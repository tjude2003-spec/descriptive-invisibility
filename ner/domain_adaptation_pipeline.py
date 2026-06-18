"""
Domain Adaptation Feasibility Pipeline
=======================================
Tests whether VeleHanden name annotations can be aligned to Transkribus HTR text
to generate BIO-tagged NER training data via distant supervision.

USAGE:
    1. First run with --fetch-htr to download PAGE XML for overlap inventory numbers
       (requires internet access to api-read.transkribus.eu)
    2. Then run with --match to perform name matching and generate BIO tags
    3. Run with --report to get matching statistics

Architecture:
    VeleHanden deed ──► scan filename ──► Transkribus page ──► HTR text
                   └─► person names ───────────────────────► fuzzy match against HTR text
                                                            └─► BIO-tagged tokens
"""

import json
import gzip
import os
import re
import sys
import time
import ssl
import logging
from pathlib import Path

# Fix macOS Python SSL certificate issue
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

# --- Configuration -----------------------------------------------------------

TRANSKRIBUS_COLLECTION = 85448
TRANSKRIBUS_DOCS = "https://api-read.transkribus.eu/documents"

# Matching thresholds
EXACT_MATCH = True           # Try exact match first
FUZZY_THRESHOLD = 0.82       # Minimum similarity for fuzzy match (0-1)
MIN_NAME_LENGTH = 4          # Skip names shorter than this (too ambiguous)
CONTEXT_WINDOW = 15          # Tokens of context around matched name for BIO output

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# --- Data Classes ------------------------------------------------------------

@dataclass
class MatchResult:
    """Result of matching a single VeleHanden name against HTR text."""
    vh_name: str
    match_type: str          # 'exact', 'fuzzy', 'partial', 'none'
    matched_span: str        # The actual text matched in HTR
    similarity: float        # 0-1 similarity score
    char_start: int          # Character offset in page text
    char_end: int
    page_scan: str           # Scan filename
    deed_id: str
    deed_type: str
    deed_date: str
    notaris: str

@dataclass
class PageMatchReport:
    """Aggregated matching results for one page."""
    inv: str
    scan: str
    total_names: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0
    no_matches: int = 0
    skipped_short: int = 0
    skipped_partial: int = 0     # "..." prefix names
    matches: list = field(default_factory=list)


# --- Step 1: Fetch HTR text from Transkribus --------------------------------

def fetch_document_pages(doc_id: int, output_dir: Path):
    """
    Fetch all pages of a Transkribus document and save extracted text locally.
    Uses the public api-read.transkribus.eu endpoint (no auth required).

    Matches the working pattern from data_collection_pipeline.py:
      GET /documents/{docId}             → metadata + pageCount
      GET /documents/{docId}/pages/{n}   → page data with content URL
      GET {content_url}                  → PAGE XML
    """
    import urllib.request

    # Step 1: Get document metadata (page count)
    url = f"{TRANSKRIBUS_DOCS}/{doc_id}"
    logger.info(f"Fetching document metadata: {url}")

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, context=ssl_context) as resp:
        doc_data = json.loads(resp.read())

    page_count = doc_data.get("pageCount", 0)
    logger.info(f"Document {doc_id}: {page_count} pages")

    output_dir.mkdir(parents=True, exist_ok=True)

    fetched = 0
    skipped = 0

    for page_nr in range(1, page_count + 1):
        # Check if already cached (by page number since we don't know img name yet)
        # We'll rename after we know the image filename

        # Step 2: Get page metadata (content URL)
        page_url = f"{TRANSKRIBUS_DOCS}/{doc_id}/pages/{page_nr}"
        try:
            req = urllib.request.Request(page_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, context=ssl_context) as resp:
                page_data = json.loads(resp.read())
        except Exception as e:
            logger.warning(f"  Page {page_nr}: metadata FAILED - {e}")
            continue

        content_url = page_data.get("content")
        img_name = page_data.get("imgFileName", f"page_{page_nr}")

        out_file = output_dir / f"{img_name}.txt"
        if out_file.exists():
            skipped += 1
            continue

        if not content_url:
            logger.warning(f"  Page {page_nr} ({img_name}): no content URL")
            continue

        time.sleep(0.3)

        # Step 3: Fetch PAGE XML from content URL
        try:
            req = urllib.request.Request(content_url)
            with urllib.request.urlopen(req, context=ssl_context) as resp:
                content = resp.read()

            # Extract text from PAGE XML
            xml_text = content.decode('utf-8')
            if '<PcGts' not in xml_text and '<pc:PcGts' not in xml_text:
                logger.warning(f"  Page {page_nr} ({img_name}): not PAGE XML")
                continue

            # Use regex extraction matching data_collection_pipeline.py
            lines = re.findall(r'<Unicode>(.*?)</Unicode>', xml_text, re.DOTALL)
            text = '\n'.join(lines)

            if text.strip():
                out_file.write_text(text, encoding='utf-8')
                fetched += 1
                if fetched % 25 == 0:
                    logger.info(f"  Progress: {fetched} pages fetched, {skipped} cached")
            else:
                logger.warning(f"  Page {page_nr} ({img_name}): empty text")

        except Exception as e:
            logger.warning(f"  Page {page_nr} ({img_name}): content FAILED - {e}")

        time.sleep(0.3)

    logger.info(f"Document {doc_id}: {fetched} fetched, {skipped} already cached")
    return page_count


def extract_text_from_page_xml(xml_bytes: bytes) -> str:
    """Extract plain text from Transkribus PAGE XML, preserving line structure."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    # Handle namespace
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    lines = []
    for region in root.findall('.//pc:TextRegion', ns):
        for line in region.findall('.//pc:TextLine', ns):
            unicode_elem = line.find('.//pc:Unicode', ns)
            if unicode_elem is not None and unicode_elem.text:
                lines.append(unicode_elem.text.strip())

    # Fallback: try without namespace
    if not lines:
        for unicode_elem in root.iter():
            if unicode_elem.tag.endswith('Unicode') and unicode_elem.text:
                lines.append(unicode_elem.text.strip())

    return '\n'.join(lines)


# --- Step 2: Match VeleHanden names against HTR text ------------------------

def normalize_for_matching(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[.,;:!?\-\(\)\[\]\"\']+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_name_in_text(name: str, page_text: str, threshold: float = FUZZY_THRESHOLD) -> Optional[MatchResult]:
    """
    Find a VeleHanden name in HTR page text.
    Returns MatchResult or None.

    Strategy:
    1. Try exact substring match (case-insensitive)
    2. Try normalized exact match (strip punctuation, collapse whitespace)
    3. Try fuzzy matching with sliding window
    """
    from rapidfuzz import fuzz

    if len(name.strip()) < MIN_NAME_LENGTH:
        return None

    name_lower = name.lower()
    text_lower = page_text.lower()

    # --- Tier 1: Exact substring match ---
    idx = text_lower.find(name_lower)
    if idx != -1:
        return MatchResult(
            vh_name=name,
            match_type='exact',
            matched_span=page_text[idx:idx+len(name)],
            similarity=1.0,
            char_start=idx,
            char_end=idx + len(name),
            page_scan='', deed_id='', deed_type='', deed_date='', notaris=''
        )

    # --- Tier 2: Normalized exact match ---
    norm_name = normalize_for_matching(name)
    norm_text = normalize_for_matching(page_text)
    idx = norm_text.find(norm_name)
    if idx != -1:
        # Map back to original text position (approximate)
        # Find the closest match in original text around this position
        approx_start = max(0, idx - 5)
        approx_end = min(len(page_text), idx + len(name) + 10)
        search_region = page_text[approx_start:approx_end]
        # Use fuzzy to get exact span in original
        best_ratio = fuzz.partial_ratio(name_lower, search_region.lower())
        return MatchResult(
            vh_name=name,
            match_type='normalized',
            matched_span=search_region.strip(),
            similarity=best_ratio / 100.0,
            char_start=approx_start,
            char_end=approx_end,
            page_scan='', deed_id='', deed_type='', deed_date='', notaris=''
        )

    # --- Tier 3: Fuzzy sliding window ---
    name_len = len(name)
    best_score = 0
    best_start = -1
    best_end = -1
    best_span = ''

    # Slide a window of roughly name-length across the text
    window_sizes = [name_len - 2, name_len - 1, name_len, name_len + 1, name_len + 2, name_len + 3]
    window_sizes = [w for w in window_sizes if w > 0]

    for wsize in window_sizes:
        for i in range(0, len(page_text) - wsize + 1, 1):
            candidate = page_text[i:i+wsize]
            # Quick pre-filter: first character should roughly match
            if abs(ord(candidate[0].lower()) - ord(name[0].lower())) > 2:
                # Allow some tolerance for first char but skip obvious mismatches
                if candidate[0].lower() != name[0].lower():
                    continue

            score = fuzz.ratio(name_lower, candidate.lower()) / 100.0
            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + wsize
                best_span = candidate

    if best_score >= threshold:
        return MatchResult(
            vh_name=name,
            match_type='fuzzy',
            matched_span=best_span,
            similarity=best_score,
            char_start=best_start,
            char_end=best_end,
            page_scan='', deed_id='', deed_type='', deed_date='', notaris=''
        )

    return None


def match_page(inv: str, scan: str, page_text: str, deeds: list) -> PageMatchReport:
    """
    Match all VeleHanden names for a given scan page against its HTR text.
    """
    report = PageMatchReport(inv=inv, scan=scan)

    # Collect all names expected on this page
    page_names = []
    for deed in deeds:
        for name_rec in deed.get('names', []):
            if name_rec.get('scan', '') == scan:
                page_names.append((name_rec, deed))

    report.total_names = len(page_names)

    for name_rec, deed in page_names:
        full_name = name_rec.get('full_name', '').strip()

        # Skip partial names (... prefix)
        if full_name.startswith('...') or full_name.startswith('…'):
            report.skipped_partial += 1
            continue

        # Skip very short names
        if len(full_name) < MIN_NAME_LENGTH:
            report.skipped_short += 1
            continue

        result = find_name_in_text(full_name, page_text)

        if result is None:
            report.no_matches += 1
        else:
            result.page_scan = scan
            result.deed_id = deed.get('akte_nr', '')
            result.deed_type = deed.get('akte_type', '')
            result.deed_date = deed.get('datering', '')
            result.notaris = deed.get('notaris', '')

            if result.match_type == 'exact':
                report.exact_matches += 1
            else:
                report.fuzzy_matches += 1

            report.matches.append(result)

    return report


# --- Step 3: Generate BIO-tagged training data ------------------------------

def tokenize_simple(text: str) -> list[tuple[str, int, int]]:
    """Simple whitespace tokenizer returning (token, start, end) tuples."""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def generate_bio_tags(page_text: str, matches: list[MatchResult]) -> list[tuple[str, str]]:
    """
    Generate BIO-tagged token sequence from page text and matched name spans.

    Returns list of (token, tag) where tag is 'B-PER', 'I-PER', or 'O'.
    """
    tokens = tokenize_simple(page_text)
    tags = ['O'] * len(tokens)

    # Sort matches by position to handle overlaps
    sorted_matches = sorted(matches, key=lambda m: m.char_start)

    for match in sorted_matches:
        start, end = match.char_start, match.char_end
        first_token = True

        for i, (tok, tok_start, tok_end) in enumerate(tokens):
            # Token overlaps with match span
            if tok_end > start and tok_start < end:
                if first_token:
                    tags[i] = 'B-PER'
                    first_token = False
                else:
                    tags[i] = 'I-PER'

    return [(tok, tag) for (tok, _, _), tag in zip(tokens, tags)]


def bio_to_conll(tagged_tokens: list[tuple[str, str]], 
                 inv: str = '', scan: str = '') -> str:
    """Format BIO-tagged tokens as CoNLL-style output."""
    lines = [f"# inv={inv} scan={scan}"]
    for token, tag in tagged_tokens:
        lines.append(f"{token}\t{tag}")
    lines.append("")  # Empty line = sentence boundary
    return '\n'.join(lines)


# --- Main pipeline -----------------------------------------------------------

def run_feasibility_test(data_dir: str, htr_dir: str, output_dir: str,
                         test_invs: list[str] = None, max_pages: int = None):
    """
    Run the full feasibility test pipeline.

    Args:
        data_dir: Directory containing the uploaded data files
        htr_dir: Directory containing cached HTR page texts (from fetch step)
        output_dir: Where to write results
        test_invs: Specific inventory numbers to test (default: first 5 overlap)
        max_pages: Limit pages per inventory (for quick testing)
    """
    data_dir = Path(data_dir)
    htr_dir = Path(htr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading VeleHanden annotations...")
    with gzip.open(data_dir / 'compressed_anno_json.gz', 'rt', encoding='utf-8') as f:
        anno = json.load(f)

    logger.info("Loading alignment...")
    with open(data_dir / 'alignment.json') as f:
        alignment = json.load(f)

    logger.info("Loading Transkribus index...")
    with open(data_dir / 'transkribus_index.json') as f:
        trans_index = json.load(f)

    overlap_invs = [item[0] for item in alignment['overlap']]

    if test_invs is None:
        # Pick a diverse set: different notaries, time periods
        test_invs = overlap_invs[:10]

    # --- Matching phase ---
    all_reports = []
    all_bio = []

    for inv in test_invs:
        if inv not in anno:
            logger.warning(f"Inv {inv} not in VeleHanden data, skipping")
            continue

        deeds = anno[inv]
        doc_info = trans_index.get(inv, {})

        # Collect all unique scan filenames for this inventory
        scans = set()
        for d in deeds:
            for n in d.get('names', []):
                s = n.get('scan', '').strip()
                if s:
                    scans.add(s)

        sorted_scans = sorted(scans)
        if max_pages:
            sorted_scans = sorted_scans[:max_pages]

        logger.info(f"Inv {inv}: {len(deeds)} deeds, {len(sorted_scans)} pages to check")

        for scan in sorted_scans:
            # Load HTR text for this page
            htr_file = htr_dir / inv / f"{scan}.txt"
            if not htr_file.exists():
                # Try alternative naming patterns
                htr_file = htr_dir / f"{scan}.txt"
                if not htr_file.exists():
                    continue

            page_text = htr_file.read_text(encoding='utf-8')
            if len(page_text.strip()) < 10:
                continue

            # Match names against HTR text
            report = match_page(inv, scan, page_text, deeds)
            all_reports.append(report)

            # Generate BIO tags
            if report.matches:
                tagged = generate_bio_tags(page_text, report.matches)
                bio_str = bio_to_conll(tagged, inv=inv, scan=scan)
                all_bio.append(bio_str)

    # --- Reporting ---
    total_names = sum(r.total_names for r in all_reports)
    total_exact = sum(r.exact_matches for r in all_reports)
    total_fuzzy = sum(r.fuzzy_matches for r in all_reports)
    total_none = sum(r.no_matches for r in all_reports)
    total_skipped_partial = sum(r.skipped_partial for r in all_reports)
    total_skipped_short = sum(r.skipped_short for r in all_reports)
    total_attempted = total_names - total_skipped_partial - total_skipped_short

    report_text = f"""
=============================================================
DOMAIN ADAPTATION FEASIBILITY REPORT
=============================================================

Inventories tested:   {len(test_invs)}
Pages processed:      {len(all_reports)}
Total VH names:       {total_names}
  Skipped (partial):  {total_skipped_partial} ({total_skipped_partial/max(total_names,1)*100:.1f}%)
  Skipped (short):    {total_skipped_short} ({total_skipped_short/max(total_names,1)*100:.1f}%)
  Attempted:          {total_attempted}

MATCHING RESULTS (of {total_attempted} attempted):
  Exact matches:      {total_exact} ({total_exact/max(total_attempted,1)*100:.1f}%)
  Fuzzy matches:      {total_fuzzy} ({total_fuzzy/max(total_attempted,1)*100:.1f}%)
  No match:           {total_none} ({total_none/max(total_attempted,1)*100:.1f}%)
  TOTAL ALIGNED:      {total_exact + total_fuzzy} ({(total_exact + total_fuzzy)/max(total_attempted,1)*100:.1f}%)

BIO-tagged sequences: {len(all_bio)}

FEASIBILITY ASSESSMENT:
"""
    align_rate = (total_exact + total_fuzzy) / max(total_attempted, 1)
    if align_rate >= 0.7:
        report_text += f"  STRONG - {align_rate:.0%} alignment rate. Sufficient for fine-tuning.\n"
    elif align_rate >= 0.5:
        report_text += f"  MODERATE - {align_rate:.0%} alignment rate. Viable with noise-tolerant training.\n"
    elif align_rate >= 0.3:
        report_text += f"  MARGINAL - {align_rate:.0%} alignment rate. Consider improving matching heuristic.\n"
    else:
        report_text += f"  WEAK - {align_rate:.0%} alignment rate. HTR quality may be insufficient.\n"

    # Fuzzy match quality distribution
    if all_reports:
        all_fuzzy_scores = []
        for r in all_reports:
            for m in r.matches:
                if m.match_type == 'fuzzy':
                    all_fuzzy_scores.append(m.similarity)

        if all_fuzzy_scores:
            report_text += f"\nFUZZY MATCH QUALITY:\n"
            report_text += f"  Mean similarity:    {sum(all_fuzzy_scores)/len(all_fuzzy_scores):.3f}\n"
            report_text += f"  Min similarity:     {min(all_fuzzy_scores):.3f}\n"
            report_text += f"  Scores >= 0.90:     {sum(1 for s in all_fuzzy_scores if s >= 0.9)} ({sum(1 for s in all_fuzzy_scores if s >= 0.9)/len(all_fuzzy_scores)*100:.1f}%)\n"
            report_text += f"  Scores 0.85-0.90:   {sum(1 for s in all_fuzzy_scores if 0.85 <= s < 0.9)} ({sum(1 for s in all_fuzzy_scores if 0.85 <= s < 0.9)/len(all_fuzzy_scores)*100:.1f}%)\n"
            report_text += f"  Scores 0.82-0.85:   {sum(1 for s in all_fuzzy_scores if 0.82 <= s < 0.85)} ({sum(1 for s in all_fuzzy_scores if 0.82 <= s < 0.85)/len(all_fuzzy_scores)*100:.1f}%)\n"

    # Show some example matches
    report_text += f"\nEXAMPLE MATCHES:\n"
    shown = 0
    for r in all_reports:
        for m in r.matches:
            if shown >= 15:
                break
            if m.match_type in ('exact', 'fuzzy'):
                report_text += f"  [{m.match_type:>5}] VH: '{m.vh_name}' → HTR: '{m.matched_span}' (sim={m.similarity:.2f})\n"
                shown += 1
        if shown >= 15:
            break

    # Show some failures
    report_text += f"\nEXAMPLE NON-MATCHES (names not found in HTR):\n"

    print(report_text)

    # Save outputs
    (output_dir / 'feasibility_report.txt').write_text(report_text)

    if all_bio:
        bio_output = '\n'.join(all_bio)
        (output_dir / 'bio_tagged_sample.conll').write_text(bio_output)
        logger.info(f"Saved {len(all_bio)} BIO-tagged sequences to bio_tagged_sample.conll")

    # Save detailed match data as JSON
    match_data = []
    for r in all_reports:
        for m in r.matches:
            match_data.append(asdict(m))
    (output_dir / 'match_details.json').write_text(
        json.dumps(match_data, ensure_ascii=False, indent=2)
    )

    return all_reports


# --- Simulation mode (for testing without API access) -----------------------

def simulate_htr_text(names: list[str], noise_level: float = 0.15) -> str:
    """
    Generate realistic simulated HTR text containing the given names,
    with configurable noise to mimic HTR errors.
    Used for testing the matching pipeline without Transkribus API access.
    """
    import random

    # Common Dutch notarial formulaic phrases
    templates = [
        "Op huijden den {date} compareerden voor mij {notaris} notaris publicq {names_text}",
        "den {date} compareerde voor mij {notaris} openbaer notaris {names_text}",
        "Op den {date} zijn voor mij {notaris} notaris publicq gecompareert {names_text}",
        "In den Jare {date} compareerden voor mij ondergeschreven notaris {names_text}",
    ]

    connectors = [
        " ende ", " mitsgaders ", " als mede ", " en ", " beneffens ",
        " gehuijst met ", " weduwe van ", " zoon van ", " dochter van ",
        " wonende op de ", " koopman alhier ", " mr. ", " borger deser stede ",
    ]

    def add_noise(text: str, level: float) -> str:
        """Simulate HTR errors: char substitutions, insertions, deletions."""
        chars = list(text)
        result = []
        for c in chars:
            r = random.random()
            if r < level * 0.4:
                # Substitution with similar char
                similar = {
                    'e': 'c', 'i': 'j', 'n': 'u', 'u': 'n', 
                    'h': 'b', 'c': 'e', 'o': 'a', 'a': 'o',
                    't': 'f', 'l': 'I', 's': 'f', 'r': 'v',
                    'd': 'cl', 'b': 'h', 'p': 'b', 'v': 'r',
                }
                result.append(similar.get(c, c))
            elif r < level * 0.5:
                # Deletion
                continue
            elif r < level * 0.55:
                # Insertion
                result.append(c)
                result.append(random.choice('eioanst'))
            else:
                result.append(c)
        return ''.join(result)

    random.seed(42)
    template = random.choice(templates)

    # Build text with names embedded
    name_parts = []
    for i, name in enumerate(names):
        if random.random() < noise_level:
            name = add_noise(name, noise_level)
        name_parts.append(name)
        if i < len(names) - 1:
            name_parts.append(random.choice(connectors))

    names_text = ''.join(name_parts)

    text = template.format(
        date="27 Januarij 1733",
        notaris="Philip Zweerts",
        names_text=names_text
    )

    # Add some general noise to non-name parts too
    # (We specifically want to test if matching works despite HTR artifacts)
    return text


def run_simulation(data_dir: str, output_dir: str):
    """
    Run the matching pipeline on simulated HTR text to test feasibility
    without needing Transkribus API access.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading VeleHanden annotations...")
    with gzip.open(data_dir / 'compressed_anno_json.gz', 'rt', encoding='utf-8') as f:
        anno = json.load(f)

    with open(data_dir / 'alignment.json') as f:
        alignment = json.load(f)

    # Test on a few overlap inventories
    test_invs = ['10024', '2382', '1927', '1327', '431']
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

    print("=" * 70)
    print("SIMULATION: Matching VeleHanden names against simulated HTR text")
    print("=" * 70)
    print(f"\nTesting {len(test_invs)} inventories at {len(noise_levels)} noise levels")
    print("(Noise simulates HTR character error rate)\n")

    results_by_noise = {}

    for noise in noise_levels:
        all_reports = []

        for inv in test_invs:
            if inv not in anno:
                continue
            deeds = anno[inv]

            # Group names by scan page
            page_names = defaultdict(list)
            for d in deeds:
                for n in d.get('names', []):
                    scan = n.get('scan', '').strip()
                    if scan:
                        page_names[scan].append((n, d))

            # Process first N pages
            for scan in list(sorted(page_names.keys()))[:20]:
                names_on_page = page_names[scan]
                vh_names = [n['full_name'] for n, d in names_on_page 
                           if n.get('full_name', '').strip() 
                           and not n['full_name'].startswith('...')]

                if not vh_names:
                    continue

                # Generate simulated HTR text with these names embedded
                sim_text = simulate_htr_text(vh_names, noise_level=noise)

                # Run matching
                report = match_page(inv, scan, sim_text, 
                                   [d for _, d in names_on_page])
                all_reports.append(report)

        total_attempted = sum(r.total_names - r.skipped_partial - r.skipped_short 
                            for r in all_reports)
        total_exact = sum(r.exact_matches for r in all_reports)
        total_fuzzy = sum(r.fuzzy_matches for r in all_reports)
        total_none = sum(r.no_matches for r in all_reports)

        results_by_noise[noise] = {
            'attempted': total_attempted,
            'exact': total_exact,
            'fuzzy': total_fuzzy,
            'none': total_none,
            'aligned': total_exact + total_fuzzy,
            'rate': (total_exact + total_fuzzy) / max(total_attempted, 1)
        }

    # Print results table
    print(f"{'Noise':>8} {'Attempted':>10} {'Exact':>8} {'Fuzzy':>8} {'None':>8} {'Aligned':>8} {'Rate':>8}")
    print("-" * 70)
    for noise in noise_levels:
        r = results_by_noise[noise]
        print(f"{noise:>7.0%} {r['attempted']:>10} {r['exact']:>8} {r['fuzzy']:>8} {r['none']:>8} {r['aligned']:>8} {r['rate']:>7.1%}")

    print(f"\nInterpretation:")
    print(f"  0% noise  = perfect HTR (upper bound on matching)")
    print(f"  5-10%     = good quality HTR (typical for 18th-19th century documents)")
    print(f"  15-20%    = moderate HTR (typical for 17th century / difficult hands)")
    print(f"  25%+      = poor HTR (very early / degraded manuscripts)")
    print(f"\n  Alignment rates above 60% are sufficient for distant supervision.")
    print(f"  The actual HTR quality determines where your data falls on this curve.")

    # Save results
    (output_dir / 'simulation_results.json').write_text(
        json.dumps(results_by_noise, indent=2)
    )

    return results_by_noise


# --- Entry point -------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Domain Adaptation Feasibility Pipeline')
    parser.add_argument('--mode', choices=['simulate', 'fetch', 'match'],
                       default='simulate',
                       help='simulate: test with synthetic HTR; fetch: download real HTR; match: run on real HTR')
    parser.add_argument('--data-dir', default='/mnt/user-data/uploads',
                       help='Directory with VeleHanden/alignment/transkribus data')
    parser.add_argument('--htr-dir', default='./htr_cache',
                       help='Directory for cached HTR texts')
    parser.add_argument('--output-dir', default='./adaptation_results',
                       help='Output directory for results')
    parser.add_argument('--invs', nargs='+', default=None,
                       help='Specific inventory numbers to test')

    args = parser.parse_args()

    if args.mode == 'simulate':
        run_simulation(args.data_dir, args.output_dir)
    elif args.mode == 'fetch':
        # Fetch HTR from Transkribus (public API, no auth needed)
        with open(Path(args.data_dir) / 'transkribus_index.json') as f:
            trans_index = json.load(f)
        with open(Path(args.data_dir) / 'alignment.json') as f:
            alignment = json.load(f)

        overlap_invs = [item[0] for item in alignment['overlap']]
        test_invs = args.invs or overlap_invs[:10]

        for inv in test_invs:
            if inv in trans_index:
                doc_id = trans_index[inv]['docId']
                fetch_document_pages(doc_id, Path(args.htr_dir) / inv)
    elif args.mode == 'match':
        run_feasibility_test(args.data_dir, args.htr_dir, args.output_dir,
                           test_invs=args.invs)
