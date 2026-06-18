#!/usr/bin/env python3
"""
Data collection pipeline for MA thesis:
"All Present, None Accounted For"

Collects three aligned data layers for fonds 5075:

1. Transkribus HTR transcriptions (collection 85448)
2. VeleHanden indexed person names (XML zip export)
3. Amsterdam City Archives finding aids (id.archief.amsterdam API)

Alignment key: inventory number

Changelog (v2):
- Fixed SAA scrape: key by file_id, not series_id (was overwriting)
- Added pagination to all API calls (series list + file children)
- Fixed alignment: handle sub-volume keys (e.g. 1373A, 1373B)
- Added error handling and retries on all HTTP calls
- Proper file handle management (with statements)
- XML parsing instead of regex for PAGE XML
- Logging throughout
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRANSKRIBUS_FACETS = "https://api-read.transkribus.eu/facets"
TRANSKRIBUS_DOCS = "https://api-read.transkribus.eu/documents"
SAA_SEARCH = "https://id.archief.amsterdam/search/records"

COLLECTION_ID = 85448
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ---------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------

def http_get(url, retries=MAX_RETRIES):
    """GET request via curl with retry logic. Returns parsed JSON or None."""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-s", "-f", "--max-time", "30", url],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.warning(
                    "GET %s failed (curl exit %d, attempt %d/%d)",
                    url, result.returncode, attempt, retries,
                )
                time.sleep(RETRY_DELAY * attempt)
                continue

            if not result.stdout.strip():
                log.warning("GET %s returned empty body (attempt %d/%d)",
                            url, attempt, retries)
                time.sleep(RETRY_DELAY * attempt)
                continue

            return json.loads(result.stdout)

        except json.JSONDecodeError as e:
            log.warning("GET %s JSON parse error: %s (attempt %d/%d)",
                        url, e, attempt, retries)
            time.sleep(RETRY_DELAY * attempt)

    log.error("GET %s failed after %d attempts", url, retries)
    return None


def http_get_text(url, retries=MAX_RETRIES):
    """GET request via curl returning raw text. For PAGE XML content URLs."""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-s", "-f", "--max-time", "60", url],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.warning(
                    "GET (text) %s failed (curl exit %d, attempt %d/%d)",
                    url, result.returncode, attempt, retries,
                )
                time.sleep(RETRY_DELAY * attempt)
                continue

            return result.stdout

        except Exception as e:
            log.warning("GET (text) %s error: %s (attempt %d/%d)",
                        url, e, attempt, retries)
            time.sleep(RETRY_DELAY * attempt)

    log.error("GET (text) %s failed after %d attempts", url, retries)
    return None


def http_post_json(url, payload, retries=MAX_RETRIES):
    """POST JSON request via curl with retry logic. Returns parsed JSON or None."""
    payload_str = json.dumps(payload)

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-s", "-f", "--max-time", "30",
                 "-X", "POST",
                 "-H", "Content-Type: application/json",
                 url,
                 "-d", payload_str],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.warning(
                    "POST %s failed (curl exit %d, attempt %d/%d)",
                    url, result.returncode, attempt, retries,
                )
                time.sleep(RETRY_DELAY * attempt)
                continue

            if not result.stdout.strip():
                log.warning("POST %s returned empty body (attempt %d/%d)",
                            url, attempt, retries)
                time.sleep(RETRY_DELAY * attempt)
                continue

            return json.loads(result.stdout)

        except json.JSONDecodeError as e:
            log.warning("POST %s JSON parse error: %s (attempt %d/%d)",
                        url, e, attempt, retries)
            time.sleep(RETRY_DELAY * attempt)

    log.error("POST %s failed after %d attempts", url, retries)
    return None


# ---------------------------------------------------------------------
# Layer 1 — Transkribus
# ---------------------------------------------------------------------

def build_transkribus_index():
    """
    Return mapping: inventory_number -> {docId, title}.

    Handles sub-volumes (e.g. 1373A, 1373B) as separate entries.
    Titles are expected as '5075_{inv}' or '{inv}_NOTA...' format.
    """
    all_docs = []
    offset = 0
    limit = 100

    log.info("Building Transkribus index for collection %d...", COLLECTION_ID)

    while True:
        payload = {
            "facets": ["f_title"],
            "filterFacets": [{"value": [str(COLLECTION_ID)], "field": "colId"}],
            "groupDocId": 1,
            "collections": [COLLECTION_ID],
            "offset": offset,
            "limit": limit,
        }

        data = http_post_json(TRANSKRIBUS_FACETS, payload)
        if data is None:
            log.error("Failed to fetch Transkribus facets at offset %d", offset)
            break

        items = data.get("items", [])
        total = data.get("total", 0)
        all_docs.extend(items)

        log.info("  fetched %d/%d documents", len(all_docs), total)

        if len(items) < limit or len(all_docs) >= total:
            break

        offset += limit
        time.sleep(REQUEST_DELAY)

    inv_to_doc = {}
    skipped = []

    for d in all_docs:
        title = d.get("title", "")
        doc_id = d.get("id")

        if not title or doc_id is None:
            skipped.append(title)
            continue

        # Parse inventory number from title
        # Expected formats: "5075_10298" or "10298_NOTA01003_merged"
        if title.startswith("5075_"):
            inv = title[5:]
            # Strip trailing suffixes like _merged but keep letter suffixes
            # "5075_1373A" -> "1373A" (keep)
            # Some titles: "5075_10298" -> "10298"
            inv = inv.split("_")[0] if "_" in inv else inv
        else:
            inv = title.split("_", 1)[0]

        if not inv or not inv[0].isdigit():
            skipped.append(title)
            continue

        if inv in inv_to_doc:
            log.warning("Duplicate inventory key '%s' (title: %s vs %s)",
                        inv, title, inv_to_doc[inv]["title"])

        inv_to_doc[inv] = {"docId": doc_id, "title": title}

    log.info("Transkribus index: %d documents, %d skipped", len(inv_to_doc), len(skipped))
    if skipped:
        log.warning("Skipped titles: %s", skipped[:10])

    return inv_to_doc


def parse_page_xml(xml_text):
    """Extract text lines from PAGE XML using proper XML parsing."""
    if not xml_text or "<PcGts" not in xml_text:
        return None

    # PAGE XML uses a namespace
    ns_match = re.match(r'.*?<PcGts[^>]+xmlns="([^"]+)"', xml_text, re.DOTALL)
    ns = {"pc": ns_match.group(1)} if ns_match else {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("PAGE XML parse error: %s", e)
        return None

    lines = []
    # Try with namespace first, then without
    unicode_elements = root.findall(".//pc:Unicode", ns) if ns else []
    if not unicode_elements:
        unicode_elements = root.findall(".//{*}Unicode")
    if not unicode_elements:
        # Fallback: regex (some PAGE XML variants are irregular)
        lines = re.findall(r'<Unicode>(.*?)</Unicode>', xml_text, re.DOTALL)
    else:
        lines = [el.text or "" for el in unicode_elements]

    return lines


def get_page_text(doc_id, page_nr):
    """Retrieve text lines from a Transkribus PAGE XML page."""

    page_data = http_get(f"{TRANSKRIBUS_DOCS}/{doc_id}/pages/{page_nr}")
    if page_data is None:
        return None

    content_url = page_data.get("content")
    if not content_url:
        log.warning("No content URL for doc %s page %d", doc_id, page_nr)
        return None

    time.sleep(REQUEST_DELAY)

    xml_text = http_get_text(content_url)
    if xml_text is None:
        return None

    lines = parse_page_xml(xml_text)
    if lines is None:
        return None

    return {
        "page_nr": page_nr,
        "lines": lines,
        "full_text": "\n".join(lines),
        "xml": xml_text,
    }


def get_document_text(doc_id, max_pages=None):
    """Retrieve full document transcription."""

    doc_data = http_get(f"{TRANSKRIBUS_DOCS}/{doc_id}")
    if doc_data is None:
        return []

    page_count = doc_data.get("pageCount", 0)
    if max_pages:
        page_count = min(page_count, max_pages)

    pages = []

    for p in range(1, page_count + 1):
        page = get_page_text(doc_id, p)
        if page:
            pages.append(page)
        time.sleep(REQUEST_DELAY)

    return pages


# ---------------------------------------------------------------------
# Layer 2 — VeleHanden
# ---------------------------------------------------------------------

def parse_velehanden(zip_path):
    """
    Parse VeleHanden XML export.

    Returns dict: inventory_number -> list of deed records.
    Each deed has: akte_nr, akte_type, datering, rubriek, notaris, names[].
    """
    vh_data = defaultdict(list)
    record_count = 0
    name_count = 0

    log.info("Parsing VeleHanden XML from %s...", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
        log.info("  found %d XML files in archive", len(xml_files))

        for fname in xml_files:
            with zf.open(fname) as f:
                try:
                    tree = ET.parse(f)
                except ET.ParseError as e:
                    log.warning("XML parse error in %s: %s", fname, e)
                    continue

                root = tree.getroot()

                for rec in root.iter("indexRecord"):
                    inv_el = rec.find("inventarisNr")
                    if inv_el is None or not inv_el.text:
                        continue

                    inv_nr = inv_el.text.strip()
                    names = []

                    for pn in rec.iter("persoonsnaam"):
                        vn = (pn.findtext("voornaam") or "").strip()
                        tv = (pn.findtext("tussenvoegsel") or "").strip()
                        an = (pn.findtext("achternaam") or "").strip()
                        sn = (pn.findtext("scanNaam") or "").strip()

                        full = " ".join(part for part in [vn, tv, an] if part)

                        names.append({
                            "voornaam": vn,
                            "tussenvoegsel": tv,
                            "achternaam": an,
                            "full_name": full,
                            "scan": sn,
                        })

                    vh_data[inv_nr].append({
                        "akte_nr": rec.findtext("akteNr", "").strip(),
                        "akte_type": rec.findtext("akteType", "").strip(),
                        "datering": rec.findtext("datering", "").strip(),
                        "rubriek": rec.findtext("rubriek", "").strip(),
                        "notaris": rec.findtext("notaris", "").strip(),
                        "names": names,
                    })

                    record_count += 1
                    name_count += len(names)

    log.info("VeleHanden: %d inventory numbers, %d deeds, %d person names",
             len(vh_data), record_count, name_count)

    return dict(vh_data)


# ---------------------------------------------------------------------
# Layer 3 — SAA Finding Aids
# ---------------------------------------------------------------------

def get_all_children(parent_id, per_page=50):
    """
    Fetch ALL child records of a given parent, with full pagination.

    Returns list of record dicts.
    """
    all_rows = []
    page = 1

    while True:
        payload = {
            "pagination": {"page": page, "perPage": per_page},
            "query": {
                "type": "FieldQuery",
                "operator": "equals",
                "field": "data.parentId",
                "value": parent_id,
            },
        }

        data = http_post_json(f"{SAA_SEARCH}?lang=nl", payload)
        if data is None:
            log.error("SAA API failed for parent %s page %d", parent_id, page)
            break

        rows = data.get("rows", [])
        total = data.get("total", len(rows))
        all_rows.extend(rows)

        if len(all_rows) >= total or not rows:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    return all_rows


def get_fonds_5075_finding_aids():
    """
    Scrape the complete finding aid hierarchy for fonds 5075.

    Structure: Fonds -> Series (notary) -> File (inventory range)

    Returns dict keyed by file_id (not series_id) to avoid overwrites.
    """
    fonds_id = "ed57f5d1-9d31-4446-a3e0-37ac1bedf989"

    log.info("Fetching fonds 5075 series list...")
    series_list = get_all_children(fonds_id, per_page=100)
    log.info("  found %d series (notary entries)", len(series_list))

    finding_aids = {}
    file_count = 0

    for i, s in enumerate(series_list):
        series_id = s.get("id")
        series_title = s.get("title", "")

        time.sleep(REQUEST_DELAY)

        files = get_all_children(series_id, per_page=50)

        for f in files:
            file_id = f.get("id")
            if not file_id:
                continue

            finding_aids[file_id] = {
                "series_title": series_title,
                "file_title": f.get("title", ""),
                "file_description": f.get("description") or "",
                "file_id": file_id,
            }
            file_count += 1

        if (i + 1) % 50 == 0:
            log.info("  processed %d/%d series (%d files so far)",
                     i + 1, len(series_list), file_count)

    log.info("SAA finding aids: %d series, %d file-level entries",
             len(series_list), len(finding_aids))

    return finding_aids


# ---------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------

def compute_overlap(transkribus_index, velehanden_data):
    """
    Compute three-way alignment between Transkribus and VeleHanden
    by inventory number.

    Handles sub-volumes correctly: 1373A and 1373B are treated as
    separate items. Matching is by exact key, not base number.

    Returns (overlap, htr_only, vh_only) where:
    - overlap: list of (trk_inv, vh_inv) pairs
    - htr_only: list of inventory numbers in Transkribus only
    - vh_only: list of inventory numbers in VeleHanden only
    """
    tr_invs = set(transkribus_index.keys())
    vh_invs = set(velehanden_data.keys())

    # Phase 1: exact match
    exact_overlap = tr_invs & vh_invs
    tr_remaining = tr_invs - exact_overlap
    vh_remaining = vh_invs - exact_overlap

    overlap_pairs = [(inv, inv) for inv in sorted(exact_overlap)]

    # Phase 2: base-number match for remaining items
    # This catches cases like Transkribus has "1373A" and VeleHanden
    # has "1373A" with slightly different formatting, or where one
    # source uses "481A" and the other uses "481A".
    #
    # We do NOT collapse sub-volumes: if Transkribus has 1373A, 1373B,
    # 1373C and VeleHanden has 1373A, 1373B, then:
    #   - 1373A and 1373B are overlap (matched exactly above)
    #   - 1373C is htr_only
    # The base-number phase only catches items that didn't exact-match
    # but share a base number with exactly one counterpart.

    def base_inv(inv):
        m = re.match(r"(\d+)", str(inv))
        return m.group(1) if m else inv

    # Group remaining items by base number
    tr_by_base = defaultdict(list)
    for inv in tr_remaining:
        tr_by_base[base_inv(inv)].append(inv)

    vh_by_base = defaultdict(list)
    for inv in vh_remaining:
        vh_by_base[base_inv(inv)].append(inv)

    # Match remaining items by base number (1-to-1 only)
    shared_bases = set(tr_by_base.keys()) & set(vh_by_base.keys())
    base_matched_tr = set()
    base_matched_vh = set()

    for base in shared_bases:
        tr_variants = tr_by_base[base]
        vh_variants = vh_by_base[base]

        # Match variants with identical keys across sources
        for t in tr_variants:
            for v in vh_variants:
                if t == v and t not in base_matched_tr and v not in base_matched_vh:
                    overlap_pairs.append((t, v))
                    base_matched_tr.add(t)
                    base_matched_vh.add(v)

        # If base has exactly one unmatched variant on each side,
        # treat as a fuzzy match (likely same volume, different naming)
        unmatched_tr = [t for t in tr_variants if t not in base_matched_tr]
        unmatched_vh = [v for v in vh_variants if v not in base_matched_vh]

        if len(unmatched_tr) == 1 and len(unmatched_vh) == 1:
            overlap_pairs.append((unmatched_tr[0], unmatched_vh[0]))
            base_matched_tr.add(unmatched_tr[0])
            base_matched_vh.add(unmatched_vh[0])
            log.info("  fuzzy-matched: Transkribus '%s' <-> VeleHanden '%s'",
                     unmatched_tr[0], unmatched_vh[0])

    htr_only = sorted(tr_remaining - base_matched_tr,
                      key=lambda x: (base_inv(x), x))
    vh_only = sorted(vh_remaining - base_matched_vh,
                     key=lambda x: (base_inv(x), x))

    log.info("Alignment: %d overlap, %d htr_only, %d vh_only",
             len(overlap_pairs), len(htr_only), len(vh_only))

    # Sanity checks
    all_tr = set(p[0] for p in overlap_pairs) | set(htr_only)
    all_vh = set(p[1] for p in overlap_pairs) | set(vh_only)

    missing_tr = tr_invs - all_tr
    missing_vh = vh_invs - all_vh

    if missing_tr:
        log.error("ALIGNMENT BUG: %d Transkribus items unaccounted for: %s",
                  len(missing_tr), sorted(missing_tr)[:10])
    if missing_vh:
        log.error("ALIGNMENT BUG: %d VeleHanden items unaccounted for: %s",
                  len(missing_vh), sorted(missing_vh)[:10])

    total_aligned = len(overlap_pairs) + len(htr_only) + len(vh_only)
    total_expected = len(tr_invs | vh_invs)
    if total_aligned != total_expected:
        log.error("ALIGNMENT BUG: aligned %d but expected %d (union of sources)",
                  total_aligned, total_expected)

    return overlap_pairs, htr_only, vh_only


# ---------------------------------------------------------------------
# Demo Retrieval
# ---------------------------------------------------------------------

def demo_retrieval(transkribus_index, velehanden_data,
                   inv_nr="10021", max_pages=3):
    """Fetch and display a sample document for verification."""

    tr_entry = transkribus_index.get(inv_nr)
    if not tr_entry:
        log.warning("Inventory %s not found in Transkribus index", inv_nr)
        return

    log.info("Demo retrieval for inventory %s (docId %s)...",
             inv_nr, tr_entry["docId"])

    pages = get_document_text(tr_entry["docId"], max_pages=max_pages)

    for page in pages:
        print(f"\n--- Page {page['page_nr']} ---")
        for line in page["lines"][:10]:
            print(f"  {line}")

    vh_deeds = velehanden_data.get(inv_nr, [])
    print(f"\nVeleHanden deeds for inv {inv_nr}: {len(vh_deeds)}")
    if vh_deeds:
        sample = vh_deeds[0]
        print(f"  First deed: {sample.get('akte_type', '?')} "
              f"({sample.get('datering', '?')}), "
              f"{len(sample.get('names', []))} persons")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Data collection pipeline for thesis: "
                    "All Present, None Accounted For"
    )
    parser.add_argument("--velehanden", required=True,
                        help="Path to VeleHanden XML zip export")
    parser.add_argument("--output", default="./thesis_data",
                        help="Output directory for cached JSON files")
    parser.add_argument("--demo-inv", default="10021",
                        help="Inventory number for demo retrieval")
    parser.add_argument("--demo-pages", type=int, default=3,
                        help="Max pages to fetch in demo")
    parser.add_argument("--skip-saa", action="store_true",
                        help="Skip SAA finding aid scrape")
    parser.add_argument("--index-only", action="store_true",
                        help="Build index only, skip demo retrieval")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore cached files and re-fetch everything")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Layer 1: Transkribus ---
    index_path = output_dir / "transkribus_index.json"
    if index_path.exists() and not args.force_refresh:
        log.info("Loading cached Transkribus index from %s", index_path)
        with open(index_path) as f:
            transkribus_index = json.load(f)
    else:
        transkribus_index = build_transkribus_index()
        with open(index_path, "w") as f:
            json.dump(transkribus_index, f, indent=2)
        log.info("Saved Transkribus index to %s", index_path)

    # --- Layer 2: VeleHanden ---
    vh_path = output_dir / "velehanden_parsed.json"
    if vh_path.exists() and not args.force_refresh:
        log.info("Loading cached VeleHanden data from %s", vh_path)
        with open(vh_path) as f:
            velehanden_data = json.load(f)
    else:
        velehanden_data = parse_velehanden(args.velehanden)
        with open(vh_path, "w") as f:
            json.dump(velehanden_data, f)
        log.info("Saved VeleHanden data to %s", vh_path)

    # --- Layer 3: SAA Finding Aids ---
    if not args.skip_saa:
        saa_path = output_dir / "saa_finding_aids.json"
        if saa_path.exists() and not args.force_refresh:
            log.info("Loading cached SAA finding aids from %s", saa_path)
            with open(saa_path) as f:
                finding_aids = json.load(f)
        else:
            finding_aids = get_fonds_5075_finding_aids()
            with open(saa_path, "w") as f:
                json.dump(finding_aids, f, indent=2)
            log.info("Saved SAA finding aids to %s", saa_path)

    # --- Alignment ---
    overlap, htr_only, vh_only = compute_overlap(
        transkribus_index, velehanden_data
    )

    alignment_path = output_dir / "alignment.json"
    with open(alignment_path, "w") as f:
        json.dump(
            {"overlap": overlap, "htr_only": htr_only, "vh_only": vh_only},
            f,
            indent=2,
        )
    log.info("Saved alignment to %s", alignment_path)

    # --- Summary ---
    log.info("=== Pipeline Summary ===")
    log.info("  Transkribus documents: %d", len(transkribus_index))
    log.info("  VeleHanden inventories: %d", len(velehanden_data))
    log.info("  Overlap: %d", len(overlap))
    log.info("  HTR-only: %d", len(htr_only))
    log.info("  VH-only: %d", len(vh_only))
    log.info("  Total aligned: %d", len(overlap) + len(htr_only) + len(vh_only))

    vh_deeds = sum(len(deeds) for deeds in velehanden_data.values())
    vh_names = sum(
        len(d.get("names", []))
        for deeds in velehanden_data.values()
        for d in deeds
    )
    log.info("  VeleHanden total deeds: %d", vh_deeds)
    log.info("  VeleHanden total person names: %d", vh_names)

    # --- Demo ---
    if not args.index_only:
        demo_retrieval(
            transkribus_index,
            velehanden_data,
            args.demo_inv,
            args.demo_pages,
        )


if __name__ == "__main__":
    main()
