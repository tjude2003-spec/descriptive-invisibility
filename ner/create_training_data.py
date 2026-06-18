#!/usr/bin/env python3
"""
Create token-level BIO training data for NER domain adaptation.

Distant supervision approach:
1. For each train-set inventory, load all HTR page texts
2. Load VeleHanden person names for that inventory
3. For each VH name, search for fuzzy matches in the HTR text
4. Mark matched spans as B-PER / I-PER tokens, everything else as O
5. Output in CoNLL format (one token per line, blank line between sentences)

Usage:
    # Quick test (2 inventories, prints stats)
    python create_training_data.py --test

    # Full run
    python create_training_data.py

    # Custom paths
    python create_training_data.py \
        --htr-cache ./htr_cache \
        --vh-csv ./final_velehanden_deeds.csv \
        --split-json ./ner_split.json \
        --output ./training_data.conll
"""

import argparse
import csv
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Text loading and cleaning (same as ner_baseline.py)
# =========================================================================

def clean_htr_text(raw_text: str) -> str:
    lines = raw_text.splitlines()

    # Remove block-level duplicates
    if len(lines) >= 4:
        for block_size in range(2, min(30, len(lines) // 2 + 1)):
            first_block = lines[:block_size]
            second_block = lines[block_size:block_size * 2]
            if first_block == second_block and all(l.strip() for l in first_block):
                lines = lines[block_size:]
                break

    cleaned = []
    for i, line in enumerate(lines):
        if i > 0 and line == lines[i - 1] and line.strip():
            continue
        line = line.replace("\xa0", " ")
        line = re.sub(r" {2,}", " ", line)
        line = line.strip()
        if not line:
            continue
        if (cleaned
                and cleaned[-1].endswith("-")
                and len(cleaned[-1]) >= 2
                and cleaned[-1][-2].isalpha()
                and line
                and line[0].islower()):
            cleaned[-1] = cleaned[-1][:-1] + line
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def load_inventory_pages(inv_dir: Path) -> list[tuple[str, str]]:
    """Load and clean HTR pages. Returns [(page_nr, text), ...]."""
    pages = []
    txt_files = sorted(inv_dir.glob("*.txt"), key=lambda p: int(
        p.stem.replace("page_", "")) if p.stem.replace("page_", "").isdigit() else 0)
    for f in txt_files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_htr_text(raw)
        if cleaned.strip():
            pages.append((f.stem, cleaned))
    return pages


# =========================================================================
# VeleHanden name loading
# =========================================================================

def load_vh_names(vh_csv: Path, target_invs: set) -> dict:
    """Load VH person names per inventory. Returns {inv: [name1, name2, ...]}."""
    names_by_inv = defaultdict(list)
    with open(vh_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inv = row.get("inventory_number", "").strip()
            if inv not in target_invs:
                continue
            names_str = row.get("person_names", "").strip()
            if names_str:
                for name in names_str.split("|"):
                    name = name.strip()
                    if name:
                        names_by_inv[inv].append(name)
    return dict(names_by_inv)


# =========================================================================
# Fuzzy span matching - find VH names in HTR text
# =========================================================================

def find_name_spans(text: str, names: list, threshold: float = 0.80) -> list:
    """
    Find approximate locations of VH names in HTR text using sliding window.

    For each unique VH name, finds candidate windows via a token index
    (windows that share at least one token with the name), then checks
    fuzzy similarity only on those candidates. This avoids O(names * tokens)
    full scans.

    Returns: [(start_char, end_char, matched_name, score), ...]
    """
    from rapidfuzz import fuzz

    if not names or not text.strip():
        return []

    # Deduplicate names
    unique_names = list(set(n.lower().strip() for n in names if n.strip()))
    if not unique_names:
        return []

    # Tokenize text preserving character offsets
    tokens = []  # [(start, end, token_text_lower), ...]
    for match in re.finditer(r'\S+', text):
        tokens.append((match.start(), match.end(), match.group().lower()))

    if not tokens:
        return []

    # Build token index: lowercased_token -> set of positions in tokens list
    token_index = defaultdict(set)
    for idx, (_, _, tok_lower) in enumerate(tokens):
        token_index[tok_lower].add(idx)

    all_spans = []

    for name in unique_names:
        name_tokens = name.split()
        window_size = len(name_tokens)

        if window_size == 0:
            continue

        # Find candidate window start positions:
        # any position where a name token appears, adjusted so the
        # window starting there could contain that token
        candidate_starts = set()
        for nt in name_tokens:
            for pos in token_index.get(nt, []):
                # This token could appear at any offset within the window
                for offset in range(window_size):
                    start = pos - offset
                    if 0 <= start <= len(tokens) - window_size:
                        candidate_starts.add(start)

        # Only fuzzy-match candidate windows
        for i in sorted(candidate_starts):
            window_tokens = tokens[i:i + window_size]
            window_text = " ".join(t[2] for t in window_tokens)

            score = fuzz.ratio(name, window_text) / 100.0

            if score >= threshold:
                span_start = window_tokens[0][0]
                span_end = window_tokens[-1][1]
                all_spans.append((span_start, span_end, name, score))

    # Remove overlapping spans (keep highest scoring)
    all_spans.sort(key=lambda x: -x[3])  # sort by score descending
    kept = []
    used_ranges = []

    for span in all_spans:
        s, e = span[0], span[1]
        overlaps = False
        for us, ue in used_ranges:
            if s < ue and e > us:  # overlap check
                overlaps = True
                break
        if not overlaps:
            kept.append(span)
            used_ranges.append((s, e))

    kept.sort(key=lambda x: x[0])  # sort by position
    return kept


# =========================================================================
# BIO tagging
# =========================================================================

def create_bio_tags(text: str, spans: list) -> list:
    """
    Convert text + matched spans into BIO-tagged token list.

    Returns: [(token, tag), ...] where tag is B-PER, I-PER, or O.
    Sentences are split on newlines.
    """
    # Build a set of character positions that are inside a PER span
    per_chars = {}  # char_pos -> 'B' or 'I'
    for start, end, name, score in spans:
        # Mark first token's chars as B, rest as I
        first_token = True
        for m in re.finditer(r'\S+', text[start:end]):
            abs_start = start + m.start()
            abs_end = start + m.end()
            tag = 'B' if first_token else 'I'
            for pos in range(abs_start, abs_end):
                per_chars[pos] = tag
            first_token = False

    # Tokenize and assign tags
    tagged_sentences = []
    current_sentence = []

    for line in text.split("\n"):
        if not line.strip():
            if current_sentence:
                tagged_sentences.append(current_sentence)
                current_sentence = []
            continue

        for m in re.finditer(r'\S+', line):
            token = m.group()
            # Determine tag based on the first character's position
            # (we need absolute position in the full text)
            # Since we split by newlines, we need to track offset
            pass

    # Simpler approach: work on full text directly
    tagged_sentences = []
    current_sentence = []

    lines = text.split("\n")
    char_offset = 0

    for line in lines:
        if not line.strip():
            if current_sentence:
                tagged_sentences.append(current_sentence)
                current_sentence = []
            char_offset += len(line) + 1  # +1 for newline
            continue

        for m in re.finditer(r'\S+', line):
            token = m.group()
            abs_start = char_offset + m.start()

            if abs_start in per_chars:
                tag = f"{per_chars[abs_start]}-PER"
            else:
                tag = "O"

            current_sentence.append((token, tag))

        char_offset += len(line) + 1

    if current_sentence:
        tagged_sentences.append(current_sentence)

    return tagged_sentences


# =========================================================================
# CoNLL output
# =========================================================================

def write_conll(tagged_sentences: list, output_path: Path, mode: str = "a"):
    """Write BIO-tagged sentences in CoNLL format."""
    with open(output_path, mode, encoding="utf-8") as f:
        for sentence in tagged_sentences:
            for token, tag in sentence:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


# =========================================================================
# Main pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create BIO training data via distant supervision"
    )
    parser.add_argument("--htr-cache", default="./htr_cache",
                        help="HTR cache directory")
    parser.add_argument("--vh-csv", default="./final_velehanden_deeds.csv",
                        help="VeleHanden deeds CSV")
    parser.add_argument("--split-json", default="./ner_split.json",
                        help="Train/dev/test split JSON")
    parser.add_argument("--output", default="./training_data.conll",
                        help="Output CoNLL file")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Fuzzy match threshold (default: 0.80)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only 2 inventories, print samples")
    parser.add_argument("--split", default="train",
                        choices=["train", "dev", "test"],
                        help="Which split to process (default: train)")
    args = parser.parse_args()

    htr_cache = Path(args.htr_cache)
    vh_csv = Path(args.vh_csv)
    split_path = Path(args.split_json)
    output_path = Path(args.output)

    # Load split
    split = json.loads(split_path.read_text())
    inv_list = split[args.split]
    if args.test:
        inv_list = inv_list[:2]

    log.info("Split '%s': %d inventories%s",
             args.split, len(inv_list), " (TEST MODE)" if args.test else "")

    # Load VH names
    log.info("Loading VH names...")
    vh_names = load_vh_names(vh_csv, set(inv_list))
    log.info("VH names loaded for %d inventories", len(vh_names))

    # Process inventories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear output file
    output_path.write_text("", encoding="utf-8")

    total_tokens = 0
    total_per_tokens = 0
    total_sentences = 0
    total_spans = 0
    skipped = 0

    t0 = time.time()

    for i, inv in enumerate(inv_list, 1):
        inv_dir = htr_cache / inv
        if not inv_dir.exists():
            log.warning("inv %s: no HTR cache directory, skipping", inv)
            skipped += 1
            continue

        pages = load_inventory_pages(inv_dir)
        if not pages:
            log.warning("inv %s: no pages after cleaning", inv)
            skipped += 1
            continue

        inv_names = vh_names.get(inv, [])
        if not inv_names:
            log.warning("inv %s: no VH names, skipping", inv)
            skipped += 1
            continue

        # Process each page
        inv_spans = 0
        inv_sentences = []

        for page_nr, page_text in pages:
            spans = find_name_spans(page_text, inv_names, args.threshold)
            inv_spans += len(spans)

            tagged = create_bio_tags(page_text, spans)
            inv_sentences.extend(tagged)

        # Write to CoNLL
        write_conll(inv_sentences, output_path, mode="a")

        # Stats
        inv_tokens = sum(len(s) for s in inv_sentences)
        inv_per = sum(1 for s in inv_sentences for t, tag in s if tag != "O")
        total_tokens += inv_tokens
        total_per_tokens += inv_per
        total_sentences += len(inv_sentences)
        total_spans += inv_spans

        elapsed = time.time() - t0
        rate = i / elapsed * 60 if elapsed > 0 else 0

        if args.test or i % 10 == 0 or i == len(inv_list):
            log.info(
                "[%d/%d] inv %s: %d pages, %d names, %d spans matched, "
                "%d tokens (%d PER) | %.1f inv/min",
                i, len(inv_list), inv, len(pages), len(inv_names),
                inv_spans, inv_tokens, inv_per, rate
            )

        # In test mode, show sample output
        if args.test and inv_sentences:
            log.info("  Sample tagged output (first sentence with PER):")
            for sent in inv_sentences:
                has_per = any(tag != "O" for _, tag in sent)
                if has_per:
                    for token, tag in sent[:30]:
                        marker = f"  <-- {tag}" if tag != "O" else ""
                        log.info("    %-30s %s%s", token, tag, marker)
                    if len(sent) > 30:
                        log.info("    ... (%d more tokens)", len(sent) - 30)
                    break

    elapsed = time.time() - t0

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING DATA CREATION SUMMARY")
    log.info("=" * 60)
    log.info("Inventories processed: %d (skipped: %d)", len(inv_list) - skipped, skipped)
    log.info("Total sentences: %d", total_sentences)
    log.info("Total tokens: %d", total_tokens)
    log.info("Total PER tokens: %d (%.2f%%)",
             total_per_tokens, total_per_tokens / max(total_tokens, 1) * 100)
    log.info("Total name spans matched: %d", total_spans)
    log.info("Time: %.0fs", elapsed)
    log.info("Output: %s (%.1f MB)",
             output_path, output_path.stat().st_size / 1e6 if output_path.exists() else 0)

    # Verify output is valid CoNLL
    if output_path.exists():
        lines = output_path.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]
        bad_lines = [l for l in non_empty if "\t" not in l]
        if bad_lines:
            log.error("VALIDATION FAILED: %d lines without tab separator", len(bad_lines))
            for bl in bad_lines[:5]:
                log.error("  Bad line: %r", bl)
        else:
            log.info("Validation: all %d non-empty lines have correct format", len(non_empty))


if __name__ == "__main__":
    main()
