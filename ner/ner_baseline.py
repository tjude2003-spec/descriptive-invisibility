#!/usr/bin/env python3
"""
NER Baseline Pipeline
======================
Runs spaCy NER on cached HTR pages, evaluates against VeleHanden ground truth.

Three modes:
    1. extract  — Run NER on cached HTR pages, output extracted entities
    2. evaluate — Compare NER extractions against VeleHanden for overlap inventories
    3. both     — Extract then evaluate (default)

USAGE:
    # Install dependencies first:
    #   pip install spacy rapidfuzz
    #   python -m spacy download nl_core_news_lg

    # Run on all completed inventories in htr_cache
    python3 ner_baseline.py

    # Run on specific inventories
    python3 ner_baseline.py --invs 10023 14256 2413

    # Extract only (no evaluation)
    python3 ner_baseline.py --mode extract

    # Evaluate only (requires prior extraction)
    python3 ner_baseline.py --mode evaluate

    # Use a different spaCy model
    python3 ner_baseline.py --model nl_core_news_md

Outputs:
    ner_output/entities_{model}.csv       — All extracted entities
    ner_output/eval_per_inventory.csv     — Per-inventory precision/recall/F1
    ner_output/eval_summary.txt           — Aggregate evaluation report
    ner_output/eval_match_details.csv     — Individual match/miss details
"""

import csv
import json
import re
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# TEXT CLEANING
# =========================================================================

def remove_block_duplicates(lines: list[str]) -> list[str]:
    """
    Remove block-level duplicates at the start of a page.

    Transkribus sometimes produces overlapping text regions that repeat
    the first N lines of a page. This detects when lines[0:N] == lines[N:2N]
    and removes the duplicate block.
    """
    if len(lines) < 4:
        return lines

    # Try block sizes from 2 up to half the page
    for block_size in range(2, min(30, len(lines) // 2 + 1)):
        first_block = lines[:block_size]
        second_block = lines[block_size:block_size * 2]
        if first_block == second_block and all(l.strip() for l in first_block):
            # Found a duplicate block — remove the first copy
            return lines[block_size:]

    return lines


def clean_htr_text(raw_text: str) -> str:
    """
    Clean HTR page text for NER processing.

    Handles:
    - Block-level duplicates at page start (Transkribus region overlap artifact)
    - Duplicate consecutive lines
    - Double/irregular whitespace
    - Non-breaking spaces
    - Word-splitting hyphens at line boundaries (rare but present)
    - Preserves original line structure for page-level traceability
    """
    lines = raw_text.splitlines()

    # First pass: remove block-level duplicates
    lines = remove_block_duplicates(lines)

    # Second pass: line-level cleaning
    cleaned = []

    for i, line in enumerate(lines):
        # Skip duplicate consecutive lines
        if i > 0 and line == lines[i - 1] and line.strip():
            continue

        # Normalise whitespace within line
        line = line.replace("\xa0", " ")      # non-breaking space
        line = re.sub(r" {2,}", " ", line)    # collapse multiple spaces
        line = line.strip()

        if not line:
            continue

        # Join word-splitting hyphens: "vooraf-\nsterven" → "voorafsterven"
        # Only when the hyphen is preceded by a letter (not standalone dashes)
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


def load_inventory_text(inv_dir: Path) -> dict[str, str]:
    """
    Load and clean all HTR pages for an inventory.
    Returns {page_nr: cleaned_text}.
    """
    pages = {}
    for f in sorted(inv_dir.glob("*.txt"), key=lambda p: int(p.stem.replace("page_", ""))):
        stem = f.stem.replace("page_", "")
        raw = f.read_text(encoding="utf-8")
        cleaned = clean_htr_text(raw)
        if cleaned.strip():
            pages[stem] = cleaned
    return pages


# =========================================================================
# NER EXTRACTION
# =========================================================================

def run_ner_on_inventory(nlp, inv_nr: str, pages: dict[str, str]) -> list[dict]:
    """
    Run spaCy NER on all pages of an inventory.
    Returns list of entity dicts.

    Uses nlp.pipe() for batched processing — significantly faster than
    calling nlp() per page.
    """
    entities = []
    page_nrs = list(pages.keys())
    texts = list(pages.values())

    # spaCy's nlp.pipe() processes texts in batches for efficiency
    for page_nr, doc in zip(page_nrs, nlp.pipe(texts, batch_size=50)):
        for ent in doc.ents:
            entities.append({
                "inventory_number": inv_nr,
                "page_number": page_nr,
                "entity_text": ent.text,
                "entity_type": ent.label_,
                "char_start": ent.start_char,
                "char_end": ent.end_char,
            })

    return entities


# =========================================================================
# EVALUATION AGAINST VELEHANDEN
# =========================================================================

# Formulaic archival Dutch terms that spaCy frequently misclassifies as PERSON.
# These are structural features of notarial deed language, not names.
FORMULAIC_STOPLIST = {
    # Legal/procedural terms
    "den requirant", "den comparant", "de comparant", "den comparanten",
    "de comparanten", "den requiranten", "de requirant",
    "den testateur", "de testateur", "den testatrice", "de testatrice",
    # Court/authority references
    "den edele hove", "den edele hove van holland", "den ed",
    "den hove", "den hove van holland", "den weled",
    "den edele gerechte", "den edele achtbare",
    # Titles and honorifics (standalone)
    "notaris publicq", "notaris publycq", "notaris publyk",
    "juffr", "juffrouw", "monsr", "de heer", "den heere",
    "mijnheer", "mevrouw", "sr",
    # Common HTR noise tagged as PERSON
    "sijn", "sy get", "fo", "ed",
    "gedateert vanden",
}

def is_formulaic(entity_text: str) -> bool:
    """Check if an entity is a formulaic archival term, not a real name."""
    return entity_text.lower().strip() in FORMULAIC_STOPLIST


def load_notary_names(summary_csv: Path, target_invs: set[str]) -> dict[str, str]:
    """
    Load notary name per inventory from VeleHanden inventory summary.
    Returns {inv_number: notary_name}.
    """
    notaries = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv = row["inventory_number"]
            if inv in target_invs and row.get("notary"):
                notaries[inv] = row["notary"]
    return notaries


def is_notary_name(entity_text: str, notary_name: str) -> bool:
    """
    Check if a NER entity is (part of) the notary's name.
    Handles partial matches: "Zweerts" matches "PHILIP ZWEERTS",
    "P Zweerts" matches "PHILIP ZWEERTS".
    """
    from rapidfuzz import fuzz

    ent_lower = entity_text.lower().strip()
    notary_lower = notary_name.lower().strip()

    # Exact match
    if ent_lower == notary_lower:
        return True

    # Entity is a substring of notary name (e.g., "Zweerts" in "Philip Zweerts")
    notary_parts = notary_lower.split()
    if ent_lower in notary_parts:
        return True

    # Fuzzy match for partial/abbreviated forms (e.g., "P Zweerts" vs "Philip Zweerts")
    score = fuzz.token_sort_ratio(ent_lower, notary_lower) / 100.0
    if score >= 0.75:
        return True

    # Entity is a contiguous subsequence of notary name tokens
    if len(ent_lower) >= 4 and ent_lower in notary_lower:
        return True

    return False


def filter_ner_entities(
    ner_entities: list[dict],
    notary_name: str | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """
    Filter NER PERSON entities, removing notary names and formulaic terms.
    Returns (filtered_entities, removal_counts).
    """
    person_labels = {"PER", "PERSON"}
    filtered = []
    removed = {"notary": 0, "formulaic": 0, "kept": 0}

    for e in ner_entities:
        if e["entity_type"] not in person_labels:
            filtered.append(e)
            continue

        if is_formulaic(e["entity_text"]):
            removed["formulaic"] += 1
            continue

        if notary_name and is_notary_name(e["entity_text"], notary_name):
            removed["notary"] += 1
            continue

        removed["kept"] += 1
        filtered.append(e)

    return filtered, removed


def load_velehanden_names(vh_csv_paths: list[Path], target_invs: set[str]) -> dict[str, list[str]]:
    """
    Load VeleHanden person names per inventory number.
    Returns {inv_number: [name1, name2, ...]}.
    Only loads inventories in target_invs for memory efficiency.
    """
    names_by_inv = defaultdict(list)

    for csv_path in vh_csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inv = row["inventory_number"]
                if inv not in target_invs:
                    continue
                if row["person_names"]:
                    # Names are pipe-delimited within each deed
                    for name in row["person_names"].split("|"):
                        name = name.strip()
                        if name:
                            names_by_inv[inv].append(name)

    return dict(names_by_inv)


def evaluate_inventory(
    ner_entities: list[dict],
    vh_names: list[str],
    fuzzy_threshold: float = 0.85,
) -> dict:
    """
    Evaluate NER PER entities against VeleHanden names for one inventory.

    Strategy:
    - For each VeleHanden name, check if any NER PER entity fuzzy-matches it (recall)
    - For each NER PER entity, check if it fuzzy-matches any VeleHanden name (precision)

    Uses rapidfuzz.process.cdist to compute the full similarity matrix in C
    with multithreading, then extracts best matches via numpy — typically
    ~20x faster than iterative fuzzy matching.

    Returns evaluation metrics dict.
    """
    from rapidfuzz import fuzz, process as rfprocess
    import numpy as np

    # Filter to person entities (spaCy uses PERSON in OntoNotes, PER in CoNLL)
    person_labels = {"PER", "PERSON"}
    ner_per = [e for e in ner_entities if e["entity_type"] in person_labels]
    ner_texts = [e["entity_text"] for e in ner_per]

    if not vh_names and not ner_texts:
        return {
            "vh_names": 0, "vh_unique": 0, "ner_per_count": 0,
            "ner_all_count": len(ner_entities),
            "vh_matched": 0, "ner_matched": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "match_details": [],
        }

    # --- Deduplicate after lowercasing (case-insensitive matching) ---
    # Keep a map from lowercase -> first original-case form for output
    ner_lower_to_orig = {}
    for t in ner_texts:
        lo = t.lower()
        if lo not in ner_lower_to_orig:
            ner_lower_to_orig[lo] = t
    ner_dedup = list(ner_lower_to_orig.keys())

    vh_lower_to_orig = {}
    for n in vh_names:
        lo = n.lower()
        if lo not in vh_lower_to_orig:
            vh_lower_to_orig[lo] = n
    vh_dedup = list(vh_lower_to_orig.keys())

    # Handle edge case: one side is empty
    if not ner_dedup or not vh_dedup:
        vh_match_details = [{"vh_name": n, "best_ner_match": "", "similarity": 0.0,
                            "matched": False, "direction": "recall"} for n in vh_names]
        ner_match_details = [{"ner_entity": t, "best_vh_match": "", "similarity": 0.0,
                             "matched": False, "direction": "precision"} for t in ner_texts]
        return {
            "vh_names": len(vh_names), "vh_unique": len(set(vh_names)),
            "ner_per_count": len(ner_texts), "ner_all_count": len(ner_entities),
            "vh_matched": 0, "ner_matched": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "match_details": vh_match_details + ner_match_details,
        }

    # --- Compute full similarity matrix in C (rows=NER, cols=VH) ---
    matrix = rfprocess.cdist(
        ner_dedup, vh_dedup,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
    ) / 100.0

    # --- Precision (NER -> VH): best VH match for each unique NER string ---
    ner_best_vh_idx = np.argmax(matrix, axis=1)
    ner_best_scores = matrix[np.arange(len(ner_dedup)), ner_best_vh_idx]

    ner_lower_results = {}
    for i, ner_lo in enumerate(ner_dedup):
        best_vh_lo = vh_dedup[ner_best_vh_idx[i]]
        ner_lower_results[ner_lo] = (vh_lower_to_orig[best_vh_lo], round(float(ner_best_scores[i]), 3))

    # --- Recall (VH -> NER): best NER match for each unique VH string ---
    vh_best_ner_idx = np.argmax(matrix, axis=0)
    vh_best_scores = matrix[vh_best_ner_idx, np.arange(len(vh_dedup))]

    vh_lower_results = {}
    for j, vh_lo in enumerate(vh_dedup):
        best_ner_lo = ner_dedup[vh_best_ner_idx[j]]
        vh_lower_results[vh_lo] = (ner_lower_to_orig[best_ner_lo], round(float(vh_best_scores[j]), 3))

    # --- Expand deduplicated results back to full lists ---
    vh_matched = 0
    vh_match_details = []
    for vh_name in vh_names:
        best_ner, best_score = vh_lower_results[vh_name.lower()]
        matched = best_score >= fuzzy_threshold
        if matched:
            vh_matched += 1
        vh_match_details.append({
            "vh_name": vh_name,
            "best_ner_match": best_ner,
            "similarity": best_score,
            "matched": matched,
            "direction": "recall",
        })

    ner_matched = 0
    ner_match_details = []
    for ner_text in ner_texts:
        best_vh, best_score = ner_lower_results[ner_text.lower()]
        matched = best_score >= fuzzy_threshold
        if matched:
            ner_matched += 1
        ner_match_details.append({
            "ner_entity": ner_text,
            "best_vh_match": best_vh,
            "similarity": best_score,
            "matched": matched,
            "direction": "precision",
        })

    # --- Metrics ---
    precision = ner_matched / max(len(ner_texts), 1)
    recall = vh_matched / max(len(vh_names), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9))

    return {
        "vh_names": len(vh_names),
        "vh_unique": len(set(vh_names)),
        "ner_per_count": len(ner_texts),
        "ner_all_count": len(ner_entities),
        "vh_matched": vh_matched,
        "ner_matched": ner_matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "match_details": vh_match_details + ner_match_details,
    }


# =========================================================================
# ENTITY TYPE SUMMARY
# =========================================================================

def entity_type_summary(entities: list[dict]) -> dict[str, int]:
    """Count entities by type across all inventories."""
    counts = defaultdict(int)
    for e in entities:
        counts[e["entity_type"]] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# =========================================================================
# MAIN
# =========================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="NER baseline pipeline: extract entities and evaluate against VeleHanden",
    )
    p.add_argument("--mode", choices=["extract", "evaluate", "both"], default="both",
                   help="Pipeline mode (default: both)")
    p.add_argument("--model", default="nl_core_news_lg",
                   help="spaCy model name (default: nl_core_news_lg)")
    p.add_argument("--invs", nargs="+", default=None,
                   help="Specific inventory numbers (default: all completed in cache)")
    p.add_argument("--data-dir", default=".",
                   help="Directory containing alignment.json and VeleHanden CSVs")
    p.add_argument("--cache-dir", default="./htr_cache",
                   help="Directory with cached HTR pages")
    p.add_argument("--output-dir", default="./ner_output",
                   help="Output directory for results")
    p.add_argument("--vh-csv", nargs="+", default=None,
                   help="Path(s) to VeleHanden deeds CSV(s). If not given, falls back to "
                        "data-dir/final_velehanden_deeds.csv, then data-dir/velehanden_deeds_part*.csv")
    p.add_argument("--fuzzy-threshold", type=float, default=0.85,
                   help="Fuzzy match threshold for evaluation (default: 0.85)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of inventories to process (for testing)")
    p.add_argument("--split-json", default=None,
                   help="Path to ner_split.json for train/dev/test inventory lists")
    p.add_argument("--split", default=None, choices=["train", "dev", "test"],
                   help="Which split to evaluate (requires --split-json)")

    args = p.parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Resolve which inventories to process
    # -----------------------------------------------------------------
    if args.split_json and args.split:
        with open(args.split_json) as f:
            split_data = json.load(f)
        inv_list = split_data[args.split]
        log.info(f"Using {args.split} split from {args.split_json}: {len(inv_list)} inventories")
    elif args.invs:
        inv_list = args.invs
    else:
        # All inventories with cached HTR pages
        inv_list = sorted(
            [d.name for d in cache_dir.iterdir()
             if d.is_dir() and not d.name.startswith("_") and any(d.glob("*.txt"))],
            key=lambda x: int(x) if x.isdigit() else 0,
        )

    if args.limit:
        inv_list = inv_list[:args.limit]

    log.info(f"Inventories to process: {len(inv_list)}")

    # -----------------------------------------------------------------
    # Determine which are overlap (have VeleHanden ground truth)
    # -----------------------------------------------------------------
    alignment_path = data_dir / "alignment.json"
    overlap_invs = set()
    if alignment_path.exists():
        with open(alignment_path) as f:
            alignment = json.load(f)
        overlap_invs = set(pair[0] for pair in alignment["overlap"])
    eval_invs = [inv for inv in inv_list if inv in overlap_invs]
    log.info(f"Of which overlap (evaluable): {len(eval_invs)}")

    # -----------------------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------------------
    all_entities = []
    entities_path = output_dir / f"entities_{args.model}.csv"

    if args.mode in ("extract", "both"):
        log.info(f"Loading spaCy model: {args.model}")
        import spacy
        nlp = spacy.load(args.model)
        log.info(f"Model loaded. Pipeline: {nlp.pipe_names}")

        t_start = time.time()
        for i, inv in enumerate(inv_list, 1):
            inv_dir = cache_dir / inv
            if not inv_dir.exists():
                log.warning(f"inv {inv}: no cache directory, skipping")
                continue

            pages = load_inventory_text(inv_dir)
            if not pages:
                log.warning(f"inv {inv}: no non-empty pages after cleaning")
                continue

            entities = run_ner_on_inventory(nlp, inv, pages)
            all_entities.extend(entities)

            per_count = sum(1 for e in entities if e["entity_type"] in ("PER", "PERSON"))
            elapsed = time.time() - t_start
            rate = i / elapsed * 60 if elapsed > 0 else 0

            log.info(
                f"[{i}/{len(inv_list)}] inv {inv}: "
                f"{len(pages)} pages, {len(entities)} entities ({per_count} PER) | "
                f"{rate:.1f} inv/min"
            )

        # Save entities CSV
        if all_entities:
            fieldnames = ["inventory_number", "page_number", "entity_text",
                          "entity_type", "char_start", "char_end"]
            with open(entities_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_entities)
            log.info(f"Saved {len(all_entities)} entities to {entities_path}")

            # Entity type distribution
            type_counts = entity_type_summary(all_entities)
            log.info(f"Entity types: {type_counts}")
        else:
            log.warning("No entities extracted")

    # -----------------------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------------------
    if args.mode in ("evaluate", "both"):
        # Load entities from CSV if evaluate-only
        if args.mode == "evaluate":
            if not entities_path.exists():
                log.error(f"No entities file at {entities_path} — run extract first")
                sys.exit(1)
            log.info(f"Loading entities from {entities_path}")
            with open(entities_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_entities = list(reader)

        if not eval_invs:
            log.warning("No overlap inventories to evaluate")
            return

        # Load VeleHanden ground truth
        log.info("Loading VeleHanden ground truth...")
        if args.vh_csv:
            vh_csvs = [Path(p) for p in args.vh_csv]
        else:
            # Try final_velehanden_deeds.csv first, fall back to part files
            single = data_dir / "final_velehanden_deeds.csv"
            if single.exists():
                vh_csvs = [single]
            else:
                vh_csvs = sorted(data_dir.glob("velehanden_deeds_part*.csv"))
        if not vh_csvs:
            log.error("No VeleHanden deeds CSV found. Use --vh-csv to specify path(s).")
            sys.exit(1)
        missing = [p for p in vh_csvs if not p.exists()]
        if missing:
            log.error(f"VH CSV(s) not found: {missing}")
            sys.exit(1)
        log.info(f"Using VH deeds file(s): {[str(p) for p in vh_csvs]}")

        vh_names = load_velehanden_names(vh_csvs, set(eval_invs))
        log.info(f"Loaded VH names for {len(vh_names)} inventories")

        # Load notary names for filtering
        summary_csv = data_dir / "final_velehanden_inventory_summary_.csv"
        notary_names = {}
        if summary_csv.exists():
            notary_names = load_notary_names(summary_csv, set(eval_invs))
            log.info(f"Loaded notary names for {len(notary_names)} inventories")
        else:
            log.warning("No final_velehanden_inventory_summary_.csv — skipping notary filtering")

        # Group NER entities by inventory
        ner_by_inv = defaultdict(list)
        for e in all_entities:
            ner_by_inv[e["inventory_number"]].append(e)

        # Evaluate each overlap inventory — both raw and filtered
        eval_results_raw = []
        eval_results_filtered = []
        all_match_details_raw = []
        all_match_details_filtered = []
        total_removed = {"notary": 0, "formulaic": 0, "kept": 0}

        log.info(f"Evaluating {len(eval_invs)} overlap inventories (threshold={args.fuzzy_threshold})...")
        for i, inv in enumerate(eval_invs, 1):
            inv_entities = ner_by_inv.get(inv, [])
            inv_vh_names = vh_names.get(inv, [])

            if not inv_entities and not inv_vh_names:
                continue

            # --- Raw evaluation (no filtering) ---
            result_raw = evaluate_inventory(inv_entities, inv_vh_names, args.fuzzy_threshold)
            eval_results_raw.append({
                "inventory_number": inv,
                "vh_names": result_raw["vh_names"],
                "vh_unique": result_raw["vh_unique"],
                "ner_per_count": result_raw["ner_per_count"],
                "ner_all_count": result_raw["ner_all_count"],
                "precision": result_raw["precision"],
                "recall": result_raw["recall"],
                "f1": result_raw["f1"],
            })
            for detail in result_raw["match_details"]:
                detail["inventory_number"] = inv
                all_match_details_raw.append(detail)

            # --- Filtered evaluation (notary + formulaic removed) ---
            notary = notary_names.get(inv)
            filtered_entities, removed = filter_ner_entities(inv_entities, notary)
            for k in total_removed:
                total_removed[k] += removed[k]

            result_filt = evaluate_inventory(filtered_entities, inv_vh_names, args.fuzzy_threshold)
            eval_results_filtered.append({
                "inventory_number": inv,
                "notary": notary or "",
                "vh_names": result_filt["vh_names"],
                "vh_unique": result_filt["vh_unique"],
                "ner_per_count": result_filt["ner_per_count"],
                "ner_all_count": result_filt["ner_all_count"],
                "removed_notary": removed["notary"],
                "removed_formulaic": removed["formulaic"],
                "precision": result_filt["precision"],
                "recall": result_filt["recall"],
                "f1": result_filt["f1"],
            })
            for detail in result_filt["match_details"]:
                detail["inventory_number"] = inv
                all_match_details_filtered.append(detail)

            if i % 10 == 0 or i == len(eval_invs):
                log.info(f"  evaluated {i}/{len(eval_invs)}")

        # ----- Save per-inventory results (filtered) -----
        eval_csv_path = output_dir / "eval_per_inventory.csv"
        eval_fields = ["inventory_number", "notary", "vh_names", "vh_unique",
                       "ner_per_count", "ner_all_count",
                       "removed_notary", "removed_formulaic",
                       "precision", "recall", "f1"]
        with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=eval_fields)
            writer.writeheader()
            writer.writerows(eval_results_filtered)
        log.info(f"Per-inventory results saved to {eval_csv_path}")

        # ----- Save match details (filtered) -----
        details_path = output_dir / "eval_match_details.csv"
        detail_fields = ["inventory_number", "direction", "vh_name", "ner_entity",
                         "best_ner_match", "best_vh_match", "similarity", "matched"]
        with open(details_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_match_details_filtered)
        log.info(f"Match details saved to {details_path}")

        # ----- Aggregate summary -----
        def compute_summary(eval_results, match_details, label):
            n = len(eval_results)
            if n == 0:
                return []
            avg_p = sum(r["precision"] for r in eval_results) / n
            avg_r = sum(r["recall"] for r in eval_results) / n
            avg_f1 = sum(r["f1"] for r in eval_results) / n
            total_vh = sum(r["vh_names"] for r in eval_results)
            total_ner = sum(r["ner_per_count"] for r in eval_results)

            total_vh_matched = sum(
                d["matched"] for d in match_details if d["direction"] == "recall"
            )
            total_ner_matched = sum(
                d["matched"] for d in match_details if d["direction"] == "precision"
            )
            micro_p = total_ner_matched / max(total_ner, 1)
            micro_r = total_vh_matched / max(total_vh, 1)
            micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

            lines = []
            lines.append(f"  [{label}]")
            lines.append(f"  Total VH names:         {total_vh}")
            lines.append(f"  Total NER PER entities: {total_ner}")
            lines.append(f"  Macro P / R / F1:       {avg_p:.4f} / {avg_r:.4f} / {avg_f1:.4f}")
            lines.append(f"  Micro P / R / F1:       {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
            return lines

        if eval_results_raw:
            report = []
            report.append("=" * 60)
            report.append(f"NER EVALUATION REPORT — {args.model}")
            report.append(f"Date: {datetime.now().isoformat()}")
            report.append(f"Fuzzy threshold: {args.fuzzy_threshold}")
            report.append("=" * 60)
            report.append("")
            report.append(f"Inventories evaluated: {len(eval_results_raw)}")
            report.append("")

            report.extend(compute_summary(eval_results_raw, all_match_details_raw, "RAW"))
            report.append("")
            report.extend(compute_summary(eval_results_filtered, all_match_details_filtered, "FILTERED"))
            report.append("")

            report.append(f"FILTERING SUMMARY:")
            report.append(f"  Notary mentions removed:    {total_removed['notary']}")
            report.append(f"  Formulaic terms removed:    {total_removed['formulaic']}")
            report.append(f"  Person entities retained:   {total_removed['kept']}")
            report.append("")

            # Entity type distribution
            type_counts = entity_type_summary(all_entities)
            report.append("ENTITY TYPE DISTRIBUTION:")
            for etype, count in type_counts.items():
                report.append(f"  {etype:12s} {count:>8,}")
            report.append("")

            # Best/worst inventories (filtered)
            by_f1 = sorted(eval_results_filtered, key=lambda r: r["f1"])
            report.append("BOTTOM 5 (lowest F1, filtered):")
            for r in by_f1[:5]:
                report.append(f"  inv {r['inventory_number']:>6s}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} (VH={r['vh_names']}, NER={r['ner_per_count']}, notary={r.get('notary','')})")
            report.append("")
            report.append("TOP 5 (highest F1, filtered):")
            for r in by_f1[-5:]:
                report.append(f"  inv {r['inventory_number']:>6s}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} (VH={r['vh_names']}, NER={r['ner_per_count']}, notary={r.get('notary','')})")

            # False positives (filtered — these are the real problem cases)
            report.append("")
            report.append("SAMPLE FALSE POSITIVES (after filtering):")
            fps = [d for d in all_match_details_filtered
                   if d["direction"] == "precision" and not d["matched"]][:15]
            for fp in fps:
                report.append(f"  inv {fp['inventory_number']}: \"{fp.get('ner_entity', '?')}\" (best VH: \"{fp.get('best_vh_match', '?')}\" sim={fp['similarity']})")

            # False negatives
            report.append("")
            report.append("SAMPLE FALSE NEGATIVES (VH names not found by NER):")
            fns = [d for d in all_match_details_filtered
                   if d["direction"] == "recall" and not d["matched"]][:15]
            for fn in fns:
                report.append(f"  inv {fn['inventory_number']}: \"{fn.get('vh_name', '?')}\" (best NER: \"{fn.get('best_ner_match', '?')}\" sim={fn['similarity']})")

            report_text = "\n".join(report)
            print("\n" + report_text)

            summary_path = output_dir / "eval_summary.txt"
            summary_path.write_text(report_text, encoding="utf-8")
            log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
