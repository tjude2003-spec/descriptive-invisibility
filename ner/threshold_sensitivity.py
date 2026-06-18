#!/usr/bin/env python3
"""
Threshold sensitivity check for the NER evaluation.

Re-runs evaluation at thresholds 0.80, 0.85, 0.90 against the cached
entity extractions for spaCy off-shelf and spaCy adapted. No
re-extraction needed — reads existing entities_{model}.csv files.

Purpose: confirm that the model ranking (spaCy adapted > spaCy off-shelf)
is robust to threshold choice in a reasonable range, so the headline
0.85 number doesn't depend on cherry-picking.

Outputs:
    threshold_sensitivity.csv — long-format table:
        model, threshold, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1
    threshold_sensitivity.txt — human-readable summary

Usage:
    python3 threshold_sensitivity.py
       [--entities-dir /path/to/ner_output]
       [--vh-csv /path/to/velehanden_deeds_part*.csv ...]
       [--split-json /path/to/ner_split.json]
       [--output-dir /path/to/output]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist as rfp_cdist


THRESHOLDS = [0.80, 0.85, 0.90]
PERSON_LABELS = {"PER", "PERSON"}


def load_entities(csv_path: Path, test_invs: set) -> dict:
    """
    Load entity CSV produced by ner_baseline.py and group by inventory.
    Filters to test inventories only.

    Returns: {inv: [name_string, name_string, ...]}
    """
    by_inv = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv = row.get("inventory_number", "").strip()
            if inv not in test_invs:
                continue
            etype = row.get("entity_type", "").strip()
            if etype not in PERSON_LABELS:
                continue
            text = row.get("entity_text", "").strip()
            if text:
                by_inv[inv].append(text)
    return dict(by_inv)


def load_vh_ground_truth(vh_csvs: list, test_invs: set) -> dict:
    """Load VeleHanden ground truth, filtered to test inventories."""
    by_inv = defaultdict(list)
    for csv_path in vh_csvs:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inv = row.get("inventory_number", "").strip()
                if inv not in test_invs:
                    continue
                names_str = row.get("person_names", "").strip()
                if names_str:
                    for name in names_str.split("|"):
                        name = name.strip()
                        if name:
                            by_inv[inv].append(name)
    return dict(by_inv)


def evaluate_inventory(ner_names: list, vh_names: list, threshold: float) -> dict:
    """
    Per-inventory evaluation matching ner_baseline.py logic exactly.

    Critical detail: matching decisions are made on deduplicated lowercase
    strings (so duplicate spelling variants don't multiply false-positive
    weight), but precision and recall denominators are MENTION COUNTS,
    not unique counts. Each mention of a matched name contributes; each
    mention of an unmatched name penalises.
    """
    if not vh_names and not ner_names:
        return {"p": 1.0, "r": 1.0, "f1": 1.0,
                "n_pred": 0, "n_gold": 0,
                "tp_p": 0, "tp_r": 0}

    pred_unique = list(set(n.lower() for n in ner_names if n.strip()))
    gold_unique = list(set(n.lower() for n in vh_names if n.strip()))

    if not pred_unique or not gold_unique:
        return {"p": 0.0, "r": 0.0, "f1": 0.0,
                "n_pred": len(ner_names), "n_gold": len(vh_names),
                "tp_p": 0, "tp_r": 0}

    matrix = rfp_cdist(
        pred_unique, gold_unique,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
    ) / 100.0

    pred_best = matrix.max(axis=1)
    gold_best = matrix.max(axis=0)

    # Per-string match flags
    pred_matched_lookup = {
        pred_unique[i]: pred_best[i] >= threshold
        for i in range(len(pred_unique))
    }
    gold_matched_lookup = {
        gold_unique[j]: gold_best[j] >= threshold
        for j in range(len(gold_unique))
    }

    # Expand back to mention-level counts (this matches ner_baseline.py)
    pred_matched_mentions = sum(
        1 for n in ner_names if pred_matched_lookup.get(n.lower(), False)
    )
    gold_matched_mentions = sum(
        1 for n in vh_names if gold_matched_lookup.get(n.lower(), False)
    )

    n_pred_mentions = len(ner_names)
    n_gold_mentions = len(vh_names)

    p = pred_matched_mentions / max(n_pred_mentions, 1)
    r = gold_matched_mentions / max(n_gold_mentions, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    return {"p": p, "r": r, "f1": f1,
            "n_pred": n_pred_mentions, "n_gold": n_gold_mentions,
            "tp_p": pred_matched_mentions, "tp_r": gold_matched_mentions}


def evaluate_at_threshold(ner_dict: dict, vh_dict: dict,
                           threshold: float) -> dict:
    """
    Evaluate one model at one threshold across all test inventories.
    Returns macro and micro metrics.
    """
    per_inv = []
    total_tp_p = 0
    total_tp_r = 0
    total_pred = 0
    total_gold = 0

    for inv in sorted(vh_dict.keys()):
        result = evaluate_inventory(
            ner_dict.get(inv, []),
            vh_dict[inv],
            threshold,
        )
        per_inv.append(result)
        total_tp_p += result["tp_p"]
        total_tp_r += result["tp_r"]
        total_pred += result["n_pred"]
        total_gold += result["n_gold"]

    if per_inv:
        macro_p = float(np.mean([r["p"] for r in per_inv]))
        macro_r = float(np.mean([r["r"] for r in per_inv]))
        macro_f1 = float(np.mean([r["f1"] for r in per_inv]))
    else:
        macro_p = macro_r = macro_f1 = 0.0

    micro_p = total_tp_p / max(total_pred, 1)
    micro_r = total_tp_r / max(total_gold, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    return {
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "n_inventories": len(per_inv),
    }


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATA_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--entities-dir",
                   default=None,
                   help="Directory containing entities_*.csv files")
    p.add_argument("--vh-csv", nargs="+", default=None,
                   help="Path(s) to velehanden_deeds_part*.csv")
    p.add_argument("--split-json",
                   default=str(DATA_DIR / "ner_split.json"))
    p.add_argument("--output-dir",
                   default="./threshold_sensitivity")
    args = p.parse_args()

    entities_dir = Path(args.entities_dir) if args.entities_dir else DATA_DIR / "ner_output"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve VH CSVs if not specified
    if args.vh_csv:
        vh_csvs = [Path(p) for p in args.vh_csv]
    else:
        candidate_dirs = [
            DATA_DIR,
        ]
        vh_csvs = []
        for d in candidate_dirs:
            vh_csvs = sorted(d.glob("velehanden_deeds_part*.csv"))
            if vh_csvs:
                break
        if not vh_csvs:
            print("[error] No velehanden_deeds_part*.csv found. "
                  "Pass --vh-csv with explicit paths.", file=sys.stderr)
            sys.exit(1)

    # Load split
    with open(args.split_json) as f:
        split = json.load(f)
    test_invs = set(split["test"])
    print(f"Test inventories: {len(test_invs)}")

    # Load VH ground truth
    print(f"Loading VH ground truth from {len(vh_csvs)} CSV(s)...")
    vh_dict = load_vh_ground_truth(vh_csvs, test_invs)
    print(f"  Inventories with VH data: {len(vh_dict)}")
    print(f"  Total VH name mentions:   {sum(len(v) for v in vh_dict.values()):,}")

    # Find entity files
    entity_files = {}
    candidates = {
        "spacy_off_shelf": ["entities_nl_core_news_lg.csv",
                              "entities_baseline.csv",
                              "entities_spacy_baseline.csv"],
        "spacy_adapted": ["entities_adapted.csv",
                           "entities_adapted_model.csv"],
    }
    for label, names in candidates.items():
        for name in names:
            path = entities_dir / name
            if path.exists():
                entity_files[label] = path
                break
        # Also look in subdirectories
        if label not in entity_files:
            for sub in entities_dir.glob("**/entities_*.csv"):
                if "adapted" in sub.name.lower() and label == "spacy_adapted":
                    entity_files[label] = sub
                    break
                if ("baseline" in sub.name.lower() or "nl_core" in sub.name.lower()) \
                        and label == "spacy_off_shelf":
                    entity_files[label] = sub
                    break

    if not entity_files:
        print(f"[error] No entity files found in {entities_dir}", file=sys.stderr)
        print("        Searched for: entities_nl_core_news_lg.csv, "
              "entities_adapted_model.csv, etc.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound entity files:")
    for label, path in entity_files.items():
        print(f"  {label}: {path}")

    # Run sensitivity check
    long_rows = []
    txt_lines = []
    txt_lines.append("=" * 70)
    txt_lines.append("THRESHOLD SENSITIVITY CHECK")
    txt_lines.append(f"Test inventories: {len(test_invs)}")
    txt_lines.append(f"Scorer: rapidfuzz.fuzz.token_sort_ratio")
    txt_lines.append(f"Thresholds: {THRESHOLDS}")
    txt_lines.append("=" * 70)
    txt_lines.append("")

    for label, path in entity_files.items():
        print(f"\n--- Loading entities for {label} ---")
        t0 = time.time()
        ner_dict = load_entities(path, test_invs)
        print(f"  Inventories with entities: {len(ner_dict)}")
        print(f"  Total mentions: {sum(len(v) for v in ner_dict.values()):,}")
        print(f"  Loaded in {time.time() - t0:.1f}s")

        txt_lines.append(f"### {label}")
        txt_lines.append(f"  Source: {path}")
        txt_lines.append(f"  Total PER mentions: "
                          f"{sum(len(v) for v in ner_dict.values()):,}")
        txt_lines.append("")
        txt_lines.append(f"  {'threshold':>10s} {'macro P':>9s} {'macro R':>9s} "
                          f"{'macro F1':>10s}  {'micro P':>9s} {'micro R':>9s} "
                          f"{'micro F1':>10s}")
        txt_lines.append("  " + "-" * 78)

        for thr in THRESHOLDS:
            print(f"  Evaluating at threshold {thr}...")
            t0 = time.time()
            metrics = evaluate_at_threshold(ner_dict, vh_dict, thr)
            elapsed = time.time() - t0
            print(f"    macro F1: {metrics['macro_f1']:.4f}, "
                  f"micro F1: {metrics['micro_f1']:.4f} "
                  f"({elapsed:.1f}s)")

            long_rows.append({
                "model": label,
                "threshold": thr,
                **{k: round(v, 4) for k, v in metrics.items()
                   if k != "n_inventories"},
                "n_inventories": metrics["n_inventories"],
            })

            txt_lines.append(
                f"  {thr:>10.2f} {metrics['macro_p']:>9.4f} "
                f"{metrics['macro_r']:>9.4f} {metrics['macro_f1']:>10.4f}  "
                f"{metrics['micro_p']:>9.4f} {metrics['micro_r']:>9.4f} "
                f"{metrics['micro_f1']:>10.4f}"
            )
        txt_lines.append("")

    # Cross-model comparison at each threshold
    txt_lines.append("=" * 70)
    txt_lines.append("RANKING ROBUSTNESS")
    txt_lines.append("=" * 70)
    txt_lines.append("")
    if "spacy_adapted" in entity_files and "spacy_off_shelf" in entity_files:
        for thr in THRESHOLDS:
            adapted = next(r for r in long_rows
                            if r["model"] == "spacy_adapted" and r["threshold"] == thr)
            offshelf = next(r for r in long_rows
                             if r["model"] == "spacy_off_shelf"
                             and r["threshold"] == thr)
            gap = adapted["macro_f1"] - offshelf["macro_f1"]
            txt_lines.append(
                f"  threshold {thr:.2f}: "
                f"adapted F1 = {adapted['macro_f1']:.4f}, "
                f"off-shelf F1 = {offshelf['macro_f1']:.4f}, "
                f"gap = +{gap:.4f}"
            )
        txt_lines.append("")
        txt_lines.append("If the gap stays positive across thresholds, ranking is robust.")

    # Write outputs
    csv_path = output_dir / "threshold_sensitivity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "threshold", "macro_p", "macro_r", "macro_f1",
                          "micro_p", "micro_r", "micro_f1", "n_inventories"],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    txt_path = output_dir / "threshold_sensitivity.txt"
    txt_content = "\n".join(txt_lines)
    txt_path.write_text(txt_content, encoding="utf-8")

    print("\n" + txt_content)
    print(f"\nOutputs:\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
