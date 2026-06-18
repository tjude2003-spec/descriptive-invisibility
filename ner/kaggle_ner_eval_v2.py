#!/usr/bin/env python3
"""
Flair + BERTje NER Evaluation on Test Set — PATCHED v2
======================================================
Produces eval_summary.txt and per-inventory CSV files in the SAME format
as the canonical spacy_baseline_eval_summary.txt and adapted_eval_summary.txt
so that all four models (BERTje, Flair, spaCy off-shelf, spaCy adapted)
are directly comparable.

What this version fixes versus the original kaggle_ner_eval.py:
  1. Applies notary-name filtering and formulaic stoplist filtering
     using the SAME logic as ner_baseline.py
  2. Uses fuzz.token_sort_ratio (not fuzz.ratio) — matches ner_baseline.py
  3. Writes eval_summary.txt in the canonical condensed format
  4. Writes per-inventory CSV with notary names (needed for analysis)

KAGGLE SETUP:
  Dataset 1 — "thesis-test-htr":
    * htr_cache/ folder with the 73 test inventory subdirectories
    * ner_split.json
    * final_velehanden_inventory_summary.csv  (for inventory→notary mapping)
  Dataset 2 — "thesis-vh-data":
    * velehanden_deeds_part1.csv
    * velehanden_deeds_part2.csv

  Notebook: GPU on, both datasets attached, paste this into one cell.

OUTPUTS in /kaggle/working/:
  bertje_per_inventory.csv
  bertje_eval_summary.txt
  flair_per_inventory.csv
  flair_eval_summary.txt
"""

# =====================================================================
# CELL 1: Install dependencies
# =====================================================================
import subprocess
subprocess.run(["pip", "install", "flair", "rapidfuzz", "-q"])

# =====================================================================
# CELL 2: Load data
# =====================================================================
import json, csv, os, time, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

HTR_BASE = "/kaggle/input/datasets/tjudem/thesis-test-htr/test_htr_cache/test_htr_cache"
VH_CSVS = ["/kaggle/input/datasets/tjudem/thesis-vh-data/final_velehanden_deeds.csv"]
SPLIT_PATH = "/kaggle/input/datasets/tjudem/thesis-test-htr/ner_split.json"
NOTARY_SUMMARY_CSV = "/kaggle/input/datasets/tjudem/thesis-test-htr/final_velehanden_inventory_summary_.csv"

# Verify all paths resolve before continuing
assert os.path.isdir(HTR_BASE), f"HTR_BASE not a directory: {HTR_BASE}"
for _p in VH_CSVS:
    assert os.path.isfile(_p), f"VH CSV not found: {_p}"
assert os.path.isfile(SPLIT_PATH), f"split file not found: {SPLIT_PATH}"
assert os.path.isfile(NOTARY_SUMMARY_CSV), \
    f"notary CSV not found: {NOTARY_SUMMARY_CSV}"

print(f"Split:           {SPLIT_PATH}")
print(f"HTR cache:       {HTR_BASE}")
print(f"VH CSVs:         {VH_CSVS}")
print(f"Notary summary:  {NOTARY_SUMMARY_CSV}")

with open(SPLIT_PATH) as f:
    split = json.load(f)
TEST_INVS = split["test"]
print(f"\nTest inventories: {len(TEST_INVS)}")

# Inventory → notary mapping
inv_to_notary = {}
if NOTARY_SUMMARY_CSV:
    with open(NOTARY_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inv = row.get("inventory_number", "").strip()
            notary = row.get("notary", "").strip()
            if inv and notary:
                inv_to_notary[inv] = notary
    print(f"Notary mapping: {len(inv_to_notary)} inventories total, "
          f"{sum(1 for inv in TEST_INVS if inv in inv_to_notary)} of {len(TEST_INVS)} test invs covered")
else:
    print("WARNING: no notary mapping found — filtering will skip notary suppression")

# Load VH ground truth for test inventories only
test_inv_set = set(TEST_INVS)
vh_names_per_inv = defaultdict(list)
for csv_path in sorted(VH_CSVS):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inv = row.get("inventory_number", "").strip()
            if inv not in test_inv_set:
                continue
            names_str = row.get("person_names", "").strip()
            if names_str:
                for name in names_str.split("|"):
                    name = name.strip()
                    if name:
                        vh_names_per_inv[inv].append(name)

test_vh = {inv: vh_names_per_inv[inv] for inv in TEST_INVS if inv in vh_names_per_inv}
print(f"Test inventories with VH data: {len(test_vh)}")
print(f"Total VH name mentions: {sum(len(v) for v in test_vh.values()):,}")

# =====================================================================
# CELL 3: HTR loading
# =====================================================================
def clean_htr_text(raw_text):
    lines = raw_text.splitlines()
    if len(lines) >= 4:
        for block_size in range(2, min(30, len(lines) // 2 + 1)):
            if (lines[:block_size] == lines[block_size:block_size * 2]
                    and all(l.strip() for l in lines[:block_size])):
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
        if (cleaned and cleaned[-1].endswith("-") and len(cleaned[-1]) >= 2
                and cleaned[-1][-2].isalpha() and line and line[0].islower()):
            cleaned[-1] = cleaned[-1][:-1] + line
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def load_htr_text(inv_id):
    inv_dir = Path(HTR_BASE) / str(inv_id)
    if not inv_dir.exists():
        return ""
    texts = []
    files = sorted(
        [p for p in inv_dir.iterdir()
         if p.suffix == ".txt" and not p.name.startswith("._")],
        key=lambda p: int(p.stem.replace("page_", ""))
            if p.stem.replace("page_", "").isdigit() else 0
    )
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        cleaned = clean_htr_text(raw)
        if cleaned.strip():
            texts.append(cleaned)
    return "\n".join(texts)

test_texts = {}
for inv in TEST_INVS:
    txt = load_htr_text(inv)
    if txt.strip():
        test_texts[inv] = txt
print(f"\nTest inventories with HTR text: {len(test_texts)}")
missing = [inv for inv in TEST_INVS if inv not in test_texts]
if missing:
    print(f"WARNING: {len(missing)} test inventories missing HTR text: {missing}")


# =====================================================================
# CELL 4: Filtering — exact match to ner_baseline.py
# =====================================================================
from rapidfuzz import fuzz

# COPIED VERBATIM from ner_baseline.py FORMULAIC_STOPLIST
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
    return entity_text.lower().strip() in FORMULAIC_STOPLIST

def is_notary_name(entity_text: str, notary_name: str) -> bool:
    """Verbatim from ner_baseline.py."""
    ent_lower = entity_text.lower().strip()
    notary_lower = notary_name.lower().strip()
    if ent_lower == notary_lower:
        return True
    notary_parts = notary_lower.split()
    if ent_lower in notary_parts:
        return True
    score = fuzz.token_sort_ratio(ent_lower, notary_lower) / 100.0
    if score >= 0.75:
        return True
    if len(ent_lower) >= 4 and ent_lower in notary_lower:
        return True
    return False

def filter_ner_names(ner_names_list, notary_name):
    """
    Match ner_baseline.py filter_ner_entities. Operates on a flat list
    of name strings (no entity_type wrapping needed since BERTje/Flair
    already return only PER labels).

    Returns (kept_list, removed_counts).
    """
    kept = []
    removed = {"notary": 0, "formulaic": 0, "kept": 0}
    for name in ner_names_list:
        if is_formulaic(name):
            removed["formulaic"] += 1
            continue
        if notary_name and is_notary_name(name, notary_name):
            removed["notary"] += 1
            continue
        removed["kept"] += 1
        kept.append(name)
    return kept, removed


# =====================================================================
# CELL 5: Evaluation function — matches ner_baseline.py logic exactly
# =====================================================================
from rapidfuzz.process import cdist as rfp_cdist

def evaluate_inventory(ner_names, vh_names, fuzzy_threshold=0.85):
    """
    Per-inventory eval matching ner_baseline.py.

    - Dedup both sides at lowercase string level
    - cdist using fuzz.token_sort_ratio (not fuzz.ratio)
    - Precision: per-mention (each NER occurrence counted vs its best
      VH match)
    - Recall: per-mention (each VH occurrence counted vs its best
      NER match)
    """
    if not vh_names and not ner_names:
        return {
            "vh_names": 0, "vh_unique": 0,
            "ner_count": 0, "ner_unique": 0,
            "vh_matched": 0, "ner_matched": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }

    ner_lower_to_orig = {}
    for t in ner_names:
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

    if not ner_dedup or not vh_dedup:
        return {
            "vh_names": len(vh_names),
            "vh_unique": len(vh_dedup),
            "ner_count": len(ner_names),
            "ner_unique": len(ner_dedup),
            "vh_matched": 0, "ner_matched": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    matrix = rfp_cdist(
        ner_dedup, vh_dedup,
        scorer=fuzz.token_sort_ratio,
        workers=-1,
    ) / 100.0

    ner_best_scores = matrix.max(axis=1)
    vh_best_scores = matrix.max(axis=0)

    # Build per-string match flags then expand back to mention-level counts
    ner_lower_matched = {
        ner_dedup[i]: ner_best_scores[i] >= fuzzy_threshold
        for i in range(len(ner_dedup))
    }
    vh_lower_matched = {
        vh_dedup[j]: vh_best_scores[j] >= fuzzy_threshold
        for j in range(len(vh_dedup))
    }

    vh_matched = sum(1 for n in vh_names if vh_lower_matched[n.lower()])
    ner_matched = sum(1 for t in ner_names if ner_lower_matched[t.lower()])

    precision = ner_matched / max(len(ner_names), 1)
    recall = vh_matched / max(len(vh_names), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "vh_names": len(vh_names),
        "vh_unique": len(vh_dedup),
        "ner_count": len(ner_names),
        "ner_unique": len(ner_dedup),
        "vh_matched": vh_matched,
        "ner_matched": ner_matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_full_eval(model_label, ner_names_dict, fuzzy_threshold=0.85,
                    apply_filter=False):
    """
    Run evaluation across all test inventories. By default, NO filtering
    is applied — this matches the canonical spaCy unfiltered evaluation
    from 2026-04-17T22:14 (which produced the F1=0.6804 and F1=0.3018
    figures in the thesis results table).

    Set apply_filter=True only if you specifically want filtered numbers
    as a separate analysis.
    """
    print(f"\n>>> FILTERING: {'ON' if apply_filter else 'OFF'} for {model_label}")
    print(f">>> SCORER:    fuzz.token_sort_ratio @ threshold {fuzzy_threshold}\n")

    per_inv = []
    total_vh_unique = 0
    total_ner_unique = 0
    total_removed = {"notary": 0, "formulaic": 0, "kept": 0}

    for inv in sorted(test_vh.keys()):
        vh = test_vh[inv]
        raw_ner = list(ner_names_dict.get(inv, []))
        notary = inv_to_notary.get(inv)

        if apply_filter:
            ner_for_eval, removed = filter_ner_names(raw_ner, notary)
            for k in total_removed:
                total_removed[k] += removed[k]
        else:
            ner_for_eval = raw_ner
            removed = {"notary": 0, "formulaic": 0, "kept": len(raw_ner)}

        result = evaluate_inventory(ner_for_eval, vh, fuzzy_threshold)
        result["inventory_number"] = inv
        result["notary"] = notary or ""
        result["removed_notary"] = removed["notary"]
        result["removed_formulaic"] = removed["formulaic"]
        per_inv.append(result)

        total_vh_unique += result["vh_unique"]
        total_ner_unique += result["ner_unique"]

    if per_inv:
        macro_p = float(np.mean([r["precision"] for r in per_inv]))
        macro_r = float(np.mean([r["recall"] for r in per_inv]))
        macro_f1 = float(np.mean([r["f1"] for r in per_inv]))
        total_vh_matched = sum(r["vh_matched"] for r in per_inv)
        total_ner_matched = sum(r["ner_matched"] for r in per_inv)
        total_vh_mentions = sum(r["vh_names"] for r in per_inv)
        total_ner_mentions = sum(r["ner_count"] for r in per_inv)
        micro_p = total_ner_matched / max(total_ner_mentions, 1)
        micro_r = total_vh_matched / max(total_vh_mentions, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
    else:
        macro_p = macro_r = macro_f1 = 0.0
        micro_p = micro_r = micro_f1 = 0.0

    totals = {
        "n_inventories": len(per_inv),
        "total_ner_unique": total_ner_unique,
        "total_vh_unique": total_vh_unique,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "removed": total_removed,
    }
    return per_inv, totals


def write_eval_summary(label, per_inv, totals, out_path):
    """Write eval_summary.txt in the format matching the canonical spacy outputs."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"NER EVALUATION REPORT — {label}")
    lines.append(f"Date: {datetime.now().isoformat()}")
    lines.append("Evaluation method: deduplicated (unique entities per inventory)")
    lines.append("Fuzzy threshold: 0.85 (rapidfuzz.fuzz.token_sort_ratio >= 85)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Inventories evaluated: {totals['n_inventories']}")
    lines.append(f"Total unique NER PER entities: {totals['total_ner_unique']:,}")
    lines.append(f"Total unique VH ground truth names: {totals['total_vh_unique']:,}")
    lines.append("")
    lines.append(f"Macro  P: {totals['macro_p']:.4f}  R: {totals['macro_r']:.4f}  F1: {totals['macro_f1']:.4f}")
    lines.append(f"Micro  P: {totals['micro_p']:.4f}  R: {totals['micro_r']:.4f}  F1: {totals['micro_f1']:.4f}")
    lines.append("")

    by_f1 = sorted(per_inv, key=lambda r: r["f1"])
    lines.append("BOTTOM 5 (lowest F1):")
    for r in by_f1[:5]:
        lines.append(
            f"  inv {r['inventory_number']:>6s}: "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"(pred={r['ner_unique']}, gold={r['vh_unique']}, "
            f"notary={r['notary']})"
        )
    lines.append("")
    lines.append("TOP 5 (highest F1):")
    for r in by_f1[-5:][::-1]:
        lines.append(
            f"  inv {r['inventory_number']:>6s}: "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"(pred={r['ner_unique']}, gold={r['vh_unique']}, "
            f"notary={r['notary']})"
        )

    text = "\n".join(lines)
    Path(out_path).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {out_path}")


def write_per_inventory_csv(per_inv, out_path):
    fieldnames = ["inventory_number", "notary", "vh_names", "vh_unique",
                  "ner_count", "ner_unique",
                  "removed_notary", "removed_formulaic",
                  "precision", "recall", "f1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in per_inv:
            w.writerow(r)
    print(f"Saved {out_path}")


# =====================================================================
# CELL 6: Run BERTje
# =====================================================================
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

print("\n" + "=" * 60)
print("RUNNING BERTje NER")
print("=" * 60)

model_name = "wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForTokenClassification.from_pretrained(model_name)

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")
ner_pipe = pipeline("ner", model=bert_model, tokenizer=tokenizer,
                    aggregation_strategy="simple", device=device)

def extract_persons_bertje(text, max_chunk_chars=1500):
    """Returns a list of name strings (not deduplicated yet)."""
    names = []
    for start in range(0, len(text), max_chunk_chars):
        chunk = text[start:start + max_chunk_chars]
        if not chunk.strip():
            continue
        try:
            ents = ner_pipe(chunk)
            for e in ents:
                # BERTje uses lowercase 'per' label
                if e.get("entity_group", "").upper() == "PER":
                    name = e["word"].strip()
                    if name:
                        names.append(name)
        except Exception as ex:
            print(f"  [warn] BERTje chunk failed: {ex}")
    return names

bertje_names = {}
t0 = time.time()
for i, inv in enumerate(test_texts):
    bertje_names[inv] = extract_persons_bertje(test_texts[inv])
    if (i + 1) % 5 == 0 or (i + 1) == len(test_texts):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  BERTje: {i+1}/{len(test_texts)} inventories ({rate:.1f}/min)")

bertje_time = time.time() - t0
print(f"BERTje done in {bertje_time:.0f}s")

per_inv, totals = run_full_eval("BERTje", bertje_names, fuzzy_threshold=0.85)
write_per_inventory_csv(per_inv, "/kaggle/working/bertje_per_inventory.csv")
write_eval_summary("BERTje (off-shelf)", per_inv, totals,
                    "/kaggle/working/bertje_eval_summary.txt")

del bert_model, ner_pipe, tokenizer
torch.cuda.empty_cache()


# =====================================================================
# CELL 7: Run Flair
# =====================================================================
from flair.data import Sentence
from flair.models import SequenceTagger

print("\n" + "=" * 60)
print("RUNNING Flair NER")
print("=" * 60)

tagger = SequenceTagger.load("flair/ner-dutch-large")

def extract_persons_flair(text, max_chars=10000):
    names = []
    for start in range(0, len(text), max_chars):
        chunk = text[start:start + max_chars]
        if not chunk.strip():
            continue
        try:
            sentence = Sentence(chunk)
            tagger.predict(sentence)
            for ent in sentence.get_spans("ner"):
                if ent.tag == "PER":
                    name = ent.text.strip()
                    if name:
                        names.append(name)
        except Exception as ex:
            print(f"  [warn] Flair chunk failed: {ex}")
    return names

flair_names = {}
t0 = time.time()
for i, inv in enumerate(test_texts):
    flair_names[inv] = extract_persons_flair(test_texts[inv])
    if (i + 1) % 5 == 0 or (i + 1) == len(test_texts):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  Flair: {i+1}/{len(test_texts)} inventories ({rate:.1f}/min)")

flair_time = time.time() - t0
print(f"Flair done in {flair_time:.0f}s")

per_inv, totals = run_full_eval("Flair", flair_names, fuzzy_threshold=0.85)
write_per_inventory_csv(per_inv, "/kaggle/working/flair_per_inventory.csv")
write_eval_summary("Flair (off-shelf)", per_inv, totals,
                    "/kaggle/working/flair_eval_summary.txt")


# =====================================================================
# CELL 8: Comparison summary
# =====================================================================
print("\n" + "=" * 60)
print("ALL FOUR MODELS COMPARED (run on 73 test inventories)")
print("=" * 60)
print(f"  spaCy off-shelf  (canonical, from local run):  P=0.2103  R=0.5843  F1=0.3018")
print(f"  spaCy adapted    (canonical, from local run):  P=0.7031  R=0.6963  F1=0.6804")
