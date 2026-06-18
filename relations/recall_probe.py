#!/usr/bin/env python3
"""
Blind relation-recall probe + miss attribution.  No argparse.

HOW TO RUN
  1. Set the paths below and pick MODE.
  2. MODE = "sample"  ->  writes recall_probe_blank.csv
       Annotate that file BY HAND, without opening relations_all.csv.
       Fill n_true_relations_on_page and true_relations_freeform per row.
  3. Save the annotated file as recall_probe_filled.csv, add a
     miss_reason column, set MODE = "score", rerun.
       miss_reason values, per missed row:
         ner_both_missed / ner_one_missed  -> NER compounding
         matcher        -> both detected, between-text logic failed
         not_a_relation -> indicator present but no real relation
         htr_garbled    -> name unrecoverable from HTR

WHY BLIND: annotate gold WITHOUT seeing model output, or the recall
estimate inherits the model's blind spots and the probe just launders
the circularity it exists to break. The discipline is the instrument;
the code cannot enforce it.
"""

import csv, re, random, math, os
from pathlib import Path
from collections import Counter, defaultdict

# ─────────────── EDIT THESE ───────────────
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HTR_CACHE as _HTR_CACHE
MODE      = "sample"                                                  # "sample" or "score"
HTR_DIR   = str(_HTR_CACHE)
RELATIONS = str(DATA_DIR / "relations_all.csv")
BLANK_OUT = "recall_probe_blank.csv"
FILLED_IN = "recall_probe_filled.csv"
N_PAGES   = 25
SEED      = 42
# ──────────────────────────────────────────

INDICATORS = [
    "sijn huisvrouw","sijn huijsvrouw","zijn huisvrouw","zijn huijsvrouw",
    "sijn huysvrouw","zijn huysvrouw","sijn huijsvrouwe","zijn huijsvrouwe",
    "en zijn huisvrouw","en zijn huijsvrouw","en sijn huisvrouw","en sijn huijsvrouw",
    "en zijn huysvrouw","en haar man","en haer man","huisvrouw van","huijsvrouw van",
    "huysvrouw van","huijsvrouwe van","vrouw van","man ende voocht van",
    "man en voocht van","gehuwd met","getrouwd met","getrouwt met","echtgenote van",
    "weduwe van","weduwe wijlen","weduwe van wijlen","wede wijlen","wede van",
    "de weduwe van","de weduwe van wijlen","weduwnaar van","weduwnaar van wijlen",
    "sijn soon","zijn zoon","sijn zoon","zijn soon","haer soon","haar zoon",
    "sijn dochter","zijn dochter","haer dochter","haar dochter","heur dochter",
    "sijn kint","zijn kind","haer kint","haar kind","soon van","zoon van",
    "dochter van","dogter van","sijn broeder","zijn broeder","zijn broer",
    "sijn broer","haer broeder","haar broeder","sijn suster","zijn zuster",
    "haer suster","haar zuster","broeder van","broer van","suster van",
    "zuster van","sijn overleden broeder","haer overleden suster",
    "sijn overleden suster","wede. van","wede.","wed. van","wed.","wede: van",
    "wede:","wedue van","weduwe","huijsvrou van","als in huwelijk hebbende",
]
INDICATORS.sort(key=len, reverse=True)
COMBINED = re.compile("(" + "|".join(re.escape(i) for i in INDICATORS) + ")",
                       re.IGNORECASE)

def page_num(stem):
    d = re.sub(r"[^0-9]", "", stem)
    return d or None

def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96; p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def do_sample():
    htr = Path(HTR_DIR)
    if not htr.exists():
        raise SystemExit(f"HTR dir not found: {htr}")
    invs = sorted(d for d in htr.iterdir()
                  if d.is_dir() and any(True for _ in d.glob("*.txt")))
    bearing, scanned = [], 0
    for d in invs:
        inv = d.name
        for fp in d.glob("*.txt"):
            try:
                if fp.stat().st_size == 0:
                    continue
                t = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            scanned += 1
            if scanned % 50000 == 0:
                print(f"  ...{scanned:,} pages scanned, {len(bearing):,} with indicators")
            if COMBINED.search(t):
                pg = page_num(fp.stem)
                if pg:
                    bearing.append((inv, pg, str(fp)))
    print(f"scanned {scanned:,} non-empty pages; "
          f"{len(bearing):,} contain >=1 indicator (corpus-wide denominator)")
    if not bearing:
        raise SystemExit("no indicator-bearing pages found - check HTR_DIR")
    random.seed(SEED)
    sample = random.sample(bearing, min(N_PAGES, len(bearing)))
    with open(BLANK_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "inventory", "page", "indicator_hits",
                    "raw_context",
                    "n_true_relations_on_page",      # YOU FILL, BLIND
                    "true_relations_freeform"])      # YOU FILL, BLIND
        for i, (inv, pg, fp) in enumerate(sample, 1):
            t = Path(fp).read_text(encoding="utf-8")
            hits = sorted(set(m.group(0).lower() for m in COMBINED.finditer(t)))
            ctx = re.sub(r"\s+", " ", t).strip()
            w.writerow([i, inv, pg, " | ".join(hits), ctx, "", ""])
    print(f"wrote {BLANK_OUT}  ({len(sample)} pages)")
    print("Annotate it blind. Do NOT open relations_all.csv yet.")
    print(f"Then save it as {FILLED_IN}, add a miss_reason column,")
    print("set MODE = 'score', and rerun this file.")

def do_score():
    if not os.path.exists(FILLED_IN):
        raise SystemExit(f"{FILLED_IN} not found - annotate the blank first")
    model = defaultdict(int)
    with open(RELATIONS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            model[(r["inventory"].strip(), r["page"].strip())] += 1
    rows = list(csv.DictReader(open(FILLED_IN, encoding="utf-8")))
    gold = caught = 0
    miss = Counter()
    bad = []
    for r in rows:
        try:
            g = int(r["n_true_relations_on_page"] or 0)
        except ValueError:
            bad.append(r["page_id"]); continue
        gold += g
        m = model.get((r["inventory"].strip(), r["page"].strip()), 0)
        caught += min(g, m)
        if g > m:
            reason = (r.get("miss_reason") or "").strip().lower()
            miss[reason or "UNLABELLED"] += (g - m)
    if bad:
        print(f"WARNING non-numeric counts on page_id {bad} - fix before trusting")
    if gold == 0:
        print("no gold annotated"); return
    rec = caught / gold
    lo, hi = wilson(caught, gold)
    print(f"\nblind relation-recall probe")
    print(f"  gold relations : {gold}")
    print(f"  caught         : {caught}")
    print(f"  recall         : {rec*100:.1f}%  (95% Wilson CI "
          f"{lo*100:.1f}-{hi*100:.1f})")
    tot = sum(miss.values())
    print(f"\nmiss attribution (n={tot}):")
    for k, v in miss.most_common():
        print(f"  {k:16s}: {v}  ({100*v/tot:.0f}%)" if tot else f"  {k}: {v}")
    ner = miss.get("ner_both_missed", 0) + miss.get("ner_one_missed", 0)
    if tot:
        print(f"\n  NER-compounding share: {100*ner/tot:.0f}%")
        print("  >50% supports the 4.3 mechanism as written;")
        print("  <50% means matcher/denominator is the real bottleneck -")
        print("  that is a finding, rewrite 4.3 to say so.")

if __name__ == "__main__":
    if MODE == "sample":
        do_sample()
    elif MODE == "score":
        do_score()
    else:
        raise SystemExit(f"MODE must be 'sample' or 'score', got {MODE!r}")
