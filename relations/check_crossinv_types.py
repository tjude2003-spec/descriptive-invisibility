#!/usr/bin/env python3
"""
Verify the cross-inventory relation-type breakdown (the "1,331 of 1,651
widow" figure) against relations_all.csv directly.

Mirrors rq3_cross_relations_v2.py's pair logic EXACTLY so the
denominator matches the patched run's 1,651:
  - same VALID_TYPES
  - same key: tuple(sorted([p1.lower(), p2.lower()]))
  - cross-inventory = pair appears in >1 distinct inventory

A pair can carry more than one relation type across its inventories
(the script stores types as a set). This reports the breakdown two
ways so you can state whichever the thesis sentence means:
  (A) by DOMINANT/ANY type   - count a pair under each type it shows
  (B) by SOLE type           - count a pair only if it has exactly one type

Set RELATIONS path, run:  python check_crossinv_types.py
"""
import csv, os
from pathlib import Path
from collections import defaultdict, Counter
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

RELATIONS = str(DATA_DIR / "relations_all.csv")
VALID_TYPES = {'widow', 'spouse', 'child', 'sibling', 'widower'}

pair = defaultdict(lambda: {'invs': set(), 'types': set()})

with open(RELATIONS, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rt = r['relation_type']
        if rt not in VALID_TYPES:
            continue
        p1 = r['person_1'].strip().lower()
        p2 = r['person_2'].strip().lower()
        k = tuple(sorted([p1, p2]))
        d = pair[k]
        d['invs'].add(r['inventory'].strip())
        d['types'].add(rt)

cross = {k: d for k, d in pair.items() if len(d['invs']) > 1}
n_cross = len(cross)
print(f"total pairs              : {len(pair):,}")
print(f"cross-inventory pairs    : {n_cross:,}   (expect 1,651)")

# (A) ANY-type: a pair counts under every type it carries
any_t = Counter()
for d in cross.values():
    for t in d['types']:
        any_t[t] += 1

# (B) SOLE-type: a pair counts once, only if it carries exactly one type
sole_t = Counter()
multi = 0
for d in cross.values():
    if len(d['types']) == 1:
        sole_t[next(iter(d['types']))] += 1
    else:
        multi += 1

print("\n(A) ANY-type breakdown (pair counted under each type it shows;")
print("    sums to >= n_cross because some pairs carry multiple types):")
for t, c in any_t.most_common():
    print(f"    {t:10s}: {c:5d}")

print(f"\n(B) SOLE-type breakdown (pair has exactly one type; "
      f"{multi} pairs carry >1 type and are excluded here):")
for t, c in sole_t.most_common():
    print(f"    {t:10s}: {c:5d}")
print(f"    (multi-type pairs        : {multi})")

# add after the sole-type block
widow_multi = sum(1 for d in cross.values() if len(d['types']) > 1 and 'widow' in d['types'])
non_widow_multi = multi - widow_multi
print(f"    (of which include widow : {widow_multi}, no widow : {non_widow_multi})")

print(f"\nINTERPRETATION")
print(f"  If the thesis sentence '1,331 of 1,651 widow' means 'widow is")
print(f"  among the types' -> compare to ANY-type widow = {any_t.get('widow',0)}.")
print(f"  If it means 'pairs that are purely widow' -> compare to")
print(f"  SOLE-type widow = {sole_t.get('widow',0)}.")
print(f"  Use whichever matches how the sentence is phrased, and make the")
print(f"  phrasing match the definition you report.")
