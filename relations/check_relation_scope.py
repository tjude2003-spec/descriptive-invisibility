#!/usr/bin/env python3
"""
Audits the TRUE scope of build_relations_fast.py.

Answers two questions the printed log does not:
  1. Does the NER file the relation script consumes span the full corpus?
  2. How many ">=2 PER" pages would silently vanish in get_page_text()
     because the HTR filename pattern does not match the cache?

Run it against the SAME ner file build_relations_fast.py points at
(the one with start_char/end_char), not a slim export.

  python check_relation_scope.py /path/to/ner_extractions.csv /path/to/htr_cache
"""
import csv, sys, os
from pathlib import Path
from collections import defaultdict, Counter

if len(sys.argv) < 3:
    sys.exit("usage: check_relation_scope.py NER_CSV HTR_CACHE_DIR")
ner_file = sys.argv[1]
htr_cache = Path(sys.argv[2])

with open(ner_file, encoding="utf-8") as f:
    cols = csv.DictReader(f).fieldnames
print("NER file :", ner_file)
print("columns  :", cols)
has_offsets = {"start_char", "end_char", "entity_label"} <= set(cols)
print("offset/label cols present:", has_offsets,
      "" if has_offsets else "  <-- NOT the file the relation script can run on")
csv.field_size_limit(10**7)

# group PER entities by (inv,page); fall back to all rows if no label column
page_entities = defaultdict(int)
inv_counter = Counter()
rows = 0
with open(ner_file, encoding="utf-8") as f:
    r = csv.DictReader(f)
    for x in r:
        rows += 1
        if "entity_label" in x and x["entity_label"] and x["entity_label"] != "PER":
            continue
        inv = x["inventory_number"].strip()
        pg = x["page_number"].strip()
        if not x["entity_text"].strip():
            continue
        page_entities[(inv, pg)] += 1
        inv_counter[inv] += 1

pages_2plus = [k for k, c in page_entities.items() if c >= 2]
print(f"\nrows scanned            : {rows:,}")
print(f"distinct inventories    : {len(inv_counter):,}")
print(f"distinct (inv,page)     : {len(page_entities):,}")
print(f"pages with >=2 entities : {len(pages_2plus):,}  (the relation-eligible set)")

# simulate get_page_text() resolution against the real cache
if not htr_cache.exists():
    print(f"\nHTR cache {htr_cache} not found - skipping attrition check")
    sys.exit(0)

resolved = 0
missing_dir = 0
missing_file = 0
miss_examples = []
for inv, pg in pages_2plus:
    d = htr_cache / inv
    if not d.exists():
        missing_dir += 1
        if len(miss_examples) < 5: miss_examples.append((inv, pg, "no inv dir"))
        continue
    if (d / f"{pg}.txt").exists() or (d / f"page_{pg}.txt").exists():
        resolved += 1
    else:
        missing_file += 1
        if len(miss_examples) < 5: miss_examples.append((inv, pg, "no page file"))

elig = len(pages_2plus)
lost = missing_dir + missing_file
print(f"\n--- silent attrition in get_page_text() ---")
print(f"relation-eligible pages : {elig:,}")
print(f"  HTR text resolves     : {resolved:,}  ({100*resolved/elig:.1f}%)")
print(f"  inv dir missing       : {missing_dir:,}")
print(f"  page file missing     : {missing_file:,}")
print(f"  TOTAL silently skipped : {lost:,}  ({100*lost/elig:.1f}%)")
if lost:
    print("  examples:", miss_examples)
    print("\n  >1-2% here means the corpus-scale RQ3 claim rests on a")
    print("  truncated denominator. Fix the filename glob before quoting scope.")
else:
    print("  clean: every relation-eligible page resolves to HTR text.")
