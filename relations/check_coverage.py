#!/usr/bin/env python3
"""
Check indicator coverage: what between-texts are we missing?
Finds frequent text between consecutive PER pairs that didn't match any indicator.
"""
import csv, re, time
from pathlib import Path
from collections import defaultdict, Counter

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HTR_CACHE

NER_FILE = str(DATA_DIR / "ner_extractions.csv")

RELATION_INDICATORS = {
    "sijn huisvrouw", "sijn huijsvrouw", "zijn huisvrouw", "zijn huijsvrouw",
    "sijn huysvrouw", "zijn huysvrouw", "sijn huijsvrouwe", "zijn huijsvrouwe",
    "en zijn huisvrouw", "en zijn huijsvrouw", "en sijn huisvrouw", "en sijn huijsvrouw",
    "en zijn huysvrouw", "en haar man", "en haer man",
    "huisvrouw van", "huijsvrouw van", "huysvrouw van", "huijsvrouwe van",
    "vrouw van", "man ende voocht van", "man en voocht van",
    "gehuwd met", "getrouwd met", "getrouwt met", "echtgenote van",
    "weduwe van", "weduwe wijlen", "weduwe van wijlen", "wede wijlen",
    "wede van", "de weduwe van", "de weduwe van wijlen",
    "weduwnaar van", "weduwnaar van wijlen",
    "sijn soon", "zijn zoon", "sijn zoon", "zijn soon",
    "haer soon", "haar zoon", "sijn dochter", "zijn dochter",
    "haer dochter", "haar dochter", "heur dochter",
    "sijn kint", "zijn kind", "haer kint", "haar kind",
    "soon van", "zoon van", "dochter van", "dogter van",
    "sijn broeder", "zijn broeder", "zijn broer", "sijn broer",
    "haer broeder", "haar broeder", "sijn suster", "zijn zuster",
    "haer suster", "haar zuster", "broeder van", "broer van",
    "suster van", "zuster van",
    "sijn overleden broeder", "haer overleden suster", "sijn overleden suster",
}

print("Loading NER entities...")
t0 = time.time()
page_entities = defaultdict(list)

with open(NER_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['entity_label'] != 'PER': continue
        text = row['entity_text'].strip()
        if not text: continue
        inv = row['inventory_number'].strip()
        page = row['page_number'].strip()
        start = int(row['start_char'])
        end = int(row['end_char'])
        page_entities[(inv, page)].append((start, end, text))

for key in page_entities:
    page_entities[key].sort()

print(f"  Loaded in {time.time()-t0:.0f}s")
print(f"  Pages with 2+ PER: {sum(1 for v in page_entities.values() if len(v) >= 2)}")

print("\nScanning between-texts...")
t1 = time.time()
unmatched = Counter()
checked = 0

for (inv, page), entities in page_entities.items():
    if len(entities) < 2: continue
    
    inv_dir = HTR_CACHE / inv
    text = None
    for pattern in [f"{page}.txt", f"page_{page}.txt"]:
        fpath = inv_dir / pattern
        if fpath.exists():
            text = fpath.read_text(encoding='utf-8')
            break
    if text is None: continue
    
    for i in range(len(entities) - 1):
        s1, e1, n1 = entities[i]
        s2, e2, n2 = entities[i + 1]
        if e1 > s2 or s2 - e1 > 150: continue
        
        between = text[e1:s2].strip().lower()
        between = re.sub(r'\s+', ' ', between)
        
        if not between or len(between.split()) > 6: continue
        checked += 1
        
        is_matched = False
        for ind in RELATION_INDICATORS:
            if between == ind or between.startswith(ind) or between.endswith(ind):
                is_matched = True
                break
        
        if not is_matched:
            unmatched[between] += 1

print(f"  Checked {checked:,} pairs in {time.time()-t1:.0f}s")
print(f"\n  Top 50 UNMATCHED between-texts (potential missing indicators):")
for text, count in unmatched.most_common(50):
    print(f"    {count:>5}x  '{text}'")
