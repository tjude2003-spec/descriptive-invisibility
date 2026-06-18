#!/usr/bin/env python3
"""
Recall estimation — mirrors exact indicators from build_relations_fast.py
"""
import csv, re, time, random
from pathlib import Path
from collections import Counter
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HTR_CACHE
RELATIONS_FILE = "relations_all.csv"

INDICATORS = [
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
    "wede. van", "wede.", "wed. van", "wed.",
    "wede: van", "wede:", "wedue van", "weduwe",
    "huijsvrou van", "als in huwelijk hebbende",
]

# Sort longest first so longer matches take priority
INDICATORS.sort(key=len, reverse=True)
COMPILED = [(ind, re.compile(r'\b' + re.escape(ind) + r'\b', re.IGNORECASE)) for ind in INDICATORS]

print(f"Indicators: {len(INDICATORS)} (matching extraction script exactly)")

print("\nLoading extracted relations...")
extracted_pages = set()
extracted_count = 0
with open(RELATIONS_FILE, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        extracted_pages.add((row['inventory'], row['page']))
        extracted_count += 1
print(f"  {extracted_count} relations across {len(extracted_pages)} pages")

print("\nScanning HTR pages...")
t0 = time.time()

# Get inventories that build_relations actually processed
ner_invs = set(k[0] for k in extracted_pages)
inv_dirs = sorted([d for d in HTR_CACHE.iterdir() if d.is_dir() and d.name in ner_invs])
random.seed(42)
sample_dirs = random.sample(inv_dirs, min(200, len(inv_dirs)))
print(f"  Sampling {len(sample_dirs)} of {len(ner_invs)} inventories with extracted relations")

found_and_extracted = 0
found_not_extracted = 0
pages_scanned = 0
missed_examples = []
missed_by_indicator = Counter()

for inv_dir in sample_dirs:
    inv = inv_dir.name
    for f in sorted(inv_dir.glob("*.txt")):
        if f.stat().st_size == 0:
            continue
        page = re.sub(r'[^0-9]', '', f.stem)
        if not page:
            continue

        text = f.read_text(encoding='utf-8')
        pages_scanned += 1

        page_has_keyword = False
        page_has_extraction = (inv, page) in extracted_pages

        for ind_text, pattern in COMPILED:
            if pattern.search(text):
                page_has_keyword = True
                if page_has_extraction:
                    found_and_extracted += 1
                else:
                    found_not_extracted += 1
                    missed_by_indicator[ind_text] += 1
                    if len(missed_examples) < 20:
                        m = pattern.search(text)
                        start = max(0, m.start() - 50)
                        end = min(len(text), m.end() + 50)
                        ctx = text[start:end].replace('\n', ' ')
                        missed_examples.append({
                            'inv': inv, 'page': page,
                            'indicator': ind_text, 'context': ctx
                        })
                break  # one keyword per page is enough

    if pages_scanned % 20000 == 0:
        elapsed = time.time() - t0
        print(f"  {pages_scanned} pages ({elapsed:.0f}s)")

elapsed = time.time() - t0
total = found_and_extracted + found_not_extracted

print(f"\nDone in {elapsed:.0f}s")
print(f"Pages scanned: {pages_scanned:,}")
print(f"Pages with keyword + extraction: {found_and_extracted}")
print(f"Pages with keyword, NO extraction: {found_not_extracted}")

if total > 0:
    recall = found_and_extracted / total
    print(f"\nRecall estimate: {found_and_extracted}/{total} = {recall*100:.1f}%")

print(f"\nMissed by indicator:")
for ind, count in missed_by_indicator.most_common(15):
    print(f"  '{ind}': {count}")

print(f"\nMissed examples:")
for ex in missed_examples[:15]:
    print(f"  inv {ex['inv']} p{ex['page']}: '{ex['indicator']}'")
    print(f"    ...{ex['context'][:120]}...")
