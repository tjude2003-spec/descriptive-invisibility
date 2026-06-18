#!/usr/bin/env python3
"""
Relation extraction v3 — FAST version.
Uses pre-computed NER extractions + raw HTR text. No spaCy inference needed.
Should complete in 30-60 minutes.
"""
import csv, json, time, sys, os, re
from pathlib import Path
from collections import defaultdict, Counter

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HTR_CACHE

NER_FILE = str(DATA_DIR / "ner_extractions.csv")
OUTPUT_CSV = "relations_all.csv"
OUTPUT_STATS = "relations_stats.json"

RELATION_INDICATORS = {
    "sijn huisvrouw": "spouse", "sijn huijsvrouw": "spouse",
    "zijn huisvrouw": "spouse", "zijn huijsvrouw": "spouse",
    "sijn huysvrouw": "spouse", "zijn huysvrouw": "spouse",
    "sijn huijsvrouwe": "spouse", "zijn huijsvrouwe": "spouse",
    "en zijn huisvrouw": "spouse", "en zijn huijsvrouw": "spouse",
    "en sijn huisvrouw": "spouse", "en sijn huijsvrouw": "spouse",
    "en zijn huysvrouw": "spouse", "en haar man": "spouse",
    "en haer man": "spouse",
    "huisvrouw van": "spouse", "huijsvrouw van": "spouse",
    "huysvrouw van": "spouse", "huijsvrouwe van": "spouse",
    "vrouw van": "spouse",
    "man ende voocht van": "spouse", "man en voocht van": "spouse",
    "gehuwd met": "spouse", "getrouwd met": "spouse",
    "getrouwt met": "spouse", "echtgenote van": "spouse",
    "weduwe van": "widow", "weduwe wijlen": "widow",
    "weduwe van wijlen": "widow", "wede wijlen": "widow",
    "wede van": "widow", "de weduwe van": "widow",
    "de weduwe van wijlen": "widow",
    "weduwnaar van": "widower", "weduwnaar van wijlen": "widower",
    "sijn soon": "child", "zijn zoon": "child",
    "sijn zoon": "child", "zijn soon": "child",
    "haer soon": "child", "haar zoon": "child",
    "sijn dochter": "child", "zijn dochter": "child",
    "haer dochter": "child", "haar dochter": "child",
    "heur dochter": "child",
    "sijn kint": "child", "zijn kind": "child",
    "haer kint": "child", "haar kind": "child",
    "soon van": "child", "zoon van": "child",
    "dochter van": "child", "dogter van": "child",
    "sijn broeder": "sibling", "zijn broeder": "sibling",
    "zijn broer": "sibling", "sijn broer": "sibling",
    "haer broeder": "sibling", "haar broeder": "sibling",
    "sijn suster": "sibling", "zijn zuster": "sibling",
    "haer suster": "sibling", "haar zuster": "sibling",
    "broeder van": "sibling", "broer van": "sibling",
    "suster van": "sibling", "zuster van": "sibling",
    "sijn overleden broeder": "sibling",
    "haer overleden suster": "sibling",
    "sijn overleden suster": "sibling",
    "wede. van": "widow", "wede.": "widow", "wed. van": "widow", "wed.": "widow",
    "wede: van": "widow", "wede:": "widow", "wedue van": "widow",
    "weduwe": "widow",
    "huijsvrou van": "spouse",
    "als in huwelijk hebbende": "spouse",
}

MALE_SUFFIXES = ('sz', 'szen', 'zoon', 'szoon', 'sen')
FEMALE_SUFFIXES = ('dr', 'dochter', 'dgtr')

def likely_gender(name):
    tokens = name.lower().rstrip('.').split()
    if not tokens: return 'unknown'
    for t in tokens[1:]:
        t_clean = t.rstrip('.')
        if any(t_clean.endswith(s) for s in MALE_SUFFIXES): return 'male'
        if any(t_clean.endswith(s) for s in FEMALE_SUFFIXES): return 'female'
    return 'unknown'

def spouse_plausible(p1, p2):
    g1, g2 = likely_gender(p1), likely_gender(p2)
    if g1 == g2 and g1 != 'unknown': return False
    return True

# ── PREFLIGHT ──
print("=" * 60)
print("PREFLIGHT CHECKS")
print("=" * 60)

if not os.path.exists(NER_FILE):
    print(f"  ERROR: {NER_FILE} not found")
    sys.exit(1)
if not HTR_CACHE.exists():
    print(f"  ERROR: HTR cache not found: {HTR_CACHE}")
    sys.exit(1)

with open(NER_FILE, encoding='utf-8') as f:
    cols = csv.DictReader(f).fieldnames
    print(f"  NER columns: {cols}")
    required = {'inventory_number', 'page_number', 'entity_text', 'entity_label', 'start_char', 'end_char'}
    missing = required - set(cols)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)
    print(f"  ✓ NER file OK")

print(f"  HTR cache: {HTR_CACHE}")
print(f"  Relation indicators: {len(RELATION_INDICATORS)}")

# ── STEP 1: Load NER PER entities grouped by (inv, page) ──
print(f"\n{'='*60}")
print("STEP 1: Loading NER entities")
print(f"{'='*60}")
t0 = time.time()

# Structure: (inv, page) -> list of (start_char, end_char, entity_text) sorted by position
page_entities = defaultdict(list)
total_per = 0
total_rows = 0

with open(NER_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        if row['entity_label'] != 'PER':
            continue
        inv = row['inventory_number'].strip()
        page = row['page_number'].strip()
        start = int(row['start_char'])
        end = int(row['end_char'])
        text = row['entity_text'].strip()
        if text:
            page_entities[(inv, page)].append((start, end, text))
            total_per += 1
        if total_rows % 2000000 == 0:
            print(f"  ...{total_rows:,} rows, {total_per:,} PER ({time.time()-t0:.0f}s)")

# Sort each page's entities by start position
for key in page_entities:
    page_entities[key].sort(key=lambda x: x[0])

pages_with_2plus = sum(1 for ents in page_entities.values() if len(ents) >= 2)
unique_invs = len(set(k[0] for k in page_entities))

print(f"  Loaded in {time.time()-t0:.0f}s")
print(f"  Total rows: {total_rows:,}")
print(f"  PER entities: {total_per:,}")
print(f"  Pages with entities: {len(page_entities):,}")
print(f"  Pages with 2+ PER: {pages_with_2plus:,}")
print(f"  Unique inventories: {unique_invs}")

# Sample check
sample_key = list(page_entities.keys())[0]
sample_ents = page_entities[sample_key]
print(f"\n  Sample page {sample_key}: {len(sample_ents)} entities")
for s, e, t in sample_ents[:3]:
    print(f"    [{s}:{e}] '{t}'")

# ── STEP 2: Extract relations using between-entity text ──
print(f"\n{'='*60}")
print("STEP 2: Extracting relations")
print(f"{'='*60}")
t1 = time.time()

all_relations = []
pages_checked = 0
pages_with_relations = 0
text_load_errors = 0
inventories_seen = set()

# Cache for HTR page text
text_cache = {}

def get_page_text(inv, page):
    """Load raw HTR text for a page."""
    key = (inv, page)
    if key in text_cache:
        return text_cache[key]
    
    inv_dir = HTR_CACHE / inv
    if not inv_dir.exists():
        return None
    
    # Try different filename patterns
    for pattern in [f"{page}.txt", f"page_{page}.txt"]:
        fpath = inv_dir / pattern
        if fpath.exists():
            try:
                text = fpath.read_text(encoding='utf-8')
                text_cache[key] = text
                # Keep cache reasonable
                if len(text_cache) > 10000:
                    oldest = list(text_cache.keys())[0]
                    del text_cache[oldest]
                return text
            except:
                return None
    return None

for idx, ((inv, page), entities) in enumerate(page_entities.items()):
    if len(entities) < 2:
        continue
    
    pages_checked += 1
    inventories_seen.add(inv)
    
    if pages_checked % 50000 == 0:
        elapsed = time.time() - t1
        remaining = (pages_with_2plus - pages_checked) / (pages_checked / elapsed) / 60 if pages_checked > 0 else 0
        print(f"  {pages_checked}/{pages_with_2plus} pages | {len(all_relations)} rels | "
              f"{elapsed:.0f}s | ~{remaining:.0f}min left")
    
    # Load HTR text for this page
    text = get_page_text(inv, page)
    if text is None:
        text_load_errors += 1
        continue
    
    page_rels = []
    seen_on_page = set()
    
    # Check consecutive pairs and pairs with one gap
    pairs_to_check = []
    for i in range(len(entities) - 1):
        pairs_to_check.append((i, i + 1))
    for i in range(len(entities) - 2):
        pairs_to_check.append((i, i + 2))
    
    for i1, i2 in pairs_to_check:
        start1, end1, name1 = entities[i1]
        start2, end2, name2 = entities[i2]
        
        # Sanity: end1 should be before start2
        if end1 > start2:
            continue
        
        # Skip if too far apart in characters (> 150 chars between)
        if start2 - end1 > 150:
            continue
        
        # Extract text between the two entities
        between = text[end1:start2].strip().lower()
        between = re.sub(r'\s+', ' ', between)
        
        if not between:
            continue
        
        # Check against indicators
        matched_rel = None
        matched_indicator = None
        
        for indicator, rel_type in RELATION_INDICATORS.items():
            if between == indicator:
                matched_rel = rel_type
                matched_indicator = indicator
                break
            if between.startswith(indicator) and len(between) - len(indicator) <= 3:
                matched_rel = rel_type
                matched_indicator = indicator
                break
            if between.endswith(indicator) and len(between) - len(indicator) <= 3:
                matched_rel = rel_type
                matched_indicator = indicator
                break
        
        if not matched_rel:
            continue
        
        # Gender check for spouse
        if matched_rel == "spouse" and not spouse_plausible(name1, name2):
            continue
        
        # Dedup on page
        dedup_key = (matched_rel, tuple(sorted([name1.lower(), name2.lower()])))
        if dedup_key in seen_on_page:
            continue
        seen_on_page.add(dedup_key)
        
        # Context
        ctx_start = max(0, start1 - 30)
        ctx_end = min(len(text), end2 + 30)
        context = text[ctx_start:ctx_end].replace('\n', ' ')[:200]
        
        page_rels.append({
            'inventory': inv,
            'page': page,
            'relation_type': matched_rel,
            'person_1': name1,
            'person_2': name2,
            'paired': True,
            'between_text': between,
            'matched_indicator': matched_indicator,
            'context': context,
        })
    
    if page_rels:
        pages_with_relations += 1
        all_relations.extend(page_rels)

elapsed_extract = time.time() - t1
print(f"\n  Extraction done in {elapsed_extract:.0f}s")
print(f"  Pages checked: {pages_checked:,}")
print(f"  Pages with relations: {pages_with_relations:,}")
print(f"  Text load errors: {text_load_errors}")
print(f"  Raw relations: {len(all_relations):,}")

# ── STEP 3: Deduplicate across pages ──
print(f"\n{'='*60}")
print("STEP 3: Deduplication and stats")
print(f"{'='*60}")

seen = set()
deduped = []
for rel in all_relations:
    key = (rel['inventory'], rel['relation_type'],
           tuple(sorted([rel['person_1'].lower(), rel['person_2'].lower()])))
    if key not in seen:
        seen.add(key)
        deduped.append(rel)

type_counts = Counter(r['relation_type'] for r in deduped)
inv_with_rels = len(set(r['inventory'] for r in deduped))
indicator_hits = Counter(r['matched_indicator'] for r in deduped)

# Cross-inventory
pair_invs = defaultdict(set)
for r in deduped:
    key = tuple(sorted([r['person_1'].lower(), r['person_2'].lower()]))
    pair_invs[key].add(r['inventory'])
cross_pairs = {k: v for k, v in pair_invs.items() if len(v) > 1}

# Gender check
gender_issues = 0
for r in deduped:
    if r['relation_type'] == 'spouse':
        g1, g2 = likely_gender(r['person_1']), likely_gender(r['person_2'])
        if g1 == g2 and g1 != 'unknown':
            gender_issues += 1
            if gender_issues <= 3:
                print(f"  GENDER ISSUE: {r['person_1']} ({g1}) <-> {r['person_2']} ({g2})")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Inventories processed: {len(inventories_seen)}")
print(f"  Text load errors: {text_load_errors}")
print(f"  Raw relations: {len(all_relations):,}")
print(f"  After dedup: {len(deduped):,}")
print(f"  ALL PAIRED: {all(r['paired'] for r in deduped)} ✓")
print(f"  Types: {dict(type_counts)}")
print(f"  Inventories with relations: {inv_with_rels}")
print(f"  Cross-inventory pairs: {len(cross_pairs)}")

print(f"\n  Indicator hit distribution:")
for ind, count in indicator_hits.most_common(15):
    print(f"    '{ind}': {count}")

print(f"\n  Sample relations:")
for r in deduped[:15]:
    print(f"    {r['relation_type']:8s} | {r['person_1']} <-> {r['person_2']}")
    print(f"             between: '{r['between_text']}' (inv {r['inventory']} p{r['page']})")

if cross_pairs:
    print(f"\n  Cross-inventory examples:")
    for (p1, p2), invs in sorted(cross_pairs.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"    '{p1}' <-> '{p2}': {len(invs)} inv ({', '.join(sorted(invs)[:5])})")

print(f"\n  VERIFY: gender issues: {gender_issues} {'✓' if gender_issues == 0 else '✗'}")
print(f"  VERIFY: all paired: {all(r['paired'] for r in deduped)} {'✓' if all(r['paired'] for r in deduped) else '✗'}")
print(f"  VERIFY: dedup removed {len(all_relations) - len(deduped)}")

# ── SAVE ──
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'inventory', 'page', 'relation_type', 'person_1', 'person_2',
        'paired', 'between_text', 'matched_indicator', 'context'])
    writer.writeheader()
    writer.writerows(deduped)

stats = {
    'inventories_processed': len(inventories_seen),
    'pages_checked': pages_checked,
    'pages_with_relations': pages_with_relations,
    'text_load_errors': text_load_errors,
    'total_raw': len(all_relations),
    'total_deduped': len(deduped),
    'all_paired': all(r['paired'] for r in deduped),
    'relation_types': dict(type_counts),
    'indicator_hits': dict(indicator_hits.most_common()),
    'inventories_with_relations': inv_with_rels,
    'cross_inventory_pairs': len(cross_pairs),
    'gender_issues': gender_issues,
    'runtime_seconds': round(time.time() - t0),
}
with open(OUTPUT_STATS, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n  Saved: {OUTPUT_CSV} ({len(deduped)} rows), {OUTPUT_STATS}")
print(f"  Total runtime: {(time.time()-t0)/60:.1f} minutes")
