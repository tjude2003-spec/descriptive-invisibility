#!/usr/bin/env python3
"""
RQ2 Population Characterization — with full debugging
"""
import csv, json, time, random, statistics, sys, os
from collections import defaultdict, Counter
from rapidfuzz import fuzz

random.seed(42)
NER_FILE = "ner_extractions_slim.csv"
OUTPUT_JSON = "rq2_results.json"

print("=" * 60)
print("PREFLIGHT CHECKS")
print("=" * 60)

if not os.path.exists(NER_FILE):
    print(f"  ERROR: {NER_FILE} not found in {os.getcwd()}")
    print(f"  Files here: {[f for f in os.listdir('.') if f.endswith('.csv')]}")
    sys.exit(1)

with open(NER_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    print(f"  Columns: {cols}")
    required = {'inventory_number', 'page_number', 'entity_text'}
    missing = required - set(cols)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)
    first = next(reader)
    print(f"  First row: {first}")
    print(f"  ✓ File structure OK")

print(f"  Counting rows...")
with open(NER_FILE, encoding='utf-8') as f:
    total_lines = sum(1 for _ in f) - 1
print(f"  Total rows: {total_lines:,}")

print(f"\n{'='*60}")
print("LOADING")
print(f"{'='*60}")
t0 = time.time()

entity_inventories = defaultdict(set)
inv_entities = defaultdict(set)
total_mentions = 0
empty_entities = 0

with open(NER_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_mentions += 1
        entity = row['entity_text'].strip()
        if not entity:
            empty_entities += 1
            continue
        entity_lower = entity.lower()
        inv = row['inventory_number']
        entity_inventories[entity_lower].add(inv)
        inv_entities[inv].add(entity_lower)
        if total_mentions % 1000000 == 0:
            print(f"  ...{total_mentions:,} rows ({time.time()-t0:.0f}s)")

unique_entities = len(entity_inventories)
unique_inventories = len(inv_entities)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Mentions: {total_mentions:,} | Empty skipped: {empty_entities}")
print(f"  Unique entities: {unique_entities:,} | Inventories: {unique_inventories}")
print(f"  VERIFY: rows read ({total_mentions:,}) vs line count ({total_lines:,}): {'✓' if abs(total_mentions - total_lines) <= 10 else '✗ MISMATCH'}")

# ── ANALYSIS 1 ──
print(f"\n{'─'*60}")
print("ANALYSIS 1: Frequency Distribution")
print(f"{'─'*60}")
inv_counts = Counter(len(invs) for invs in entity_inventories.values())
singletons = inv_counts[1]
appear_2 = inv_counts[2]
appear_3_5 = sum(inv_counts[i] for i in range(3, 6))
appear_6_10 = sum(inv_counts[i] for i in range(6, 11))
appear_11_plus = sum(v for k, v in inv_counts.items() if k >= 11)
bucket_sum = singletons + appear_2 + appear_3_5 + appear_6_10 + appear_11_plus

print(f"  Singletons: {singletons:,} ({singletons/unique_entities*100:.1f}%)")
print(f"  2 inv: {appear_2:,} ({appear_2/unique_entities*100:.1f}%)")
print(f"  3-5 inv: {appear_3_5:,} ({appear_3_5/unique_entities*100:.1f}%)")
print(f"  6-10 inv: {appear_6_10:,} ({appear_6_10/unique_entities*100:.1f}%)")
print(f"  11+ inv: {appear_11_plus:,} ({appear_11_plus/unique_entities*100:.1f}%)")
print(f"  VERIFY: bucket sum {bucket_sum:,} == unique {unique_entities:,}: {'✓' if bucket_sum == unique_entities else '✗'}")

top_names = sorted(entity_inventories.items(), key=lambda x: len(x[1]), reverse=True)[:20]
print(f"\n  Top 20:")
for e, invs in top_names:
    print(f"    '{e}': {len(invs)} inv")

with open("rq2_frequency_distribution.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['inventory_count', 'num_entities'])
    for k in sorted(inv_counts):
        w.writerow([k, inv_counts[k]])
print(f"  Saved: rq2_frequency_distribution.csv")

# ── ANALYSIS 2 ──
print(f"\n{'─'*60}")
print("ANALYSIS 2: Cross-Inventory Appearance Rate")
print(f"{'─'*60}")
cross_inv = sum(1 for invs in entity_inventories.values() if len(invs) > 1)
cross_3 = sum(1 for invs in entity_inventories.values() if len(invs) > 2)
cross_5 = sum(1 for invs in entity_inventories.values() if len(invs) > 5)
cross_10 = sum(1 for invs in entity_inventories.values() if len(invs) > 10)

print(f"  >1 inv: {cross_inv:,} ({cross_inv/unique_entities*100:.1f}%)")
print(f"  >2 inv: {cross_3:,} ({cross_3/unique_entities*100:.1f}%)")
print(f"  >5 inv: {cross_5:,} ({cross_5/unique_entities*100:.1f}%)")
print(f"  >10 inv: {cross_10:,} ({cross_10/unique_entities*100:.1f}%)")
print(f"  VERIFY: singletons({singletons}) + cross({cross_inv}) = {singletons+cross_inv} == {unique_entities}: {'✓' if singletons+cross_inv == unique_entities else '✗'}")

densities = sorted(len(ents) for ents in inv_entities.values())
print(f"\n  Entity density per inventory:")
print(f"    Mean: {statistics.mean(densities):.1f} | Median: {statistics.median(densities):.1f}")
print(f"    Min: {min(densities)} | Max: {max(densities)} | Stdev: {statistics.stdev(densities):.1f}")
print(f"    25th: {densities[len(densities)//4]} | 75th: {densities[3*len(densities)//4]}")

cross_count = 0
with open("rq2_cross_inventory.csv", 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['entity', 'inventory_count', 'inventories'])
    for entity, invs in sorted(entity_inventories.items(), key=lambda x: len(x[1]), reverse=True):
        if len(invs) > 1:
            w.writerow([entity, len(invs), ';'.join(sorted(invs))])
            cross_count += 1
print(f"  Saved: rq2_cross_inventory.csv ({cross_count} rows)")
print(f"  VERIFY: CSV rows ({cross_count}) == cross_inv count ({cross_inv}): {'✓' if cross_count == cross_inv else '✗'}")

# ── ANALYSIS 3 ──
print(f"\n{'─'*60}")
print("ANALYSIS 3: Orthographic Variation Density")
print(f"{'─'*60}")
multi_token = [e for e in entity_inventories.keys() if len(e.split()) >= 2]
print(f"  Multi-token: {len(multi_token):,} of {unique_entities:,}")

sample_size = min(10000, len(multi_token))
sample = random.sample(multi_token, sample_size)
print(f"  Sample: {sample_size} entities")
print(f"  Examples: {sample[:3]}")

est_comparisons = sample_size * sample_size
print(f"  Estimated comparisons: ~{est_comparisons:,}")
print(f"  This may take 15-30 minutes. Starting...")

t1 = time.time()
has_near_dup = 0
near_dup_examples = []

for i, name in enumerate(sample):
    if i % 500 == 0:
        elapsed = time.time() - t1
        if i > 0:
            rate = i / elapsed
            remaining = (sample_size - i) / rate
            print(f"    {i}/{sample_size} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining | dupes so far: {has_near_dup}")
        else:
            print(f"    {i}/{sample_size} | starting...")
    
    best_score = 0
    best_match = ""
    
    for j, other in enumerate(sample):
        if i == j:
            continue
        if abs(len(name) - len(other)) > max(len(name), len(other)) * 0.4:
            continue
        score = fuzz.token_sort_ratio(name, other) / 100.0
        if score > best_score:
            best_score = score
            best_match = other
    
    if best_score >= 0.85 and name != best_match:
        has_near_dup += 1
        if len(near_dup_examples) < 20:
            near_dup_examples.append({'name': name, 'match': best_match, 'score': round(best_score, 3)})

elapsed3 = time.time() - t1
variation_rate = has_near_dup / sample_size
print(f"\n  Done in {elapsed3:.0f}s")
print(f"  Near-duplicates: {has_near_dup} ({variation_rate*100:.1f}%)")
print(f"  Extrapolated: ~{int(variation_rate * len(multi_token)):,}")

print(f"\n  Examples:")
for ex in near_dup_examples[:15]:
    print(f"    '{ex['name']}' <-> '{ex['match']}' ({ex['score']})")

# Verify a few
print(f"\n  VERIFY: re-computing 3 scores...")
for ex in near_dup_examples[:3]:
    recomputed = fuzz.token_sort_ratio(ex['name'], ex['match']) / 100.0
    match = abs(recomputed - ex['score']) < 0.001
    print(f"    '{ex['name']}' vs '{ex['match']}': stored={ex['score']}, recomputed={recomputed:.3f} {'✓' if match else '✗'}")

# ── SAVE ──
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

results = {
    "total_mentions": total_mentions,
    "unique_entities": unique_entities,
    "unique_inventories": unique_inventories,
    "empty_entities_skipped": empty_entities,
    "singleton_count": singletons,
    "singleton_rate": round(singletons / unique_entities, 4),
    "cross_inventory_count": cross_inv,
    "cross_inventory_rate": round(cross_inv / unique_entities, 4),
    "cross_inventory_3plus": cross_3,
    "cross_inventory_5plus": cross_5,
    "cross_inventory_10plus": cross_10,
    "density_mean": round(statistics.mean(densities), 1),
    "density_median": round(statistics.median(densities), 1),
    "density_min": min(densities),
    "density_max": max(densities),
    "density_stdev": round(statistics.stdev(densities), 1),
    "multi_token_entities": len(multi_token),
    "orthographic_variation_sample_size": sample_size,
    "orthographic_variation_rate": round(variation_rate, 4),
    "orthographic_variation_count": has_near_dup,
    "top_20_names": [{"name": e, "inventory_count": len(invs)} for e, invs in top_names],
    "near_duplicate_examples": near_dup_examples[:10],
    "verification": {
        "bucket_sum_equals_unique": bucket_sum == unique_entities,
        "singleton_plus_cross_equals_unique": singletons + cross_inv == unique_entities,
        "row_count_matches_line_count": abs(total_mentions - total_lines) <= 10,
        "cross_inv_csv_matches_count": cross_count == cross_inv,
    }
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Saved: {OUTPUT_JSON}")
print(f"\n  FINAL CHECKS:")
for check, passed in results['verification'].items():
    status = '✓' if passed else '✗ FAILED'
    print(f"    {check}: {status}")

print(f"\n  Total runtime: {time.time()-t0:.0f}s")
