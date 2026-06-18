"""
Orthographic variation with decomposition — v3.
Family-name check + token-reorder inflation check.

Requires: name_decomposition_v2_full.csv
Output: rq2_variation_decomposed_v3.json
"""
import csv, json, time, random
from rapidfuzz import fuzz, process

random.seed(42)

FAMILY_THRESHOLD = 80  # fuzz.ratio below this = different family name

print("Loading decomposed entities...")
t0 = time.time()

entities = {}
multi_token = []

with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        ent = row['entity'].strip()
        entities[ent] = row
        if len(ent.split()) >= 2:
            multi_token.append(ent)

print(f"  {len(multi_token):,} multi-token ({time.time()-t0:.1f}s)")

sample = random.sample(multi_token, min(10000, len(multi_token)))
print(f"  Sampled {len(sample):,}")

print("\nFinding near-duplicates...")
t1 = time.time()

results = {
    'total_sampled': len(sample),
    'has_near_duplicate': 0,
    'genuine_variant': 0,
    'given_match_family_differs': 0,
    'coincidental_match': 0,
    'given_missing': 0,
    'token_reorder_inflated': 0,
    'examples_genuine': [],
    'examples_given_match_family_differs': [],
    'examples_coincidental': [],
    'examples_reorder': [],
}

for i, ent in enumerate(sample):
    if i % 2000 == 0 and i > 0:
        print(f"  ...{i}/{len(sample)} ({time.time()-t1:.0f}s)")

    matches = process.extract(ent, sample, scorer=fuzz.token_sort_ratio,
                               score_cutoff=85, limit=3)

    best = None
    for match_str, score, idx in matches:
        if match_str != ent:
            best = (match_str, score / 100.0)
            break

    if best is None:
        continue

    match_str, score = best
    results['has_near_duplicate'] += 1

    # Check if match depends on token reordering
    plain_ratio = fuzz.ratio(ent, match_str) / 100.0
    if score - plain_ratio > 0.15:
        results['token_reorder_inflated'] += 1
        if len(results['examples_reorder']) < 10:
            results['examples_reorder'].append({
                'entity': ent, 'match': match_str,
                'token_sort': round(score, 3), 'plain': round(plain_ratio, 3),
            })

    ent_given = entities.get(ent, {}).get('given_name', '')
    match_given = entities.get(match_str, {}).get('given_name', '')

    if not ent_given or not match_given:
        results['given_missing'] += 1
        continue

    given_match = fuzz.ratio(ent_given, match_given) >= 80

    if not given_match:
        results['coincidental_match'] += 1
        if len(results['examples_coincidental']) < 15:
            results['examples_coincidental'].append({
                'entity': ent, 'match': match_str, 'score': round(score, 3),
                'ent_given': ent_given, 'match_given': match_given,
            })
        continue

    # Given names match — now check family names
    ent_family = entities.get(ent, {}).get('family_name', '').strip()
    match_family = entities.get(match_str, {}).get('family_name', '').strip()

    if ent_family and match_family:
        family_sim = fuzz.ratio(ent_family, match_family)
        if family_sim >= FAMILY_THRESHOLD:
            results['genuine_variant'] += 1
            if len(results['examples_genuine']) < 15:
                results['examples_genuine'].append({
                    'entity': ent, 'match': match_str, 'score': round(score, 3),
                    'shared_given': ent_given,
                    'ent_family': ent_family, 'match_family': match_family,
                    'family_sim': round(family_sim, 1),
                })
        else:
            results['given_match_family_differs'] += 1
            if len(results['examples_given_match_family_differs']) < 15:
                results['examples_given_match_family_differs'].append({
                    'entity': ent, 'match': match_str, 'score': round(score, 3),
                    'shared_given': ent_given,
                    'ent_family': ent_family, 'match_family': match_family,
                    'family_sim': round(family_sim, 1),
                })
    else:
        results['genuine_variant'] += 1
        if len(results['examples_genuine']) < 15:
            results['examples_genuine'].append({
                'entity': ent, 'match': match_str, 'score': round(score, 3),
                'shared_given': ent_given,
                'ent_family': ent_family or '(none)',
                'match_family': match_family or '(none)',
                'family_sim': -1,
            })

n = results['has_near_duplicate']
not_genuine = results['coincidental_match'] + results['given_match_family_differs']
classifiable = results['genuine_variant'] + not_genuine

results['near_duplicate_rate'] = round(n / results['total_sampled'] * 100, 2)
results['genuine_pct'] = round(results['genuine_variant'] / max(classifiable, 1) * 100, 2)
results['given_match_family_differs_pct'] = round(
    results['given_match_family_differs'] / max(classifiable, 1) * 100, 2)
results['coincidental_pct'] = round(results['coincidental_match'] / max(classifiable, 1) * 100, 2)

results['fragmentation_rate'] = round(results['genuine_variant'] / results['total_sampled'] * 100, 2)
results['noise_rate'] = round(not_genuine / results['total_sampled'] * 100, 2)

with open('rq2_variation_decomposed_v3.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("ORTHOGRAPHIC VARIATION — DECOMPOSED v3")
print(f"{'='*60}")
print(f"  Sampled: {results['total_sampled']:,}")
print(f"  Near-duplicates (>=0.85): {n:,} ({results['near_duplicate_rate']}%)")
print(f"  Family name threshold: {FAMILY_THRESHOLD}")
print(f"\n  Of {classifiable:,} classifiable:")
print(f"    Genuine (given+family match): {results['genuine_variant']:,} ({results['genuine_pct']}%)")
print(f"    Given match, family differs:  {results['given_match_family_differs']:,} ({results['given_match_family_differs_pct']}%)")
print(f"    Coincidental (diff given):    {results['coincidental_match']:,} ({results['coincidental_pct']}%)")
print(f"    Missing given name:           {results['given_missing']:,}")
print(f"\n  Effective rates (against 10K sample):")
print(f"    Fragmentation rate: {results['fragmentation_rate']}%")
print(f"    Noise rate:         {results['noise_rate']}%")
print(f"\n  Token-reorder inflated: {results['token_reorder_inflated']} of {n} near-duplicates")
for ex in results['examples_reorder'][:5]:
    print(f"    {ex['entity']:35s} <> {ex['match']:35s} (sort={ex['token_sort']}, plain={ex['plain']})")
print(f"\n  Genuine examples:")
for ex in results['examples_genuine'][:5]:
    fam = f"fam={ex['ent_family']}/{ex['match_family']}"
    sim = f" sim={ex['family_sim']}" if ex.get('family_sim', -1) >= 0 else ""
    print(f"    {ex['entity']:35s} <> {ex['match']:35s} (given={ex['shared_given']}, {fam}{sim}, {ex['score']})")
print(f"\n  Given-match-but-different-family examples:")
for ex in results['examples_given_match_family_differs'][:5]:
    print(f"    {ex['entity']:35s} <> {ex['match']:35s} (given={ex['shared_given']}, fam={ex['ent_family']}/{ex['match_family']} sim={ex['family_sim']}, {ex['score']})")
print(f"\n  Coincidental examples:")
for ex in results['examples_coincidental'][:5]:
    print(f"    {ex['entity']:35s} <> {ex['match']:35s} ({ex['ent_given']}!={ex['match_given']}, {ex['score']})")
print(f"\n  Runtime: {time.time()-t0:.0f}s")
