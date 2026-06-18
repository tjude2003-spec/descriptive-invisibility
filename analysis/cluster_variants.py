"""
Orthographic variant clustering.
Reuses the same comparison logic as rq2_variation_decomposed_v3.py
but collects ALL genuine variant pairs and clusters them
using connected components.

Requires: name_decomposition_v2_full.csv, rq2_cross_inventory.csv
Output:   variant_clusters.json, variant_clusters.csv
"""
import csv, json, time, random
from collections import defaultdict
from rapidfuzz import fuzz, process

try:
    import networkx as nx
except ImportError:
    print("Installing networkx...")
    import subprocess
    subprocess.check_call(["pip", "install", "networkx", "--break-system-packages", "-q"])
    import networkx as nx

random.seed(42)
FAMILY_THRESHOLD = 80

# ── Load entities ──
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

print(f"  {len(multi_token):,} multi-token entities ({time.time()-t0:.1f}s)")

# Same 10K sample as v3
sample = random.sample(multi_token, min(10000, len(multi_token)))
print(f"  Sampled {len(sample):,}")

# ── Load inventory locations ──
print("Loading cross-inventory data...")
inv_lookup = {}
with open('rq2_cross_inventory.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        inv_lookup[row['entity']] = row['inventories'].split(';')

# ── Find all genuine variant pairs ──
print("\nFinding genuine variant pairs...")
t1 = time.time()

genuine_pairs = []
stats = {
    'total_sampled': len(sample),
    'has_near_duplicate': 0,
    'genuine': 0,
    'coincidental': 0,
    'given_match_family_differs': 0,
    'given_missing': 0,
}

for i, ent in enumerate(sample):
    if i % 2000 == 0 and i > 0:
        print(f"  ...{i}/{len(sample)} ({time.time()-t1:.0f}s)")

    matches = process.extract(ent, sample, scorer=fuzz.token_sort_ratio,
                               score_cutoff=85, limit=5)

    for match_str, score, idx in matches:
        if match_str == ent:
            continue

        stats['has_near_duplicate'] += 1
        score = score / 100.0

        ent_given = entities.get(ent, {}).get('given_name', '')
        match_given = entities.get(match_str, {}).get('given_name', '')

        if not ent_given or not match_given:
            stats['given_missing'] += 1
            continue

        if fuzz.ratio(ent_given, match_given) < 80:
            stats['coincidental'] += 1
            continue

        ent_family = entities.get(ent, {}).get('family_name', '').strip()
        match_family = entities.get(match_str, {}).get('family_name', '').strip()

        is_genuine = False
        if ent_family and match_family:
            if fuzz.ratio(ent_family, match_family) >= FAMILY_THRESHOLD:
                is_genuine = True
            else:
                stats['given_match_family_differs'] += 1
        else:
            is_genuine = True

        if is_genuine:
            stats['genuine'] += 1
            genuine_pairs.append((ent, match_str, round(score, 3)))

# Deduplicate pairs (A,B) and (B,A)
seen = set()
unique_pairs = []
for a, b, s in genuine_pairs:
    key = tuple(sorted([a, b]))
    if key not in seen:
        seen.add(key)
        unique_pairs.append((a, b, s))

print(f"\n  Near-duplicates found: {stats['has_near_duplicate']:,}")
print(f"  Genuine pairs (before dedup): {stats['genuine']:,}")
print(f"  Unique genuine pairs: {len(unique_pairs):,}")

# ── Build graph and cluster ──
print("\nClustering...")
G = nx.Graph()
for a, b, s in unique_pairs:
    G.add_edge(a, b, similarity=s)

components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)

print(f"  {len(components):,} clusters")
print(f"  Largest cluster: {len(components[0])} members")
sizes = [len(c) for c in components]
print(f"  Size distribution: {sum(1 for s in sizes if s == 2)} pairs, "
      f"{sum(1 for s in sizes if s == 3)} triples, "
      f"{sum(1 for s in sizes if s >= 4)} larger")

# ── Build output ──
clusters = []
for cid, comp in enumerate(components):
    members = []
    all_inventories = set()
    for name in sorted(comp):
        invs = inv_lookup.get(name, [])
        all_inventories.update(invs)
        stype = entities.get(name, {}).get('structure_type', 'unknown')
        members.append({
            'name': name,
            'structure_type': stype,
            'inventory_count': len(invs),
            'inventories': invs[:20],  # cap for readability
        })
    clusters.append({
        'cluster_id': cid,
        'size': len(comp),
        'members': members,
        'combined_inventory_count': len(all_inventories),
    })

# ── Save JSON ──
output = {
    'stats': stats,
    'unique_pairs': len(unique_pairs),
    'num_clusters': len(clusters),
    'size_distribution': {
        'pairs': sum(1 for s in sizes if s == 2),
        'triples': sum(1 for s in sizes if s == 3),
        'larger': sum(1 for s in sizes if s >= 4),
    },
    'clusters': clusters,
}

with open('variant_clusters.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Save CSV (one row per cluster member) ──
with open('variant_clusters.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['cluster_id', 'cluster_size', 'name', 'structure_type',
                'inventory_count', 'sample_inventories'])
    for cl in clusters:
        for m in cl['members']:
            w.writerow([
                cl['cluster_id'], cl['size'], m['name'],
                m['structure_type'], m['inventory_count'],
                ';'.join(m['inventories'][:10]),
            ])

print(f"\nSaved variant_clusters.json and variant_clusters.csv")
print(f"Top 5 clusters:")
for cl in clusters[:5]:
    names = [m['name'] for m in cl['members']]
    print(f"  [{cl['cluster_id']}] ({cl['size']}): {', '.join(names[:6])}")

print(f"\nTotal runtime: {time.time()-t0:.0f}s")
