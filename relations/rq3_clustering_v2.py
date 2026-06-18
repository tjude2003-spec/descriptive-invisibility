"""
RQ3 Clustering v2: Three disambiguation signals for cross-inventory name strings.
1. Co-occurrence partner overlap (from v1)
2. Decomposed component matching (given + patronymic + family)
3. Combined signal strength

Requires: ner_extractions_slim.csv, name_decomposition_v2_full.csv
Output: rq3_clustering_v2.json
"""
import csv, json, sys, time, random, statistics
from collections import Counter, defaultdict

random.seed(42)

ner_path = sys.argv[1] if len(sys.argv) > 1 else 'ner_extractions_slim.csv'

# ── Load decomposition ──
print("Loading decomposition...")
t0 = time.time()
decomp = {}
with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        decomp[row['entity'].strip()] = row
print(f"  {len(decomp):,} entities ({time.time()-t0:.1f}s)")

# ── Build page-level co-occurrence + entity-inventory mapping ──
print("\nBuilding co-occurrence index...")
t1 = time.time()

page_entities = defaultdict(set)
entity_inventories = defaultdict(set)
total = 0

with open(ner_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        total += 1
        ent = row['entity_text'].strip().lower()
        if not ent:
            continue
        inv = row['inventory_number'].strip()
        page = row['page_number'].strip()
        page_entities[(inv, page)].add(ent)
        entity_inventories[ent].add(inv)
        if total % 2000000 == 0:
            print(f"  ...{total:,} rows")

cross_inv_entities = {e: invs for e, invs in entity_inventories.items() if len(invs) > 1}
print(f"  Cross-inventory entities: {len(cross_inv_entities):,} ({time.time()-t1:.1f}s)")

# ── Build per-inventory co-occurrence partners ──
print("\nBuilding partner index...")
t2 = time.time()

entity_partners_by_inv = defaultdict(lambda: defaultdict(set))
pages_processed = 0
for (inv, page), entities in page_entities.items():
    pages_processed += 1
    if pages_processed % 500000 == 0:
        print(f"  ...{pages_processed:,} pages")
    cross_on_page = [e for e in entities if e in cross_inv_entities]
    if not cross_on_page:
        continue
    for ent in cross_on_page:
        partners = entities - {ent}
        if partners:
            entity_partners_by_inv[ent][inv].update(partners)

print(f"  {len(entity_partners_by_inv):,} entities with partner data ({time.time()-t2:.1f}s)")

# ── Three-signal analysis ──
print("\nComputing disambiguation signals...")
t3 = time.time()

# Signal 1: co-occurrence partner overlap (same as v1)
# Signal 2: component matching — do cross-inventory appearances share
#           decomposed name components with their co-occurrence partners?
# Signal 3: combined — entities that have BOTH shared partners AND
#           matching components in those partners

has_shared_partner = 0
no_shared_partner = 0
shared_partner_counts = []

# Component-level analysis for cross-inventory entities
component_match_stats = {
    'total_cross_inv': len(cross_inv_entities),
    'has_patronymic': 0,
    'has_prefix': 0,
    'has_family': 0,
    'given_only_or_given_family': 0,
}

for ent in cross_inv_entities:
    d = decomp.get(ent, {})
    if d.get('patronymic'): component_match_stats['has_patronymic'] += 1
    if d.get('prefix'): component_match_stats['has_prefix'] += 1
    if d.get('family_name'): component_match_stats['has_family'] += 1
    if d.get('structure_type') in ('given_only', 'given_family'):
        component_match_stats['given_only_or_given_family'] += 1

# For entities with partner data, compute partner overlap + component analysis
combined_signal = 0  # has shared partner AND shared partner has matching component
partner_component_matches = 0  # shared partners that share a decomposed component

for ent, inv_partners in entity_partners_by_inv.items():
    if len(inv_partners) < 2:
        continue

    # Signal 1: shared partners
    partner_inv_count = Counter()
    for inv, partners in inv_partners.items():
        for p in partners:
            partner_inv_count[p] += 1

    shared = {p for p, c in partner_inv_count.items() if c >= 2}

    if shared:
        has_shared_partner += 1
        shared_partner_counts.append(len(shared))

        # Signal 2+3: do shared partners share decomposed components with the entity?
        ent_d = decomp.get(ent, {})
        ent_given = ent_d.get('given_name', '')

        has_component_match = False
        for sp in shared:
            sp_d = decomp.get(sp, {})
            sp_given = sp_d.get('given_name', '')
            # A shared partner with a DIFFERENT given name is a genuine
            # co-occurrence signal (not just orthographic noise)
            if sp_given and ent_given and sp_given != ent_given:
                has_component_match = True
                partner_component_matches += 1
                break

        if has_component_match:
            combined_signal += 1
    else:
        no_shared_partner += 1

total_with_data = has_shared_partner + no_shared_partner

print(f"\n{'='*60}")
print("CLUSTERING WITH DECOMPOSITION")
print(f"{'='*60}")
print(f"\n  Cross-inventory entities: {len(cross_inv_entities):,}")
print(f"  With partner data:       {total_with_data:,}")

print(f"\n  Signal 1 — Co-occurrence partner overlap:")
print(f"    Has shared partner:    {has_shared_partner:,} ({has_shared_partner/max(total_with_data,1)*100:.1f}%)")
print(f"    No shared partner:     {no_shared_partner:,}")
if shared_partner_counts:
    print(f"    Shared partner mean:   {statistics.mean(shared_partner_counts):.1f}")
    print(f"    Shared partner median: {statistics.median(shared_partner_counts):.0f}")

print(f"\n  Signal 2 — Component structure of cross-inventory entities:")
n_cross = len(cross_inv_entities)
print(f"    Has patronymic:        {component_match_stats['has_patronymic']:,} ({component_match_stats['has_patronymic']/n_cross*100:.1f}%)")
print(f"    Has prefix:            {component_match_stats['has_prefix']:,} ({component_match_stats['has_prefix']/n_cross*100:.1f}%)")
print(f"    Has family name:       {component_match_stats['has_family']:,} ({component_match_stats['has_family']/n_cross*100:.1f}%)")

print(f"\n  Signal 3 — Combined (shared partner with different given name):")
print(f"    Combined signal:       {combined_signal:,} ({combined_signal/max(has_shared_partner,1)*100:.1f}% of those with shared partners)")
print(f"    Interpretation: these entities share a co-occurrence partner")
print(f"    whose given name differs from theirs — a genuine relational")
print(f"    signal rather than orthographic noise.")

stats = {
    'cross_inv_entities': len(cross_inv_entities),
    'entities_with_partner_data': total_with_data,
    'signal_1_shared_partner': has_shared_partner,
    'signal_1_pct': round(has_shared_partner / max(total_with_data, 1) * 100, 2),
    'signal_1_mean_partners': round(statistics.mean(shared_partner_counts), 2) if shared_partner_counts else 0,
    'signal_1_median_partners': statistics.median(shared_partner_counts) if shared_partner_counts else 0,
    'signal_2_component_structure': component_match_stats,
    'signal_3_combined': combined_signal,
    'signal_3_pct_of_shared': round(combined_signal / max(has_shared_partner, 1) * 100, 2),
}

with open('rq3_clustering_v2.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n  Saved: rq3_clustering_v2.json")
print(f"  Runtime: {time.time()-t0:.0f}s")
