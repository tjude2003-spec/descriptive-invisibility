"""
RQ3 Case Study v2: Sub-networks with decomposed name components.
Shows how component-level matching reveals cluster structure.

Usage:
    python rq3_case_study_v2.py "JAN VERLEIJ"
    python rq3_case_study_v2.py "PHILIP ZWEERTS"
    python rq3_case_study_v2.py --list

Requires: relations_all.csv, name_decomposition_v2_full.csv,
          htr_inventory_notary_mapping.csv
"""
import csv, json, sys, time
from collections import Counter, defaultdict

try:
    import networkx as nx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'networkx', '--break-system-packages', '-q'])
    import networkx as nx

VALID_TYPES = {'widow', 'spouse', 'child', 'sibling', 'widower'}

# ── Load decomposition ──
decomp = {}
with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        decomp[row['entity'].strip()] = row

# ── Load notary mapping ──
inv_notary = {}
notary_invs = defaultdict(set)
for mapfile in ['htr_inventory_notary_mapping.csv', 'final_velehanden_inventory_summary_.csv']:
    try:
        with open(mapfile, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                inv = row['inventory_number'].strip()
                notary = row['notary'].strip()
                if notary and inv not in inv_notary:
                    inv_notary[inv] = notary
                    notary_invs[notary].add(inv)
        break
    except FileNotFoundError:
        continue

if len(sys.argv) < 2 or sys.argv[1] == '--list':
    notary_rels = Counter()
    with open('relations_all.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['relation_type'] not in VALID_TYPES:
                continue
            inv = row['inventory'].strip()
            if inv in inv_notary:
                notary_rels[inv_notary[inv]] += 1
    print("Available notaries:")
    for notary, count in notary_rels.most_common(20):
        print(f"  {notary:35s} {count:5d} relations")
    sys.exit(0)

target_notary = sys.argv[1].upper().strip()
target_invs = notary_invs.get(target_notary, set())
if not target_invs:
    matches = [n for n in notary_invs if target_notary in n]
    if matches:
        print(f"No exact match. Did you mean: {matches}")
    sys.exit(1)

safe_name = target_notary.lower().replace(' ', '_').replace('.', '')

# ── Build sub-network ──
G = nx.Graph()
type_counts = Counter()

with open('relations_all.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['relation_type'] not in VALID_TYPES:
            continue
        inv = row['inventory'].strip()
        if inv not in target_invs:
            continue
        p1 = row['person_1'].strip().lower()
        p2 = row['person_2'].strip().lower()
        rtype = row['relation_type']
        type_counts[rtype] += 1

        for p in (p1, p2):
            if p not in G and p in decomp:
                d = decomp[p]
                G.add_node(p, given=d['given_name'], patronymic=d['patronymic'],
                          prefix=d['prefix'], family=d['family_name'],
                          gender=d['gender_marker'], structure=d['structure_type'])

        if G.has_edge(p1, p2):
            G[p1][p2]['weight'] += 1
            G[p1][p2]['inventories'].add(inv)
        else:
            G.add_edge(p1, p2, weight=1, relation_type=rtype, inventories={inv})

components = list(nx.connected_components(G))
components_sorted = sorted(components, key=len, reverse=True)

# ── Analyze clusters with decomposition ──
cluster_analyses = []
for comp in components_sorted[:10]:
    subg = G.subgraph(comp)
    members = []
    given_groups = defaultdict(list)

    for node in sorted(comp):
        d = decomp.get(node, {})
        info = {
            'entity': node,
            'given': d.get('given_name', ''),
            'patronymic': d.get('patronymic', ''),
            'prefix': d.get('prefix', ''),
            'family': d.get('family_name', ''),
            'gender': d.get('gender_marker', ''),
        }
        members.append(info)
        if info['given']:
            given_groups[info['given']].append(node)

    # Count unique given names
    unique_givens = set(m['given'] for m in members if m['given'])
    # Could component-matching collapse this?
    # If unique_givens <= 2 for a component of size > 2, it's likely fragmentation
    collapsible = len(unique_givens) <= 2 and len(comp) > 2

    relations = []
    for u, v, d in subg.edges(data=True):
        relations.append({
            'person_1': u, 'person_2': v,
            'type': d.get('relation_type', '?'), 'weight': d['weight']
        })

    cluster_analyses.append({
        'size': len(comp),
        'edges': subg.number_of_edges(),
        'unique_given_names': len(unique_givens),
        'given_name_groups': {g: sorted(nodes) for g, nodes in given_groups.items()},
        'collapsible_by_given': collapsible,
        'members': members,
        'relations': relations,
    })

# ── Summary stats ──
n_nodes = G.number_of_nodes()
n_comps = len(components)
comps_3plus = [c for c in components if len(c) >= 3]
collapsible_count = sum(1 for ca in cluster_analyses if ca.get('collapsible_by_given', False))

# For ALL components >= 3, check collapsibility
all_collapsible = 0
for comp in components:
    if len(comp) < 3:
        continue
    givens = set()
    for node in comp:
        d = decomp.get(node, {})
        g = d.get('given_name', '')
        if g:
            givens.add(g)
    if len(givens) <= 2:
        all_collapsible += 1

stats = {
    'notary': target_notary,
    'nodes': n_nodes,
    'edges': G.number_of_edges(),
    'total_relations': sum(type_counts.values()),
    'relation_types': dict(type_counts.most_common()),
    'components': n_comps,
    'components_3plus': len(comps_3plus),
    'collapsible_by_given_name': all_collapsible,
    'collapsible_pct_of_3plus': round(all_collapsible / max(len(comps_3plus), 1) * 100, 1),
    'top_clusters': cluster_analyses,
}

with open(f'rq3_case_{safe_name}_v2.json', 'w') as f:
    json.dump(stats, f, indent=2, default=list)

print(f"\n{'='*60}")
print(f"CASE STUDY: {target_notary} (with decomposition)")
print(f"{'='*60}")
print(f"  Nodes: {n_nodes:,}  Edges: {G.number_of_edges():,}")
print(f"  Components: {n_comps:,}  (≥3 members: {len(comps_3plus)})")
print(f"  Collapsible by given-name matching: {all_collapsible} of {len(comps_3plus)} ({all_collapsible/max(len(comps_3plus),1)*100:.0f}%)")

for i, ca in enumerate(cluster_analyses[:5]):
    print(f"\n  Cluster {i+1} (size={ca['size']}, givens={ca['unique_given_names']}, "
          f"collapsible={ca['collapsible_by_given']})")
    for g, nodes in ca['given_name_groups'].items():
        print(f"    given='{g}': {', '.join(nodes[:5])}")
    for m in ca['members'][:8]:
        parts = [f"giv={m['given']}"]
        if m['patronymic']: parts.append(f"pat={m['patronymic']}")
        if m['prefix']: parts.append(f"pfx={m['prefix']}")
        if m['family']: parts.append(f"fam={m['family']}")
        print(f"      {m['entity']:40s} {', '.join(parts)}")

print(f"\n  Saved: rq3_case_{safe_name}_v2.json")
