"""
RQ3 Network Stats v3: Formalized component classification.

Two criteria distinguish orthographic fragmentation from patronymic collision:
  1. Given-name ratio: unique given names <= 50% of component nodes
  2. Max degree: no node with degree > 15

Orthographic fragmentation produces small clusters of spelling variants with
low degree (each variant connects to the same few relational partners).
Patronymic collision produces high-degree hubs where unrelated persons sharing
a name string (e.g. "jan jansz") aggregate into apparent connectivity.

Requires: relations_all.csv, name_decomposition_v2_full.csv
Output: rq3_network_stats_v3.json
"""
import csv, json, time
from collections import Counter, defaultdict

try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'networkx', '--break-system-packages', '-q'])
    import networkx as nx

VALID_TYPES = {'widow', 'spouse', 'child', 'sibling', 'widower'}
MAX_DEGREE_THRESHOLD = 15

# ── Load decomposition ──
print("Loading name decomposition...")
t0 = time.time()
decomp = {}
with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        decomp[row['entity'].strip()] = row

print(f"  {len(decomp):,} decomposed entities ({time.time()-t0:.1f}s)")

# ── Build graph ──
print("Loading relations...")
G = nx.Graph()
type_counts = Counter()
total_relations = 0

with open('relations_all.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['relation_type'] not in VALID_TYPES:
            continue
        total_relations += 1
        p1 = row['person_1'].strip().lower()
        p2 = row['person_2'].strip().lower()
        rtype = row['relation_type']
        inv = row['inventory'].strip()
        type_counts[rtype] += 1

        for p in (p1, p2):
            if p not in G and p in decomp:
                d = decomp[p]
                G.add_node(p,
                    given=d['given_name'], patronymic=d['patronymic'],
                    prefix=d['prefix'], family=d['family_name'],
                    gender=d['gender_marker'], structure=d['structure_type'])

        if G.has_edge(p1, p2):
            G[p1][p2]['weight'] += 1
            G[p1][p2]['inventories'].add(inv)
            G[p1][p2]['types'].add(rtype)
        else:
            G.add_edge(p1, p2, weight=1, inventories={inv}, types={rtype},
                      relation_type=rtype)

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print(f"  {total_relations:,} relations, {n_nodes:,} nodes, {n_edges:,} edges")

# ── Component analysis with degree filter ──
print("\nAnalyzing components...")
components = list(nx.connected_components(G))
giant_comp = max(components, key=len)
giant_size = len(giant_comp)

# Edge-level given-name analysis
shared_given_edges = 0
different_given_edges = 0
missing_decomp_edges = 0
for u, v, d in G.edges(data=True):
    u_given = G.nodes[u].get('given', '') if u in G.nodes else ''
    v_given = G.nodes[v].get('given', '') if v in G.nodes else ''
    if not u_given or not v_given:
        missing_decomp_edges += 1
        continue
    if u_given == v_given:
        shared_given_edges += 1
    else:
        different_given_edges += 1

# Component classification — uniform criteria, no ad hoc exclusions
comp_counts = {
    'size_1': 0,
    'size_2': 0,
    'size_3plus': 0,
    'collapsible': 0,           # low given ratio AND low max degree
    'patronymic_artifact': 0,   # low given ratio BUT high max degree
    'no_convergence': 0,        # given ratio > 0.5
}
collapsible_sizes = Counter()
patronymic_artifact_details = []

for comp in components:
    if len(comp) == 1:
        comp_counts['size_1'] += 1
        continue
    if len(comp) == 2:
        comp_counts['size_2'] += 1
        continue

    comp_counts['size_3plus'] += 1

    givens = set()
    max_deg = 0
    for node in comp:
        g = G.nodes[node].get('given', '')
        if g:
            givens.add(g)
        max_deg = max(max_deg, G.degree(node))

    ratio = len(givens) / len(comp)

    if ratio <= 0.5 and max_deg <= MAX_DEGREE_THRESHOLD:
        comp_counts['collapsible'] += 1
        collapsible_sizes[len(comp)] += 1
    elif ratio <= 0.5 and max_deg > MAX_DEGREE_THRESHOLD:
        comp_counts['patronymic_artifact'] += 1
        patronymic_artifact_details.append({
            'size': len(comp),
            'unique_givens': len(givens),
            'given_ratio': round(ratio, 4),
            'max_degree': max_deg,
        })
    else:
        comp_counts['no_convergence'] += 1

# Gender distribution
gender_in_network = Counter()
structure_in_network = Counter()
for node in G.nodes():
    gender_in_network[G.nodes[node].get('gender', 'unknown')] += 1
    structure_in_network[G.nodes[node].get('structure', 'unknown')] += 1

# ── Output ──
stats = {
    'nodes': n_nodes,
    'edges': n_edges,
    'total_relations': total_relations,
    'relation_types': dict(type_counts.most_common()),
    'components_total': len(components),
    'giant_component_size': giant_size,
    'shared_given_edges': shared_given_edges,
    'shared_given_edges_pct': round(shared_given_edges / max(n_edges, 1) * 100, 2),
    'different_given_edges': different_given_edges,
    'missing_decomp_edges': missing_decomp_edges,
    'component_classification': {
        'criteria': f'collapsible = given_ratio <= 0.5 AND max_degree <= {MAX_DEGREE_THRESHOLD}; '
                    f'patronymic_artifact = given_ratio <= 0.5 AND max_degree > {MAX_DEGREE_THRESHOLD}',
        'size_1_isolates': comp_counts['size_1'],
        'size_2_pairs': comp_counts['size_2'],
        'size_3plus_total': comp_counts['size_3plus'],
        'collapsible': comp_counts['collapsible'],
        'patronymic_artifact': comp_counts['patronymic_artifact'],
        'no_convergence': comp_counts['no_convergence'],
        'collapsible_size_distribution': dict(collapsible_sizes.most_common(20)),
        'patronymic_artifact_details': sorted(patronymic_artifact_details,
                                               key=lambda x: -x['size']),
    },
    'gender_in_network': dict(gender_in_network.most_common()),
    'structure_types_in_network': dict(structure_in_network.most_common()),
}

with open('rq3_network_stats_v3.json', 'w') as f:
    json.dump(stats, f, indent=2)

# ── Print ──
print(f"\n{'='*60}")
print("NETWORK STATS v3")
print(f"{'='*60}")
print(f"  Nodes: {n_nodes:,}  Edges: {n_edges:,}  Components: {len(components):,}")
print(f"  Giant component: {giant_size:,} nodes")

print(f"\n  Edge-level given-name analysis:")
print(f"    Shared given name:    {shared_given_edges:,} ({shared_given_edges/n_edges*100:.1f}%)")
print(f"    Different given name: {different_given_edges:,} ({different_given_edges/n_edges*100:.1f}%)")

print(f"\n  Component classification (size >= 3, max_degree threshold = {MAX_DEGREE_THRESHOLD}):")
print(f"    Total 3+ components:    {comp_counts['size_3plus']:,}")
print(f"    Collapsible:            {comp_counts['collapsible']:,}")
print(f"    Patronymic artifact:    {comp_counts['patronymic_artifact']:,}")
print(f"    No convergence:         {comp_counts['no_convergence']:,}")

if patronymic_artifact_details:
    print(f"\n  Patronymic artifact components:")
    for pa in sorted(patronymic_artifact_details, key=lambda x: -x['size']):
        print(f"    size={pa['size']:,}  givens={pa['unique_givens']}  "
              f"ratio={pa['given_ratio']:.3f}  max_deg={pa['max_degree']}")

print(f"\n  Gender in network:")
for g, c in gender_in_network.most_common():
    print(f"    {g:18s} {c:8,}")
print(f"\n  Saved: rq3_network_stats_v3.json")
