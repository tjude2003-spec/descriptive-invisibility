"""
Check max-degree distribution of components meeting given-name ratio <= 0.5.
Is there a natural gap around 15, or is the threshold arbitrary?

Requires: relations_all.csv, name_decomposition_v2_full.csv
"""
import csv, time
from collections import Counter

try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'networkx', '--break-system-packages', '-q'])
    import networkx as nx

VALID_TYPES = {'widow', 'spouse', 'child', 'sibling', 'widower'}

print("Loading decomposition...")
decomp = {}
with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        decomp[row['entity'].strip()] = row

print("Building graph...")
G = nx.Graph()
with open('relations_all.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['relation_type'] not in VALID_TYPES:
            continue
        p1 = row['person_1'].strip().lower()
        p2 = row['person_2'].strip().lower()
        for p in (p1, p2):
            if p not in G and p in decomp:
                G.add_node(p, given=decomp[p]['given_name'].strip().lower())
        if G.has_edge(p1, p2):
            G[p1][p2]['weight'] += 1
        else:
            G.add_edge(p1, p2, weight=1)

components = list(nx.connected_components(G))

# Collect max degree for all components with 3+ nodes AND given ratio <= 0.5
max_degrees = []
for comp in components:
    if len(comp) < 3:
        continue
    givens = set()
    max_deg = 0
    for node in comp:
        g = G.nodes[node].get('given', '')
        if g:
            givens.add(g)
        max_deg = max(max_deg, G.degree(node))
    ratio = len(givens) / len(comp)
    if ratio <= 0.5:
        max_degrees.append((max_deg, len(comp)))

max_degrees.sort()

print(f"\n{len(max_degrees)} components with 3+ nodes and given ratio <= 0.5")
print(f"\nMax-degree distribution:")

# Bin by degree ranges
bins = Counter()
for md, sz in max_degrees:
    bins[md] += 1

for deg in sorted(bins):
    bar = '#' * min(bins[deg], 80)
    print(f"  deg {deg:>4d}: {bins[deg]:>4d}  {bar}")

# Show the gap region explicitly
print(f"\n--- Around threshold 15 ---")
for deg in range(10, 25):
    count = bins.get(deg, 0)
    print(f"  deg {deg:>2d}: {count}")

print(f"\nComponents with max_deg > 10:")
for md, sz in max_degrees:
    if md > 10:
        print(f"  max_deg={md}, size={sz}")
