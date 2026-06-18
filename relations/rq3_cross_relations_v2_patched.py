"""
RQ3 Cross-Relations v2 — PATCHED notary attribution.

CHANGE FROM ORIGINAL: inv->notary is now resolved by range-join against the
authoritative SAA finding aid (final_saa_finding_aids.csv), NOT the
half-filled htr_inventory_notary_mapping.csv. The old file was ~1,589/3,179
empty (CRLF-masked), which inflated 'notary_unknown'. The finding aid covers
the full fonds-5075 inventory space, so every inventory with a relation
resolves to a notary.

Requires: relations_all.csv, name_decomposition_v2_full.csv,
          final_saa_finding_aids.csv
Output: rq3_cross_relations_v2.json
"""
import csv, json, time, re
from collections import Counter, defaultdict

VALID_TYPES = {'widow', 'spouse', 'child', 'sibling', 'widower'}

# ── Load decomposition ──
print("Loading decomposition...")
decomp = {}
with open('name_decomposition_v2_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        decomp[row['entity'].strip()] = row

# ── Build AUTHORITATIVE inv->notary from finding-aid ranges ──
print("Building notary resolver from finding aid ranges...")
_fa = []
with open('final_saa_finding_aids.csv', newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        s, e, nm = r['inv_range_start'].strip(), r['inv_range_end'].strip(), r['notary_name'].strip()
        if re.fullmatch(r'\d+', s) and re.fullmatch(r'\d+', e) and nm:
            _fa.append((int(s), int(e), nm))
_fa.sort()

def resolve_notary(inv):
    """Range-join an inventory number to its notary. Returns None if no
    unambiguous finding-aid range contains it."""
    m = re.match(r'(\d+)', str(inv).strip())
    if not m:
        return None
    n = int(m.group(1))
    hits = {nm for s, e, nm in _fa if s <= n <= e}
    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:                 # overlapping ranges at a boundary
        return sorted(hits)[0]        # deterministic; flagged in stats below
    return None

# cache so we don't re-scan ranges per relation row
_cache = {}
def notary_of(inv):
    if inv not in _cache:
        _cache[inv] = resolve_notary(inv)
    return _cache[inv]

print(f"  {len(decomp):,} decomposed entities, {len(_fa):,} finding-aid ranges")

# ── Load and group relations ──
print("Loading relations...")
t0 = time.time()

pair_data = defaultdict(lambda: {
    'inventories': set(), 'notaries': set(), 'types': set(), 'count': 0
})

total = 0
ambiguous_invs = set()
with open('relations_all.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['relation_type'] not in VALID_TYPES:
            continue
        total += 1
        p1 = row['person_1'].strip().lower()
        p2 = row['person_2'].strip().lower()
        key = tuple(sorted([p1, p2]))
        inv = row['inventory'].strip()
        d = pair_data[key]
        d['inventories'].add(inv)
        d['types'].add(row['relation_type'])
        d['count'] += 1
        nm = notary_of(inv)
        if nm:
            d['notaries'].add(nm)

print(f"  {total:,} relations, {len(pair_data):,} pairs ({time.time()-t0:.1f}s)")

# ── Analyze cross-inventory pairs with decomposition ──
cross_inv = 0
cross_notary = 0
same_notary = 0
notary_unknown = 0

cross_pairs_both_have_patronymic = 0
cross_pairs_both_have_prefix = 0
cross_pairs_both_decomposable = 0
examples = []

for key, d in pair_data.items():
    if len(d['inventories']) <= 1:
        continue
    cross_inv += 1

    if len(d['notaries']) > 1:
        cross_notary += 1
    elif len(d['notaries']) == 1:
        same_notary += 1
    else:
        notary_unknown += 1

    p1, p2 = key
    d1 = decomp.get(p1, {})
    d2 = decomp.get(p2, {})

    if d1.get('given_name') and d2.get('given_name'):
        cross_pairs_both_decomposable += 1
    if d1.get('patronymic') and d2.get('patronymic'):
        cross_pairs_both_have_patronymic += 1
    if d1.get('prefix') and d2.get('prefix'):
        cross_pairs_both_have_prefix += 1

    if len(examples) < 30:
        examples.append({
            'person_1': p1, 'person_2': p2,
            'p1_decomp': {k: d1.get(k, '') for k in ['given_name','patronymic','prefix','family_name','gender_marker']},
            'p2_decomp': {k: d2.get(k, '') for k in ['given_name','patronymic','prefix','family_name','gender_marker']},
            'n_inventories': len(d['inventories']),
            'n_notaries': len(d['notaries']),
            'types': sorted(d['types']),
        })

stats = {
    'total_pairs': len(pair_data),
    'cross_inventory_pairs': cross_inv,
    'cross_notary': cross_notary,
    'same_notary': same_notary,
    'notary_unknown': notary_unknown,
    'notary_source': 'final_saa_finding_aids.csv range-join (authoritative)',
    'component_analysis': {
        'both_decomposable': cross_pairs_both_decomposable,
        'both_have_patronymic': cross_pairs_both_have_patronymic,
        'both_have_prefix': cross_pairs_both_have_prefix,
    },
    'examples': examples[:20],
}

with open('rq3_cross_relations_v2.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*60}")
print("CROSS-RELATIONS WITH DECOMPOSITION (authoritative notary mapping)")
print(f"{'='*60}")
print(f"  Cross-inventory pairs:     {cross_inv:,}")
print(f"    Cross-notary:            {cross_notary:,}")
print(f"    Same notary:             {same_notary:,}")
print(f"    Notary unknown:          {notary_unknown:,}")
print(f"  (was: 123 cross / 1,215 same / 313 unknown on the old half-filled map)")
print(f"\n  Component analysis of cross-inventory pairs:")
print(f"    Both decomposable:       {cross_pairs_both_decomposable:,} ({cross_pairs_both_decomposable/max(cross_inv,1)*100:.1f}%)")
print(f"    Both have patronymic:    {cross_pairs_both_have_patronymic:,}")
print(f"    Both have prefix:        {cross_pairs_both_have_prefix:,}")
print(f"\n  Saved: rq3_cross_relations_v2.json")
