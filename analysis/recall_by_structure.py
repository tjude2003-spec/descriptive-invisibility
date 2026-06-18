"""
NER Recall Stratified by Name Structure
========================================
Decomposes VeleHanden ground-truth names in the test set, then measures
recall separately for each structure type. Shows whether the NER model's
regime disfavors the same structural properties that make names
unrecognizable to external authority layers.

Requires:
    - ner_split.json (test inventory list)
    - VeleHanden CSVs (final_velehanden_deeds.csv)
    - NER extraction output (ner_extractions.csv — adapted model)
    
Usage:
    python recall_by_structure.py

Adjust paths below as needed.
"""
import csv, json, os, time, sys
from collections import defaultdict, Counter
from rapidfuzz import fuzz, process as rfprocess
import numpy as np

# ── PATHS ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR
SPLIT_JSON = str(DATA_DIR / "ner_split.json")
VH_CSVS = [
    str(DATA_DIR / "final_velehanden_deeds.csv"),
]
NER_CSV = str(DATA_DIR / "ner_extractions.csv")
THRESHOLD = 0.85

# ── Decomposition (same logic as decompose_names_v2.py, simplified for VH names) ──
PREFIXES = {'van','de','der','den','het','ter','ten',"'t",'op','in','tot','uit','uijt'}
PREFIX_PAIRS = {('van','de'),('van','der'),('van','den'),('van','het'),
                ('in','de'),('in',"'t"),('op','de'),('op','den'),('uit','de'),('uijt','de')}
RELIABLE_PAT = ('sz','szen','szoon','zoon','szn','sdr','dochter','dgtr','dogter')
AMBIGUOUS_PAT = ('sen','ssen','se','sse')

# Build a minimal given-name vocab from VH names themselves
# (populated in main after loading VH data)
given_name_vocab = set()

def extract_pat_stem(token):
    t = token.lower().rstrip('.')
    for suf in sorted(RELIABLE_PAT, key=len, reverse=True):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)], suf, 'reliable'
    for suf in sorted(AMBIGUOUS_PAT, key=len, reverse=True):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)], suf, 'ambiguous'
    return None, None, None

def is_patronymic(token):
    stem, suf, rel = extract_pat_stem(token)
    if stem is None: return False
    if rel == 'reliable': return True
    if rel == 'ambiguous':
        candidates = {stem, stem.rstrip('s'), stem + 's'}
        return bool(candidates & given_name_vocab)
    return False

def find_prefix(tokens, start=1):
    for i in range(start, len(tokens)):
        tl = tokens[i].lower().rstrip('.')
        if i + 1 < len(tokens) and (tl, tokens[i+1].lower().rstrip('.')) in PREFIX_PAIRS:
            return i, i + 2
        if tl in PREFIXES:
            return i, i + 1
    return None

def classify_vh_name(name_str):
    """Classify a VeleHanden name into structure type."""
    tokens = name_str.lower().strip().split()
    tokens = [t.rstrip('.') for t in tokens if t.strip()]
    if not tokens:
        return 'unknown'
    if len(tokens) == 1:
        return 'given only'

    # Check for patronymic (not first token)
    has_pat = False
    pat_idx = None
    for i in range(1, len(tokens)):
        if is_patronymic(tokens[i]):
            has_pat = True
            pat_idx = i
            break

    # Check for prefix
    pfx = find_prefix(tokens, start=1)
    has_prefix = pfx is not None

    # Determine family name presence
    if has_pat:
        remaining = tokens[pat_idx + 1:]
        has_family = len(remaining) > 0
        if has_family:
            # Check if remaining starts with prefix
            pfx2 = find_prefix(remaining, start=0)
            if pfx2 is not None and pfx2[0] == 0:
                after = remaining[pfx2[1]:]
                has_family = len(after) > 0
                has_prefix = True
        if has_family:
            return 'patronymic + family'
        else:
            return 'patronymic only'
    elif has_prefix:
        # prefix found; family = tokens after prefix
        after = tokens[pfx[1]:]
        if after:
            return 'prefix + family'
        else:
            return 'given only'  # prefix without family name is edge case
    else:
        if len(tokens) >= 2:
            return 'family only'
        return 'given only'


def main():
    global given_name_vocab

    # ── Load test split ──
    print("Loading test split...")
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_invs = set(str(i) for i in split['test'])
    print(f"  {len(test_invs)} test inventories")

    # ── Load VH names for test inventories ──
    print("Loading VeleHanden names...")
    vh_by_inv = defaultdict(list)
    for csv_path in VH_CSVS:
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                inv = row['inventory_number'].strip()
                if inv not in test_invs:
                    continue
                if row['person_names']:
                    for name in row['person_names'].split('|'):
                        name = name.strip()
                        if name:
                            vh_by_inv[inv].append(name)
    
    # Build given-name vocab from VH names
    first_counts = Counter()
    for inv, names in vh_by_inv.items():
        for n in names:
            tokens = n.strip().split()
            if len(tokens) >= 2:
                first = tokens[0].lower().strip('.,;:()')
                if first.isalpha() and len(first) >= 2:
                    first_counts[first] += 1
    given_name_vocab = {n for n, c in first_counts.items() if c >= 5}
    # Add common Dutch names
    given_name_vocab |= {
        'jan','pieter','cornelis','jacob','hendrik','willem','gerrit',
        'claes','dirck','abraham','isaac','daniel','johannes','michiel',
        'anna','maria','elisabeth','catharina','geertruij','jannetje',
        'adriaen','harmen','albert','andries','barent','david','evert',
        'frans','govert','hans','joost','laurens','lucas','maarten',
        'nicolaes','paulus','simon','thomas','wouter',
    }
    print(f"  Given-name vocab: {len(given_name_vocab)} names")

    total_vh = sum(len(names) for names in vh_by_inv.values())
    print(f"  {len(vh_by_inv)} inventories, {total_vh} VH names")

    # ── Load NER predictions for test inventories ──
    print("Loading NER predictions...")
    ner_by_inv = defaultdict(list)
    total_ner = 0
    with open(NER_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inv = row['inventory_number'].strip()
            if inv not in test_invs:
                continue
            label = row.get('entity_label', row.get('entity_type', '')).upper()
            if label in ('PER', 'PERSON'):
                ner_by_inv[inv].append(row['entity_text'].strip())
                total_ner += 1
    print(f"  {len(ner_by_inv)} inventories, {total_ner} NER PER predictions")

    # ── Per-inventory matching with structure tracking ──
    print(f"\nMatching (threshold={THRESHOLD})...")
    t0 = time.time()

    structure_stats = defaultdict(lambda: {'total': 0, 'matched': 0})
    inv_count = 0

    for inv in sorted(vh_by_inv.keys()):
        vh_names = vh_by_inv[inv]
        ner_names = ner_by_inv.get(inv, [])

        # Deduplicate (same as ner_baseline.py)
        vh_dedup_map = {}
        for n in vh_names:
            lo = n.lower()
            if lo not in vh_dedup_map:
                vh_dedup_map[lo] = n
        vh_dedup = list(vh_dedup_map.keys())

        ner_dedup_map = {}
        for n in ner_names:
            lo = n.lower()
            if lo not in ner_dedup_map:
                ner_dedup_map[lo] = n
        ner_dedup = list(ner_dedup_map.keys())

        if not vh_dedup:
            continue

        # Classify each VH name by structure
        vh_structures = {}
        for vh_lo in vh_dedup:
            vh_structures[vh_lo] = classify_vh_name(vh_lo)

        if not ner_dedup:
            # All VH names are unmatched
            for vh_lo in vh_dedup:
                s = vh_structures[vh_lo]
                structure_stats[s]['total'] += 1
            inv_count += 1
            continue

        # Compute similarity matrix
        matrix = rfprocess.cdist(
            ner_dedup, vh_dedup,
            scorer=fuzz.token_sort_ratio,
            workers=-1,
        ) / 100.0

        # Best NER match per VH name (recall direction)
        vh_best_ner_idx = np.argmax(matrix, axis=0)
        vh_best_scores = matrix[vh_best_ner_idx, np.arange(len(vh_dedup))]

        for j, vh_lo in enumerate(vh_dedup):
            s = vh_structures[vh_lo]
            structure_stats[s]['total'] += 1
            if vh_best_scores[j] >= THRESHOLD:
                structure_stats[s]['matched'] += 1

        inv_count += 1
        if inv_count % 10 == 0:
            print(f"  {inv_count}/{len(vh_by_inv)} inventories ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {inv_count} inventories in {elapsed:.0f}s")

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"NER RECALL BY GROUND-TRUTH NAME STRUCTURE")
    print(f"Threshold: {THRESHOLD}, Test inventories: {inv_count}")
    print(f"{'='*70}")
    print(f"{'Structure':30s} {'Total':>8s} {'Matched':>8s} {'Recall':>8s}")
    print('-' * 70)

    ordered = ['patronymic only', 'given only', 'family only',
               'prefix + family', 'patronymic + family']
    results = {'threshold': THRESHOLD, 'n_inventories': inv_count, 'strata': {}}

    grand_total = 0
    grand_matched = 0
    for s in ordered:
        d = structure_stats.get(s, {'total': 0, 'matched': 0})
        total = d['total']
        matched = d['matched']
        recall = matched / total if total > 0 else 0
        grand_total += total
        grand_matched += matched
        print(f"  {s:28s} {total:8d} {matched:8d} {recall:8.3f}")
        results['strata'][s] = {
            'total': total, 'matched': matched, 'recall': round(recall, 4)
        }

    # Any unlisted strata
    for s, d in sorted(structure_stats.items()):
        if s not in ordered:
            total = d['total']
            matched = d['matched']
            recall = matched / total if total > 0 else 0
            grand_total += total
            grand_matched += matched
            print(f"  {s:28s} {total:8d} {matched:8d} {recall:8.3f}")
            results['strata'][s] = {
                'total': total, 'matched': matched, 'recall': round(recall, 4)
            }

    overall = grand_matched / grand_total if grand_total > 0 else 0
    print('-' * 70)
    print(f"  {'OVERALL':28s} {grand_total:8d} {grand_matched:8d} {overall:8.3f}")
    results['overall'] = {'total': grand_total, 'matched': grand_matched,
                          'recall': round(overall, 4)}

    with open('recall_by_structure.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: recall_by_structure.json")


if __name__ == '__main__':
    main()
