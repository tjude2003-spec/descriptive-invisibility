"""
Decompose NER entity strings into given_name | patronymic | prefix | family_name.
v2: improved parse order, stricter patronymic detection, HTR cleanup.

Improvements over v1:
  - Scan for patronymic BEFORE prefix (fixes "garrit adamsz de reijger")
  - Ambiguous suffixes (-sen, -ssen) require stem to match a known given name
  - Given name vocabulary built from corpus (first tokens of multi-token entities)
  - HTR artifact cleanup (strip punctuation, newlines, digits)
  - Handles full structure: given + patronymic + prefix + family

Usage:
    python decompose_names_v2.py /path/to/ner_extractions_slim.csv

Output: name_decomposition_v2_stats.json, name_decomposition_v2_sample.csv
"""
import csv, json, re, sys, time, random
from collections import Counter, defaultdict

random.seed(42)

# ── Configuration ──

PREFIXES = {
    'van', 'de', 'der', 'den', 'het', 'ter', 'ten',
    "'t", 'op', 'in', 'tot', 'uit', 'uijt',
}
PREFIX_PAIRS = {
    ('van', 'de'), ('van', 'der'), ('van', 'den'), ('van', 'het'),
    ('in', 'de'), ('in', "'t"), ('op', 'de'), ('op', 'den'),
    ('uit', 'de'), ('uijt', 'de'),
}

# Reliable patronymic suffixes — unambiguously encode filiation
RELIABLE_PAT_SUFFIXES = (
    'sz', 'szen', 'szoon', 'zoon', 'szn',
    'sdr', 'sdr', 'dochter', 'dgtr', 'dogter',
)

# Ambiguous suffixes — could be patronymic OR hereditary surname
# Only treated as patronymic if stem matches a known given name
AMBIGUOUS_PAT_SUFFIXES = (
    'sen', 'ssen', 'se', 'sse',
)

# Female patronymic suffixes (subset of reliable)
FEMALE_PAT_MARKERS = ('dochter', 'dgtr', 'dogter', 'sdr', 'sdr', 'dr')


def clean_entity(s):
    """Clean HTR artifacts from entity string."""
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'^[^a-zA-Z]+', '', s)      # strip leading non-alpha
    s = re.sub(r'[^a-zA-Z.]+$', '', s)     # strip trailing non-alpha (keep trailing .)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def build_given_name_vocab(ner_path, min_freq=20):
    """
    Build a set of known Dutch given names from the corpus itself.
    Uses the first token of multi-token entities, filtered by frequency.
    """
    first_token_counts = Counter()
    total = 0
    with open(ner_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            total += 1
            ent = row['entity_text'].strip().lower()
            tokens = ent.split()
            if len(tokens) >= 2:
                first = tokens[0].strip('.,;:()[]')
                if first.isalpha() and len(first) >= 2:
                    first_token_counts[first] += 1
            if total % 2000000 == 0:
                print(f"  ...{total:,} rows (vocab)")

    # Keep names appearing as first token at least min_freq times
    vocab = {name for name, count in first_token_counts.items() if count >= min_freq}
    print(f"  Given name vocabulary: {len(vocab)} names (freq >= {min_freq})")
    return vocab


def extract_patronymic_stem(token):
    """
    If token ends with a patronymic suffix, return (stem, suffix, reliability).
    stem = the given name embedded in the patronymic.
    """
    t = token.lower().rstrip('.')

    # Check reliable suffixes first (longest match)
    for suf in sorted(RELIABLE_PAT_SUFFIXES, key=len, reverse=True):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)], suf, 'reliable'

    # Check ambiguous suffixes
    for suf in sorted(AMBIGUOUS_PAT_SUFFIXES, key=len, reverse=True):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)], suf, 'ambiguous'

    return None, None, None


def is_patronymic(token, given_names):
    """Check if token is a patronymic. Ambiguous suffixes require stem in given_names."""
    stem, suf, reliability = extract_patronymic_stem(token)
    if stem is None:
        return False
    if reliability == 'reliable':
        return True
    if reliability == 'ambiguous':
        # Check if stem looks like a known given name
        # Also check common stem variants (e.g., "jans" -> "jan", "pieters" -> "pieter")
        candidates = {stem, stem.rstrip('s'), stem + 's'}
        return bool(candidates & given_names)
    return False


def patronymic_gender(token):
    """Extract gender signal from patronymic suffix."""
    t = token.lower().rstrip('.')
    for suf in FEMALE_PAT_MARKERS:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return 'female'
    # If it has a reliable male suffix
    for suf in ('sz', 'szen', 'szoon', 'zoon', 'szn'):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return 'male'
    # Ambiguous -sen/-ssen that passed the vocab check — likely male
    for suf in ('sen', 'ssen'):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return 'male_ambiguous'
    return 'unknown'


def find_prefix_position(tokens, first_allowed=0):
    """Find the position of a prefix particle in the token list. Returns (start, end) or None.
    first_allowed controls where to start looking — caller ensures prefix isn't at an
    invalid position (e.g. first_allowed=1 to prevent prefix as first token of the name)."""
    for i in range(first_allowed, len(tokens)):
        tl = tokens[i].lower().rstrip('.')
        # Check two-token prefix first
        if i + 1 < len(tokens):
            pair = (tl, tokens[i+1].lower().rstrip('.'))
            if pair in PREFIX_PAIRS:
                return i, i + 2
        # Single-token prefix
        if tl in PREFIXES:
            return i, i + 1
    return None


def decompose(name_string, given_names):
    """
    Decompose a name string into structured components.
    Parse order: clean → find patronymic → find prefix → assign.
    """
    cleaned = clean_entity(name_string)
    if not cleaned:
        return None

    tokens = cleaned.split()
    if not tokens:
        return None

    result = {
        'given_name': '',
        'patronymic': '',
        'prefix': '',
        'family_name': '',
        'structure_type': 'unknown',
        'gender_marker': 'unknown',
        'pat_reliability': '',
    }

    # ── Single token ──
    if len(tokens) == 1:
        if is_patronymic(tokens[0], given_names):
            result['patronymic'] = tokens[0]
            result['structure_type'] = 'patronymic_only'
            result['gender_marker'] = patronymic_gender(tokens[0])
            stem, suf, rel = extract_patronymic_stem(tokens[0])
            result['pat_reliability'] = rel
        else:
            result['given_name'] = tokens[0]
            result['structure_type'] = 'given_only'
        return result

    # ── Multi-token: find patronymic first ──
    pat_idx = None
    for i in range(1, len(tokens)):  # patronymic never first token
        if is_patronymic(tokens[i], given_names):
            pat_idx = i
            break

    if pat_idx is not None:
        result['given_name'] = ' '.join(tokens[:pat_idx])
        result['patronymic'] = tokens[pat_idx]
        result['gender_marker'] = patronymic_gender(tokens[pat_idx])
        stem, suf, rel = extract_patronymic_stem(tokens[pat_idx])
        result['pat_reliability'] = rel

        remaining = tokens[pat_idx + 1:]

        if not remaining:
            result['structure_type'] = 'given_patronymic'
        else:
            # Check if remaining starts with a prefix
            pfx = find_prefix_position(remaining, first_allowed=0)
            if pfx is not None and pfx[0] == 0:
                result['prefix'] = ' '.join(remaining[pfx[0]:pfx[1]])
                after_pfx = remaining[pfx[1]:]
                if after_pfx:
                    result['family_name'] = ' '.join(after_pfx)
                    result['structure_type'] = 'given_pat_prefix_family'
                else:
                    result['structure_type'] = 'given_pat_prefix'
            else:
                result['family_name'] = ' '.join(remaining)
                result['structure_type'] = 'given_patronymic_family'
        return result

    # ── No patronymic: find prefix ──
    pfx = find_prefix_position(tokens, first_allowed=1)  # prefix never first token

    if pfx is not None:
        result['given_name'] = ' '.join(tokens[:pfx[0]])
        result['prefix'] = ' '.join(tokens[pfx[0]:pfx[1]])
        after_pfx = tokens[pfx[1]:]
        if after_pfx:
            result['family_name'] = ' '.join(after_pfx)
            result['structure_type'] = 'given_prefix_family'
        else:
            result['structure_type'] = 'given_prefix'
        return result

    # ── No patronymic, no prefix: given + family ──
    result['given_name'] = tokens[0]
    result['family_name'] = ' '.join(tokens[1:])
    result['structure_type'] = 'given_family'
    return result


# ── Main ──

ner_path = sys.argv[1] if len(sys.argv) > 1 else 'ner_extractions_slim.csv'

print("STEP 1: Building given name vocabulary from corpus...")
t0 = time.time()
given_names = build_given_name_vocab(ner_path, min_freq=20)

# Add some known Dutch given names the corpus might miss at freq threshold
SUPPLEMENTS = {
    'jan', 'pieter', 'cornelis', 'jacob', 'hendrik', 'willem', 'gerrit',
    'claes', 'dirck', 'abraham', 'isaac', 'daniel', 'johannes', 'michiel',
    'anna', 'maria', 'elisabeth', 'catharina', 'geertruij', 'jannetje',
    'marritje', 'aaltje', 'grietje', 'trijntje', 'annetje', 'lijsbeth',
    'adriaen', 'harmen', 'albert', 'andries', 'barent', 'bartholome',
    'christiaan', 'david', 'evert', 'frans', 'govert', 'hans',
    'joost', 'laurens', 'lourens', 'lucas', 'maarten', 'marten',
    'nicolaes', 'paulus', 'robbert', 'simon', 'thomas', 'wouter',
}
given_names |= SUPPLEMENTS
print(f"  Final vocabulary: {len(given_names)} names ({time.time()-t0:.1f}s)")

print(f"\nSTEP 2: Loading unique entities...")
t1 = time.time()
entities = set()
total = 0
with open(ner_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        total += 1
        ent = row['entity_text'].strip()
        if ent:
            entities.add(ent.lower())
        if total % 2000000 == 0:
            print(f"  ...{total:,} rows")
print(f"  {len(entities):,} unique entities ({time.time()-t1:.1f}s)")

print(f"\nSTEP 3: Decomposing...")
t2 = time.time()

type_counts = Counter()
gender_counts = Counter()
reliability_counts = Counter()
component_flags = Counter()
token_counts = Counter()
examples = defaultdict(list)

results = []
for ent in entities:
    d = decompose(ent, given_names)
    if d is None:
        continue
    results.append((ent, d))
    type_counts[d['structure_type']] += 1
    if d['gender_marker'] != 'unknown':
        gender_counts[d['gender_marker']] += 1
    if d['pat_reliability']:
        reliability_counts[d['pat_reliability']] += 1
    if d['given_name']: component_flags['has_given'] += 1
    if d['patronymic']: component_flags['has_patronymic'] += 1
    if d['prefix']: component_flags['has_prefix'] += 1
    if d['family_name']: component_flags['has_family'] += 1
    token_counts[len(ent.split())] += 1

    if len(examples[d['structure_type']]) < 10:
        examples[d['structure_type']].append({
            'original': ent,
            'given': d['given_name'],
            'patronymic': d['patronymic'],
            'prefix': d['prefix'],
            'family': d['family_name'],
            'gender': d['gender_marker'],
            'reliability': d['pat_reliability'],
        })

n = len(results)
print(f"  Done: {n:,} entities ({time.time()-t2:.1f}s)")

# ── Report ──
print(f"\n{'='*60}")
print(f"NAME DECOMPOSITION v2 — RESULTS")
print(f"{'='*60}")
print(f"  Total: {n:,}")

print(f"\n  Structure types:")
for t, c in type_counts.most_common():
    print(f"    {t:30s} {c:8,} ({c/n*100:5.1f}%)")

print(f"\n  Component coverage:")
for comp in ['has_given', 'has_patronymic', 'has_prefix', 'has_family']:
    c = component_flags[comp]
    print(f"    {comp:20s} {c:8,} ({c/n*100:.1f}%)")

print(f"\n  Patronymic reliability:")
for r, c in reliability_counts.most_common():
    print(f"    {r:15s} {c:8,}")

print(f"\n  Gender from patronymic:")
for g, c in gender_counts.most_common():
    print(f"    {g:18s} {c:8,} ({c/n*100:.1f}%)")

print(f"\n  Token counts:")
for tc in sorted(token_counts):
    if token_counts[tc] > 100:
        print(f"    {tc} tokens: {token_counts[tc]:8,} ({token_counts[tc]/n*100:.1f}%)")

print(f"\n  Examples:")
for stype, _ in type_counts.most_common():
    print(f"\n    [{stype}]")
    for ex in examples[stype][:5]:
        parts = []
        if ex['given']: parts.append(f"giv={ex['given']}")
        if ex['patronymic']: parts.append(f"pat={ex['patronymic']}{'*' if ex['reliability']=='ambiguous' else ''}")
        if ex['prefix']: parts.append(f"pfx={ex['prefix']}")
        if ex['family']: parts.append(f"fam={ex['family']}")
        if ex['gender'] != 'unknown': parts.append(f"gen={ex['gender']}")
        print(f"      {ex['original']:40s} → {', '.join(parts)}")

# ── Save ──
stats = {
    'total_entities': n,
    'structure_types': {t: {'count': c, 'pct': round(c/n*100, 2)} for t, c in type_counts.most_common()},
    'components': {comp: {'count': component_flags[comp], 'pct': round(component_flags[comp]/n*100, 2)}
                   for comp in ['has_given', 'has_patronymic', 'has_prefix', 'has_family']},
    'patronymic_reliability': dict(reliability_counts),
    'gender_from_patronymic': dict(gender_counts),
    'token_distribution': {str(k): v for k, v in sorted(token_counts.items())},
    'given_name_vocab_size': len(given_names),
    'examples': {t: examples[t] for t in type_counts},
}

with open('name_decomposition_v2_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Sample CSV for manual validation
sample = random.sample(results, min(1000, len(results)))
with open('name_decomposition_v2_sample.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['original', 'structure_type', 'given_name', 'patronymic', 'prefix',
                'family_name', 'gender_marker', 'pat_reliability'])
    for ent, d in sample:
        w.writerow([ent, d['structure_type'], d['given_name'], d['patronymic'],
                    d['prefix'], d['family_name'], d['gender_marker'], d['pat_reliability']])

# Full decomposed corpus for downstream use
print(f"\n  Writing full decomposed corpus...")
t3 = time.time()
with open('name_decomposition_v2_full.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['entity', 'structure_type', 'given_name', 'patronymic', 'prefix',
                'family_name', 'gender_marker', 'pat_reliability'])
    for ent, d in results:
        w.writerow([ent, d['structure_type'], d['given_name'], d['patronymic'],
                    d['prefix'], d['family_name'], d['gender_marker'], d['pat_reliability']])
print(f"  Done ({time.time()-t3:.1f}s)")

print(f"\n  Saved: name_decomposition_v2_stats.json")
print(f"         name_decomposition_v2_sample.csv (1000 random for validation)")
print(f"         name_decomposition_v2_full.csv (all {n:,} entities)")
print(f"\n  Total runtime: {time.time()-t0:.0f}s")
