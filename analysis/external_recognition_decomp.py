#!/usr/bin/env python3
"""
External-recognition coverage test — decomposed + surface, side by side.

Three measurements on ONE seed-42 sorted draw:

  A. SAA decomposed-to-decomposed   (PRIMARY, strongest)
       extracted given_name vs SAA givenName  (fuzz.ratio >= 0.80)
       AND extracted family_name vs SAA baseSurname (fuzz.ratio >= 0.65)
       Same thresholds the orthographic-variation analysis already uses,
       so the chapter has ONE matching standard.
       NOTE: entities with no family_name component (bare patronymic
       forms) CANNOT match a family-bearing record by construction.
       That is the structural-invisibility thesis at the matching
       layer, NOT a measured exclusion rate. Reported separately.

  B. SAA surface-fuzzy   (for comparison only — shows how much the
       matching METHOD drove the 1.21% surface number)

  C. Ecartico surface-fuzzy   (corroborating referent; SUBSET file =
       lower bound; collision-prone, hence kept surface only)

Run:  python external_recognition_decomp.py
"""
import csv, random, math
from collections import defaultdict
from rapidfuzz import fuzz, process

DECOMP   = "name_decomposition_v2_full.csv"
ECARTICO = "ecartico_persons.csv"
SAA      = "export-records-2026-04-15T06-32-57-485564001.csv"
SEED, N  = 42, 10000
G_THR, F_THR = 0.85, 0.85          # threshold for external recognition, selected after manual inspection of results at 0.65 and 0.80 for all rq2 analyses. 

# ---------- extracted population: keep decomposed parts ----------
rows = []
with open(DECOMP, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        e = r["entity"].strip()
        if not e:
            continue
        rows.append((e,
                     (r.get("given_name") or "").strip().lower(),
                     (r.get("family_name") or "").strip().lower()))
def stratum(gn, fn, has_pat, has_pre):
    if fn and has_pre: return "prefix+family"
    if fn:             return "family_only"
    if has_pat:        return "patronymic_only"
    return "given_only_or_other"

# need patronymic/prefix flags for strata — reread minimally
flags = {}
with open(DECOMP, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        e=r["entity"].strip()
        if e:
            flags[e]=(bool((r.get("patronymic") or "").strip()),
                      bool((r.get("prefix") or "").strip()))

ent_index = {e:(gn,fn) for e,gn,fn in rows}
entities = sorted(ent_index)
random.seed(SEED)
sample = random.sample(entities, min(N, len(entities)))
print(f"population {len(entities):,} | sample {len(sample):,} (seed {SEED}, sorted)")

# ---------- SAA: keep decomposed parts AND a surface form ----------
saa_pairs = []          # (givenName_lc, baseSurname_lc)
saa_surface = set()
with open(SAA, encoding="utf-8") as f:
    rd = csv.DictReader(f, delimiter=";")
    gk="Person.pnv:hasName.pnv:givenName"
    pk="Person.pnv:hasName.pnv:surnamePrefix"
    sk="Person.pnv:hasName.pnv:baseSurname"
    ak="Person.schema:alternateName"
    for r in rd:
        g=(r.get(gk) or "").strip().lower()
        p=(r.get(pk) or "").strip().lower()
        s=(r.get(sk) or "").strip().lower()
        if s:                        # need a surname to decomposed-match
            saa_pairs.append((g, s))
        surf=" ".join(x for x in [g,p,s] if x).strip()
        if surf: saa_surface.add(surf)
        alt=(r.get(ak) or "").strip().lower()
        if alt: saa_surface.add(alt)
# index SAA surnames -> set of given names, for fast family-first matching
saa_surnames = sorted({s for _,s in saa_pairs})
saa_by_surname = defaultdict(list)
for g,s in saa_pairs:
    saa_by_surname[s].append(g)
saa_surface = sorted(saa_surface)
print(f"SAA: {len(saa_pairs):,} (given,surname) pairs | "
      f"{len(saa_surface):,} surface forms")

eca = sorted({(r.get('name') or '').strip().lower()
              for r in csv.DictReader(open(ECARTICO, encoding='utf-8'))
              if (r.get('name') or '').strip()})
print(f"Ecartico: {len(eca):,} names "
      f"({'FULL' if len(eca)>40000 else 'SUBSET = LOWER BOUND'})")

def wilson(k,n):
    if n==0: return (0.0,0.0)
    z=1.96;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

def report(label, matched_flags, note=""):
    n=len(sample)
    k=sum(matched_flags.values())
    lo,hi=wilson(k,n)
    print(f"\n===== {label} {note} =====")
    print(f"  overall: {k} / {n} = {100*k/n:.2f}%  CI[{100*lo:.2f},{100*hi:.2f}]")
    strat=defaultdict(lambda:{"n":0,"m":0})
    for e in sample:
        gn,fn=ent_index[e]; hp,hx=flags.get(e,(False,False))
        s=stratum(gn,fn,hp,hx)
        strat[s]["n"]+=1
        if matched_flags[e]: strat[s]["m"]+=1
    for s in ["prefix+family","family_only","patronymic_only","given_only_or_other"]:
        d=strat[s]
        if d["n"]==0: print(f"    {s:22s} n=0"); continue
        lo,hi=wilson(d["m"],d["n"])
        print(f"    {s:22s}{d['n']:6d}{d['m']:6d}{100*d['m']/d['n']:7.2f}%"
              f"  CI[{100*lo:.2f},{100*hi:.2f}]")

# ---- A. SAA decomposed: family-first then given-name check ----
mA={}
nofam=0
for e in sample:
    gn,fn=ent_index[e]
    if not fn:                       # no family component -> cannot match by construction
        mA[e]=False; nofam+=1; continue
    hit=False
    # find SAA surnames within F_THR of this family name
    fr=process.extract(fn, saa_surnames, scorer=fuzz.ratio,
                        score_cutoff=F_THR*100, limit=None)
    for sname,_,_ in fr:
        for sg in saa_by_surname[sname]:
            if gn and sg and fuzz.ratio(gn,sg)/100.0 >= G_THR:
                hit=True; break
        if hit: break
    mA[e]=hit
    # Save SAA matches to CSV for verification
import csv as csv_mod
saa_matches = []
for e in sample:
    if mA[e]:
        gn, fn = ent_index[e]
        # re-find the match to get the SAA name
        fr = process.extract(fn, saa_surnames, scorer=fuzz.ratio,
                             score_cutoff=F_THR*100, limit=None)
        for sname, fs, _ in fr:
            for sg in saa_by_surname[sname]:
                gs = fuzz.ratio(gn, sg) / 100.0 if gn and sg else 0
                if gs >= G_THR:
                    saa_matches.append({
                        'ner_entity': e,
                        'ner_given': gn, 'ner_family': fn,
                        'saa_given': sg, 'saa_surname': sname,
                        'given_score': round(gs * 100, 1),
                        'family_score': round(fs, 1),
                    })
                    break
            else:
                continue
            break

with open('saa_matches_for_verification.csv', 'w', newline='') as f:
    w = csv_mod.DictWriter(f, fieldnames=[
        'ner_entity', 'ner_given', 'ner_family',
        'saa_given', 'saa_surname', 'given_score', 'family_score',
        'genuine', 'notes'])
    w.writeheader()
    for m in saa_matches:
        m['genuine'] = ''
        m['notes'] = ''
        w.writerow(m)
print(f"Saved {len(saa_matches)} SAA matches to saa_matches_for_verification.csv")
report("A. SAA DECOMPOSED (given>=0.85 AND family>=0.85)", mA,
       "[PRIMARY]")
print(f"    note: {nofam} sampled entities have NO family component and")
print(f"          cannot match by construction (structural, not measured).")

# ---- B. SAA surface fuzzy (comparison) ----
mB={e:(process.extractOne(e.lower(), saa_surface,
        scorer=fuzz.token_sort_ratio, score_cutoff=85) is not None)
    for e in sample}
report("B. SAA SURFACE FUZZY (token_sort_ratio>=0.85)", mB,
       "[comparison only]")

# ---- C. Ecartico surface fuzzy (corroborating) ----
mC={e:(process.extractOne(e.lower(), eca,
        scorer=fuzz.token_sort_ratio, score_cutoff=85) is not None)
    for e in sample}
report("C. ECARTICO SURFACE FUZZY (token_sort_ratio>=0.85)", mC,
       "[corroborating; subset=lower bound]")

print("\nINTERPRETATION")
print("  A vs B: gap = how much the surface method inflated/changed the")
print("          SAA number. A is the defensible primary figure.")
print("  Thin-name strata in A near 0 is EXPECTED and structural -")
print("  report as a property of the matching definition, not a finding.")
