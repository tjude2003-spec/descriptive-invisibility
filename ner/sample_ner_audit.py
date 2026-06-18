#!/usr/bin/env python3
"""
Stratified sampler for a manual PRECISION / generalization audit of NER output.

WHAT THIS SUPPORTS:
  - Precision of the adapted model (of what it extracted, how much is a real person).
  - Generalization spot-check on inventories/notaries WITHOUT VeleHanden coverage
    (filter with --restrict-inventories pointing at the extension-corpus list).

WHAT THIS DOES NOT SUPPORT:
  - Recall. You cannot estimate recall from the model's own output.
  - The circularity in 3.9. That requires page-level blind annotation of the
    SOURCE TEXT, not a sample of extractions. Different artifact entirely.

Usage:
  python sample_ner_audit.py ner_extractions.csv -n 400 --seed 42 \
      [--restrict-inventories extension_inventories.txt] \
      [--notary-map inv_to_notary.csv] -o ner_audit_sample.csv
"""
import argparse, csv, math, random, sys
from collections import defaultdict

def ci_halfwidth(n, p=0.5, z=1.96):
    return z * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")

def detect(colnames, candidates):
    low = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("-n", "--n", type=int, default=400, help="target total sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default="ner_audit_sample.csv")
    ap.add_argument("--label-value", default="PER",
                    help="value in the label column to keep (default PER)")
    ap.add_argument("--restrict-inventories", default=None,
                    help="optional file: one inventory number per line; "
                         "sample is drawn only from these (use for the "
                         "no-VeleHanden generalization audit)")
    ap.add_argument("--notary-map", default=None,
                    help="optional CSV with columns inventory_number,notary; "
                         "if given, stratification is by notary")
    args = ap.parse_args()
    random.seed(args.seed)

    with open(args.infile, newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        ent_c = detect(cols, ["entity_text", "entity", "text", "name", "mention", "string"])
        inv_c = detect(cols, ["inventory_number", "inventory", "inv", "inv_number"])
        lab_c = detect(cols, ["entity_label", "label", "ent_label", "type", "tag"])
        pg_c  = detect(cols, ["page", "page_number", "ner_page", "pagenr"])
        if not ent_c or not inv_c:
            sys.exit(f"Could not find entity/inventory columns in: {cols}")

        restrict = None
        if args.restrict_inventories:
            with open(args.restrict_inventories) as rf:
                restrict = {ln.strip() for ln in rf if ln.strip()}

        rows = []
        for r in rdr:
            if lab_c and args.label_value and r.get(lab_c, "").strip().upper() != args.label_value.upper():
                continue
            inv = (r.get(inv_c) or "").strip()
            ent = (r.get(ent_c) or "").strip()
            if not inv or not ent:
                continue
            if restrict is not None and inv not in restrict:
                continue
            rows.append({
                "inventory_number": inv,
                "page": (r.get(pg_c) or "").strip() if pg_c else "",
                "entity": ent,
            })

    if not rows:
        sys.exit("No rows matched the filters. Check --label-value / --restrict-inventories.")

    notary_of = {}
    if args.notary_map:
        with open(args.notary_map, newline="", encoding="utf-8") as nf:
            nr = csv.DictReader(nf)
            nic = detect(nr.fieldnames or [], ["inventory_number", "inventory", "inv"])
            ncc = detect(nr.fieldnames or [], ["notary", "notary_name", "notaris"])
            for r in nr:
                notary_of[(r.get(nic) or "").strip()] = (r.get(ncc) or "").strip()

    # stratum = notary if a map is given, else inventory number
    strata = defaultdict(list)
    for row in rows:
        key = notary_of.get(row["inventory_number"], row["inventory_number"]) \
              if args.notary_map else row["inventory_number"]
        strata[key].append(row)

    # proportional allocation, >=1 per non-empty stratum, capped at stratum size
    total = len(rows)
    target = min(args.n, total)
    alloc = {}
    for k, lst in strata.items():
        alloc[k] = min(len(lst), max(1, round(target * len(lst) / total)))
    # trim/grow to hit target exactly
    keys = sorted(strata, key=lambda k: -len(strata[k]))
    while sum(alloc.values()) > target:
        for k in keys:
            if sum(alloc.values()) <= target: break
            if alloc[k] > 1: alloc[k] -= 1
    while sum(alloc.values()) < target:
        for k in keys:
            if sum(alloc.values()) >= target: break
            if alloc[k] < len(strata[k]): alloc[k] += 1

    sample = []
    for k, lst in strata.items():
        sample.extend(random.sample(lst, alloc[k]))
    random.shuffle(sample)

    with open(args.out, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["id", "inventory_number", "page", "entity",
                    "is_person", "notes"])
        for i, row in enumerate(sample, 1):
            w.writerow([i, row["inventory_number"], row["page"],
                        row["entity"], "", ""])

    n = len(sample)
    print(f"Wrote {n} rows to {args.out}")
    print(f"Strata ({'notary' if args.notary_map else 'inventory'}): {len(strata)} "
          f"covered; pool size {total}")
    print(f"95% CI half-width (worst case p=0.5): +/-{ci_halfwidth(n)*100:.1f}pp")
    print(f"95% CI half-width (at p=0.70):        +/-{ci_halfwidth(n,0.70)*100:.1f}pp")
    print(f"Rule-of-three: if 0 non-persons observed, upper-95% rate <= {3/n*100:.2f}%")
    print(f"Seed {args.seed} -- rerun reproduces this exact sample.")

if __name__ == "__main__":
    main()
