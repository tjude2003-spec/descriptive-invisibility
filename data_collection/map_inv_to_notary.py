#!/usr/bin/env python3
"""
Map inventory numbers to notaries using SAA finding aid ranges.

Uses narrowest-range heuristic: when an inventory number falls within
multiple notaries' ranges, the narrowest containing range wins.
Validated at 99.4% against VeleHanden ground truth (339 items).

For letter-suffix inventories (e.g., 1970A), uses exact string match.

USAGE:
    python3 map_inv_to_notary.py --from-cache ./htr_cache --saa-csv final_saa_finding_aids.csv
    python3 map_inv_to_notary.py --from-transkribus final_transkribus_index_.csv --saa-csv final_saa_finding_aids.csv
    python3 map_inv_to_notary.py --invs 10021 10022 1970A --saa-csv final_saa_finding_aids.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_saa_entries(saa_csv_path):
    numeric_entries = []
    exact_entries = {}
    with open(saa_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            start_str = row.get("inv_range_start", "").strip()
            end_str = row.get("inv_range_end", "").strip()
            notary_num = row.get("notary_number", "").strip()
            notary_name = row.get("notary_name", "").strip()
            try:
                s = int(start_str)
                e = int(end_str)
                numeric_entries.append((s, e, e - s + 1, notary_num, notary_name))
            except (ValueError, TypeError):
                exact_entries[start_str] = (notary_num, notary_name)
                if end_str and end_str != start_str:
                    exact_entries[end_str] = (notary_num, notary_name)
    numeric_entries.sort(key=lambda x: x[2])
    log.info(f"Loaded {len(numeric_entries)} numeric + {len(exact_entries)} exact-match entries")
    return numeric_entries, exact_entries


def lookup_notary(inv_number, numeric_entries, exact_entries):
    if inv_number in exact_entries:
        return exact_entries[inv_number]
    inv_base = "".join(c for c in inv_number if c.isdigit())
    if not inv_base:
        return None, None
    inv_num = int(inv_base)
    for s, e, span, notary_num, notary_name in numeric_entries:
        if s <= inv_num <= e:
            return notary_num, notary_name
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saa-csv", required=True)
    parser.add_argument("--invs", nargs="+", default=None)
    parser.add_argument("--from-cache", default=None)
    parser.add_argument("--from-transkribus", default=None)
    parser.add_argument("--output", default="./inventory_notary_map.csv")
    args = parser.parse_args()

    numeric_entries, exact_entries = load_saa_entries(args.saa_csv)

    if args.invs:
        inv_list = args.invs
    elif args.from_cache:
        cache_dir = Path(args.from_cache)
        inv_list = sorted([d.name for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])
    elif args.from_transkribus:
        inv_list = []
        with open(args.from_transkribus, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                inv_list.append(row["inventory_number"])
    else:
        log.error("Specify --invs, --from-cache, or --from-transkribus")
        sys.exit(1)

    log.info(f"Inventories to map: {len(inv_list)}")

    output_path = Path(args.output)
    mapped = 0
    unmapped = 0
    unmapped_list = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["inventory_number", "notary_number", "notary_name"])
        writer.writeheader()
        for inv in inv_list:
            notary_num, notary_name = lookup_notary(inv, numeric_entries, exact_entries)
            writer.writerow({"inventory_number": inv, "notary_number": notary_num or "", "notary_name": notary_name or ""})
            if notary_num:
                mapped += 1
            else:
                unmapped += 1
                unmapped_list.append(inv)

    notaries = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["notary_name"]:
                notaries.add(row["notary_name"])

    log.info(f"Mapped: {mapped}/{len(inv_list)} ({mapped/len(inv_list)*100:.1f}%)")
    log.info(f"Unmapped: {unmapped}")
    log.info(f"Unique notaries: {len(notaries)}")
    log.info(f"Output: {output_path}")
    if unmapped_list:
        log.info(f"Unmapped: {unmapped_list[:20]}")


if __name__ == "__main__":
    main()
