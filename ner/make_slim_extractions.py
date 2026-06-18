#!/usr/bin/env python3
"""
Create ner_extractions_slim.csv from ner_extractions.csv.

Drops offset columns (start_char, end_char, entity_label) that are only
needed by build_relations_fast.py. The slim version is consumed by
rq2_analysis.py, decompose_names_v2.py, match_ecartico.py,
match_ecartico_decomposed.py, and rq3_clustering_v2.py.

Usage:
    python make_slim_extractions.py [/path/to/ner_extractions.csv]

Writes ner_extractions_slim.csv alongside the input file.
"""
import csv, sys, os

input_path = sys.argv[1] if len(sys.argv) > 1 else "ner_extractions.csv"
output_path = os.path.join(os.path.dirname(input_path), "ner_extractions_slim.csv")

KEEP = {"inventory_number", "page_number", "entity_text"}

with open(input_path, encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    keep_cols = [c for c in reader.fieldnames if c in KEEP]
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=keep_cols)
        writer.writeheader()
        n = 0
        for row in reader:
            writer.writerow({c: row[c] for c in keep_cols})
            n += 1

print(f"Wrote {n:,} rows to {output_path}")
print(f"  Kept columns: {keep_cols}")
