"""
Thesis pipeline configuration — COPY THIS FILE to config.py and edit the paths.

    cp config.example.py config.py

All scripts that need absolute paths import from config.py.
"""
from pathlib import Path

# ── Base directories ─────────────────────────────────────────────────
# DATA_DIR should contain: ner_extractions.csv, ner_extractions_slim.csv,
#   relations_all.csv, ner_split.json, name_decomposition_v2_full.csv,
#   rq2_frequency_distribution.csv, name_decomposition_v2_stats.json,
#   rq3_case_philip_zweerts_v2.json, rq3_case_jan_verleij_v2.json,
#   final_velehanden_deeds.csv (or velehanden_deeds_part*.csv),
#   final_velehanden_inventory_summary_.csv, final_saa_finding_aids.csv,
#   alignment.json, transkribus_index.json, ecartico_persons.csv,
#   _manifest_merged.json, htr_inventory_notary_mapping.csv,
#   export-records-*.csv (SAA authority export)
DATA_DIR = Path("/path/to/your/thesis_data")

# HTR_CACHE contains per-inventory subdirectories of .txt page files
HTR_CACHE = DATA_DIR / "htr_cache"

# MODEL_DIR contains the adapted spaCy NER model (model-best/)
MODEL_DIR = Path("/path/to/your/ner_model/model-best")
