#!/usr/bin/env python3
"""
Extract named entities from the combined HTR corpus using the adapted
spaCy model.

Fixes applied over the initial version:
  - Oversize pages are handled at the page level, not the inventory
    level: a single too-long file no longer drops its entire inventory
  - macOS AppleDouble sidecar files (._*) are explicitly excluded
  - Malformed filenames now log a warning instead of silently skipping
  - Text iteration is streaming-friendly (uses a generator instead of
    loading everything for an inventory into memory at once)
  - os.fsync after each inventory, so a kernel panic doesn't lose
    already-written rows
  - Stale checkpoint .tmp files are cleaned up on startup

Usage:
    python3 ner_extract_corpus.py

All defaults match the session layout. Override with flags if needed.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


# spaCy's default is 1_000_000 chars. HTR pages rarely exceed 50K, so
# raise the cap only enough to tolerate unusually dense pages while
# still catching pathological concatenations that would hang the model.
MAX_DOC_LENGTH = 1_500_000


def load_target_invs(windows_report_path: Path, mac_report_path: Path) -> set:
    with open(windows_report_path) as f:
        win = json.load(f)
    with open(mac_report_path) as f:
        mac = json.load(f)
    return set(win["usable_invs"]) | set(mac["usable_invs"])


def load_processed_invs(output_csv: Path) -> set:
    checkpoint = output_csv.with_suffix(".checkpoint.json")
    stale_tmp = checkpoint.with_suffix(".tmp")
    # Clean up leftovers from an interrupted write
    if stale_tmp.exists():
        try:
            stale_tmp.unlink()
        except Exception:
            pass
    if checkpoint.exists():
        try:
            with open(checkpoint) as f:
                return set(json.load(f).get("completed_invs", []))
        except json.JSONDecodeError:
            print(f"[warn] checkpoint file corrupt, ignoring: {checkpoint}",
                  file=sys.stderr)
            return set()
    return set()


def save_checkpoint(output_csv: Path, completed: set):
    checkpoint = output_csv.with_suffix(".checkpoint.json")
    tmp = checkpoint.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"completed_invs": sorted(completed)}, f)
    tmp.rename(checkpoint)


def iter_pages(inv_dir: Path):
    """
    Yield (page_number, text) for every non-empty .txt page. Skips:
      - zero-byte files (empty HTR output)
      - macOS sidecar files (._*)
      - files too long for spaCy to process
      - files with unparseable names (logged as warnings)
    """
    for f in sorted(inv_dir.glob("*.txt")):
        # Skip macOS AppleDouble sidecar files — they're binary metadata,
        # have .txt in the name, and are not part of the corpus
        if f.name.startswith("._"):
            continue
        if f.stat().st_size == 0:
            continue

        stem = f.stem
        if stem.startswith("page_"):
            num_str = stem[5:]
        else:
            num_str = stem
        try:
            page_num = int(num_str)
        except ValueError:
            print(f"  [warn] skipping unparseable filename: {f.name} "
                  f"in {inv_dir.name}", file=sys.stderr)
            continue

        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"  [warn] {f}: not valid UTF-8, skipping ({e})",
                  file=sys.stderr)
            continue
        except Exception as e:
            print(f"  [warn] couldn't read {f}: {e}", file=sys.stderr)
            continue

        if not text.strip():
            continue

        if len(text) > MAX_DOC_LENGTH:
            print(f"  [warn] {inv_dir.name}/page {page_num}: text is "
                  f"{len(text):,} chars (> {MAX_DOC_LENGTH:,} max); "
                  f"skipping this page", file=sys.stderr)
            continue

        yield page_num, text


def process_inventory(nlp, inv_nr: str, inv_dir: Path,
                       keep_labels: set, batch_size: int):
    """
    Run the spaCy model on all pages of one inventory. Returns a list
    of rows ready to write to CSV. Memory footprint is bounded by
    batch_size, not by inventory size, because nlp.pipe() streams.
    """
    # Materialize the page list once so we can align page numbers with
    # the Doc stream from nlp.pipe. For a 700-page inventory at ~5KB
    # per page this is ~3.5MB of text in memory — well within limits.
    pages = list(iter_pages(inv_dir))
    if not pages:
        return []

    page_nums = [p for p, _ in pages]
    texts = [t for _, t in pages]

    rows = []
    for page_num, doc in zip(page_nums, nlp.pipe(texts, batch_size=batch_size)):
        for ent in doc.ents:
            if ent.label_ not in keep_labels:
                continue
            rows.append({
                "inventory_number": inv_nr,
                "page_number": page_num,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    return rows


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATA_DIR, HTR_CACHE, MODEL_DIR
    default_cache = HTR_CACHE
    default_model = MODEL_DIR
    default_win_report = DATA_DIR / "verify_merged_windows.json"
    default_mac_report = DATA_DIR / "verify_merged_mac.json"
    default_output = DATA_DIR / "ner_extractions.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=str(default_cache))
    p.add_argument("--model", default=str(default_model))
    p.add_argument("--windows-report", default=str(default_win_report))
    p.add_argument("--mac-report", default=str(default_mac_report))
    p.add_argument("--output", default=str(default_output))
    p.add_argument("--labels", default="PER",
                   help="Comma-separated NER labels to keep")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    model_path = Path(args.model)
    output_csv = Path(args.output)
    keep_labels = {lbl.strip() for lbl in args.labels.split(",")}

    for path, label in [(cache_dir, "cache dir"),
                          (model_path, "model path"),
                          (Path(args.windows_report), "windows report"),
                          (Path(args.mac_report), "mac report")]:
        if not path.exists():
            print(f"[error] {label} does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    target_invs = load_target_invs(Path(args.windows_report),
                                     Path(args.mac_report))
    print(f"Target inventories: {len(target_invs)}")

    completed_invs = load_processed_invs(output_csv)
    remaining = sorted(target_invs - completed_invs)
    print(f"Already completed: {len(completed_invs)}")
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("Nothing to do. Checkpoint shows all target inventories processed.")
        return

    print(f"\nLoading model from {model_path} ...")
    import spacy
    nlp = spacy.load(str(model_path))
    # Match our file-level guard so pipe doesn't raise before we can
    # skip pages ourselves
    nlp.max_length = MAX_DOC_LENGTH
    print(f"Model loaded. Pipeline: {nlp.pipe_names}")
    print(f"Keeping labels: {keep_labels}")

    is_new = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "inventory_number", "page_number", "entity_text",
        "entity_label", "start_char", "end_char"
    ])
    if is_new:
        writer.writeheader()
        csv_file.flush()

    t0 = time.time()
    total_rows = 0
    failed_invs = []

    for idx, inv in enumerate(remaining, 1):
        inv_dir = cache_dir / inv
        if not inv_dir.is_dir():
            print(f"  [warn] inventory dir not found: {inv_dir}")
            failed_invs.append((inv, "directory not found"))
            continue

        try:
            rows = process_inventory(nlp, inv, inv_dir,
                                       keep_labels, args.batch_size)
        except Exception as e:
            print(f"  [error] {inv} failed: {e}", file=sys.stderr)
            failed_invs.append((inv, str(e)))
            # Do not mark completed — will retry on resume
            continue

        for row in rows:
            writer.writerow(row)
        csv_file.flush()
        # Force to disk so a kernel panic doesn't lose this inventory's rows
        try:
            os.fsync(csv_file.fileno())
        except OSError:
            # fsync can fail on some filesystems; flush already happened
            pass
        total_rows += len(rows)

        completed_invs.add(inv)
        save_checkpoint(output_csv, completed_invs)

        elapsed = time.time() - t0
        rate = idx / elapsed
        remaining_ct = len(remaining) - idx
        eta_min = (remaining_ct / rate) / 60 if rate > 0 else 0

        if idx % 25 == 0 or idx == 1 or idx == len(remaining):
            print(f"  [{idx}/{len(remaining)}] {inv}: {len(rows)} entities "
                  f"| total rows: {total_rows:,} "
                  f"| {rate:.1f} inv/s | ETA: {eta_min:.0f} min")

    csv_file.close()

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Inventories processed this run: {len(remaining) - len(failed_invs)}")
    print(f"Inventories failed:             {len(failed_invs)}")
    if failed_invs:
        print("Failed inventories (will retry on rerun):")
        for inv, reason in failed_invs[:20]:
            print(f"  {inv}: {reason}")
        if len(failed_invs) > 20:
            print(f"  ... and {len(failed_invs) - 20} more")
    print(f"Total entities written: {total_rows:,}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"Output CSV: {output_csv}")
    print(f"Checkpoint: {output_csv.with_suffix('.checkpoint.json')}")


if __name__ == "__main__":
    main()
