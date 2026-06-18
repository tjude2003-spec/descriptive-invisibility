#!/usr/bin/env python3
"""
Verify that manifest.json accurately reflects what's in htr_cache/.

Tailored to the actual manifest schema:
    {
      "<inv>": {
        "inventory_number": "...",
        "doc_id": int,
        "page_count": int,          # expected total from Transkribus
        "pages_fetched": int,       # downloaded fresh this run
        "pages_cached": int,        # already on disk, kept
        "pages_failed": int,
        "failed_pages": [int, ...],
        "completed": bool,
        "timestamp": "..."
      }
    }

Checks:
  1. Every 'completed' entry should have pages_fetched+pages_cached
     files actually present on disk.
  2. Inventory dirs on disk NOT in the manifest (collector crashed
     before writing final state).
  3. Not-completed entries: usually one failed page (near-complete,
     still usable) vs severe failure (skip).
  4. page_count == 0 outliers (empty Transkribus source docs).

Outputs a JSON report for cross-machine merging.

Usage:
    python verify_manifest.py \
        --manifest ./htr_cache/manifest.json \
        --cache-dir ./htr_cache \
        --output verify_report_windows.json \
        --label windows
"""

import argparse
import json
import sys
from pathlib import Path


# Usable threshold: an inventory is considered usable for NER if at least
# this fraction of expected pages are present as non-empty files on disk.
# Handles two failure modes in one rule:
#   - manifest-recorded failed pages (typically 1 per inventory)
#   - zero-byte files from interrupted writes (several per inventory)
# 0.98 = at most 2% of pages can be missing/empty.
USABLE_MIN_FRACTION = 0.98


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print(f"[ERROR] unexpected manifest shape: {type(raw)}", file=sys.stderr)
        sys.exit(1)
    return raw


def scan_cache_dir(cache_dir: Path) -> dict:
    """Return {inv_number: {total_files, non_empty_count}}."""
    if not cache_dir.exists():
        print(f"[ERROR] cache dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    on_disk = {}
    for sub in sorted(cache_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name in {"__pycache__", "logs"}:
            continue

        txt_files = list(sub.glob("*.txt"))
        non_empty = [f for f in txt_files if f.stat().st_size > 0]

        on_disk[sub.name] = {
            "total_files": len(txt_files),
            "non_empty_count": len(non_empty),
        }

    return on_disk


def classify_entry(entry: dict, disk_info) -> dict:
    """
    status is one of:
      ok                  — completed, disk count == expected
      near_complete       — within NEAR_COMPLETE_TOLERANCE pages
      partial             — substantially incomplete
      disk_missing        — manifest has it, directory doesn't exist
      disk_empty          — directory exists but no non-empty files
      empty_transkribus   — page_count == 0 from the source
      page_count_mismatch — disk count disagrees with manifest
    """
    inv = entry["inventory_number"]
    expected = entry.get("page_count", 0)
    manifest_got = entry.get("pages_fetched", 0) + entry.get("pages_cached", 0)
    completed = entry.get("completed", False)
    failed_count = entry.get("pages_failed", 0)

    base = {
        "inv": inv,
        "expected": expected,
        "manifest_got": manifest_got,
        "completed": completed,
        "failed_count": failed_count,
    }

    if disk_info is None:
        return {**base, "status": "disk_missing", "disk_got": 0}

    disk_got = disk_info["non_empty_count"]
    base["disk_got"] = disk_got

    if expected == 0:
        return {**base, "status": "empty_transkribus"}

    if disk_got == 0:
        return {**base, "status": "disk_empty"}

    fraction = disk_got / expected if expected > 0 else 0.0
    base["fraction_present"] = round(fraction, 4)

    if completed and disk_got == expected:
        return {**base, "status": "ok"}

    if fraction >= USABLE_MIN_FRACTION:
        return {**base, "status": "near_complete",
                "missing_pages": entry.get("failed_pages", [])}

    if disk_got != manifest_got:
        # Below usable fraction AND manifest overclaims: genuinely broken
        return {**base, "status": "page_count_mismatch",
                "missing_pages": entry.get("failed_pages", [])}

    return {**base, "status": "partial",
            "missing_pages": entry.get("failed_pages", [])}


def reconcile(manifest: dict, on_disk: dict) -> dict:
    manifest_invs = set(manifest.keys())
    disk_invs = set(on_disk.keys())

    classified = [
        classify_entry(entry, on_disk.get(inv))
        for inv, entry in manifest.items()
    ]

    unlogged_on_disk = [
        {"inv": inv, "non_empty_pages": on_disk[inv]["non_empty_count"]}
        for inv in sorted(disk_invs - manifest_invs)
    ]

    buckets = {}
    for rec in classified:
        buckets.setdefault(rec["status"], []).append(rec)

    # Usable = ok + near_complete. NER can tolerate a missing page.
    usable_invs = sorted(
        [r["inv"] for r in buckets.get("ok", [])] +
        [r["inv"] for r in buckets.get("near_complete", [])]
    )

    return {
        "counts": {
            "manifest_total": len(manifest_invs),
            "disk_total": len(disk_invs),
            "ok": len(buckets.get("ok", [])),
            "near_complete": len(buckets.get("near_complete", [])),
            "partial": len(buckets.get("partial", [])),
            "disk_missing": len(buckets.get("disk_missing", [])),
            "disk_empty": len(buckets.get("disk_empty", [])),
            "page_count_mismatch": len(buckets.get("page_count_mismatch", [])),
            "empty_transkribus": len(buckets.get("empty_transkribus", [])),
            "unlogged_on_disk": len(unlogged_on_disk),
            "usable_for_ner": len(usable_invs),
        },
        "usable_invs": usable_invs,
        "by_status": buckets,
        "unlogged_on_disk": unlogged_on_disk,
    }


def print_report(report: dict, machine_label: str):
    c = report["counts"]
    print(f"\n=== Manifest verification: {machine_label} ===")
    print(f"  manifest entries:       {c['manifest_total']}")
    print(f"  disk directories:       {c['disk_total']}")
    print(f"  ----")
    print(f"  ok (completed, match):           {c['ok']}")
    print(f"  near-complete (>={USABLE_MIN_FRACTION*100:.0f}% pages present): {c['near_complete']}")
    print(f"  ----")
    print(f"  USABLE FOR NER:                  {c['usable_for_ner']}")
    print(f"  ----")
    print(f"  partial (skip):                  {c['partial']}")
    print(f"  disk missing:                    {c['disk_missing']}")
    print(f"  disk empty:                      {c['disk_empty']}")
    print(f"  page count mismatch:             {c['page_count_mismatch']}")
    print(f"  empty transkribus src:           {c['empty_transkribus']}")
    print(f"  on disk but unlogged:            {c['unlogged_on_disk']}")

    for cat in ("partial", "disk_missing", "disk_empty", "page_count_mismatch"):
        items = report["by_status"].get(cat, [])
        if items:
            sample = [r["inv"] for r in items[:5]]
            print(f"\n  [!] first 5 {cat}: {sample}")

    if report["unlogged_on_disk"]:
        sample = [u["inv"] for u in report["unlogged_on_disk"][:5]]
        print(f"\n  [!] first 5 unlogged-on-disk: {sample}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default=None,
                   help="e.g. 'mac' or 'windows'")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    label = args.label or output_path.stem

    manifest = load_manifest(manifest_path)
    on_disk = scan_cache_dir(cache_dir)
    report = reconcile(manifest, on_disk)
    report["machine_label"] = label
    report["manifest_path"] = str(manifest_path.resolve())
    report["cache_dir"] = str(cache_dir.resolve())

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report, label)
    print(f"\n  full report: {output_path}")


if __name__ == "__main__":
    main()
