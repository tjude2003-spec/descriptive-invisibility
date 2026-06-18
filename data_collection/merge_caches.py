#!/usr/bin/env python3
"""
Merge the nested htr_cache (Mac) into the top-level htr_cache (Windows).

Current layout assumed:
    htr_cache/
    ├── _manifest.json              (Windows manifest)
    ├── 10023/                      (Windows inventories)
    ├── ... 1,920 Windows invs ...
    └── htr_cache/                  (nested Mac cache)
        ├── _manifest.json          (Mac manifest)
        ├── 10023/                  (overlap: same inv on both)
        ├── 10024/
        └── ... 549 Mac invs ...

Goal: flatten so htr_cache/ contains all inventories from both machines.
For overlapping inventories, keep the copy with more non-empty pages.

Strategy:
  1. For each inventory in the nested folder:
     - If NOT in top-level: move it up
     - If in top-level: compare non-empty page counts, keep the winner
  2. Preserve both manifests separately (_manifest_windows.json,
     _manifest_mac.json) rather than trying to merge them — they're needed
     for provenance in the thesis.
  3. Delete the now-empty nested htr_cache folder.

DRY RUN by default. Pass --execute to actually move files.

Usage:
    # Preview what would happen:
    python3 merge_caches.py --root "/Volumes/External HD/htr_cache"

    # Actually do it:
    python3 merge_caches.py --root "/Volumes/External HD/htr_cache" --execute
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def count_non_empty_pages(inv_dir: Path) -> int:
    """Count .txt files with size > 0 in an inventory directory."""
    if not inv_dir.exists():
        return 0
    return sum(1 for f in inv_dir.glob("*.txt") if f.stat().st_size > 0)


def count_total_files(inv_dir: Path) -> int:
    """Count all .txt files (including zero-byte) in an inventory directory."""
    if not inv_dir.exists():
        return 0
    return sum(1 for f in inv_dir.glob("*.txt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Path to the top-level htr_cache directory")
    p.add_argument("--execute", action="store_true",
                   help="Actually move files (default: dry run)")
    args = p.parse_args()

    root = Path(args.root)
    nested = root / "htr_cache"

    if not root.exists():
        print(f"[ERROR] root does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    if not nested.exists():
        print(f"[ERROR] nested folder does not exist: {nested}", file=sys.stderr)
        sys.exit(1)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"=== Merge caches — {mode} ===")
    print(f"Root: {root}")
    print(f"Nested: {nested}\n")

    # List inventory directories in both places
    top_invs = {d.name for d in root.iterdir()
                if d.is_dir() and d.name != "htr_cache"}
    nested_invs = {d.name for d in nested.iterdir() if d.is_dir()}

    only_in_nested = sorted(nested_invs - top_invs)
    only_in_top = sorted(top_invs - nested_invs)
    in_both = sorted(top_invs & nested_invs)

    print(f"inventories only in top-level (Windows-only): {len(only_in_top)}")
    print(f"inventories only in nested (Mac-only):         {len(only_in_nested)}")
    print(f"inventories in BOTH:                            {len(in_both)}")
    print()

    # --- Step 1: Move Mac-only inventories up to top-level ---
    print("--- Step 1: moving Mac-only inventories up ---")
    for inv in only_in_nested:
        src = nested / inv
        dst = root / inv
        if args.execute:
            shutil.move(str(src), str(dst))
        if len(only_in_nested) <= 20 or only_in_nested.index(inv) < 5:
            print(f"  move: {src.name}  →  {dst.parent.name}/{dst.name}")
    if len(only_in_nested) > 20:
        print(f"  ... and {len(only_in_nested) - 5} more")
    print()

    # --- Step 2: Resolve duplicates ---
    print(f"--- Step 2: resolving {len(in_both)} duplicates ---")
    keep_top = []
    keep_nested = []
    tied = []

    for inv in in_both:
        top_dir = root / inv
        nested_dir = nested / inv
        top_count = count_non_empty_pages(top_dir)
        nested_count = count_non_empty_pages(nested_dir)

        if top_count > nested_count:
            keep_top.append((inv, top_count, nested_count))
            if args.execute:
                shutil.rmtree(nested_dir)
        elif nested_count > top_count:
            keep_nested.append((inv, top_count, nested_count))
            if args.execute:
                shutil.rmtree(top_dir)
                shutil.move(str(nested_dir), str(top_dir))
        else:
            tied.append((inv, top_count))
            # Tie: keep top (Windows), delete nested (Mac)
            # They have the same page count; arbitrary choice is fine
            if args.execute:
                shutil.rmtree(nested_dir)

    print(f"  top (Windows) had more pages:   {len(keep_top)}")
    print(f"  nested (Mac) had more pages:    {len(keep_nested)}")
    print(f"  tied (kept top by convention):  {len(tied)}")

    # Show a few examples of where they differed
    if keep_top:
        print(f"\n  examples where Windows won (first 5):")
        for inv, t, n in keep_top[:5]:
            print(f"    {inv}: top={t}, nested={n}, kept top")
    if keep_nested:
        print(f"\n  examples where Mac won (first 5):")
        for inv, t, n in keep_nested[:5]:
            print(f"    {inv}: top={t}, nested={n}, kept nested")

    # Flag big disagreements — these warrant a manual look
    big_diffs = [
        (inv, t, n) for inv, t, n in keep_top + keep_nested
        if abs(t - n) > 5
    ]
    if big_diffs:
        print(f"\n  [!] {len(big_diffs)} duplicates differed by more than 5 pages:")
        for inv, t, n in big_diffs[:20]:
            print(f"    {inv}: top={t}, nested={n}, diff={abs(t-n)}")
        if len(big_diffs) > 20:
            print(f"    ... and {len(big_diffs) - 20} more")

    # --- Step 3: Rename and preserve manifests ---
    print("\n--- Step 3: preserving both manifests ---")
    windows_manifest = root / "_manifest.json"
    mac_manifest = nested / "_manifest.json"

    if windows_manifest.exists():
        target = root / "_manifest_windows.json"
        print(f"  {windows_manifest.name}  →  {target.name}")
        if args.execute:
            shutil.move(str(windows_manifest), str(target))

    if mac_manifest.exists():
        target = root / "_manifest_mac.json"
        print(f"  (nested) {mac_manifest.name}  →  {target.name}")
        if args.execute:
            shutil.move(str(mac_manifest), str(target))

    # --- Step 4: Remove empty nested dir ---
    print("\n--- Step 4: cleaning up ---")
    if args.execute:
        # At this point the nested folder should be empty
        remaining = list(nested.iterdir())
        if remaining:
            print(f"  [!] nested folder not empty, has: {[p.name for p in remaining[:10]]}")
            print("  NOT removing it — investigate manually")
        else:
            nested.rmdir()
            print(f"  removed empty {nested}")
    else:
        print(f"  would remove {nested} (if empty)")

    print(f"\n=== {mode} complete ===")
    if not args.execute:
        print("Re-run with --execute to actually perform the merge.")


if __name__ == "__main__":
    main()
