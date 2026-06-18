#!/usr/bin/env python3
"""
Fetch HTR Corpus from Transkribus (Parallel Version)
=====================================================
Same as fetch_htr_corpus.py but fetches pages in parallel within each
inventory. Uses the same cache directory and manifest format — fully
compatible, picks up where the sequential version left off.

USAGE:
    # Fetch evaluation corpus with 4 parallel workers (default)
    python3 fetch_htr_corpus_fast.py --corpus eval

    # Faster: 8 workers, minimal sleep
    python3 fetch_htr_corpus_fast.py --corpus eval --workers 8 --sleep 0.05

    # Resume after interruption (same command)
    python3 fetch_htr_corpus_fast.py --corpus eval

    # Dry run
    python3 fetch_htr_corpus_fast.py --corpus eval --dry-run

Requires: Python 3.9+, no external dependencies.
"""

import json
import re
import ssl
import sys
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL context
# ---------------------------------------------------------------------------
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRANSKRIBUS_API = "https://api-read.transkribus.eu/documents"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(url: str, accept: str = "application/json", retries: int = MAX_RETRIES):
    """GET with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": accept})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt == retries:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(wait)


def get_page_count(doc_id: int) -> int:
    raw = api_get(f"{TRANSKRIBUS_API}/{doc_id}")
    return json.loads(raw).get("pageCount", 0)


def fetch_single_page(doc_id: int, page_nr: int, out_path: Path, sleep: float) -> dict:
    """
    Fetch one page. Returns a result dict.
    Designed to be called from a thread pool.
    """
    result = {"page_nr": page_nr, "status": "skipped"}

    # Skip if already cached
    if out_path.exists():
        result["status"] = "cached"
        return result

    # Also check legacy naming
    legacy = out_path.parent / f"page_{page_nr}.txt"
    if legacy.exists():
        result["status"] = "cached"
        return result

    try:
        # Step 1: get content URL
        raw = api_get(f"{TRANSKRIBUS_API}/{doc_id}/pages/{page_nr}")
        page_meta = json.loads(raw)
        content_url = page_meta.get("content")
        if not content_url:
            result["status"] = "failed"
            result["error"] = "no content URL"
            return result

        if sleep > 0:
            time.sleep(sleep)

        # Step 2: fetch PAGE XML
        xml_bytes = api_get(content_url, accept="application/xml")
        xml_text = xml_bytes.decode("utf-8")

        # Step 3: extract text
        lines = re.findall(r"<Unicode>(.*?)</Unicode>", xml_text, re.DOTALL)
        text = "\n".join(lines)

        out_path.write_text(text, encoding="utf-8")
        result["status"] = "fetched"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Inventory-level parallel fetch
# ---------------------------------------------------------------------------
def fetch_inventory_parallel(
    inv_nr: str, doc_id: int, cache_dir: Path,
    workers: int, sleep: float
) -> dict:
    """Fetch all pages for one inventory using parallel workers."""
    inv_dir = cache_dir / inv_nr
    inv_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "inventory_number": inv_nr,
        "doc_id": doc_id,
        "page_count": 0,
        "pages_fetched": 0,
        "pages_cached": 0,
        "pages_failed": 0,
        "failed_pages": [],
        "completed": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        n_pages = get_page_count(doc_id)
    except Exception as e:
        log.error(f"inv {inv_nr} (doc {doc_id}): failed to get page count — {e}")
        status["error"] = str(e)
        return status

    status["page_count"] = n_pages

    if n_pages == 0:
        log.warning(f"inv {inv_nr}: 0 pages reported")
        status["completed"] = True
        return status

    # Check how many are already cached before spawning threads
    pages_to_fetch = []
    for pg in range(1, n_pages + 1):
        page_path = inv_dir / f"{pg}.txt"
        legacy_path = inv_dir / f"page_{pg}.txt"
        if page_path.exists() or legacy_path.exists():
            status["pages_cached"] += 1
        else:
            pages_to_fetch.append((pg, page_path))

    if not pages_to_fetch:
        status["completed"] = True
        return status

    # Parallel fetch of uncached pages
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_single_page, doc_id, pg, path, sleep
            ): pg
            for pg, path in pages_to_fetch
        }

        done_count = 0
        for future in as_completed(futures):
            pg = futures[future]
            done_count += 1
            try:
                result = future.result()
                if result["status"] == "fetched":
                    status["pages_fetched"] += 1
                elif result["status"] == "cached":
                    status["pages_cached"] += 1
                elif result["status"] == "failed":
                    status["pages_failed"] += 1
                    status["failed_pages"].append(pg)
                    log.warning(f"  inv {inv_nr} page {pg}: {result.get('error', '?')}")
            except Exception as e:
                status["pages_failed"] += 1
                status["failed_pages"].append(pg)
                log.warning(f"  inv {inv_nr} page {pg}: {e}")

            # Progress every 100 pages
            if done_count % 100 == 0:
                log.info(f"  {done_count}/{len(pages_to_fetch)} pages done")

    total_ok = status["pages_fetched"] + status["pages_cached"]
    status["completed"] = (total_ok == n_pages)
    return status


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Corpus resolution
# ---------------------------------------------------------------------------
def resolve_inventory_list(corpus, alignment_path, transkribus_index, explicit_invs=None):
    with open(alignment_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    if explicit_invs:
        inv_list = explicit_invs
    elif corpus == "eval":
        inv_list = [pair[0] for pair in alignment["overlap"]]
    elif corpus == "extension":
        inv_list = alignment["htr_only"]
    elif corpus == "all":
        inv_list = [pair[0] for pair in alignment["overlap"]] + alignment["htr_only"]
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    pairs = []
    missing = []
    for inv in inv_list:
        if inv in transkribus_index:
            pairs.append((inv, transkribus_index[inv]["docId"]))
        else:
            missing.append(inv)

    if missing:
        log.warning(f"{len(missing)} inventories not in transkribus_index")

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Fetch HTR corpus from Transkribus (parallel version)",
    )
    p.add_argument("--corpus", choices=["eval", "extension", "all"], default="eval")
    p.add_argument("--invs", nargs="+", default=None)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--cache-dir", default="./htr_cache")
    p.add_argument("--workers", type=int, default=4,
                    help="Parallel page fetches per inventory (default: 4)")
    p.add_argument("--sleep", type=float, default=0.05,
                    help="Sleep between API calls per thread (default: 0.05s)")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ti_path = data_dir / "transkribus_index.json"
    if not ti_path.exists():
        log.error(f"transkribus_index.json not found at {ti_path}")
        sys.exit(1)

    with open(ti_path, "r", encoding="utf-8") as f:
        transkribus_index = json.load(f)

    alignment_path = data_dir / "alignment.json"
    if not alignment_path.exists():
        log.error(f"alignment.json not found at {alignment_path}")
        sys.exit(1)

    pairs = resolve_inventory_list(
        args.corpus, alignment_path, transkribus_index, args.invs
    )

    log.info(f"Corpus: {args.corpus if not args.invs else 'manual'}")
    log.info(f"Inventories: {len(pairs)} | Workers: {args.workers} | Sleep: {args.sleep}s")
    log.info(f"Cache: {cache_dir.resolve()}")

    if args.dry_run:
        print(f"\n{'inv':>8}  {'docId':>10}  cached?")
        print("-" * 35)
        for inv, doc_id in pairs:
            inv_dir = cache_dir / inv
            has_cache = inv_dir.exists() and any(inv_dir.glob("*.txt"))
            print(f"{inv:>8}  {doc_id:>10}  {'YES' if has_cache else 'no'}")
        n_cached = sum(
            1 for inv, _ in pairs
            if (cache_dir / inv).exists() and any((cache_dir / inv).glob("*.txt"))
        )
        print(f"\n{n_cached}/{len(pairs)} already have cached pages.")
        return

    manifest_path = cache_dir / "_manifest.json"
    manifest = load_manifest(manifest_path)

    already_done = {
        inv for inv, status in manifest.items()
        if status.get("completed") and status.get("pages_failed", 0) == 0
    }
    to_fetch = [(inv, doc_id) for inv, doc_id in pairs if inv not in already_done]

    log.info(f"Already completed: {len(already_done)} | To fetch: {len(to_fetch)}")

    if not to_fetch:
        log.info("Nothing to fetch — all inventories already completed.")
        return

    t_start = time.time()
    total_pages_so_far = 0

    for i, (inv, doc_id) in enumerate(to_fetch, 1):
        elapsed = time.time() - t_start
        if total_pages_so_far > 0:
            pages_per_sec = total_pages_so_far / elapsed
            # Estimate remaining pages (use running average)
            avg_pages = total_pages_so_far / max(i - 1, 1)
            remaining_pages = avg_pages * (len(to_fetch) - i + 1)
            remaining_min = remaining_pages / max(pages_per_sec, 0.01) / 60
        else:
            remaining_min = 0

        log.info(
            f"[{i}/{len(to_fetch)}] inv {inv} (doc {doc_id}) | "
            f"~{remaining_min:.0f}min remaining"
        )

        status = fetch_inventory_parallel(
            inv, doc_id, cache_dir, args.workers, args.sleep
        )
        manifest[inv] = status

        total_pages_so_far += status.get("pages_fetched", 0) + status.get("pages_cached", 0)

        total_pages = status["pages_fetched"] + status["pages_cached"]
        log.info(
            f"  → {total_pages}/{status['page_count']} pages "
            f"({status['pages_fetched']} new, {status['pages_cached']} cached, "
            f"{status['pages_failed']} failed)"
        )

        save_manifest(manifest_path, manifest)

    # Final summary
    elapsed = time.time() - t_start
    total_fetched = sum(s.get("pages_fetched", 0) for s in manifest.values())
    total_cached = sum(s.get("pages_cached", 0) for s in manifest.values())
    total_failed = sum(s.get("pages_failed", 0) for s in manifest.values())
    n_complete = sum(1 for s in manifest.values() if s.get("completed"))

    print()
    print("=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    print(f"Inventories completed:   {n_complete}/{len(manifest)}")
    print(f"Pages fetched (new):     {total_fetched}")
    print(f"Pages already cached:    {total_cached}")
    print(f"Pages failed:            {total_failed}")
    print(f"Time elapsed:            {elapsed/60:.1f} minutes")
    print(f"Throughput:              {(total_fetched)/(elapsed/60):.0f} pages/min")
    print(f"Manifest:                {manifest_path.resolve()}")

    if total_failed > 0:
        print(f"\nFAILED PAGES (re-run to retry):")
        for inv, s in manifest.items():
            if s.get("failed_pages"):
                print(f"  inv {inv}: pages {s['failed_pages']}")


if __name__ == "__main__":
    main()
