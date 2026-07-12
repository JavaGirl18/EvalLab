#!/usr/bin/env python3
"""
Ingestion script — fetches and extracts article content for dataset items that have
a source_url but empty source_text. Writes results back to the target corpus in place.
Safe to re-run: already-populated items are skipped unless --force is passed.

Fields populated when available:
  source_text            — main article body (always updated on fetch)
  source_title           — page title (only if currently empty or a placeholder)
  metadata.publisher     — site name (only if currently empty)
  metadata.published_date — article date (only if currently empty)
  metadata.retrieved_at  — ISO 8601 UTC timestamp of this fetch (always updated)
  metadata.ingestion_error — error message if fetch failed (cleared on success)

Usage (run from backend/):
    python3 ingest_articles.py                                   # pilot_dataset.json
    python3 ingest_articles.py --corpus benchmark/benchmark_corpus.json
    python3 ingest_articles.py --force      # re-fetch items that already have source_text
    python3 ingest_articles.py --dry-run    # print what would be fetched without writing

The /research-eval endpoint is not involved here. It only reads static source_text.
"""

from __future__ import annotations

import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import trafilatura

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS = "pilot_dataset.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def fetch_and_extract(url: str) -> dict:
    """
    Fetches the page at url and extracts article content using trafilatura.
    Returns a dict with keys: text, title, publisher, published_date.
    Any field trafilatura could not determine will be None.
    Raises RuntimeError if the page could not be fetched or text is empty.
    """
    log.info("  Fetching %s", url)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError("fetch failed — no content returned (blocked, timeout, or bad URL)")

    result = trafilatura.bare_extraction(
        downloaded,
        include_comments=False,
        include_tables=False,
        with_metadata=True,
    )

    # trafilatura 2.0+ returns a Document object, not a dict
    def _get(key: str) -> str | None:
        val = getattr(result, key, None) if result else None
        return val.strip() if isinstance(val, str) and val.strip() else None

    text = _get("text")
    if not text:
        raise RuntimeError("extraction returned empty text — page may be JS-rendered or paywalled")

    return {
        "text":           text,
        "title":          _get("title"),
        "publisher":      _get("sitename"),
        "published_date": _get("date"),
    }


def _is_placeholder(value: str) -> bool:
    return not value or value.strip().startswith("[Placeholder")


def ingest(force: bool = False, dry_run: bool = False, corpus: str = DEFAULT_CORPUS) -> None:
    dataset_path = DATA_DIR / corpus
    if not dataset_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    changed = 0
    skipped = 0
    failed = 0

    for item in dataset:
        item_id  = item["id"]
        url      = item.get("source_url", "").strip()
        existing = item.get("source_text", "").strip()

        if not url:
            log.info("SKIP  %s — no source_url", item_id)
            skipped += 1
            continue

        if existing and not force:
            log.info("SKIP  %s — source_text already populated (use --force to re-fetch)", item_id)
            skipped += 1
            continue

        if dry_run:
            log.info("DRY   %s — would fetch %s", item_id, url)
            continue

        try:
            extracted = fetch_and_extract(url)
            meta = item.setdefault("metadata", {})

            # source_text — always update on a successful fetch
            item["source_text"] = extracted["text"]
            log.info("OK    %s — extracted %d words", item_id, len(extracted["text"].split()))

            # source_title — only update if currently empty or a placeholder
            if extracted["title"] and _is_placeholder(item.get("source_title", "")):
                item["source_title"] = extracted["title"]
                log.info("      title: %s", extracted["title"])

            # metadata.publisher — only update if currently empty
            if extracted["publisher"] and not meta.get("publisher"):
                meta["publisher"] = extracted["publisher"]
                log.info("      publisher: %s", extracted["publisher"])

            # metadata.published_date — only update if currently empty
            if extracted["published_date"] and not meta.get("published_date"):
                meta["published_date"] = extracted["published_date"]
                log.info("      published_date: %s", extracted["published_date"])

            meta["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            meta.pop("ingestion_error", None)
            changed += 1

        except Exception as exc:
            log.warning("FAIL  %s — %s", item_id, exc)
            item.setdefault("metadata", {})
            item["metadata"]["ingestion_error"] = str(exc)
            item["metadata"]["retrieved_at"] = ""
            failed += 1

    if not dry_run and changed > 0:
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            f.write("\n")
        log.info("Wrote updated dataset to %s", dataset_path)

    log.info("Done — %d fetched, %d skipped, %d failed", changed, skipped, failed)
    if failed:
        log.warning("%d item(s) failed. Check metadata.ingestion_error in the dataset file.", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus",  default=DEFAULT_CORPUS,
                        help=f"Path relative to data/ (default: {DEFAULT_CORPUS})")
    parser.add_argument("--force",   action="store_true", help="Re-fetch items that already have source_text")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched without writing")
    args = parser.parse_args()

    ingest(force=args.force, dry_run=args.dry_run, corpus=args.corpus)
