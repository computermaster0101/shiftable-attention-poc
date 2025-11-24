#!/usr/bin/env python3
"""
prepare_general_corpus.py

Download and build a "general" text corpus composed of:

- Dictionary / thesaurus sources (Project Gutenberg).
- Wiktionary (Kaikki Wiktextract JSONL).
- Simple English Wikipedia articles (XML dump).

The final cleaned .txt files are written to:

    shiftable_project/data/general/

This script is intended to live in:

    shiftable_project/data/prep/general/prepare_general_corpus.py

It uses only the Python standard library plus an optional `requests` dependency.
If `requests` is not installed, it falls back to urllib.
"""

import bz2
import gzip
import io
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Approximate limits to avoid exploding disk/memory use.
# Set to None to process entire datasets.
MAX_WIKTIONARY_LINES: Optional[int] = 1_000_000    # lines from Kaikki JSONL
MAX_WIKIPEDIA_PAGES: Optional[int] = 50_000        # pages from Simple English Wikipedia

# Source URLs
GUTENBERG_FERNALD_URL = "https://www.gutenberg.org/ebooks/28900.txt.utf-8"
GUTENBERG_ROGET_URL = "https://www.gutenberg.org/ebooks/10681.txt.utf-8"
WIKTEXTRACT_JSONL_GZ_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
SIMPLE_WIKI_PAGES_ARTICLES_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_paths() -> dict:
    """
    Compute important paths relative to this script.

    Returns a dict with:
      - base_dir: shiftable_project root directory
      - data_dir: shiftable_project/data
      - general_dir: shiftable_project/data/general
      - prep_dir: shiftable_project/data/prep/general (this script's directory)
      - download_dir: shiftable_project/data/prep/general/downloads
    """
    script_path = Path(__file__).resolve()
    prep_dir = script_path.parent
    # We expect .../shiftable_project/data/prep/general/prepare_general_corpus.py
    # So base_dir is three levels up from data/
    data_dir = prep_dir.parents[1] / "data"
    base_dir = data_dir.parent
    general_dir = data_dir / "general"
    download_dir = prep_dir / "downloads"

    general_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "general_dir": general_dir,
        "prep_dir": prep_dir,
        "download_dir": download_dir,
    }


def ensure_requests():
    """
    Try to import `requests`. If unavailable, return None and the script will fall
    back to urllib for downloads.
    """
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        return None


def download_file(url: str, dest: Path) -> None:
    """
    Download a file from `url` to `dest`. Uses `requests` if available,
    otherwise falls back to urllib.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[download] Skipping download; file already exists: {dest}")
        return

    print(f"[download] Downloading from: {url}")
    print(f"[download] Saving to: {dest}")

    requests = ensure_requests()
    if requests is not None:
        # Stream download to avoid high memory usage
        with requests.get(url, stream=True) as r:  # type: ignore[attr-defined]
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    else:
        # Fallback to urllib
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)

    print(f"[download] Completed: {dest}")


def clean_gutenberg_text(raw_text: str) -> str:
    """
    Remove the standard Project Gutenberg header/footer when present.
    If markers are not found, returns the original text.
    """
    lines = raw_text.splitlines()
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if line.startswith("*** START OF THE PROJECT GUTENBERG EBOOK"):
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if line.startswith("*** END OF THE PROJECT GUTENBERG EBOOK"):
            end_idx = i
            break

    # If no markers were found, return original text
    if start_idx == 0 and end_idx == len(lines):
        return raw_text

    body = "\n".join(lines[start_idx:end_idx])
    return body.strip() + "\n"


# ---------------------------------------------------------------------------
# Gutenberg sources (Fernald + Roget)
# ---------------------------------------------------------------------------

def build_gutenberg_sources(paths: dict) -> None:
    general_dir: Path = paths["general_dir"]
    download_dir: Path = paths["download_dir"]

    fernald_raw = download_dir / "fernald_english_synonyms.txt"
    roget_raw = download_dir / "roget_thesaurus.txt"

    # Download raw Project Gutenberg text files
    download_file(GUTENBERG_FERNALD_URL, fernald_raw)
    download_file(GUTENBERG_ROGET_URL, roget_raw)

    # Clean and write to general corpus directory
    fernald_out = general_dir / "gutenberg_fernald_synonyms.txt"
    roget_out = general_dir / "gutenberg_roget_thesaurus.txt"

    print(f"[gutenberg] Cleaning Fernald text -> {fernald_out}")
    with open(fernald_raw, "r", encoding="utf-8", errors="ignore") as f_in:
        cleaned = clean_gutenberg_text(f_in.read())
    with open(fernald_out, "w", encoding="utf-8") as f_out:
        f_out.write(cleaned)

    print(f"[gutenberg] Cleaning Roget text -> {roget_out}")
    with open(roget_raw, "r", encoding="utf-8", errors="ignore") as f_in:
        cleaned = clean_gutenberg_text(f_in.read())
    with open(roget_out, "w", encoding="utf-8") as f_out:
        f_out.write(cleaned)

    print("[gutenberg] Done building Gutenberg dictionary/thesaurus sources.")


# ---------------------------------------------------------------------------
# Wiktionary via Kaikki (Wiktextract JSONL)
# ---------------------------------------------------------------------------

def build_wiktionary_source(paths: dict) -> None:
    general_dir: Path = paths["general_dir"]
    download_dir: Path = paths["download_dir"]

    wikt_gz_path = download_dir / "raw-wiktextract-data.jsonl.gz"
    download_file(WIKTEXTRACT_JSONL_GZ_URL, wikt_gz_path)

    out_path = general_dir / "wiktionary_english_glosses.txt"
    if out_path.exists():
        print(f"[wiktionary] Output already exists, will overwrite: {out_path}")

    print(f"[wiktionary] Extracting English glosses -> {out_path}")
    count_lines = 0
    count_written = 0

    with gzip.open(wikt_gz_path, "rt", encoding="utf-8") as f_in, \
            open(out_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            count_lines += 1
            if MAX_WIKTIONARY_LINES is not None and count_lines > MAX_WIKTIONARY_LINES:
                print(f"[wiktionary] Reached MAX_WIKTIONARY_LINES={MAX_WIKTIONARY_LINES}, stopping.")
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # Skip malformed lines
                continue

            # We only keep entries where language is English
            if obj.get("lang") != "English":
                continue

            word = obj.get("word")
            pos = obj.get("pos") or obj.get("pos_category") or ""
            senses = obj.get("senses") or []

            if not word or not senses:
                continue

            # Build a compact textual representation combining glosses and examples
            gloss_texts = []
            for sense in senses:
                gloss = sense.get("gloss")
                if not gloss:
                    continue
                examples = sense.get("examples") or []
                example_texts = []
                for ex in examples:
                    # examples can be string or dict
                    if isinstance(ex, str):
                        example_texts.append(ex)
                    elif isinstance(ex, dict):
                        ex_t = ex.get("text")
                        if ex_t:
                            example_texts.append(ex_t)
                if example_texts:
                    gloss_texts.append(gloss + " Examples: " + " | ".join(example_texts))
                else:
                    gloss_texts.append(gloss)

            if not gloss_texts:
                continue

            header = f"{word} ({pos})".strip()
            text = header + ": " + " || ".join(gloss_texts)
            f_out.write(text.replace("\n", " ") + "\n")
            count_written += 1

            if count_written and count_written % 100000 == 0:
                print(f"[wiktionary] Written {count_written} entries...")

    print(f"[wiktionary] Done. Processed lines: {count_lines}, written entries: {count_written}.")


# ---------------------------------------------------------------------------
# Simple English Wikipedia (pages-articles dump)
# ---------------------------------------------------------------------------

def iter_simplewiki_pages(xml_stream: io.BufferedReader):
    """
    Simple streaming parser over a MediaWiki 'pages-articles' XML dump.

    Yields (title, text) tuples. We do not attempt to clean wiki markup here;
    we just output the raw wikitext.

    This is intentionally minimal to avoid extra dependencies.
    """
    import xml.etree.ElementTree as ET

    # We use iterparse to stream through the XML.
    # Each <page> element contains a <title> and a <revision>/<text>.
    context = ET.iterparse(xml_stream, events=("end",))
    for event, elem in context:
        if elem.tag.endswith("page"):
            title_elem = elem.find("./{*}title")
            rev_elem = elem.find("./{*}revision")
            text_elem = None
            if rev_elem is not None:
                text_elem = rev_elem.find("./{*}text")
            title = title_elem.text if title_elem is not None else ""
            text = text_elem.text if text_elem is not None else ""
            yield title, text if text is not None else ""
            elem.clear()  # free memory


def build_simplewiki_source(paths: dict) -> None:
    general_dir: Path = paths["general_dir"]
    download_dir: Path = paths["download_dir"]

    dump_path = download_dir / "simplewiki-latest-pages-articles.xml.bz2"
    download_file(SIMPLE_WIKI_PAGES_ARTICLES_URL, dump_path)

    out_path = general_dir / "simple_wikipedia_articles.txt"
    if out_path.exists():
        print(f"[simplewiki] Output already exists, will overwrite: {out_path}")

    print(f"[simplewiki] Extracting pages from Simple English Wikipedia -> {out_path}")
    count_pages = 0

    with bz2.open(dump_path, "rb") as f_in, \
            open(out_path, "w", encoding="utf-8") as f_out:

        for title, text in iter_simplewiki_pages(f_in):
            if not title and not text:
                continue

            count_pages += 1
            # Write simple separator-based representation
            f_out.write("### TITLE ###\n")
            f_out.write((title or "").replace("\n", " ") + "\n")
            f_out.write("### TEXT ###\n")
            f_out.write((text or "") + "\n\n")

            if MAX_WIKIPEDIA_PAGES is not None and count_pages >= MAX_WIKIPEDIA_PAGES:
                print(f"[simplewiki] Reached MAX_WIKIPEDIA_PAGES={MAX_WIKIPEDIA_PAGES}, stopping.")
                break

            if count_pages and count_pages % 5000 == 0:
                print(f"[simplewiki] Processed {count_pages} pages...")

    print(f"[simplewiki] Done. Extracted pages: {count_pages}.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    paths = get_paths()

    print("=== prepare_general_corpus.py ===")
    print(f"Base directory    : {paths['base_dir']}")
    print(f"Data directory    : {paths['data_dir']}")
    print(f"General corpus dir: {paths['general_dir']}")
    print(f"Prep directory    : {paths['prep_dir']}")
    print()

    # Brief summary
    summary = textwrap.dedent(
        f"""
        This script will download and build a general corpus from:
          - Project Gutenberg English Synonyms and Antonyms (Fernald)
          - Project Gutenberg Roget's Thesaurus
          - Wiktionary (Kaikki raw Wiktextract JSONL; English entries only)
          - Simple English Wikipedia (pages-articles dump)

        Final cleaned .txt outputs will be placed into:

          {paths['general_dir']}

        Approximate controls:
          MAX_WIKTIONARY_LINES   = {MAX_WIKTIONARY_LINES}
          MAX_WIKIPEDIA_PAGES    = {MAX_WIKIPEDIA_PAGES}

        You can adjust these constants at the top of this script if needed.
        """
    ).strip()
    print(summary)
    print()

    # Run all builders
    try:
        build_gutenberg_sources(paths)
    except Exception as e:
        print(f"[ERROR] While building Gutenberg sources: {e}", file=sys.stderr)

    try:
        build_wiktionary_source(paths)
    except Exception as e:
        print(f"[ERROR] While building Wiktionary source: {e}", file=sys.stderr)

    try:
        build_simplewiki_source(paths)
    except Exception as e:
        print(f"[ERROR] While building Simple English Wikipedia source: {e}", file=sys.stderr)

    print("\nAll steps completed (with any above errors noted).")


if __name__ == "__main__":
    main()
