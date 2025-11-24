# General Corpus Prep – Dictionaries, Thesauruses, Encyclopedias

This directory contains a helper script to build the **general** corpus for
`shiftable_project` by automatically downloading and processing:

- **Dictionaries / Thesauruses**
  - *English Synonyms and Antonyms* (Fernald, Project Gutenberg)
  - *Roget's Thesaurus of English Words and Phrases* (Roget, Project Gutenberg)
- **Dictionary-style entries (Wiktionary)**
  - Kaikki.org / Wiktextract JSONL dump (English glosses)
- **Encyclopedia**
  - Simple English Wikipedia pages-articles XML dump

All processed text is written as `.txt` files into:

```text
shiftable_project/data/general/
```


## 1. Location in the Project

Put these files in:

```text
shiftable_project/
  data/
    general/            # <- final corpus .txt files will end up here
    prep/
      general/
        prepare_general_corpus.py
        README_general_corpus.md
        downloads/      # (created automatically; raw downloads)
```


## 2. Dependencies

The script is primarily standard-library based, but uses:

- **Python 3.8+** recommended.
- Optional: **requests** (for nicer HTTP downloads). If `requests` is missing,
  it will fall back to `urllib` from the standard library.

Install `requests` (recommended):

```bash
pip install requests
```

No other third-party packages are required.


## 3. What It Downloads

### 3.1 Project Gutenberg – Dictionaries / Thesauruses

From Project Gutenberg:

1. **English Synonyms and Antonyms** (James C. Fernald) – Gutenberg #28900  
   - Text URL (Plain Text UTF‑8):  
     `https://www.gutenberg.org/ebooks/28900.txt.utf-8`

2. **Roget's Thesaurus of English Words and Phrases** (Peter Mark Roget) – Gutenberg #10681  
   - Text URL (Plain Text UTF‑8):  
     `https://www.gutenberg.org/ebooks/10681.txt.utf-8`

The script:

- Downloads each to `downloads/`.
- Strips Project Gutenberg header/footer when possible.
- Writes cleaned versions to:

  ```text
  shiftable_project/data/general/gutenberg_fernald_synonyms.txt
  shiftable_project/data/general/gutenberg_roget_thesaurus.txt
  ```

These provide classical dictionary & thesaurus style text.


### 3.2 Wiktionary – Kaikki / Wiktextract

From Kaikki.org (“Raw data downloads extracted from Wiktionary”):

- JSONL (gzipped) raw Wiktextract data for the English Wiktionary edition:  
  `https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`

The data is a **JSON Lines** file in which each line is one entry containing
fields like `word`, `lang`, `pos`, `senses`, `examples`, etc.

The script:

- Downloads `raw-wiktextract-data.jsonl.gz` into `downloads/`.
- Streams it line-by-line (without loading everything into memory).
- Keeps only entries where `lang == "English"`.
- For each entry, builds compact lines like:

  ```text
  word (pos): gloss1 [Examples: ...] || gloss2 ...
  ```

- Writes them to:

  ```text
  shiftable_project/data/general/wiktionary_english_glosses.txt
  ```

By default, it processes up to `MAX_WIKTIONARY_LINES` lines for safety (see
configuration section below).


### 3.3 Simple English Wikipedia – Encyclopedia

From Wikimedia dumps for **Simple English Wikipedia**:

- Pages-articles dump (XML, bz2-compressed):  
  `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

The script:

- Downloads the dump into `downloads/`.
- Streams it with `xml.etree.ElementTree.iterparse`, page by page.
- For each `<page>`:
  - Extracts `<title>` and `<revision>/<text>` (raw wikitext).
- Writes them out into:

  ```text
  shiftable_project/data/general/simple_wikipedia_articles.txt
  ```

The output format is simple:

```text
### TITLE ###
Some page title
### TEXT ###
Raw wiki markup text...

### TITLE ###
Next page title
### TEXT ###
More wiki markup...
```

By default, it processes up to `MAX_WIKIPEDIA_PAGES` pages for safety (configurable).


## 4. Configuration

At the top of `prepare_general_corpus.py`, you’ll find:

```python
# Approximate limits to avoid exploding disk/memory use.
# Set to None to process entire datasets.
MAX_WIKTIONARY_LINES: Optional[int] = 1_000_000    # lines from Kaikki JSONL
MAX_WIKIPEDIA_PAGES: Optional[int] = 50_000        # pages from Simple English Wikipedia
```

You can edit these constants to:

- Increase the coverage (set to `None` for full dataset).
- Decrease them for faster, smaller test runs.

**Note:**

- The Kaikki JSONL is very large (tens of GB uncompressed). Streaming helps,
  but even so, processing *all* of it can be slow.
- The Simple English Wikipedia dump is much smaller than full enwiki but still
  large; 50,000 pages is usually plenty for a solid PoC.


## 5. Running the Script

From your project root (`shiftable_project/`), run:

```bash
cd shiftable_project/data/prep/general
python prepare_general_corpus.py
```

You should see log output like:

```text
=== prepare_general_corpus.py ===
Base directory    : /path/to/shiftable_project
Data directory    : /path/to/shiftable_project/data
General corpus dir: /path/to/shiftable_project/data/general
Prep directory    : /path/to/shiftable_project/data/prep/general

[download] Downloading from: https://www.gutenberg.org/ebooks/28900.txt.utf-8
[download] Saving to: /.../downloads/fernald_english_synonyms.txt
...
[gutenberg] Cleaning Fernald text -> /.../data/general/gutenberg_fernald_synonyms.txt
...
[wiktionary] Extracting English glosses -> /.../data/general/wiktionary_english_glosses.txt
...
[simplewiki] Extracting pages from Simple English Wikipedia -> /.../data/general/simple_wikipedia_articles.txt
...
All steps completed (with any above errors noted).
```

After it completes (or partially completes), check:

```text
shiftable_project/data/general/
  gutenberg_fernald_synonyms.txt
  gutenberg_roget_thesaurus.txt
  wiktionary_english_glosses.txt
  simple_wikipedia_articles.txt
```

These files will then be picked up by your generalist training pipeline
(`train_generalist.py` or the SMA API initialization logic).


## 6. Licensing Notes (High-Level)

This script pulls from several external sources. For **internal research and PoC**
use, you are generally fine, but for **redistribution** or **commercial** use,
please review and comply with each source’s license:

- **Project Gutenberg texts** – public domain in the USA, but Project Gutenberg
  has its own terms of use for distribution.
- **Kaikki / Wiktextract data** – based on Wiktionary; subject to corresponding
  licenses (CC BY-SA, GFDL). See the Kaikki site and Wiktextract docs.
- **Simple English Wikipedia dump** – Wikipedia content is CC BY-SA and GFDL;
  re-use requires attribution and share-alike.

For this PoC, the script simply automates “getting the text”. You should add
appropriate attribution and license notes in your top-level project README if
you intend to share models or corpora built from this data.


## 7. Integration with SMA / Generalist Training

Once the script has generated the `.txt` files in:

```text
shiftable_project/data/general/
```

your SMA project’s initialization flow will:

1. Use **all `.txt` files in `data/general/`** to build the tokenizer.
2. Train the **generalist** language model on this corpus.
3. Save artifacts to `shiftable_project/outputs/general/`.

From there, your existing `sma_poc` API can initialize and train
`ShiftableTransformerLM` with specialist heads on top of this richer
generalist corpus.
