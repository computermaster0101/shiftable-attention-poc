# Emergent Expert Corpus Prep

This directory contains a helper script to turn **emergent queries**
(from `emergence_log.jsonl`) into a training corpus for a **new specialist
expert** domain in your SMA project.

The idea is:

1. The SMA API runs and logs queries that appear *outside* the coverage of
   existing specialists to:

   ```text
   shiftable_project/outputs/shiftable/emergence_log.jsonl
   ```

2. You use this script to extract those emergent queries and their model
   completions into a `.txt` corpus for a new expert domain under:

   ```text
   shiftable_project/data/<expert_name>/
   ```

3. You then add that new expert to the SMA model by calling the existing
   `/specialists` API endpoint or using the UI.


## 1. Location in the Project

Place these files under:

```text
shiftable_project/
  data/
    prep/
      emergence/
        prepare_emergent_expert_corpus.py
        README_emergent_expert_corpus.md
```

The script expects that the SMA PoC produces the emergence log at:

```text
shiftable_project/outputs/shiftable/emergence_log.jsonl
```


## 2. What the Script Does

`prepare_emergent_expert_corpus.py`:

- Reads `emergence_log.jsonl` line-by-line (JSONL format).
- Treats a record as **emergent** if:
  - `specialists` list is empty, OR
  - `routing.is_unknown == true`, OR
  - `routing.reason` is `"unknown_domain"` or `"no_domain_stats"`.
- Optionally filters out very short prompts.
- Formats each selected record into a text block of the form:

  ```text
  ### PROMPT ###
  <original prompt>

  ### COMPLETION ###
  <model completion>


  ```

- Writes all selected blocks into a `.txt` corpus file under:

  ```text
  shiftable_project/data/<expert_name>/
  ```

By default, the output filename is:

```text
emergent_<expert_name>.txt
```


## 3. Dependencies

The script uses only the Python standard library:

- `argparse`
- `json`
- `textwrap`
- `pathlib`

No additional packages are required.


## 4. Running the Script

From your **shiftable_project** root directory:

```bash
cd shiftable_project/data/prep/emergence

python prepare_emergent_expert_corpus.py --expert-name NEWDOMAIN
```

Replace `NEWDOMAIN` with the name you want for the new specialist,
for example `cyberrisk` or `governance`.


### 4.1 Common Options

The full set of CLI arguments:

```bash
python prepare_emergent_expert_corpus.py \
  --expert-name NEWDOMAIN \
  --max-samples 1000 \
  --min-prompt-length 10 \
  --output-filename emergent_NEWDOMAIN.txt
```

- `--expert-name` (required)  
  Name of the new expert/domain. This also becomes the directory name under `data/`.

  Example: `--expert-name cyberrisk` → corpus directory:
  `shiftable_project/data/cyberrisk/`

- `--max-samples` (optional, default: `1000`)  
  Maximum number of emergent records to include in the corpus.  
  Use `--max-samples -1` to include *all* emergent records.

- `--min-prompt-length` (optional, default: `10`)  
  Minimum number of characters in the prompt required to include a record.
  This helps filter out trivial or empty prompts.

- `--output-filename` (optional)  
  Explicit filename to use for the corpus. If omitted, the script uses:

  ```text
  emergent_<expert_name>.txt
  ```


### 4.2 Example Commands

**Example 1 – Basic usage**

```bash
python prepare_emergent_expert_corpus.py --expert-name cyberrisk
```

This will:

- Create `shiftable_project/data/cyberrisk/` (if it doesn’t exist).
- Scan `emergence_log.jsonl` for emergent records.
- Keep up to 1000 records (default).
- Save them to:

  ```text
  shiftable_project/data/cyberrisk/emergent_cyberrisk.txt
  ```


**Example 2 – Unlimited samples, longer prompts only**

```bash
python prepare_emergent_expert_corpus.py \
  --expert-name gov_ai \
  --max-samples -1 \
  --min-prompt-length 40
```

This will include all emergent records whose prompts have at least 40 characters
and save them to:

```text
shiftable_project/data/gov_ai/emergent_gov_ai.txt
```


**Example 3 – Custom corpus filename**

```bash
python prepare_emergent_expert_corpus.py \
  --expert-name quant_strategy \
  --output-filename emergent_quant_batch1.txt
```

Creates:

```text
shiftable_project/data/quant_strategy/emergent_quant_batch1.txt
```


## 5. Integrating the New Expert into SMA

Once you have generated a corpus for `NEWDOMAIN`, you can integrate it into
the SMA PoC as follows:

1. Confirm the corpus exists:

   ```text
   shiftable_project/data/NEWDOMAIN/emergent_NEWDOMAIN.txt
   ```

2. Start (or restart) the SMA API server, if it’s not already running:

   ```bash
   cd sma_poc
   uvicorn main:app --reload
   ```

3. Use the API or UI to register the new specialist:

   - **UI**: Open `http://localhost:8000/`, go to the Specialists panel, and
     type `NEWDOMAIN` into the “Add Specialist” form, then click **Add Specialist**.

   - **API**: Call the `/specialists` endpoint:

     ```bash
     curl -X POST http://localhost:8000/specialists \
       -H "Content-Type: application/json" \
       -d '{"name": "NEWDOMAIN"}'
     ```

4. The SMA system will:

   - Retrain the shiftable model including the new specialist.
   - Rebuild domain statistics (centroids + diagonal covariances).
   - Reload the model and geometric router.

From that point on, the router and emergent-logging logic can route some queries
to this new expert and continue to log **truly novel** queries for future experts.


## 6. Notes

- The script intentionally does **minimal** filtering and formatting so that
  your training pipeline keeps as much signal as possible.
- You can open the resulting `.txt` corpus files directly to inspect the kinds
  of emergent queries that led to your new expert.
- If you want more sophisticated selection (e.g., clustering, scoring by
  similarity), you can add that on top of this script or in a separate data
  prep tool.

This helper is designed to be a simple, reliable bridge from **emergent-domain
logging** to a **ready-to-train specialist corpus** for the SMA framework.
