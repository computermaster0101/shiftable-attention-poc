# SMA PoC – Shiftable Multi-Agent Language Model API

This project (`sma_poc`) wraps your **shiftable_project** and **shiftable_attention** package
in a FastAPI service that:

- Trains a **generalist** Transformer language model on first startup.
- Builds a **ShiftableTransformerLM** that adds **specialist heads** on top of the generalist.
- Exposes an HTTP API for:
  - Health / status
  - Text generation (generalist + specialists)
  - Listing specialists
  - Adding a new specialist
  - Deleting a specialist
- Serves a simple **web UI** for chatting with the model and managing specialists.

The entire flow is self-initializing: when the API is started for the first time,
it trains any missing components before serving requests.


## 1. Project Structure

Expected directory layout:

```text
sma_poc/
  main.py
  app/
    __init__.py
    api.py
    config.py
    model_manager.py
  frontend/
    index.html
  shiftable_project/
    __init__.py
    tokenizer.py
    data.py
    models.py
    train_generalist.py    # (not required by API, but useful to keep)
    train_specialists.py   # (not required by API, but useful to keep)
    data/
      general/             # general corpus (.txt)
      datascience/         # initial specialist corpus example
      businessadvisor/     # initial specialist corpus example
    outputs/
      general/
        tokenizer.json
        generalist.pt
      shiftable/
        shiftable_specialists.pt
  shiftable_attention/
    # GRCLM Implementation 
```

> Note: When you start the API **for the first time**, the outputs in
> `shiftable_project/outputs/` will **not exist**. The API handles this
> by training the required models during initialization.


## 2. Components Overview

### 2.1 shiftable_project

This folder contains the core modeling and data utilities:

- `tokenizer.py` – simple trainable tokenizer.
- `data.py` – utilities to load `.txt` files and build LM datasets.
- `models.py`
  - `BaseTransformerLM`: the **generalist** model (standard Transformer encoder LM).
  - `ShiftableTransformerLM`: a model built from `ShiftableTransformerBlock` layers
    (from `shiftable_attention`), which:
    - Reuse the generalist's self-attention as a **base expert**.
    - Add **specialist experts** per domain (e.g., `datascience`, `businessadvisor`).
    - Learn a **gating network** to mix base + specialists per layer.

### 2.2 shiftable_attention

Additional package that defines `ShiftableTransformerBlock`
and the underlying shiftable attention logic. This is imported as:

```python
from shiftable_attention import ShiftableTransformerBlock
```

### 2.3 app (API server)

- `config.py` – central config:
  - Paths to data and model outputs.
  - Training hyperparameters.
  - Default generation settings.

- `model_manager.py` – the runtime brain:
  - On first use:
    - Trains the **generalist** model if needed.
    - Trains the **shiftable** model for all discovered specialists if needed.
  - Loads tokenizer + models into memory.
  - Provides:
    - `get_status()`
    - `list_specialists()`
    - `add_specialist(name)`
    - `delete_specialist(name)`
    - `generate(prompt, max_new_tokens, temperature, top_k)`

- `api.py` – FastAPI routes:
  - `GET /` – serves the frontend UI.
  - `GET /health` – health + status information.
  - `GET /specialists` – list specialists.
  - `POST /specialists` – add a specialist.
  - `DELETE /specialists/{name}` – delete a specialist.
  - `POST /generate` – generate text from the model.

### 2.4 frontend

- `index.html` – a simple single-page UI:
  - Left panel:
    - Chat interface with the model (generalist + specialists).
    - Simple conversation history.
  - Right panel:
    - List of current specialists and delete buttons.
    - Form to add a new specialist by name (after corpus is present).
    - Status messages for long operations (retraining).


## 3. Data Layout and Specialists

All corpora live under:

```text
shiftable_project/data/
```

- **General corpus** (required):

  ```text
  shiftable_project/data/general/
    file1.txt
    file2.txt
    ...
  ```

- **Specialist corpora** (optional, discovered automatically):

  ```text
  shiftable_project/data/datascience/
    ds1.txt
    ds2.txt
    ...
  shiftable_project/data/businessadvisor/
    biz1.txt
    biz2.txt
    ...
  shiftable_project/data/<specialist_name>/
    your_domain_text1.txt
    your_domain_text2.txt
    ...
  ```

Rules:

- Any directory **under `data/` that is not `general`** is treated as a specialist.
- Each directory must contain at least one `.txt` file.
- On startup and specialist changes, `model_manager` calls
  `_discover_specialist_names()` to determine the full set of specialist domains.


## 4. Initialization Flow (First Startup)

When you start the API for the first time (no checkpoints yet), the following happens:

1. **`ModelManager.ensure_initialized()`** is called at startup.
2. If `tokenizer.json` or `generalist.pt` is missing:
   - `_train_generalist()` runs:
     - Builds a tokenizer from `data/general/*.txt`.
     - Trains a `BaseTransformerLM` on the general corpus.
     - Saves:
       - `outputs/general/tokenizer.json`
       - `outputs/general/generalist.pt`
3. If `shiftable_specialists.pt` is missing:
   - `_train_shiftable_for_all_specialists()` runs:
     - Discovers specialists by listing `data/*` excluding `general`.
     - Builds `ShiftableTransformerLM` with **num_specialists = number of directories found**.
     - Trains on a concatenation of:
       - general dataset (if `data/general` exists)
       - each specialist dataset
     - Saves:
       - `outputs/shiftable/shiftable_specialists.pt`
4. `_load_tokenizer_and_shiftable_model()`:
   - Loads tokenizer from `tokenizer.json`.
   - Loads generalist from `generalist.pt`.
   - Constructs `ShiftableTransformerLM.from_base_model(...)` with the discovered specialists.
   - Loads trained weights from `shiftable_specialists.pt` into this model.
   - Stores tokenizer, shift model, and specialist list in memory.

On subsequent startups, if all these artifacts already exist, the API just loads them directly.


## 5. How the Generalist and Specialists Interact

The `ShiftableTransformerLM` is built from `ShiftableTransformerBlock` layers, each of which has:

- A **base attention** branch (copied from the generalist, typically frozen).
- Multiple **specialist attention** branches (one per specialist).
- A **gating network** per layer that outputs a soft mixture over:
  - base + all specialists.

During training:

- The model sees a mixture of examples from:
  - `general` corpus.
  - All specialist corpora.
- For each example, the gate learns to assign more weight to whichever combination of
  base + specialist heads minimizes LM loss for that text.

At inference (generation), the **same gating mechanism** runs per layer:

- It does **not** hard-select one specialist.
- Instead, it uses a learned **soft mixture** of the base and all specialists appropriate to the prompt.


## 6. Running the API

### 6.1 Install Dependencies

From the `sma_poc` root, in your virtual environment:

```bash
pip install fastapi uvicorn torch
```

Also ensure that your `shiftable_attention` package is importable (e.g., located at
`sma_poc/shiftable_attention` or installed into the environment).

### 6.2 Start the Server

From `sma_poc/`:

```bash
uvicorn main:app --reload
```

The first run may take longer, because it will:

- Train the generalist model (if not already trained).
- Train the shiftable model for all discovered specialists (if not already trained).


### 6.3 Open the UI

Open in your browser:

- `http://localhost:8000/`

You’ll see:

- **Chat panel** (left): talk to the generalist + specialists.
- **Specialists panel** (right):
  - Current specialists list.
  - Buttons to delete specialists (if more than one exists).
  - Form to add a new specialist.


## 7. API Endpoints

Base URL is typically `http://localhost:8000`.


### 7.1 `GET /`

Serves the `frontend/index.html` UI.

### 7.2 `GET /health`

Returns status information.

Example response:

```json
{
  "initialized": true,
  "device": "cuda:0",
  "tokenizer_path": "shiftable_project/outputs/general/tokenizer.json",
  "generalist_ckpt_path": "shiftable_project/outputs/general/generalist.pt",
  "shiftable_ckpt_path": "shiftable_project/outputs/shiftable/shiftable_specialists.pt",
  "specialists": ["datascience", "businessadvisor"]
}
```

### 7.3 `GET /specialists`

Returns a list of currently configured specialists.

Example:

```json
["datascience", "businessadvisor", "marketing"]
```

### 7.4 `POST /generate`

Generate text from the model (generalist + specialists).

**Request body:**

```json
{
  "prompt": "Act as a data science advisor and explain cross-validation:",
  "max_new_tokens": 80,
  "temperature": 0.9,
  "top_k": 50
}
```

- `prompt` – input text.
- `max_new_tokens` – number of tokens to generate.
- `temperature` – 0 for greedy, >0 for sampling.
- `top_k` – 0 for full distribution, >0 to restrict to top-k tokens.

**Response:**

```json
{
  "prompt": "... original prompt ...",
  "completion": "... model's new text ...",
  "full_text": "... prompt + completion ...",
  "specialists": ["datascience", "businessadvisor", "marketing"]
}
```

### 7.5 `POST /specialists` – Add Specialist

Adds a new specialist and retrains the shiftable model on all specialists.

**Important:** Before calling this endpoint, you must:

1. Create a directory: `shiftable_project/data/<name>/`
2. Add one or more `.txt` files as the specialist corpus.

**Request body:**

```json
{
  "name": "marketing"
}
```

**Response (success):**

```json
{
  "message": "Specialist 'marketing' is now configured.",
  "specialists": ["datascience", "businessadvisor", "marketing"]
}
```

If the directory is missing or empty, you’ll get a `400` error with details.


### 7.6 `DELETE /specialists/{name}` – Delete Specialist

Deletes an existing specialist and retrains the shiftable model over the remaining specialists.

Behavior:

- Removes the corpus directory:
  - `shiftable_project/data/{name}`
- Rebuilds and retrains the shiftable model for all remaining specialists.
- Will **not** allow deleting the last remaining specialist (at least one must exist).

**Example:**

```bash
curl -X DELETE http://localhost:8000/specialists/marketing
```

**Response (success):**

```json
{
  "message": "Specialist 'marketing' has been deleted.",
  "specialists": ["datascience", "businessadvisor"]
}
```


## 8. Using the Frontend UI

1. Start the server:

   ```bash
   uvicorn main:app --reload
   ```

2. Open the browser at `http://localhost:8000/`.

3. **Chat with the model:**
   - Type in the prompt box and press **Enter** or click **Send**.
   - The conversation history is passed in the prompt so you have basic context.

4. **Manage specialists:**
   - The right panel lists current specialists.
   - **Delete** buttons remove a specialist (and its corpus directory) and retrain the model.
   - The **Add Specialist** form lets you register a new specialist **after** placing its corpus in
     `shiftable_project/data/<name>/`.

   Example for new specialist:
   - Create: `shiftable_project/data/marketing/`
   - Add: `marketing1.txt`, `marketing2.txt`.
   - In the UI, type `marketing` in the **Add Specialist** form, click **Add Specialist**.
   - The model will be retrained and `marketing` will appear in the list.


## 9. Notes and Extensions

- The current design uses a **soft mixture-of-experts** gate per layer:
  - For each layer, the gate computes a distribution over base + all specialists.
  - Outputs are a weighted sum of each expert's attention output.
  - This allows **partial routing** to multiple specialists when appropriate.

- Possible extensions (not implemented here but compatible with the architecture):
  - Add a `/inspect_gates` endpoint to return the gate weights per layer for a given prompt.
  - Add explicit domain tokens (e.g., `[datascience]`) in prompts to influence routing.
  - Implement hard routing by taking argmax over gate logits per layer.

The current PoC is fully functional end-to-end for:

- Training a generalist and its specialists.
- Serving them via an API.
- Managing specialist lifecycles dynamically.
- Interacting through a minimal but practical web UI.
