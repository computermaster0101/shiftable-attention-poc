# Shiftable Generalist + Specialist Transformer LM

This project demonstrates how to use the `shiftable_attention` package to:

- Train a **generalist** Transformer language model (LM)
- Build a **ShiftableTransformerLM** that:
  - Reuses and effectively freezes the generalist's base attention layers
  - Adds **two specialist heads**:
    - `datascience`
    - `businessadvisor`
- Train the specialist branches and their domain gate on specialist corpora

You provide the text corpora for:

- The **generalist** model
- The **datascience** specialist
- The **businessadvisor** specialist

All code is fully implemented with no placeholders.


## Project Layout

Place the extracted `shiftable_attention` package and the training scripts in the same directory:

```text
your_project/
  shiftable_attention/           # from your Archive.zip (installed or importable)
  tokenizer.py
  data.py
  models.py
  train_generalist.py
  train_specialists.py
```

Make sure you can `import shiftable_attention` from this directory (or install it into your environment).


## Dependencies

- Python 3.9+
- PyTorch
- Your `shiftable_attention` package (the one from `Archive.zip`)

Install PyTorch according to your CUDA / CPU setup from the official docs, then ensure that
`shiftable_attention` is on the Python path (e.g., by keeping it in the same directory
as the training scripts or installing it as a package).


## Data Expectations

Each corpus is a directory of **plain text `.txt` files**. For example:

```text
data/
  general/
    file1.txt
    file2.txt
    ...
  datascience/
    ds1.txt
    ds2.txt
    ...
  business/
    biz1.txt
    biz2.txt
    ...
```

Each `.txt` file can contain one or many lines of text; the scripts will tokenize and build
next-token-prediction samples from them.


## 1. Train the Generalist Model

Use `train_generalist.py` on your **general** corpus to train a base Transformer LM and build
a tokenizer vocabulary.

Example:

```bash
python train_generalist.py   --data_dir data/general   --output_dir outputs/general   --seq_len 128   --d_model 256   --n_heads 8   --n_layers 4   --dim_feedforward 1024   --batch_size 32   --epochs 5   --lr 3e-4
```

This will:

- Train a `BaseTransformerLM` on `data/general`
- Create and save:
  - `outputs/general/tokenizer.json`
  - `outputs/general/generalist.pt`

`generalist.pt` contains:

- The model weights
- A config dict (dimensions, vocab size, etc.)


## 2. Train the Specialists (datascience + businessadvisor)

Once the generalist is trained, use `train_specialists.py` to create and train the
shiftable model with the two specialist heads:

- Specialist 1: `datascience`
- Specialist 2: `businessadvisor`

Example:

```bash
python train_specialists.py   --tokenizer_path outputs/general/tokenizer.json   --generalist_ckpt outputs/general/generalist.pt   --datascience_dir data/datascience   --business_dir data/business   --general_dir data/general   --output_dir outputs/shiftable   --seq_len 128   --batch_size 32   --epochs 5   --lr 3e-4
```

Arguments:

- `--tokenizer_path`  
  Path to the tokenizer JSON saved during generalist training.

- `--generalist_ckpt`  
  Path to `generalist.pt` checkpoint.

- `--datascience_dir`  
  Directory containing the **data science** specialist corpus (`.txt` files).

- `--business_dir`  
  Directory containing the **business advisor** specialist corpus (`.txt` files).

- `--general_dir` (optional but recommended)  
  General corpus to mix in during specialist training. Use this to keep the specialists from
  drifting too far from general usage.

- `--output_dir`  
  Directory where the **shiftable specialist model** checkpoint will be stored.


### What This Step Does

`train_specialists.py` will:

1. Load the tokenizer and generalist model.
2. Build a `ShiftableTransformerLM` using your `shiftable_attention.ShiftableTransformerBlock`,
   with two specialist heads named:
   - `datascience`
   - `businessadvisor`
3. Optionally freeze the embeddings and LM head so that only specialist attention and gating
   layers are updated (you can change this behavior by editing the script).
4. Train on a concatenation of the domain datasets (general + datascience + business).


### Output

The script saves a checkpoint, e.g.:

```text
outputs/shiftable/shiftable_specialists.pt
```

This file contains:

- The shiftable model's weights
- Configuration (vocab size, dimensions, number and names of specialists)


## 3. Next Steps (Optional)

You can now:

- Add an inference / generation script that:
  - Loads `shiftable_specialists.pt` and `tokenizer.json`
  - Generates text given a prompt
  - Inspects / prints gate activations for:
    - base (generalist)
    - datascience specialist
    - businessadvisor specialist

The training scripts in this repo are self-contained; once your corpora are in place and
dependencies are installed, you can run them as-is.
