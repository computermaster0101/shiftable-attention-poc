# Shiftable Attention
### *Modular Attention Extensions for Building Specialist Language Models Without Full Retraining*

Shiftable Attention (SMA) is a modular attention framework that allows you to:

- Start with a **fully trained generalist base model**
- Freeze the entire base model (no forgetting)
- Attach **specialist attention branches**
- Train only the new specialist modules + a **domain gate**
- Dynamically shift attention between generalist and specialists
- Expand the model infinitely with new experts
- Avoid catastrophic forgetting entirely

SMA enables practical **continual domain specialization** without retraining your base model.

---

# 🔧 Key Features

- ✔ **Shiftable Multi-Head Attention (SMA)** – attaches specialist attention to frozen layers  
- ✔ **Domain Gate** – routes attention per-layer based on input  
- ✔ **Multi-specialist support** – add 1, 5, or 50 experts  
- ✔ **Drop-in replacement** for standard `nn.TransformerEncoderLayer`  
- ✔ **Base model remains untouched**  
- ✔ **Supports any tokenizer & corpus**  
- ✔ **Ready for dictionary → specialist workflows**

---

# 📦 Installation

Clone and install:

```bash
git clone https://github.com/yourname/shiftable-attention
cd shiftable-attention
pip install -e .
```

Requires:

- Python 3.9+
- PyTorch 2.0+

---

# 📁 Project Structure

```
shiftable_attention/
├── shiftable_attention/
│   ├── __init__.py
│   ├── gate.py                 # Domain gating module
│   ├── sma.py                  # Shiftable Multi-Head Attention
│   └── blocks.py               # Transformer block wrapping SMA
├── models.py                    # Base transformer LM
├── datasets.py                  # Tokenized dataset utilities
├── shiftable_lm.py              # Specialist LM wrapper
├── train_sma_lm.py              # Base + specialist training pipeline
└── README.md
```

---

# 🚀 Quick Start

## 1️⃣ Train the Base Model

Your general corpus should contain dictionaries, thesauri, encyclopedias, etc.

```bash
python train_sma_lm.py --general_corpus data/general_corpus.txt
```

This trains:

- Token embeddings  
- Positional embeddings  
- Transformer encoder  
- LM head  

Output: `base_lm.pt`

---

## 2️⃣ Train a Specialist Model (Shiftable Attention)

Example: specialize for **Data Engineering / Data Science**

```bash
python train_sma_lm.py --special_corpus data/data_engineering_corpus.txt
```

This:

- Loads your frozen base LM
- Creates 1 or more specialist branches
- Trains only specialist attention + gate
- Learns attention shift behavior

Output: `shiftable_lm.pt`

---

# 🧠 What Is “Attention Shift”?

Shiftable attention merges multiple attention paths:

```
Y = g_base * BaseAttention(X)
  + g_spec * SpecialistAttention(X)
```

Where:

- `BaseAttention` = frozen base transformer attention  
- `SpecialistAttention` = newly trained attention  
- `g_base`, `g_spec` are gate weights learned dynamically  

The model decides **which expert to use per input**.

---

# 📊 Inspecting Gate Behavior

```python
logits, gates = model(input_ids, return_gates=True)

for layer, g in enumerate(gates):
    print(f"Layer {layer}:", g.mean(dim=0))
```

Example output:

```
Layer 0: [0.91 base, 0.09 specialist]
Layer 1: [0.62 base, 0.38 specialist]
Layer 2: [0.20 base, 0.80 specialist]
```

Meaning:

- Early layers rely on general knowledge  
- Later layers shift sharply into specialist mode on domain-matching text  

---

# 🧱 Using SMA Blocks Directly

```python
from shiftable_attention import ShiftableTransformerBlock

block = ShiftableTransformerBlock(
    d_model=512,
    num_heads=8,
    dim_feedforward=2048,
    base_mha=pretrained_layer.self_attn,
    num_specialists=3
)
```

Attach this into your own architecture.

---

# 🌱 Adding Multiple Specialists

Supports any number of specialists:

```python
model = ShiftableLanguageModel(
    base_model,
    num_specialists=4
)
```

Possible domains:

- Math specialist  
- Data Engineering specialist  
- Writing specialist  
- Medical specialist  

Each learns separately.

---

# 🛠 Example Inference

```python
from shiftable_lm import ShiftableLanguageModel

model = ShiftableLanguageModel(base_model, num_specialists=1)
model.load_state_dict(torch.load("shiftable_lm.pt"))
model.eval()

prompt = "Explain distributed SQL query planning..."
ids = tokenizer.encode(prompt, return_tensors="pt")

logits, gates = model(ids, return_gates=True)
print(gates[-1])
```

---

# 📚 Recommended Corpus Strategy

### Base Corpus (generalist)
- Dictionaries  
- Thesauri  
- Encyclopedias  
- Wikipedia  
- Textbook excerpts  
- Glossaries  

### Specialist Corpus
- Data engineering  
- Big data & Spark  
- SQL  
- Data modeling  
- ML engineering  
- Cloud architecture  

---

# 🧪 Loss Functions

Default: Cross-entropy LM loss.

You can also add:

### Gate entropy regularization
Encourage decisive routing:

```python
entropy = -(gate * gate.log()).sum(dim=-1).mean()
loss = lm_loss + 0.01 * entropy
```

### Gate sparsity penalties
Force experts to activate distinctly.

---

# ⚡ Internals Summary

Each layer does:

1. Frozen **base** MHA  
2. Trainable **specialist** MHA  
3. Gate → softmax([base, spec])  
4. Blended output  

Base model is *never touched*.

---

# 🙌 Contributions

PRs welcome.  
Add new specialists, better gates, or more efficient attention variants.  
This repo is a sandbox for building modular, expandable AI systems.

