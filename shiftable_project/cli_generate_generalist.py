from __future__ import annotations

"""CLI utility to test *generalist-only* generation.

This bypasses the app stack (router, shiftable model, emergence logic, etc.)
and loads only:
  - tokenizer.json
  - generalist.pt

Example:
  python -m shiftable_project.cli_generate_generalist \
    --tokenizer shiftable_project/outputs/generalist/tokenizer.json \
    --ckpt shiftable_project/outputs/generalist/generalist.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 80 \
    --temperature 0.8 \
    --top_k 50

Tip:
  If you suspect the model is suffering from 'future token leakage' (trained
  with an encoder without a causal mask), try adding `--causal` to force a
  causal attention mask during generation.
"""

import argparse
from typing import Optional

import torch

from .tokenizer import SimpleTokenizer
from .models import BaseTransformerLM


def _logits_generalist(
    model: BaseTransformerLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    causal: bool,
) -> torch.Tensor:
    """Compute logits, optionally forcing a causal attention mask."""
    if not causal:
        return model(input_ids, attention_mask=attention_mask)

    # Manual forward to inject a causal mask without changing model code.
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    x = model.tok_emb(input_ids) + model.pos_emb(positions)

    if attention_mask is not None:
        key_padding_mask = attention_mask == 0
    else:
        key_padding_mask = input_ids == model.pad_id

    # True entries are disallowed positions for PyTorch's bool attn_mask.
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    x = model.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
    return model.lm_head(x)


@torch.no_grad()
def generate(
    model: BaseTransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    causal: bool,
    gen_len: int | None = None,
    bidirectional_steps: int = 0,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_specials=False)
    prefix = [tokenizer.bos_id] + prompt_ids

    # -------------------------------
    # Bidirectional "refinement" decode
    # -------------------------------
    if bidirectional_steps and (gen_len is not None):
        # Start with PAD tokens for the unknown future.
        seq = prefix + [tokenizer.pad_id] * int(gen_len)

        for _ in range(int(bidirectional_steps)):
            context = seq[-model.max_seq_len :]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = (x != tokenizer.pad_id).long()

            logits = _logits_generalist(model, x, attn_mask, causal=False)  # bidirectional on purpose
            start_pos = max(0, len(context) - (len(seq) - len(prefix)))  # where generated region begins inside `context`

            # Resample only the generated region.
            for i in range(start_pos, logits.size(1)):
                next_logits = logits[0, i, :]

                if temperature <= 0:
                    tok_id = int(torch.argmax(next_logits).item())
                else:
                    scaled = next_logits / float(temperature)
                    if 0 < top_k < scaled.numel():
                        values, indices = torch.topk(scaled, top_k)
                        filtered = torch.full_like(scaled, float("-inf"))
                        filtered[indices] = values
                    else:
                        filtered = scaled
                    probs = torch.softmax(filtered, dim=-1)
                    tok_id = int(torch.multinomial(probs, num_samples=1).item())

                # write back into the full `seq` at the matching absolute position
                abs_i = (len(seq) - len(context)) + i
                if abs_i >= len(prefix):  # only modify generated part
                    seq[abs_i] = tok_id

        completion_ids = seq[len(prefix) :]
        # Stop at EOS if it appears
        if tokenizer.eos_id in completion_ids:
            completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_id)]
        return tokenizer.decode(completion_ids, skip_specials=True)

    # -------------------------------
    # Standard autoregressive decode
    # -------------------------------
    generated = prefix[:]

    for _ in range(int(max_new_tokens)):
        context = generated[-model.max_seq_len :]
        context_t = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = (context_t != tokenizer.pad_id).long()

        logits = _logits_generalist(model, context_t, attention_mask, causal=causal)
        next_logits = logits[0, -1, :]

        if temperature <= 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            scaled = next_logits / float(temperature)
            if 0 < top_k < scaled.numel():
                values, indices = torch.topk(scaled, top_k)
                filtered = torch.full_like(scaled, float("-inf"))
                filtered[indices] = values
            else:
                filtered = scaled
            probs = torch.softmax(filtered, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    completion_ids = generated[len(prefix) :]
    return tokenizer.decode(completion_ids, skip_specials=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generalist-only CLI generation test")
    p.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    p.add_argument("--ckpt", type=str, required=True, help="Path to generalist.pt")
    p.add_argument("--prompt", type=str, required=True, help="Prompt text")
    p.add_argument("--max_new_tokens", type=int, default=128, help="AR mode length (default)")
    p.add_argument("--gen_len", type=int, default=None, help="If set, use bidirectional refine mode with this many tokens")
    p.add_argument("--bidirectional_steps", type=int, default=0, help=">0 enables bidirectional refine decoding")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--causal", action="store_true", help="Force causal attention mask during generation (AR only)")

    args = p.parse_args()

    tokenizer = SimpleTokenizer.load(args.tokenizer)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", {})
    if not config:
        raise RuntimeError(f"Checkpoint missing config: {args.ckpt}")

    model = BaseTransformerLM(**config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        causal=bool(args.causal),
        gen_len=(int(args.gen_len) if args.gen_len is not None else None),
        bidirectional_steps=int(args.bidirectional_steps),
    )
    print(out)


if __name__ == "__main__":
    main()
