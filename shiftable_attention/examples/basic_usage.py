import torch
import torch.nn as nn

from shiftable_attention import ShiftableTransformerBlock


def main():
    batch_size = 4
    seq_len = 16
    d_model = 128
    num_heads = 8

    # Step 1: create and (in a real setup) pretrain a base MHA
    base_mha = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.1,
        batch_first=True,
    )

    # (Here we skip actual pretraining and just demo usage.)

    # Step 2: wrap in a ShiftableTransformerBlock with 2 specialists
    block = ShiftableTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=256,
        dropout=0.1,
        base_mha=base_mha,
        num_specialists=2,
        use_cls_token_pool=False,
    )

    # Step 3: freeze base_mha parameters
    for p in block.self_attn.base_mha.parameters():
        p.requires_grad = False

    # Only train specialist branches & gate (and optionally FFN/norms)
    trainable_params = [p for p in block.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Dummy input
    x = torch.randn(batch_size, seq_len, d_model)

    # Dummy training loop step
    block.train()
    optimizer.zero_grad()

    out, gate = block(x, return_gate=True)  # out: [b, s, d], gate: [b, 1 + num_specialists]

    # For demo: simple self-supervised objective like reconstructing x
    loss = ((out - x) ** 2).mean()
    loss.backward()
    optimizer.step()

    print("Output shape:", out.shape)
    print("Gate shape:", gate.shape)
    print("Gate example:", gate[0].detach().cpu())


if __name__ == "__main__":
    main()
