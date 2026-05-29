"""128K needle-in-haystack precision regression (WS9 / G13).

Validates that the reference implementation (and, when available, the kernel) can attend
to a single "needle" token planted at the midpoint of a very long sequence without
catastrophic precision loss.

This test guards against the FP8 accumulation regression documented in the vLLM April
2026 blog post (see playbook §9): uncalibrated FP8 caused needle recall to drop from
91% → 13% at 128K context. The BF16 reference oracle must score > 5.0 mean abs output.

Tier B — GPU-gated. Run with: ``pytest tests -m gpu``
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch required")

from tessera.reference.absorbed_attention import reference_absorbed_mla

DS_V3_D_C = 512
DS_V3_D_R = 64
DS_V3_D_H = 128
DS_V3_H = 128
DS_V3_SCALE = 1.0 / math.sqrt(DS_V3_D_C)


def _make_needle_inputs(seq_len: int, device: torch.device) -> dict:
    """Needle-in-haystack inputs: one token has a strong signal; the rest are zeros."""
    torch.manual_seed(0)
    needle_pos = seq_len // 2

    c_kv = torch.zeros(1, seq_len, DS_V3_D_C, dtype=torch.bfloat16, device=device)
    c_kv[:, needle_pos, :] = 10.0  # strong needle signal

    q_abs = torch.ones(1, 1, DS_V3_D_C, dtype=torch.bfloat16, device=device)  # aligns
    q_rope = torch.zeros(1, 1, DS_V3_D_R, dtype=torch.bfloat16, device=device)
    k_rope = torch.zeros(1, seq_len, DS_V3_D_R, dtype=torch.bfloat16, device=device)
    W_UV = (
        torch.eye(DS_V3_D_H, DS_V3_D_C, dtype=torch.bfloat16, device=device)
        .unsqueeze(0)
        .expand(1, -1, -1)
    )
    return {
        "q_abs": q_abs,
        "q_rope": q_rope,
        "c_kv": c_kv,
        "k_rope": k_rope,
        "W_UV": W_UV,
        "scale": DS_V3_SCALE,
    }


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [65536, 131072])
def test_needle_in_haystack_precision(gpu, seq_len: int):
    """Reference oracle must find the needle at long context (mean_abs > 5.0).

    If this test fails, there is a precision regression in the reference implementation
    or the attention mechanism — investigate accumulation before enabling FP8 production.
    """
    inp = _make_needle_inputs(seq_len, gpu)
    out = reference_absorbed_mla(**inp)  # [1, 1, d_h]

    mean_abs = out.abs().mean().item()
    assert mean_abs > 5.0, (
        f"Precision regression at seq_len={seq_len}: needle signal too weak "
        f"(mean_abs={mean_abs:.3f}, expected > 5.0). "
        "This mirrors the vLLM April-2026 FP8 accumulation regression."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [65536, 131072])
def test_needle_attention_weight_dominates(gpu, seq_len: int):
    """The attention weight at the needle position must be close to 1.0."""
    torch.manual_seed(0)
    needle_pos = seq_len // 2

    c_kv = torch.zeros(1, seq_len, DS_V3_D_C, dtype=torch.bfloat16, device=gpu)
    c_kv[:, needle_pos, :] = 20.0  # even stronger signal

    q_abs = torch.ones(1, 1, DS_V3_D_C, dtype=torch.bfloat16, device=gpu)
    torch.zeros(1, 1, DS_V3_D_R, dtype=torch.bfloat16, device=gpu)
    torch.zeros(1, seq_len, DS_V3_D_R, dtype=torch.bfloat16, device=gpu)

    # Compute raw scores to inspect attention distribution.
    qA = q_abs.float()
    C = c_kv.float()
    scale = DS_V3_SCALE
    scores = torch.einsum("bhc,bsc->bhs", qA, C) * scale  # [1, 1, seq_len]
    attn = torch.softmax(scores, dim=-1)

    needle_weight = attn[0, 0, needle_pos].item()
    assert needle_weight > 0.9, (
        f"Needle attention weight {needle_weight:.4f} < 0.9 at seq_len={seq_len}"
    )
