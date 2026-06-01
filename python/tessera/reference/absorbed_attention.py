"""PyTorch reference implementation of absorbed MLA attention.

This is the correctness oracle used by all parity tests (WS3 / ADR-0011). It runs
entirely in Python/PyTorch with FP32 internal accumulation; it is never called on the
hot path — only in tests.

Absorption trick recap (from the playbook §2.2):
    q_absorbed = q_content @ W_UK    (query absorbs W_UK; computed once per decode step)
    attention over c_kv directly — W_UV also absorbed into the output transform.

This implementation computes all tensors in FP32 to avoid accumulation errors that
would mask real kernel bugs. The output is cast back to ``q_abs.dtype`` so the caller
can use the result in dtype-sensitive comparisons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def reference_absorbed_mla(
    q_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    W_UV: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference absorbed MLA attention with FP32 accumulation.

    Args:
        q_abs:  ``[B, H, d_c]`` — query with ``W_UK`` absorbed (content component).
        q_rope: ``[B, H, d_r]`` — query RoPE component.
        c_kv:   ``[B, S, d_c]`` — KV latent (position-independent, ``c_kv`` in the
                playbook). Shared across heads.
        k_rope: ``[B, S, d_r]`` — decoupled RoPE keys (MQA-style; broadcast across H).
        W_UV:   ``[H, d_h, d_c]`` — value up-projection absorbed into output transform.
        scale:  softmax scale factor (typically ``1 / sqrt(d_c)``).

    Returns:
        ``[B, H, d_h]`` attention output, same dtype as ``q_abs``.
    """
    import torch  # local import — module-level import is skipped when torch is absent

    # Upcast to FP32 for accumulation stability (avoids the vLLM April-2026 precision
    # regression documented in the playbook §9 and WS9 needle-in-haystack tests).
    qA = q_abs.float()  # [B, H, d_c]
    qR = q_rope.float()  # [B, H, d_r]
    C = c_kv.float()  # [B, S, d_c]
    KR = k_rope.float()  # [B, S, d_r]
    WV = W_UV.float()  # [H, d_h, d_c]

    # Content attention scores: q_absorbed · c_kv^T  →  [B, H, S]
    content_scores = torch.einsum("bhc,bsc->bhs", qA, C) * scale

    # RoPE attention scores: q_rope · k_rope^T (MQA: k_rope broadcast over heads) → [B, H, S]
    rope_scores = torch.einsum("bhd,bsd->bhs", qR, KR) * scale

    scores = content_scores + rope_scores  # [B, H, S]
    attn = torch.softmax(scores, dim=-1)  # [B, H, S]

    # Weighted sum over c_kv: attn · c_kv  →  [B, H, d_c]
    weighted_ckv = torch.einsum("bhs,bsc->bhc", attn, C)

    # Apply absorbed W_UV: [B, H, d_h]
    out = torch.einsum("bhc,hdc->bhd", weighted_ckv, WV)

    return out.to(q_abs.dtype)
