"""Tests for the PyTorch reference oracle (WS3 / ADR-0011).

These tests run on CPU — no GPU required. They verify the reference implementation
against hand-computed cases and validate the absorption trick numerics.

Skip gracefully if PyTorch is not installed (e.g. in minimal CI environments).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch not installed; skipping reference oracle tests"
)

from tessera.reference.absorbed_attention import reference_absorbed_mla

# ─────────────────────────── fixtures ────────────────────────────────────────


def _make_inputs(B: int, H: int, S: int, d_c: int, d_r: int, d_h: int, seed: int = 0):
    """Return a dict of deterministic random tensors matching reference_absorbed_mla's signature."""
    torch.manual_seed(seed)
    dtype = torch.float32
    return {
        "q_abs": torch.randn(B, H, d_c, dtype=dtype),
        "q_rope": torch.randn(B, H, d_r, dtype=dtype),
        "c_kv": torch.randn(B, S, d_c, dtype=dtype),
        "k_rope": torch.randn(B, S, d_r, dtype=dtype),
        "W_UV": torch.randn(H, d_h, d_c, dtype=dtype),
        "scale": 1.0 / math.sqrt(d_c),
    }


# ─────────────────────────── sanity checks ───────────────────────────────────


class TestReferenceMlaShape:
    """Output shape and dtype must be correct for various configs."""

    @pytest.mark.parametrize(
        ("B", "H", "S", "d_c", "d_r", "d_h"),
        [
            (1, 1, 2, 4, 2, 4),
            (2, 4, 8, 16, 4, 16),
            (1, 8, 64, 32, 8, 32),
        ],
    )
    def test_output_shape(self, B, H, S, d_c, d_r, d_h):
        inp = _make_inputs(B, H, S, d_c, d_r, d_h)
        out = reference_absorbed_mla(**inp)
        assert out.shape == (B, H, d_h), f"expected {(B, H, d_h)}, got {out.shape}"

    def test_output_dtype_matches_input(self):
        inp = _make_inputs(1, 1, 2, 4, 2, 4)
        for dtype in (torch.float32, torch.bfloat16):
            inp2 = {k: v.to(dtype) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            out = reference_absorbed_mla(**inp2)
            assert out.dtype == dtype, f"output dtype {out.dtype} != input dtype {dtype}"


class TestReferenceMlaNumerics:
    """Hand-computed ground-truth for B=1, H=1, S=2, d_c=4, d_r=2."""

    def _tiny_case(self) -> dict:
        """Minimal deterministic config for hand-verification."""
        # All-ones query and KV → uniform attention → output is mean of c_kv rows.
        B, H, S, d_c, d_r, d_h = 1, 1, 2, 4, 2, 4
        q_abs = torch.ones(B, H, d_c)
        q_rope = torch.zeros(B, H, d_r)  # no RoPE contribution
        c_kv = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])  # [1, 2, 4]
        k_rope = torch.zeros(B, S, d_r)  # no RoPE contribution
        W_UV = torch.eye(d_h, d_c).unsqueeze(0)  # [1, 4, 4] identity
        scale = 1.0 / math.sqrt(d_c)
        return {
            "q_abs": q_abs,
            "q_rope": q_rope,
            "c_kv": c_kv,
            "k_rope": k_rope,
            "W_UV": W_UV,
            "scale": scale,
        }

    def test_uniform_attention_yields_mean_of_ckv(self):
        """With all-ones query and zero RoPE, attention is uniform → output = mean(c_kv) @ W_UV."""
        inp = self._tiny_case()
        out = reference_absorbed_mla(**inp)  # [1, 1, 4]

        # With W_UV = identity, output should be mean(c_kv, dim=S) = [0.5, 0, 0, 0.5].
        expected = torch.tensor([0.5, 0.0, 0.0, 0.5])
        torch.testing.assert_close(out.squeeze(), expected, atol=1e-5, rtol=1e-5)

    def test_needle_signal_dominates(self):
        """A strong needle in c_kv should dominate the output when query aligns with it."""
        B, H, S, d_c, d_r, d_h = 1, 1, 16, 8, 2, 8
        needle_pos = 7
        c_kv = torch.zeros(B, S, d_c)
        c_kv[:, needle_pos, :] = 10.0  # strong signal at needle_pos

        q_abs = torch.ones(B, H, d_c)  # aligns with needle
        q_rope = torch.zeros(B, H, d_r)
        k_rope = torch.zeros(B, S, d_r)
        W_UV = torch.eye(d_h, d_c).unsqueeze(0)
        scale = 1.0 / math.sqrt(d_c)

        out = reference_absorbed_mla(q_abs, q_rope, c_kv, k_rope, W_UV, scale)
        # Output magnitude should be dominated by the needle (close to 10.0 in each dim).
        assert out.abs().mean().item() > 5.0, "needle signal must dominate output"

    def test_deterministic_across_calls(self):
        """Same inputs produce bit-identical outputs."""
        inp = _make_inputs(2, 4, 8, 16, 4, 16, seed=42)
        out1 = reference_absorbed_mla(**inp)
        out2 = reference_absorbed_mla(**inp)
        torch.testing.assert_close(out1, out2)

    def test_scale_zero_gives_uniform_attention(self):
        """scale=0 makes all scores equal → softmax → uniform → mean of c_kv rows."""
        B, H, S, d_c, d_r, d_h = 1, 1, 4, 4, 2, 4
        q_abs = torch.randn(B, H, d_c)
        q_rope = torch.randn(B, H, d_r)
        c_kv = torch.randn(B, S, d_c)
        k_rope = torch.randn(B, S, d_r)
        W_UV = torch.eye(d_h, d_c).unsqueeze(0)

        out = reference_absorbed_mla(q_abs, q_rope, c_kv, k_rope, W_UV, scale=0.0)
        expected_ckv_mean = c_kv.mean(dim=1)  # [B, d_c]
        # With identity W_UV, output = c_kv_mean broadcast over heads.
        torch.testing.assert_close(out.squeeze(1), expected_ckv_mean, atol=1e-5, rtol=1e-5)
