# ADR-0011 — PyTorch reference oracle for absorbed MLA attention

**Status**: Accepted  
**Sprint**: 1 (WS3)  
**Supersedes**: n/a  

---

## Context

Tessera's kernel backends (FlashMLA, FlashInfer, Triton fallback) produce BF16 or FP8
attention outputs. To verify that these kernels implement the absorbed MLA attention
correctly, we need a known-correct reference implementation to compare against.

The alternatives considered were:

1. **Compare kernels against each other** (FlashMLA vs FlashInfer): catches *differences*
   but cannot tell which is wrong if both have the same bug. Also requires two working GPU
   environments.
2. **Use the DeepSeek reference implementation**: written for training, not inference; uses
   different tensor layouts and does not model the absorbed-attention formulation used by
   Tessera's block layout.
3. **Mathematical derivation in Rust**: possible but would live behind a `#[cfg(test)]` wall,
   not usable from Python test infrastructure.
4. **PyTorch FP32 CPU oracle**: universally available, no GPU required, easy to read and
   audit, produces ground truth that GPU kernels must match within a tolerance.

The vLLM April 2026 FP8 regression (128K needle-in-haystack recall fell from 91% to 13%
with uncalibrated FP8) demonstrated that even heavily tested production kernels can regress
on long-context precision. A CPU oracle that runs in CI on every push provides the earliest
possible regression signal.

## Decision

Ship `python/tessera/reference/absorbed_attention.py` containing
`reference_absorbed_mla(q_abs, q_rope, c_kv, k_rope, W_UV, scale) -> Tensor`.

The implementation:

1. **Accumulates in FP32** regardless of input dtype. All einsum contractions are cast to
   `float32` before computation and cast back to the input dtype on output. This ensures
   numerical fidelity is limited only by the FP32 mantissa (24 bits), not BF16 (7 bits).
2. **Operates on full sequences** (not paged). It receives `c_kv` as a dense
   `[batch, seq_len, d_c]` tensor. The paging abstraction is transparent to the oracle.
3. **Models the absorbed formulation exactly**:
   - Content scores: `einsum("bhc,bsc->bhs", q_abs_f32, c_kv_f32) * scale`
   - RoPE scores: `einsum("bhd,bsd->bhs", q_rope_f32, k_rope_f32) * scale`
   - Attention weights: `softmax(content_scores + rope_scores, dim=-1)`
   - Weighted sum: `einsum("bhs,bsc->bhc", attn, c_kv_f32)`
   - Projection: `einsum("bhc,hdc->bhd", weighted_ckv, W_UV_f32)`
4. **Imports torch locally** — the reference module is importable without torch; only
   calling `reference_absorbed_mla` requires it. This keeps the import footprint of the
   test suite clean.

Parity tolerance for GPU kernels: `atol=1e-2, rtol=1e-3` (BF16 headroom; tightened to
`atol=1e-3` for FP32 inputs). The needle-in-haystack test uses an absolute signal threshold
(`mean_abs > 5.0`) that would catch any precision regression similar to the vLLM incident.

## Consequences

**Good**

- Pure CPU reference: runs in every CI environment, no GPU required, no environment
  variables, no CUDA driver.
- Mathematical transparency: the oracle reads like the paper (Attention Is All You Need +
  MLA formulation from DeepSeek-V2 tech report). Any reviewer can verify correctness by
  inspection in < 5 minutes.
- FP32 accumulation catches BF16 overflow early. The needle test (`seq_len=131072`,
  `needle_signal=10.0`) was specifically designed to catch the vLLM April 2026 regression
  class.
- Tier B GPU parity tests (`test_flash_mla_parity.py`, `test_flash_infer_parity.py`) share
  the same oracle with no additional infrastructure — cloud burst adds one `pytest -m gpu`
  invocation.

**Trade-offs**

- FP32 accumulation means the oracle is *more* precise than BF16 production kernels. The
  tolerance band (`atol=1e-2`) must be wide enough to accommodate legitimate BF16 rounding
  differences. If a kernel is numerically correct but the test fails due to tight tolerance,
  the tolerance should be widened with a comment explaining why, not the kernel fixed.
- The oracle does not model FP8 quantization error. FP8 parity tests must pass a separately
  calibrated `fp8_scales` tensor; the oracle output is used as the *pre-quantization* ground
  truth, not post-quantization.
- A CPU oracle at `seq_len=131072` takes ~10 seconds on a 32-core machine (the einsum
  scales as `O(S * d_c * H)`). The GPU parity tests are therefore Tier B (GPU-gated) to
  avoid making CI slow. Only the small-tensor sanity tests (`S ≤ 64`) run in Tier A CI.

## Alternatives considered

- **Numba / Cython oracle**: faster but harder to audit; adds a build dependency. Rejected.
- **JAX oracle**: cleaner autodiff story but adds another heavy dependency. Rejected.
- **Existing FlashMLA Python reference**: not maintained separately from the CUDA kernel;
  subject to the same regression risk we are trying to catch. Rejected.
