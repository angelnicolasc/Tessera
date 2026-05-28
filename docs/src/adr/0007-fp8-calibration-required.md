# ADR-0007 — FP8 `c_kv` storage requires per-layer calibration

**Status:** Accepted, 2026-05-21.

## Context

vLLM's April 2026 blog post documented a severe regression on the FlashMLA FP8 path: 128K
needle-in-haystack recall dropped from 91% to 13% when FP8 was enabled with a fixed,
uncalibrated scale factor. Root cause: imprecise FP32 accumulation in Tensor Cores
overlapped with quantisation error.

Two alternatives we rejected:

* **Ship FP8 with a fixed scale.** Reproduces the recall regression.
* **Disable FP8 until calibration ships.** Wastes the 2× compression composition with MLA.

## Decision

FP8 storage in Tessera requires:

1. **Per-layer calibration.** `python/tessera/fp8_calibrate.py` ships the harness. Operators
   run it offline against a representative corpus to produce a `{layer_idx: scale}` map.
2. **`k_rope` stays BF16, always.** The position-dependent RoPE dot-product is the most
   precision-sensitive path. FP8-quantising it reproduced the regression in the vLLM
   investigation.
3. **128K needle-in-haystack validation before enabling FP8 in production.**

The block layout reserves space for per-layer scale factors (`fp8_scale_block_bytes`); the
block manager treats their absence under BF16 as zero overhead.

## Consequences

* FP8 isn't a "flip a flag" feature. The harness is mandatory before production rollout —
  this is documented in the user-facing README and in the BlockConfig validator.
* `compute_per_layer_scales` is exposed publicly so calibration data can be inspected, diff'd
  across runs, and version-controlled alongside the model.
