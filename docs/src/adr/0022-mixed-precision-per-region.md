# ADR-0022 — Mixed precision per region inside a V4 compressed entry

**Status:** Accepted, 2026-05-28.

## Context

V4 stores each compressed KV entry as a heterogeneous record (paper §2.3.4):

* **Last 64 dims** (RoPE) — **BF16** (2 bytes/elem). Precision-sensitive: vLLM's April
  2026 analysis traced the long-context FP8 regression to imprecise accumulation in the
  position-dependent path. The paper deliberately keeps RoPE in BF16.
* **Remaining `head_dim - 64`** elements (content) — **FP8 E4M3** (1 byte/elem). The 2×
  storage savings on the dominant region.
* **Indexer entries** (CSA only) — **FP4 E2M1** (0.5 bytes/elem). The Lightning Indexer's
  selection scores tolerate aggressive quantisation; FP4 doubles its throughput.

Tessera's Sprint 0 `CkvDtype { Bf16, Fp8E4m3 }` describes a single dtype for the whole
primary region. That doesn't fit V4 where three precisions co-resident in one block.

## Decision

Two extensions to `CkvDtype`:

1. **New variant `Fp4E2m1`** — first-class 4-bit dtype. `bytes_for_elements(n)` returns
   `(n + 1) / 2` (sub-byte packing, ceiling).
2. **New sentinel `MixedBf16Fp8Fp4`** — signals "the scheme variant carries its own
   layout". `bytes()` returns 0 to make accidental use as a scalar bytes-per-elem clearly
   wrong; callers must consult `scheme.bytes_per_token_per_layer()` instead.

V4 scheme variants (V4Csa / V4Hca / V4Swa) encode their own per-region layout. Each
variant's `bytes_per_token_per_layer()` decomposes into:

```rust
let bf16_bytes = CkvDtype::Bf16.bytes_for_elements(rope_dim);
let fp8_bytes  = CkvDtype::Fp8E4m3.bytes_for_elements(head_dim - rope_dim);
let fp4_bytes  = CkvDtype::Fp4E2m1.bytes_for_elements(indexer_head_dim);  // CSA only
let entry_bytes = bf16_bytes + fp8_bytes + fp4_bytes;
let per_token   = entry_bytes / k;  // k = k1 for CSA, k2 for HCA, 1 for SWA
```

Validation rules:

* V4 configs require `ckv_dtype == MixedBf16Fp8Fp4` (Python config enforces; the Rust
  scheme is self-describing so this is a defensive check at the Python boundary).
* `requires_per_layer_scales` is `true` for `Fp4E2m1` and `MixedBf16Fp8Fp4` — both need
  calibration like FP8 does (ADR-0007's regression concern applies).

## Consequences

* V4 sizing is precise. `primary_block_bytes()` for V4-Pro at 128 tokens / block produces
  the byte count predicted by the paper (~2% of GQA8-BF16 at 1M context).
* The Sprint 0 path is untouched. `CkvDtype::Bf16` and `CkvDtype::Fp8E4m3` behave
  identically; V3 / MHA configs see no change.
* PyO3 surface: `tessera._native.CkvDtype` gains `Fp4E2m1` and `MixedBf16Fp8Fp4`. Python
  `BlockConfig.ckv_dtype` literal extends to `"fp4_e2m1" | "mixed_bf16_fp8_fp4"`.
* FP4 calibration tooling (`fp8_calibrate.py`) is generalisable to FP4 with one constant
  swap (`FP4_E2M1_MAX = 6.0` instead of `FP8_E4M3_MAX = 448.0`). That generalisation is
  Sprint 6 work.

## Why not a scalar `BlockConfig.precision_layout` field?

We considered modelling the layout as a separate `PrecisionLayout` struct parallel to
`CompressionScheme`. Rejected: the precision is intrinsic to V4's scheme definition —
RoPE BF16 / content FP8 / indexer FP4 is a fact about CSA, not an independent dimension.
Keeping it inside the scheme variant means a single `match` resolves both the structural
and precision facts.
