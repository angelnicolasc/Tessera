# ADR-0020 — DeepSeek-V4 hybrid attention: CSA + HCA + SWA

**Status:** Accepted, 2026-05-28. Supersedes the placeholder semantics in ADR-0004.

## Context

DeepSeek-V4 (paper §2.3, May 2026) ships a hybrid attention architecture that **violates
the assumptions behind PagedAttention** by design (§3.5.1 of the paper, verbatim):

> "The hybrid attention mechanism violates fundamental assumptions behind PagedAttention
> and its variants."

This is the exact problem Tessera was built for. The Sprint 0 design carried a
placeholder `CompressionScheme::DsaHierarchical { coarse_dim, fine_dim, swa_window }`
based on guesses (the internal `reversion-tessera02.md` draft) that turned out wrong in
every detail: the real V4 has three distinct attention types interleaved per layer, not
two compression dims plus an SWA window.

The published V4 architecture comprises:

| Component | Compression | Selection | Storage |
|---|---|---|---|
| **CSA** (Compressed Sparse Attention) | `k1 = 4` tokens → 1 entry (overlapping windows) | DSA top-k via Lightning Indexer (FP4) | Paged KV cache |
| **HCA** (Heavily Compressed Attention) | `k2 = 128` tokens → 1 entry | Full attention over compressed entries | Paged KV cache |
| **SWA** (Sliding Window) | Uncompressed | Most recent `win = 128` tokens | State Cache (per-request arena) |

Three engineering implications:

1. **Per-layer schemes** — V4 alternates CSA / HCA per layer with optional SWA branch.
   Tessera's single-`scheme` `MlaBlockConfig` doesn't model this. (Resolved in ADR-0021.)
2. **Mixed precision per region** — BF16 RoPE + FP8 content + FP4 indexer co-resident in
   each compressed entry. (Resolved in ADR-0022.)
3. **Two-tier KV cache** — State Cache (per-request, fixed) + paged KV cache (block pool).
   (Resolved in ADR-0023.)

## Decision

Three new `CompressionScheme` variants added to the `#[non_exhaustive]` enum:

```rust
CompressionScheme::V4Csa {
    k1: u32,                  // 4
    head_dim: u32,            // 512
    num_heads: u32,           // 64 Flash / 128 Pro
    rope_dim: u32,            // 64 (trailing BF16 dims)
    indexer_head_dim: u32,    // 128
    num_indexer_heads: u32,   // 64
    top_k: u32,               // 512 Flash / 1024 Pro
}

CompressionScheme::V4Hca {
    k2: u32,                  // 128
    head_dim: u32,            // 512
    num_heads: u32,
    rope_dim: u32,            // 64
}

CompressionScheme::V4Swa {
    window: u32,              // 128
    head_dim: u32,            // 512
    num_heads: u32,
    rope_dim: u32,            // 64
}
```

`CompressionScheme::DsaHierarchical` is marked `#[deprecated]` but retained for migration —
construction is permitted (with a compiler warning), but `MlaBlockConfig::new` rejects it
with an error message pointing at V4Csa / V4Hca / V4Swa.

A new self-describing method `bytes_per_token_per_layer()` computes V4 storage cost
honouring the variant's internal mixed precision. The legacy
`primary_bytes_per_token(num_layers, dtype_bytes)` continues to work for V3 MLA + MHA;
for V4 schemes the `dtype_bytes` parameter is ignored and accounting goes through the
new method.

## Consequences

* Tessera now natively models V4. CPU-side accounting (`primary_block_bytes`,
  `compression_ratio_vs_mha_bf16`) yields correct numbers for `deepseek_v4_flash.toml`
  and `deepseek_v4_pro.toml` (added in WS-V4-F).
* The `#[non_exhaustive]` foundation pays off again — every `match` site on
  `CompressionScheme` was caught by the compiler and updated with the new arms. Zero
  silent behaviour drift.
* The migration path for callers using `DsaHierarchical` is explicit: the deprecation
  attribute surfaces the rename at compile time, and runtime panics include the migration
  pointer.
* Sprint 5's V4 compliance is the **block layout layer only**. The actual V4 kernels
  (Lightning Indexer FP4, CSA top-k selection, HCA core attention) live in DeepSeek's
  TileLang reference implementation and integrate via the kernel dispatch layer in a
  future sprint.

## What this doesn't include

* **Lightning Indexer kernel** — model-architecture component (§2.3.1). Out of scope for
  the block manager; lives in the model itself or in the kernel backend.
* **Top-k selector kernel** — same.
* **Manifold-Constrained Hyper-Connections (mHC)** — training-time technique; no
  inference-time storage implication.
* **Muon optimizer** — training only.
* **MoE expert parallelism** — orthogonal scheduler concern.
