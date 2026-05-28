# Introduction

> PagedAttention was designed for MHA. DeepSeek's MLA compresses KV cache **56.9×** (V3);
> DeepSeek-V4 (May 2026 paper preview) introduces hybrid CSA + HCA + SWA attention with a
> two-tier paged + state cache and explicitly states that the hybrid scheme *"violates
> fundamental assumptions behind PagedAttention and its variants"* (§3.5.1). Mainstream
> serving stacks in 2026 still allocate and evict blocks sized for MHA. **Tessera is the
> block manager that knows what MLA and V4 actually store** — it owns the layout +
> accounting layer and mounts on upstream kernels instead of reinventing them.

## What Tessera Is

Tessera is a **block manager**, not a kernel. It owns the on-device memory that holds a
DeepSeek-style KV cache (V3 MLA or V4 hybrid) and exposes it to attention kernels
(FlashMLA on Hopper / Blackwell, FlashInfer on Ampere+, Triton fallback, future TileLang
V4) through paged page tables and per-region byte layouts they consume directly.

For **V3 / Kimi-K2 (MLA)** it adds three things mainstream serving stacks generally lack:

1. **MLA-aware block sizing.** Blocks are 64 tokens deep — FlashMLA's native paged block
   size — and structured around `c_kv` and `k_rope` rather than full K/V tensors.
2. **Content-addressed `c_kv` sharing.** Because `c_kv` is position-independent (the
   decoupled RoPE component lives separately in `k_rope`), two requests that process the
   same document at *different* prompt positions can share `c_kv` blocks. We have not
   found this implemented in mainstream open-source serving stacks (RadixAttention, vLLM
   APC, SGLang) as of 2026 — they hash a token prefix that includes position context.
3. **Ref-counted copy-on-write** for safe multi-agent sharing without copy overhead.

For **DeepSeek-V4 (hybrid)** Sprint 5 adds:

4. **Per-layer compression schemes.** `MlaBlockConfig::with_per_layer_schemes` accepts a
   `Vec<CompressionScheme>` so CSA, HCA and SWA layers interleave per the paper's
   `layer_pattern`. See [ADR-0021](adr/0021-per-layer-schemes.md).
5. **State Cache.** Per-request fixed-size arena for SWA + uncompressed-tail tokens.
   Structurally separate from the paged block pool (different lifetime, no
   content-addressed sharing). See [ADR-0023](adr/0023-state-cache.md) and the
   [State Cache](state_cache.md) component page.
6. **On-disk KV tier.** `DiskBackend` implements `DeviceBackend` over a filesystem-backed
   region store, with the three SWA caching strategies (Full / Periodic / Zero) the
   paper proposes. See [ADR-0024](adr/0024-disk-backend.md) and
   [Disk Backend](disk_backend.md).

## What Tessera Is Not

* **Not a kernel.** FlashMLA, FlashInfer and the V4 TileLang reference impl are open
  source and well-maintained upstream. Tessera mounts on them.
* **Not a fork of vLLM.** Tessera registers as a vLLM V1 `BlockAllocator` plugin.
* **Not an MLA implementation.** Tessera assumes the model already uses MLA / V4; it
  does not add either to an MHA model (`mha_fallback.toml` is a degraded compatibility
  mode, not feature parity).

## Repository tour

| Path | Contents |
|---|---|
| `crates/tessera-core` | Rust block manager + state cache + transport + observability. Generic over `DeviceBackend` (CPU mock / CUDA / Disk). |
| `crates/tessera-index` | `IndexBackend` trait + `usearch` HNSW + `DistributedSegmentIndex`. |
| `crates/tessera-py` | PyO3 bindings: `tessera._native`. |
| `python/tessera` | Python orchestration: config (pydantic v2), segment index, kernel dispatch, vLLM plugin (rank-aware), FP8 calibration, multi-rank coordinator. |
| `models/*.toml` | Reference configs (DeepSeek-V3, V4-Flash, V4-Pro, Kimi-K2, MHA fallback). |
| `docs/src/adr` | 24 Architecture Decision Records — read these next. |
| `Research/DeepseekV4/gap-analysis.md` | Paper-vs-implementation audit. |
