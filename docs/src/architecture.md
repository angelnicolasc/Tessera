# Architecture

> **Current status (v0.6.0-sprint5)**: this page reflects the surface as of Sprint 5.
> Components labelled *cloud-burst* are wired in the type system but their runtime bodies
> are gated on a future GPU session. ADR history is preserved under
> [Architecture Decision Records](adr/0001-block-size-64.md) — pages here describe the
> current shape, ADRs describe how it got there.

```text
                ┌───────────────────────────────────────────────────────────────────┐
                │                  Tessera Block Manager (Rust)                     │
                │                                                                   │
vLLM V1 ──────► │   MLA / V4 Block Allocator                                        │
                │     block_size_tokens: 64 (MLA / FlashMLA-native)                 │
                │                       128 (V4 hybrid = lcm(k1=4, k2=128))         │
                │     primary region dtype:                                         │
                │       MLA: BF16 or FP8 E4M3                                       │
                │       V4 : MixedBf16Fp8Fp4 (RoPE BF16 + content FP8 + indexer FP4)│
                │     k_rope: always BF16                                           │
                │     content hash: xxhash3 over primary bytes only                 │
                │     schemes: single (MLA / MHA) OR per-layer Vec (V4 hybrid)      │
                │                                                                   │
                │   Lifecycle + Eviction                                            │
                │     release_request(req_id) — O(owned) atomic teardown            │
                │     evict_one() tiered LRU: orphaned → cold unindexed             │
                │                              → cold indexed; shared never evicted │
                │                                                                   │
                │   Segment Index (async, off hot-path)                             │
                │     Layer 1: xxhash3 — O(1), sync                                 │
                │     Layer 2: usearch HNSW — async, ~500μs P99 budget              │
                │     DistributedSegmentIndex: tier-aware fan-out budget            │
                │                                                                   │
                │   Cross-Agent Share Table                                         │
                │     req_id ↔ block_id, ref-counted, copy-on-write                 │
                │                                                                   │
                │   State Cache (per-request arena, V4)                             │
                │     SWA tokens + uncompressed tail; fixed size; arena-allocated   │
                │                                                                   │
                │   Cross-rank coordination                                         │
                │     trait RankTransport: MockTransport | P2pCudaTransport ✦      │
                │                          | NcclTransport ✦✦                      │
                │     LatencyInjector: chaos rig (tier latency + jitter + drops)    │
                │     reserve-then-stream PD-disagg: prepare → stream → commit      │
                │                                                                   │
                │   Storage backends (trait DeviceBackend)                          │
                │     CpuMockBackend (default, deterministic)                       │
                │     CudaBackend ✦ (feature `cuda`)                                │
                │     DiskBackend (on-disk KV tier, 3 SWA strategies)               │
                │                                                                   │
                │   Observability                                                   │
                │     7 + 7 Prometheus families (block + rank/transport)            │
                │     OTLP tracing: Python (init_tracing) + Rust (feature           │
                │                   `otel-rust`, ADR-0017) on same endpoint         │
                └────────────────────────────┬──────────────────────────────────────┘
                                             │
                ┌────────────────────────────▼──────────────────────────────────────┐
                │              Kernel Dispatch                                       │
                │                                                                    │
                │   SM ≥ 9.0  → FlashMLA (paged, BF16/FP8)                          │
                │   Ampere+   → FlashInfer MLA backend                              │
                │   V4 ✦      → DeepSeek TileLang ref (Lightning Indexer + CSA/HCA) │
                │   Fallback  → vLLM Triton MLA                                     │
                │                                                                    │
                │   Tessera supplies: block pointers, per-region byte layouts,      │
                │   reservation tokens — kernels consume directly.                  │
                └────────────────────────────────────────────────────────────────────┘
```

✦ runtime body cloud-burst gated.   ✦✦ runtime body Sprint 6+.

## Block Layout — MLA (V3 / Kimi-K2)

Each block carries `block_size_tokens=64` tokens of MLA latents across all layers, plus a
64-byte cache-line-aligned header.

| Region | Size (DeepSeek-V3, BF16) | Size (DeepSeek-V3, FP8 `c_kv`) |
|---|---|---|
| Header | 64 B | 64 B |
| `c_kv` | 61 × 64 × 512 × 2 = 3.82 MB | 1.91 MB |
| `k_rope` | 61 × 64 × 64 × 2 = 501 KB | 501 KB (always BF16) |
| FP8 scales | — | 61 × 4 = 244 B |
| **Total / block** | **≈ 4.31 MB** | **≈ 2.40 MB** |

`k_rope` is never quantised to FP8; the position-dependent RoPE path is precision-sensitive
at long context (see [ADR-0007](adr/0007-fp8-calibration-required.md)).

## Block Layout — V4 hybrid (CSA + HCA + SWA)

V4 uses **per-layer schemes** interleaved across layers (see
[ADR-0021](adr/0021-per-layer-schemes.md)). Block size is `lcm(k1, k2) = 128` original
tokens; each block carries 32 CSA entries + 1 HCA entry per CSA/HCA layer. SWA tokens
and uncompressed-tail tokens live in the per-request State Cache
([ADR-0023](adr/0023-state-cache.md)), not in the paged block pool.

Per-token-per-layer storage cost (verified against paper §2.3.4):

| Scheme | Compression | Per-token bytes/layer | Composition |
|---|---|---|---|
| CSA  | `k1 = 4` | **160 B** | `(64·BF16 + 448·FP8 + 128·FP4)/4` |
| HCA  | `k2 = 128` | **4 B** | `(64·BF16 + 448·FP8)/128` |
| SWA  | uncompressed (`win=128`) | 576 B (in State Cache) | `64·BF16 + 448·FP8` |

The on-disk tier ([ADR-0024](adr/0024-disk-backend.md)) extends this layout to persistent
storage with three SWA caching strategies (Full / Periodic / Zero).

## Crate Layout

| Crate | Role | Hot-path? |
|---|---|---|
| `tessera-core` | Block manager, share table, lifecycle, eviction, state cache, transport, metrics, OTLP bridge, `DeviceBackend` (CpuMock / Cuda / Disk) | Yes (Rust) |
| `tessera-index` | `IndexBackend` trait, `UsearchIndex` HNSW, `DistributedSegmentIndex` with tier-aware budgets | Off-hot-path |
| `tessera-py` | PyO3 module `tessera._native` — boundary types only | Boundary |

## Extension Hooks

Four non-exhaustive seams keep future work additive without breaking the public surface:

1. `CompressionScheme` (`#[non_exhaustive]`): `MlaLatent`, `MhaFull`, `V4Csa`, `V4Hca`,
   `V4Swa` are sibling variants. `DsaHierarchical` is retained `#[deprecated]` for
   migration only. See [ADR-0020](adr/0020-v4-hybrid-attention.md);
   [ADR-0004](adr/0004-compression-scheme-enum.md) is the historical record of why the
   enum exists.
2. `IndexBackend`: pluggable ANN backend behind a 5-method trait. See
   [ADR-0005](adr/0005-index-backend-trait.md).
3. `KernelBackend` enum: dispatch target. `FLASH_ATTN4` is an experimental stub that
   raises `NotImplementedError` until the upstream backend matures; the V4 TileLang
   reference impl gets the same stub-then-implement treatment. See
   [Kernel Dispatch](kernel_dispatch.md).
4. `RankTransport` trait (Sprint 3+): three impls (`MockTransport`, `P2pCudaTransport`,
   `NcclTransport`) cover in-process tests, intra-node NVLink, and multi-node IB
   respectively. See [ADR-0015](adr/0015-p2p-vs-nccl-transport.md).

## Related pages

- [DeepSeek-V4 Compliance](v4_compliance.md) — gap summary vs the V4 paper preview.
- [Multi-GPU (Tensor Parallelism)](multi_gpu.md) — rank-aware coordination + reserve-then-stream.
- [Request Lifecycle](lifecycle.md), [Eviction Policy](eviction.md).
- [Chaos Testing](chaos.md) — `LatencyInjector` + proptest + hypothesis layers.
- [Testing](testing.md), [Benchmarks](benchmarks.md), [FAQ](faq.md).
