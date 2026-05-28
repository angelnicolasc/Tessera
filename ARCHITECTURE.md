# Architecture

This is the short, scannable architecture overview for Tessera as of
`v0.6.0-sprint5`. The long-form version lives in `docs/src/` and is built with
`mdbook build docs`; start at `docs/src/architecture.md` for diagrams, math,
component pages, and ADR links.

## Current Scope

Tessera is a KV block manager and memory hierarchy for MLA-style inference. It
does not ship an attention kernel. Its job is to allocate, account for, share,
evict, persist, and expose the correct storage layout to upstream kernels such
as FlashMLA, FlashInfer, Triton, and the future DeepSeek-V4 TileLang reference
backend.

Current implementation status:

| Surface | Status |
|---|---|
| DeepSeek-V3 / Kimi-K2 MLA layout | Implemented; CPU-validated |
| DeepSeek-V4 CSA / HCA / SWA block layout | Implemented; CPU-validated |
| V4 mixed precision accounting (BF16 + FP8 + FP4) | Implemented; CPU-validated |
| Per-layer V4 schemes | Implemented; CPU-validated |
| State Cache for V4 SWA + tail tokens | Implemented as an independent arena; plugin lifecycle integration pending |
| DiskBackend V4 persistence layer | Implemented as a filesystem-backed `DeviceBackend`; mmap + block-manager spillover pending |
| Multi-rank coordination over MockTransport | Implemented; CPU chaos-tested |
| P2pCuda / NCCL runtime bodies | API wired; runtime validation pending cloud-burst / Sprint 6+ |
| GPU kernel parity and real vLLM-engine integration | Harness wired; GPU-gated |

## Layers

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ vLLM V1 engine                                                              │
│   scheduler / executor / plugin discovery                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ TesseraBlockAllocator plugin                                                │
│   allocate · post_prefill_seal · find_shared_prefix · free · transfer       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Python orchestration                                                        │
│   config.py              pydantic v2 model + V4 / disk-cache config         │
│   segment_index.py       exact xxh3 + async HNSW lookup                     │
│   kernel_dispatch.py     FlashMLA / FlashInfer / Triton / FA4 stub selector │
│   fp8_calibrate.py       per-layer FP8 calibration harness                  │
│   multi_rank.py          CPU multi-rank coordinator for tests               │
│   reference/             FP32 PyTorch oracle for GPU parity tests           │
├─────────────────────────────────────────────────────────────────────────────┤
│ PyO3 boundary                                                               │
│   tessera._native exposes block manager, config, transport, and index APIs  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Rust core                                                                   │
│   tessera-core                                                              │
│     block_manager      allocation, seal/dedup, CoW, eviction, lifecycle     │
│     config             MLA / MHA / V4 schemes and byte accounting           │
│     state_cache        V4 per-request SWA + tail arena                      │
│     device             CpuMock, Cuda, Disk backends                         │
│     transport          Mock, P2pCuda stub, NCCL stub, LatencyInjector       │
│     observability      metrics + optional otel-rust bridge                  │
│   tessera-index                                                              │
│     UsearchIndex + DistributedSegmentIndex                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Storage Model

### V3 / MLA

For DeepSeek-V3-style MLA, each block stores `block_size_tokens = 64` tokens
across all layers:

| Region | Meaning |
|---|---|
| `c_kv` | Position-independent latent content, BF16 or calibrated FP8 |
| `k_rope` | Position-dependent RoPE component, always BF16 |
| FP8 scales | Optional per-layer scale factors for FP8 `c_kv` |

The block manager hashes stored `c_kv` bytes for exact deduplication and uses an
async HNSW layer for approximate segment lookup off the hot path.

### V4 / Hybrid

For DeepSeek-V4, Tessera models the block layout and accounting layer:

| Scheme | Storage owner | Accounting |
|---|---|---|
| CSA | Paged KV block pool | `(64*BF16 + 448*FP8 + 128*FP4) / k1`, with `k1 = 4` |
| HCA | Paged KV block pool | `(64*BF16 + 448*FP8) / k2`, with `k2 = 128` |
| SWA | State Cache | Uncompressed `64*BF16 + 448*FP8`, capped by request window |

`MlaBlockConfig` supports either a homogeneous scheme or a per-layer
`schemes_per_layer` vector. The V4 configs use `block_size_tokens = 128`, the
`lcm(k1, k2)` required by the active compression boundaries.

## Request Lifetime

`TesseraBlockManager` tracks private block ownership with a reverse index:

```text
req_id -> Vec<BlockId>
```

`release_request(req_id)` atomically tears down all private blocks for a request.
Cross-agent sharing is handled separately by `CrossAgentShareTable`, which owns
shared-reference bookkeeping and returns blocks to the manager only when the last
owner releases them.

For V4, `StateCache` has its own request arena and independent
`release_request(req_id)`. Wiring block-manager release and State Cache release
into one vLLM plugin lifecycle operation is tracked as Sprint 6+ integration
work.

## Distributed Path

Tessera uses one block manager per rank. Cross-rank behavior is transport
abstracted:

| Transport | Purpose | Status |
|---|---|---|
| `MockTransport` | CPU tests, multi-rank coordinator, chaos testing | Implemented |
| `LatencyInjector<T>` | Deterministic latency/drop wrapper for any transport | Implemented |
| `P2pCudaTransport` | Intra-node NVLink / CUDA IPC path | API wired; runtime body cloud-burst gated |
| `NcclTransport` | Multi-node InfiniBand / NCCL path | Feature-gated stub; runtime Sprint 6+ |

Prefill/decode disaggregation uses the reserve-then-stream protocol from
ADR-0018:

```text
reserve capacity on destination
stream block payloads
commit by releasing source request
abort by releasing destination reservation
```

The CPU MockTransport path exercises these semantics under proptest and
Hypothesis chaos suites before the real GPU transports land.

## Extension Points

Tessera keeps four load-bearing seams intentionally narrow:

| Seam | Purpose |
|---|---|
| `DeviceBackend` | CPU mock, CUDA, and Disk storage through one alloc/read/write contract |
| `CompressionScheme` | MLA, MHA fallback, and V4 CSA/HCA/SWA accounting |
| `IndexBackend` | Exact + approximate segment lookup without tying the core to one ANN implementation |
| `RankTransport` | Mock, CUDA P2P, NCCL, and latency-injected transport behavior |

The historical `DsaHierarchical` placeholder remains only for migration. V4's
real public semantics are the sibling `V4Csa`, `V4Hca`, and `V4Swa` variants
documented in ADR-0020 through ADR-0024.

## Decision Log

The mdBook contains 24 ADRs. The most important current ones are:

| ADR | Subject |
|---|---|
| [0001](docs/src/adr/0001-block-size-64.md) | MLA block size and FlashMLA-native paging |
| [0007](docs/src/adr/0007-fp8-calibration-required.md) | FP8 calibration and long-context precision guardrails |
| [0009](docs/src/adr/0009-per-request-lifecycle.md) | Per-request lifecycle via reverse index |
| [0010](docs/src/adr/0010-eviction-policy.md) | Tiered LRU eviction |
| [0018](docs/src/adr/0018-reserve-then-stream-pd-disagg.md) | Transactional PD-disaggregation |
| [0019](docs/src/adr/0019-latency-injection-chaos-testing.md) | Latency injection and chaos testing |
| [0020](docs/src/adr/0020-v4-hybrid-attention.md) | DeepSeek-V4 hybrid attention model |
| [0021](docs/src/adr/0021-per-layer-schemes.md) | Per-layer schemes in `MlaBlockConfig` |
| [0022](docs/src/adr/0022-mixed-precision-per-region.md) | V4 mixed precision per region |
| [0023](docs/src/adr/0023-state-cache.md) | V4 State Cache |
| [0024](docs/src/adr/0024-disk-backend.md) | V4 DiskBackend |
