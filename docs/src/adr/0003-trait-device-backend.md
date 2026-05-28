# ADR-0003 — `trait DeviceBackend` with CPU mock and CUDA impls

**Status:** Accepted, 2026-05-21.

## Context

Tessera must run two ways:

1. On developer workstations and CI runners that may not have a CUDA GPU — for unit tests,
   integration tests, type checking, and the cross-agent sharing benchmark.
2. On real H100/H800/B200 hardware in production — backed by `cudarc`.

Hard-coupling the block manager to CUDA would mean:

* No deterministic Rust tests (CI tests would require GPU runners).
* The cross-agent share table and segment-index logic couldn't be exercised end-to-end on
  Windows / macOS developer machines.

Hard-coupling everything to a CPU mock would lose access to real GPU semantics.

## Decision

Block manager is generic over `B: DeviceBackend`. Implementations:

* `CpuMockBackend` — `Vec<u8>` per region; deterministic; the default for `TesseraBlockManager::new`.
* `CudaBackend` — `cudarc`-backed; feature-gated behind `--features cuda`.

The trait surface is intentionally narrow: `alloc_region`, `memcpy`, `read_bytes`,
`fill_pattern`, `name`. The hot path through the block manager is `unsafe`-free.

## Consequences

* `cargo test --workspace` runs anywhere. CI matrix doesn't need GPU runners until Sprint 1.
* Two PyO3 classes exposed: `BlockManager` (CPU mock) and `BlockManagerCuda` (feature-gated).
* `read_bytes` performs a `dtoh` on real CUDA. That cost dominates `seal()` at production
  block-rates and is tracked by [ADR-0008](0008-hnsw-off-hot-path.md): a future on-device
  hash kernel removes the transfer.
