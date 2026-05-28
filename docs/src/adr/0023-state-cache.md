# ADR-0023 — State Cache: per-request arena for V4 SWA + tail tokens

**Status:** Accepted, 2026-05-28.

## Context

V4 (paper §3.5.1) splits the KV cache into **two tiers**:

1. **KV Cache** — the paged block pool Tessera already manages. Holds compressed CSA /
   HCA entries.
2. **State Cache** — per-request fixed-size arena. Holds:
   - SWA KV entries for the most recent `win = 128` tokens (uncompressed).
   - Uncompressed tail tokens awaiting CSA / HCA compression (less than `k1` or `k2`
     tokens accumulated since the last compression block).

The State Cache is structurally different from the block pool:

* **Per-request lifetime.** Allocated on first use; freed atomically on
  `release_request`.
* **Fixed size per request.** No paging, no eviction within a request — one contiguous
  arena per request.
* **No content-address sharing.** Both SWA entries and tail tokens are position-bound;
  the xxh3 dedup pipeline doesn't apply.

Trying to model this with the existing block pool fails on every front: ref-counting
(unnecessary), eviction tiers (irrelevant), content hash (position-dependent storage).

## Decision

New `state_cache.rs` module in `tessera-core` with:

* `StateCacheConfig { swa_bytes_per_request, tail_bytes_per_request, max_requests }` —
  parametric sizing. `for_v4(head_dim, rope_dim, num_layers, win, max_uncompressed_tail,
  max_requests)` derives a config from the V4 paper constants.
* `StateCache<B: DeviceBackend>` — pre-allocates `max_requests * (swa + tail)` bytes on
  `B` at construction; lends slot indices in O(1) via a `Mutex<Vec<u32>>` free-list
  matching the block manager's contention model.
* `RequestArena` trait — minimal `acquire(req_id) -> (swa_ptr, tail_ptr)` /
  `release(req_id)` surface. Forward-looking: future arches (Mamba state, KV projection
  scratch) plug in as `RequestArena` impls.

`StateCache` is **independent** of the block manager. The block manager remains the
sole owner of the compressed KV blocks; the state cache is the sole owner of the
per-request arena. Sprint 5 doesn't wire them together at the orchestration layer —
that's an integration concern the Python plugin or kernel dispatcher handles when the
real V4 kernels arrive.

## Consequences

* V4's State Cache is now a first-class component, sized correctly from the paper's
  constants.
* The trait surface (`RequestArena`) is intentionally tiny — only `acquire` /
  `release` / `name`. Future request-scoped allocators (mHC scratch buffers,
  speculative-decode shadow caches) inherit the contract without code duplication.
* CPU mock tests verify lifecycle, OOM behaviour, and configuration derivation. The
  real-GPU path uses the same `DeviceBackend` abstraction; nothing in `StateCache`
  changes between CPU mock and CUDA.
* Sprint 5 ships sizing + lifecycle. Sprint 6+ will add:
  - Eviction of the entire State Cache when a request migrates between ranks (interacts
    with ADR-0018 PD-disagg).
  - Disk overflow for SWA when running ultra-long requests (composes with ADR-0024).
