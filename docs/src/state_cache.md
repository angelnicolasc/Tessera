# State Cache (V4 per-request arena)

DeepSeek-V4 splits the KV cache into two tiers (paper §3.5.1):

1. **KV Cache** — the paged block pool the block manager has owned since Sprint 0. Holds
   compressed CSA / HCA entries.
2. **State Cache** — fixed-size arena per request. Holds:
   - SWA KV entries for the most recent `win = 128` tokens (uncompressed).
   - Uncompressed tail tokens awaiting CSA / HCA compression (less than `k1` or `k2`
     tokens accumulated since the last compression boundary).

The State Cache is structurally different from the block pool:

| Property | Block pool (KV cache) | State Cache (V4) |
|---|---|---|
| Lifetime | Per block; reusable across requests via dedup + share | Per request; allocated on first use, freed on `release_request` |
| Sizing | Variable (paged) | Fixed per request |
| Content-addressed sharing | Yes (`xxh3` + HNSW) | No — position-dependent storage |
| Eviction | Tiered LRU ([ADR-0010](adr/0010-eviction-policy.md)) | None (arena lifetime tied to request) |

Trying to model V4 SWA and tail tokens with the block pool fails on every front:
ref-counting is unnecessary, eviction tiers are irrelevant, content hash doesn't apply.
That's why Sprint 5 introduces a sibling component.

## API surface

```rust
use tessera_core::{StateCache, StateCacheConfig, CpuMockBackend, RequestArena};

let cfg = StateCacheConfig::for_v4(
    /* head_dim            */ 512,
    /* rope_dim            */ 64,
    /* num_layers          */ 61,
    /* win                 */ 128,
    /* max_uncompressed_tail */ 127,
    /* max_requests        */ 8,
);
let sc = StateCache::new(cfg, CpuMockBackend::new()).unwrap();

let (swa_ptr, tail_ptr) = sc.allocate_for_request(req_id)?;
// ... write SWA + tail bytes via the backend ...
sc.release_request(req_id);
```

`for_v4(...)` derives sizing from the V4 paper constants directly. For V4-Pro
(`head_dim=512`, `rope_dim=64`, `num_layers=61`, `win=128`) the SWA region is
`576 × 61 × 128 = 4,497,408` bytes per request.

## `RequestArena` trait

Sprint 5 abstracts the per-request arena pattern behind a minimal trait so future
request-scoped allocators (e.g. Mamba state, speculative-decode shadow caches) plug in
without code duplication:

```rust
pub trait RequestArena: Send + Sync {
    fn acquire(&self, req_id: u64) -> Result<(DevicePtr, DevicePtr)>;
    fn release(&self, req_id: u64) -> bool;
    fn name(&self) -> &'static str;
}
```

`StateCache` is the Sprint 5 concrete impl. Object-safety is verified by
`crates/tessera-core/src/state_cache.rs::tests::request_arena_trait_is_object_safe`.

## Lifecycle semantics

- `allocate_for_request(req_id)` is idempotent — calling twice for the same request
  returns the same pointer pair (no double-allocation).
- `release_request(req_id)` returns `bool` indicating whether a slot was actually freed
  (idempotent for unknown requests).
- Pool exhaustion returns `TesseraError::OutOfBlocks { used, total }` matching the
  block-manager contract.

## What's not yet wired

The block manager's `release_request` (Sprint 1, ADR-0009) and the State Cache's
`release_request` are currently independent calls. The vLLM plugin will tie them
together in Sprint 6+ so a single `TesseraBlockAllocator.free(block)` on a request's
last block also reaps its State Cache slot (TD-037).

See [ADR-0023](adr/0023-state-cache.md) for the design rationale.
