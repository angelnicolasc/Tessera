# ADR-0010 — Tiered LRU eviction policy

**Status**: Accepted  
**Sprint**: 1 (WS2)  
**Supersedes**: n/a  

---

## Context

Sprint 0 `allocate` returned `TesseraError::OutOfBlocks` when the manager's budget was
exhausted. In a multi-agent, long-context inference workload this is a fatal error: the GPU
KV cache is always finite, and the serving stack must gracefully reclaim capacity rather than
crash.

An eviction policy for MLA KV blocks must account for three dimensions that vanilla LRU
ignores:

1. **Sharing**: blocks shared across multiple requests (`ref_count > 1`) cannot be evicted
   without corrupting the other owners' KV state.
2. **Index membership**: blocks that are indexed in the segment index (HNSW layer) provide
   approximate-match value to future requests. Evicting them discards that value permanently
   (unless the request is re-prefilled).
3. **Orphaning**: blocks with `ref_count == 0` are already dead weight — the owning request
   has been released. They should be the first eviction candidates.

## Decision

Tessera uses a **tiered LRU** policy with four tiers ordered by eviction priority:

| Tier | Condition | Policy |
|------|-----------|--------|
| **a** | `ref_count == 0` (orphaned) | Evict immediately — no value, no risk |
| **b** | `ref_count == 1` AND `indexed == false` | LRU within pool — cheapest non-shared candidate |
| **c** | `ref_count == 1` AND `indexed == true` | LRU within pool — prefer to keep (index value) |
| **d** | `ref_count > 1` (shared) | **Never evict** — live owners would be corrupted |

Within each tier, the candidate with the smallest `last_touched` epoch (least recently used)
is preferred. Epochs are monotonic 64-bit counters incremented on every `primary_ptr` and
`rope_ptr` access; no wall-clock is involved, making the ordering deterministic and free of
platform time-source drift.

`BlockMeta` gains a `last_touched: Arc<AtomicU64>` field updated by `fetch_add` on each
access — the atomic is cheap on the common path and allows the eviction scan to read epochs
without holding any lock.

`allocate` attempts `evict_one()` exactly once on `OutOfBlocks` before returning the error.
`evict_one` acquires a read lock, scans all blocks, selects the best candidate by
tier+epoch, then releases the read lock and evicts via `free_block_internal`. This is O(n)
in the number of blocks but n is bounded by the GPU memory budget, which for a 80 GB H100
at DS-V3 sizing is ~180 000 blocks — scan completes in < 1 ms on modern hardware.

## Consequences

**Good**

- Shared blocks are unconditionally safe from eviction. No correctness risk from concurrent
  multi-agent sharing.
- Tier a catches leaked/orphaned blocks before they accumulate, acting as a passive GC pass
  on each allocation pressure event.
- The epoch-based LRU correctly identifies cold blocks in long-running pipelines (e.g., a
  batch of 128K-context requests where older prefix blocks are not accessed during decode).
- `tessera_evictions_total{tier}` labeled counter gives per-tier observability in Grafana.

**Trade-offs**

- Eviction is O(n) in blocks. For very large managers (> 500 K blocks) a dedicated
  eviction data structure (tiered min-heap) would be faster. Deferred to a future sprint
  if profiling shows eviction scan time in the critical path. Intentionally **not**
  ticketed (this idea was assigned "TD-021" in the original Sprint 1 draft; Sprint 3
  reassigned TD-021 to the `P2pCudaTransport` runtime body, so a new id will be filed if
  / when the optimisation becomes load-bearing).
- `evict_one` is called at most once per `allocate` failure. If the first evicted block does
  not free enough space (e.g., it was tier-c but the allocation needs a large contiguous
  region — not applicable since our blocks are fixed-size), the caller gets `OutOfBlocks`.
  For fixed-size blocks this is not an issue; each eviction frees exactly one block.
- Epochs are per-access, not per-token. A block that is read many times per request has a
  high epoch but is not necessarily "hot" from a caching perspective. This is acceptable:
  hotter blocks should indeed survive longer.

## Alternatives considered

- **Global LRU (no tiers)**: rejected because it would evict shared blocks, corrupting live
  requests.
- **Ref-count only (never evict ref_count>0)**: identical to tier d but without tier
  prioritisation. Rejected because it cannot distinguish an unindexed cold block (tier b)
  from an indexed warm block (tier c).
- **Random eviction among tier-a candidates**: simpler but ignores recency; would evict a
  just-orphaned block before an ancient one, reducing effective prefix hit rate.
- **Background eviction thread**: more complex, requires lock-based coordination with
  allocate. Deferred; the synchronous evict-on-pressure approach is correct for Sprint 1.
