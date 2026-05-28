# ADR-0014 — Multi-rank architecture: per-rank manager + RankTransport trait

**Status:** Accepted, 2026-05-21.

## Context

DeepSeek-V3 production runs TP=8 intra-node. MLA replicates `c_kv` across ranks because it
is small (68 KB/token vs 3.8 MB MHA) — communication beats sharding. The real coordination
problem is therefore *not* "shard the KV cache" but:

1. Each rank owns its own HBM region and block manager.
2. When rank-A seals a `c_kv` that rank-B will also see, cross-rank dedup avoids recompute.
3. Per-rank ownership must remain trivially mappable to global identity for share tables,
   distributed indexes, and PD-disaggregation.

Two alternatives we rejected:

* **Single global block manager with a `rank` field on each block.** Fans out lock
  contention; serialises allocate/free across ranks; doesn't match the physical
  ownership of HBM.
* **One block manager, share state via NCCL collectives.** Couples allocation latency to
  network RTT. Untenable on the decode hot path.

## Decision

* **Per-rank `TesseraBlockManager`.** Each owns:
  * `rank: RankId`
  * `world: Arc<World>` — shared topology description.
  * Its own contiguous device regions (primary, rope, optional FP8 scales).
* **`GlobalBlockId = (RankId, BlockId)`.** Constructed only at cross-rank boundaries.
* **`trait RankTransport`** with three implementations (ADR-0015). Block manager remains
  transport-agnostic — it doesn't know whether a peer is reached via NVLink or NCCL.
* **Backward compatible.** `TesseraBlockManager::new(config, memory_bytes)` retained as a
  singleton-world convenience (`RankId::ZERO`, `World::singleton()`). The new
  `new_with_world(config, memory_bytes, rank, world)` is the explicit multi-rank path. Tests
  Sprint 0-2 keep working unchanged.

## Consequences

* API surface grows by one constructor + a handful of accessors (`rank()`, `world()`,
  `global_id()`, `transfer_request_to_rank()`). All existing public methods unchanged.
* Cross-rank events flow through `Arc<dyn RankTransport>` — virtual dispatch cost is
  measured in nanoseconds; dwarfed by NVLink RTT (~10 µs) or NCCL RTT (~100 µs+).
* The PyO3 layer ships `PyRankId`, `PyWorld`, and `BlockManager.with_world` as additive
  surface; no Python caller has to touch them until they go multi-rank.
* PD-disaggregation hook (ADR-0016) sits naturally on top of the transport trait — no
  separate abstraction layer needed.

## Deviation from initial plan

The initial Sprint 3 plan called for renaming `TesseraBlockManager::new` → `new_singleton`
and introducing `new` as the rank-aware constructor. Pragmatic deviation in execution:
**kept `new()` as the singleton path** and added `new_with_world()` as the explicit
multi-rank constructor. This minimises churn across the existing 10+ test files for zero
semantic difference; we accept that the function name is slightly less explicit in
exchange for a clean diff.
