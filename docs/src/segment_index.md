# Segment Index

Two layers. Both content-addressed over `c_kv` bytes (never `k_rope` — position-dependent).

## Layer 1 — Exact xxhash3 (synchronous)

* O(1) lookup.
* Lives in the block manager's `content_index: DashMap<u64, BlockId>`.
* Sub-microsecond on a hot cache line.
* Catches identical token sequences — the common case in multi-agent pipelines that share a
  document.

## Layer 2 — usearch HNSW (asynchronous, off hot path)

* Indexes the *mean* of `c_kv` across `(layer, token)` → `[d_c]` descriptor.
* Lives in `tessera-index` behind the `IndexBackend` trait
  ([ADR-0005](adr/0005-index-backend-trait.md)).
* Executes in a dedicated thread-pool task — never on the prefill or decode critical path.
* **Latency budget**: 500 µs by default. Exceeding the budget returns `None`, which is a
  correctness-preserving miss (the request just computes its own `c_kv`).

## Why usearch and not FAISS

usearch supports incremental `remove` without rebuilding the index. FAISS HNSW requires a
full rebuild on delete. KV cache eviction patterns are incompatible with full rebuild —
this constraint is architectural, not a performance preference.

## Failure modes

| Event | Behaviour | Cost |
|---|---|---|
| Layer 1 hit | block re-used, ref-count++ | sub-µs |
| Layer 1 miss → Layer 2 hit | block re-used after async resolve | ≤ budget |
| Layer 2 timeout | block computed fresh; metric incremented | budget µs |
| Layer 2 miss | block computed fresh | budget µs |

The HNSW path is never on the decode critical path. It is consulted at prefill, where the
budget is more forgiving and the result delivered via `oneshot` channel.
