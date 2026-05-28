# ADR-0009 — Per-request block lifecycle via reverse index

**Status**: Accepted  
**Sprint**: 1 (WS1)  
**Supersedes**: n/a  

---

## Context

Sprint 0 block manager tracked blocks via a global free-list and a `DashMap<BlockId, BlockMeta>`
forward index. When a request completed or was cancelled, its private blocks had to be freed
individually by the caller — but there was no record of *which* block IDs belonged to a given
request. In practice this meant:

- A crashed vLLM worker left orphaned blocks; the only recovery was a full manager restart.
- The `CrossAgentShareTable` only tracked *shared* blocks, leaving private blocks (prefix and
  decode) entirely unaccounted for at the request granularity.
- The vLLM V1 `free(block)` protocol delivered one block ID per call; the plugin had to call
  `free` in a loop, one block at a time, which made the hot path O(blocks-per-request) in
  Python instead of O(1) FFI calls.

## Decision

Add a reverse index `req_blocks: DashMap<u64, Vec<BlockId>>` to `TesseraBlockManager`. The
invariants are:

1. `allocate(req_id, ...)` inserts the new `block_id` into `req_blocks[req_id]`.
2. `free(block_id)` removes the entry from `req_blocks` before dropping the block.
3. A new method `release_request(req_id) -> u32` atomically removes the request's entry,
   calls `free(block_id)` for each owned block, and returns the count freed. This is the
   canonical teardown path.
4. Shared blocks are *not* tracked in `req_blocks`; the `CrossAgentShareTable` retains
   ownership of cross-agent references. This preserves the single-responsibility split
   established in Sprint 0.

The vLLM plugin's `free(block)` is rewired to call `manager.release_request(req_id)` —
one FFI call regardless of how many blocks the request owned.

## Consequences

**Good**

- Crash safety: any process that calls `release_request` on shutdown frees all private blocks
  in a single operation. Request-level leaks become impossible unless the *process* dies.
- O(1) vLLM `free` call from Python (single PyO3 crossing).
- Eviction can use the reverse index to verify ownership before force-freeing a block (see
  ADR-0010).
- Counter `tessera_request_releases_total` provides request-lifecycle observability at zero
  additional overhead.

**Trade-offs**

- Extra `DashMap` write on every `allocate`. Benchmark `share_table_concurrency` shows this
  is dominated by the block allocation itself; net overhead < 2% in microbench.
- Shared blocks (canonical from dedup) are *not* auto-freed via `release_request`. Callers
  that use `share_table.add_share` must call `share_table.release_request` first, then
  `manager.release_request`. This two-phase teardown is documented in `vllm_plugin.py::free`.
- `req_blocks` values are `Vec<BlockId>` (unbounded growth). A pathological request that
  allocates millions of blocks will hold a large Vec until teardown. This is acceptable:
  such a request would exhaust GPU memory long before the Vec became a concern.

## Alternatives considered

- **Caller-managed free list** (Sprint 0 approach): rejected because it pushes memory
  correctness onto every caller and has no crash-safety story.
- **Reference counting only** (no reverse index): rejected because ref counts tell you
  *when* to free, not *what* to free for a given request.
- **Separate lifecycle manager object**: rejected as over-engineering; the block manager is
  the natural owner of its allocations.
