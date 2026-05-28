# Request Lifecycle

Tessera tracks every private block allocation per request via a reverse index
`req_blocks: DashMap<u64, Vec<BlockId>>`. This enables atomic, crash-safe teardown of all
blocks owned by a request in a single call.

---

## Sequence diagram

```
vLLM scheduler                  TesseraBlockAllocator         BlockManager (Rust)
      │                                  │                           │
      │── allocate_mutable_block() ──→   │                           │
      │                                  │── allocate(req_id, …) ──→ │
      │                                  │                           │ insert req_blocks[req_id] ← bid
      │                                  │ ←── BlockId ─────────────  │
      │ ←── _TesseraBlock ─────────────  │                           │
      │                                  │                           │
      │  (prefill writes c_kv to block)  │                           │
      │                                  │                           │
      │── post_prefill_seal(bid, c_kv) → │                           │
      │                                  │── seal(bid) ─────────────→ │
      │                                  │                           │ compute content hash
      │                                  │                           │ if duplicate: free_block_internal(bid)
      │                                  │                           │              return canonical_bid
      │                                  │                           │ else: insert exact index; schedule HNSW add
      │                                  │ ←── (canonical_bid, …) ── │
      │                                  │                           │
      │  (decode loop: primary_ptr / rope_ptr calls update epoch)    │
      │                                  │                           │
      │── free(block) ──────────────→    │                           │
      │                                  │── release_request(req_id) │
      │                                  │                           │ remove req_blocks[req_id]
      │                                  │                           │ for each bid: free(bid)
      │                                  │                           │ decrement ref_counts
      │                                  │ ←── freed_count ──────── │
      │                                  │                           │
      │                                  │ if shared blocks:         │
      │                                  │   share_table.release_request(req_id)
      │                                  │   for bid in to_release: manager.free(bid)
```

---

## Key invariants

1. **Every block allocated for a request ends up in `req_blocks[req_id]`.**  
   Even blocks that are later deduplicated (seal returns a different canonical ID) are
   freed via `free_block_internal` *at seal time* and removed from `req_blocks` then.

2. **`release_request` is idempotent.**  
   If `req_id` is not in `req_blocks` (e.g., already released, or a no-op from an unknown
   request), `release_request` returns 0 without error.

3. **Shared blocks are handled separately.**  
   `req_blocks` only tracks *private* blocks. Blocks shared via `CrossAgentShareTable` are
   released through `share_table.release_request(req_id)`, which returns the block IDs
   whose ref-count dropped to zero — those are then freed via `manager.free`.

4. **Eviction is transparent.**  
   If the eviction policy force-frees a block still in `req_blocks`, `free_block_internal`
   removes it from the reverse index. When `release_request` later tries to free it again,
   `free(block_id)` returns `Ok(())` on an unknown block (the block is simply gone).

---

## Metrics

| Metric | Meaning |
|--------|---------|
| `tessera_request_releases_total` | Total blocks freed via `release_request`. Rising rate under stable load indicates requests are being released normally. Flat under load may indicate a lifecycle leak. |
| `tessera_blocks_used` | Gauge: currently allocated blocks. Should track request count × blocks-per-request. |

---

## Design rationale

See [ADR-0009](adr/0009-per-request-lifecycle.md) for the full decision record.

The short version: caller-managed free lists (Sprint 0 approach) push correctness onto every
integration point and have no crash-safety story. The reverse index inside the block manager
makes lifecycle correctness a property of the manager, not of its callers.
