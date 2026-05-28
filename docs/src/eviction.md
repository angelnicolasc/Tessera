# Eviction Policy

Tessera uses a **tiered LRU** eviction policy that understands MLA's sharing semantics.
When `allocate` runs out of free blocks it calls `evict_one()` before returning
`OutOfBlocks`. This ensures the manager degrades gracefully under memory pressure rather
than crashing the serving loop.

---

## Tier diagram

```
Priority (evict first → last)
┌────────────────────────────────────────────────────────────┐
│  Tier a — ref_count == 0                                   │
│  Orphaned blocks. No owner. Evict immediately.             │
├────────────────────────────────────────────────────────────┤
│  Tier b — ref_count == 1, NOT indexed                      │
│  Private, cold, not in segment index. Cheapest loss.       │
├────────────────────────────────────────────────────────────┤
│  Tier c — ref_count == 1, indexed                          │
│  Private, cold, BUT in HNSW index — approximate-match      │
│  value. Prefer to keep; evict only when a and b exhausted. │
├────────────────────────────────────────────────────────────┤
│  Tier d — ref_count > 1 (SHARED)                           │
│  Multiple live owners. NEVER evicted.                      │
└────────────────────────────────────────────────────────────┘
```

Within tiers a–c, the candidate with the smallest `last_touched` epoch is chosen (least
recently used). Epochs are monotonic 64-bit counters incremented on `primary_ptr` and
`rope_ptr` accesses — no wall-clock involved.

---

## Implementation details

### `BlockMeta.last_touched`

Every block carries `last_touched: Arc<AtomicU64>`. Each call to `primary_ptr()` or
`rope_ptr()` performs a lock-free `fetch_add(1, Relaxed)` on the process-global epoch
counter and stores the result. The scan in `evict_one` reads epochs without holding any
lock; only the final force-free acquires the write lock.

### `evict_one()` algorithm

```
1. Acquire read lock on block table.
2. Scan all blocks; for each compute (tier, epoch).
3. Track (best_tier, best_epoch, best_block_id).
4. Release read lock.
5. If candidate found and tier != d:
     free_block_internal(candidate)
     increment tessera_evictions_total{tier}
```

`free_block_internal` removes the block from `req_blocks`, drops the device memory, and
removes the content hash from the segment index if present. It acquires the write lock
internally.

### Metric

```
tessera_evictions_total{tier="a"} — orphaned evictions
tessera_evictions_total{tier="b"} — cold-unindexed evictions
tessera_evictions_total{tier="c"} — cold-indexed evictions
```

A sustained rate of tier-c evictions under normal load indicates the GPU memory budget is
too small for the workload prefix set — consider increasing `gpu_memory_bytes` or reducing
`block_size_tokens`.

---

## Design rationale

See [ADR-0010](adr/0010-eviction-policy.md) for the full decision record, including
alternatives considered (random eviction, background eviction thread, ref-count-only
policy) and their trade-offs.

---

## Known limitations

- `evict_one` is O(n) in blocks. For managers > 500 K blocks a tiered min-heap would reduce
  scan time. Filed as future work; not on the critical path for current deployment sizes.
- One eviction per `allocate` call. If many blocks are simultaneously orphaned (e.g., mass
  request cancellation), the next `n` allocations will each trigger one eviction rather than
  a bulk reclaim. This is correct but suboptimal; a future `evict_batch(n)` call can fix it.
