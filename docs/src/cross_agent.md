# Cross-Agent Sharing

`c_kv` is position-independent. Two agents that process the same document at different
prompt positions have identical `c_kv` — even though their full token prefix differs. Tessera
exploits this directly; no estimation, no adjustment factor.

## Share Table

`CrossAgentShareTable` maintains two indexes:

* `block_to_owners: DashMap<BlockId, Vec<req_id>>`
* `req_to_blocks: DashMap<req_id, Vec<BlockId>>`

Plus a `total_shared_refs: AtomicU64` powering the `tessera_sharing_rate` Prometheus gauge.

## Copy-on-Write

**Invariant:** a shared block is never mutated in place.

Sequence for a write to a shared block:

```text
1. Requests A and B share block_id=42.
2. A receives a token that extends block 42's token range.
3. Block manager detects ref_count > 1.
4. cow_fork(42, req_a) → new block_id=87. GPU memcpy of c_kv + k_rope + scales.
5. A's block table points at 87. Ref-count on 42 decremented.
6. B keeps reading from 42 unaffected.
```

The CoW protocol composes correctly with:

* **Speculative decoding** — candidate tokens at new positions allocate fresh private blocks.
* **Beam search** — common prefix blocks are shared; diverged suffixes are forked copies.

## Metric — Sharing Rate

```text
sharing_rate = shared_blocks / total_shared_refs_added
```

The cross-agent benchmark in `benchmarks/sharing_bench.py` exercises this in a pure CPU-mock
configuration so its expected output is deterministic.
