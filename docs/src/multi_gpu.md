# Multi-GPU (Tensor Parallelism)

**Current status (v0.6.0-sprint5)**: intra-node coordination is implemented end-to-end on
the CPU mock transport; multi-node NCCL transport is type-wired but its runtime body is
deferred to Sprint 6+. Real-GPU validation of `P2pCudaTransport` (NVLink IPC) is gated on
the cloud-burst session. The reserve-then-stream PD-disaggregation protocol (ADR-0018)
is implemented and chaos-tested under random drop rates (ADR-0019).

## Topology

DeepSeek-V3 production typically runs TP=8 intra-node. MLA's `c_kv` is small enough
(68 KB per token, all layers) that **replicating** beats sharding for the latent
representation: communication cost is lower than the memory saved. The interesting
multi-GPU problem for Tessera is therefore *coordination*, not sharding. DeepSeek-V4 keeps
the same structural decision (per-rank KV cache; cross-rank fan-out for shared prefixes).

```text
                       single node, TP=4 (NVLink-connected)
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ rank-0   │  │ rank-1   │  │ rank-2   │  │ rank-3   │
   │ HBM+blk  │  │ HBM+blk  │  │ HBM+blk  │  │ HBM+blk  │
   │ manager  │  │ manager  │  │ manager  │  │ manager  │
   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
              NVLink P2P (P2pCudaTransport ✦)
        cross-node: NCCL (NcclTransport, Sprint 6+)
```

✦ runtime impl gated on cloud-burst session.

## Components (current)

| Component | Role | Source |
|---|---|---|
| `RankId(u32)` | Stable rank identifier | `crates/tessera-core/src/rank.rs` |
| `World { local, size, topology }` | Topology description (single-node or multi-node) | same |
| `LatencyTier` | `IntraNode` / `IntraRack` / `CrossRack` classifier (ADR-0019) | same |
| `GlobalBlockId(RankId, BlockId)` | Cross-rank block identity | `crates/tessera-core/src/block.rs` |
| `trait RankTransport` | Pluggable inter-rank message bus | `crates/tessera-core/src/transport/mod.rs` |
| `MockTransport` | In-process channel impl (tests, CI) — implemented | `transport/mock.rs` |
| `LatencyInjector<T>` | Chaos wrapper: tier-aware latency + jitter + drop_rate — implemented | `transport/latency.rs` |
| `P2pCudaTransport` | NVLink P2P impl — API wired, runtime body cloud-burst gated | `transport/p2p_cuda.rs` |
| `NcclTransport` | Multi-node NCCL — feature-flag stub, runtime Sprint 6+ | `transport/nccl.rs` |
| `DistributedSegmentIndex` | Cross-rank hash lookup with tier-aware budget | `crates/tessera-index/src/distributed.rs` |
| `ReservationToken` + `reserve_slots`/`release_reservation` | Reserve-then-stream PD-disagg (ADR-0018) | `transport/mod.rs` |
| `transfer_request_to_rank` | PD-disaggregation entry point (transactional) | `block_manager.rs` |

## Sequences

### Cross-rank `c_kv` share (prefill hit)

```text
  rank-1                              rank-0
  ───────                             ───────
  prefill computes c_kv hash 0xCAFE
  segment_index.lookup_exact(0xCAFE)
    → local miss
  distributed_index.lookup_hash(0xCAFE)
    → transport.query_hash(rank=0, 0xCAFE)
                                      → ShareTable / hash table lookup
                                      ← return Some(block_id=42)
    ← Some((rank=0, block=42))
  result: (0, 42)  — caller decides:
     • reverse-pull via transport.fetch_block, OR
     • route subsequent decode to rank-0
```

### PD-disaggregation transfer (reserve-then-stream, ADR-0018)

```text
  prefill-rank-0                              decode-rank-3
  ──────────────                              ──────────────
  manager.transfer_request_to_rank(
      req_id=99, target=3, transport)

  [1] RESERVE                                 reserve_incoming(req_id=99, count=N)
      transport.reserve_slots(target=3,       │  forces eviction if needed
                              req_id=99,      │  fails OutOfBlocks cleanly
                              count=N)        │  on capacity miss
                                  ─────────►  ▼
                                              returns ReservationToken
                                  ◄─────────

  [2] STREAM (loop over N owned blocks)
      for each block:
        payload = export_payload(block)
        transport.push_block(target=3, payload)
                                  ─────────►  import_payload(payload)
                                              consume_reservation_slot()
                                  ◄─────────  returns new local block_id

      on ANY error mid-stream:
        transport.release_reservation(target, token)
                                  ─────────►  release_reservation_local(token)
        return Err(...)  — source state untouched

  [3] COMMIT (only on full success)
      release_request(req_id=99)             destination owns N new blocks
      source's used_blocks → 0
      cross_rank_transfer counters incremented
```

The reserve-then-stream protocol guarantees atomicity: every transfer either fully
completes (source releases, destination owns) or fully aborts (source retains
everything, no leaked reservations on destination). The chaos suite exercises this under
random drop rates 0.0–0.75; the proptest invariant `transfer_atomicity_under_chaos`
pins the contract.

## Selecting a transport

```python
from tessera import _native

if world_size == 1:
    transport = _native.MockTransport.singleton()
elif intra_node:
    # Runtime body cloud-burst gated; stub compiles + errors with a migration message.
    transport = _native.P2pCudaTransport(...)
else:
    # Sprint 6+ runtime; stub compiles under `--features nccl`.
    transport = _native.NcclTransport(...)
```

In the Python plugin, this lives in `TesseraBlockAllocator.__init__(rank, world_size,
transport=...)`. When `transport is None`, multi-rank coordination is silently disabled
and the allocator behaves identically to the singleton-world path.

## Test harness — CPU only

`tests/multirank/` exercises every rank-aware code path on the CPU mock backend, without
any GPU. The `MultiRankCoordinator` helper (`python/tessera/multi_rank.py`) spins up N
block managers, N interconnected mock transports, and cross-wires their peer slots in
one line:

```python
from tessera.multi_rank import spawn_multirank_world
coord = spawn_multirank_world(config, world_size=4)
coord.managers[0].transfer_request_to_rank(
    7, target=3, transport=coord.transports[0]
)
```

Chaos coverage comes from two layers (ADR-0019):

- **Rust proptest** (`crates/tessera-core/tests/proptest_chaos.rs`): random op sequences
  over the block manager + random `LatencyProfile` (drop_rate, jitter, tier latencies).
  Invariants checked: `used_blocks` bounded, `free` safe on evicted blocks,
  `release_request` exact-count fidelity, `transfer_request_to_rank` atomicity.
- **Python hypothesis** (`tests/test_hypothesis_*`): boundary fuzz across the PyO3 seam.

## What's deferred (current)

The items below are wired in the type system and tested with `MockTransport`; they need
hardware or a future sprint for their runtime bodies:

* `P2pCudaTransport` runtime body — NVLink IPC handles via `cudarc` (TD-021,
  cloud-burst gated).
* `NcclTransport` runtime body — multi-node IB (TD-022, Sprint 6+).
* Zero-copy `BlockPayload` via CUDA IPC import (TD-026, cloud-burst).
* Disk-tier eviction policy integration with the block manager's tiered LRU (TD-036,
  Sprint 6).
* `StateCache` lifecycle integration in the vLLM plugin (TD-037, Sprint 6).

Items previously listed here as deferred have shipped:

| Previously deferred | Now closed by |
|---|---|
| Reserve-then-stream rollback semantics | ADR-0018 (Sprint 4 / WS3) |
| `MockTransport` chaos testing / latency injection | ADR-0019 (Sprint 4 / WS1+WS4+WS5) |
| `Topology::MultiNode` semantics | Sprint 4 / WS2 (`node_of`, `is_same_node`, `peer_tier`) |
