# ADR-0018 — Reserve-then-stream transactional PD-disaggregation

**Status:** Accepted, 2026-05-28. Supersedes the push-mode semantics in ADR-0016.

## Context

ADR-0016 introduced `transfer_request_to_rank(req_id, target)` as a push-mode primitive.
Sprint 3 left two gaps documented as TD-024:

1. **Target capacity isn't checked up front.** If the destination is OOM the first
   `push_block` fails halfway through, leaving the source with partially-transferred state.
   Rollback is the caller's problem.
2. **Concurrent eviction race.** While a transfer is in progress, the destination's
   eviction policy could evict blocks the transfer just imported. With no explicit pinning,
   the just-arrived blocks could be reaped before the destination's scheduler sees them.

Sprint 4 needs to ship robust failure semantics before cloud-burst validation lands, so
producers can rely on `transfer_request_to_rank` rather than reimplementing rollback.

Two-phase commit was considered (`prepare` + `commit` + `abort` messages on every block).
Rejected: doubles the transport round-trips for the common-case success path. The cost
hits decode-latency-sensitive workloads.

## Decision

**Reserve-then-stream** — a lighter variant of 2PC where capacity reservation happens once
up front, then individual blocks stream without per-block prepare/commit.

```text
source                                           destination
──────                                           ──────────────
1. transport.reserve_slots(target, req_id, n)    ┌─ reserve_incoming(req_id, n)
   ─────────────────────────────────────────────►│  (forces eviction if needed;
                                                 │   fails OOM cleanly)
   ◄──────────────────────────────────── token   └─ insert ReservationEntry

2. for each block:                               ┌─ import_payload(req_id, ...)
   transport.push_block(target, payload)         │  consume_reservation_slot(token)
   ─────────────────────────────────────────────►│
   ◄──────────────────────── new_block_id (or Err)
   on ANY Err: transport.release_reservation     ┌─ release_reservation_local(token)
              ─────────────────────────────────► │  drop ReservationEntry; eviction may
                                                 │   reclaim now-unpinned capacity

3. on full success: release_request(req_id)
   (source-local; no extra transport call)
```

The destination's `ReservationEntry { req_id, remaining }` pins capacity until either:
* every slot is consumed by a matching `push_block` (transfer succeeded), or
* the source surrenders via `release_reservation` (transfer aborted).

Sprint 4 implementation choices:

* `ReservationToken = (rank << 48) | counter` — globally unique even across destination
  restarts within a deployment.
* `reserve_incoming` forces eviction up to `count` blocks before failing. The new
  `RESERVATIONS_ACTIVE{rank}` gauge surfaces leaked reservations to monitoring.
* The source's atomic invariant: on any `Err`, the source's `req_blocks[req_id]` is
  untouched. The caller can retry safely. Tests in
  `tests/pd_disagg_transactional.rs` verify this for three failure modes (target OOM,
  mid-stream push drop, empty request).

## Consequences

* Common-case latency cost: **one extra transport round-trip per transfer** (the reserve
  step). At intra-node NVLink latencies (~5 µs) this is invisible; at cross-node IB it's
  the right tradeoff for the atomicity guarantee.
* Sprint 3's ADR-0016 push-mode is now subsumed; the new contract supersedes it.
* `RankTransport` trait grew two methods. Existing stubs (`P2pCudaTransport`,
  `NcclTransport`) implement them as deferred errors citing TD-021/TD-022.
* The `BlockManagerPeerAdapter` in `crates/tessera-py/src/lib.rs` was updated to forward
  reserves to `reserve_incoming` so the PyO3 path enforces real capacity.
* TD-024 closed; TD-029 opened in DEVLOG as a follow-up: under sustained heavy
  concurrency, reservation-vs-eviction interplay may need additional fairness
  guarantees (e.g. tier-d analog for reserved blocks).

## Test coverage (Sprint 4)

* `crates/tessera-core/tests/pd_disagg_transactional.rs`:
  - Full-success path drains source and lands all blocks at destination.
  - Destination OOM aborts cleanly; source state untouched.
  - Mid-stream push failure under `LatencyProfile::ALL_DROPS`: source state untouched, no
    reservations leaked.
  - Empty request returns 0 without touching transport.
* `proptest_chaos.rs` exercises atomicity invariants under random drop rates 0.0–0.75.
