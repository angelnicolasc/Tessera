# ADR-0019 — Latency injection + chaos testing

**Status:** Accepted, 2026-05-28.

## Context

Tessera's distributed protocols (cross-rank share, distributed segment index, reserve-then-
stream PD-disaggregation) all assume the transport layer can fail. Until Sprint 4, those
failure paths were code paths nobody had actually exercised — every test used the
`MockTransport` with `tokio::yield_now` and assumed success. TD-025 tracked the gap.

A CPU-only chaos rig must satisfy three constraints to be useful:

1. **Deterministic.** Tests need to reproduce failures by seed for forensics. Random
   wall-clock-based jitter makes regression analysis impossible.
2. **Realistic.** The latency distribution should track real NVLink / NVSwitch / IB cost
   profiles so timeouts and budgets behave as they would in production.
3. **Composable.** Any `RankTransport` implementation should be wrappable without code
   changes in the transport itself.

## Decision

`LatencyInjector<T: RankTransport>` (`crates/tessera-core/src/transport/latency.rs`) wraps
any transport with:

```rust
pub struct LatencyProfile {
    pub intra_node_us: u64,    // typical: 5
    pub intra_rack_us: u64,    // typical: 50
    pub cross_rack_us: u64,    // typical: 500
    pub jitter_us: u64,        // ±range
    pub drop_rate: f32,        // [0.0, 1.0]
}
```

Three preset constants:

* `INTRA_NODE_REALISTIC` — sane default for NVLink benchmarks.
* `STRESS_MULTI_RACK` — large cross-rack base + jitter + 5% drops.
* `ALL_DROPS` — 100% failure for negative testing.

Determinism is enforced by an explicit `ChaCha8Rng` seeded via
`LatencyInjector::new(transport, profile, local, topology, seed)`. The injector picks the
tier per call from the wrapped topology + destination rank, samples `±jitter_us`, sleeps
via `tokio::time::sleep` (compatible with `tokio::time::pause` in tests), and rolls a
drop decision against `drop_rate`.

Drops increment `tessera_latency_injected_drops_total{op}` so dashboards distinguish
chaos-rig traffic from real failures.

Two layers of chaos coverage:

* **Rust proptest** (`tests/proptest_chaos.rs`): 64 cases × random op sequences over the
  block manager. Properties: `used_blocks` bounded, no panics on `free` of evicted blocks,
  `release_request` exact count, `transfer_request_to_rank` atomicity under random
  `drop_rate`.
* **Python hypothesis** (`tests/test_hypothesis_allocator.py`,
  `tests/test_hypothesis_distributed.py`): 64 cases × random op sequences through the
  PyO3 boundary. Properties: Python tracking matches `used_blocks`, seal idempotence,
  `lookup_hash` shape invariants. Catches Python ↔ Rust type-conversion bugs that pure
  Rust tests can't see.

## Consequences

* Every distributed protocol in Tessera is now stress-tested against jitter + drops
  before cloud-burst validation. When the real `P2pCudaTransport` lands, the contract is
  already battle-tested.
* `LatencyInjector` is production-grade — operators can run it in staging by wrapping
  the real transport with a small `drop_rate` to surface race conditions before they hit
  live traffic.
* Limitations (TD-028, TD-031 in DEVLOG):
  - Per-op drop selectivity is not yet supported (all ops share `drop_rate`). Sprint 5
    can add `LatencyProfile::per_op` if production needs it.
  - Hypothesis covers small worlds (≤ 4 ranks); larger-world fuzz needs a slower CI
    cadence.

## Related

* ADR-0014 (multi-rank architecture) — defines the `RankTransport` contract the chaos
  layer wraps.
* ADR-0018 (reserve-then-stream PD-disagg) — primary consumer of the chaos rig.
