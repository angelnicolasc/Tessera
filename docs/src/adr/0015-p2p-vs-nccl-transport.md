# ADR-0015 — Cross-rank transport selection: P2pCuda vs NCCL vs Mock

**Status:** Accepted 2026-05-21. **Status update (Sprint 4–5)**:

* `MockTransport`: implemented + exercised by every multi-rank test.
* `LatencyInjector<T>` chaos wrapper (originally flagged as future work here) shipped in
  Sprint 4 — see [ADR-0019](0019-latency-injection-chaos-testing.md). Every transport
  can be wrapped with a deterministic tier-aware latency + drop profile for chaos
  testing.
* `P2pCudaTransport`: API + dispatcher still wired; runtime body remains cloud-burst
  gated (TD-021).
* `NcclTransport`: compile-only stub behind `--features nccl`; runtime body now
  scheduled for **Sprint 6+** (was "Sprint 4" in this ADR's original text), pending the
  multi-node test harness.

## Context

Cross-rank communication has fundamentally different cost profiles depending on whether the
peers share a node:

| Topology | Mechanism | Latency (typical) | Bandwidth |
|---|---|---|---|
| Same node | NVLink P2P (`cuCtxEnablePeerAccess` + `cuMemcpyPeerAsync`) | 5–15 µs | 600 GB/s |
| Different nodes (same DGX) | NVLink-switch + NCCL | 20–40 µs | 600 GB/s |
| Different nodes (IB) | NCCL over InfiniBand | 100 µs – 1 ms | 50–100 GB/s |
| Tests / CI / dev | In-process channels | sub-µs | unbounded |

A single "unified transport" would force either (a) NCCL overhead for intra-node calls or
(b) NVLink semantics on configurations that have no NVLink. Both are wrong choices for the
majority of deployments.

Tessera also needs a CI-runnable, GPU-free transport so the multi-rank surface can be
validated without dedicated hardware on every PR.

## Decision

Three implementations of `RankTransport`:

1. **`MockTransport`** (`transport::mock`) — in-process channels via `tokio::sync::mpsc`.
   Always compiled. Used by all multi-rank tests, the `MultiRankCoordinator` Python helper,
   and any environment that needs to exercise the multi-rank code path without hardware.
   Records every call into an `EventLog` for assertion-grade test introspection.

2. **`P2pCudaTransport`** (`transport::p2p_cuda`, feature `cuda`) — NVLink P2P, the
   production intra-node path. Wraps `cudarc::driver::CudaDevice` + `cuCtxEnablePeerAccess`
   + IPC handles. Sprint 3 ships the struct, methods, and dispatcher wiring; the runtime
   bodies return a structured error citing TD-021 and are implemented in the cloud-burst
   session with multi-GPU hardware available.

3. **`NcclTransport`** (`transport::nccl`, feature `nccl`) — multi-node fan-out. Same shape,
   runtime impl deferred to Sprint 4 (TD-022). The feature flag toggles compile-time
   inclusion without pulling in an extra dependency in Sprint 3.

Selection lives in the consumer — typically the Python plugin's `__init__`. The block
manager itself remains transport-agnostic: it accepts `Arc<dyn RankTransport>`.

## Consequences

* `cargo build --features cuda` compiles `P2pCudaTransport` and exercises its API surface
  on every CI run, catching cudarc drift early (the same mechanism that closed TD-001).
* `cargo build --features nccl` compiles `NcclTransport`; the runtime bodies are stubbed
  with diagnostic errors that name TD-022 so any production user attempting to use
  multi-node before Sprint 4 lands gets a clear migration path.
* Tests run anywhere — no GPU runner needed for the multi-rank suite.
* `MockTransport` is not a "performance reference"; its `tokio::yield_now` cooperative
  yields are scheduler-deterministic but real transports have very different latency
  distributions. Sprint 4 will add a `LatencyInjector` adapter for chaos testing
  (TD-025).
