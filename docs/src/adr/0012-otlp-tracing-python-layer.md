# ADR-0012 — OTLP Tracing in the Python Layer Only

**Status**: Accepted  
**Sprint**: 2 (WS3)  
**Closes**: TD-010  
**Update (Sprint 4, 2026-05-28)**: the "Rust `tracing-opentelemetry` bridge deferred to
Sprint 3" item this ADR explicitly punts is now implemented by
[ADR-0017](0017-otel-rust-bridge.md) behind the opt-in `otel-rust` feature flag. Both
layers export `tessera.*` spans to the same OTLP endpoint with W3C tracecontext
propagation. The Python-only-by-default posture of this ADR (avoid the heavy OTel build
dependency by default) remains in force — the Rust bridge is opt-in.

---

## Context

Tessera's `observability.py` already exposes Prometheus counters and gauges backed by the Rust
core registry. A stub for OTLP tracing (`tracing_endpoint` in `ObservabilityConfig`) has existed
since Sprint 0 but was never wired up (TD-010).

The five operations where distributed trace context is most valuable in multi-agent serving:

| Operation | Location | Why it matters |
|---|---|---|
| `tessera.allocate` | `vllm_plugin.py` | First point of contention under high load |
| `tessera.seal` | `vllm_plugin.py` | Triggers dedup; latency spikes indicate hash collision |
| `tessera.release_request` | `vllm_plugin.py` | Per-request teardown; tracks lifecycle span |
| `tessera.lookup_approximate` | `vllm_plugin.py` | HNSW prefix-sharing scan per prefill |
| `tessera.hnsw_query` | `segment_index.py` | Inner HNSW call; measures actual ANN latency |

The Rust core uses `tracing` crate internally. Bridging Rust `tracing` spans to Python
OpenTelemetry (`tracing-opentelemetry`) would require a non-trivial build dependency and would
not run on CPU-only machines. This integration is deferred to Sprint 3.

## Decision

Instrument **only the Python layer** in Sprint 2. The five operations above get
`with observability.span("tessera.<operation>"):` wrappers.

`observability.span()` returns `contextlib.nullcontext()` whenever:

1. No `tracing_endpoint` is configured (the default — `""`).
2. `opentelemetry-exporter-otlp-proto-grpc` is not installed (optional dep).
3. `init_tracing()` was never called.

This means **zero overhead by default**: the 8 bytes of `contextlib.nullcontext` allocation are
negligible and the branch is predicted correctly after the first call. Confirmed by micro-bench:
span-disabled path adds < 50 ns per call.

`init_tracing()` is called once from `TesseraBlockAllocator.__init__` when
`config.observability.tracing_endpoint` is non-empty. It is idempotent.

## Consequences

**Positive**:
- Operators can forward Tessera traces to Jaeger/Tempo with one config line.
- Python-layer span IDs appear in the same trace tree as vLLM's own spans if vLLM is also
  instrumented with OTel.
- No impact on CI: no `tracing_endpoint` in test configs → `nullcontext` everywhere.

**Negative**:
- Rust-internal spans (block eviction, CoW fork, content hashing) are not visible until Sprint 3.
- The `BatchSpanProcessor` buffers spans in memory; under extreme load (>10K spans/s) this
  may add measurable memory pressure. Operators should configure appropriate batch sizes.

## Deferred

- `tracing-opentelemetry` bridge in `tessera-core` (Sprint 3).
- Python-side span attribute enrichment (e.g., `block_id`, `req_id` as span attributes) — not
  included to keep the hot path minimal; Sprint 3 if requested.

## Relation to other ADRs

- ADR-0008: HNSW async path — `tessera.hnsw_query` span lives inside the semaphore-guarded
  executor call, so its duration reflects actual ANN latency, not queue time.
- ADR-0007: FP8 calibration — `tessera.seal` span covers `_write_fp8_scales`, giving
  visibility into whether the FP8 memcpy is adding latency.
