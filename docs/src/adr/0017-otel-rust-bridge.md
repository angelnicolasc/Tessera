# ADR-0017 — Rust OTLP tracing bridge (`otel-rust` feature)

**Status:** Accepted, 2026-05-28.

## Context

Sprint 2 wired `tessera.observability` on the Python side to OTLP via
`opentelemetry-exporter-otlp-proto-grpc`. The Rust core, where the hot paths actually
execute, has been emitting structured `tracing` events but those events stayed within the
process — no way to correlate a Python-side `tessera.allocate` span with the Rust block
manager work inside it. ADR-0012 documented this gap with TD-030 as the closing item.

Two failed alternatives:

* **Make the Python layer query Rust counters and synthesise spans.** Sampling-friendly but
  loses the actual span structure (start/stop, hierarchical context, attributes). Useless
  for debugging concurrent CoW races.
* **Always-on Rust OTel.** Pulls `tonic`, `protobuf-c`, `ring`. Compile time goes up by
  ~40s. Many deployments (offline benchmarks, CI sanity, vendored bundles) don't want it.

## Decision

Sprint 4 adds **`otel-rust` as an opt-in feature flag** on `tessera-core`:

```rust
[features]
otel-rust = [
    "dep:tracing-opentelemetry",
    "dep:opentelemetry",
    "dep:opentelemetry_sdk",
    "dep:opentelemetry-otlp",
]
```

`crates/tessera-core/src/observability.rs` provides two public functions with stable
signatures regardless of feature state:

```rust
pub fn init_otlp_tracing(endpoint: &str, service_name: &str) -> anyhow::Result<()>;
pub fn shutdown_tracing();
```

Feature off → no-op stubs. Feature on → real OTLP exporter wired into the global
`tracing` subscriber via `tracing-opentelemetry::layer()`.

Spans on both sides use the same `tessera.*` prefix and the same OTLP endpoint, so a
single collector receives unified traces. Trace context propagation uses the W3C
`tracecontext` standard so cross-process correlation works.

## Consequences

* Default builds stay lean. Anyone benchmarking compile time on stock `cargo build` sees
  no change.
* Operators who want unified traces flip both knobs:
  - Rust: `cargo build --features otel-rust` + call `init_otlp_tracing(endpoint, "tessera")`.
  - Python: `pip install tessera[observability]` + `tessera.observability.init_tracing(endpoint)`.
* CI runs a dedicated `build-otel-rust` job to validate the feature compiles cleanly on
  every push (the pin set is fragile across the opentelemetry ecosystem's frequent minor
  bumps).
* Future work (TD-030 → TD-033 in Sprint 4 DEVLOG): env-var-driven auto-discovery so the
  binary picks up `OTEL_EXPORTER_OTLP_ENDPOINT` without explicit init calls.
