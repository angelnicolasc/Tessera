//! Rust-side OTLP tracing bridge (Sprint 4 / WS6 / ADR-0017).
//!
//! Mirror of the Python `tessera.observability` API. When both sides initialise against the
//! same OTLP endpoint, their spans interleave naturally on the receiving collector with a
//! shared service name and consistent `tessera.*` span naming.
//!
//! The bridge is **default-off**. The OTel ecosystem pulls tonic / protobuf-c / ring which
//! is a non-trivial build dependency; operators that don't need traces shouldn't pay for
//! them. The public API surface is identical with the feature on or off — `init_otlp_tracing`
//! and `span!` macros compile in both modes; the off-mode versions are no-ops with zero
//! runtime cost.

/// Initialise the OTLP tracer. When the `otel-rust` feature is disabled this is a no-op
/// that returns `Ok(())` so callers can wire it unconditionally.
///
/// Args:
/// - `endpoint`: OTLP gRPC endpoint (e.g. `http://localhost:4317`). Empty / `""` disables.
/// - `service_name`: shows up as `service.name` resource attribute. Sprint 4 default is
///   `"tessera"`; multi-tenant deployments override per-process.
#[cfg(feature = "otel-rust")]
pub fn init_otlp_tracing(endpoint: &str, service_name: &str) -> anyhow::Result<()> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    if endpoint.trim().is_empty() {
        return Ok(());
    }

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| anyhow::anyhow!("OTLP exporter build failed: {e}"))?;

    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(Resource::new(vec![KeyValue::new(
            "service.name",
            service_name.to_string(),
        )]))
        .build();

    let tracer = provider.tracer("tessera");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Best-effort: if a global subscriber is already installed, we silently skip rather
    // than panic. Production processes typically set up the subscriber once at startup;
    // tests may call this multiple times.
    let _ = tracing_subscriber::registry().with(layer).try_init();

    // Stash the provider so the batch exporter keeps draining on shutdown.
    opentelemetry::global::set_tracer_provider(provider);

    tracing::info!(
        endpoint,
        service_name,
        "tessera OTLP tracing bridge initialised"
    );
    Ok(())
}

/// No-op stub when the `otel-rust` feature is disabled. Identical signature so callers
/// don't need cfg-gates.
#[cfg(not(feature = "otel-rust"))]
pub fn init_otlp_tracing(_endpoint: &str, _service_name: &str) -> anyhow::Result<()> {
    Ok(())
}

/// Gracefully flush + shut down the tracer. Call on process termination to avoid losing
/// the last batch of spans. No-op without the feature.
#[cfg(feature = "otel-rust")]
pub fn shutdown_tracing() {
    opentelemetry::global::shutdown_tracer_provider();
}

/// No-op tracing shutdown when the `otel-rust` feature is disabled.
#[cfg(not(feature = "otel-rust"))]
pub fn shutdown_tracing() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_with_empty_endpoint_is_noop() {
        // Should never fail; should not require an actual collector.
        init_otlp_tracing("", "tessera").unwrap();
    }
}
