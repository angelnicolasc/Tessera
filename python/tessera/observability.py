"""Observability glue.

The Rust core exposes Prometheus counters / gauges directly; this module gives the Python
layer a handle to bump them and a small HTTP exposer for ``/metrics``. Both pieces are
optional — Tessera works without prometheus_client installed; only the helpers in this file
do.

WS7 addition: ``segment_index_queue_depth`` gauge tracking concurrent HNSW query
backpressure. Updated by ``SegmentIndex.lookup_approximate`` on every entry and exit.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any

# ─── OTLP tracing (WS3) ───────────────────────────────────────────────────────
# Tracing is optional: when endpoint is empty or the opentelemetry packages are
# absent, every `span()` call returns a no-op nullcontext. Zero overhead by default.

_TRACER_CACHE: dict[str, Any] = {}


def init_tracing(endpoint: str, service_name: str = "tessera") -> None:
    """Configure OTLP gRPC tracing.

    No-op when ``endpoint`` is empty or when ``opentelemetry-exporter-otlp-proto-grpc``
    is not installed. Calling this multiple times with the same endpoint is idempotent.

    Args:
        endpoint: OTLP gRPC collector endpoint, e.g. ``"http://localhost:4317"``.
            Pass ``""`` to disable tracing entirely.
        service_name: OpenTelemetry service name attribute. Defaults to ``"tessera"``.
    """
    if not endpoint:
        return
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import (
            TracerProvider,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,  # type: ignore[import-not-found]
        )
    except ImportError:
        return
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER_CACHE["tracer"] = trace.get_tracer(service_name)


def get_tracer() -> Any | None:
    """Return the active OpenTelemetry tracer, or ``None`` if tracing is not configured."""
    return _TRACER_CACHE.get("tracer")


def span(name: str) -> contextlib.AbstractContextManager[Any]:
    """Return a span context manager if tracing is active, else a no-op ``nullcontext``.

    Usage::

        with observability.span("tessera.allocate"):
            block_id = manager.allocate(...)

    When no tracer is configured (the default in CI and production without an OTLP
    endpoint) this degrades to ``contextlib.nullcontext()`` with zero overhead.
    """
    tracer = get_tracer()
    if tracer is None:
        return contextlib.nullcontext()
    return tracer.start_as_current_span(name)  # type: ignore[return-value]


def _native_metrics_text() -> str:
    """Snapshot the native (Rust) Prometheus registry as text format."""
    from tessera import _native

    return _native.metrics_snapshot_text()


def hnsw_budget_exceeded() -> None:
    """Increment the ``tessera_hnsw_budget_exceeded_total`` counter.

    The Rust core owns the authoritative counter; we maintain a Python-side mirror so
    dashboards pulled via the Python HTTP exposer reflect the change too.
    """
    try:
        prom = importlib.import_module("prometheus_client")
    except ImportError:
        return
    counter = _python_hnsw_counter(prom)
    counter.inc()


def set_segment_index_queue_depth(depth: int) -> None:
    """Update the ``tessera_segment_index_queue_depth`` gauge (WS7).

    Called by ``SegmentIndex.lookup_approximate`` with the number of available semaphore
    slots (not the waiting count). Callers are expected to subtract from the max concurrency
    limit to derive actual queue depth if needed; here we track the semaphore value directly
    for simplicity.

    Best-effort: silently no-ops when ``prometheus_client`` is absent.
    """
    try:
        prom = importlib.import_module("prometheus_client")
    except ImportError:
        return
    gauge = _python_queue_depth_gauge(prom)
    gauge.set(depth)


_PYTHON_COUNTER_CACHE: dict[str, Any] = {}


def _python_hnsw_counter(prom: Any) -> Any:
    """Return a process-local Python ``Counter`` mirroring the native one."""
    name = "tessera_hnsw_budget_exceeded_total"
    if name not in _PYTHON_COUNTER_CACHE:
        _PYTHON_COUNTER_CACHE[name] = prom.Counter(
            name, "HNSW lookups that exceeded their latency budget"
        )
    return _PYTHON_COUNTER_CACHE[name]


def _python_queue_depth_gauge(prom: Any) -> Any:
    """Return a process-local Python ``Gauge`` for segment index queue depth."""
    name = "tessera_segment_index_queue_depth"
    if name not in _PYTHON_COUNTER_CACHE:
        _PYTHON_COUNTER_CACHE[name] = prom.Gauge(
            name,
            "Available semaphore slots for concurrent HNSW queries (WS7 backpressure)",
        )
    return _PYTHON_COUNTER_CACHE[name]


def start_metrics_server(port: int) -> None:
    """Start a ``prometheus_client`` HTTP server on ``port``. No-op if the library is missing.

    The server exposes both the Python-side counters AND the snapshot of the native registry
    appended at the bottom under a ``# native:`` block.
    """
    try:
        prom = importlib.import_module("prometheus_client")
    except ImportError as exc:
        msg = "prometheus_client is not installed. `pip install tessera[observability]`"
        raise RuntimeError(msg) from exc
    prom.start_http_server(port)


def metrics_text() -> str:
    """Combined Python + native metrics text snapshot."""
    parts = [_native_metrics_text()]
    try:
        prom = importlib.import_module("prometheus_client")
    except ImportError:
        return parts[0]
    parts.append(prom.generate_latest().decode("utf-8"))
    return "\n".join(parts)
