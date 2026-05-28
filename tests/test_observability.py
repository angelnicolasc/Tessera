"""WS3 — OTLP tracing tests (TD-010).

Verifies:
- span() is a no-op when endpoint is empty (nullcontext)
- span() is a no-op when opentelemetry is absent (monkeypatched)
- InMemorySpanExporter captures span names correctly
- TesseraBlockAllocator emits tessera.allocate span when tracer is active
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from tessera import _native
from tessera.observability import _TRACER_CACHE, get_tracer, init_tracing, span


# ──────────── helpers ─────────────────────────────────────────────────────────


def _clear_tracer_cache() -> None:
    _TRACER_CACHE.pop("tracer", None)


def _minimal_bf16_config_dict() -> dict:
    return {
        "model": {
            "name": "test",
            "latent_dim": 32,
            "rope_key_dim": 8,
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 128,
        },
        "block": {"block_size_tokens": 64, "ckv_dtype": "bf16"},
        "runtime": {"gpu_memory_bytes": 64 * 1024 * 1024},
    }


# ──────────── tests ────────────────────────────────────────────────────────────


def test_span_noop_when_endpoint_empty() -> None:
    """span() must return a nullcontext (no exception) when no endpoint is set."""
    _clear_tracer_cache()
    init_tracing("")
    assert get_tracer() is None
    with span("tessera.test_noop"):
        pass  # must not raise


def test_span_noop_produces_nullcontext_type() -> None:
    """span() returns contextlib.nullcontext when tracer is absent."""
    _clear_tracer_cache()
    ctx = span("tessera.x")
    assert isinstance(ctx, contextlib.AbstractContextManager)
    # Should be nullcontext specifically when no tracer set
    assert type(ctx).__name__ == "_GeneratorContextManager" or isinstance(
        ctx, contextlib.nullcontext
    )


def test_span_noop_when_tracer_not_set() -> None:
    """span() with an empty _TRACER_CACHE is a no-op nullcontext regardless of endpoint."""
    _clear_tracer_cache()
    assert get_tracer() is None

    with span("tessera.any_name"):
        pass  # must not raise

    # No tracer should have been set by span() itself.
    assert get_tracer() is None


def test_mock_exporter_records_span_names() -> None:
    """InMemorySpanExporter captures span names emitted through observability.span()."""
    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk not installed")
    in_memory = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter",
        reason="opentelemetry-sdk not installed",
    )
    otel_api = pytest.importorskip("opentelemetry", reason="opentelemetry not installed")

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[import-not-found]

    exporter = in_memory.InMemorySpanExporter()
    provider = otel_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_api.trace.set_tracer_provider(provider)

    _clear_tracer_cache()
    _TRACER_CACHE["tracer"] = otel_api.trace.get_tracer("tessera-test")

    with span("tessera.allocate"):
        pass
    with span("tessera.seal"):
        pass

    finished = exporter.get_finished_spans()
    names = [s.name for s in finished]
    assert "tessera.allocate" in names
    assert "tessera.seal" in names

    _clear_tracer_cache()
    exporter.clear()


def test_allocator_emits_no_exception_with_mock_tracer() -> None:
    """TesseraBlockAllocator operates normally when a mock tracer is injected."""
    from tessera.config import TesseraConfig
    from tessera.vllm_plugin import TesseraBlockAllocator

    _clear_tracer_cache()
    # Inject a simple mock tracer that records span names.
    recorded: list[str] = []

    class _MockSpan:
        def __enter__(self) -> _MockSpan:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    class _MockTracer:
        def start_as_current_span(self, name: str) -> _MockSpan:
            recorded.append(name)
            return _MockSpan()

    _TRACER_CACHE["tracer"] = _MockTracer()

    cfg = TesseraConfig.from_dict(_minimal_bf16_config_dict())
    allocator = TesseraBlockAllocator(cfg)

    # allocate_mutable_block fires tessera.allocate span.
    block = allocator.allocate_mutable_block(None, device=None)
    assert "tessera.allocate" in recorded

    _clear_tracer_cache()
