"""Verify the ``TesseraBlockAllocator`` exposes the V1 ``BlockAllocator`` protocol.

This test uses introspection only — it does NOT require vLLM to be installed. The full
integration test (instantiating an allocator and running it through vLLM's engine) lives in
``tests/integration/`` (Sprint 1).
"""

from __future__ import annotations

import inspect

import pytest

from tessera.vllm_plugin import REQUIRED_V1_METHODS, TesseraBlockAllocator


@pytest.mark.parametrize("method", REQUIRED_V1_METHODS)
def test_required_method_is_defined(method: str) -> None:
    assert hasattr(TesseraBlockAllocator, method)
    fn = getattr(TesseraBlockAllocator, method)
    assert callable(fn), f"{method} must be callable"


def test_post_prefill_seal_is_present() -> None:
    assert callable(TesseraBlockAllocator.post_prefill_seal)


def test_find_shared_prefix_signature_returns_optional_list() -> None:
    sig = inspect.signature(TesseraBlockAllocator.find_shared_prefix)
    # Python 3.11 returns the annotation as a string by default with `from __future__ import
    # annotations` in our source; that's fine — we just check the params shape.
    params = list(sig.parameters.keys())
    assert params == ["self", "req_id", "c_kv_blocks"]


def test_allocate_mutable_block_signature() -> None:
    sig = inspect.signature(TesseraBlockAllocator.allocate_mutable_block)
    params = list(sig.parameters.keys())
    assert params == ["self", "prev_block", "device"]
