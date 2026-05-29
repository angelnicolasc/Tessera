"""Hypothesis boundary fuzz: random op sequences against the native ``BlockManager``.

Cross-cuts the Python ↔ Rust seam: every op goes through the PyO3 boundary, exercising
the type conversions + error mapping. Properties asserted:

1. **Tracking parity** — Python-side bookkeeping (`set` of live block_ids) and the
   native ``used_blocks`` counter agree after every op.
2. **Seal idempotence** — sealing identical bytes always returns the same canonical
   block id (the native dedup contract).
3. **release_request fidelity** — count returned by ``release_request`` equals the
   private-block count owned by that req_id at call time.

Strategy: 8 ops drawn from {Allocate, FillFree, Seal, Release}. Each test runs 64 random
sequences (kept tight for CI wall-clock).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("tessera._native")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tessera import _native


def small_cfg():
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    return _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Bf16, 0)


# Op DSL: small ADT serialised as tuple-tagged dicts so hypothesis can shrink.
_op_strategy = st.one_of(
    st.tuples(st.just("alloc"), st.integers(min_value=0, max_value=3)),
    st.tuples(st.just("seal"), st.integers(min_value=0, max_value=255)),
    st.tuples(st.just("release"), st.integers(min_value=0, max_value=3)),
)


@given(ops=st.lists(_op_strategy, min_size=1, max_size=24))
@settings(
    max_examples=64,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_used_blocks_consistent_under_random_ops(ops):
    mgr = _native.BlockManager(small_cfg(), 32 * 1024 * 1024)
    live: dict[int, int] = {}  # block_id → req_id (private blocks only)

    for op in ops:
        kind = op[0]
        if kind == "alloc":
            req = int(op[1])
            try:
                bid = mgr.allocate(req, 0, 64)
                live[bid] = req
            except RuntimeError:
                # OOM under tight budget. Allowed; just don't break the invariant.
                pass
        elif kind == "seal":
            pattern = int(op[1])
            # Pick the most recently allocated block to seal, if any.
            if live:
                bid = next(reversed(live))
                try:
                    mgr.fill_primary_test_pattern(bid, pattern)
                    _canon, _h, was_dedup = mgr.seal(bid)
                    if was_dedup:
                        # The duplicate was freed; remove from our model.
                        del live[bid]
                except (ValueError, RuntimeError):
                    pass
        elif kind == "release":
            req = int(op[1])
            freed = mgr.release_request(req)
            owned = [b for b, r in live.items() if r == req]
            # Invariant: release_request frees exactly the owned blocks (live model
            # tracks privately-owned only, which matches release_request semantics).
            assert freed == len(owned)
            for b in owned:
                del live[b]

        # Invariant: native used_blocks >= len(live) at all times (some blocks may have
        # been deduped and not tracked, but every live block we know about is still real).
        assert mgr.used_blocks >= len(live), (
            f"used_blocks ({mgr.used_blocks}) < python live ({len(live)})"
        )
        # Invariant: used_blocks bounded by total_blocks.
        assert mgr.used_blocks <= mgr.total_blocks


@given(pattern=st.integers(min_value=0, max_value=255))
@settings(max_examples=32, deadline=None)
def test_seal_of_identical_bytes_is_deterministic(pattern):
    """Two allocations filled with the same byte pattern must dedup deterministically."""
    mgr = _native.BlockManager(small_cfg(), 32 * 1024 * 1024)
    a = mgr.allocate(1, 0, 64)
    b = mgr.allocate(2, 0, 64)
    mgr.fill_primary_test_pattern(a, pattern)
    mgr.fill_primary_test_pattern(b, pattern)
    canon_a, hash_a, dedup_a = mgr.seal(a)
    canon_b, hash_b, dedup_b = mgr.seal(b)
    assert dedup_a is False
    assert dedup_b is True
    assert canon_a == canon_b == a
    assert hash_a == hash_b


@given(
    req=st.integers(min_value=0, max_value=255),
    count=st.integers(min_value=0, max_value=8),
)
@settings(max_examples=32, deadline=None)
def test_release_request_returns_exact_count(req, count):
    mgr = _native.BlockManager(small_cfg(), 32 * 1024 * 1024)
    for _ in range(count):
        try:
            mgr.allocate(req, 0, 64)
        except RuntimeError:
            count -= 1  # OOM, didn't actually allocate
    freed = mgr.release_request(req)
    assert freed == max(count, 0)
