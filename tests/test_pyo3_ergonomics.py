"""WS1 — PyO3 ergonomics tests (TD-016).

Verifies that all PyO3-exposed classes have working __repr__, __len__, __sizeof__.
These make the objects usable in debug sessions (print, len(), sys.getsizeof()).
No GPU required — runs entirely on the CPU mock backend.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera import _native


def _make_bf16_config() -> _native.MlaBlockConfig:
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    return _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Bf16, 0)


def _make_manager(n_blocks: int = 32) -> _native.BlockManager:
    cfg = _make_bf16_config()
    return _native.BlockManager(cfg, n_blocks * cfg.total_block_bytes())


def test_block_manager_repr_contains_class_name() -> None:
    mgr = _make_manager()
    r = repr(mgr)
    assert "BlockManager" in r


def test_block_manager_repr_reflects_used_blocks() -> None:
    mgr = _make_manager(32)
    bid = mgr.allocate(1, 0, 64)
    mgr.fill_primary_test_pattern(bid, 0xAA)
    mgr.seal(bid)
    r = repr(mgr)
    assert "used=1" in r


def test_block_manager_len_equals_used_blocks() -> None:
    mgr = _make_manager(32)
    assert len(mgr) == 0
    bid = mgr.allocate(1, 0, 64)
    mgr.fill_primary_test_pattern(bid, 0x01)
    mgr.seal(bid)
    assert len(mgr) == 1
    mgr.release_request(1)
    assert len(mgr) == 0


def test_block_manager_sizeof_positive() -> None:
    mgr = _make_manager()
    assert mgr.__sizeof__() > 0


def test_share_table_repr_contains_class_name() -> None:
    st = _native.ShareTable()
    r = repr(st)
    assert "ShareTable" in r


def test_share_table_repr_and_len() -> None:
    mgr = _make_manager(32)
    st = _native.ShareTable()

    bid = mgr.allocate(1, 0, 64)
    mgr.fill_primary_test_pattern(bid, 0x11)
    canon, _, _ = mgr.seal(bid)

    st.add_share(req_id=1, block_id=canon)
    st.add_share(req_id=2, block_id=canon)

    assert len(st) == 1  # one shared block (canon), two owners
    assert "ShareTable" in repr(st)
    assert "shared_blocks=1" in repr(st)


def test_ckv_dtype_repr_bf16() -> None:
    r = repr(_native.CkvDtype.Bf16)
    assert "Bf16" in r


def test_ckv_dtype_repr_fp8() -> None:
    r = repr(_native.CkvDtype.Fp8E4m3)
    assert "Fp8E4m3" in r


def test_mla_block_config_repr() -> None:
    cfg = _make_bf16_config()
    r = repr(cfg)
    assert len(r) > 0  # MlaBlockConfig delegates to Rust Debug; just check non-empty


def test_mla_block_config_sizeof_positive() -> None:
    cfg = _make_bf16_config()
    assert cfg.__sizeof__() > 0


def test_usearch_index_repr_after_add() -> None:
    idx = _native.UsearchIndex(16)
    descriptor = np.random.rand(16).astype(np.float32)
    idx.add(42, descriptor)
    r = repr(idx)
    assert "UsearchIndex" in r
    assert "len=1" in r


def test_fp8_scales_ptr_none_for_bf16() -> None:
    """fp8_scales_ptr returns None when ckv_dtype is BF16."""
    mgr = _make_manager(8)
    bid = mgr.allocate(1, 0, 64)
    assert mgr.fp8_scales_ptr(bid) is None


def test_fp8_scales_ptr_returns_int_for_fp8() -> None:
    """fp8_scales_ptr returns an int (pointer as usize) when ckv_dtype is FP8."""
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    cfg = _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Fp8E4m3, 0)
    mgr = _native.BlockManager(cfg, 16 * cfg.total_block_bytes())
    bid = mgr.allocate(1, 0, 64)
    ptr = mgr.fp8_scales_ptr(bid)
    assert ptr is not None
    assert isinstance(ptr, int)
    assert ptr > 0
