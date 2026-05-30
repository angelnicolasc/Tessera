"""WS2 — FP8 production pipeline tests (TD-018).

Covers:
- save_scales / load_scales round-trip
- TesseraConfig.fp8_scales lazy-loading from file
- fp8_scales returns None for BF16 configs
- fp8_scales returns None when path does not exist
- post_prefill_seal with an FP8 block manager does not raise
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tessera import _native
from tessera.config import TesseraConfig
from tessera.fp8_calibrate import load_scales, save_scales

# ──────────── helpers ─────────────────────────────────────────────────────────


def _fp8_config_dict(fp8_scales_path: str | None = None) -> dict:
    return {
        "model": {
            "name": "test-fp8",
            "latent_dim": 32,
            "rope_key_dim": 8,
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 128,
        },
        "block": {
            "block_size_tokens": 64,
            "ckv_dtype": "fp8_e4m3",
            **({"fp8_scales_path": fp8_scales_path} if fp8_scales_path else {}),
        },
        "runtime": {"gpu_memory_bytes": 128 * 1024 * 1024},
    }


def _bf16_config_dict() -> dict:
    return {
        "model": {
            "name": "test-bf16",
            "latent_dim": 32,
            "rope_key_dim": 8,
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 128,
        },
        "block": {"block_size_tokens": 64, "ckv_dtype": "bf16"},
        "runtime": {"gpu_memory_bytes": 128 * 1024 * 1024},
    }


# ──────────── tests ────────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    scales_file = tmp_path / "scales.json"
    original = {0: 0.00123, 1: 0.00456, 2: 0.00789}
    save_scales(original, scales_file)

    loaded = load_scales(scales_file)
    assert loaded == original


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    scales_file = tmp_path / "deep" / "nested" / "scales.json"
    save_scales({0: 1.0}, scales_file)
    assert scales_file.exists()
    data = json.loads(scales_file.read_text())
    assert data == {"0": 1.0}


def test_config_loads_scales_from_path(tmp_path: Path) -> None:
    scales_file = tmp_path / "scales.json"
    expected = {0: 0.001, 1: 0.002, 2: 0.003, 3: 0.004}
    save_scales(expected, scales_file)

    cfg = TesseraConfig.from_dict(_fp8_config_dict(str(scales_file)))
    scales = cfg.fp8_scales
    assert scales is not None
    assert scales == expected


def test_config_fp8_scales_cached_across_calls(tmp_path: Path) -> None:
    scales_file = tmp_path / "scales.json"
    save_scales({0: 0.5}, scales_file)

    cfg = TesseraConfig.from_dict(_fp8_config_dict(str(scales_file)))
    first = cfg.fp8_scales
    second = cfg.fp8_scales
    assert first is second  # same object — lazy cache


def test_config_fp8_scales_none_when_bf16() -> None:
    cfg = TesseraConfig.from_dict(_bf16_config_dict())
    assert cfg.fp8_scales is None


def test_config_fp8_scales_none_when_path_absent() -> None:
    cfg = TesseraConfig.from_dict(_fp8_config_dict(fp8_scales_path=None))
    assert cfg.fp8_scales is None


def test_config_fp8_scales_none_when_file_missing(tmp_path: Path) -> None:
    missing = str(tmp_path / "no_such_file.json")
    cfg = TesseraConfig.from_dict(_fp8_config_dict(fp8_scales_path=missing))
    assert cfg.fp8_scales is None


def test_post_prefill_seal_no_exception_fp8(tmp_path: Path) -> None:
    """post_prefill_seal with an FP8 block + loaded scales does not raise."""
    scales_file = tmp_path / "scales.json"
    save_scales({i: float(i + 1) * 0.001 for i in range(4)}, scales_file)

    from tessera.vllm_plugin import TesseraBlockAllocator

    cfg = TesseraConfig.from_dict(_fp8_config_dict(str(scales_file)))
    allocator = TesseraBlockAllocator(cfg)

    # Allocate and fill a block so it can be sealed.
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Fp8E4m3, 0)
    block_id = allocator._manager.allocate(1, 0, 64)
    allocator._manager.fill_primary_test_pattern(block_id, 0xAB)

    # Shape must match `[num_layers, block_size_tokens, d_c]` (4, 64, 32) — the segment
    # index's descriptor_from_ckv asserts ndim==3 and reduces over (layer, token).
    c_kv = np.zeros((4, 64, 32), dtype=np.float32)
    # Must not raise — _write_fp8_scales is a no-op on the CPU mock.
    canonical_id = allocator.post_prefill_seal(block_id, c_kv)
    assert isinstance(canonical_id, int)
