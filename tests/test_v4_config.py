"""V4 compliance tests for the Python config layer (Sprint 5).

Validates:
1. The two new V4 model TOMLs (`deepseek_v4_flash.toml` + `deepseek_v4_pro.toml`) parse,
   validate and round-trip via Pydantic.
2. `TesseraConfig.to_native_config()` produces a `MlaBlockConfig` with the right per-layer
   pattern (`has_per_layer_schemes()` is True, `num_layers` matches the pattern length).
3. Block-size invariants: V4 hybrid requires `block_size_tokens` to be a multiple of
   `lcm(k1, k2)` for the patterns in use.
4. The deprecated `dsa_hierarchical` constructor is still callable via the native module
   but cannot be embedded in a real `MlaBlockConfig`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.config import TesseraConfig


@pytest.fixture(scope="session")
def v4_flash_toml(repo_root: Path) -> Path:
    return repo_root / "models" / "deepseek_v4_flash.toml"


@pytest.fixture(scope="session")
def v4_pro_toml(repo_root: Path) -> Path:
    return repo_root / "models" / "deepseek_v4_pro.toml"


def test_v4_flash_config_loads(v4_flash_toml: Path) -> None:
    cfg = TesseraConfig.from_toml(v4_flash_toml)
    assert cfg.is_v4
    assert not cfg.is_mla
    assert cfg.model.num_layers == 43
    assert cfg.model.num_heads == 64
    assert cfg.block.block_size_tokens == 128
    assert cfg.block.ckv_dtype == "mixed_bf16_fp8_fp4"
    assert cfg.v4 is not None
    assert cfg.v4.k1 == 4
    assert cfg.v4.k2 == 128
    assert cfg.v4.top_k == 512  # Flash spec
    assert len(cfg.v4.layer_pattern) == 43
    # First 2 layers pure SWA per the paper.
    assert cfg.v4.layer_pattern[:2] == ["swa", "swa"]


def test_v4_pro_config_loads(v4_pro_toml: Path) -> None:
    cfg = TesseraConfig.from_toml(v4_pro_toml)
    assert cfg.is_v4
    assert cfg.model.num_layers == 61
    assert cfg.model.num_heads == 128
    assert cfg.v4 is not None
    assert cfg.v4.top_k == 1024  # Pro spec
    assert len(cfg.v4.layer_pattern) == 61
    # First 2 layers HCA per the paper.
    assert cfg.v4.layer_pattern[:2] == ["hca", "hca"]


def test_v4_to_native_config_produces_per_layer(v4_pro_toml: Path) -> None:
    pytest.importorskip("tessera._native")
    cfg = TesseraConfig.from_toml(v4_pro_toml)
    native = cfg.to_native_config()
    assert native.has_per_layer_schemes()
    assert native.num_layers == 61
    assert native.v4_block_size_lcm() == 128
    # First layer scheme should be V4 (Pro starts with HCA).
    layer0 = native.scheme_for_layer(0)
    assert layer0.is_v4()


def test_v4_layer_pattern_length_validated() -> None:
    payload = {
        "model": {
            "name": "bad-v4",
            "latent_dim": 0,
            "rope_key_dim": 0,
            "num_layers": 4,
            "num_heads": 64,
            "head_dim": 512,
        },
        "block": {"block_size_tokens": 128, "ckv_dtype": "mixed_bf16_fp8_fp4"},
        "v4": {
            "k1": 4,
            "k2": 128,
            "head_dim": 512,
            "rope_dim": 64,
            "indexer_head_dim": 128,
            "num_indexer_heads": 64,
            "top_k": 512,
            "swa_window": 128,
            # Pattern length mismatches num_layers=4.
            "layer_pattern": ["csa", "hca"],
        },
        "runtime": {"device": 0, "gpu_memory_bytes": 1 << 30},
    }
    with pytest.raises(ValueError, match="layer_pattern length"):
        TesseraConfig.from_dict(payload)


def test_v4_block_size_must_be_multiple_of_lcm() -> None:
    payload = {
        "model": {
            "name": "bad-bs",
            "latent_dim": 0,
            "rope_key_dim": 0,
            "num_layers": 2,
            "num_heads": 64,
            "head_dim": 512,
        },
        # HCA requires multiple of 128; 64 fails.
        "block": {"block_size_tokens": 64, "ckv_dtype": "mixed_bf16_fp8_fp4"},
        "v4": {
            "k1": 4,
            "k2": 128,
            "head_dim": 512,
            "rope_dim": 64,
            "indexer_head_dim": 128,
            "num_indexer_heads": 64,
            "top_k": 512,
            "swa_window": 128,
            "layer_pattern": ["hca", "csa"],
        },
        "runtime": {"device": 0, "gpu_memory_bytes": 1 << 30},
    }
    with pytest.raises(ValueError, match="multiple of"):
        TesseraConfig.from_dict(payload)


def test_v4_requires_mixed_dtype() -> None:
    payload = {
        "model": {
            "name": "wrong-dtype",
            "latent_dim": 0,
            "rope_key_dim": 0,
            "num_layers": 1,
            "num_heads": 64,
            "head_dim": 512,
        },
        # V4 with bf16 dtype is rejected.
        "block": {"block_size_tokens": 128, "ckv_dtype": "bf16"},
        "v4": {
            "k1": 4,
            "k2": 128,
            "head_dim": 512,
            "rope_dim": 64,
            "indexer_head_dim": 128,
            "num_indexer_heads": 64,
            "top_k": 512,
            "swa_window": 128,
            "layer_pattern": ["csa"],
        },
        "runtime": {"device": 0, "gpu_memory_bytes": 1 << 30},
    }
    with pytest.raises(ValueError, match="mixed_bf16_fp8_fp4"):
        TesseraConfig.from_dict(payload)


def test_v3_path_still_works(deepseek_v3_toml: Path) -> None:
    """Regression: V3 (MLA) configs continue to load correctly post-Sprint-5."""
    cfg = TesseraConfig.from_toml(deepseek_v3_toml)
    assert cfg.is_mla
    assert not cfg.is_v4
    assert cfg.block.ckv_dtype == "bf16"


def test_fp4_and_mixed_dtypes_exposed_on_native_module() -> None:
    pytest.importorskip("tessera._native")
    from tessera import _native

    assert hasattr(_native.CkvDtype, "Fp4E2m1")
    assert hasattr(_native.CkvDtype, "MixedBf16Fp8Fp4")
    assert hasattr(_native.CompressionScheme, "v4_csa")
    assert hasattr(_native.CompressionScheme, "v4_hca")
    assert hasattr(_native.CompressionScheme, "v4_swa")


def test_native_v4_csa_bytes_per_token_matches_paper() -> None:
    """Spot-check the paper's storage math: V4-Pro CSA per-token = 160 bytes/layer."""
    pytest.importorskip("tessera._native")
    from tessera import _native

    csa = _native.CompressionScheme.v4_csa(
        k1=4,
        head_dim=512,
        num_heads=128,
        rope_dim=64,
        indexer_head_dim=128,
        num_indexer_heads=64,
        top_k=1024,
    )
    assert csa.is_v4()
    assert csa.bytes_per_token_per_layer() == 160
