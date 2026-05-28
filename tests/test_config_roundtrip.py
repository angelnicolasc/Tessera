"""Configuration roundtrip and validation tests. No native module required."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.config import REQUIRED_BLOCK_SIZE_TOKENS, TesseraConfig


def test_deepseek_v3_loads_and_validates(deepseek_v3_toml: Path) -> None:
    cfg = TesseraConfig.from_toml(deepseek_v3_toml)
    assert cfg.is_mla
    assert cfg.model.latent_dim == 512
    assert cfg.model.rope_key_dim == 64
    assert cfg.model.num_layers == 61
    assert cfg.block.block_size_tokens == REQUIRED_BLOCK_SIZE_TOKENS


def test_compression_ratio_matches_playbook(deepseek_v3_toml: Path) -> None:
    cfg = TesseraConfig.from_toml(deepseek_v3_toml)
    ratio = cfg.compression_ratio_vs_mha_bf16()
    # Playbook §1.2 quotes ~56.9. Allow the 55–58 band.
    assert 55.0 < ratio < 58.0, f"ratio {ratio} out of band"


def test_dump_then_load_is_stable(deepseek_v3_toml: Path) -> None:
    cfg1 = TesseraConfig.from_toml(deepseek_v3_toml)
    cfg2 = TesseraConfig.from_dict(cfg1.model_dump())
    assert cfg1 == cfg2


def test_mla_requires_block_size_64() -> None:
    payload = {
        "model": {
            "name": "test-mla",
            "latent_dim": 512,
            "rope_key_dim": 64,
            "num_layers": 61,
            "num_heads": 128,
            "head_dim": 128,
        },
        "block": {"block_size_tokens": 16, "ckv_dtype": "bf16"},
        "runtime": {"device": 0, "gpu_memory_bytes": 1 << 30},
    }
    with pytest.raises(ValueError, match="block_size_tokens"):
        TesseraConfig.from_dict(payload)


def test_mha_fallback_loads(mha_fallback_toml: Path) -> None:
    cfg = TesseraConfig.from_toml(mha_fallback_toml)
    assert not cfg.is_mla
    assert cfg.model.latent_dim == 0
    assert cfg.model.rope_key_dim == 0
