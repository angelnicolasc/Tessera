"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


@pytest.fixture(scope="session")
def deepseek_v3_toml(repo_root: Path) -> Path:
    """Path to the DeepSeek-V3 canonical config."""
    return repo_root / "models" / "deepseek_v3.toml"


@pytest.fixture(scope="session")
def mha_fallback_toml(repo_root: Path) -> Path:
    return repo_root / "models" / "mha_fallback.toml"


@pytest.fixture
def native():
    """Native ``tessera._native`` module. Skips when the extension isn't built."""
    try:
        from tessera import _native
    except ImportError:
        pytest.skip("tessera._native not built — run `maturin develop`")
    return _native
