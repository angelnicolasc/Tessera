"""GPU test fixtures. Imported by all tests in tests/gpu/.

All GPU tests depend on the ``gpu`` fixture, which auto-skips when CUDA is unavailable.
This ensures ``pytest tests -m "not gpu"`` runs cleanly on CPU-only machines, and
``pytest tests -m gpu`` just works on a CUDA node — no other changes needed.

WS9 additions: ``ds_v3_config`` and ``ds_v3_manager`` fixtures load the DeepSeek-V3 model
config and construct a live ``BlockManager`` for integration tests against a real GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu():
    """Session-scoped fixture that skips the test if CUDA is unavailable.

    Usage::

        def test_something(gpu):
            # only runs when a CUDA device is present
            ...
    """
    if not _cuda_available():
        pytest.skip("CUDA device required for GPU tests; skipping on CPU-only machine")
    import torch

    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def ds_v3_scale() -> float:
    """Softmax scale for DeepSeek-V3 config (1 / sqrt(d_c=512))."""
    import math

    return 1.0 / math.sqrt(512)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def ds_v3_config(repo_root: Path):
    """Session-scoped TesseraConfig loaded from models/deepseek_v3.toml.

    Available on both CPU and GPU — the config itself does not require a device. Tests that
    need a GPU should depend on both ``gpu`` and ``ds_v3_config``.
    """
    from tessera.config import TesseraConfig

    toml_path = repo_root / "models" / "deepseek_v3.toml"
    return TesseraConfig.from_toml(toml_path)


@pytest.fixture(scope="session")
def ds_v3_manager(ds_v3_config, gpu):  # type: ignore[no-untyped-def]
    """Session-scoped BlockManager constructed from the DeepSeek-V3 config.

    Depends on ``gpu`` so it auto-skips on CPU-only machines. The manager uses the
    GPU memory budget from ``ds_v3_config.runtime.gpu_memory_bytes``.
    """
    from tessera import _native

    native_cfg = ds_v3_config.to_native_config()
    return _native.BlockManager(native_cfg, ds_v3_config.runtime.gpu_memory_bytes)
