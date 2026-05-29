"""Kernel throughput benchmark stub. Requires a CUDA-capable GPU.

Marked ``slow + gpu`` so it is skipped on CPU-only CI runners. Implementing the actual
FlashMLA invocation is Sprint 1 work; Sprint 0 ships the harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import pytest

from tessera.config import TesseraConfig
from tessera.kernel_dispatch import get_mla_backend


@pytest.mark.gpu
@pytest.mark.slow
def test_kernel_throughput_smoke() -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    config = TesseraConfig.from_toml(ROOT / "models" / "deepseek_v3.toml")
    backend = get_mla_backend(config)
    assert backend.is_available(), f"backend {backend.backend_name} not importable"
    # Real measurement loop deferred to Sprint 1 (see issue tracker).
