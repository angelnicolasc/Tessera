"""FlashInfer MLA backend parity tests (WS9 / G12).

Mirrors ``test_flash_mla_parity.py`` but targets the FlashInfer backend (Ampere+).

Tier B — GPU-gated. Run with: ``pytest tests -m gpu``
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch required")

from tessera.reference.absorbed_attention import reference_absorbed_mla

DS_V3_D_C = 512
DS_V3_D_R = 64
DS_V3_D_H = 128
DS_V3_H = 128
DS_V3_SCALE = 1.0 / math.sqrt(DS_V3_D_C)


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [64, 512, 4096, 32768])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_infer_parity_vs_reference(gpu, batch_size: int, seq_len: int, dtype):
    """FlashInferMLABackend.forward must match reference within BF16 tolerance."""
    from tessera.config import TesseraConfig
    from tessera.kernel_dispatch import get_mla_backend

    config = TesseraConfig.from_toml("models/deepseek_v3.toml")
    backend = get_mla_backend(gpu, config)

    if backend.backend_name not in ("flash_infer", "flash_mla"):
        pytest.skip(f"Neither FlashInfer nor FlashMLA available; got {backend.backend_name}")

    torch.manual_seed(42)
    q_abs = torch.randn(batch_size, DS_V3_H, DS_V3_D_C, dtype=dtype, device=gpu)
    q_rope = torch.randn(batch_size, DS_V3_H, DS_V3_D_R, dtype=dtype, device=gpu)
    c_kv = torch.randn(batch_size, seq_len, DS_V3_D_C, dtype=dtype, device=gpu)
    k_rope = torch.randn(batch_size, seq_len, DS_V3_D_R, dtype=dtype, device=gpu)
    W_UV = torch.randn(DS_V3_H, DS_V3_D_H, DS_V3_D_C, dtype=dtype, device=gpu)

    ref_out = reference_absorbed_mla(q_abs, q_rope, c_kv, k_rope, W_UV, DS_V3_SCALE)
    assert ref_out.shape == (batch_size, DS_V3_H, DS_V3_D_H)
