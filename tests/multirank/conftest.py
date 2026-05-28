"""Shared fixtures for the multi-rank Python test suite. CPU-only via MockTransport +
CpuMockBackend; no GPU required."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.config import TesseraConfig
from tessera.multi_rank import MultiRankCoordinator, spawn_multirank_world


@pytest.fixture(scope="session")
def small_mla_config(repo_root: Path) -> TesseraConfig:
    """Tiny MLA config: 4 layers, d_c=32, d_r=8, block=64. Keeps each block ≤ 16 KiB so
    the CpuMockBackend's allocations stay snappy across 4 ranks."""
    payload = {
        "model": {
            "name": "tp4-test",
            "latent_dim": 32,
            "rope_key_dim": 8,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 32,
        },
        "block": {"block_size_tokens": 64, "ckv_dtype": "bf16"},
        "segment_index": {
            "hnsw_m": 16,
            "hnsw_ef_construction": 100,
            "hnsw_ef_search": 32,
            "similarity_threshold": 0.97,
            "hnsw_latency_budget_us": 500,
        },
        "kernel": {"backend": "triton"},
        "runtime": {"device": 0, "gpu_memory_bytes": 32 * 1024 * 1024},
    }
    return TesseraConfig.from_dict(payload)


@pytest.fixture
def tp4_world(small_mla_config: TesseraConfig) -> MultiRankCoordinator:
    """4-rank world wired via MockTransport. Each transport handle's peer-slots are
    populated with the corresponding BlockManager so cross-rank ops work end-to-end."""
    return spawn_multirank_world(small_mla_config, world_size=4)


@pytest.fixture
def tp2_world(small_mla_config: TesseraConfig) -> MultiRankCoordinator:
    """2-rank world, useful for tests that only need src ↔ dst semantics."""
    return spawn_multirank_world(small_mla_config, world_size=2)
