"""Multiprocessing-backed isolation smoke test. Validates that the rank-aware constructor
works across true process boundaries (no GIL contention, separate address spaces).

Sprint 3 ships a CPU-only validation: each worker spawns its own ``BlockManager`` instance
with a distinct rank and reports its `used_blocks` count back to the parent. Cross-rank
**transport** semantics still require a shared `MockTransport`, which doesn't survive
process boundaries — that's a Sprint 4 concern (real `P2pCudaTransport` solves it via
CUDA IPC). For now, isolation is the testable invariant.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import pytest

from tessera.config import TesseraConfig


def _worker(rank: int, world_size: int, config_payload: dict, q: Any) -> None:
    """Constructed in a child process; reports `(rank, used_blocks)` after one allocation."""
    # Imports must happen in the child to avoid pickling extension types.
    from tessera import _native

    cfg = TesseraConfig.from_dict(config_payload)
    world = _native.World.single_node(local=rank, size=world_size)
    mgr = _native.BlockManager.with_world(
        cfg.to_native_config(),
        cfg.runtime.gpu_memory_bytes,
        rank=rank,
        world=world,
    )
    mgr.allocate(req_id=rank, token_start=0, token_end=64)
    q.put((rank, mgr.used_blocks, mgr.rank))


@pytest.mark.integration
def test_multiprocess_workers_isolate_state() -> None:
    payload = {
        "model": {
            "name": "mp-test",
            "latent_dim": 32,
            "rope_key_dim": 8,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 32,
        },
        "block": {"block_size_tokens": 64, "ckv_dtype": "bf16"},
        "kernel": {"backend": "triton"},
        "runtime": {"device": 0, "gpu_memory_bytes": 16 * 1024 * 1024},
    }
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 4, payload, q)) for r in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

    results = sorted([q.get(timeout=5) for _ in range(4)])
    # Each worker allocated exactly 1 block and reported its own rank.
    assert results == [(0, 1, 0), (1, 1, 1), (2, 1, 2), (3, 1, 3)]
