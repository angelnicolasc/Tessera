"""Multi-rank orchestration helpers.

The Sprint 3 multi-rank surface composes three native primitives:

* :class:`tessera._native.World` — topology description.
* :class:`tessera._native.MockTransport` (or :class:`P2pCudaTransport`,
  :class:`NcclTransport` in later sprints) — cross-rank message bus.
* :class:`tessera._native.BlockManager.with_world` — per-rank manager bound to the world.

The :class:`MultiRankCoordinator` below stitches them together for the common test pattern:
N block managers + N interconnected mock transports, with each transport's peer slot wired
to the corresponding manager. Tests then orchestrate request lifecycles via the
coordinator's accessors.

For production deployments you replace :class:`tessera._native.MockTransport` with the real
``P2pCudaTransport`` (intra-node NVLink P2P) or ``NcclTransport`` (multi-node IB). The
coordinator shape stays identical — only the transport changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera._native import BlockManager, MockTransport, World
    from tessera.config import TesseraConfig


# Synthetic request id used by mock transports when fulfilling an incoming `push_block`.
# Real deployments will route via the destination's vLLM scheduler instead.
DEFAULT_PUSH_ACCEPT_REQ_ID: int = 1


@dataclass(frozen=True, slots=True)
class MultiRankCoordinator:
    """One world, N block managers, N mock transports wired together.

    Attributes:
        world: Shared :class:`World` (single-node for Sprint 3).
        managers: List of :class:`BlockManager`, one per rank (`managers[r]` is rank r).
        transports: List of :class:`MockTransport`, one per rank, peer-wired to every other.
    """

    world: World
    managers: list[BlockManager]
    transports: list[MockTransport]

    @property
    def world_size(self) -> int:
        return self.world.size

    def __len__(self) -> int:
        return self.world_size


def spawn_multirank_world(
    config: TesseraConfig,
    world_size: int,
    *,
    accept_req_id: int = DEFAULT_PUSH_ACCEPT_REQ_ID,
) -> MultiRankCoordinator:
    """Construct N block managers + N interconnected mock transports.

    Each transport's peer slot for rank `r` is wired to ``managers[r]`` via
    :meth:`MockTransport.register_block_manager_peer`. After construction every transport
    handle can answer cross-rank fetch / push / query_hash on behalf of any rank in the
    world.

    Args:
        config: Tessera configuration. Each manager receives a fresh ``BlockManager`` built
            from this config (the GPU memory budget comes from ``config.runtime.gpu_memory_bytes``).
        world_size: Number of ranks to spawn. Must be ≥ 1.
        accept_req_id: Synthetic request id the destination side uses when accepting pushed
            blocks. Tests can override to make `release_request` operate predictably.

    Returns:
        A :class:`MultiRankCoordinator`. Worlds of size 1 still get a coordinator, but with
        a single rank and no peers — useful for retrocompat scaffolding.

    Raises:
        ValueError: ``world_size < 1``.
    """
    if world_size < 1:
        msg = f"world_size must be >= 1; got {world_size}"
        raise ValueError(msg)

    from tessera import _native

    if world_size == 1:
        world = _native.World.singleton()
        mgr = _native.BlockManager.with_world(
            config.to_native_config(),
            config.runtime.gpu_memory_bytes,
            rank=0,
            world=world,
        )
        transport = _native.MockTransport.singleton()
        transport.register_block_manager_peer(0, mgr, accept_req_id)
        return MultiRankCoordinator(world=world, managers=[mgr], transports=[transport])

    transports = _native.MockTransport.new_world(world_size)
    managers: list[BlockManager] = []
    # Build managers first so we can register each one as the peer on every transport handle.
    worlds = [_native.World.single_node(local=r, size=world_size) for r in range(world_size)]
    for r in range(world_size):
        mgr = _native.BlockManager.with_world(
            config.to_native_config(),
            config.runtime.gpu_memory_bytes,
            rank=r,
            world=worlds[r],
        )
        managers.append(mgr)

    # Cross-wire: every transport handle answers for every rank.
    for handle in transports:
        for r in range(world_size):
            handle.register_block_manager_peer(r, managers[r], accept_req_id)

    return MultiRankCoordinator(
        # All ranks share an equivalent World description; we expose rank-0's for the
        # coordinator's `.world` accessor. Per-rank worlds remain accessible via
        # `managers[r].world` on the Rust side if needed.
        world=worlds[0],
        managers=managers,
        transports=transports,
    )
