//! Rank, world and topology types for multi-GPU / tensor-parallel deployments.
//!
//! Every [`crate::block_manager::TesseraBlockManager`] owns exactly one [`RankId`] and a
//! shared handle to a [`World`] describing the deployment topology. Sprint 3 ships the
//! intra-node ([`Topology::SingleNode`]) path fully; multi-node is wired through the type
//! system but [`Topology::MultiNode`] is unused at runtime until Sprint 4. See
//! `docs/src/adr/0014-multi-rank-architecture.md`.

use serde::{Deserialize, Serialize};

/// Stable per-process rank identifier within a world. Wrapping `u32` in a newtype keeps the
/// rank namespace distinct from block ids and request ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RankId(pub u32);

impl RankId {
    /// Convenience: rank-0, the canonical singleton rank for non-distributed deployments.
    pub const ZERO: Self = Self(0);

    /// Raw integer value (for FFI / metric labels).
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for RankId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

impl From<u32> for RankId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<RankId> for u32 {
    fn from(v: RankId) -> Self {
        v.0
    }
}

/// Latency tier classifying the cost of reaching a peer. Used by chaos injection (Sprint 4
/// `LatencyInjector`) and topology-aware budget scaling in `DistributedSegmentIndex`. The
/// ordering reflects increasing wall-clock cost: `IntraNode < IntraRack < CrossRack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LatencyTier {
    /// Same node (NVLink P2P territory in production; ~5 µs typical).
    IntraNode,
    /// Different node, same rack (NVSwitch / NVLink Switch System; ~50 µs typical).
    IntraRack,
    /// Different rack (InfiniBand / RoCE; 500 µs+ typical).
    CrossRack,
}

impl LatencyTier {
    /// Short metric-label-friendly identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IntraNode => "intra_node",
            Self::IntraRack => "intra_rack",
            Self::CrossRack => "cross_rack",
        }
    }
}

/// Stable per-node identifier. Reserved for [`Topology::MultiNode`] in Sprint 4; the type
/// exists today so the public API of [`World`] does not change when multi-node lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Raw integer value.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Deployment topology. `SingleNode` covers everything Sprint 3 implements; `MultiNode` is
/// cabled into the type system but its transport (NCCL) is deferred to Sprint 4. See
/// `docs/src/adr/0015-p2p-vs-nccl-transport.md` for the selection rationale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Topology {
    /// All ranks live on a single node. Cross-rank transfers go over NVLink P2P (real
    /// implementation) or in-process channels (test mock).
    SingleNode,

    /// Multi-node deployment. `node_of[rank.0 as usize]` maps each rank to its node. Cross-
    /// node transfers go through NCCL; intra-node transfers still use P2P. The
    /// `NcclTransport` runtime impl is deferred to Sprint 4 — the variant is exposed today
    /// so configs can be authored against the stable API surface.
    MultiNode {
        /// Mapping from `RankId.0` to its hosting `NodeId`. Length must equal `World::size`.
        node_of: Vec<NodeId>,
    },
}

impl Topology {
    /// Whether this topology spans more than one node.
    pub fn is_multi_node(&self) -> bool {
        matches!(self, Self::MultiNode { .. })
    }

    /// Node id hosting the given rank under this topology. Returns `None` for ranks out of
    /// range. Under [`Topology::SingleNode`] every rank reports `NodeId(0)`.
    pub fn node_of(&self, rank: RankId) -> Option<NodeId> {
        match self {
            Self::SingleNode => Some(NodeId(0)),
            Self::MultiNode { node_of } => node_of.get(rank.raw() as usize).copied(),
        }
    }

    /// Whether two ranks live on the same node. Falls back to `false` if either rank is
    /// out of range (defensive: classifying an unknown peer as remote is safer than the
    /// reverse).
    pub fn is_same_node(&self, a: RankId, b: RankId) -> bool {
        match (self.node_of(a), self.node_of(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }
}

/// Description of the surrounding multi-rank deployment as observed by one process. Shared
/// across all collaborating components (block manager, distributed segment index, transport,
/// vLLM plugin) via `Arc<World>` so that updates to the world remain a single point of
/// truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct World {
    /// This process's rank.
    pub local: RankId,
    /// Total number of ranks in the world.
    pub size: u32,
    /// Topology — drives transport selection.
    pub topology: Topology,
}

impl World {
    /// Construct a world from explicit parts. Returns `None` if `local.raw() >= size` or if a
    /// multi-node mapping has the wrong length.
    pub fn new(local: RankId, size: u32, topology: Topology) -> Option<Self> {
        if local.raw() >= size {
            return None;
        }
        if let Topology::MultiNode { ref node_of } = topology {
            if node_of.len() != size as usize {
                return None;
            }
        }
        Some(Self {
            local,
            size,
            topology,
        })
    }

    /// Canonical single-rank world. Used by [`crate::block_manager::TesseraBlockManager::new`]
    /// for non-distributed deployments and by every Sprint 0-2 test.
    pub const fn singleton() -> Self {
        Self {
            local: RankId::ZERO,
            size: 1,
            topology: Topology::SingleNode,
        }
    }

    /// All non-local ranks. Useful for fan-out (e.g. broadcast on seal).
    pub fn peers(&self) -> impl Iterator<Item = RankId> + '_ {
        (0..self.size).map(RankId).filter(move |r| *r != self.local)
    }

    /// Whether this world has any peers (i.e. `size > 1`).
    pub const fn has_peers(&self) -> bool {
        self.size > 1
    }

    /// Latency tier classifying the cost of reaching `other` from this rank. Returns
    /// `LatencyTier::IntraNode` for self-references — callers should not normally invoke
    /// with `other == self.local`.
    pub fn peer_tier(&self, other: RankId) -> LatencyTier {
        if other == self.local {
            return LatencyTier::IntraNode;
        }
        if self.topology.is_same_node(self.local, other) {
            LatencyTier::IntraNode
        } else {
            // Sprint 4 doesn't yet model rack-vs-cross-rack at the World level (no
            // `rack_of` mapping); any cross-node hop is classified as IntraRack. A future
            // extension can introduce CrossRack by adding rack metadata to Topology.
            LatencyTier::IntraRack
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::singleton()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_is_rank0_size1() {
        let w = World::singleton();
        assert_eq!(w.local, RankId::ZERO);
        assert_eq!(w.size, 1);
        assert!(!w.has_peers());
        assert_eq!(w.peers().count(), 0);
    }

    #[test]
    fn rank_display_uses_r_prefix() {
        assert_eq!(RankId(7).to_string(), "r7");
        assert_eq!(NodeId(2).to_string(), "n2");
    }

    #[test]
    fn world_new_rejects_out_of_range_local() {
        assert!(World::new(RankId(4), 4, Topology::SingleNode).is_none());
        assert!(World::new(RankId(3), 4, Topology::SingleNode).is_some());
    }

    #[test]
    fn world_new_rejects_inconsistent_node_mapping() {
        let bad = World::new(
            RankId(0),
            4,
            Topology::MultiNode {
                node_of: vec![NodeId(0); 3], // wrong length
            },
        );
        assert!(bad.is_none());
        let good = World::new(
            RankId(0),
            4,
            Topology::MultiNode {
                node_of: vec![NodeId(0), NodeId(0), NodeId(1), NodeId(1)],
            },
        );
        assert!(good.is_some());
    }

    #[test]
    fn peers_excludes_local() {
        let w = World::new(RankId(1), 4, Topology::SingleNode).unwrap();
        let peers: Vec<_> = w.peers().collect();
        assert_eq!(peers, vec![RankId(0), RankId(2), RankId(3)]);
    }
}
