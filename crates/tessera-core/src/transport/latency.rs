//! `LatencyInjector` — wrap any [`RankTransport`] with a deterministic latency / drop
//! profile. Used by Sprint 4's chaos test suite to validate that the distributed protocols
//! survive realistic multi-node failure modes without requiring real hardware.
//!
//! Three injection knobs:
//!
//! * **Tier-aware latency** — `intra_node_us` / `intra_rack_us` / `cross_rack_us` simulate
//!   the latency distribution of real NVLink (intra-node), NVSwitch (intra-rack), and IB
//!   (cross-rack). The injector picks the appropriate tier per call by consulting the
//!   wrapped transport's [`Topology`] + destination rank.
//! * **Jitter** — `jitter_us` adds a uniform `±jitter_us` to each call's sleep. Drives
//!   tests of budget-aware timeouts; with `jitter_us == 0` the injector is fully
//!   deterministic given a fixed seed.
//! * **Drop rate** — `drop_rate ∈ [0.0, 1.0]` causes each call to fail with
//!   `anyhow!("simulated drop")` with the given probability. The Tessera contract treats
//!   transport failures as recoverable (caller retries or aborts cleanly), so the chaos
//!   suite uses this to verify rollback semantics in `transfer_request_to_rank`.
//!
//! Determinism: the injector owns a `Mutex<ChaCha8Rng>` seeded explicitly. Tests that need
//! reproducible failures use `LatencyInjector::new(transport, profile, seed)`; production
//! / non-test uses can fall back to `LatencyInjector::with_entropy(...)` which seeds from
//! the OS entropy pool.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::block::BlockId;
use crate::rank::{LatencyTier, RankId, Topology};

use super::{BlockPayload, RankTransport};

/// Tunable latency profile for [`LatencyInjector`]. Defaults to realistic numbers for a
/// single-node NVLink deployment; multi-node simulations bump the cross-rack tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyProfile {
    /// Base latency for peers on the same node (NVLink P2P territory).
    pub intra_node_us: u64,
    /// Base latency for peers on different nodes within the same rack (NVSwitch).
    pub intra_rack_us: u64,
    /// Base latency for peers in a different rack (IB).
    pub cross_rack_us: u64,
    /// Symmetric jitter range applied per call (in microseconds). Set to 0 for fully
    /// deterministic latency given a fixed seed.
    pub jitter_us: u64,
    /// Probability ∈ [0.0, 1.0] that any individual transport call fails with a simulated
    /// drop error. Tessera's contract treats such failures as recoverable.
    pub drop_rate: f32,
}

impl LatencyProfile {
    /// Realistic intra-node profile: ~5 µs NVLink baseline, no jitter, no drops. Useful as
    /// a sanity-test default.
    pub const INTRA_NODE_REALISTIC: Self = Self {
        intra_node_us: 5,
        intra_rack_us: 50,
        cross_rack_us: 500,
        jitter_us: 0,
        drop_rate: 0.0,
    };

    /// Multi-rack stress profile: large cross-rack base + jitter + occasional drops.
    pub const STRESS_MULTI_RACK: Self = Self {
        intra_node_us: 10,
        intra_rack_us: 100,
        cross_rack_us: 800,
        jitter_us: 200,
        drop_rate: 0.05,
    };

    /// Pure chaos: every call drops. Useful for negative tests (verify graceful failure).
    pub const ALL_DROPS: Self = Self {
        intra_node_us: 0,
        intra_rack_us: 0,
        cross_rack_us: 0,
        jitter_us: 0,
        drop_rate: 1.0,
    };

    /// Zero-latency, zero-drop profile. Equivalent to no injection, useful as a baseline.
    pub const ZERO: Self = Self {
        intra_node_us: 0,
        intra_rack_us: 0,
        cross_rack_us: 0,
        jitter_us: 0,
        drop_rate: 0.0,
    };

    /// Base latency in microseconds for a given tier.
    pub const fn base_us(&self, tier: LatencyTier) -> u64 {
        match tier {
            LatencyTier::IntraNode => self.intra_node_us,
            LatencyTier::IntraRack => self.intra_rack_us,
            LatencyTier::CrossRack => self.cross_rack_us,
        }
    }
}

impl Default for LatencyProfile {
    fn default() -> Self {
        Self::INTRA_NODE_REALISTIC
    }
}

/// Wraps any [`RankTransport`] implementation with a [`LatencyProfile`]. Calls are routed
/// through the inner transport unmodified except for the injected sleep + drop decision.
pub struct LatencyInjector<T: RankTransport> {
    inner: Arc<T>,
    profile: LatencyProfile,
    /// Local-rank assignment, used to classify peer tiers via the world's topology when
    /// the inner transport doesn't carry that information itself.
    local: RankId,
    topology: Topology,
    rng: Mutex<ChaCha8Rng>,
}

impl<T: RankTransport> std::fmt::Debug for LatencyInjector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatencyInjector")
            .field("inner_name", &self.inner.name())
            .field("profile", &self.profile)
            .field("local", &self.local)
            .field("topology_is_multi_node", &self.topology.is_multi_node())
            .finish()
    }
}

impl<T: RankTransport> LatencyInjector<T> {
    /// Construct an injector with explicit seed (deterministic). Use this in tests.
    pub fn new(
        inner: Arc<T>,
        profile: LatencyProfile,
        local: RankId,
        topology: Topology,
        seed: u64,
    ) -> Self {
        Self {
            inner,
            profile,
            local,
            topology,
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }

    /// Construct an injector seeded from the OS entropy pool. Use this in production
    /// staging chaos rigs where reproducibility is undesirable.
    pub fn with_entropy(
        inner: Arc<T>,
        profile: LatencyProfile,
        local: RankId,
        topology: Topology,
    ) -> Self {
        let mut seeder = rand::thread_rng();
        let seed = seeder.next_u64();
        Self::new(inner, profile, local, topology, seed)
    }

    /// Borrow the wrapped transport (for introspection).
    pub fn inner(&self) -> &Arc<T> {
        &self.inner
    }

    /// Borrow the active profile.
    pub const fn profile(&self) -> &LatencyProfile {
        &self.profile
    }

    fn tier_for(&self, peer: RankId) -> LatencyTier {
        // Re-implement the tier resolution here rather than going through `World::peer_tier`
        // because the injector is constructed with a snapshot of the topology and may
        // outlive any specific `World` instance.
        match &self.topology {
            Topology::SingleNode => LatencyTier::IntraNode,
            Topology::MultiNode { node_of } => {
                let local_node = node_of.get(self.local.raw() as usize);
                let peer_node = node_of.get(peer.raw() as usize);
                match (local_node, peer_node) {
                    (Some(a), Some(b)) if a == b => LatencyTier::IntraNode,
                    // Sprint 4 doesn't yet model rack-vs-non-rack at the World level;
                    // anything cross-node is treated as IntraRack. CrossRack is reserved
                    // for an explicit future extension that adds `rack_of` mappings.
                    (Some(_), Some(_)) => LatencyTier::IntraRack,
                    _ => LatencyTier::IntraRack,
                }
            }
        }
    }

    /// Sample a delay for `peer` (base tier latency + symmetric jitter). Result is rounded
    /// to whole microseconds.
    fn sample_delay(&self, peer: RankId) -> Duration {
        let tier = self.tier_for(peer);
        let base = self.profile.base_us(tier);
        if self.profile.jitter_us == 0 {
            return Duration::from_micros(base);
        }
        let mut rng = self.rng.lock();
        let jitter_range = self.profile.jitter_us as i64;
        let delta = rng.gen_range(-jitter_range..=jitter_range);
        let total = base as i64 + delta;
        Duration::from_micros(total.max(0) as u64)
    }

    /// Decide whether this call should be dropped. `drop_rate == 0.0` always returns false.
    fn should_drop(&self) -> bool {
        if self.profile.drop_rate <= 0.0 {
            return false;
        }
        if self.profile.drop_rate >= 1.0 {
            return true;
        }
        let mut rng = self.rng.lock();
        rng.gen::<f32>() < self.profile.drop_rate
    }

    async fn before(&self, peer: RankId, op: &'static str) -> anyhow::Result<()> {
        let delay = self.sample_delay(peer);
        if delay > Duration::ZERO {
            tokio::time::sleep(delay).await;
        }
        if self.should_drop() {
            crate::metrics::LATENCY_INJECTED_DROPS_TOTAL
                .with_label_values(&[op])
                .inc();
            return Err(anyhow::anyhow!(
                "LatencyInjector: simulated drop on {op} to peer {peer}"
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl<T: RankTransport> RankTransport for LatencyInjector<T> {
    async fn broadcast_seal(
        &self,
        src: RankId,
        block_id: BlockId,
        content_hash: u64,
        descriptor: Vec<f32>,
    ) -> anyhow::Result<()> {
        // For broadcast, we inject latency for the worst-case peer (the slowest tier).
        // This is conservative: a real implementation parallelises peer sends, so the
        // observed wall-clock is dominated by the slowest. Picking an arbitrary non-source
        // rank as the "worst case" is a reasonable approximation under SingleNode (all
        // equal) and MultiNode (mixed). Tests can construct deterministic worlds where
        // this matters.
        let any_peer = if src.raw() == 0 { RankId(1) } else { RankId(0) };
        self.before(any_peer, "broadcast_seal").await?;
        self.inner
            .broadcast_seal(src, block_id, content_hash, descriptor)
            .await
    }

    async fn fetch_block(&self, src: RankId, block_id: BlockId) -> anyhow::Result<BlockPayload> {
        self.before(src, "fetch_block").await?;
        self.inner.fetch_block(src, block_id).await
    }

    async fn push_block(&self, dst: RankId, payload: BlockPayload) -> anyhow::Result<BlockId> {
        self.before(dst, "push_block").await?;
        self.inner.push_block(dst, payload).await
    }

    async fn announce_release(&self, src: RankId, block_id: BlockId) -> anyhow::Result<()> {
        let any_peer = if src.raw() == 0 { RankId(1) } else { RankId(0) };
        self.before(any_peer, "announce_release").await?;
        self.inner.announce_release(src, block_id).await
    }

    async fn query_hash(&self, dst: RankId, content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        self.before(dst, "query_hash").await?;
        self.inner.query_hash(dst, content_hash).await
    }

    async fn reserve_slots(
        &self,
        dst: RankId,
        req_id: u64,
        count: u32,
    ) -> anyhow::Result<super::ReservationToken> {
        self.before(dst, "reserve_slots").await?;
        self.inner.reserve_slots(dst, req_id, count).await
    }

    async fn release_reservation(
        &self,
        dst: RankId,
        token: super::ReservationToken,
    ) -> anyhow::Result<()> {
        self.before(dst, "release_reservation").await?;
        self.inner.release_reservation(dst, token).await
    }

    fn topology(&self) -> &Topology {
        &self.topology
    }

    fn name(&self) -> &'static str {
        "latency_injector"
    }
}
