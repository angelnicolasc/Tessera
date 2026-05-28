//! In-process mock transport backed by Tokio mpsc channels.
//!
//! Each rank in the simulated world owns a [`MockTransport`] handle. The handle holds a
//! shared registry of payload-providers (one per rank) plus a shared event log so tests can
//! assert exact message fidelity (e.g. "rank 0's seal broadcast reached every other rank
//! exactly once"). Determinism: every async operation completes within a single
//! `tokio::yield_now`; no real wall-clock delay is introduced unless tests explicitly add it.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::block::BlockId;
use crate::rank::{RankId, Topology};

use super::{BlockPayload, RankTransport, ReservationToken};

/// Single record of a cross-rank event, captured by the shared `EventLog` for assertions.
#[derive(Debug, Clone)]
pub enum MockEvent {
    /// `broadcast_seal(src, block_id, content_hash)` fan-out to all peers.
    BroadcastSeal {
        /// Originating rank.
        src: RankId,
        /// Originating block.
        block_id: BlockId,
        /// Content hash announced.
        content_hash: u64,
    },
    /// `fetch_block(src, block_id)` issued by some rank.
    Fetch {
        /// Source rank (the one being fetched from).
        src: RankId,
        /// Block being fetched.
        block_id: BlockId,
    },
    /// `push_block(dst, ...)` issued by some rank.
    Push {
        /// Destination rank.
        dst: RankId,
        /// Payload byte length (avoid logging the entire bytes).
        bytes: usize,
    },
    /// `announce_release(src, block_id)`.
    Release {
        /// Originating rank.
        src: RankId,
        /// Block being released.
        block_id: BlockId,
    },
    /// `query_hash(dst, hash)`.
    QueryHash {
        /// Destination rank queried.
        dst: RankId,
        /// Content hash queried.
        content_hash: u64,
    },
    /// `reserve_slots(dst, req_id, count)`.
    Reserve {
        /// Destination rank being reserved on.
        dst: RankId,
        /// Request id holding the reservation.
        req_id: u64,
        /// Count of slots requested.
        count: u32,
    },
    /// `release_reservation(dst, token)`.
    ReleaseReservation {
        /// Destination rank.
        dst: RankId,
        /// Token surrendered.
        token: ReservationToken,
    },
}

/// Shared log of every `MockTransport` operation across a simulated world. Wrapped in
/// `Arc<Mutex<Vec<...>>>` so tests can snapshot and assert.
pub type EventLog = Arc<Mutex<Vec<MockEvent>>>;

/// Per-rank handler registered with the shared registry. Lets tests inject payload and
/// query-hash logic for the rank "owned" by each [`MockTransport`] handle. In typical use
/// the test wires the block manager of that rank as the provider.
pub trait MockPeer: Send + Sync {
    /// Produce the payload for `block_id` from this rank's local storage.
    fn provide_block(&self, block_id: BlockId) -> anyhow::Result<BlockPayload>;
    /// Accept a pushed payload and return the local block id assigned.
    fn accept_pushed(&self, payload: BlockPayload) -> anyhow::Result<BlockId>;
    /// Look up a content hash; return the local block id if present.
    fn lookup_hash(&self, content_hash: u64) -> anyhow::Result<Option<BlockId>>;
    /// Notification: a peer announced a seal. Default: ignore.
    fn on_seal_announce(
        &self,
        _src: RankId,
        _block_id: BlockId,
        _content_hash: u64,
        _descriptor: &[f32],
    ) -> anyhow::Result<()> {
        Ok(())
    }
    /// Notification: a peer announced a release. Default: ignore.
    fn on_release_announce(&self, _src: RankId, _block_id: BlockId) -> anyhow::Result<()> {
        Ok(())
    }

    /// Reserve `count` slots for `req_id`. Implementations that don't model capacity can
    /// return a synthetic token via the default (always-succeed) path.
    fn reserve(&self, _req_id: u64, _count: u32) -> anyhow::Result<ReservationToken> {
        Ok(ReservationToken(0))
    }

    /// Release a previously held reservation. Default: ignore.
    fn release_reservation(&self, _token: ReservationToken) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Default no-op peer used when a rank's handler hasn't been wired yet. Returns empty
/// payloads / `None` from queries. Useful for `MockTransport::new_world(N)` smoke tests.
pub struct NullPeer;
impl MockPeer for NullPeer {
    fn provide_block(&self, _block_id: BlockId) -> anyhow::Result<BlockPayload> {
        Ok(BlockPayload { c_kv: vec![], k_rope: vec![], fp8_scales: None })
    }
    fn accept_pushed(&self, _payload: BlockPayload) -> anyhow::Result<BlockId> {
        Ok(BlockId(0))
    }
    fn lookup_hash(&self, _content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
}

/// Shared registry mapping each rank to its handler. Wrapped in `Arc<Mutex<...>>` so tests
/// can swap in handlers after construction.
type PeerRegistry = Arc<Mutex<HashMap<RankId, Arc<dyn MockPeer>>>>;

/// One handle per rank. Cloning a `MockTransport` clones the inner `Arc`s; the resulting
/// handles share the registry, event log, and reservation counter.
#[derive(Clone)]
pub struct MockTransport {
    local: RankId,
    world_size: u32,
    registry: PeerRegistry,
    events: EventLog,
    /// Shared monotonic counter for ReservationToken generation. Wrapped in `Arc<AtomicU64>`
    /// so all cloned handles agree on a strictly-increasing token space.
    reservation_counter: std::sync::Arc<AtomicU64>,
}

impl std::fmt::Debug for MockTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTransport")
            .field("local", &self.local)
            .field("world_size", &self.world_size)
            .field("events", &self.events.lock().len())
            .finish()
    }
}

impl MockTransport {
    /// Construct N interconnected `MockTransport` handles, one per rank. Each handle starts
    /// with a `NullPeer` and tests can `register_peer` to swap in their wiring.
    pub fn new_world(size: u32) -> Vec<Self> {
        let registry: PeerRegistry = Arc::new(Mutex::new({
            let mut m = HashMap::new();
            for r in 0..size {
                m.insert(RankId(r), Arc::new(NullPeer) as Arc<dyn MockPeer>);
            }
            m
        }));
        let events: EventLog = Arc::new(Mutex::new(Vec::new()));
        let reservation_counter = std::sync::Arc::new(AtomicU64::new(1));
        (0..size)
            .map(|r| Self {
                local: RankId(r),
                world_size: size,
                registry: Arc::clone(&registry),
                events: Arc::clone(&events),
                reservation_counter: std::sync::Arc::clone(&reservation_counter),
            })
            .collect()
    }

    /// Convenience for tests that want a single-rank world (zero peers). Mostly useful as a
    /// default when API callers require *some* transport handle.
    pub fn singleton() -> Self {
        Self::new_world(1).into_iter().next().unwrap()
    }

    /// Construct N handles wrapped in [`super::LatencyInjector`] per the supplied topology.
    /// Returns the trait-object form so callers can hand them to consumers that take
    /// `Arc<dyn RankTransport>`.
    ///
    /// Each handle's tier resolution uses the supplied [`Topology`] (which must have
    /// `node_of` length == `size` when multi-node). Drops + jitter are deterministic per
    /// handle: handle `r` is seeded with `base_seed XOR r`.
    pub fn with_topology(
        size: u32,
        topology: Topology,
        profile: super::LatencyProfile,
        base_seed: u64,
    ) -> Vec<std::sync::Arc<dyn RankTransport>> {
        let bases = Self::new_world(size);
        bases
            .into_iter()
            .enumerate()
            .map(|(idx, m)| {
                let inner = std::sync::Arc::new(m);
                let injector = super::LatencyInjector::new(
                    inner,
                    profile,
                    RankId(idx as u32),
                    topology.clone(),
                    base_seed ^ idx as u64,
                );
                let dyn_t: std::sync::Arc<dyn RankTransport> = std::sync::Arc::new(injector);
                dyn_t
            })
            .collect()
    }

    /// Replace the handler for rank `r`. Returns the previous handler so tests can chain.
    pub fn register_peer(&self, r: RankId, peer: Arc<dyn MockPeer>) -> Option<Arc<dyn MockPeer>> {
        self.registry.lock().insert(r, peer)
    }

    /// Snapshot the event log. Each call returns a clone — the log keeps accumulating.
    pub fn events(&self) -> Vec<MockEvent> {
        self.events.lock().clone()
    }

    /// Number of events recorded so far.
    pub fn event_count(&self) -> usize {
        self.events.lock().len()
    }

    /// Clear the event log. Useful between test phases.
    pub fn clear_events(&self) {
        self.events.lock().clear();
    }

    /// This rank's id.
    pub const fn local(&self) -> RankId {
        self.local
    }

    /// Lookup peer handle by rank.
    fn peer(&self, r: RankId) -> Option<Arc<dyn MockPeer>> {
        self.registry.lock().get(&r).cloned()
    }

    fn record(&self, ev: MockEvent) {
        self.events.lock().push(ev);
    }
}

#[async_trait]
impl RankTransport for MockTransport {
    async fn broadcast_seal(
        &self,
        src: RankId,
        block_id: BlockId,
        content_hash: u64,
        descriptor: Vec<f32>,
    ) -> anyhow::Result<()> {
        self.record(MockEvent::BroadcastSeal { src, block_id, content_hash });
        // Fan out to every peer (excluding source).
        let mut peers: Vec<(RankId, Arc<dyn MockPeer>)> = Vec::new();
        {
            let reg = self.registry.lock();
            for (r, p) in reg.iter() {
                if *r != src {
                    peers.push((*r, Arc::clone(p)));
                }
            }
        }
        for (r, peer) in peers {
            // Notify; deliver descriptor by slice.
            let _ = peer.on_seal_announce(src, block_id, content_hash, &descriptor);
            crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
                .with_label_values(&[&src.to_string(), &r.to_string(), "broadcast_seal"])
                .inc();
        }
        // Cooperative yield so concurrent tasks can interleave deterministically.
        tokio::task::yield_now().await;
        Ok(())
    }

    async fn fetch_block(
        &self,
        src: RankId,
        block_id: BlockId,
    ) -> anyhow::Result<BlockPayload> {
        self.record(MockEvent::Fetch { src, block_id });
        let peer = self
            .peer(src)
            .ok_or_else(|| anyhow::anyhow!("MockTransport: no peer registered for rank {src}"))?;
        let payload = peer.provide_block(block_id)?;
        crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&[&src.to_string(), &self.local.to_string(), "fetch"])
            .inc();
        tokio::task::yield_now().await;
        Ok(payload)
    }

    async fn push_block(
        &self,
        dst: RankId,
        payload: BlockPayload,
    ) -> anyhow::Result<BlockId> {
        let bytes = payload.byte_len();
        self.record(MockEvent::Push { dst, bytes });
        let peer = self
            .peer(dst)
            .ok_or_else(|| anyhow::anyhow!("MockTransport: no peer registered for rank {dst}"))?;
        let assigned = peer.accept_pushed(payload)?;
        crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&[&self.local.to_string(), &dst.to_string(), "push"])
            .inc();
        tokio::task::yield_now().await;
        Ok(assigned)
    }

    async fn announce_release(
        &self,
        src: RankId,
        block_id: BlockId,
    ) -> anyhow::Result<()> {
        self.record(MockEvent::Release { src, block_id });
        let peers: Vec<(RankId, Arc<dyn MockPeer>)> = {
            let reg = self.registry.lock();
            reg.iter()
                .filter(|(r, _)| **r != src)
                .map(|(r, p)| (*r, Arc::clone(p)))
                .collect()
        };
        for (r, peer) in peers {
            let _ = peer.on_release_announce(src, block_id);
            crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
                .with_label_values(&[&src.to_string(), &r.to_string(), "release_announce"])
                .inc();
        }
        tokio::task::yield_now().await;
        Ok(())
    }

    async fn query_hash(
        &self,
        dst: RankId,
        content_hash: u64,
    ) -> anyhow::Result<Option<BlockId>> {
        self.record(MockEvent::QueryHash { dst, content_hash });
        let peer = self
            .peer(dst)
            .ok_or_else(|| anyhow::anyhow!("MockTransport: no peer registered for rank {dst}"))?;
        let result = peer.lookup_hash(content_hash)?;
        tokio::task::yield_now().await;
        Ok(result)
    }

    async fn reserve_slots(
        &self,
        dst: RankId,
        req_id: u64,
        count: u32,
    ) -> anyhow::Result<ReservationToken> {
        self.record(MockEvent::Reserve { dst, req_id, count });
        let peer = self
            .peer(dst)
            .ok_or_else(|| anyhow::anyhow!("MockTransport: no peer registered for rank {dst}"))?;
        // Ask the peer to actually reserve; if the peer's default no-op returns 0, mint a
        // fresh token from our shared counter instead so callers always get unique ids.
        let peer_token = peer.reserve(req_id, count)?;
        let token = if peer_token.raw() == 0 {
            ReservationToken(self.reservation_counter.fetch_add(1, Ordering::Relaxed))
        } else {
            peer_token
        };
        crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&[&self.local.to_string(), &dst.to_string(), "reserve"])
            .inc();
        tokio::task::yield_now().await;
        Ok(token)
    }

    async fn release_reservation(
        &self,
        dst: RankId,
        token: ReservationToken,
    ) -> anyhow::Result<()> {
        self.record(MockEvent::ReleaseReservation { dst, token });
        let peer = self
            .peer(dst)
            .ok_or_else(|| anyhow::anyhow!("MockTransport: no peer registered for rank {dst}"))?;
        peer.release_reservation(token)?;
        crate::metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&[&self.local.to_string(), &dst.to_string(), "release_reservation"])
            .inc();
        tokio::task::yield_now().await;
        Ok(())
    }

    fn topology(&self) -> &Topology {
        &Topology::SingleNode
    }

    fn name(&self) -> &'static str {
        "mock"
    }
}

/// Helper for tests that need to wait on a one-shot signal from a peer. Re-exported here so
/// tests can `use tessera_core::transport::mock::oneshot_pair;` without depending on Tokio
/// directly.
pub fn oneshot_pair<T>() -> (oneshot::Sender<T>, oneshot::Receiver<T>) {
    oneshot::channel()
}
