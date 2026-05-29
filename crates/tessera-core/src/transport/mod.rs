//! Cross-rank transport abstraction.
//!
//! Three implementations:
//!
//! * [`MockTransport`] — in-process channels backed by Tokio mpsc. Used by every Sprint 3
//!   test; deterministic, no network or GPU required.
//! * `P2pCudaTransport` (feature `cuda`) — NVLink P2P via `cudarc`. Sprint 3 ships the
//!   struct + method signatures wired into the dispatcher; the runtime implementation
//!   bodies return a documented `Err(...)` and are completed in the combined cloud-burst
//!   session on real hardware.
//! * `NcclTransport` (feature `nccl`) — multi-node fan-out via NCCL. Sprint 3 ships the
//!   stub so that downstream callers can target the stable API today; the runtime impl
//!   lands in Sprint 4 once the multi-node test harness exists.
//!
//! Selection logic lives in the consumer (e.g. the Python plugin) and is informed by
//! [`crate::rank::World::topology`]. See `docs/src/adr/0015-p2p-vs-nccl-transport.md`.

pub mod latency;
pub mod mock;

#[cfg(feature = "cuda")]
pub mod p2p_cuda;

#[cfg(feature = "nccl")]
pub mod nccl;

pub use latency::{LatencyInjector, LatencyProfile};
pub use mock::MockTransport;

#[cfg(feature = "cuda")]
pub use p2p_cuda::P2pCudaTransport;

#[cfg(feature = "nccl")]
pub use nccl::NcclTransport;

use async_trait::async_trait;

use crate::block::BlockId;
use crate::rank::{RankId, Topology};

/// Opaque handle returned by [`RankTransport::reserve_slots`] and surrendered to
/// [`RankTransport::release_reservation`]. The numeric value is implementation-defined
/// (mock uses a monotonic counter; production transports may encode rank+req_id+epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationToken(pub u64);

impl ReservationToken {
    /// Raw integer for FFI / debugging.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ReservationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tok:{}", self.0)
    }
}

/// The serialised payload of a single block transferred across a rank boundary. The payload
/// owns the bytes so that the transport can move them across threads / processes. For
/// production NVLink P2P the implementation can pass shared-memory handles instead of
/// copying — see ADR-0015 and TD-026.
#[derive(Debug, Clone)]
pub struct BlockPayload {
    /// Primary region (`c_kv`) bytes.
    pub c_kv: Vec<u8>,
    /// `k_rope` region bytes (always BF16).
    pub k_rope: Vec<u8>,
    /// Optional FP8 per-layer scales (only present when the source block is FP8-stored).
    pub fp8_scales: Option<Vec<u8>>,
}

impl BlockPayload {
    /// Approximate heap footprint of this payload — useful for metric histograms.
    pub fn byte_len(&self) -> usize {
        self.c_kv.len() + self.k_rope.len() + self.fp8_scales.as_ref().map_or(0, Vec::len)
    }
}

/// Cross-rank transport contract. Every implementation is `Send + Sync` so it can be cloned
/// (typically `Arc<dyn RankTransport>`) and held by both the block manager and the
/// distributed segment index without internal locking.
#[async_trait]
pub trait RankTransport: Send + Sync {
    /// Announce that `src` has sealed `block_id` with the given content hash and ANN
    /// descriptor. Peers may decide to record this in their distributed segment index for
    /// future lookups. Idempotent: receivers must tolerate duplicates.
    async fn broadcast_seal(
        &self,
        src: RankId,
        block_id: BlockId,
        content_hash: u64,
        descriptor: Vec<f32>,
    ) -> anyhow::Result<()>;

    /// Fetch the full payload of `block_id` from rank `src`. Used for cross-rank
    /// content-addressed sharing and for the PD-disaggregation reverse-pull path.
    async fn fetch_block(&self, src: RankId, block_id: BlockId) -> anyhow::Result<BlockPayload>;

    /// Push a payload to rank `dst`, returning the local block id assigned by the
    /// destination's block manager. Used by the PD-disaggregation push-mode path. The
    /// destination is responsible for allocating the block and writing the bytes.
    async fn push_block(&self, dst: RankId, payload: BlockPayload) -> anyhow::Result<BlockId>;

    /// Announce release of `block_id` so peers can drop any cached references.
    async fn announce_release(&self, src: RankId, block_id: BlockId) -> anyhow::Result<()>;

    /// Look up whether peer `dst` is holding a block whose content hash matches the given
    /// value. Used by [`crate::block_manager::TesseraBlockManager`]'s distributed lookup
    /// fast-path before a full descriptor fan-out. Returns `Ok(None)` for "no match".
    async fn query_hash(&self, dst: RankId, content_hash: u64) -> anyhow::Result<Option<BlockId>>;

    /// Reserve `count` block slots on peer `dst` for `req_id`. Used by the reserve-then-stream
    /// PD-disaggregation protocol (ADR-0018): the source rank calls this before its first
    /// `push_block` so the destination can refuse early if it lacks capacity. On success
    /// the destination has pinned the slots; the source must either consume them with
    /// matching `push_block` calls or surrender them via [`release_reservation`]. Idempotent
    /// for a given `(dst, req_id)` only at the implementation's discretion — callers
    /// should not rely on it.
    async fn reserve_slots(
        &self,
        dst: RankId,
        req_id: u64,
        count: u32,
    ) -> anyhow::Result<ReservationToken>;

    /// Release a reservation previously obtained via [`reserve_slots`]. Used on the
    /// rollback path when a `transfer_request_to_rank` aborts mid-stream. Safe to call
    /// even when partial pushes have already consumed some of the reserved slots — the
    /// implementation releases only the unused remainder.
    async fn release_reservation(&self, dst: RankId, token: ReservationToken)
        -> anyhow::Result<()>;

    /// Describes the world this transport spans. Used by selection logic and metrics.
    fn topology(&self) -> &Topology;

    /// Human-readable name for diagnostics, metrics labels, and logs.
    fn name(&self) -> &'static str;
}
