//! `P2pCudaTransport` — NVLink P2P intra-node transport. **Stub, runtime impl deferred.**
//!
//! The struct, methods, and dispatcher wiring are complete so callers can target the stable
//! API today. The bodies return a structured error pointing to the tracking ticket; the real
//! implementation will land in the combined cloud-burst session on multi-GPU hardware.
//!
//! # Implementation outline (cloud-burst)
//!
//! Each rank holds:
//!
//! * One `cudarc::driver::CudaDevice` per peer rank in the world.
//! * A registry mapping `(peer_rank, BlockId) -> IpcMemHandle` returned by
//!   `cuMemExportToShareableHandle`. The handle is imported on demand via
//!   `cuMemImportFromShareableHandle` to obtain a peer-accessible device pointer.
//! * `cuCtxEnablePeerAccess` invoked once per (local, peer) pair at startup.
//!
//! `fetch_block` performs an asynchronous `cuMemcpyPeerAsync` from the peer device into a
//! transient buffer; the byte payload is then passed into `BlockPayload` (or, in the optimal
//! path, the raw shareable handle is returned and the destination's block manager imports
//! it in place — zero-copy. See TD-026).
//!
//! # Tracking
//!
//! - Issue label `tier-b-cloud-burst`
//! - TD-021 in `DEVLOG.md`
//! - ADR-0015 captures the rationale for the dispatcher pattern.

#![allow(missing_docs)]

use async_trait::async_trait;

use crate::block::BlockId;
use crate::rank::{RankId, Topology};

use super::{BlockPayload, RankTransport};

/// NVLink P2P transport. Construct via [`P2pCudaTransport::new`]; that constructor records
/// the world layout but does **not** initialise CUDA contexts (that happens on first use in
/// the cloud-burst impl).
#[derive(Debug, Clone)]
pub struct P2pCudaTransport {
    local: RankId,
    world_size: u32,
    topology: Topology,
}

impl P2pCudaTransport {
    /// Construct a P2P transport for the given world. The topology is captured at
    /// construction; runtime CUDA initialisation is lazy.
    pub fn new(local: RankId, world_size: u32, topology: Topology) -> Self {
        Self {
            local,
            world_size,
            topology,
        }
    }

    fn deferred<T>(op: &'static str) -> anyhow::Result<T> {
        Err(anyhow::anyhow!(
            "P2pCudaTransport::{op}: runtime impl deferred to cloud-burst session (TD-021). \
             The dispatcher is wired; this stub guarantees the API surface stays stable."
        ))
    }
}

#[async_trait]
impl RankTransport for P2pCudaTransport {
    async fn broadcast_seal(
        &self,
        _src: RankId,
        _block_id: BlockId,
        _content_hash: u64,
        _descriptor: Vec<f32>,
    ) -> anyhow::Result<()> {
        Self::deferred("broadcast_seal")
    }

    async fn fetch_block(&self, _src: RankId, _block_id: BlockId) -> anyhow::Result<BlockPayload> {
        Self::deferred("fetch_block")
    }

    async fn push_block(&self, _dst: RankId, _payload: BlockPayload) -> anyhow::Result<BlockId> {
        Self::deferred("push_block")
    }

    async fn announce_release(&self, _src: RankId, _block_id: BlockId) -> anyhow::Result<()> {
        Self::deferred("announce_release")
    }

    async fn query_hash(
        &self,
        _dst: RankId,
        _content_hash: u64,
    ) -> anyhow::Result<Option<BlockId>> {
        Self::deferred("query_hash")
    }

    async fn reserve_slots(
        &self,
        _dst: RankId,
        _req_id: u64,
        _count: u32,
    ) -> anyhow::Result<super::ReservationToken> {
        Self::deferred("reserve_slots")
    }

    async fn release_reservation(
        &self,
        _dst: RankId,
        _token: super::ReservationToken,
    ) -> anyhow::Result<()> {
        Self::deferred("release_reservation")
    }

    fn topology(&self) -> &Topology {
        &self.topology
    }

    fn name(&self) -> &'static str {
        "p2p_cuda"
    }
}

// Suppress dead-code warnings for the `local` / `world_size` fields on the stub — they will
// be read by the runtime implementation.
#[allow(dead_code)]
const fn _used_in_runtime_impl(t: &P2pCudaTransport) -> (RankId, u32) {
    (t.local, t.world_size)
}
