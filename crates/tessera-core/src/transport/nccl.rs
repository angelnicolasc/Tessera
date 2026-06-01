//! `NcclTransport` — multi-node fan-out via NCCL. **Stub, Sprint 4 runtime impl.**
//!
//! Multi-node IB has fundamentally different latency / bandwidth characteristics than
//! intra-node NVLink P2P, which is why the transport is selected via topology rather than a
//! single unified backend (see ADR-0015). The Sprint 3 deliverable here is the **type
//! plumbing**: the struct + methods exist, the dispatcher routes here when topology is
//! `MultiNode`, and the body returns a structured error pointing to TD-022.
//!
//! # Implementation outline (Sprint 4)
//!
//! * `NcclCommunicator::new(world_size, rank, unique_id)` once at startup. Unique-id
//!   exchange uses a CPU-side rendezvous (file, TCP, env-var) so the multi-node test
//!   harness can run in CI without GPU.
//! * `broadcast_seal` → `ncclAllGather` of fixed-size announcement records (rank, block_id,
//!   content_hash, descriptor). Use a dedicated NCCL stream so seals don't block decode.
//! * `fetch_block` → point-to-point `ncclSend` / `ncclRecv` on the producer/consumer pair.
//!   Note: NCCL P2P requires both ranks to issue paired calls — the consumer initiates with
//!   a `request_block` message, producer replies with the payload.
//! * Network-aware budget for the distributed segment index (`DistributedSegmentIndex`
//!   already supports configurable timeouts; multi-node configs should pick a larger
//!   `total_budget`). See TD-023.

#![allow(missing_docs)]

use async_trait::async_trait;

use crate::block::BlockId;
use crate::rank::{RankId, Topology};

use super::{BlockPayload, RankTransport};

#[derive(Debug, Clone)]
pub struct NcclTransport {
    local: RankId,
    world_size: u32,
    topology: Topology,
}

impl NcclTransport {
    pub fn new(local: RankId, world_size: u32, topology: Topology) -> Self {
        Self {
            local,
            world_size,
            topology,
        }
    }

    fn deferred<T>(op: &'static str) -> anyhow::Result<T> {
        Err(anyhow::anyhow!(
            "NcclTransport::{op}: runtime impl deferred to Sprint 4 (TD-022). \
             The dispatcher is wired; multi-node configs can author against the stable API."
        ))
    }
}

#[async_trait]
impl RankTransport for NcclTransport {
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
        "nccl"
    }
}

#[allow(dead_code)]
const fn _used_in_runtime_impl(t: &NcclTransport) -> (RankId, u32) {
    (t.local, t.world_size)
}
