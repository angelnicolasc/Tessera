//! Tessera core: MLA-aware KV block manager.
//!
//! This crate provides the foundational types and algorithms that distinguish Tessera from
//! existing PagedAttention-style block managers. The crate is **kernel-agnostic**: it does not
//! invoke FlashMLA or any attention kernel directly. Instead it provides:
//!
//! 1. A typed [`config::MlaBlockConfig`] anchored on [`config::CompressionScheme`] — a sealed
//!    enum that lets future compression schemes (DSA c4a/c128a, MHA fallback) land as additive
//!    variants without breaking the public surface.
//! 2. A [`block_manager::TesseraBlockManager`] generic over [`device::DeviceBackend`], so the
//!    same code runs against a deterministic CPU mock in tests and against real CUDA HBM in
//!    production.
//! 3. A [`cross_agent::CrossAgentShareTable`] that ref-counts blocks shared between requests and
//!    provides the bookkeeping required for safe copy-on-write.
//! 4. Prometheus [`metrics`] for the seven observability signals the playbook calls out.
//!
//! See `ARCHITECTURE.md` and the ADRs under `docs/src/adr/` for the rationale behind each
//! design choice.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod block;
pub mod block_manager;
pub mod config;
pub mod content_hash;
pub mod cross_agent;
pub mod device;
pub mod error;
pub mod metrics;
pub mod observability;
pub mod rank;
pub mod state_cache;
pub mod transport;

pub use block::{BlockId, BlockMeta, GlobalBlockId, TokenRange, SHARED_SENTINEL};
pub use block_manager::{SealOutcome, TesseraBlockManager};
pub use config::{CkvDtype, CompressionScheme, MlaBlockConfig};
pub use cross_agent::CrossAgentShareTable;
pub use device::{CpuMockBackend, DeviceBackend, DevicePtr, DiskBackend, SwaCachingStrategy};
pub use error::{Result, TesseraError};
pub use rank::{LatencyTier, NodeId, RankId, Topology, World};
pub use state_cache::{RequestArena, StateCache, StateCacheConfig};
pub use transport::{
    BlockPayload, LatencyInjector, LatencyProfile, MockTransport, RankTransport, ReservationToken,
};
