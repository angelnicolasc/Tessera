//! Tessera segment index — pluggable ANN backend for content-addressed `c_kv` sharing.
//!
//! Two-layer design:
//!
//! * Layer 1: exact xxhash3 match — lives in `tessera-core` next to the block manager because
//!   it is on the hot path and cannot afford a virtual dispatch.
//! * Layer 2: this crate. Approximate nearest-neighbour over the mean `c_kv` vector. Pluggable
//!   via [`IndexBackend`] so that Engram and Lightning-Indexer replacements can land later
//!   without touching the Python orchestration layer (ADR-0005).
//!
//! The descriptor pipeline (mean over layers and tokens) is intentionally **not** abstracted
//! by this crate — it is specific to MLA storage and will change shape with DSA. Each
//! `IndexBackend` impl assumes the caller hands it a 1-D feature vector of the right
//! dimensionality.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod distributed;
mod usearch_index;

pub use distributed::{DistributedHit, DistributedSegmentIndex, TierBudget};
pub use usearch_index::{UsearchConfig, UsearchIndex};

/// A single ANN match: the matched block id and the similarity score (cosine in `[-1, 1]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexMatch {
    /// Matching block id.
    pub block_id: u32,
    /// Cosine similarity in `[-1, 1]`. Higher is more similar.
    pub similarity: f32,
}

/// Pluggable ANN backend over `c_kv` mean descriptors.
///
/// Implementations must be `Send + Sync` because the Python segment-index layer calls into
/// them from arbitrary `asyncio` worker threads.
pub trait IndexBackend: Send + Sync {
    /// Add or replace a descriptor for `block_id`.
    fn add(&self, block_id: u32, descriptor: &[f32]) -> anyhow::Result<()>;

    /// Search for the top-`k` nearest neighbours of `descriptor`.
    fn query(&self, descriptor: &[f32], k: usize) -> anyhow::Result<Vec<IndexMatch>>;

    /// Remove the entry for `block_id`. Implementations should accept "id was never present"
    /// silently.
    fn remove(&self, block_id: u32) -> anyhow::Result<()>;

    /// Number of entries currently indexed.
    fn len(&self) -> usize;

    /// Whether the index has zero entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Implementation name for diagnostics.
    fn name(&self) -> &'static str;
}
