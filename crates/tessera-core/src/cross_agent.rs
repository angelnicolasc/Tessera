//! Cross-agent share table.
//!
//! Tracks which blocks are currently shared between multiple requests. The block manager
//! itself maintains per-block ref-counts; this table layers two things on top:
//!
//! 1. **Reverse mapping** — given a request id, find the list of shared blocks it holds. The
//!    block manager has no symmetric reverse mapping by design (it would have to lock the
//!    whole metadata map on every request scan); the share table keeps this index out of band
//!    so request completion can decrement everything in one walk.
//! 2. **Sharing metrics** — `tessera_sharing_rate` reflects the proportion of allocated
//!    block-references that are shared (vs. private), which is the headline number the
//!    cross-agent benchmark wants.
//!
//! Copy-on-write enforcement is mechanically separate: it is performed by the block manager's
//! `cow_fork` when a write to a shared block is detected. The share table only knows about
//! ownership; it does not police mutation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use crate::block::BlockId;

/// Cross-agent share-state index.
#[derive(Debug, Default)]
pub struct CrossAgentShareTable {
    block_to_owners: DashMap<BlockId, Vec<u64>>,
    req_to_blocks: DashMap<u64, Vec<BlockId>>,
    /// Cumulative count of `add_share` calls. Used as the denominator of `sharing_rate`.
    total_shared_refs: Arc<AtomicU64>,
}

impl CrossAgentShareTable {
    /// Construct a fresh, empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `req_id` as an additional owner of `block_id`. Caller must have already
    /// incremented the block's ref-count in the block manager.
    pub fn add_share(&self, req_id: u64, block_id: BlockId) {
        self.block_to_owners.entry(block_id).or_default().push(req_id);
        self.req_to_blocks.entry(req_id).or_default().push(block_id);
        self.total_shared_refs.fetch_add(1, Ordering::Relaxed);
        crate::metrics::SHARING_RATE.set(self.sharing_rate());
    }

    /// Release every shared block held by `req_id`. Returns the list of `block_id`s that the
    /// caller must now `free` on the block manager (because this request was their last
    /// shared owner — the block manager's per-block ref-count handles the actual free).
    ///
    /// Note: the *block manager's* ref-count must still be decremented for **every** entry in
    /// the returned vector; the share table doesn't reach into the block manager itself.
    pub fn release_request(&self, req_id: u64) -> Vec<BlockId> {
        let blocks = self.req_to_blocks.remove(&req_id).map(|(_, v)| v).unwrap_or_default();
        let mut to_free = Vec::with_capacity(blocks.len());
        for block_id in blocks {
            let now_empty = {
                let mut entry = self.block_to_owners.entry(block_id).or_default();
                entry.retain(|r| *r != req_id);
                entry.is_empty()
            };
            // Always return the block id — the caller must release their ref-count in the
            // block manager. If we were the last owner the manager will free it; otherwise it
            // will just decrement.
            to_free.push(block_id);
            if now_empty {
                self.block_to_owners.remove(&block_id);
            }
        }
        crate::metrics::SHARING_RATE.set(self.sharing_rate());
        to_free
    }

    /// Look up all current owners of `block_id`. Returns `None` if the block is not shared.
    pub fn owners(&self, block_id: BlockId) -> Option<Vec<u64>> {
        self.block_to_owners.get(&block_id).map(|e| e.value().clone())
    }

    /// Distinct shared blocks tracked.
    pub fn shared_block_count(&self) -> usize {
        self.block_to_owners.len()
    }

    /// Fraction of `add_share`-tracked references that correspond to currently shared blocks.
    /// Returns 0.0 when no shares have been registered yet.
    pub fn sharing_rate(&self) -> f64 {
        let total = self.total_shared_refs.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let shared = self.block_to_owners.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let total_f = total as f64;
        shared / total_f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_release_roundtrip() {
        let st = CrossAgentShareTable::new();
        st.add_share(1, BlockId(10));
        st.add_share(2, BlockId(10));
        st.add_share(2, BlockId(11));

        assert_eq!(st.shared_block_count(), 2);
        let owners = st.owners(BlockId(10)).unwrap();
        assert_eq!(owners.len(), 2);

        let freed_for_req2 = st.release_request(2);
        assert_eq!(freed_for_req2.len(), 2);
        // Block 10 should still have req 1.
        assert_eq!(st.owners(BlockId(10)).unwrap(), vec![1]);
        // Block 11 had only req 2, so it should be gone.
        assert!(st.owners(BlockId(11)).is_none());
    }
}
