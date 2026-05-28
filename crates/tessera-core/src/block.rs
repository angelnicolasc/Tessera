//! Block identity and metadata.
//!
//! These types are deliberately small and `Copy` where possible: the block manager keeps one
//! [`BlockMeta`] per allocated block and the per-block memory budget should be measured in
//! cache lines, not pages.

use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::Arc;

use crate::config::CkvDtype;
use crate::rank::RankId;

/// Sentinel `req_id` denoting a multi-owner block. Resolve owners via
/// [`crate::cross_agent::CrossAgentShareTable`].
pub const SHARED_SENTINEL: u64 = u64::MAX;

/// Globally addressable block identifier — `(rank, block)`. Sprint 3 introduces this so
/// cross-rank components (transport, distributed segment index, cross-agent share table)
/// can refer unambiguously to a specific block on a specific rank. Within a single rank's
/// block manager the local [`BlockId`] is still the canonical handle; `GlobalBlockId` is
/// only constructed at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalBlockId {
    /// Rank that owns the block.
    pub rank: RankId,
    /// Local block id on that rank.
    pub block: BlockId,
}

impl GlobalBlockId {
    /// Construct a global id from its parts.
    pub const fn new(rank: RankId, block: BlockId) -> Self {
        Self { rank, block }
    }
}

impl std::fmt::Display for GlobalBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:b{}", self.rank, self.block.raw())
    }
}

/// Strongly-typed block identifier. Wrapping a `u32` in a newtype keeps the block index name-
/// space distinct from generic counters and lets the type system catch the most common class
/// of indexing bug: passing a `block_id` where a `req_id` is expected (and vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Underlying integer for FFI / Python interop.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for BlockId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<BlockId> for u32 {
    fn from(value: BlockId) -> Self {
        value.0
    }
}

/// Half-open `[start, end)` range of token positions covered by a single block. Stored as a
/// `Range<u32>`-shaped struct to keep `BlockMeta` `Clone` without paying for a `Range` clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenRange {
    /// Inclusive start position.
    pub start: u32,
    /// Exclusive end position.
    pub end: u32,
}

impl TokenRange {
    /// Convenience constructor; debug-asserts that `end >= start`.
    pub const fn new(start: u32, end: u32) -> Self {
        debug_assert!(end >= start, "TokenRange end must be >= start");
        Self { start, end }
    }

    /// Number of tokens covered.
    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    /// Whether the range is empty.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

impl From<Range<u32>> for TokenRange {
    fn from(r: Range<u32>) -> Self {
        Self::new(r.start, r.end)
    }
}

/// Per-block bookkeeping kept by the block manager. Cloning is cheap because `ref_count` and
/// `last_touched` are `Arc<Atomic*>` — the underlying atomics are shared, not copied.
#[derive(Debug, Clone)]
pub struct BlockMeta {
    /// Stable identifier for this block.
    pub block_id: BlockId,

    /// Owning request, or `Some(SHARED_SENTINEL)` if the block is shared between multiple
    /// requests (look up the owner set in the share table). `None` indicates the block is
    /// currently in the free pool.
    pub req_id: Option<u64>,

    /// Reference count. Shared so multiple `BlockMeta` snapshots over time agree on the same
    /// counter. The block is eligible to be returned to the free pool when this hits zero.
    pub ref_count: Arc<AtomicU32>,

    /// xxhash3 over the raw `c_kv` bytes. Position-independent. Filled by
    /// [`crate::block_manager::TesseraBlockManager::seal`] after prefill writes the latents.
    pub content_hash: u64,

    /// Whether this block's mean-`c_kv` vector has been added to the segment index. The Rust
    /// core only flips this flag; the actual index lives in `tessera-index` and the async
    /// Python orchestration layer.
    pub indexed: bool,

    /// Token range covered by this block.
    pub token_range: TokenRange,

    /// Storage dtype of the `c_kv` region for this block (may differ from manager default if
    /// later we support mixed dtypes — Sprint 0 always matches the manager's config).
    pub ckv_dtype: CkvDtype,

    /// Monotonic epoch counter incremented each time the block is accessed via `primary_ptr`
    /// or `rope_ptr`. Used by the LRU eviction policy to identify least-recently-touched
    /// blocks within an eviction tier. Epoch is a global counter from the block manager, not
    /// wall-clock time — this avoids clock skew and is lock-free.
    pub last_touched: Arc<AtomicU64>,
}

impl BlockMeta {
    /// Construct fresh metadata for a newly allocated block. `ref_count` starts at 1 and
    /// `last_touched` is initialised to `initial_epoch`.
    pub fn fresh(
        block_id: BlockId,
        req_id: u64,
        token_range: TokenRange,
        ckv_dtype: CkvDtype,
        initial_epoch: u64,
    ) -> Self {
        Self {
            block_id,
            req_id: Some(req_id),
            ref_count: Arc::new(AtomicU32::new(1)),
            content_hash: 0,
            indexed: false,
            token_range,
            ckv_dtype,
            last_touched: Arc::new(AtomicU64::new(initial_epoch)),
        }
    }
}
