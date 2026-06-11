//! Per-request State Cache for DeepSeek-V4 hybrid attention.
//!
//! V4 introduces a **two-tier KV cache** (paper §3.5.1):
//! 1. **KV Cache** — paged blocks holding compressed CSA / HCA entries. This is the
//!    pool [`crate::block_manager::TesseraBlockManager`] already manages.
//! 2. **State Cache** — fixed-size arena per request holding:
//!    - SWA KV entries for the most recent `win` tokens (uncompressed).
//!    - Uncompressed tail tokens awaiting CSA / HCA compression (less than `k1` or `k2`
//!      tokens since the last compression block).
//!
//! State Cache differs from the block pool in three structural ways:
//! 1. **Per-request lifetime.** Allocated on first use, freed on `release_request`.
//! 2. **Fixed size per request.** No paging — one contiguous arena per request.
//! 3. **No content-address sharing.** SWA entries are position-dependent (just like
//!    `k_rope` in V3 MLA); the tail buffer is also position-bound.
//!
//! See ADR-0023.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;

use crate::device::{DeviceBackend, DevicePtr, RegionKind};
use crate::error::{Result, TesseraError};

/// One per-request state arena. Holds two contiguous sub-regions: the SWA buffer (for the
/// most recent `win` tokens) and the uncompressed-tail buffer (for tokens awaiting
/// compression). The arena is allocated on first `allocate_for_request` and freed via
/// `release_request`.
#[derive(Debug, Clone, Copy)]
struct StateEntry {
    /// Pointer to the SWA region of this request's arena.
    swa_ptr: DevicePtr,
    /// Pointer to the uncompressed-tail region.
    tail_ptr: DevicePtr,
    /// Arena slot index within the global state-cache pool.
    slot: u32,
}

/// Configuration for the state cache. All sizes are in bytes; size depends on the active
/// model's per-layer V4Swa scheme parameters and the number of layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateCacheConfig {
    /// Bytes per request for the SWA sub-region.
    pub swa_bytes_per_request: u64,
    /// Bytes per request for the uncompressed-tail sub-region.
    pub tail_bytes_per_request: u64,
    /// Max concurrent requests the cache supports. Pre-allocates `max_requests *
    /// (swa_bytes + tail_bytes)`.
    pub max_requests: u32,
}

impl StateCacheConfig {
    /// Total device bytes pre-allocated by this configuration.
    pub fn total_bytes(&self) -> u64 {
        u64::from(self.max_requests) * (self.swa_bytes_per_request + self.tail_bytes_per_request)
    }

    /// Derive a config from V4 parameters. `head_dim`, `rope_dim`, `num_layers` and `win`
    /// come from the V4Swa scheme; `max_uncompressed_tail` is the worst-case tail length
    /// awaiting compression — typically `max(k1, k2) - 1` for canonical V4-Pro that's 127.
    pub fn for_v4(
        head_dim: u32,
        rope_dim: u32,
        num_layers: u32,
        win: u32,
        max_uncompressed_tail: u32,
        max_requests: u32,
    ) -> Self {
        // Per token: BF16(rope_dim) + FP8(head_dim - rope_dim) = bf16*2 + fp8*1 bytes.
        let bytes_per_token =
            u64::from(rope_dim) * 2 + u64::from(head_dim.saturating_sub(rope_dim));
        let swa_bytes_per_request = bytes_per_token * u64::from(num_layers) * u64::from(win);
        let tail_bytes_per_request =
            bytes_per_token * u64::from(num_layers) * u64::from(max_uncompressed_tail);
        Self {
            swa_bytes_per_request,
            tail_bytes_per_request,
            max_requests,
        }
    }
}

/// State Cache manager. Sized at construction; thereafter `allocate_for_request` /
/// `release_request` lend / return arena slots in O(1).
#[derive(Debug)]
pub struct StateCache<B: DeviceBackend> {
    config: StateCacheConfig,
    backend: B,

    /// Active slots: `req_id → StateEntry`.
    active: DashMap<u64, StateEntry>,

    /// Free-list of slot indices. Wrapped in `Mutex<Vec<u32>>` to match the block
    /// manager's free-list contention model.
    free_slots: Mutex<Vec<u32>>,

    /// Region bases.
    swa_base: DevicePtr,
    tail_base: DevicePtr,

    /// Used-slot counter (atomic for lock-free `utilization` reads).
    used: Arc<AtomicU32>,
}

impl<B: DeviceBackend> StateCache<B> {
    /// Construct a new state cache, pre-allocating both regions on `backend`.
    pub fn new(config: StateCacheConfig, backend: B) -> Result<Self> {
        if config.max_requests == 0 {
            return Err(TesseraError::InvalidConfig(
                "StateCacheConfig::max_requests must be > 0".into(),
            ));
        }
        let swa_total = config.swa_bytes_per_request * u64::from(config.max_requests);
        let tail_total = config.tail_bytes_per_request * u64::from(config.max_requests);
        let swa_base = backend
            .alloc_region(swa_total.max(1), RegionKind::Primary)
            .map_err(TesseraError::Backend)?;
        let tail_base = backend
            .alloc_region(tail_total.max(1), RegionKind::Rope)
            .map_err(TesseraError::Backend)?;
        let free_slots: Vec<u32> = (0..config.max_requests).rev().collect();
        Ok(Self {
            config,
            backend,
            active: DashMap::new(),
            free_slots: Mutex::new(free_slots),
            swa_base,
            tail_base,
            used: Arc::new(AtomicU32::new(0)),
        })
    }

    /// Borrow the configuration.
    pub const fn config(&self) -> &StateCacheConfig {
        &self.config
    }

    /// Allocate a state arena slot for `req_id`. If the request already holds a slot the
    /// existing pointers are returned (idempotent). Returns `OutOfBlocks` when the pool
    /// is exhausted.
    pub fn allocate_for_request(&self, req_id: u64) -> Result<(DevicePtr, DevicePtr)> {
        if let Some(entry) = self.active.get(&req_id) {
            return Ok((entry.swa_ptr, entry.tail_ptr));
        }
        let slot = {
            let mut free = self.free_slots.lock();
            free.pop().ok_or(TesseraError::OutOfBlocks {
                used: self.used.load(Ordering::Relaxed),
                total: self.config.max_requests,
            })?
        };
        let swa_ptr = self
            .swa_base
            .offset(u64::from(slot) * self.config.swa_bytes_per_request);
        let tail_ptr = self
            .tail_base
            .offset(u64::from(slot) * self.config.tail_bytes_per_request);
        let entry = StateEntry {
            swa_ptr,
            tail_ptr,
            slot,
        };
        self.active.insert(req_id, entry);
        self.used.fetch_add(1, Ordering::Relaxed);
        Ok((swa_ptr, tail_ptr))
    }

    /// Release the state arena held by `req_id`. Idempotent (releasing a non-active
    /// request is a no-op). Returns `true` when a slot was actually freed.
    pub fn release_request(&self, req_id: u64) -> bool {
        let removed = self.active.remove(&req_id);
        if let Some((_, entry)) = removed {
            self.free_slots.lock().push(entry.slot);
            self.used.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Look up the current arena pointers for `req_id`, without allocating.
    pub fn ptrs_for(&self, req_id: u64) -> Option<(DevicePtr, DevicePtr)> {
        self.active.get(&req_id).map(|e| (e.swa_ptr, e.tail_ptr))
    }

    /// Number of requests currently holding a slot.
    pub fn used(&self) -> u32 {
        self.used.load(Ordering::Relaxed)
    }

    /// Total slot capacity.
    pub const fn capacity(&self) -> u32 {
        self.config.max_requests
    }

    /// Utilisation in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.used.load(Ordering::Relaxed) as f64 / self.config.max_requests as f64
        }
    }

    /// Backend handle.
    pub const fn backend(&self) -> &B {
        &self.backend
    }
}

/// Trait abstracting "a per-request arena" for any future architectures that need their
/// own state-space tier beyond V4's SWA-and-tail layout. Sprint 5 only ships `StateCache`
/// as the concrete impl; the trait stays minimal so future variants (e.g. Mamba state, KV
/// projection scratch) plug in without breaking callers.
pub trait RequestArena: Send + Sync {
    /// Acquire (or look up) per-request pointers. Implementations may return multiple
    /// pointers; Sprint 5 uses `(swa_ptr, tail_ptr)`.
    fn acquire(&self, req_id: u64) -> Result<(DevicePtr, DevicePtr)>;
    /// Release per-request state. Returns `true` if state was released.
    fn release(&self, req_id: u64) -> bool;
    /// Diagnostic name.
    fn name(&self) -> &'static str;
}

impl<B: DeviceBackend> RequestArena for StateCache<B> {
    fn acquire(&self, req_id: u64) -> Result<(DevicePtr, DevicePtr)> {
        self.allocate_for_request(req_id)
    }
    fn release(&self, req_id: u64) -> bool {
        self.release_request(req_id)
    }
    fn name(&self) -> &'static str {
        "v4_state_cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuMockBackend;

    fn cfg() -> StateCacheConfig {
        StateCacheConfig::for_v4(
            /* head_dim */ 512, /* rope_dim */ 64, /* num_layers */ 4,
            /* win */ 128, /* max_uncompressed_tail */ 127, /* max_requests */ 8,
        )
    }

    #[test]
    fn allocation_lifecycle_is_idempotent() {
        let sc = StateCache::new(cfg(), CpuMockBackend::new()).unwrap();
        let (swa1, tail1) = sc.allocate_for_request(7).unwrap();
        let (swa2, tail2) = sc.allocate_for_request(7).unwrap();
        // Sprint 5.1: DevicePtr is now a `(region, offset, len)` handle. Identity
        // comparison via the derived PartialEq still works.
        assert_eq!((swa1, tail1), (swa2, tail2));
        assert_eq!(sc.used(), 1);
        assert!(sc.release_request(7));
        assert_eq!(sc.used(), 0);
        // Releasing an unknown request is a no-op.
        assert!(!sc.release_request(42));
    }

    #[test]
    fn exhausted_pool_returns_out_of_blocks() {
        let sc = StateCache::new(cfg(), CpuMockBackend::new()).unwrap();
        for r in 0..8 {
            sc.allocate_for_request(r).unwrap();
        }
        let err = sc.allocate_for_request(99).unwrap_err();
        assert!(matches!(err, TesseraError::OutOfBlocks { .. }));
    }

    #[test]
    fn for_v4_config_matches_paper_constants() {
        // V4-Pro SWA: head_dim=512, rope_dim=64, 61 layers, win=128.
        // Per token bytes: 64*2 + (512-64)*1 = 128 + 448 = 576.
        // SWA region: 576 * 61 * 128 = 4,497,408 bytes per request.
        let c = StateCacheConfig::for_v4(512, 64, 61, 128, 127, 4);
        assert_eq!(c.swa_bytes_per_request, 576 * 61 * 128);
        assert_eq!(c.tail_bytes_per_request, 576 * 61 * 127);
    }

    #[test]
    fn request_arena_trait_is_object_safe() {
        let sc: Arc<dyn RequestArena> =
            Arc::new(StateCache::new(cfg(), CpuMockBackend::new()).unwrap());
        sc.acquire(1).unwrap();
        assert!(sc.release(1));
        assert_eq!(sc.name(), "v4_state_cache");
    }
}
