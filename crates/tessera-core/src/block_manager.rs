//! [`TesseraBlockManager`]: the heart of the core crate.
//!
//! The manager owns a fixed pool of MLA blocks backed by three parallel device regions:
//! the primary `c_kv` region, the secondary `k_rope` region, and (when FP8 is active) a small
//! per-layer scale-factor region. It exposes:
//!
//! * [`allocate`](TesseraBlockManager::allocate) — pull a free block off the free list;
//!   attempts LRU eviction before returning [`TesseraError::OutOfBlocks`].
//! * [`seal`](TesseraBlockManager::seal) — finalise a block after prefill writes its data,
//!   compute its content hash, and dedup against any existing identical block.
//! * [`cow_fork`](TesseraBlockManager::cow_fork) — create a private writable copy of a shared
//!   block, the foundation of safe multi-agent sharing.
//! * [`free`](TesseraBlockManager::free) — decrement the ref-count; physically returns the
//!   block to the free list only when it reaches zero.
//! * [`release_request`](TesseraBlockManager::release_request) — atomically free all private
//!   blocks allocated for a given `req_id` (TD-004 / WS1). Returns the count freed.
//!
//! **Eviction policy** (WS2 / ADR-0010): when the free list is exhausted, `allocate` invokes
//! `evict_one` before failing. Eviction priority (most eligible first):
//!   * Tier a — `ref_count == 0` (orphaned, no owner)
//!   * Tier b — `ref_count == 1`, not indexed (inactive, low reuse value)
//!   * Tier c — `ref_count == 1`, indexed (inactive, higher reuse value — avoid)
//!   * Tier d — `ref_count > 1` (shared) — **never evicted**
//!
//! Within each tier the least-recently-touched block wins (epoch-based LRU).
//!
//! Everything that touches "device" memory goes through the [`DeviceBackend`] trait. With the
//! default [`CpuMockBackend`] the entire manager is exercisable in `cargo test` on any
//! machine; with `CudaBackend` (feature `cuda`) it runs against real HBM.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};

use crate::block::{BlockId, BlockMeta, GlobalBlockId, TokenRange, SHARED_SENTINEL};
use crate::config::MlaBlockConfig;
use crate::content_hash::{ContentHasher, Xxh3Hasher};
use crate::device::{CpuMockBackend, DeviceBackend, DevicePtr, RegionKind};
use crate::error::{Result, TesseraError};
use crate::rank::{RankId, World};
use crate::transport::ReservationToken;

/// Result of [`TesseraBlockManager::seal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealOutcome {
    /// The canonical block id for this content. When a duplicate is detected this differs
    /// from the input block id; callers should use this value going forward.
    pub canonical_block: BlockId,
    /// Content hash of the sealed block's `c_kv` region.
    pub content_hash: u64,
    /// `true` when the seal collapsed a duplicate (the original block id has been freed and
    /// the canonical block's ref-count has been incremented to keep the caller alive).
    pub was_dedup: bool,
}

/// Active PD-disagg reservation entry held by this rank as the **destination** of a
/// pending transfer.
//
// Fields are populated when a reservation is created and consulted by the diagnostics
// path; the struct itself is the inventory entry. `dead_code` suppresses the lint
// while `consume_reservation_slot` (below) is wired through the import path.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct ReservationEntry {
    /// Request id the reservation belongs to.
    req_id: u64,
    /// Slots still pinned (not yet consumed by `import_payload` from a push_block).
    remaining: u32,
}

/// Which eviction tier a block falls into. Used for metrics and eviction priority ordering.
#[derive(Debug, Clone, Copy)]
enum EvictionTier {
    /// ref_count == 0: orphaned — safe to evict immediately.
    A,
    /// ref_count == 1, not indexed: inactive with low reuse value.
    B,
    /// ref_count == 1, indexed: inactive but has reuse value; evict last.
    C,
}

impl EvictionTier {
    const fn label(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::C => "c",
        }
    }
}

/// MLA-aware block manager.
///
/// Generic over [`DeviceBackend`] and [`ContentHasher`]; both default to in-process choices
/// (`CpuMockBackend` and `Xxh3Hasher`) so `TesseraBlockManager::new(config, gpu_memory_bytes)`
/// "just works" without ceremony in tests.
#[derive(Debug)]
pub struct TesseraBlockManager<B: DeviceBackend = CpuMockBackend, H: ContentHasher = Xxh3Hasher> {
    config: MlaBlockConfig,
    backend: B,
    hasher: H,

    /// This manager's rank. Defaults to [`RankId::ZERO`] under [`World::singleton`].
    rank: RankId,
    /// Cached Prometheus label for `rank` to avoid `to_string` on each metric tick.
    rank_label: Arc<str>,
    /// Shared description of the surrounding world (size, topology). Sprint 3 ships
    /// `SingleNode`; `MultiNode` is reserved for Sprint 4.
    world: Arc<World>,

    blocks: RwLock<HashMap<BlockId, BlockMeta>>,
    content_index: DashMap<u64, BlockId>,
    free_list: Mutex<Vec<BlockId>>,

    /// Per-request block ownership index. `req_id → Vec<BlockId>` for private blocks only.
    /// Populated by `allocate`; cleaned up by `free_block_internal` and `release_request`.
    /// Enables O(owned-blocks) request teardown without scanning the full block map.
    req_blocks: DashMap<u64, Vec<BlockId>>,

    /// Global monotonic epoch counter. Incremented on each `allocate` and each `primary_ptr`
    /// / `rope_ptr` access. Stored in `BlockMeta::last_touched` for LRU eviction ordering.
    next_touch_epoch: AtomicU64,

    /// Active reservations (Sprint 4 / ADR-0018). Maps token → `(req_id, remaining_slots)`.
    /// When `remaining_slots == 0` the reservation is fully consumed and removed.
    reservations: DashMap<ReservationToken, ReservationEntry>,
    /// Monotonic counter for reservation token minting on this rank. Tokens encode
    /// `(rank << 48) | counter` so they're globally unique within a deployment.
    next_reservation_id: AtomicU64,

    primary_base: DevicePtr,
    rope_base: DevicePtr,
    fp8_scales_base: Option<DevicePtr>,

    total_blocks: u32,
    used_blocks: Arc<AtomicU32>,
}

impl TesseraBlockManager<CpuMockBackend, Xxh3Hasher> {
    /// Construct a singleton-world manager backed by the default CPU mock and xxh3 hasher.
    /// Intended for tests, single-GPU deployments, and developer workstations. For
    /// multi-rank deployments use [`TesseraBlockManager::new_with_world`]. For custom
    /// backends use [`TesseraBlockManager::with_backend`].
    pub fn new(config: MlaBlockConfig, memory_bytes: u64) -> Result<Self> {
        Self::with_backend(config, memory_bytes, CpuMockBackend::new(), Xxh3Hasher)
    }

    /// Construct a multi-rank manager (intra-node TP=N typical use). Equivalent to
    /// [`TesseraBlockManager::with_backend_and_world`] using the default CPU mock backend
    /// and `Xxh3Hasher`. The `rank` and `world` are stored on the manager and surfaced via
    /// [`TesseraBlockManager::rank`] / [`TesseraBlockManager::world`] for transports and
    /// distributed components to coordinate.
    pub fn new_with_world(
        config: MlaBlockConfig,
        memory_bytes: u64,
        rank: RankId,
        world: Arc<World>,
    ) -> Result<Self> {
        Self::with_backend_and_world(
            config,
            memory_bytes,
            CpuMockBackend::new(),
            Xxh3Hasher,
            rank,
            world,
        )
    }
}

impl<B: DeviceBackend, H: ContentHasher> TesseraBlockManager<B, H> {
    /// Construct a manager from an explicit backend and hasher (singleton world). The
    /// `memory_bytes` budget determines `total_blocks` after the per-block size is computed
    /// from the config.
    pub fn with_backend(
        config: MlaBlockConfig,
        memory_bytes: u64,
        backend: B,
        hasher: H,
    ) -> Result<Self> {
        Self::with_backend_and_world(
            config,
            memory_bytes,
            backend,
            hasher,
            RankId::ZERO,
            Arc::new(World::singleton()),
        )
    }

    /// Most explicit constructor: every dependency named. Validates that `rank.raw() <
    /// world.size` before allocating any device memory.
    pub fn with_backend_and_world(
        config: MlaBlockConfig,
        memory_bytes: u64,
        backend: B,
        hasher: H,
        rank: RankId,
        world: Arc<World>,
    ) -> Result<Self> {
        if rank.raw() >= world.size {
            return Err(TesseraError::InvalidConfig(format!(
                "rank {rank} is out of range for world size {}",
                world.size
            )));
        }
        let per_block = config.total_block_bytes();
        if per_block == 0 || memory_bytes < per_block {
            return Err(TesseraError::InvalidConfig(format!(
                "memory_bytes={memory_bytes} too small for at least one block of {per_block} bytes"
            )));
        }
        let total_blocks = u32::try_from(memory_bytes / per_block)
            .map_err(|_| TesseraError::InvalidConfig("total_blocks does not fit in u32".into()))?;

        let primary_base = backend
            .alloc_region(
                config.primary_block_bytes() * u64::from(total_blocks),
                RegionKind::Primary,
            )
            .map_err(TesseraError::Backend)?;
        let rope_base = backend
            .alloc_region(
                config.rope_block_bytes() * u64::from(total_blocks),
                RegionKind::Rope,
            )
            .map_err(TesseraError::Backend)?;
        let fp8_scales_base = if config.fp8_scale_block_bytes() > 0 {
            Some(
                backend
                    .alloc_region(
                        config.fp8_scale_block_bytes() * u64::from(total_blocks),
                        RegionKind::Fp8Scales,
                    )
                    .map_err(TesseraError::Backend)?,
            )
        } else {
            None
        };

        let free_list: Vec<BlockId> = (0..total_blocks).map(BlockId).collect();

        tracing::info!(
            backend = backend.name(),
            hasher = hasher.name(),
            rank = %rank,
            world_size = world.size,
            total_blocks,
            per_block_bytes = per_block,
            compression_ratio_vs_mha = config.compression_ratio_vs_mha_bf16(),
            "TesseraBlockManager initialised"
        );

        let rank_label: Arc<str> = Arc::from(rank.to_string().into_boxed_str());
        crate::metrics::BLOCKS_PER_RANK
            .with_label_values(&[&rank_label])
            .set(0.0);

        Ok(Self {
            config,
            backend,
            hasher,
            rank,
            rank_label,
            world,
            blocks: RwLock::new(HashMap::with_capacity(total_blocks as usize)),
            content_index: DashMap::new(),
            free_list: Mutex::new(free_list),
            req_blocks: DashMap::new(),
            next_touch_epoch: AtomicU64::new(0),
            reservations: DashMap::new(),
            next_reservation_id: AtomicU64::new(1),
            primary_base,
            rope_base,
            fp8_scales_base,
            total_blocks,
            used_blocks: Arc::new(AtomicU32::new(0)),
        })
    }

    /// This manager's rank within its [`World`].
    pub const fn rank(&self) -> RankId {
        self.rank
    }

    /// Shared handle to the world description.
    pub fn world(&self) -> &Arc<World> {
        &self.world
    }

    /// Lift a local [`BlockId`] into a [`GlobalBlockId`] so it can travel across rank
    /// boundaries (transport, distributed segment index, share table).
    pub const fn global_id(&self, block_id: BlockId) -> GlobalBlockId {
        GlobalBlockId::new(self.rank, block_id)
    }

    /// Serialise the device-side payload of `block_id` into a host-owned [`crate::transport::BlockPayload`].
    /// Used by transports (mock, future P2pCuda) to ship a block across a rank boundary.
    ///
    /// Returns [`TesseraError::UnknownBlock`] if the block was evicted between the caller's
    /// reference and this call.
    pub fn export_payload(&self, block_id: BlockId) -> Result<crate::transport::BlockPayload> {
        let primary_ptr = self
            .primary_ptr(block_id)
            .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
        let primary_len = usize::try_from(self.config.primary_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("primary bytes overflow usize".into()))?;
        let c_kv = self
            .backend
            .read_bytes(primary_ptr, primary_len)
            .map_err(TesseraError::Backend)?;

        let k_rope = if self.config.rope_block_bytes() > 0 {
            let rope_ptr = self
                .rope_ptr(block_id)
                .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
            let rope_len = usize::try_from(self.config.rope_block_bytes())
                .map_err(|_| TesseraError::InvalidConfig("rope bytes overflow usize".into()))?;
            self.backend
                .read_bytes(rope_ptr, rope_len)
                .map_err(TesseraError::Backend)?
        } else {
            Vec::new()
        };

        let fp8_scales = if let Some(scales_ptr) = self.fp8_scales_ptr(block_id) {
            let scales_len =
                usize::try_from(self.config.fp8_scale_block_bytes()).map_err(|_| {
                    TesseraError::InvalidConfig("fp8 scale bytes overflow usize".into())
                })?;
            Some(
                self.backend
                    .read_bytes(scales_ptr, scales_len)
                    .map_err(TesseraError::Backend)?,
            )
        } else {
            None
        };

        Ok(crate::transport::BlockPayload {
            c_kv,
            k_rope,
            fp8_scales,
        })
    }

    /// Import a [`crate::transport::BlockPayload`] arriving from another rank, returning the
    /// newly allocated local [`BlockId`]. The block is registered under `req_id` so that
    /// `release_request(req_id)` reaps it normally.
    ///
    /// Used by the PD-disaggregation push path (`RankTransport::push_block` -> dst block
    /// manager). The payload's `c_kv` length must match `primary_block_bytes()`; mismatches
    /// produce [`TesseraError::InvalidConfig`] for caller-side debugging.
    pub fn import_payload(
        &self,
        req_id: u64,
        token_range: TokenRange,
        payload: &crate::transport::BlockPayload,
    ) -> Result<BlockId> {
        let expected_primary = usize::try_from(self.config.primary_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("primary bytes overflow usize".into()))?;
        if payload.c_kv.len() != expected_primary {
            return Err(TesseraError::InvalidConfig(format!(
                "import_payload: c_kv length {} doesn't match primary_block_bytes {}",
                payload.c_kv.len(),
                expected_primary
            )));
        }
        let block_id = self.allocate(req_id, token_range)?;

        // Write primary.
        let primary_ptr = self
            .primary_ptr(block_id)
            .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
        self.backend
            .write_bytes(primary_ptr, &payload.c_kv)
            .map_err(TesseraError::Backend)?;

        // Write rope (if any).
        if self.config.rope_block_bytes() > 0 {
            let rope_ptr = self
                .rope_ptr(block_id)
                .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
            self.backend
                .write_bytes(rope_ptr, &payload.k_rope)
                .map_err(TesseraError::Backend)?;
        }

        // Write FP8 scales (if any).
        if let (Some(scales), Some(ptr)) = (&payload.fp8_scales, self.fp8_scales_ptr(block_id)) {
            self.backend
                .write_bytes(ptr, scales)
                .map_err(TesseraError::Backend)?;
        }

        Ok(block_id)
    }

    /// Reserve `count` slots on this rank for an incoming PD-disagg transfer owned by
    /// `req_id`. Forces eviction if needed; fails with [`TesseraError::OutOfBlocks`] when
    /// the manager cannot satisfy the request even after eviction. Returns an opaque
    /// [`ReservationToken`] the source rank uses on subsequent `push_block` / rollback
    /// calls.
    ///
    /// The token encodes `(rank << 48) | counter` so tokens from different ranks never
    /// collide, even when the destination crashes and restarts.
    pub fn reserve_incoming(&self, req_id: u64, count: u32) -> Result<ReservationToken> {
        // Check capacity: free_list + evictable blocks must cover `count`.
        let free_now = self.total_blocks.saturating_sub(self.used_blocks());
        if free_now < count {
            // Try to evict the shortfall in a tight loop. evict_one is best-effort, so we
            // bound the loop to (count - free_now) attempts to avoid spinning forever on
            // an all-pinned pool.
            let needed = count - free_now;
            for _ in 0..needed {
                self.evict_one();
                if self.total_blocks.saturating_sub(self.used_blocks()) >= count {
                    break;
                }
            }
            let free_after = self.total_blocks.saturating_sub(self.used_blocks());
            if free_after < count {
                crate::metrics::TRANSFER_ABORTS_TOTAL
                    .with_label_values(&["destination_capacity"])
                    .inc();
                return Err(TesseraError::OutOfBlocks {
                    used: self.used_blocks(),
                    total: self.total_blocks,
                });
            }
        }

        let counter = self.next_reservation_id.fetch_add(1, Ordering::Relaxed);
        let raw = (u64::from(self.rank.raw()) << 48) | counter;
        let token = ReservationToken(raw);
        self.reservations.insert(
            token,
            ReservationEntry {
                req_id,
                remaining: count,
            },
        );
        crate::metrics::RESERVATIONS_ACTIVE
            .with_label_values(&[&self.rank_label])
            .set(self.reservations.len() as f64);
        Ok(token)
    }

    /// Release a previously held reservation. Safe to call even when some slots have been
    /// consumed by intervening `import_payload` calls — only the unused remainder is
    /// surrendered (the consumed blocks live independently as private blocks of the
    /// destination's request id).
    pub fn release_reservation_local(&self, token: ReservationToken) -> Result<()> {
        self.reservations.remove(&token);
        crate::metrics::RESERVATIONS_ACTIVE
            .with_label_values(&[&self.rank_label])
            .set(self.reservations.len() as f64);
        Ok(())
    }

    /// Consume one slot from a reservation. Called from `import_payload` when accepting an
    /// incoming pushed block. Returns `Ok(())` whether or not the reservation entry exists
    /// (allowing a non-reserved import path for backward compatibility with the Sprint 3
    /// tests).
    #[allow(dead_code)] // wired in by import_payload in Sprint 4; kept for transactional path
    fn consume_reservation_slot(&self, token: Option<ReservationToken>) {
        let Some(tok) = token else { return };
        if let Some(mut entry) = self.reservations.get_mut(&tok) {
            entry.remaining = entry.remaining.saturating_sub(1);
            let drained = entry.remaining == 0;
            drop(entry);
            if drained {
                self.reservations.remove(&tok);
            }
        }
        crate::metrics::RESERVATIONS_ACTIVE
            .with_label_values(&[&self.rank_label])
            .set(self.reservations.len() as f64);
    }

    /// Atomically transfer every private block owned by `req_id` to rank `target` via the
    /// reserve-then-stream protocol (ADR-0018):
    ///
    /// 1. **Reserve.** Call `transport.reserve_slots(target, req_id, count)`. Target either
    ///    pins capacity (`Ok(token)`) or aborts cleanly (`Err`).
    /// 2. **Stream.** For each block, call `transport.push_block(target, payload)`. Any
    ///    failure mid-stream triggers `transport.release_reservation(target, token)` to
    ///    surrender the unused remainder, and the source retains every block (no partial
    ///    free, no orphaned state).
    /// 3. **Commit.** On full success, call `release_request(req_id)` locally.
    ///
    /// Returns the count of blocks transferred on success, or `TesseraError::Backend`
    /// wrapping the underlying transport / reservation failure.
    pub async fn transfer_request_to_rank(
        &self,
        req_id: u64,
        target: RankId,
        transport: &Arc<dyn crate::transport::RankTransport>,
    ) -> Result<u32> {
        // Snapshot ownership.
        let block_ids: Vec<BlockId> = self
            .req_blocks
            .get(&req_id)
            .map(|e| e.value().clone())
            .unwrap_or_default();
        if block_ids.is_empty() {
            return Ok(0);
        }
        let count = u32::try_from(block_ids.len()).unwrap_or(u32::MAX);

        // Phase 1: reserve.
        let token = transport
            .reserve_slots(target, req_id, count)
            .await
            .map_err(|e| {
                crate::metrics::TRANSFER_ABORTS_TOTAL
                    .with_label_values(&["reserve_failed"])
                    .inc();
                TesseraError::Backend(e)
            })?;

        // Phase 2: stream. On any failure, surrender the reservation and bail.
        for block_id in &block_ids {
            let payload = match self.export_payload(*block_id) {
                Ok(p) => p,
                Err(e) => {
                    let _ = transport.release_reservation(target, token).await;
                    crate::metrics::TRANSFER_ABORTS_TOTAL
                        .with_label_values(&["export_failed"])
                        .inc();
                    return Err(e);
                }
            };
            if let Err(e) = transport.push_block(target, payload).await {
                let _ = transport.release_reservation(target, token).await;
                crate::metrics::TRANSFER_ABORTS_TOTAL
                    .with_label_values(&["push_failed"])
                    .inc();
                return Err(TesseraError::Backend(e));
            }
            crate::metrics::PD_DISAGG_TRANSFERS_TOTAL
                .with_label_values(&[&self.rank_label, &target.to_string()])
                .inc();
        }

        // Phase 3: commit (release source-side).
        let released = self.release_request(req_id);
        Ok(released)
    }

    /// Allocate a fresh block for `req_id`. On free-list exhaustion, attempts one eviction
    /// before returning [`TesseraError::OutOfBlocks`]. The new block has `ref_count=1` and is
    /// registered in the per-request ownership index.
    pub fn allocate(&self, req_id: u64, token_range: TokenRange) -> Result<BlockId> {
        let block_id = {
            let mut free = self.free_list.lock();
            if let Some(id) = free.pop() {
                id
            } else {
                // OOM: attempt one eviction before giving up.
                drop(free);
                self.evict_one();
                let mut free2 = self.free_list.lock();
                free2.pop().ok_or(TesseraError::OutOfBlocks {
                    used: self.used_blocks.load(Ordering::Relaxed),
                    total: self.total_blocks,
                })?
            }
        };

        let epoch = self.next_touch_epoch.fetch_add(1, Ordering::Relaxed);
        let meta = BlockMeta::fresh(block_id, req_id, token_range, self.config.ckv_dtype, epoch);
        self.blocks.write().insert(block_id, meta);

        // Track private ownership for release_request().
        self.req_blocks.entry(req_id).or_default().push(block_id);

        self.used_blocks.fetch_add(1, Ordering::Relaxed);
        crate::metrics::BLOCK_UTILIZATION.set(self.utilization());
        crate::metrics::BLOCKS_PER_RANK
            .with_label_values(&[&self.rank_label])
            .set(f64::from(self.used_blocks.load(Ordering::Relaxed)));
        Ok(block_id)
    }

    /// Increment the reference count of `block_id`. Used by the share table when registering
    /// a new owner. Errors if the block is unknown.
    pub fn increment_ref(&self, block_id: BlockId) -> Result<u32> {
        let blocks = self.blocks.read();
        let meta = blocks
            .get(&block_id)
            .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
        let prev = meta.ref_count.fetch_add(1, Ordering::AcqRel);
        Ok(prev + 1)
    }

    /// Compute the content hash, register it in the content index, and dedup against any
    /// existing identical block. The caller should swap to [`SealOutcome::canonical_block`]
    /// going forward — if `was_dedup` is `true`, `block_id` has been freed.
    ///
    /// **Sprint 5.1 hardening (ADR-0026)**: a matching `content_index` entry triggers a
    /// **byte-equality check** between the candidate block and the canonical block before
    /// dedup is committed. xxh3 is not collision-resistant against adversaries (crafted
    /// collisions are well-known), so the hash only fast-paths the lookup; the dedup
    /// authorisation requires identical bytes. On hash collision the candidate is
    /// installed as a fresh entry — the share table keeps two distinct blocks with the
    /// same hash, and a future seal of the same content collapses onto whichever installs
    /// first. The dedup-rate metric degrades; the security invariant holds.
    pub fn seal(&self, block_id: BlockId) -> Result<SealOutcome> {
        let hash = self.hash_primary(block_id)?;
        if let Some(existing) = self.content_index.get(&hash) {
            let canonical = *existing.value();
            drop(existing);
            // Byte-verify before committing to dedup. xxh3 is non-cryptographic; an
            // adversary that can submit content can craft collisions against the
            // canonical block and (without this verification) be handed a pointer to it.
            if self.blocks_have_equal_primary(block_id, canonical)? {
                self.increment_ref(canonical)?;
                self.free_block_internal(block_id)?;
                crate::metrics::EXACT_DEDUP_HITS.inc();
                return Ok(SealOutcome {
                    canonical_block: canonical,
                    content_hash: hash,
                    was_dedup: true,
                });
            }
            crate::metrics::DEDUP_HASH_COLLISIONS.inc();
            tracing::warn!(
                hash,
                candidate = ?block_id,
                canonical = ?canonical,
                "seal: hash match but bytes differ — installing as fresh block (xxh3 collision)"
            );
            // Fall through: install the candidate as a fresh, non-aliased block. The
            // content_index entry already points at `canonical`; we don't overwrite it.
            {
                let mut blocks = self.blocks.write();
                let meta = blocks
                    .get_mut(&block_id)
                    .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
                meta.content_hash = hash;
            }
            return Ok(SealOutcome {
                canonical_block: block_id,
                content_hash: hash,
                was_dedup: false,
            });
        }
        // Atomic install: use DashMap::entry to avoid the get/insert race where two
        // identical seals both pass the get and both insert (silently breaking the dedup
        // invariant — see audit H3).
        let inserted = self.content_index.entry(hash).or_insert(block_id);
        let canonical = *inserted.value();
        drop(inserted);
        if canonical != block_id {
            // Another thread won the race with an identical (or hash-colliding) block.
            // Verify bytes before deduping.
            if self.blocks_have_equal_primary(block_id, canonical)? {
                self.increment_ref(canonical)?;
                self.free_block_internal(block_id)?;
                crate::metrics::EXACT_DEDUP_HITS.inc();
                return Ok(SealOutcome {
                    canonical_block: canonical,
                    content_hash: hash,
                    was_dedup: true,
                });
            }
            // Race + hash collision — install fresh, keep both.
            crate::metrics::DEDUP_HASH_COLLISIONS.inc();
        }
        {
            let mut blocks = self.blocks.write();
            let meta = blocks
                .get_mut(&block_id)
                .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
            meta.content_hash = hash;
        }
        Ok(SealOutcome {
            canonical_block: block_id,
            content_hash: hash,
            was_dedup: false,
        })
    }

    /// Compare the primary (`c_kv`) bytes of two blocks for equality. Returns `false` if
    /// either block has been evicted between caller's reference and this call (treats a
    /// missing block as "not equal").
    fn blocks_have_equal_primary(&self, a: BlockId, b: BlockId) -> Result<bool> {
        let n = usize::try_from(self.config.primary_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("primary bytes overflow usize".into()))?;
        let ptr_a = match self.primary_ptr(a) {
            Some(p) => p,
            None => return Ok(false),
        };
        let ptr_b = match self.primary_ptr(b) {
            Some(p) => p,
            None => return Ok(false),
        };
        let bytes_a = self
            .backend
            .read_bytes(ptr_a, n)
            .map_err(TesseraError::Backend)?;
        let bytes_b = self
            .backend
            .read_bytes(ptr_b, n)
            .map_err(TesseraError::Backend)?;
        Ok(bytes_a == bytes_b)
    }

    /// **Sprint 5.1 hardening** — write per-layer FP8 scale factors for `block_id`
    /// atomically under the block manager's read lock. Replaces the prior pattern of
    /// returning a raw `usize` to Python for `ctypes.memmove`, which had a TOCTOU race
    /// between the pointer fetch and the memmove (eviction could recycle the block; the
    /// memmove would land in another tenant's data — see audit C3).
    ///
    /// The number of `scales` provided must match the number of layers in the config.
    /// No-op when FP8 is not active for the active config.
    pub fn write_fp8_scales(&self, block_id: BlockId, scales: &[f32]) -> Result<()> {
        let expected_len = self.config.num_layers as usize;
        if scales.len() != expected_len {
            return Err(TesseraError::InvalidConfig(format!(
                "write_fp8_scales: expected {} scales (one per layer), got {}",
                expected_len,
                scales.len()
            )));
        }
        // Hold the read lock for the duration of the write so the block cannot be evicted
        // (eviction takes the write lock).
        let blocks = self.blocks.read();
        if !blocks.contains_key(&block_id) {
            return Err(TesseraError::UnknownBlock(block_id.raw()));
        }
        let Some(base) = self.fp8_scales_base else {
            // No FP8 region for this config — silently no-op so callers can use a single
            // code path across MLA-BF16 and MLA-FP8 configs.
            return Ok(());
        };
        let ptr = base.offset(u64::from(block_id.raw()) * self.config.fp8_scale_block_bytes());
        // Re-interpret &[f32] as &[u8] via to_le_bytes streaming. f32::to_le_bytes
        // produces a deterministic 4-byte little-endian encoding, which matches what
        // PyTorch / numpy use for `.float32`.
        let mut bytes = Vec::with_capacity(scales.len() * 4);
        for s in scales {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let expected_bytes = usize::try_from(self.config.fp8_scale_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("fp8 scale bytes overflow usize".into()))?;
        if bytes.len() > expected_bytes {
            return Err(TesseraError::InvalidConfig(format!(
                "write_fp8_scales: encoded {} bytes exceed region {}",
                bytes.len(),
                expected_bytes
            )));
        }
        self.backend
            .write_bytes(ptr, &bytes)
            .map_err(TesseraError::Backend)?;
        drop(blocks);
        Ok(())
    }

    /// Fork a private writable copy of `block_id` for `new_req_id`. The new block is fully
    /// independent: mutations to it do not propagate to the original. The original block's
    /// ref-count is not decremented here — callers should `free` the original after the fork
    /// when they no longer hold their reference.
    ///
    /// **Sprint 5.1 hardening**: the source pointers are resolved through the checked
    /// [`Self::primary_ptr`] / [`Self::rope_ptr`] / [`Self::fp8_scales_ptr`] accessors. If
    /// the source block has been evicted between the caller's reference and the memcpy
    /// the fork fails with [`TesseraError::UnknownBlock`] rather than silently reading
    /// from recycled memory (audit H2).
    pub fn cow_fork(&self, block_id: BlockId, new_req_id: u64) -> Result<BlockId> {
        let token_range = {
            let blocks = self.blocks.read();
            blocks
                .get(&block_id)
                .ok_or(TesseraError::UnknownBlock(block_id.raw()))?
                .token_range
        };
        // Pin the source against eviction for the duration of the fork. evict_one will
        // not touch a block whose ref_count > 1.
        let _src_ref = self.increment_ref(block_id)?;
        let new_id = match self.allocate(new_req_id, token_range) {
            Ok(id) => id,
            Err(e) => {
                let _ = self.free(block_id);
                return Err(e);
            }
        };
        let result = (|| -> Result<()> {
            let src_primary = self
                .primary_ptr(block_id)
                .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
            let dst_primary = self
                .primary_ptr(new_id)
                .ok_or(TesseraError::UnknownBlock(new_id.raw()))?;
            self.backend
                .memcpy(src_primary, dst_primary, self.config.primary_block_bytes())
                .map_err(TesseraError::Backend)?;
            if self.config.rope_block_bytes() > 0 {
                let src_rope = self
                    .rope_ptr(block_id)
                    .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
                let dst_rope = self
                    .rope_ptr(new_id)
                    .ok_or(TesseraError::UnknownBlock(new_id.raw()))?;
                self.backend
                    .memcpy(src_rope, dst_rope, self.config.rope_block_bytes())
                    .map_err(TesseraError::Backend)?;
            }
            if let Some(scales_ptr_src) = self.fp8_scales_ptr(block_id) {
                let scales_ptr_dst = self
                    .fp8_scales_ptr(new_id)
                    .ok_or(TesseraError::UnknownBlock(new_id.raw()))?;
                self.backend
                    .memcpy(
                        scales_ptr_src,
                        scales_ptr_dst,
                        self.config.fp8_scale_block_bytes(),
                    )
                    .map_err(TesseraError::Backend)?;
            }
            Ok(())
        })();
        // Release the source pin regardless of outcome.
        let _ = self.free(block_id);
        result?;
        crate::metrics::COW_FORKS.inc();
        Ok(new_id)
    }

    /// Release a reference to `block_id`. Physically returns it to the free pool only when
    /// the ref-count drops to zero. If `block_id` is not found (e.g. was already evicted),
    /// returns `Ok(())` — a safe no-op consistent with the eviction contract.
    pub fn free(&self, block_id: BlockId) -> Result<()> {
        let should_free = {
            let blocks = self.blocks.read();
            let Some(meta) = blocks.get(&block_id) else {
                // Block was evicted between the caller's last use and this free call.
                // Treat as already freed — no double-panic, no double-free.
                return Ok(());
            };
            let prev = meta.ref_count.fetch_sub(1, Ordering::AcqRel);
            prev == 1
        };
        if should_free {
            self.free_block_internal(block_id)?;
        }
        Ok(())
    }

    /// Free all private blocks allocated for `req_id` and remove the ownership entry.
    /// Returns the number of blocks freed. Blocks shared with other requests (via
    /// `CrossAgentShareTable`) are **not** freed here — use the share table's
    /// `release_request` for those, then call `free` on each returned block id.
    ///
    /// This is the primary per-request teardown path (TD-004 / ADR-0009).
    pub fn release_request(&self, req_id: u64) -> u32 {
        let block_ids = self
            .req_blocks
            .remove(&req_id)
            .map(|(_, v)| v)
            .unwrap_or_default();
        let count = u32::try_from(block_ids.len()).unwrap_or(u32::MAX);
        for block_id in block_ids {
            let _ = self.free(block_id);
        }
        if count > 0 {
            crate::metrics::REQUEST_RELEASES_TOTAL.inc_by(f64::from(count));
        }
        count
    }

    /// Total compressed block bytes for this manager.
    pub const fn config(&self) -> &MlaBlockConfig {
        &self.config
    }

    /// Backend handle (for advanced wiring; most callers do not need this).
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Memory utilisation in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let used = self.used_blocks.load(Ordering::Relaxed) as f64;
        #[allow(clippy::cast_precision_loss)]
        let total = self.total_blocks as f64;
        used / total
    }

    /// Number of blocks currently allocated.
    pub fn used_blocks(&self) -> u32 {
        self.used_blocks.load(Ordering::Relaxed)
    }

    /// Total blocks managed (independent of free vs. used).
    pub const fn total_blocks(&self) -> u32 {
        self.total_blocks
    }

    /// Pointer to the primary (`c_kv`) region of `block_id`. Updates the block's
    /// `last_touched` epoch for LRU eviction ordering. Returns `None` if not allocated.
    pub fn primary_ptr(&self, block_id: BlockId) -> Option<DevicePtr> {
        let blocks = self.blocks.read();
        let meta = blocks.get(&block_id)?;
        let epoch = self.next_touch_epoch.fetch_add(1, Ordering::Relaxed);
        meta.last_touched.store(epoch, Ordering::Relaxed);
        Some(self.primary_ptr_unchecked(block_id))
    }

    /// Pointer to the rope (`k_rope`) region of `block_id`. Updates the block's
    /// `last_touched` epoch for LRU eviction ordering. Returns `None` if not allocated.
    pub fn rope_ptr(&self, block_id: BlockId) -> Option<DevicePtr> {
        let blocks = self.blocks.read();
        let meta = blocks.get(&block_id)?;
        let epoch = self.next_touch_epoch.fetch_add(1, Ordering::Relaxed);
        meta.last_touched.store(epoch, Ordering::Relaxed);
        Some(self.rope_ptr_unchecked(block_id))
    }

    /// Pointer to the FP8 scale region of `block_id`. Returns `None` when FP8 is inactive or
    /// the block is unknown.
    pub fn fp8_scales_ptr(&self, block_id: BlockId) -> Option<DevicePtr> {
        let base = self.fp8_scales_base?;
        let blocks = self.blocks.read();
        blocks
            .contains_key(&block_id)
            .then(|| base.offset(u64::from(block_id.raw()) * self.config.fp8_scale_block_bytes()))
    }

    /// Test helper: fill a deterministic pattern over the primary region of `block_id`. Used
    /// by `seal_dedup` and `cow_isolation` integration tests.
    #[doc(hidden)]
    pub fn fill_primary_test_pattern(&self, block_id: BlockId, byte: u8) -> Result<()> {
        let ptr = self
            .primary_ptr(block_id)
            .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
        let n = usize::try_from(self.config.primary_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("primary bytes overflow usize".into()))?;
        self.backend
            .fill_pattern(ptr, byte, n)
            .map_err(TesseraError::Backend)
    }

    // ─────────────────────────── internals ───────────────────────────────────

    fn primary_ptr_unchecked(&self, block_id: BlockId) -> DevicePtr {
        self.primary_base
            .offset(u64::from(block_id.raw()) * self.config.primary_block_bytes())
    }

    fn rope_ptr_unchecked(&self, block_id: BlockId) -> DevicePtr {
        self.rope_base
            .offset(u64::from(block_id.raw()) * self.config.rope_block_bytes())
    }

    /// Hash the primary region using the pluggable `hash_device` method (WS10 seam).
    fn hash_primary(&self, block_id: BlockId) -> Result<u64> {
        let ptr = self
            .primary_ptr(block_id)
            .ok_or(TesseraError::UnknownBlock(block_id.raw()))?;
        let n = usize::try_from(self.config.primary_block_bytes())
            .map_err(|_| TesseraError::InvalidConfig("primary bytes overflow usize".into()))?;
        self.hasher
            .hash_device(&self.backend, ptr, n)
            .map_err(TesseraError::Backend)
    }

    /// Physically return `block_id` to the free pool. Removes the block from the metadata
    /// map, the content index (if indexed), and the per-request ownership index.
    ///
    /// This is the single physical-free path; all logical-free paths route through here.
    fn free_block_internal(&self, block_id: BlockId) -> Result<()> {
        let removed_meta = {
            let mut blocks = self.blocks.write();
            blocks.remove(&block_id)
        };
        if let Some(meta) = removed_meta {
            // Clean up per-request ownership index.
            if let Some(req_id) = meta.req_id {
                if req_id != SHARED_SENTINEL {
                    if let Some(mut list) = self.req_blocks.get_mut(&req_id) {
                        list.retain(|&id| id != block_id);
                    }
                }
            }
            // Clean up content index (only if the block was ever sealed).
            if meta.content_hash != 0 {
                self.content_index
                    .remove_if(&meta.content_hash, |_, v| *v == block_id);
            }
        }
        self.free_list.lock().push(block_id);
        // saturating_sub guards against the rare eviction race where used_blocks could
        // transiently undercount; Relaxed ordering is sufficient (only diagnostic).
        self.used_blocks.fetch_sub(1, Ordering::Relaxed);
        crate::metrics::BLOCK_UTILIZATION.set(self.utilization());
        crate::metrics::BLOCKS_PER_RANK
            .with_label_values(&[&self.rank_label])
            .set(f64::from(self.used_blocks.load(Ordering::Relaxed)));
        Ok(())
    }

    /// Attempt to evict one block to reclaim a free slot. Prefers orphaned blocks (tier a),
    /// then unindexed inactive (tier b), then indexed inactive (tier c). Within each tier,
    /// the least-recently-touched block (lowest `last_touched` epoch) is selected. Shared
    /// blocks (`ref_count > 1`, tier d) are **never** evicted.
    ///
    /// This function is best-effort: it does nothing if no evictable block exists.
    fn evict_one(&self) {
        let candidate = {
            let blocks = self.blocks.read();

            let mut best_a: Option<(BlockId, u64)> = None;
            let mut best_b: Option<(BlockId, u64)> = None;
            let mut best_c: Option<(BlockId, u64)> = None;

            for (id, meta) in blocks.iter() {
                let rc = meta.ref_count.load(Ordering::Acquire);
                let ts = meta.last_touched.load(Ordering::Relaxed);

                match rc {
                    0 => {
                        // Tier a: orphaned — always prefer oldest epoch.
                        if best_a.is_none_or(|(_, t)| ts < t) {
                            best_a = Some((*id, ts));
                        }
                    }
                    1 if !meta.indexed => {
                        if best_b.is_none_or(|(_, t)| ts < t) {
                            best_b = Some((*id, ts));
                        }
                    }
                    1 => {
                        if best_c.is_none_or(|(_, t)| ts < t) {
                            best_c = Some((*id, ts));
                        }
                    }
                    // Tier d: ref_count > 1, shared — never evict.
                    _ => {}
                }
            }

            best_a
                .map(|(id, _)| (EvictionTier::A, id))
                .or_else(|| best_b.map(|(id, _)| (EvictionTier::B, id)))
                .or_else(|| best_c.map(|(id, _)| (EvictionTier::C, id)))
        };

        if let Some((tier, block_id)) = candidate {
            // Force-free the block regardless of ref_count. The owner may later call
            // free(block_id), which will find the block absent and return Ok(()) safely.
            let _ = self.free_block_internal(block_id);
            crate::metrics::EVICTIONS_TOTAL
                .with_label_values(&[tier.label()])
                .inc();
            tracing::debug!(?block_id, tier = tier.label(), "evicted block");
        }
    }
}
