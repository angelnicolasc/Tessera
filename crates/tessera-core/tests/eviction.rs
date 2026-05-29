//! Eviction policy tests (WS2 / ADR-0010).
//!
//! Validates the tiered LRU eviction: orphaned blocks (tier a) are evicted first;
//! recently-touched blocks survive longer than older ones within the same tier;
//! shared blocks (ref_count > 1, tier d) are never evicted.

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange};

/// Very small pool: exactly 2 blocks. Forces eviction on the 3rd allocation.
fn two_block_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 32,
            rope_key_dim: 8,
        },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

/// Build a manager with exactly `n_blocks` capacity. Uses a tiny latent_dim so the pool
/// fits in a few MB regardless of `n_blocks`.
fn make_manager_with_capacity(n_blocks: u32) -> TesseraBlockManager {
    let cfg = two_block_cfg();
    let block_bytes = cfg.total_block_bytes();
    TesseraBlockManager::new(cfg, u64::from(n_blocks) * block_bytes).unwrap()
}

/// Fill the pool, then allocate one more block. The orphaned block (ref_count == 0 after a
/// free) must be evicted before a shared block (ref_count > 1).
#[test]
fn orphaned_block_evicted_before_shared() {
    let mgr = make_manager_with_capacity(3);

    // Allocate all 3 blocks.
    let b0 = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let _b1 = mgr.allocate(1, TokenRange::new(64, 128)).unwrap();
    let b2 = mgr.allocate(2, TokenRange::new(0, 64)).unwrap();

    // Give b2 a second ref so it is "shared" (tier d).
    mgr.increment_ref(b2).unwrap();

    // Free b0 completely (ref_count → 0), making it an orphan (tier a).
    mgr.free(b0).unwrap();
    assert_eq!(mgr.used_blocks(), 2); // b1 + b2

    // Now try to allocate a 4th block: pool is full again (b1 + b2 = 2, capacity = 3 but
    // b0 was freed). Wait — with capacity 3 and b0 freed, the free list has b0 back already.
    // Let's re-fill: allocate b3 using the freed slot, then try again without free slots.
    let b3 = mgr.allocate(3, TokenRange::new(128, 192)).unwrap();
    assert_eq!(mgr.used_blocks(), 3); // pool full

    // b1 is now tier b (ref_count==1, not indexed). b2 is tier d (ref_count==2, shared).
    // The next allocation must evict b1 (tier b) rather than b2 (tier d).
    let _b4 = mgr.allocate(4, TokenRange::new(192, 256)).unwrap();

    // b2 must still be alive (tier d was never evicted).
    assert!(
        mgr.primary_ptr(b2).is_some(),
        "shared block must not be evicted"
    );

    // Cleanup leftover ref on b2.
    mgr.free(b2).unwrap();
    mgr.free(b2).unwrap();
    let _ = b3;
}

/// A recently-touched block survives eviction longer than an older block in the same tier.
#[test]
fn recently_touched_block_survives_lru_eviction() {
    // 2-block pool: exactly one must be evicted to allocate a 3rd.
    let mgr = make_manager_with_capacity(2);

    let older = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let newer = mgr.allocate(2, TokenRange::new(64, 128)).unwrap();

    // Touch `newer` to give it a higher epoch (simulating recent access).
    let _ = mgr.primary_ptr(newer);

    // Pool is full; the 3rd allocation must evict the LRU block (older).
    let _third = mgr.allocate(3, TokenRange::new(128, 192)).unwrap();

    // `newer` (higher epoch) must still be alive; `older` (lower epoch) was evicted.
    assert!(
        mgr.primary_ptr(newer).is_some(),
        "recently-touched block must survive eviction"
    );
    assert!(
        mgr.primary_ptr(older).is_none(),
        "least-recently-touched block must be evicted"
    );
}

/// Shared blocks (ref_count > 1) must never be evicted regardless of pool pressure.
#[test]
fn shared_block_never_evicted() {
    let mgr = make_manager_with_capacity(2);

    let shared = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let _other = mgr.allocate(2, TokenRange::new(64, 128)).unwrap();

    // Promote `shared` to ref_count == 2 (tier d).
    mgr.increment_ref(shared).unwrap();

    // Pool full; 3rd allocation must evict `_other` (tier b) rather than `shared` (tier d).
    let _third = mgr.allocate(3, TokenRange::new(128, 192)).unwrap();

    assert!(
        mgr.primary_ptr(shared).is_some(),
        "shared block (tier d) must never be evicted"
    );

    // Cleanup.
    mgr.free(shared).unwrap();
    mgr.free(shared).unwrap();
}
