//! Sprint 5.1 hardening — `seal()` byte verification.
//!
//! Verifies that the block manager rejects dedup when two blocks hash to the same value but
//! their bytes differ. Without this check (audit C1 / ADR-0026), an adversary that can craft
//! xxh3 collisions would be handed a pointer to another tenant's block via the share table.
//!
//! The test installs a deliberately collision-prone `ContentHasher` that returns the same
//! 64-bit value for every input, exercising the worst case: every seal hits the dedup path
//! in `content_index`. The block manager must still keep distinct blocks distinct.

use std::sync::Arc;

use tessera_core::block::TokenRange;
use tessera_core::config::{CkvDtype, CompressionScheme, MlaBlockConfig};
use tessera_core::content_hash::ContentHasher;
use tessera_core::device::CpuMockBackend;
use tessera_core::TesseraBlockManager;

#[derive(Debug)]
struct CollidingHasher {
    fixed: u64,
}

impl ContentHasher for CollidingHasher {
    fn hash(&self, _bytes: &[u8]) -> u64 {
        self.fixed
    }
    fn name(&self) -> &'static str {
        "colliding-test-hasher"
    }
}

fn make_manager() -> TesseraBlockManager<CpuMockBackend, CollidingHasher> {
    let scheme = CompressionScheme::MlaLatent {
        latent_dim: 32,
        rope_key_dim: 8,
    };
    let cfg = MlaBlockConfig::new(scheme, 4, 64, CkvDtype::Bf16, 0).expect("valid MLA config");
    let per_block = cfg.total_block_bytes();
    TesseraBlockManager::with_backend(
        cfg,
        per_block * 16,
        CpuMockBackend::new(),
        CollidingHasher { fixed: 0xDEADBEEF },
    )
    .expect("manager")
}

#[test]
fn seal_rejects_dedup_when_bytes_differ_under_hash_collision() {
    let mgr = Arc::new(make_manager());

    // Block A with pattern 0xAA
    let a = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(a, 0xAA).unwrap();
    let seal_a = mgr.seal(a).unwrap();
    assert!(!seal_a.was_dedup);
    assert_eq!(seal_a.canonical_block, a);

    // Block B with pattern 0xBB — different bytes, but the hasher returns the same hash.
    let b = mgr.allocate(2, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(b, 0xBB).unwrap();
    let seal_b = mgr.seal(b).unwrap();

    // Sprint 5.1 contract: must NOT dedup despite the hash collision.
    assert!(
        !seal_b.was_dedup,
        "byte-different blocks must not dedup even on hash collision"
    );
    assert_ne!(
        seal_b.canonical_block, a,
        "B must remain a distinct canonical block"
    );

    // Both blocks must still be addressable.
    assert!(mgr.primary_ptr(a).is_some());
    assert!(mgr.primary_ptr(seal_b.canonical_block).is_some());

    // Metric: tessera_dedup_hash_collisions_total must have incremented.
    let snap = tessera_core::metrics::snapshot_text();
    assert!(
        snap.contains("tessera_dedup_hash_collisions_total"),
        "metric must be registered: {snap}"
    );
}

#[test]
fn seal_still_dedups_when_bytes_match() {
    let mgr = Arc::new(make_manager());

    let a = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(a, 0x42).unwrap();
    let seal_a = mgr.seal(a).unwrap();
    assert!(!seal_a.was_dedup);

    let b = mgr.allocate(2, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(b, 0x42).unwrap();
    let seal_b = mgr.seal(b).unwrap();

    // Same bytes, same hash → genuine dedup.
    assert!(seal_b.was_dedup);
    assert_eq!(seal_b.canonical_block, seal_a.canonical_block);
}
