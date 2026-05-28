//! Copy-on-write fork: mutations to the fork must not propagate to the original.

use tessera_core::{
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn small_cfg() -> MlaBlockConfig {
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

#[test]
fn cow_fork_produces_independent_block() {
    let mgr = TesseraBlockManager::new(small_cfg(), 16 * 1024 * 1024).unwrap();
    let original = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(original, 0x01).unwrap();

    // Seal so it lives in the content index.
    let seal_a = mgr.seal(original).unwrap();
    assert!(!seal_a.was_dedup);

    let forked = mgr.cow_fork(original, 2).unwrap();
    assert_ne!(forked, original);

    // Mutate the fork. The original's content hash should still equal seal_a.content_hash.
    mgr.fill_primary_test_pattern(forked, 0xFF).unwrap();

    // Sealing the fork must NOT collide with the original (different bytes ⇒ different hash).
    let seal_b = mgr.seal(forked).unwrap();
    assert!(!seal_b.was_dedup);
    assert_ne!(seal_a.content_hash, seal_b.content_hash);
}
