//! Content-addressed deduplication: two blocks with identical `c_kv` bytes must collapse.

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange};

fn small_mla_cfg() -> MlaBlockConfig {
    // Use a tiny model footprint so the test runs in milliseconds even with the mock backend.
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 32,
            rope_key_dim: 8,
        },
        4,  // 4 layers
        64, // mandatory block size
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

#[test]
fn first_seal_records_canonical_block() {
    let mgr = TesseraBlockManager::new(small_mla_cfg(), 16 * 1024 * 1024).unwrap();
    let id = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(id, 0xAA).unwrap();
    let out = mgr.seal(id).unwrap();
    assert!(!out.was_dedup);
    assert_eq!(out.canonical_block, id);
}

#[test]
fn second_seal_with_identical_bytes_dedups() {
    let mgr = TesseraBlockManager::new(small_mla_cfg(), 16 * 1024 * 1024).unwrap();
    let a = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let b = mgr.allocate(2, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(a, 0xCC).unwrap();
    mgr.fill_primary_test_pattern(b, 0xCC).unwrap();

    let first = mgr.seal(a).unwrap();
    assert!(!first.was_dedup);

    let second = mgr.seal(b).unwrap();
    assert!(second.was_dedup);
    assert_eq!(second.canonical_block, a);
    // The duplicate must have been returned to the free pool.
    assert_eq!(mgr.used_blocks(), 1);
}

#[test]
fn different_bytes_do_not_dedup() {
    let mgr = TesseraBlockManager::new(small_mla_cfg(), 16 * 1024 * 1024).unwrap();
    let a = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let b = mgr.allocate(2, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(a, 0x11).unwrap();
    mgr.fill_primary_test_pattern(b, 0x22).unwrap();

    assert!(!mgr.seal(a).unwrap().was_dedup);
    assert!(!mgr.seal(b).unwrap().was_dedup);
    assert_eq!(mgr.used_blocks(), 2);
}
