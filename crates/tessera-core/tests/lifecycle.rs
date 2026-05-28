//! Per-request lifecycle tests (WS1 / ADR-0009).
//!
//! Validates that `release_request` correctly frees all private blocks for a request, that
//! shared blocks are unaffected when only one owner releases, and that the `used_blocks`
//! counter reaches zero after a complete teardown.

use tessera_core::{
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn small_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent { latent_dim: 32, rope_key_dim: 8 },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

fn make_manager() -> TesseraBlockManager {
    TesseraBlockManager::new(small_cfg(), 32 * 1024 * 1024).unwrap()
}

/// Allocate N blocks for a single request, release the request, verify used_blocks reaches
/// zero.
#[test]
fn alloc_n_then_release_request_drops_to_zero() {
    let mgr = make_manager();
    let req_a: u64 = 42;
    let n = 5;
    for i in 0..n {
        mgr.allocate(req_a, TokenRange::new(i as u32 * 64, (i as u32 + 1) * 64)).unwrap();
    }
    assert_eq!(mgr.used_blocks(), n);

    let freed = mgr.release_request(req_a);
    assert_eq!(freed, n, "release_request must report the freed count");
    assert_eq!(mgr.used_blocks(), 0, "all blocks must be returned to free pool");
}

/// Two requests allocate disjoint blocks; releasing one request leaves the other untouched.
#[test]
fn release_request_partial_respects_other_owner() {
    let mgr = make_manager();
    let req_a: u64 = 1;
    let req_b: u64 = 2;

    let a_block = mgr.allocate(req_a, TokenRange::new(0, 64)).unwrap();
    let b_block = mgr.allocate(req_b, TokenRange::new(64, 128)).unwrap();

    assert_eq!(mgr.used_blocks(), 2);

    // Release req_a only.
    let freed = mgr.release_request(req_a);
    assert_eq!(freed, 1);
    assert_eq!(mgr.used_blocks(), 1, "req_b's block must survive");

    // req_b's block is still accessible.
    assert!(mgr.primary_ptr(b_block).is_some(), "block must still be allocated");
    // req_a's block is gone.
    assert!(mgr.primary_ptr(a_block).is_none(), "block must be freed");
}

/// Shared blocks (ref_count > 1) are not freed by the block manager's release_request —
/// the share table governs shared ownership.
#[test]
fn shared_block_survives_one_release() {
    let mgr = make_manager();
    let req_a: u64 = 10;

    let block = mgr.allocate(req_a, TokenRange::new(0, 64)).unwrap();

    // Simulate a second owner by incrementing ref_count (normally done via share table).
    mgr.increment_ref(block).unwrap();

    // Release req_a's private ownership.
    let _freed = mgr.release_request(req_a);

    // The block is still alive because ref_count > 0 (it was 2, now 1 after the free in
    // release_request decremented it).
    assert_eq!(mgr.used_blocks(), 1, "shared block must still be allocated");
    assert!(mgr.primary_ptr(block).is_some());

    // Manual cleanup of the remaining reference.
    mgr.free(block).unwrap();
    assert_eq!(mgr.used_blocks(), 0);
}

/// release_request on an unknown req_id is a no-op that returns 0.
#[test]
fn release_unknown_request_is_noop() {
    let mgr = make_manager();
    let freed = mgr.release_request(9999);
    assert_eq!(freed, 0);
    assert_eq!(mgr.used_blocks(), 0);
}
