//! Stress tests for concurrency and memory pressure (WS6).
//!
//! These tests exercise `TesseraBlockManager` under conditions that unit tests do not cover:
//! multi-threaded concurrent allocation (thread-safety of `DashMap` / atomics) and sequential
//! fill-to-capacity allocation (eviction keeps the system alive and `used_blocks ≤ total`).

use std::sync::Arc;
use std::thread;

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange};

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

fn make_manager(n_blocks: u32) -> TesseraBlockManager {
    let cfg = small_cfg();
    let budget = u64::from(n_blocks) * cfg.total_block_bytes();
    TesseraBlockManager::new(cfg, budget).unwrap()
}

/// 50 threads each perform an alloc → release_request cycle concurrently. After all threads
/// join, `used_blocks` must be 0 — no ref-count leak under concurrent access.
#[test]
fn concurrent_alloc_release_no_leak() {
    const N_THREADS: u32 = 50;
    const BLOCKS_PER_THREAD: u32 = 3;

    let mgr = Arc::new(make_manager(N_THREADS * BLOCKS_PER_THREAD + 20));

    let handles: Vec<_> = (0..N_THREADS)
        .map(|req_id| {
            let mgr = Arc::clone(&mgr);
            thread::spawn(move || {
                let req_id = u64::from(req_id + 1);
                let token_base = (req_id as u32) * BLOCKS_PER_THREAD * 64;
                let mut block_ids = Vec::with_capacity(BLOCKS_PER_THREAD as usize);
                for i in 0..BLOCKS_PER_THREAD {
                    let start = token_base + i * 64;
                    let bid = mgr
                        .allocate(req_id, TokenRange::new(start, start + 64))
                        .unwrap();
                    block_ids.push(bid);
                }
                // Release the request — all private blocks must be freed.
                mgr.release_request(req_id);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(
        mgr.used_blocks(),
        0,
        "ref-count leak: {} blocks still live after all threads exited",
        mgr.used_blocks()
    );
}

/// Allocate blocks up to near-full capacity, then allocate beyond capacity. The block manager
/// must evict Tier-b blocks (allocated but not shared) rather than panicking or returning an
/// error. `used_blocks` must never exceed `total_blocks`.
#[test]
fn eviction_under_full_pressure_no_panic() {
    let n_total: u32 = 50;
    let mgr = make_manager(n_total);

    // Fill to 48/50 (96%).
    for i in 0..48u32 {
        let _ = mgr
            .allocate(u64::from(i + 1), TokenRange::new(0, 64))
            .unwrap();
    }
    assert_eq!(mgr.used_blocks(), 48);

    // Attempt 10 more allocations. Each should trigger an eviction of an existing Tier-b block
    // rather than returning OutOfBlocks.
    for i in 0..10u32 {
        let bid = mgr.allocate(u64::from(1000 + i), TokenRange::new(0, 64));
        assert!(
            bid.is_ok(),
            "allocation {i} should succeed via eviction; got: {bid:?}"
        );
    }

    assert!(
        mgr.used_blocks() <= mgr.total_blocks(),
        "used_blocks {} exceeded total_blocks {}",
        mgr.used_blocks(),
        mgr.total_blocks()
    );
}
