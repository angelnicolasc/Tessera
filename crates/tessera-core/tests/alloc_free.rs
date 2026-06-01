//! Allocation / free / utilization round-trip invariants.

use std::collections::HashSet;

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange};

fn ds_v3_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 512,
            rope_key_dim: 64,
        },
        61,
        64,
        CkvDtype::Bf16,
        0,
    )
    .expect("DeepSeek-V3 reference config is valid")
}

#[test]
fn allocate_then_free_returns_to_pool() {
    let mgr = TesseraBlockManager::new(ds_v3_cfg(), 256 * 1024 * 1024).unwrap();
    let initial = mgr.total_blocks();
    assert!(initial > 0, "block budget must yield at least one block");

    let ids: Vec<_> = (0u64..10)
        .map(|i| mgr.allocate(i, TokenRange::new(0, 64)).unwrap())
        .collect();
    assert_eq!(mgr.used_blocks(), 10);
    assert_eq!(ids.iter().copied().collect::<HashSet<_>>().len(), 10);

    for id in ids {
        mgr.free(id).unwrap();
    }
    assert_eq!(mgr.used_blocks(), 0);
}

#[test]
fn oom_returns_typed_error() {
    // Tiny budget — at most a handful of blocks. We pin every allocated block via
    // `increment_ref` so it lands in eviction tier d (`ref_count > 1`, never evicted);
    // otherwise the manager's one-shot eviction would happily reclaim a tier-b block
    // on every alloc call and the loop would never surface OOM. (This was the bug
    // behind the Windows runner's 5-hour 30 GB hang — the original test relied on a
    // guarantee that eviction does not make.)
    let mgr = TesseraBlockManager::new(ds_v3_cfg(), 8 * 1024 * 1024).unwrap();
    let total = mgr.total_blocks();
    for i in 0..total {
        let token = i * 64;
        let id = mgr
            .allocate(u64::from(i) + 1, TokenRange::new(token, token + 64))
            .unwrap();
        mgr.increment_ref(id).unwrap(); // pin → tier d, eviction-immune
    }
    // Free list is drained AND every block is pinned; the next call must fail.
    let err = mgr.allocate(0, TokenRange::new(0, 64)).unwrap_err();
    assert!(matches!(
        err,
        tessera_core::TesseraError::OutOfBlocks { .. }
    ));
}

#[test]
fn compression_ratio_deepseek_v3_is_about_57x() {
    let cfg = ds_v3_cfg();
    let r = cfg.compression_ratio_vs_mha_bf16();
    assert!(
        (55.0..58.0).contains(&r),
        "compression ratio {r} not in the playbook-claimed (55, 58) band"
    );
}
