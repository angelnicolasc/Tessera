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
    // Tiny budget — at most a handful of blocks.
    let mgr = TesseraBlockManager::new(ds_v3_cfg(), 8 * 1024 * 1024).unwrap();
    let mut allocated = Vec::new();
    while let Ok(id) = mgr.allocate(1, TokenRange::new(0, 64)) {
        allocated.push(id);
    }
    // Next attempt must fail with the structured error.
    let err = mgr.allocate(1, TokenRange::new(0, 64)).unwrap_err();
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
