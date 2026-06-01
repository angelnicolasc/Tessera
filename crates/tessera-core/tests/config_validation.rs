//! Configuration validation: enforces every invariant the rest of the crate relies on.

#![allow(deprecated)] // DsaHierarchical retained for migration; deprecation tested elsewhere

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraError};

#[test]
fn rejects_zero_layers() {
    let err = MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 512,
            rope_key_dim: 64,
        },
        0,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap_err();
    assert!(matches!(err, TesseraError::InvalidConfig(_)));
}

#[test]
fn rejects_wrong_block_size_under_mla() {
    let err = MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 512,
            rope_key_dim: 64,
        },
        61,
        16,
        CkvDtype::Bf16,
        0,
    )
    .unwrap_err();
    assert!(matches!(err, TesseraError::InvalidConfig(_)));
}

#[test]
fn allows_mha_fallback() {
    let cfg = MlaBlockConfig::new(
        CompressionScheme::MhaFull {
            num_heads: 32,
            head_dim: 128,
        },
        24,
        16,
        CkvDtype::Bf16,
        0,
    )
    .expect("MHA fallback config should validate");
    assert!(cfg.primary_block_bytes() > 0);
    assert_eq!(cfg.rope_block_bytes(), 0);
}

#[test]
fn rejects_dsa_as_deprecated() {
    // Sprint 5 (post V4 paper): DsaHierarchical is deprecated; MlaBlockConfig::new rejects
    // with a migration message pointing at V4Csa / V4Hca / V4Swa.
    let err = MlaBlockConfig::new(
        CompressionScheme::DsaHierarchical {
            coarse_dim: 128,
            fine_dim: 32,
            swa_window: 128,
        },
        61,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap_err();
    assert!(matches!(err, TesseraError::InvalidConfig(_)));
}

#[test]
fn accepts_v4_csa_with_block_size_multiple_of_k1() {
    let cfg = MlaBlockConfig::new(
        CompressionScheme::V4Csa {
            k1: 4,
            head_dim: 512,
            num_heads: 128,
            rope_dim: 64,
            indexer_head_dim: 128,
            num_indexer_heads: 64,
            top_k: 1024,
        },
        61,
        128,
        CkvDtype::MixedBf16Fp8Fp4,
        0,
    )
    .expect("V4 CSA with block_size=128 must validate");
    assert!(cfg.scheme.is_v4());
    assert_eq!(cfg.block_size_tokens, 128);
}

#[test]
fn accepts_v4_hybrid_per_layer_schemes() {
    use tessera_core::CompressionScheme as C;
    let layers = vec![
        C::V4Hca {
            k2: 128,
            head_dim: 512,
            num_heads: 128,
            rope_dim: 64,
        },
        C::V4Csa {
            k1: 4,
            head_dim: 512,
            num_heads: 128,
            rope_dim: 64,
            indexer_head_dim: 128,
            num_indexer_heads: 64,
            top_k: 1024,
        },
    ];
    let cfg = MlaBlockConfig::with_per_layer_schemes(layers, 128, CkvDtype::MixedBf16Fp8Fp4, 0)
        .expect("V4 per-layer hybrid config must validate");
    assert!(cfg.has_per_layer_schemes());
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.v4_block_size_lcm(), 128);
}
