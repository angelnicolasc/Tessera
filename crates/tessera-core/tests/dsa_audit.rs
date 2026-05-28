//! DSA deprecation audit (originally Sprint 2 WS4 / TD-017; revised Sprint 5 / ADR-0020).
//!
//! `CompressionScheme::DsaHierarchical` was Sprint 0's placeholder for DeepSeek-V4's
//! hierarchical compression. The V4 paper (May 2026) made it obsolete — the real
//! architecture is CSA + HCA + SWA, modelled as three sibling variants. The variant is
//! kept for backward compatibility but every code path that touches it now:
//!   1. Surfaces a deprecation warning (compile-time, via `#[deprecated]`).
//!   2. Either panics with a migration-pointing message OR fails validation with one.
//!
//! This test file pins both behaviours so future refactors don't silently drift.

#![allow(deprecated)]

use std::panic;

use tessera_core::{CkvDtype, CompressionScheme, MlaBlockConfig, TesseraError};

fn dsa_scheme() -> CompressionScheme {
    CompressionScheme::DsaHierarchical {
        coarse_dim: 128,
        fine_dim: 32,
        swa_window: 128,
    }
}

#[test]
fn dsa_primary_bytes_panics_with_v4_migration_message() {
    let scheme = dsa_scheme();
    let result = panic::catch_unwind(move || scheme.primary_bytes_per_token(4, 2));
    let err = result.expect_err("primary_bytes_per_token(DSA) must panic");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic>");
    assert!(
        msg.contains("deprecated"),
        "panic must say 'deprecated'; got: {msg}"
    );
    assert!(
        msg.contains("V4Csa") || msg.contains("V4Hca") || msg.contains("V4Swa"),
        "panic must point at the V4 migration target; got: {msg}"
    );
}

#[test]
fn dsa_rope_bytes_panics_with_v4_migration_message() {
    let scheme = dsa_scheme();
    let result = panic::catch_unwind(move || scheme.rope_bytes_per_token(4));
    let err = result.expect_err("rope_bytes_per_token(DSA) must panic");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic>");
    assert!(msg.contains("deprecated"));
    assert!(msg.contains("V4Csa") || msg.contains("V4Hca") || msg.contains("V4Swa"));
}

#[test]
fn dsa_mla_config_new_returns_invalid_config_with_adr_0020_link() {
    let err = MlaBlockConfig::new(dsa_scheme(), 4, 64, CkvDtype::Bf16, 0)
        .expect_err("DSA scheme must be rejected by MlaBlockConfig::new");
    assert!(
        matches!(err, TesseraError::InvalidConfig(_)),
        "expected InvalidConfig, got: {err:?}"
    );
    if let TesseraError::InvalidConfig(msg) = &err {
        assert!(
            msg.contains("ADR-0020") || msg.contains("deprecated"),
            "error message should reference ADR-0020; got: {msg}"
        );
        assert!(
            msg.contains("V4Csa") || msg.contains("V4Hca") || msg.contains("V4Swa"),
            "error message should point at V4 migration target; got: {msg}"
        );
    }
}
