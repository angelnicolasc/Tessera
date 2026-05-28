//! Configuration types for the Tessera block manager.
//!
//! [`CompressionScheme`] is the central extension hook of Sprint 0. It is marked
//! `#[non_exhaustive]` so that adding a future variant (e.g. real DSA c4a/c128a logic) cannot
//! happen silently: `cargo build` flags every `match` site in the codebase that has not yet
//! been taught the new variant. That is, refactors expand from a compiler-generated worklist
//! instead of a manual audit. See `docs/src/adr/0004-compression-scheme-enum.md`.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TesseraError};

/// FlashMLA's native paged block size. Anything else here is a configuration error; the kernel
/// assumes a 64-token page boundary internally.
pub const REQUIRED_BLOCK_SIZE_TOKENS: u32 = 64;

/// Supported on-device storage dtypes for the primary KV region.
///
/// Sprint 0 shipped BF16 + FP8 E4M3 for V3-style MLA storage. Sprint 5 adds FP4 E2M1 for the
/// V4 Lightning Indexer + a `MixedBf16Fp8Fp4` flag for the V4 hybrid layout (BF16 RoPE +
/// FP8 content + FP4 indexer co-resident in a single compressed entry — see ADR-0022).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CkvDtype {
    /// IEEE-style bfloat16, 2 bytes per element. Default; no calibration required.
    Bf16,
    /// FP8 E4M3 (1 byte per element). Requires per-layer scale factors from calibration.
    Fp8E4m3,
    /// FP4 E2M1 (4 bits per element = 0.5 bytes; packed two-per-byte). Used by the V4
    /// Lightning Indexer (§2.3.1 of the V4 paper). Requires per-block scale factors.
    Fp4E2m1,
    /// Sentinel marker: the scheme variant carries its own mixed-precision layout, and the
    /// scalar `ckv_dtype` is informational only. V4 hybrid schemes (`V4Csa`/`V4Hca`/`V4Swa`)
    /// always use this — they encode `(BF16 RoPE region, FP8 content region, FP4 indexer
    /// region)` inside the variant itself. See ADR-0022.
    MixedBf16Fp8Fp4,
}

impl CkvDtype {
    /// Bits per element of storage under this dtype. Use [`bytes_for_elements`] when you
    /// need a byte count from a known element count.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Bf16 => 16,
            Self::Fp8E4m3 => 8,
            Self::Fp4E2m1 => 4,
            // Mixed layouts: 0 indicates "ask the scheme". Callers must not use this for
            // sizing — they should read the scheme's `bytes_per_token_per_layer`.
            Self::MixedBf16Fp8Fp4 => 0,
        }
    }

    /// Bytes per element, rounding up for sub-byte dtypes. Returns 0 for
    /// `MixedBf16Fp8Fp4` to signal that scheme-level accounting is required.
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4m3 => 1,
            // FP4 is 4 bits — one byte holds two elements; callers must consult
            // `bytes_for_elements` for accurate sizing.
            Self::Fp4E2m1 => 1,
            Self::MixedBf16Fp8Fp4 => 0,
        }
    }

    /// Exact byte count to store `n` elements of this dtype, honouring sub-byte packing.
    pub const fn bytes_for_elements(self, n: u64) -> u64 {
        match self {
            Self::Bf16 => n * 2,
            Self::Fp8E4m3 => n,
            // FP4 packs two elements per byte; ceiling-divide so an odd count rounds up.
            Self::Fp4E2m1 => (n + 1) / 2,
            // Caller error: mixed dtype must be sized from the scheme.
            Self::MixedBf16Fp8Fp4 => 0,
        }
    }

    /// Whether per-layer scale factors must be present on every block when storing under
    /// this dtype. FP8 and FP4 both quantise; mixed inherits from its components.
    pub const fn requires_per_layer_scales(self) -> bool {
        matches!(self, Self::Fp8E4m3 | Self::Fp4E2m1 | Self::MixedBf16Fp8Fp4)
    }
}

/// How a single token's KV is stored in the block layout.
///
/// `#[non_exhaustive]` here is load-bearing: it is the mechanism that allows DSA hierarchical
/// compression to be added later without breaking downstream code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CompressionScheme {
    /// Standard DeepSeek-V2/V3/V4 / Kimi-K2 MLA: a latent `c_kv` plus a decoupled `k_rope` key.
    /// `c_kv` is position-independent; `k_rope` is not. Fully implemented in Sprint 0.
    MlaLatent {
        /// `d_c` — latent dimension. 512 for DeepSeek-V3/V4.
        latent_dim: u32,
        /// `d_r` — decoupled RoPE key dimension. 64 for DeepSeek-V3/V4.
        rope_key_dim: u32,
    },

    /// **Deprecated** (Sprint 5): pre-V4-paper placeholder using nomenclature
    /// `c4a`/`c128a` that turned out not to match DeepSeek-V4's actual architecture. The
    /// real V4 layout uses three sibling variants — [`Self::V4Csa`], [`Self::V4Hca`], and
    /// [`Self::V4Swa`] — applied per-layer (interleaved). Construction is still permitted
    /// for legacy TOML parsing; `MlaBlockConfig::new` rejects it with a migration message.
    /// See `docs/src/adr/0020-v4-hybrid-attention.md`.
    #[deprecated(
        since = "0.6.0",
        note = "DeepSeek-V4's real architecture is CSA + HCA + SWA interleaved per layer. \
                Use CompressionScheme::V4Csa / V4Hca / V4Swa instead. See ADR-0020."
    )]
    DsaHierarchical {
        /// Coarse-grain compressed dimension (legacy `c4a` tier guess).
        coarse_dim: u32,
        /// Fine-grain compressed dimension (legacy `c128a` tier guess).
        fine_dim: u32,
        /// Local sliding-window size in tokens.
        swa_window: u32,
    },

    /// MHA fallback for non-MLA models. Stores full K and V tensors. Used when `latent_dim=0`
    /// is set in TOML configs. The block size constraint is relaxed in this mode by upstream
    /// callers (the value lives in [`MlaBlockConfig::block_size_tokens`] anyway).
    MhaFull {
        /// Number of attention heads.
        num_heads: u32,
        /// Per-head dimension.
        head_dim: u32,
    },

    // ────────── DeepSeek-V4 hybrid attention (Sprint 5 / ADR-0020) ──────────
    //
    // V4 alternates CSA / HCA layers, with SWA always running as a supplementary branch.
    // Block sizing therefore needs PER-LAYER schemes (see ADR-0021 and
    // `MlaBlockConfig::schemes_per_layer`). The variants below describe one layer each;
    // typical V4 deployments carry a `Vec<CompressionScheme>` of length `num_layers`.
    /// **V4 CSA layer** — Compressed Sparse Attention.
    ///
    /// Compresses `k1` consecutive tokens (overlapping windows) into one main KV entry +
    /// one Lightning-Indexer entry. The DSA top-k selector then picks `top_k` of those
    /// compressed entries for the core attention.
    ///
    /// Per compressed main KV entry (shared-MQA, one across all query heads):
    /// `(head_dim - rope_dim)` content elements stored in **FP8** + `rope_dim` RoPE
    /// elements stored in **BF16**.
    ///
    /// Per compressed indexer entry: `indexer_head_dim` elements stored in **FP4**.
    V4Csa {
        /// `k1` — token compression ratio (4 in V4-Flash and V4-Pro).
        k1: u32,
        /// Per-head dimension `d_h` for both attention and KV entries (512 in V4-* models).
        head_dim: u32,
        /// Total number of query heads at this layer (64 Flash, 128 Pro). Stored for the
        /// kernel; the main KV entry itself is shared-MQA.
        num_heads: u32,
        /// Number of trailing dimensions reserved for RoPE (BF16). 64 in both V4 models.
        rope_dim: u32,
        /// Indexer head dimension `d_I` (128 in both V4 models).
        indexer_head_dim: u32,
        /// Number of indexer query heads `n_I` (64 in both V4 models).
        num_indexer_heads: u32,
        /// `top_k` of compressed entries selected by the sparse attention path (512 Flash,
        /// 1024 Pro). Informational — does not affect block-byte computation.
        top_k: u32,
    },

    /// **V4 HCA layer** — Heavily Compressed Attention.
    ///
    /// Compresses `k2` consecutive tokens (non-overlapping) into one shared-MQA entry. No
    /// sparse selection — every query attends to every compressed HCA entry. Per entry
    /// storage is identical to V4Csa's main entry: BF16 RoPE region + FP8 content region.
    V4Hca {
        /// `k2` — token compression ratio (128 in V4-Flash and V4-Pro).
        k2: u32,
        /// Per-head dimension `d_h` (512).
        head_dim: u32,
        /// Total number of query heads at this layer.
        num_heads: u32,
        /// Number of trailing dimensions reserved for RoPE (BF16). 64.
        rope_dim: u32,
    },

    /// **V4 SWA layer** — Sliding Window Attention branch.
    ///
    /// Uncompressed; stores raw KV for the most recent `window` tokens of each request.
    /// Lives in the State Cache (ADR-0023), **not** in the paged block pool. Sprint 5
    /// supports this variant in `MlaBlockConfig` so per-layer maps can describe pure-SWA
    /// layers (V4-Flash's first two layers) and the supplementary SWA branch attached to
    /// CSA / HCA layers.
    V4Swa {
        /// Window size in original tokens (128 in V4-Flash and V4-Pro).
        window: u32,
        /// Per-head dimension (512).
        head_dim: u32,
        /// Number of query heads.
        num_heads: u32,
        /// RoPE region size (BF16); same as CSA / HCA (64).
        rope_dim: u32,
    },
}

#[allow(deprecated)] // legacy DsaHierarchical variant kept for migration
impl CompressionScheme {
    /// Bytes of `c_kv` (or equivalent primary KV region) stored per token across all layers.
    ///
    /// `dtype_bytes` applies to MLA/MHA only — V4 variants encode their own mixed precision
    /// internally and **ignore the parameter**. Callers that want a precise V4 sizing should
    /// prefer [`Self::bytes_per_token_per_layer`] which is self-describing.
    pub fn primary_bytes_per_token(self, num_layers: u32, dtype_bytes: u32) -> u64 {
        match self {
            Self::MlaLatent { latent_dim, .. } => {
                u64::from(latent_dim) * u64::from(num_layers) * u64::from(dtype_bytes)
            }
            Self::DsaHierarchical { .. } => {
                todo!(
                    "DsaHierarchical is deprecated — migrate to V4Csa / V4Hca / V4Swa. \
                     See ADR-0020."
                )
            }
            Self::MhaFull { num_heads, head_dim } => {
                // Both K and V are stored fully.
                2 * u64::from(num_heads) * u64::from(head_dim) * u64::from(num_layers)
                    * u64::from(dtype_bytes)
            }
            // V4 variants: dtype_bytes ignored; precision is mixed and encoded in
            // `bytes_per_token_per_layer`. Multiply per-layer cost by num_layers.
            Self::V4Csa { .. } | Self::V4Hca { .. } | Self::V4Swa { .. } => {
                self.bytes_per_token_per_layer() * u64::from(num_layers)
            }
        }
    }

    /// Bytes of the secondary, position-dependent region per token (e.g. `k_rope` for MLA).
    /// Returns zero for compression schemes that do not split storage in two — for V4
    /// schemes the RoPE region is already accounted for inside
    /// [`Self::bytes_per_token_per_layer`].
    pub fn rope_bytes_per_token(self, num_layers: u32) -> u64 {
        match self {
            Self::MlaLatent { rope_key_dim, .. } => {
                // k_rope is always BF16 (2 bytes); see ADR-0007.
                u64::from(rope_key_dim) * u64::from(num_layers) * 2
            }
            Self::DsaHierarchical { .. } => todo!(
                "DsaHierarchical is deprecated — migrate to V4Csa / V4Hca / V4Swa. \
                 See ADR-0020."
            ),
            Self::MhaFull { .. } => 0,
            // V4: the RoPE BF16 region is bundled into `bytes_per_token_per_layer` because
            // V4 entries are MQA (shared across heads) and the RoPE region is part of the
            // entry, not a parallel pool.
            Self::V4Csa { .. } | Self::V4Hca { .. } | Self::V4Swa { .. } => 0,
        }
    }

    /// **Sprint 5 / V4** — Self-describing per-layer byte cost per ORIGINAL token, honouring
    /// the variant's internal precision split. For V4 the precision is fixed by the paper:
    /// RoPE region in BF16, content region in FP8, indexer region (CSA only) in FP4.
    ///
    /// Returns 0 for non-V4 variants — callers should use [`Self::primary_bytes_per_token`]
    /// + [`Self::rope_bytes_per_token`] for MLA/MHA.
    pub fn bytes_per_token_per_layer(self) -> u64 {
        match self {
            Self::V4Csa {
                k1,
                head_dim,
                rope_dim,
                indexer_head_dim,
                ..
            } => {
                // Per CSA compressed entry, MQA (shared across heads):
                //   rope_dim       elements in BF16   (2 bytes each)
                //   head_dim - rope_dim elements in FP8 (1 byte each)
                //   indexer_head_dim elements in FP4   (0.5 byte each; packed)
                let bf16 = CkvDtype::Bf16.bytes_for_elements(u64::from(rope_dim));
                let fp8 = CkvDtype::Fp8E4m3.bytes_for_elements(
                    u64::from(head_dim).saturating_sub(u64::from(rope_dim)),
                );
                let fp4 = CkvDtype::Fp4E2m1.bytes_for_elements(u64::from(indexer_head_dim));
                let bytes_per_entry = bf16 + fp8 + fp4;
                // CSA produces 1 compressed entry per k1 original tokens.
                // (The 2k1 overlap is an attention-time read pattern, not a storage factor.)
                bytes_per_entry / u64::from(k1.max(1))
            }
            Self::V4Hca {
                k2,
                head_dim,
                rope_dim,
                ..
            } => {
                let bf16 = CkvDtype::Bf16.bytes_for_elements(u64::from(rope_dim));
                let fp8 = CkvDtype::Fp8E4m3.bytes_for_elements(
                    u64::from(head_dim).saturating_sub(u64::from(rope_dim)),
                );
                (bf16 + fp8) / u64::from(k2.max(1))
            }
            Self::V4Swa { head_dim, rope_dim, .. } => {
                // Uncompressed — each original token contributes one full entry. SWA caps
                // total stored tokens at `window` per request via the State Cache; this
                // function gives the per-token cost regardless of window.
                let bf16 = CkvDtype::Bf16.bytes_for_elements(u64::from(rope_dim));
                let fp8 = CkvDtype::Fp8E4m3.bytes_for_elements(
                    u64::from(head_dim).saturating_sub(u64::from(rope_dim)),
                );
                bf16 + fp8
            }
            _ => 0,
        }
    }

    /// Convenience: MLA latent dim, if this is an MLA variant. Returns `None` otherwise
    /// (including for V4 variants, which use a different storage model).
    pub fn mla_latent_dim(self) -> Option<u32> {
        match self {
            Self::MlaLatent { latent_dim, .. } => Some(latent_dim),
            _ => None,
        }
    }

    /// Convenience: MLA RoPE-key dim, if this is an MLA variant. Returns `None` otherwise.
    pub fn mla_rope_key_dim(self) -> Option<u32> {
        match self {
            Self::MlaLatent { rope_key_dim, .. } => Some(rope_key_dim),
            _ => None,
        }
    }

    /// Whether the variant belongs to the V4 hybrid family.
    pub fn is_v4(self) -> bool {
        matches!(self, Self::V4Csa { .. } | Self::V4Hca { .. } | Self::V4Swa { .. })
    }
}

/// Validated block manager configuration. Construct via [`MlaBlockConfig::new`] for
/// homogeneous-scheme configs (V3-style MLA, MHA fallback) or
/// [`MlaBlockConfig::with_per_layer_schemes`] for V4 hybrid layouts (Sprint 5 / ADR-0021).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlaBlockConfig {
    /// Primary compression scheme. For homogeneous configs (`schemes_per_layer.is_none()`)
    /// this applies to every layer. For V4 hybrid configs it is the **first layer's**
    /// scheme and acts as a back-compat hint; per-layer resolution goes through
    /// [`Self::scheme_for_layer`].
    pub scheme: CompressionScheme,
    /// `L` — number of transformer layers.
    pub num_layers: u32,
    /// Block size in tokens. Sprint 0 enforces this equals [`REQUIRED_BLOCK_SIZE_TOKENS`] (64)
    /// for MLA schemes to match FlashMLA's native paged block size. V4 hybrid configs use
    /// `lcm(k1, k2) = 128` (paper §3.5.1).
    pub block_size_tokens: u32,
    /// Storage dtype for the primary region (`c_kv` in MLA, K/V in MHA fallback). For V4
    /// schemes this is informational — the variant carries its own mixed precision.
    pub ckv_dtype: CkvDtype,
    /// CUDA device ordinal. Ignored by the CPU mock backend.
    pub device: i32,
    /// **Sprint 5** — per-layer schemes for V4 hybrid configs. `None` means homogeneous
    /// (use [`Self::scheme`] for every layer). When present, `Vec` length must equal
    /// `num_layers` and is enforced by [`Self::with_per_layer_schemes`].
    ///
    /// `Arc<Vec<...>>` so multiple block managers can share the layout description without
    /// cloning the vector on every `scheme_for_layer` lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schemes_per_layer: Option<std::sync::Arc<Vec<CompressionScheme>>>,
}

impl MlaBlockConfig {
    /// Construct after validating every invariant. Returns
    /// [`TesseraError::InvalidConfig`] with a precise message otherwise.
    pub fn new(
        scheme: CompressionScheme,
        num_layers: u32,
        block_size_tokens: u32,
        ckv_dtype: CkvDtype,
        device: i32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TesseraError::InvalidConfig(
                "num_layers must be > 0".into(),
            ));
        }
        if block_size_tokens == 0 {
            return Err(TesseraError::InvalidConfig(
                "block_size_tokens must be > 0".into(),
            ));
        }
        #[allow(deprecated)]
        match scheme {
            CompressionScheme::MlaLatent {
                latent_dim,
                rope_key_dim,
            } => {
                if latent_dim == 0 || rope_key_dim == 0 {
                    return Err(TesseraError::InvalidConfig(
                        "MLA latent_dim and rope_key_dim must both be > 0".into(),
                    ));
                }
                if block_size_tokens != REQUIRED_BLOCK_SIZE_TOKENS {
                    return Err(TesseraError::InvalidConfig(format!(
                        "block_size_tokens must equal {REQUIRED_BLOCK_SIZE_TOKENS} for MLA \
                         (got {block_size_tokens}) — FlashMLA's paged block size is fixed."
                    )));
                }
            }
            CompressionScheme::MhaFull { num_heads, head_dim } => {
                if num_heads == 0 || head_dim == 0 {
                    return Err(TesseraError::InvalidConfig(
                        "MHA num_heads and head_dim must both be > 0".into(),
                    ));
                }
                // MHA fallback does not constrain block_size_tokens here.
            }
            CompressionScheme::DsaHierarchical { .. } => {
                return Err(TesseraError::InvalidConfig(
                    "DsaHierarchical is deprecated (Sprint 5). Migrate to CompressionScheme::\
                     V4Csa / V4Hca / V4Swa per-layer. See ADR-0020.".into(),
                ));
            }
            CompressionScheme::V4Csa { k1, head_dim, rope_dim, .. } => {
                if k1 == 0 || head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
                    return Err(TesseraError::InvalidConfig(format!(
                        "V4Csa: k1, head_dim, rope_dim must be > 0 with rope_dim ≤ head_dim \
                         (got k1={k1} head_dim={head_dim} rope_dim={rope_dim})"
                    )));
                }
                if block_size_tokens % k1 != 0 {
                    return Err(TesseraError::InvalidConfig(format!(
                        "V4Csa: block_size_tokens ({block_size_tokens}) must be a multiple of \
                         k1 ({k1}) so compressed entries align to block boundaries."
                    )));
                }
            }
            CompressionScheme::V4Hca { k2, head_dim, rope_dim, .. } => {
                if k2 == 0 || head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
                    return Err(TesseraError::InvalidConfig(format!(
                        "V4Hca: k2, head_dim, rope_dim must be > 0 with rope_dim ≤ head_dim \
                         (got k2={k2} head_dim={head_dim} rope_dim={rope_dim})"
                    )));
                }
                if block_size_tokens % k2 != 0 {
                    return Err(TesseraError::InvalidConfig(format!(
                        "V4Hca: block_size_tokens ({block_size_tokens}) must be a multiple of \
                         k2 ({k2})."
                    )));
                }
            }
            CompressionScheme::V4Swa { window, head_dim, rope_dim, .. } => {
                if window == 0 || head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
                    return Err(TesseraError::InvalidConfig(format!(
                        "V4Swa: window, head_dim, rope_dim must be > 0 with rope_dim ≤ head_dim \
                         (got window={window} head_dim={head_dim} rope_dim={rope_dim})"
                    )));
                }
                // V4Swa lives in the State Cache (ADR-0023); block_size_tokens is informational
                // for SWA-only configs.
            }
        }
        Ok(Self {
            scheme,
            num_layers,
            block_size_tokens,
            ckv_dtype,
            device,
            schemes_per_layer: None,
        })
    }

    /// **Sprint 5** — Construct with explicit per-layer schemes (V4 hybrid interleaved
    /// CSA / HCA / SWA). The `schemes` vector length must equal `num_layers`; each entry
    /// describes one transformer layer in order. The scalar `scheme` field stays as a
    /// "primary" hint (first layer's scheme) for backward compatibility with callers that
    /// haven't been taught the per-layer API yet.
    ///
    /// Validates each per-layer scheme using the same rules as [`Self::new`] would for a
    /// homogeneous config.
    pub fn with_per_layer_schemes(
        schemes: Vec<CompressionScheme>,
        block_size_tokens: u32,
        ckv_dtype: CkvDtype,
        device: i32,
    ) -> Result<Self> {
        if schemes.is_empty() {
            return Err(TesseraError::InvalidConfig(
                "with_per_layer_schemes: schemes vector must be non-empty".into(),
            ));
        }
        let num_layers = u32::try_from(schemes.len()).map_err(|_| {
            TesseraError::InvalidConfig("num_layers does not fit in u32".into())
        })?;
        // Validate each layer individually using the homogeneous path. We tolerate that
        // `new()` will fail on MLA block_size_tokens != 64; per-layer V4 configs are
        // expected to use V4 variants throughout, so this is the strict path.
        for (idx, scheme) in schemes.iter().enumerate() {
            let probe = Self::new(*scheme, 1, block_size_tokens, ckv_dtype, device);
            if let Err(e) = probe {
                return Err(TesseraError::InvalidConfig(format!(
                    "with_per_layer_schemes: layer {idx} rejected: {e}"
                )));
            }
        }
        Ok(Self {
            scheme: schemes[0],
            num_layers,
            block_size_tokens,
            ckv_dtype,
            device,
            schemes_per_layer: Some(std::sync::Arc::new(schemes)),
        })
    }

    /// Whether this config uses per-layer schemes.
    pub fn has_per_layer_schemes(&self) -> bool {
        self.schemes_per_layer.is_some()
    }

    /// Resolve the scheme for a specific layer index. For homogeneous configs returns
    /// `self.scheme`; for V4 hybrid configs returns the entry at `layer_idx`.
    pub fn scheme_for_layer(&self, layer_idx: u32) -> CompressionScheme {
        match &self.schemes_per_layer {
            Some(v) => v.get(layer_idx as usize).copied().unwrap_or(self.scheme),
            None => self.scheme,
        }
    }

    /// Bytes used by the primary region of one block (e.g. `c_kv` for MLA).
    ///
    /// For V4 hybrid configs with per-layer schemes this sums each layer's contribution,
    /// honouring the variant-specific precision via `bytes_per_token_per_layer`.
    pub fn primary_block_bytes(&self) -> u64 {
        match &self.schemes_per_layer {
            Some(per_layer) => {
                let mut total = 0u64;
                for scheme in per_layer.iter() {
                    if scheme.is_v4() {
                        total += scheme.bytes_per_token_per_layer()
                            * u64::from(self.block_size_tokens);
                    } else {
                        // Mixed V4 + non-V4 per-layer maps fall back to the homogeneous
                        // accounting for non-V4 layers.
                        total += scheme.primary_bytes_per_token(1, self.ckv_dtype.bytes())
                            * u64::from(self.block_size_tokens);
                    }
                }
                total
            }
            None => {
                self.scheme
                    .primary_bytes_per_token(self.num_layers, self.ckv_dtype.bytes())
                    * u64::from(self.block_size_tokens)
            }
        }
    }

    /// Bytes used by the secondary, position-dependent region of one block (`k_rope` in MLA).
    /// V4 schemes return 0 because the RoPE region is bundled into the primary entry.
    pub fn rope_block_bytes(&self) -> u64 {
        match &self.schemes_per_layer {
            Some(per_layer) => {
                let mut total = 0u64;
                for scheme in per_layer.iter() {
                    total += scheme.rope_bytes_per_token(1)
                        * u64::from(self.block_size_tokens);
                }
                total
            }
            None => {
                self.scheme.rope_bytes_per_token(self.num_layers)
                    * u64::from(self.block_size_tokens)
            }
        }
    }

    /// Bytes of per-layer FP8 scale factors stored per block (0 unless FP8 is active).
    pub fn fp8_scale_block_bytes(&self) -> u64 {
        if self.ckv_dtype.requires_per_layer_scales() {
            // One f32 per layer.
            u64::from(self.num_layers) * 4
        } else {
            0
        }
    }

    /// Total bytes consumed by a single block, including a 64-byte aligned header.
    pub fn total_block_bytes(&self) -> u64 {
        const HEADER_BYTES: u64 = 64;
        HEADER_BYTES
            + self.primary_block_bytes()
            + self.rope_block_bytes()
            + self.fp8_scale_block_bytes()
    }

    /// Compression ratio relative to a hypothetical MHA-BF16 baseline with the same number of
    /// tokens per block and the same number of layers. Useful as a sanity-check metric.
    ///
    /// For V4 schemes the baseline is `(num_heads, 128)` BF16 GQA — the paper's stated
    /// reference (§2.3.4 quotes ~2% vs GQA8 BF16 at 1M).
    #[allow(deprecated)]
    pub fn compression_ratio_vs_mha_bf16(&self) -> f64 {
        let (num_heads, head_dim) = match self.scheme {
            CompressionScheme::MhaFull { num_heads, head_dim } => (num_heads, head_dim),
            // For MLA we cannot know head/head_dim from the scheme alone. The portfolio metric
            // uses DeepSeek-V3 reference (128 heads × 128 dim) — the same constants the
            // playbook uses for its 56.9× headline number.
            CompressionScheme::MlaLatent { .. } => (128, 128),
            // V4: use the variant's own num_heads × 128 (V3 reference head_dim).
            CompressionScheme::V4Csa { num_heads, .. }
            | CompressionScheme::V4Hca { num_heads, .. }
            | CompressionScheme::V4Swa { num_heads, .. } => (num_heads, 128),
            CompressionScheme::DsaHierarchical { .. } => unreachable!(
                "DsaHierarchical configs are rejected by MlaBlockConfig::new"
            ),
        };
        let mha_bytes = 2u64
            * u64::from(num_heads)
            * u64::from(head_dim)
            * u64::from(self.num_layers)
            * u64::from(self.block_size_tokens)
            * 2; // MHA baseline is always BF16
        #[allow(clippy::cast_precision_loss)]
        let ratio = mha_bytes as f64 / self.total_block_bytes() as f64;
        ratio
    }

    /// **Sprint 5 / V4** — Compute `lcm(k1, k2)` over per-layer V4 schemes. Returns 1 when
    /// no V4 layer is present. Used to validate that `block_size_tokens` aligns with the
    /// compression boundaries (§3.5.1: "each cache block covers `lcm(k1, k2)` original
    /// tokens").
    pub fn v4_block_size_lcm(&self) -> u32 {
        const fn gcd(mut a: u32, mut b: u32) -> u32 {
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            a
        }
        const fn lcm(a: u32, b: u32) -> u32 {
            if a == 0 || b == 0 {
                return a.max(b);
            }
            (a / gcd(a, b)).saturating_mul(b)
        }

        let schemes: Vec<CompressionScheme> = match &self.schemes_per_layer {
            Some(v) => v.iter().copied().collect(),
            None => vec![self.scheme],
        };
        let mut acc = 1u32;
        for s in schemes {
            match s {
                CompressionScheme::V4Csa { k1, .. } => acc = lcm(acc, k1),
                CompressionScheme::V4Hca { k2, .. } => acc = lcm(acc, k2),
                _ => {}
            }
        }
        acc.max(1)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    fn ds_v3_scheme() -> CompressionScheme {
        CompressionScheme::MlaLatent {
            latent_dim: 512,
            rope_key_dim: 64,
        }
    }

    fn v4_pro_csa() -> CompressionScheme {
        CompressionScheme::V4Csa {
            k1: 4,
            head_dim: 512,
            num_heads: 128,
            rope_dim: 64,
            indexer_head_dim: 128,
            num_indexer_heads: 64,
            top_k: 1024,
        }
    }

    fn v4_pro_hca() -> CompressionScheme {
        CompressionScheme::V4Hca { k2: 128, head_dim: 512, num_heads: 128, rope_dim: 64 }
    }

    fn v4_pro_swa() -> CompressionScheme {
        CompressionScheme::V4Swa { window: 128, head_dim: 512, num_heads: 128, rope_dim: 64 }
    }

    #[test]
    fn deepseek_v3_compression_ratio_is_about_57x() {
        let cfg =
            MlaBlockConfig::new(ds_v3_scheme(), 61, 64, CkvDtype::Bf16, 0).unwrap();
        let ratio = cfg.compression_ratio_vs_mha_bf16();
        assert!(
            (55.0..58.0).contains(&ratio),
            "compression ratio {ratio} not in (55, 58) — playbook claims ~56.9"
        );
    }

    #[test]
    fn invalid_block_size_rejected_for_mla() {
        let err = MlaBlockConfig::new(ds_v3_scheme(), 61, 16, CkvDtype::Bf16, 0)
            .expect_err("block_size != 64 must fail under MLA");
        assert!(matches!(err, TesseraError::InvalidConfig(_)));
    }

    #[test]
    fn dsa_variant_is_rejected_by_new() {
        let scheme = CompressionScheme::DsaHierarchical {
            coarse_dim: 128,
            fine_dim: 32,
            swa_window: 128,
        };
        let err = MlaBlockConfig::new(scheme, 61, 64, CkvDtype::Bf16, 0)
            .expect_err("DsaHierarchical is deprecated; must be rejected");
        assert!(matches!(err, TesseraError::InvalidConfig(_)));
    }

    #[test]
    #[should_panic(expected = "DsaHierarchical is deprecated")]
    fn dsa_primary_bytes_panics_with_diagnostic_message() {
        let scheme = CompressionScheme::DsaHierarchical {
            coarse_dim: 128,
            fine_dim: 32,
            swa_window: 128,
        };
        let _ = scheme.primary_bytes_per_token(61, 2);
    }

    #[test]
    fn fp8_halves_primary_bytes() {
        let bf16_cfg =
            MlaBlockConfig::new(ds_v3_scheme(), 61, 64, CkvDtype::Bf16, 0).unwrap();
        let fp8_cfg =
            MlaBlockConfig::new(ds_v3_scheme(), 61, 64, CkvDtype::Fp8E4m3, 0).unwrap();
        assert_eq!(bf16_cfg.primary_block_bytes(), 2 * fp8_cfg.primary_block_bytes());
        assert_eq!(fp8_cfg.fp8_scale_block_bytes(), 61 * 4);
        assert_eq!(bf16_cfg.fp8_scale_block_bytes(), 0);
    }

    // ────────── Sprint 5 — V4 compliance tests ──────────

    #[test]
    fn v4_csa_per_token_bytes_match_paper() {
        // V4-Pro CSA: k1=4, head_dim=512, rope_dim=64, indexer=128
        // BF16(64) = 128 B; FP8(448) = 448 B; FP4(128) = 64 B; total entry = 640 B
        // Per token: 640 / 4 = 160 B/layer.
        assert_eq!(v4_pro_csa().bytes_per_token_per_layer(), 160);
    }

    #[test]
    fn v4_hca_per_token_bytes_match_paper() {
        // V4-Pro HCA: k2=128, head_dim=512, rope_dim=64
        // BF16(64) = 128 B; FP8(448) = 448 B; total entry = 576 B
        // Per token: 576 / 128 = 4 B/layer (with rounding).
        assert_eq!(v4_pro_hca().bytes_per_token_per_layer(), 4);
    }

    #[test]
    fn v4_swa_per_token_bytes_match_paper() {
        // Uncompressed per token: 128 + 448 = 576 B/layer (mirrors HCA entry).
        assert_eq!(v4_pro_swa().bytes_per_token_per_layer(), 576);
    }

    #[test]
    fn v4_csa_passes_validation_with_block_size_multiple_of_k1() {
        let cfg = MlaBlockConfig::new(v4_pro_csa(), 61, 128, CkvDtype::MixedBf16Fp8Fp4, 0).unwrap();
        assert_eq!(cfg.scheme.is_v4(), true);
        assert_eq!(cfg.block_size_tokens, 128);
    }

    #[test]
    fn v4_csa_rejects_block_size_not_multiple_of_k1() {
        // k1=4; block_size=30 → 30 % 4 != 0
        let err =
            MlaBlockConfig::new(v4_pro_csa(), 61, 30, CkvDtype::MixedBf16Fp8Fp4, 0).unwrap_err();
        assert!(matches!(err, TesseraError::InvalidConfig(_)));
    }

    #[test]
    fn v4_hca_rejects_block_size_not_multiple_of_k2() {
        let err =
            MlaBlockConfig::new(v4_pro_hca(), 61, 64, CkvDtype::MixedBf16Fp8Fp4, 0).unwrap_err();
        assert!(matches!(err, TesseraError::InvalidConfig(_)));
    }

    #[test]
    fn v4_pro_per_layer_lcm_is_128() {
        // V4-Pro: 2 leading HCA + interleaved CSA/HCA for the remaining 59.
        // lcm(k1=4, k2=128) = 128.
        let mut layers = Vec::with_capacity(61);
        layers.push(v4_pro_hca());
        layers.push(v4_pro_hca());
        for i in 0..59 {
            layers.push(if i % 2 == 0 { v4_pro_csa() } else { v4_pro_hca() });
        }
        let cfg = MlaBlockConfig::with_per_layer_schemes(
            layers, 128, CkvDtype::MixedBf16Fp8Fp4, 0,
        )
        .unwrap();
        assert_eq!(cfg.num_layers, 61);
        assert_eq!(cfg.v4_block_size_lcm(), 128);
        assert!(cfg.has_per_layer_schemes());
    }

    #[test]
    fn ckv_dtype_fp4_packs_two_elements_per_byte() {
        assert_eq!(CkvDtype::Fp4E2m1.bytes_for_elements(2), 1);
        assert_eq!(CkvDtype::Fp4E2m1.bytes_for_elements(3), 2);
        assert_eq!(CkvDtype::Fp4E2m1.bytes_for_elements(128), 64);
        assert_eq!(CkvDtype::Fp4E2m1.bits(), 4);
    }

    #[test]
    fn mixed_dtype_signals_scheme_level_accounting() {
        assert_eq!(CkvDtype::MixedBf16Fp8Fp4.bytes(), 0);
        assert_eq!(CkvDtype::MixedBf16Fp8Fp4.bytes_for_elements(1000), 0);
        assert!(CkvDtype::MixedBf16Fp8Fp4.requires_per_layer_scales());
    }
}
