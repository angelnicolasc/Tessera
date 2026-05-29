"""Typed Tessera configuration. Parses ``tessera.toml`` into a Pydantic model with strict
validation, then projects onto the native ``MlaBlockConfig`` when the block manager needs it.

The Python config layer carries fields that the Rust core does not need (Prometheus port,
HNSW tuning knobs, kernel backend selection) — keeping them out of the Rust struct means the
core crate has no opinion on observability or kernel dispatch.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REQUIRED_BLOCK_SIZE_TOKENS = 64


class ModelConfig(BaseModel):
    """Model architecture constants. The MLA-specific dims drive block sizing in the core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    latent_dim: Annotated[int, Field(ge=0, description="d_c; 0 selects MHA fallback")]
    rope_key_dim: Annotated[int, Field(ge=0, description="d_r; ignored under MHA fallback")]
    num_layers: Annotated[int, Field(gt=0)]
    num_heads: Annotated[int, Field(gt=0)]
    head_dim: Annotated[int, Field(gt=0)]


class BlockConfig(BaseModel):
    """Block-layout configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_size_tokens: Annotated[int, Field(gt=0)]
    ckv_dtype: Literal["bf16", "fp8_e4m3", "fp4_e2m1", "mixed_bf16_fp8_fp4"]
    fp8_scales_path: str | None = None
    """Path to a JSON file with per-layer FP8 scale factors ``{"0": 0.00123, ...}``.

    Required when ``ckv_dtype == "fp8_e4m3"`` for production use (ADR-0007). If ``None``
    the scale defaults to 1.0 per layer — correct only for testing, NOT for production.
    """


class V4Config(BaseModel):
    """**Sprint 5 / V4** — DeepSeek-V4 hybrid attention specification.

    Required when ``ckv_dtype == "mixed_bf16_fp8_fp4"`` and the model uses V4 hybrid
    attention. See ADR-0020 / ADR-0021.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k1: Annotated[int, Field(gt=0, description="CSA token compression ratio (paper: 4)")]
    k2: Annotated[int, Field(gt=0, description="HCA token compression ratio (paper: 128)")]
    head_dim: Annotated[int, Field(gt=0, description="per-head dim (paper: 512)")]
    rope_dim: Annotated[int, Field(gt=0, description="trailing RoPE BF16 dims (paper: 64)")]
    indexer_head_dim: Annotated[
        int, Field(gt=0, description="Lightning Indexer head dim (paper: 128)")
    ]
    num_indexer_heads: Annotated[
        int, Field(gt=0, description="Lightning Indexer query heads (paper: 64)")
    ]
    top_k: Annotated[int, Field(gt=0, description="sparse top-k (paper: 512 Flash / 1024 Pro)")]
    swa_window: Annotated[int, Field(gt=0, description="SWA window (paper: 128)")]
    """Layer pattern, length must equal num_layers. Each entry is one of
    ``{"csa", "hca", "swa"}`` describing the layer's attention scheme. The first 2 layers
    in V4-Flash are pure SWA; V4-Pro's first 2 are HCA; subsequent layers interleave
    CSA / HCA."""
    layer_pattern: list[Literal["csa", "hca", "swa"]]


class DiskCacheConfig(BaseModel):
    """**Sprint 5 / V4** — On-disk KV cache configuration (ADR-0024)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    root: str | None = None
    """Filesystem directory for region files. Created if missing. Required when enabled."""
    swa_strategy: Literal["full", "periodic", "zero"] = "periodic"
    swa_checkpoint_interval_tokens: Annotated[int, Field(gt=0)] = 4096


class SegmentIndexConfig(BaseModel):
    """HNSW segment-index tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hnsw_m: Annotated[int, Field(ge=4, le=128)] = 32
    hnsw_ef_construction: Annotated[int, Field(ge=16, le=1024)] = 200
    hnsw_ef_search: Annotated[int, Field(ge=1, le=1024)] = 64
    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.97
    hnsw_latency_budget_us: Annotated[int, Field(ge=10, le=10_000)] = 500


class KernelConfig(BaseModel):
    """Kernel dispatch configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["auto", "flash_mla", "flash_infer", "flash_attn4", "triton"] = "auto"
    softmax_scale: float | None = None


class RuntimeConfig(BaseModel):
    """Runtime resource configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Annotated[int, Field(ge=0)] = 0
    gpu_memory_bytes: Annotated[int, Field(gt=0)]


class ObservabilityConfig(BaseModel):
    """Metrics and tracing endpoints. All fields optional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prometheus_port: Annotated[int, Field(ge=1, le=65_535)] | None = 9090
    tracing_endpoint: str = ""


class TesseraConfig(BaseModel):
    """Top-level Tessera configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelConfig
    block: BlockConfig
    segment_index: SegmentIndexConfig = SegmentIndexConfig()
    kernel: KernelConfig = KernelConfig()
    runtime: RuntimeConfig
    observability: ObservabilityConfig = ObservabilityConfig()
    v4: V4Config | None = None
    """Sprint 5 / V4. Present when the model uses V4 hybrid attention (CSA/HCA/SWA).
    When set, ``model.latent_dim`` is ignored and per-layer schemes are constructed from
    ``v4.layer_pattern`` instead."""
    disk_cache: DiskCacheConfig = DiskCacheConfig()
    """Sprint 5 / V4. On-disk KV cache configuration. Disabled by default."""

    # ------------------------ validation -----------------------------------

    @field_validator("block")
    @classmethod
    def _enforce_block_size(cls, v: BlockConfig) -> BlockConfig:
        # MLA configs must have block_size == 64. We can only fully enforce after we know
        # whether the model is MLA or MHA; defer the cross-field check to model_validator.
        return v

    @model_validator(mode="after")
    def _enforce_block_size_invariants(self) -> TesseraConfig:
        is_mla = self.model.latent_dim > 0 and self.v4 is None
        is_v4 = self.v4 is not None
        if is_mla and self.block.block_size_tokens != REQUIRED_BLOCK_SIZE_TOKENS:
            msg = (
                f"MLA configs require block_size_tokens == {REQUIRED_BLOCK_SIZE_TOKENS} "
                f"(got {self.block.block_size_tokens}). FlashMLA's paged block size is fixed."
            )
            raise ValueError(msg)
        if not is_mla and not is_v4 and self.model.rope_key_dim != 0:
            msg = "MHA fallback (latent_dim=0, no v4) must also have rope_key_dim=0"
            raise ValueError(msg)
        if is_v4:
            v4 = self.v4
            assert v4 is not None
            if len(v4.layer_pattern) != self.model.num_layers:
                msg = (
                    f"v4.layer_pattern length ({len(v4.layer_pattern)}) must equal "
                    f"model.num_layers ({self.model.num_layers})"
                )
                raise ValueError(msg)
            # Block size must be a multiple of lcm(k1, k2) for the patterns in use.
            from math import gcd

            def lcm(a: int, b: int) -> int:
                return (a * b) // gcd(a, b) if a and b else max(a, b)

            need = 1
            for layer in v4.layer_pattern:
                if layer == "csa":
                    need = lcm(need, v4.k1)
                elif layer == "hca":
                    need = lcm(need, v4.k2)
            if self.block.block_size_tokens % need != 0:
                msg = (
                    f"V4 hybrid requires block_size_tokens % lcm(k1, k2) == 0; "
                    f"need multiple of {need}, got {self.block.block_size_tokens}"
                )
                raise ValueError(msg)
            if self.block.ckv_dtype != "mixed_bf16_fp8_fp4":
                msg = "V4 hybrid requires ckv_dtype == 'mixed_bf16_fp8_fp4' (see ADR-0022)"
                raise ValueError(msg)
        return self

    # ------------------------ constructors ---------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> TesseraConfig:
        """Load from a ``tessera.toml`` file."""
        path = Path(path)
        with path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TesseraConfig:
        """Load from an in-memory dict (mostly for tests)."""
        return cls.model_validate(data)

    # ------------------------ derived values -------------------------------

    @property
    def is_mla(self) -> bool:
        """``True`` when the config selects MLA storage (latent_dim > 0) and **not** V4."""
        return self.model.latent_dim > 0 and self.v4 is None

    @property
    def is_v4(self) -> bool:
        """``True`` when the config selects V4 hybrid attention."""
        return self.v4 is not None

    # Lazy-loaded FP8 scale cache — not part of the pydantic model (mutable).
    _fp8_scales_cache: dict[int, float] | None | bool = False  # False = not loaded yet

    @property
    def fp8_scales(self) -> dict[int, float] | None:
        """Per-layer FP8 scale factors loaded from ``block.fp8_scales_path``.

        Returns ``None`` when: (a) ``ckv_dtype`` is ``"bf16"``, (b) ``fp8_scales_path`` is
        not set, or (c) the file is empty. Loaded once and cached.
        """
        if self.block.fp8_scales_path is None or self.block.ckv_dtype != "fp8_e4m3":
            return None
        # Use object __dict__ to bypass frozen pydantic model for the mutable cache.
        cache = object.__getattribute__(self, "__dict__").get("_fp8_scales_loaded")
        if cache is not None:
            return cache  # type: ignore[return-value]
        path = Path(self.block.fp8_scales_path)
        if not path.exists():
            return None
        raw: dict[str, float] = json.loads(path.read_text())
        scales = {int(k): float(v) for k, v in raw.items()}
        object.__getattribute__(self, "__dict__")["_fp8_scales_loaded"] = scales
        return scales

    def compression_ratio_vs_mha_bf16(self) -> float:
        """Effective compression vs. MHA BF16 at this config. Sanity-check metric."""
        m = self.model
        b = self.block
        dtype_bytes = 2 if b.ckv_dtype == "bf16" else 1
        if self.is_mla:
            ckv_bytes = m.latent_dim * b.block_size_tokens * m.num_layers * dtype_bytes
            rope_bytes = m.rope_key_dim * b.block_size_tokens * m.num_layers * 2  # always BF16
            fp8_scales = m.num_layers * 4 if b.ckv_dtype == "fp8_e4m3" else 0
            total = 64 + ckv_bytes + rope_bytes + fp8_scales
        else:
            kv_bytes = (
                2 * m.num_heads * m.head_dim * b.block_size_tokens * m.num_layers * dtype_bytes
            )
            total = 64 + kv_bytes
        mha_bf16 = 2 * m.num_heads * m.head_dim * b.block_size_tokens * m.num_layers * 2
        return mha_bf16 / total

    def to_native_config(self) -> Any:
        """Project onto ``tessera._native.MlaBlockConfig``. Imports the native module lazily.

        For V4 hybrid configs returns a per-layer config (one scheme per ``layer_pattern``
        entry); for MLA / MHA returns a single-scheme config (Sprint 0 path).
        """
        from tessera import _native

        ckv_dtype_map = {
            "bf16": _native.CkvDtype.Bf16,
            "fp8_e4m3": _native.CkvDtype.Fp8E4m3,
            "fp4_e2m1": _native.CkvDtype.Fp4E2m1,
            "mixed_bf16_fp8_fp4": _native.CkvDtype.MixedBf16Fp8Fp4,
        }
        ckv_dtype = ckv_dtype_map[self.block.ckv_dtype]

        if self.is_v4:
            v4 = self.v4
            assert v4 is not None
            schemes = []
            for layer in v4.layer_pattern:
                if layer == "csa":
                    schemes.append(
                        _native.CompressionScheme.v4_csa(
                            v4.k1,
                            v4.head_dim,
                            self.model.num_heads,
                            v4.rope_dim,
                            v4.indexer_head_dim,
                            v4.num_indexer_heads,
                            v4.top_k,
                        )
                    )
                elif layer == "hca":
                    schemes.append(
                        _native.CompressionScheme.v4_hca(
                            v4.k2, v4.head_dim, self.model.num_heads, v4.rope_dim
                        )
                    )
                elif layer == "swa":
                    schemes.append(
                        _native.CompressionScheme.v4_swa(
                            v4.swa_window, v4.head_dim, self.model.num_heads, v4.rope_dim
                        )
                    )
            return _native.MlaBlockConfig.with_per_layer_schemes(
                schemes,
                self.block.block_size_tokens,
                ckv_dtype,
                self.runtime.device,
            )

        if self.is_mla:
            scheme = _native.CompressionScheme.mla_latent(
                self.model.latent_dim, self.model.rope_key_dim
            )
        else:
            scheme = _native.CompressionScheme.mha_full(self.model.num_heads, self.model.head_dim)
        return _native.MlaBlockConfig(
            scheme,
            self.model.num_layers,
            self.block.block_size_tokens,
            ckv_dtype,
            self.runtime.device,
        )
