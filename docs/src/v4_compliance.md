# DeepSeek-V4 Compliance

Sprint 5 (2026-05-28) brings Tessera's block manager into structural alignment with
DeepSeek-V4 (paper preview, May 2026). All Sprint 5 work is **CPU-validated**; the real
V4 kernels (Lightning Indexer, top-k selector, CSA/HCA cores) integrate via the kernel
dispatch layer in a future GPU session.

## What changed

| Sprint 0-4 assumption | V4 reality | Tessera response (Sprint 5) |
|---|---|---|
| Single `CompressionScheme` per config | Per-layer interleaved CSA / HCA / SWA | [ADR-0021](adr/0021-per-layer-schemes.md): `schemes_per_layer: Option<Arc<Vec<…>>>` |
| Block size fixed at 64 (FlashMLA-native) | V4 wants `lcm(k1, k2) = 128` | `MlaBlockConfig::v4_block_size_lcm()`, per-variant validation |
| One dtype per region | BF16 RoPE + FP8 content + FP4 indexer co-resident | [ADR-0022](adr/0022-mixed-precision-per-region.md): `CkvDtype::Fp4E2m1` + `MixedBf16Fp8Fp4` |
| One KV cache pool | Two-tier: KV Cache + State Cache (per-request arena) | [ADR-0023](adr/0023-state-cache.md): `state_cache::StateCache` |
| In-memory only | V4 ships three on-disk SWA strategies | [ADR-0024](adr/0024-disk-backend.md): `DiskBackend: DeviceBackend` |
| `DsaHierarchical { coarse_dim, fine_dim, swa_window }` placeholder | Three sibling variants | [ADR-0020](adr/0020-v4-hybrid-attention.md): `V4Csa` / `V4Hca` / `V4Swa`; `DsaHierarchical` deprecated |

## V4 reference configurations

Two new TOMLs ship with Sprint 5, encoding the paper's §4.2.1 dimensions verbatim:

* `models/deepseek_v4_flash.toml` — 284B total / 13B activated, 43 layers, first 2 SWA
* `models/deepseek_v4_pro.toml` — 1.6T total / 49B activated, 61 layers, first 2 HCA

Each ships with a `[v4]` block carrying `k1=4`, `k2=128`, `head_dim=512`, `rope_dim=64`,
`indexer_head_dim=128`, plus an explicit `layer_pattern` array.

## Per-token storage cost (Tessera's accounting, verified against the paper)

For V4-Pro, per-layer per-token in bytes:

```text
CSA  (k1=4):  (64·BF16 + 448·FP8 + 128·FP4) / 4  =  (128 + 448 + 64) / 4  = 160 B
HCA  (k2=128): (64·BF16 + 448·FP8)           / 128 =  (128 + 448)    / 128 =   4 B
SWA  (uncompressed): 64·BF16 + 448·FP8                                   = 576 B  (caps at win=128 per request)
```

`tests/test_v4_config.py::test_native_v4_csa_bytes_per_token_matches_paper` and the
sibling Rust tests in `crates/tessera-core/src/config.rs` pin these numbers.

## Layer-pattern semantics

V4-Pro's pattern (paper §4.2.1):
> "For the first two layers, we use HCA. For the subsequent layers, CSA and HCA are
> used in an interleaved manner."

Tessera encodes that exactly in `deepseek_v4_pro.toml::v4.layer_pattern`:

```text
hca, hca,                          # layers 0-1
csa, hca, csa, hca, …, csa         # layers 2-60 (59 alternating)
```

The Pydantic config validator enforces `len(layer_pattern) == num_layers` and that
`block_size_tokens` is a multiple of `lcm(k1, k2)` for the patterns in use.

## What this sprint does **not** ship

* **V4 kernel runtime** — Lightning Indexer, top-k selector, CSA/HCA core attention.
  Lives in DeepSeek's TileLang reference impl on HuggingFace; integrates via
  `kernel_dispatch.py` in a future sprint.
* **FP4 calibration tooling** — `fp8_calibrate.py` will generalise to FP4 with a single
  constant swap; deferred until the kernels are wired.
* **Disk tier integration with block manager** — `DiskBackend` is a sibling
  `DeviceBackend` impl; orchestration code that spills evicted blocks to disk is Sprint
  6+ work.

## Status flag

`TesseraConfig.is_v4` (Python) and `CompressionScheme::is_v4()` (Rust) report whether
the active config selects the V4 hybrid path. Sprint 5 callers that need to branch on
this — most won't, because `to_native_config()` resolves the per-layer dispatch
internally — can use it as a single point of truth.

## References

**Internal**

* DeepSeek-V4 paper PDF (vendored): `Research/DeepseekV4/DeepSeek_V4-paper-oficial.pdf`
* Gap analysis: `Research/DeepseekV4/gap-analysis.md`
* ADRs 0020 (V4 hybrid architecture), 0021 (per-layer schemes), 0022 (mixed precision
  per region), 0023 (State Cache), 0024 (Disk Backend).

**External (public)**

* **DeepSeek-V4** model collection on Hugging Face — `huggingface.co/collections/deepseek-ai/deepseek-v4`
  (paper, weights, inference reference impl).
* **DeepSeek-V4-Pro inference reference** — `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
  (TileLang kernels for Lightning Indexer + CSA / HCA cores).
* **FlashMLA** (Hopper / Blackwell MLA kernel) — `github.com/deepseek-ai/FlashMLA`.
* **FlashInfer MLA backend** — `github.com/flashinfer-ai/flashinfer`.
* **vLLM** V1 BlockAllocator protocol — `github.com/vllm-project/vllm`.
* **KVCOMM** (cross-agent KV reuse, Oct 2025) — public arXiv preprint.
* **TokenDance** (collective KV sharing for multi-agent serving, Apr 2026) — public
  arXiv preprint.
