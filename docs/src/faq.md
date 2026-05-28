# FAQ

### FlashMLA is already open source. What does Tessera add?

FlashMLA is a *kernel*. No block manager, no cross-agent share table, no segment index, no
vLLM plugin. Tessera is the memory hierarchy that feeds FlashMLA correctly. Analogy: Linux's
page allocator versus a GPU driver — one provides physical pages, the other computes over
their contents.

### How does this differ from SGLang's RadixAttention?

RadixAttention is keyed on token-prefix match and indexes full K/V. Two requests
processing the same document at different prompt positions don't collide on its prefix
hash because position participates in the key. Tessera hashes `c_kv` content only —
position is absent by construction — and catches that case directly. The trade-off:
RadixAttention covers any model; Tessera's content-addressed sharing path is
MLA / V4-aware and degrades to standard prefix sharing under MHA fallback.

### TyphoonMLA (Huawei) is in the same space — is it competition?

TyphoonMLA is a kernel optimisation for shared-prefix scenarios in mixed naive/absorb mode.
It lives at the kernel layer; it has no block manager or cross-agent infrastructure. Tessera
and TyphoonMLA operate at different layers; the kernel dispatcher can route to a
TyphoonMLA-flavoured backend in the future.

### Why exploit MLA's native position-independence instead of KVCOMM's adjustment?

KVCOMM works on any architecture by estimating an adjustment factor per token. For MLA, the
adjustment is mathematically zero — `c_kv` is position-independent by construction. Tessera
gives the stronger guarantee with less compute and zero approximation error.

### What happens if a model doesn't use MLA?

The `mha_fallback.toml` sets `latent_dim=0` which selects `CompressionScheme::MhaFull`.
Block sizing falls back to standard K/V; the segment index and share table continue to work
(they're content-agnostic). Kernel dispatch defaults to Triton.

### Minimum GPU for Tessera to beat stock vLLM?

Any GPU that can run a DeepSeek-V3 inference. The block manager and segment index
improvements are GPU-agnostic. The kernel throughput gain (FlashMLA vs stock) requires
Hopper or newer for the full bandwidth advantage; Ampere sees a smaller but real gain via
FlashInfer.

### What does Tessera add for DeepSeek-V4?

V4 (paper preview, May 2026) replaces the V3 MLA cache with a hybrid CSA + HCA + SWA
layout, a two-tier paged + per-request state cache, and an on-disk shared-prefix tier.
Tessera implements the block-layout/accounting layer for all three: `CompressionScheme::
{V4Csa, V4Hca, V4Swa}` as sibling variants, per-layer scheme maps in `MlaBlockConfig`,
`state_cache::StateCache` per-request arena, and `device::DiskBackend` with the three
SWA caching strategies the paper proposes. See `docs/src/v4_compliance.md` for the gap
summary and per-token byte accounting that matches the paper's published constants.

### When can I use this in production?

**Current status (v0.6.0-sprint5, CPU-validated):**

- ✅ Block manager, share table, segment index, distributed transports — all exercised
  by Rust unit + integration + proptest, Python pytest + hypothesis, on every PR.
- ✅ V4 hybrid layout (CSA/HCA/SWA per-layer interleaved) — type system + byte
  accounting verified against the paper's §2.3.4 constants.
- ✅ Multi-rank intra-node coordination — `MockTransport` exercised E2E; `P2pCuda` /
  `NCCL` transport stubs compile against feature flags.
- ✅ PD-disaggregation with reserve-then-stream transactional semantics (ADR-0018).
- ✅ Distribution: manylinux_2_28 wheels + ghcr.io image built per push.

**Cloud-burst gated** (wired but not yet run on real GPU):

- FlashMLA / FlashInfer parity vs the PyTorch reference oracle.
- 128K needle-in-haystack precision regression.
- Tessera-vs-stock-vLLM throughput benchmark.
- `P2pCudaTransport` runtime (NVLink IPC); `NcclTransport` runtime (multi-node IB).
- `CudaXxh3Hasher` on-device hashing kernel.
- V4 kernel integration via DeepSeek's TileLang reference impl.

The CPU-only path is production-shaped (typed errors, eviction, lifecycle, observability,
backpressure, chaos-tested). What's left for a true production rollout is the GPU
validation pass — see `Sprint 5 status` in the README for the full table.
