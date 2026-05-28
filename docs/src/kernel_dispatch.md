# Kernel Dispatch

Tessera does not ship a custom attention kernel; it dispatches to upstream kernels. The
rationale is in [ADR-0002](adr/0002-no-custom-wmma.md). The trade-off is explicit: Tessera
owns the block-layout / accounting layer; kernel authorship stays with the model's
upstream maintainers.

```python
class KernelBackend(str, Enum):
    FLASH_MLA   = "flash_mla"     # SM ≥ 9.0 (Hopper / Blackwell), MLA / V3
    FLASH_INFER = "flash_infer"   # Ampere+, MLA fallback
    FLASH_ATTN4 = "flash_attn4"   # Explicit experimental stub; raises
                                  # NotImplementedError until upstream backend matures.
    TRITON      = "triton"        # vLLM built-in fallback
```

## Selection

`select_backend_kind(requested)` resolves the active backend:

1. If `requested != "auto"` use the explicit choice.
2. Else if CUDA capability ≥ 9.0 **and** `flash_mla` importable → `FLASH_MLA`.
3. Else if `flashinfer` importable with `BatchMLAPagedAttentionWrapper` → `FLASH_INFER`.
4. Else → `TRITON`.

`FLASH_ATTN4` is opt-in only. Requesting it returns a backend whose `forward` raises
`NotImplementedError` with a pointer to the tracking label — the dispatcher treats it as
an explicit experimental seam rather than a silent no-op so misuse fails loudly.

A future DeepSeek-V4 TileLang backend (Lightning Indexer + CSA/HCA cores) follows the
same pattern: a `KernelBackend::TILELANG_V4` variant lands as a stub that raises until
upstream stabilises and the kernel is integration-tested under a cloud-burst session.

## What Tessera Hands the Kernel

**MLA / V3 (`FLASH_MLA`, `FLASH_INFER`, `TRITON`)**

```text
q_absorbed         [batch, num_heads, d_c]
q_rope             [batch, num_heads, d_r]
c_kv_block_table   [batch, max_blocks]      ← Tessera block ids
k_rope_block_table [batch, max_blocks]      ← Tessera block ids
seqlens            [batch]
```

Block ids resolve to device pointers via `primary_ptr` / `rope_ptr` before the kernel
call. The block manager owns the pointer arithmetic; the kernel reads contiguous memory.

**V4 hybrid** (future TileLang backend)

V4 schemes carry per-layer layout — the dispatch layer reads `MlaBlockConfig::
scheme_for_layer(idx)` to size per-region pointers (RoPE BF16 + content FP8 + indexer
FP4) and hands those alongside per-layer reservation tokens. The kernel signature is
not yet stable upstream; Tessera's accounting is paper-aligned (see
[v4_compliance.md](v4_compliance.md)) so the integration is a wrapper change, not a
layout rework.
