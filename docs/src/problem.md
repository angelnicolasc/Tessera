# The Problem

## MLA vs MHA Storage

DeepSeek-V2 / V3 and Kimi-K2 replace the conventional MHA KV cache with a compressed
latent representation (MLA — Multi-head Latent Attention):

```text
MHA — per token, per layer:
    K[layer, token] ∈ ℝ^(num_heads × head_dim)
    V[layer, token] ∈ ℝ^(num_heads × head_dim)

MLA — per token, per layer:
    c_kv[layer, token]   ∈ ℝ^(d_c)        # position-independent
    k_rope[layer, token] ∈ ℝ^(d_r)        # decoupled RoPE, position-dependent
```

K and V are reconstructed on the fly via up-projection matrices `W_UK`, `W_UV`. They are
**never stored**. This is not a software optimisation — it is a different thing in memory.

**DeepSeek-V4** (May 2026 paper preview) **extends** this idea with a hybrid architecture
rather than reusing the V3 MLA layout directly: per-layer interleaved CSA (Compressed
Sparse Attention, `k1 = 4` tokens → 1 compressed entry + Lightning Indexer) + HCA
(Heavily Compressed Attention, `k2 = 128` tokens → 1 entry) + SWA (Sliding Window,
uncompressed for the most recent `win = 128` tokens). The compressed entries use mixed
precision (BF16 RoPE + FP8 content + FP4 indexer). The shared structural property is
that *position-independent content lives separately from position-dependent context* —
which is what makes content-addressed sharing possible across both V3 and V4. See
[v4_compliance.md](v4_compliance.md) for V4 details.

## The 56.9× Headline Number

For DeepSeek-V3 (`L=61`, `H=128`, `d_h=128`, `d_c=512`, `d_r=64`, BF16):

```text
MHA bytes/token:  2 · 128 · 128 · 61 · 2  = 3,997,696   ≈ 3.81 MB
MLA bytes/token:  (512 + 64) · 61 · 2    =    70,272   ≈ 68.6 KB
Ratio:            3,997,696 / 70,272      ≈ 56.9×

With FP8 c_kv (2× additional):           ≈ 101.7× total
```

At 128K context per request:

| Format | KV cache size |
|---|---|
| MHA BF16 | 488 GB (impossible on a single node) |
| MLA BF16 | 8.6 GB (fits on one A100-80G) |
| MLA FP8 c_kv | 4.7 GB (fits with headroom for weights) |

## Four Inefficiencies in 2026 Production Stacks

1. **Frameworks expand before caching.** Many MLA integrations in vLLM, TGI, SGLang expand
   `W_UK · c_kv` and `W_UV · c_kv` eagerly and store the result — discarding the entire
   compression gain at the block manager layer.
2. **Wrong block size.** PagedAttention's default 16-token blocks are sized for MHA
   payloads. Even when frameworks DO store `c_kv`, the block accounting, eviction heuristics,
   and prefix-hash comparisons run on a granularity that's two orders of magnitude off.
3. **Prefix caching ignores position-independence.** vLLM APC and SGLang RadixAttention
   require exact token-prefix match. `c_kv` contains no RoPE — it CAN be shared between
   requests that process the same document at different positions. We have not found
   this implemented in mainstream open-source serving stacks as of 2026.
4. **Multi-agent systems recompute shared context.** TokenDance (Apr 2026) measures this
   in production. KVCOMM (Oct 2025) shows 7.8× speed-up when cross-agent caching is done
   right.
5. **V4 hybrid layouts break PagedAttention assumptions** (paper §3.5.1, verbatim). The
   per-layer interleaved CSA / HCA / SWA and the two-tier paged + State Cache demand a
   block manager that models heterogeneity at the layout layer.

Tessera addresses all five at the block-layout / accounting layer — kernel authorship
stays upstream.
