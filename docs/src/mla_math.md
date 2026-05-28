# MLA Mathematics

## Naive Forward Pass

For decode step at position `t`, layer `l`:

```python
q = W_DQ @ x[t]
q_content, q_rope = split(q)
q_rope = RoPE(q_rope, pos=t)

c_kv   = cache[l, t]          # [d_c]
k_rope = cache_rope[l, t]     # [d_r], RoPE baked in at prefill

K_content = W_UK @ c_kv       # expensive HBM traffic per token
V         = W_UV @ c_kv
K = concat(K_content, k_rope) # k_rope broadcast across heads

scores = softmax(q @ K.T / sqrt(d_h))
out    = scores @ V
```

## The Absorption Trick

Because `K_content = W_UK · c_kv`:

```text
q_content · K_content^T
  = q_content · (W_UK · c_kv)^T
  = (q_content · W_UK) · c_kv^T
  = q_absorbed · c_kv^T          ← W_UK folded into the query, computed once
```

Similarly `scores · V = scores · (W_UV · c_kv) = (scores · W_UV) · c_kv` — `W_UV` is absorbed
into the output transform, never applied per cached token.

**Result:** attention operates directly on `c_kv` without ever materialising full K and V in
HBM. This is what FlashMLA implements, and what Tessera's block manager is designed to feed.

## The Position Split

```text
c_kv   (d_c=512): position-independent  →  content-addressable  →  shareable across agents
k_rope (d_r=64) : position-dependent    →  not shareable, stored per-position
```

`d_r / (d_c + d_r) = 11%` of latent storage. The remaining 89% is shareable. Tessera's
segment index hashes only `c_kv`; `k_rope` is stored contiguously but never indexed.
