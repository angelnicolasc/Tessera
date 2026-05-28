# FP8 Storage

This page covers FP8 quantisation of MLA `c_kv` (V3 / Kimi-K2). DeepSeek-V4 uses a
**different storage model**: each compressed entry mixes BF16 (RoPE), FP8 (content), and
FP4 (Lightning Indexer) co-resident in one record. See
[ADR-0022](adr/0022-mixed-precision-per-region.md) and
[v4_compliance.md](v4_compliance.md) for the V4 mixed-precision layout. The calibration
discipline below — per-layer scales, BF16-always RoPE, 128K validation — extends to V4
(FP4 calibration generalises with one constant swap; tracked as TD-034).

FP8 `c_kv` storage composes orthogonally with MLA compression:

| Configuration | `c_kv` element | vs MHA BF16 |
|---|---|---|
| MHA BF16 (baseline) | 2 B | 1× |
| MHA FP8 | 1 B | ~2× |
| MLA BF16 | 2 B, 512+64 dims | **~57×** |
| MLA FP8 `c_kv` | 1 B (`c_kv`), 2 B (`k_rope`) | **~102×** |

At 128K context per request:

```text
MLA BF16   →  8.5 GB   fits A100-80G with model weights
MLA FP8    →  4.7 GB   fits A100-80G with 2 concurrent long-context requests
```

## Mandatory Calibration

Uncalibrated FP8 with FlashMLA caused a documented regression in April 2026 (vLLM blog):
128K needle-in-haystack recall fell from 91% → 13% due to imprecise FP32 accumulation in the
Tensor Cores. Tessera's mitigations:

1. **Per-layer scale factors** from a calibration corpus. See `fp8_calibrate.py`.
2. **`k_rope` always BF16.** The position-dependent path is the most precision-sensitive.
3. **128K needle-in-haystack validation** before FP8 is enabled in production
   ([ADR-0007](adr/0007-fp8-calibration-required.md)).

## Hash Interaction

The content hash is computed over the **stored bytes**. Two blocks with identical FP8 bytes
hash identically; two blocks quantised with different scale factors will hash differently
even if their BF16 originals were close. This is correct: exact dedup requires byte-level
agreement. Semantic similarity is handled by the HNSW layer.
