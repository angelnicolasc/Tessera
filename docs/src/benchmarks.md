# Benchmarks

## Rust micro-benches (Criterion)

```bash
cargo bench --workspace
```

| Bench | What it measures |
|---|---|
| `block_manager/alloc_free` | Allocation + free throughput (operations / s) on the CPU mock |
| `content_hash/xxh3` | xxhash3 throughput across 4 KiB → 4 MiB block sizes |
| `share_table/concurrent_add` | `add_share` throughput under 1 / 4 / 8 threads |

These benches don't require a GPU. Their value is establishing a floor: any future change
that regresses the bookkeeping cost by more than 2× shows up here immediately.

## Python benchmarks

```bash
python -m benchmarks.memory_report     # MHA vs MLA BF16 vs MLA FP8 table
python -m benchmarks.sharing_bench     # Cross-agent dedup rate (CPU mock OK)
pytest benchmarks/kernel_bench.py      # FlashMLA throughput (skipped on CPU)
```

### Memory report — DeepSeek-V3 (MLA)

```text
Model: deepseek-v3  (L=61, H=128, d_h=128, d_c=512, d_r=64)
Block: size_tokens=64, ckv_dtype=bf16
Compression vs MHA BF16: 56.6x

 Context     MHA BF16     MLA BF16     MLA FP8   Ratio (MLA/MHA)
     8K       30.5 GB       0.54 GB       0.30 GB          56.6x
    32K      122.0 GB       2.17 GB       1.21 GB          56.2x
   128K      488.0 GB       8.68 GB       4.84 GB          56.2x
   512K     1952.0 GB      34.72 GB      19.36 GB          56.2x
  1024K     3904.0 GB      69.44 GB      38.72 GB          56.2x
```

### Per-token accounting — DeepSeek-V4 (hybrid)

`benchmarks.memory_report` covers V3 MLA today; V4 sizing is computed structurally
because the layout is per-layer heterogeneous (see
[v4_compliance.md](v4_compliance.md)). Per the V4 paper §2.3.4, the V4-Pro layout
(`k1=4`, `k2=128`, `head_dim=512`, `rope_dim=64`, indexer=128 dims, FP8 content + BF16
RoPE + FP4 indexer):

| Layer kind | Compression | Per-token bytes / layer | Composition |
|---|---|---|---|
| CSA  | `k1 = 4` (overlapping) | **160 B** | `(64·BF16 + 448·FP8 + 128·FP4)/4` |
| HCA  | `k2 = 128` (non-overlapping) | **4 B** | `(64·BF16 + 448·FP8)/128` |
| SWA  | uncompressed (`win = 128`, cap per request) | 576 B in State Cache | `64·BF16 + 448·FP8` |

The paper headlines ~**2% of GQA8 BF16** at 1M context. Tessera's accounting matches
the paper's per-token constants verbatim — pinned by
`tests/test_v4_config.py::test_native_v4_csa_bytes_per_token_matches_paper` (Python)
and `crates/tessera-core/src/config.rs::tests::v4_csa_per_token_bytes_match_paper`
(Rust).

A `python -m benchmarks.memory_report --model v4-pro` extension is listed under
[Roadmap](#roadmap) below; the structural numbers above are the authoritative source
until that lands.

## Roadmap

* End-to-end V4-Pro / V4-Flash memory report integrated into the `memory_report`
  script.
* Tessera-vs-stock-vLLM throughput benchmark (`benchmarks/vllm_compare.py` skeleton
  already wired, GPU-gated).
* Nightly Criterion regression check with > 20 % drift alerting (Sprint 4 workflow
  in place).
