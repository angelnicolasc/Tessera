# Testing

## Layers

| Layer | Tooling | What it covers |
|---|---|---|
| Rust unit | `cargo test --workspace` | Config validation, hash determinism, CPU mock invariants |
| Rust integration | `crates/tessera-core/tests/*` | Alloc/free, seal/dedup, CoW isolation, share table, lifecycle, eviction |
| Rust proptest | `crates/tessera-core/tests/proptest_invariants.rs` | 6 property invariants over randomised block manager inputs |
| Index recall | `crates/tessera-index/tests/recall.rs` | Top-1 self-recall ≥ 0.95 on random vectors |
| Python | `pytest tests -m "not gpu"` | Config roundtrip, segment index two-layer behaviour, kernel dispatch, vLLM plugin protocol shape, FP8 calibration math, reference oracle sanity, E2E CPU integration |
| GPU integration | `pytest -m gpu` | FlashMLA / FlashInfer parity vs reference oracle; 128K needle-in-haystack precision regression |

---

## Property-based testing (proptest)

`crates/tessera-core/tests/proptest_invariants.rs` exercises six invariants over randomised
inputs using the [proptest](https://docs.rs/proptest) framework:

| Invariant | Description |
|-----------|-------------|
| `round_trip_alloc_free_reaches_zero` | Balanced alloc/free sequence → `used_blocks == 0` |
| `seal_dedup_is_consistent` | Identical content always seals to the same canonical block |
| `dedup_reduces_to_one_unique_block` | N seals of identical content → exactly 1 live block |
| `cow_fork_is_isolated` | Mutation in a CoW fork never changes the original block's hash |
| `eviction_never_frees_shared_block` | `ref_count > 1` blocks survive any eviction pressure |
| `release_request_frees_exactly_owned_blocks` | `release_request(req_id)` frees exactly the blocks allocated for that req |

Run: `cargo test --workspace proptest`

---

## PyTorch reference oracle

`python/tessera/reference/absorbed_attention.py` provides `reference_absorbed_mla` — a
pure-CPU, FP32-accumulation implementation of absorbed MLA attention. It is the ground truth
for all GPU parity tests.

Key properties:
- No GPU required. Runs in every CI environment.
- Accumulates in FP32 regardless of input dtype — avoids BF16 rounding errors that could
  mask precision regressions in the oracle itself.
- Designed to catch the vLLM April 2026 regression class (uncalibrated FP8 causing 128K
  needle-in-haystack recall to drop from 91% → 13%).

See [ADR-0011](adr/0011-pytorch-reference-oracle.md) for rationale.

---

## GPU-gated tests (Tier B)

All tests in `tests/gpu/` are decorated with `@pytest.mark.gpu` and use the `gpu` fixture
from `tests/gpu/conftest.py`. The fixture skips automatically if `torch.cuda.is_available()`
returns `False`.

**Cloud burst** — to run the GPU-gated suite:

```bash
pytest -m gpu -v
```

The GPU parity test harness is wired against the PyTorch reference oracle and skips
gracefully when CUDA is unavailable. **V4 runtime integration** (Lightning Indexer +
CSA/HCA cores via DeepSeek's TileLang reference impl) and **real vLLM-engine
integration tests** remain cloud-burst / Sprint 6+ gated — see the `Sprint 5 status`
table in the README for the full list.

Files:

| File | What it tests |
|------|---------------|
| `test_flash_mla_parity.py` | FlashMLA vs reference oracle; parametrised over `(batch, seqlen, dtype)` |
| `test_flash_infer_parity.py` | FlashInfer vs reference oracle; same parametrisation |
| `test_needle_haystack.py` | 128K precision: `mean_abs > 5.0`; needle attention weight `> 0.9` |

Tolerance: `atol=1e-2, rtol=1e-3` for BF16 kernel outputs vs FP32 oracle.

---

## Determinism

All Tier A tests run with deterministic seeds. The CPU mock backend produces the same
bytes on every machine, so `seal_dedup` and `cow_isolation` are reproducible cross-platform.

---

## Coverage Gates (hard)

| Component | Gate | Enforcer |
|-----------|------|----------|
| `tessera-core` | ≥ 85% line | `coverage.yml` — parses `cargo-llvm-cov --json`; `exit 1` if below |
| `python/tessera` | ≥ 75% line | `coverage.yml` — `pytest --cov-fail-under=75` |

CI rejects any PR that drops below these thresholds. Codecov uploads provide per-file
breakdowns for forensics.
