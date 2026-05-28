# Chaos Testing

Sprint 4 adds two complementary fuzz/chaos layers that exercise Tessera's distributed
protocols under failure modes. Everything runs on **CPU only** — the chaos rig is the
mechanism that lets us validate the multi-rank robustness story before cloud-burst GPU
work begins.

## The injection layer

`crates/tessera-core/src/transport/latency.rs` ships `LatencyInjector<T>` — wraps any
`RankTransport` with a tunable [`LatencyProfile`](adr/0019-latency-injection-chaos-testing.md):

```rust
LatencyProfile {
    intra_node_us: 5,        // NVLink baseline
    intra_rack_us: 50,       // NVSwitch
    cross_rack_us: 500,      // IB
    jitter_us: 100,          // ± symmetric
    drop_rate: 0.05,         // 5% transport failures
}
```

Three preset constants for common scenarios: `INTRA_NODE_REALISTIC`, `STRESS_MULTI_RACK`,
`ALL_DROPS`. Determinism via explicit seed (`ChaCha8Rng`).

## Layer 1 — Rust proptest

`crates/tessera-core/tests/proptest_chaos.rs`:

| Property | Coverage |
|----------|----------|
| `alloc_free_release_keeps_used_blocks_consistent` | Random op sequences (Allocate / Seal / Free / ReleaseRequest) keep `used_blocks ≤ total_blocks` |
| `free_of_known_block_never_panics` | `free` on previously allocated ids is always safe, even after eviction reaped them |
| `release_request_fidelity` | `release_request(req)` count equals private blocks owned by `req` at call time |
| `transfer_atomicity_under_chaos` | `transfer_request_to_rank` is all-or-nothing: source either fully drains or fully retains, never partial — under `drop_rate ∈ [0, 0.75]` |

Run: `cargo test --workspace proptest`

## Layer 2 — Python hypothesis (PyO3 boundary fuzz)

The Rust core can be correct in isolation while the PyO3 boundary mangles a type
conversion. Hypothesis covers that crossover:

`tests/test_hypothesis_allocator.py`:

| Property | Why it matters |
|----------|----------------|
| `test_used_blocks_consistent_under_random_ops` | Python's live-set tracking and `manager.used_blocks` stay in sync through arbitrary op sequences |
| `test_seal_of_identical_bytes_is_deterministic` | The dedup contract is observable from Python (canonical block, `was_dedup` flag) |
| `test_release_request_returns_exact_count` | The return type marshalling across PyO3 reports the right integer |

`tests/test_hypothesis_distributed.py`:

| Property | Why it matters |
|----------|----------------|
| `test_distributed_lookup_hash_shape` | Result is `None` or `(int, int)` for any hash, no exceptions |
| `test_distributed_lookup_safe_on_any_world_size` | Singleton + 2/3/4-rank worlds all return safe misses for unknown hashes |
| `test_usearch_add_query_roundtrip` | numpy ↔ PyO3 ↔ usearch roundtrip preserves vector identity |

Run: `pytest tests/test_hypothesis_*.py`

## Running the full chaos suite

```bash
# Rust + Python combined chaos cycle (CPU only)
cargo test --workspace proptest
pytest tests/test_hypothesis_allocator.py tests/test_hypothesis_distributed.py -v
```

CI runs both on every PR. Failures land in `tests/proptest-regressions/` and
`.hypothesis/examples/` for forensics.

## Bounds & follow-ups

| Limitation | Tracking | Follow-up |
|------------|----------|-----------|
| Per-op drop selectivity not supported | TD-028 | Sprint 5 if production calls for it |
| Hypothesis covers world_size ∈ [1, 4] only | TD-031 | Slower CI cadence for larger fuzz |
| Reservation-vs-eviction fairness under sustained load | TD-029 | Sprint 5 (tier-d analog for reserved blocks) |
