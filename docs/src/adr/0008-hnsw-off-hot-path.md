# ADR-0008 — HNSW is off the hot path; budget violations are safe misses

**Status:** Accepted, 2026-05-21.

## Context

The original Tessera v0.1 design ran HNSW lookup synchronously on the prefill hot path.
A reviewer (Apr 2026) flagged: at typical `ef=64`, HNSW lookup is ~10–50 µs P50 with a long
tail; that latency is unbearable on TTFT-sensitive workloads.

We considered:

* **Skip HNSW entirely.** Loses the cross-tokenisation match case (BPE boundary drift across
  agents producing slightly different `c_kv`).
* **Trade recall for latency by setting `ef=1`.** Defeats the purpose; HNSW becomes a coin flip.

## Decision

* **Layer 1 (exact xxhash3)** stays synchronous and on the hot path. Sub-µs.
* **Layer 2 (HNSW)** is async, off the hot path. Wrapped in `asyncio.wait_for` with an
  explicit **500 µs P99 budget** (configurable via `hnsw_latency_budget_us`).
* **Budget exceeded → return `None`.** This is a *correctness-preserving* miss: the request
  computes its own `c_kv` instead of reusing a sibling's. Bumps `tessera_hnsw_budget_exceeded_total`.

Additionally tracked here for a future sprint: **on-device content hashing.** Today the
block manager does `dtoh` to compute the seal hash. At production block-rates the round-trip
dominates `seal()`. A custom hashing kernel run on the device removes the copy entirely.

## Consequences

* TTFT is bounded by the budget. The budget can be tightened per deployment without code
  changes (TOML config).
* The Python `ThreadPoolExecutor` for the async HNSW is single-worker by default. Increase
  if the deployment serves many concurrent prefills.
* Observability requirement: monitor `tessera_hnsw_budget_exceeded_total / tessera_hnsw_match_hits_total`.
  A high ratio means raising `expansion_search` won't help — the underlying CPU is starved.
