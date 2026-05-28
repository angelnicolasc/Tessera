# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The file is
managed by [release-please](https://github.com/googleapis/release-please) — additions on
`main` derived from Conventional Commit messages will appear here automatically.

## [Unreleased]

## [0.6.0-sprint5] — 2026-05-27

**DeepSeek-V4 Compliance.** Brings Tessera's block manager into structural alignment with
DeepSeek-V4 (paper preview, May 2026 — §2.3 hybrid attention, §3.5.1 KV cache, §3.5.2 on-disk
storage). All work CPU-validated; real V4 kernel integration follows in a future GPU
session.

### Added

- **WS-V4-A** — `CompressionScheme::{V4Csa, V4Hca, V4Swa}` variants; `CkvDtype::Fp4E2m1` + `MixedBf16Fp8Fp4`; `bytes_per_token_per_layer()` self-describing accounting; `DsaHierarchical` marked `#[deprecated]` with V4 migration pointer. ADR-0020, ADR-0022.
- **WS-V4-B** — `MlaBlockConfig::with_per_layer_schemes(...)` + `schemes_per_layer: Option<Arc<Vec<…>>>` field; `scheme_for_layer(idx)`, `has_per_layer_schemes()`, `v4_block_size_lcm()`. ADR-0021.
- **WS-V4-C** — `state_cache::{StateCache, StateCacheConfig, RequestArena}` for V4's per-request SWA + tail arena. `StateCacheConfig::for_v4(...)` derives sizing from paper constants. ADR-0023.
- **WS-V4-D** — `device::DiskBackend` implementing `DeviceBackend` over filesystem-backed regions; `SwaCachingStrategy::{Full, Periodic, Zero}` matching paper §3.5.2; cross-process shared-prefix cache as a free side-effect. ADR-0024.
- **WS-V4-E** — PyO3 + Python config:
  - Native module gains `CkvDtype.{Fp4E2m1, MixedBf16Fp8Fp4}`, `CompressionScheme.{v4_csa, v4_hca, v4_swa}`, `MlaBlockConfig.with_per_layer_schemes`, `scheme_for_layer`, `has_per_layer_schemes`, `v4_block_size_lcm`.
  - Python `TesseraConfig` gains `V4Config` and `DiskCacheConfig` sub-models; `is_v4` property; `to_native_config()` builds per-layer hybrid configs.
  - `_native.pyi` stubs updated.
- **WS-V4-F** — `models/deepseek_v4_flash.toml` and `models/deepseek_v4_pro.toml` with paper §4.2.1 dimensions verbatim.
- **WS-V4-G** — 8 new Rust tests in config.rs (V4 byte accounting, validation, LCM, FP4 packing); `dsa_audit.rs` updated for deprecation messages; `tests/test_v4_config.py` (8 tests, CPU-only).
- **WS-V4-H** — ADRs 0020-0024; `docs/src/v4_compliance.md`; `Research/DeepseekV4/gap-analysis.md`; SUMMARY.md "Model Compliance" section; ARCHITECTURE.md ADR table extended to 24.

### Changed

- `CompressionScheme::DsaHierarchical` is `#[deprecated(since = "0.6.0")]`. Construction still permitted; `MlaBlockConfig::new` rejects with a migration message pointing at V4Csa/V4Hca/V4Swa.
- Sprint 5 dependencies: `tempfile = "3.13"` added as dev-dep on `tessera-core`.

### Closed tech debt

- "DSA c4a/c128a semantics (await DeepSeek-V4 public specs)" — closed by the entire WS-V4-* series. The paper's actual nomenclature (CSA/HCA/SWA) replaced our placeholder guess.

### New tech debt

- TD-034: FP4 calibration tooling — `fp8_calibrate.py` generalises with one constant swap (`FP4_E2M1_MAX = 6.0`); deferred until V4 kernel runtime arrives.
- TD-035: `DiskBackend` mirrored buffers should swap to `memmap2` zero-copy behind a `disk-mmap` feature (production hardening).
- TD-036: Disk-tier eviction policy — block manager's tiered LRU (ADR-0010) spilling to `DiskBackend` is Sprint 6+ orchestration.
- TD-037: `StateCache` ↔ block manager lifecycle integration in the vLLM plugin (Sprint 6+).
- TD-038: V4 kernel runtime (Lightning Indexer + CSA/HCA cores) — pending DeepSeek's TileLang reference integration.

## [0.5.0-sprint4] — 2026-05-18

**Distributed Robustness — CPU-only chaos validation.** Endurece el protocolo distribuido
intra-node con transactional PD-disagg, chaos injection (Rust proptest + Python hypothesis),
topology-aware budgets, y bridge OTLP Rust. Cloud-burst GPU pendiente para Sprint 5.

### Added

**WS1 — `LatencyInjector` + `LatencyProfile`**
- `crates/tessera-core/src/transport/latency.rs`: wraps any `RankTransport` with
  tier-aware latency (intra_node/intra_rack/cross_rack µs) + symmetric jitter + drop_rate.
- Presets: `INTRA_NODE_REALISTIC`, `STRESS_MULTI_RACK`, `ALL_DROPS`, `ZERO`.
- Deterministic via explicit `ChaCha8Rng` seed; `LatencyInjector::with_entropy` for
  non-deterministic staging chaos.
- Metric: `tessera_latency_injected_drops_total{op}`.
- `tests/latency_injection.rs`: 6 tests (intra-node latency honored, multi-node tier
  picks correctly, ALL_DROPS / ZERO, determinism by seed, metric increments).

**WS2 — Multi-node topology semantics + budget tier scaling**
- `rank.rs`: `LatencyTier { IntraNode, IntraRack, CrossRack }`; `Topology::node_of`,
  `Topology::is_same_node`; `World::peer_tier(other) -> LatencyTier`.
- `crates/tessera-index/src/distributed.rs`: `TierBudget` struct + `DistributedSegmentIndex::
  with_tier_multipliers`; `effective_budget_for(peer)` exposes per-peer budget.
  Fan-out budget = max-per-peer tier budget (no longer min'd against base, allowing
  multi-node lookups to expand within their allotment).
- `MockTransport::with_topology(size, topology, profile, seed)` builder that wraps
  every handle in a `LatencyInjector` per supplied topology.
- `tests/topology.rs`: 4 tests (single-node classification, multi-node mapping, tier
  string stability, builder wrap verification).
- `tests/distributed.rs`: gains `budget_scales_per_tier_multi_node` test.

**WS3 — Reserve-then-stream transactional PD-disagg (supersedes ADR-0016 push-mode)**
- `RankTransport` trait gains `reserve_slots(dst, req_id, count) -> ReservationToken` +
  `release_reservation(dst, token) -> ()`. Three impls (mock, p2p_cuda stub, nccl stub)
  updated.
- `ReservationToken(u64)` opaque newtype with `(rank << 48) | counter` encoding for
  globally unique tokens.
- `TesseraBlockManager::reserve_incoming(req_id, count)` forces eviction up to `count`
  before failing with `OutOfBlocks`.
- `transfer_request_to_rank` rewritten with 3-phase protocol: reserve → stream → commit.
  Any mid-stream failure triggers `release_reservation` rollback; source state retained
  on any abort. Closes TD-024.
- `BlockManagerPeerAdapter` in PyO3 lib.rs forwards `reserve` to `reserve_incoming` so
  capacity is enforced through the Python boundary.
- Metrics: `RESERVATIONS_ACTIVE{rank}` gauge, `TRANSFER_ABORTS_TOTAL{reason}` counter.
- `tests/pd_disagg_transactional.rs`: 4 tests (full success, destination OOM aborts
  cleanly, mid-stream ALL_DROPS rolls back without leaks, empty request returns 0).

**WS4 — Rust proptest chaos suite**
- `tests/proptest_chaos.rs`: 4 properties × 64 cases each — round-trip consistency,
  safe free of evicted blocks, release_request fidelity, transfer atomicity under
  random `drop_rate ∈ [0, 0.75]`.
- Bounded wall-clock (`max_shrink_iters = 256`).

**WS5 — Python hypothesis boundary fuzz**
- `pyproject.toml`: `hypothesis>=6.112` in `dev` extra.
- `tests/test_hypothesis_allocator.py`: 3 properties × 64 cases — Python ↔ Rust tracking
  parity, seal idempotence, release_request exact count.
- `tests/test_hypothesis_distributed.py`: 3 properties × 64 cases — `lookup_hash` shape,
  safe miss across world sizes, usearch add/query roundtrip.

**WS6 — Rust `tracing-opentelemetry` bridge (`otel-rust` feature)**
- Workspace deps: `tracing-opentelemetry 0.27` + `opentelemetry 0.26` family.
- `crates/tessera-core/src/observability.rs`: `init_otlp_tracing(endpoint, service_name)`
  + `shutdown_tracing()`. No-op stubs when feature disabled.
- Composes with the Python OTLP exporter (Sprint 2): both emit `tessera.*` spans to the
  same endpoint; W3C tracecontext propagation.
- Closes TD-030.

**WS7 — Documentation**
- ADR-0017 (Rust OTLP bridge), ADR-0018 (reserve-then-stream), ADR-0019 (latency
  injection + chaos).
- `docs/src/chaos.md`: methodology + tables of properties tested.
- `docs/src/SUMMARY.md` + `ARCHITECTURE.md`: ADR table extended to 19.
- README + `Sprint 4 status` table; version badge → 0.5.0.

**WS8 — DEVLOG Sprint 4 entry** (see `DEVLOG.md`).

**WS9 — CI**
- `.github/workflows/ci.yml`: new `build-otel-rust` job (`cargo build --features otel-rust`).
- `test-python` + `test-multirank-python` jobs include `hypothesis` install.

### Changed

- `RankTransport` trait grew two methods (`reserve_slots`, `release_reservation`); not
  breaking for external consumers since the trait is consumed dyn-dispatch.
- `transfer_request_to_rank` semantics now reserve-then-stream (ADR-0018 supersedes
  ADR-0016's push-mode). Existing call sites compile unchanged; behaviour is strictly
  more robust.

### Closed tech debt

- TD-024 (rollback under destination failure) — closed by WS3.
- TD-025 (chaos / latency injection) — closed by WS1+WS4+WS5.
- TD-027 (`Topology::MultiNode` semantics) — closed by WS2.
- TD-030 (Rust OTLP bridge) — closed by WS6.

### New tech debt

- TD-028: `LatencyProfile.drop_rate` is global, not per-op.
- TD-029: Reservation-vs-eviction fairness under sustained heavy load (monitoring item).
- TD-031: Hypothesis covers `world_size ∈ [1, 4]` only.
- TD-032: `ReservationToken` is opaque u64; cross-deployment collision resistance not
  formally proven.
- TD-033: OTLP env-var auto-discovery (`OTEL_EXPORTER_OTLP_ENDPOINT`) not yet wired —
  init still requires explicit call.

## [0.4.0-sprint3] — 2026-04-24

**Multi-GPU / Tensor Parallelism (intra-node, NVLink-ready).** CPU-only mainline; cloud-burst
session validates the GPU runtime.

### Added

**WS1+WS6 — Rank-aware block manager**
- `crates/tessera-core/src/rank.rs`: `RankId(u32)` newtype, `NodeId(u32)`, `Topology { SingleNode, MultiNode { node_of } }` (`#[non_exhaustive]`), `World { local, size, topology }` with `singleton()`, `new()`, `peers()`, `has_peers()`.
- `block.rs`: `GlobalBlockId(RankId, BlockId)` for cross-rank identity.
- `block_manager.rs`: `rank`, `world`, `rank_label` fields; `new_with_world()` + `with_backend_and_world()` constructors. Existing `new()` retained as singleton convenience (deviation from initial plan documented in ADR-0014).
- `tests/rank_basic.rs`: 5 tests covering singleton retrocompat, multi-rank construction, out-of-range rejection, global id formatting, multi-node validation.

**WS2 — `trait RankTransport` + 3 impls**
- `transport/mod.rs`: trait with `broadcast_seal`, `fetch_block`, `push_block`, `announce_release`, `query_hash`; `BlockPayload { c_kv, k_rope, fp8_scales }`.
- `transport/mock.rs`: `MockTransport` with in-process tokio mpsc channels; `EventLog` for assertions; `MockPeer` trait + `BlockManagerPeerAdapter`.
- `transport/p2p_cuda.rs` (feature `cuda`): `P2pCudaTransport` stub — compiles, returns structured error on call citing TD-021 for cloud-burst.
- `transport/nccl.rs` (feature `nccl`): `NcclTransport` stub — Sprint 4 runtime impl, TD-022.
- `Cargo.toml`: workspace deps `tokio`, `async-trait`, `futures`; tessera-core gains `nccl` feature flag (compile-only).
- `tests/transport_mock.rs`: 6 async tests (broadcast fan-out, fetch payload, push assignment, query_hash hit/miss, release propagation, topology).
- ADR-0015.

**WS3 — `DistributedSegmentIndex`**
- `crates/tessera-index/src/distributed.rs`: hash-only fan-out lookup with split local/remote budget; `DistributedHit { global, local }`; first-hit-wins via `FuturesUnordered + tokio::select!`.
- `tests/distributed.rs`: 4 tests (remote hit returns global id, all miss returns None, singleton short-circuits, budget exhausted → safe miss).

**WS7+WS8 — PD-disaggregation hook + multi-rank metrics**
- `block_manager.rs::transfer_request_to_rank(req_id, target, transport)`: push-mode transfer; on full success releases source-side locally. Reuses `export_payload` + transport.push_block + destination's `import_payload`.
- `DeviceBackend::write_bytes` added; impls in `cpu_mock.rs` + `cuda.rs` (htod_sync_copy_into).
- `metrics.rs`: 7 new families — `BLOCKS_PER_RANK`, `CROSS_RANK_TRANSFERS_TOTAL{src,dst,kind}`, `PD_DISAGG_TRANSFERS_TOTAL{src,dst}`, `DISTRIBUTED_INDEX_LOCAL_HITS_TOTAL`, `DISTRIBUTED_INDEX_REMOTE_HITS_TOTAL{src_rank}`, `DISTRIBUTED_INDEX_MISSES_TOTAL`, `DISTRIBUTED_INDEX_FANOUT_LATENCY_SECONDS` histogram.
- `tests/pd_disagg.rs`: 3 cases (transfer-all-blocks, unknown-req returns 0, payload roundtrip preserves bytes).
- `tests/metrics_multirank.rs`: 3 tests (cross-rank transfer counter increments on fetch, broadcast increments per-peer, all new families appear in snapshot text).
- ADR-0016.

**WS4 — PyO3 bindings + Python rank-aware plugin + `multi_rank.py`**
- `crates/tessera-py/src/lib.rs`: process-wide `TOKIO_RT` (`LazyLock<tokio::runtime::Runtime>`); 4 new classes — `PyRankId`, `PyWorld`, `PyMockTransport`, `PyDistributedSegmentIndex`; `BlockManager.with_world()` staticmethod; `BlockManager.{rank, world_size, global_id, transfer_request_to_rank}`. Inner switched to `Arc<TesseraBlockManager>` to enable adapter wiring.
- Fixed preexisting bug: `fp8_scales_ptr` no longer attempts `ptr as usize` on a struct (uses `.raw`).
- `python/tessera/multi_rank.py`: `MultiRankCoordinator` frozen dataclass + `spawn_multirank_world(config, world_size, accept_req_id=1)` helper.
- `python/tessera/vllm_plugin.py`: `TesseraBlockAllocator.__init__` gains `rank`, `world_size`, `transport` kwargs (backward-compatible defaults); routes to `BlockManager.with_world` when `world_size > 1`; constructs `DistributedSegmentIndex` when transport is supplied. `find_shared_prefix` now checks local-then-remote and returns `int | (rank, block) | None` per input block. New `transfer_request_to_rank(req_id, target)` method.
- `python/tessera/_native.pyi`: stubs for all new classes/methods.
- `python/tessera/__init__.py`: lazy-imports `RankId`, `World`, `MockTransport`, `DistributedSegmentIndex`, `MultiRankCoordinator`, `spawn_multirank_world`; `__version__` bumped to 0.4.0.

**WS5 — Multi-rank Python E2E**
- `tests/multirank/conftest.py`: `small_mla_config`, `tp4_world`, `tp2_world` fixtures.
- `tests/multirank/test_e2e_tp4.py`: 6 integration tests (rank isolation, payload roundtrip, rank-local release, transport event log fidelity, distributed index safe-miss path, singleton retrocompat).
- `tests/multirank/test_mp_isolation.py`: multiprocessing.spawn-based 4-worker validation that the rank-aware constructor survives true process boundaries.

**WS9+WS10 — ADRs + Documentation**
- ADR-0014 (multi-rank architecture; documents the `new()`-as-singleton deviation).
- ADR-0015 (P2pCuda vs NCCL transport selection).
- ADR-0016 (PD-disaggregation hook).
- `docs/src/multi_gpu.md`: TP=4 ASCII diagram + cross-rank share sequence + PD-disagg sequence + transport selection guidance.
- `docs/src/SUMMARY.md`: new "Distributed" section linking multi_gpu.md + 3 new ADRs.
- `ARCHITECTURE.md`: ADR table extended to 16.

**WS11 — DEVLOG Sprint 3 entry + TD-021..TD-027** (see `DEVLOG.md`).

**WS12 — CI updates**
- `.github/workflows/ci.yml`: new `build-nccl` job (Ubuntu, `cargo build --features nccl`).
- Test matrix continues to run `--features cuda` compile-only validation.

### Changed

- `TesseraBlockAllocator.__init__` signature: gained `rank`, `world_size`, `transport` kwargs. Defaults preserve Sprint 0-2 behaviour exactly. Version bump 0.4.0 reflects pre-1.0 freedom.
- `find_shared_prefix` return type extended from `list[int | None]` to `list[int | tuple[int, int] | None]` to express remote hits.
- `BlockMeta`-based `last_touched` (Sprint 1) and `ref_count` semantics unchanged; `transfer_request_to_rank` respects ADR-0010 by refusing to move blocks with `ref_count > 1` (would require explicit unsharing — Sprint 4).

## [0.3.0-sprint2] — 2026-03-27

### Added

**WS1 — PyO3 ergonomics (TD-016)**
- `PyCkvDtype.__repr__` returning `"CkvDtype.Bf16"` / `"CkvDtype.Fp8E4m3"`.
- `PyBlockManager.__repr__`, `__len__`, `__sizeof__`, `fp8_scales_ptr(block_id) → int | None`.
- `PyShareTable.__repr__`, `__len__`.
- `PyUsearchIndex.__repr__` with len and name.
- `python/tessera/_native.pyi` stubs updated for all new methods.
- `tests/test_pyo3_ergonomics.py`: 12 tests for all repr/len/sizeof/fp8_scales_ptr.

**WS2 — FP8 production pipeline (TD-018)**
- `BlockConfig.fp8_scales_path: str | None` field in `TesseraConfig`.
- `TesseraConfig.fp8_scales` property: lazy JSON load, cached, returns `None` for BF16.
- `fp8_calibrate.save_scales()` and `fp8_calibrate.load_scales()` for round-trip persistence.
- `TesseraBlockAllocator._write_fp8_scales()`: memcpy scale array to block's FP8 region on seal.
- `FlashMLABackend.forward_from_config()`: reads scales from config automatically.
- `tests/test_fp8_pipeline.py`: 8 tests covering round-trip, lazy caching, and no-exception sealing.

**WS3 — OTLP tracing (TD-010)**
- `observability.init_tracing(endpoint, service_name)`: configures OTLP gRPC exporter; no-op when empty or library absent.
- `observability.get_tracer()`, `observability.span(name)`: zero-overhead nullcontext when not configured.
- `TesseraBlockAllocator`: 4 operations instrumented — `tessera.allocate`, `tessera.seal`, `tessera.release_request`, `tessera.lookup_approximate`.
- `SegmentIndex.lookup_approximate`: `tessera.hnsw_query` span wrapping the executor call.
- `pyproject.toml`: `opentelemetry-exporter-otlp-proto-grpc>=1.27` added to `observability` extra.
- `tests/test_observability.py`: 5 tests (noop, absent otel, mock exporter, allocator span).
- ADR-0012: OTLP tracing Python layer rationale.

**WS4 — DSA audit (TD-017)**
- `CompressionScheme::DsaHierarchical` `unimplemented!` → `todo!` with ADR-0004 link and "DeepSeek-V4 specs" context in both `primary_bytes_per_token` and `rope_bytes_per_token`.
- `crates/tessera-core/tests/dsa_audit.rs`: 3 Rust tests (`catch_unwind` on both todo paths + `InvalidConfig` from `new`).
- `tests/test_dsa_audit.py`: 2 Python tests (construction succeeds, `MlaBlockConfig` raises `ValueError`).

**WS5 — Distribution (TD-019)**
- `.github/workflows/wheel.yml`: `cibuildwheel@v2.21` building `cp311` + `cp312` `manylinux_2_28` wheels; smoke-test step; artifact upload.
- `Dockerfile`: multi-stage `rust:1.82-slim` builder + `python:3.11-slim` runtime; smoke-test `RUN` layer.
- `.github/workflows/docker.yml`: `docker/build-push-action@v6` → `ghcr.io/<repo>:sprint2|latest|sha-<short>`; smoke-test after push.
- `.dockerignore`: excludes `target/`, `tests/`, `benchmarks/`, `docs/book/`, `.git/`.
- ADR-0013: manylinux_2_28 wheel and ghcr.io distribution rationale.

**WS6 — Stress tests**
- `tests/test_stress.py`: 4 Python stress tests — 50-thread lifecycle no-leaks, 95% memory pressure with eviction, eviction metrics reflection, 10K alloc/free throughput floor (> 50K ops/s).
- `crates/tessera-core/tests/stress.rs`: 2 Rust stress tests — concurrent alloc/release via Arc + 50 threads; eviction under full pressure no-panic.

**WS7 — Documentation**
- ADR-0012, ADR-0013 linked in `docs/src/SUMMARY.md`.
- `CHANGELOG.md`: this entry (`[0.3.0-sprint2]`).
- `DEVLOG.md`: Sprint 2 section with TD resolution column.
- `README.md`: Sprint 2 status table; badge updated.

### Changed

- `observability.py`: `span()` returns `contextlib.nullcontext()` (no-op) by default; `init_tracing()` must be called to activate.
- `vllm_plugin.py`: `free()` → `release_request()` rewiring retained; `__init__` now calls `init_tracing()` and stores `_fp8_scales`.
- `config.rs`: `unimplemented!` → `todo!` with ADR-0004 + DeepSeek-V4 link in DSA match arms.

## [0.2.0-sprint1] — 2026-02-27

### Added

**WS1 — Per-request lifecycle**
- `TesseraBlockManager` gains `req_blocks: DashMap<u64, Vec<BlockId>>` reverse ownership index.
- `release_request(req_id) -> u32` atomically frees all private blocks for a request.
- `tessera_request_releases_total` Prometheus counter.
- `PyBlockManager.release_request` exposed via PyO3.
- `tests/lifecycle.rs`: 4 integration tests for lifecycle semantics.
- ADR-0009: per-request lifecycle via reverse index.

**WS2 — Tiered LRU eviction**
- `BlockMeta` gains `last_touched: Arc<AtomicU64>` for epoch-based LRU ordering.
- `evict_one()` implements tiered LRU: orphaned (a) → cold unindexed (b) → cold indexed (c); tier d (shared) never evicted.
- `allocate` attempts `evict_one` before returning `OutOfBlocks`.
- `tessera_evictions_total{tier=a|b|c}` labeled Prometheus counter.
- `tests/eviction.rs`: 3 integration tests for eviction semantics.
- ADR-0010: tiered LRU eviction policy.

**WS3 — PyTorch reference oracle**
- `python/tessera/reference/absorbed_attention.py`: `reference_absorbed_mla` FP32-accumulation CPU oracle.
- `tests/test_reference_attention.py`: shape/dtype, hand-computed uniform-attention case, needle dominance, determinism, scale=0.
- ADR-0011: PyTorch reference oracle rationale.

**WS4 — E2E CPU integration test**
- `tests/integration/test_e2e_cpu.py`: two agents share a document; verifies dedup, share table, partial release, reference attention identity.

**WS5 — Proptest invariants**
- `crates/tessera-core/tests/proptest_invariants.rs`: 6 property-based invariants (round-trip, seal idempotency, dedup commutativity, CoW isolation, eviction safety, release_request fidelity).

**WS6 — CUDA CI**
- `.github/workflows/ci.yml`: new `build-cuda` job (Ubuntu, CUDA 12.4, `cargo build --workspace --features cuda`). Closes TD-001.

**WS7 — SegmentIndex async polish**
- Process-global `_GLOBAL_EXECUTOR` singleton (cpu_count()//2 workers).
- `asyncio.Semaphore` backpressure (default 16 concurrent HNSW queries).
- `tessera_segment_index_queue_depth` Prometheus gauge.
- `find_shared_prefix` now fires async Layer 2 HNSW lookup for exact-hash misses.

**WS8 — Cross-agent sharing benchmark**
- `benchmarks/sharing_bench.py`: redesigned with N=16 agents × unique system prompt + shared document; reports dedup rate and KV savings ratio.

**WS9 — GPU stubs**
- `tests/gpu/conftest.py`, `test_flash_mla_parity.py`, `test_flash_infer_parity.py`, `test_needle_haystack.py`: full GPU parity + 128K regression suite, auto-skips without CUDA.
- `benchmarks/vllm_compare.py`: Tessera vs vLLM block allocator throughput skeleton.
- `backends/flash_mla.py`: shape validation, FP8 scale forwarding, `build_page_tables()` helper.
- `backends/flash_infer.py`: wrapper memoised per config key (TD-006 closed).
- `backends/triton_fallback.py`: probes ordered candidate paths for `triton_mla_decode` (TD-007 closed).

**WS10 — On-device hash seam**
- `ContentHasher::hash_device` default method: `read_bytes + hash` fallback; future `CudaXxh3Hasher` overrides.
- `block_manager::hash_primary` uses `hasher.hash_device`.

**WS11 — Coverage gates + nightly bench**
- `.github/workflows/coverage.yml`: hard gates — Python `--cov-fail-under=75`; Rust ≥ 85% line (exit 1 via JSON parse).
- `.github/workflows/bench.yml`: nightly cron 04:00 UTC; Criterion baseline save/restore; regression > 20% → fail + open GitHub issue.

**WS12 — DEVLOG**
- `DEVLOG.md`: Sprint 0 full inventory, tech debt table TD-001 through TD-020, Sprint 1 resolution column.

**WS13 — Documentation**
- `docs/src/eviction.md`: eviction policy + tier diagram.
- `docs/src/lifecycle.md`: request lifecycle sequence diagram.
- `docs/src/testing.md`: proptest, reference oracle, and GPU-gated sections.
- `docs/src/SUMMARY.md`: linked new pages and ADRs 0009–0011.
- `ARCHITECTURE.md`: refreshed with Sprint 1 blocks and 11-ADR table.

### Fixed

- cudarc 0.12 API drift: `device_ptr()` returns `u64` directly; removed invalid dereference in `CudaBackend::primary_ptr` (TD-001).

[Unreleased]: https://github.com/angelnicolasc/tessera/compare/v0.3.0-sprint2...HEAD
[0.3.0-sprint2]: https://github.com/angelnicolasc/tessera/compare/v0.2.0-sprint1...v0.3.0-sprint2
[0.2.0-sprint1]: https://github.com/angelnicolasc/tessera/compare/v0.0.0...v0.2.0-sprint1
