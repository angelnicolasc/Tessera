# Summary

[Introduction](intro.md)

# Foundations

- [The Problem](problem.md)
- [MLA Mathematics](mla_math.md)
- [Architecture](architecture.md)

# Components

- [Kernel Dispatch](kernel_dispatch.md)
- [Segment Index](segment_index.md)
- [Cross-Agent Sharing](cross_agent.md)
- [FP8 Storage](fp8.md)
- [State Cache (V4)](state_cache.md)
- [Disk Backend (V4)](disk_backend.md)

# Request Management

- [Request Lifecycle](lifecycle.md)
- [Eviction Policy](eviction.md)

# Distributed

- [Multi-GPU (Tensor Parallelism)](multi_gpu.md)

# Model Compliance

- [DeepSeek-V4 Compliance](v4_compliance.md)

# Engineering

- [Testing](testing.md)
- [Chaos Testing](chaos.md)
- [Benchmarks](benchmarks.md)
- [FAQ](faq.md)

# Architecture Decision Records

- [ADR-0001 — Block size 64](adr/0001-block-size-64.md)
- [ADR-0002 — Mount on FlashMLA; no custom WMMA](adr/0002-no-custom-wmma.md)
- [ADR-0003 — `trait DeviceBackend`](adr/0003-trait-device-backend.md)
- [ADR-0004 — `CompressionScheme` `#[non_exhaustive]`](adr/0004-compression-scheme-enum.md)
- [ADR-0005 — `trait IndexBackend`](adr/0005-index-backend-trait.md)
- [ADR-0006 — vLLM V1 allocator](adr/0006-vllm-v1-allocator.md)
- [ADR-0007 — FP8 calibration required](adr/0007-fp8-calibration-required.md)
- [ADR-0008 — HNSW off the hot path](adr/0008-hnsw-off-hot-path.md)
- [ADR-0009 — Per-request lifecycle](adr/0009-per-request-lifecycle.md)
- [ADR-0010 — Tiered LRU eviction](adr/0010-eviction-policy.md)
- [ADR-0011 — PyTorch reference oracle](adr/0011-pytorch-reference-oracle.md)
- [ADR-0012 — OTLP tracing (Python layer)](adr/0012-otlp-tracing-python-layer.md)
- [ADR-0013 — manylinux_2_28 distribution](adr/0013-manylinux-distribution.md)
- [ADR-0014 — Multi-rank architecture](adr/0014-multi-rank-architecture.md)
- [ADR-0015 — P2pCuda vs NCCL transport](adr/0015-p2p-vs-nccl-transport.md)
- [ADR-0016 — PD-disaggregation hook (superseded by ADR-0018)](adr/0016-pd-disaggregation-hook.md)
- [ADR-0017 — Rust OTLP tracing bridge](adr/0017-otel-rust-bridge.md)
- [ADR-0018 — Reserve-then-stream PD-disagg](adr/0018-reserve-then-stream-pd-disagg.md)
- [ADR-0019 — Latency injection + chaos testing](adr/0019-latency-injection-chaos-testing.md)
- [ADR-0020 — DeepSeek-V4 hybrid attention](adr/0020-v4-hybrid-attention.md)
- [ADR-0021 — Per-layer schemes](adr/0021-per-layer-schemes.md)
- [ADR-0022 — Mixed precision per region](adr/0022-mixed-precision-per-region.md)
- [ADR-0023 — State Cache](adr/0023-state-cache.md)
- [ADR-0024 — Disk backend (V4 on-disk KV)](adr/0024-disk-backend.md)
- [ADR-0025 — Handle-based DevicePtr](adr/0025-handle-based-device-ptr.md)
- [ADR-0026 — seal() byte verification](adr/0026-seal-byte-verification.md)
- [ADR-0027 — Atomic FP8 scale write](adr/0027-atomic-fp8-scale-write.md)
