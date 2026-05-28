# ADR-0024 — `DiskBackend`: on-disk KV cache tier with three SWA strategies

**Status:** Accepted, 2026-05-28.

## Context

V4 paper §3.5.2 introduces persistent on-disk KV storage:

> "When serving DeepSeek-V4, we leverage an on-disk KV cache storage mechanism to
> eliminate repeated prefilling for shared-prefix requests."

For CSA / HCA, the paper persists every compressed entry — the per-token cost is small
enough that disk writes don't dominate. For SWA the volume is ~8× larger; the paper
proposes three trade-offs and lets operators choose:

| Strategy | Disk cost | Compute cost on hit |
|---|---|---|
| **Full SWA Caching** | High (every SWA entry written) | Zero recompute |
| **Periodic Checkpointing** | Tunable (snapshot every `c` tokens) | Recompute tail since last checkpoint |
| **Zero SWA Caching** | None | Recompute `win × L` tokens to reconstruct SWA |

Tessera needs to expose this choice without forcing operators to reinvent the storage
layer. Existing options considered:

* **Layer it above `BlockManager`.** Forces every caller to implement persistence; the
  storage choice leaks into application code. Rejected.
* **Build it as a `KvCacheStore` trait separate from `DeviceBackend`.** Two parallel
  storage abstractions to teach the block manager. Rejected for code-path complexity.
* **Make it a `DeviceBackend` impl.** Disk addresses look like device pointers to the
  block manager; the existing alloc / read / write / memcpy contract maps cleanly to
  filesystem semantics via a mirrored host buffer.

## Decision

`DiskBackend: DeviceBackend` (Sprint 5) implements all five `DeviceBackend` methods over
a filesystem-backed storage layer:

* `alloc_region(bytes, kind)` → opens / creates one backing file per region under the
  configured `root`; mirrors content in a host buffer (`Vec<u8>`) for fast access.
* `memcpy` / `read_bytes` / `write_bytes` / `fill_pattern` → operate on the host
  buffer; modified buffers flush to disk on `flush_all` or `Drop`.
* Persistence semantics: regions are recovered from existing files on backend reopen,
  enabling **cross-process shared-prefix cache** out of the box.

`SwaCachingStrategy` enum:

```rust
pub enum SwaCachingStrategy {
    Full,
    Periodic { checkpoint_interval_tokens: u32 },
    Zero,
}
```

`DiskBackend::should_persist_swa(token_pos) -> bool` decides per-token persistence per
the active strategy. Integration code (Sprint 6+, when the V4 prefill loop is wired)
consults this helper before invoking `write_bytes` on the SWA region.

Sprint 5 ships:

* The trait impl + filesystem persistence (memcpy / read / write / fill via mirrored
  host buffer).
* All three strategies with deterministic `should_persist_swa` semantics.
* Tests: alloc/write/read roundtrip, persistence across reopens, strategy decisions.

What Sprint 5 does **not** ship:

* `memmap2`-backed zero-copy storage. The mirrored-buffer approach is correct for
  Sprint 5 (tests + CPU-only validation); production deployments will swap to mmap
  behind a `disk-mmap` feature in Sprint 6+.
* Eviction policy at the disk tier. The block manager's tiered LRU (ADR-0010) governs
  in-memory blocks; what falls out of memory could spill to disk via `DiskBackend`,
  but that policy wiring is Sprint 6+.
* Compression at rest. CSA / HCA entries are already heavily compressed; zstd over the
  disk file would help SWA storage under `Full` strategy. Deferred.

## Consequences

* The `DeviceBackend` abstraction proves its design value for the third time (after CPU
  mock and CUDA): adding disk as a sibling needs no changes to the block manager.
* Operators choose persistence strategy via `tessera.toml`'s `[disk_cache]` section.
* Cross-process sharing becomes a free side-effect — two Tessera processes pointing at
  the same `root` see each other's compressed entries.
* The disk tier composes with the cross-agent share table from Sprint 0: agents that
  load the same document during prefill can hit the shared disk cache before the
  in-memory share table picks them up.
