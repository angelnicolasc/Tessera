# Disk Backend (V4 on-disk KV tier)

DeepSeek-V4 paper §3.5.2 proposes on-disk persistence for the KV cache to eliminate
repeated prefill on shared-prefix requests. Compressed CSA / HCA entries are cheap to
persist (small per-token footprint); SWA is roughly 8× larger and the paper proposes
three trade-offs.

Tessera implements the storage layer as a sibling `DeviceBackend` impl, not a parallel
abstraction. The block manager treats disk addresses identically to GPU / CPU device
pointers; only the persistence semantics differ. See
[ADR-0024](adr/0024-disk-backend.md) for the design rationale.

## Caching strategies

| Strategy | Disk cost | Compute cost on prefix hit |
|---|---|---|
| **Full SWA Caching** | High — every SWA entry written | Zero recompute |
| **Periodic Checkpointing** | Tunable — snapshot every `c` tokens | Recompute tail since last checkpoint |
| **Zero SWA Caching** | None | Recompute `win × L` tokens to reconstruct SWA |

```rust
use tessera_core::{DiskBackend, SwaCachingStrategy};

let be = DiskBackend::new(
    std::path::PathBuf::from("/var/cache/tessera"),
    SwaCachingStrategy::Periodic { checkpoint_interval_tokens: 4096 },
)?;

// DeviceBackend methods work as expected:
let region = be.alloc_region(bytes, RegionKind::Primary)?;
be.write_bytes(region, &payload)?;
let recovered = be.read_bytes(region, payload.len())?;

// Strategy decision helper for integration code:
if be.should_persist_swa(token_pos) {
    be.write_bytes(swa_region, &swa_entry)?;
}

// Explicit flush before shutdown (also called automatically on Drop):
be.flush_all()?;
```

## Persistence semantics

Each allocated region maps to one backing file under the configured root directory.
Sprint 5's storage model is a mirrored `Vec<u8>` host buffer per region — reads / writes
hit the buffer, modified buffers flush to disk on `flush_all` or `Drop`. Cross-process
shared-prefix cache is a free side-effect: a second Tessera process pointing at the same
root recovers existing regions on reopen.

The mirror-buffer approach is correct but allocates per region. Production hardening
(TD-035, Sprint 6+) swaps to `memmap2`-backed zero-copy storage behind a `disk-mmap`
feature flag.

## Configuration

`models/deepseek_v4_*.toml` ships a `[disk_cache]` section:

```toml
[disk_cache]
enabled = false
root = "/var/cache/tessera"
swa_strategy = "periodic"
swa_checkpoint_interval_tokens = 4096
```

The Python `DiskCacheConfig` model validates these fields. When `enabled = false` (the
default) the block manager never touches disk.

## What's not yet wired

* Block-manager spillover — when in-memory eviction (ADR-0010) reaps a block, it could
  optionally migrate to the disk tier rather than disappearing. That orchestration is
  Sprint 6+ (TD-036).
* Compression at rest — CSA / HCA entries are already heavily compressed; zstd on disk
  would help the `Full` SWA strategy. Deferred.
* The mirror-buffer to mmap upgrade (TD-035, Sprint 6+).

## Tests

`crates/tessera-core/src/device/disk.rs::tests`:

- `alloc_write_read_roundtrip` — basic correctness.
- `persistence_across_backend_reopens` — close one backend, open another with the same
  root, recover bytes.
- `should_persist_swa_honours_strategy` — all three strategies make the right decisions
  on representative token positions.
- `strategy_as_str_is_stable` — label stability for metric / TOML round-trips.
- `memcpy_cross_region_copies_bytes` — cross-region copy doesn't share buffers.
