# ADR-0025 — Handle-based `DevicePtr`

**Status:** Accepted, 2026-06-11. Sprint 5.1 hardening.

## Context

The Sprint 0 `DevicePtr` carried a raw host/device address as `usize` plus the allocation
length:

```rust
pub struct DevicePtr {
    pub raw: usize,
    pub len: u64,
}
```

Every `DeviceBackend::locate(ptr)` reconstructed the `(region_index, offset)` pair by
walking the backend's region list and asking "does `ptr.raw` fall inside this region's
address range?". This was correct in practice but came with three structural problems:

1. **Aliasing.** Two regions whose host allocations happened to fall adjacent in the
   allocator's bins would have `[base_a, end_a)` abutting `[base_b, end_b)`. A pointer
   constructed from `end_a` (i.e. `base_b`) would be reported as belonging to region B —
   silently. No test could fail this; only chance allocator behaviour decides.
2. **No origin validation.** A handle travelling through the PyO3 boundary or arriving
   from a serialised payload could not be checked back against "did this backend
   produce me?". The backend just compared address ranges.
3. **Leakage.** The `raw: usize` field encouraged downstream code to do pointer arithmetic
   on it (and the Python plugin did exactly this — `ctypes.memmove(ptr, ...)` — opening
   the TOCTOU race documented in ADR-0027).

The hardening audit (`audit C4` + `H1`) made this load-bearing in the multi-tenant story:
the disk-backend cross-process cache and the PyO3 FP8 write path both depended on
`locate()` being correct.

## Decision

`DevicePtr` becomes an opaque handle:

```rust
pub struct DevicePtr {
    region: u32,
    offset: u64,
    len: u64,
}
```

* `region` is the index of the backing allocation inside the backend's region vec.
  Allocation is monotonic — region ids are not reused across allocations during a
  backend's lifetime — so a stale handle reliably either resolves to a present region or
  yields an `OutOfBounds` error.
* `offset` is the byte offset from the start of that region.
* `len` is the remaining addressable length from the handle.

All three backend impls (`CpuMockBackend`, `CudaBackend`, `DiskBackend`) drop their
range-comparison `locate()` and use `regions[ptr.region() as usize]` directly. Lookup
becomes O(1); aliasing becomes structurally impossible.

The one place that genuinely needs the raw device address — kernel-side GPU dispatch —
goes through a new explicit method on the trait:

```rust
fn device_address(&self, _ptr: DevicePtr) -> Option<usize> { None }
```

`CudaBackend` implements it by storing the cudarc device pointer in its `Region` struct
and resolving `device_addr + offset` at the call site. CPU mock and Disk return `None`.

## Consequences

* **PyO3 `fp8_scales_ptr` removed.** The method previously returned `ptr.raw` as `usize`
  for `ctypes.memmove`. Without a public `raw` field that pattern is no longer
  expressible; the replacement is the atomic `BlockManager.write_fp8_scales(block_id,
  scales)` documented in ADR-0027.
* **State Cache test** had compared two `DevicePtr`s via `.raw`; now compares the full
  handle (derived `PartialEq`).
* **CudaXxh3HasherStub** still works — it uses `read_bytes(ptr, len)` on the cached
  backend handle, which goes through `locate()` and is bounds-checked.
* New tests in `cpu_mock.rs::tests` exercise out-of-bounds and unknown-region handles to
  pin the invariant.

## Migration

The public field `DevicePtr::raw` is removed. Downstream Rust code that depended on it
should use `DevicePtr::region_offset()` / `DevicePtr::len()` for handle data and
`DeviceBackend::device_address(ptr)` only when a kernel call genuinely needs the raw
device pointer.
