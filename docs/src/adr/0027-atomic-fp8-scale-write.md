# ADR-0027 — Atomic FP8 scale write via `BlockManager::write_fp8_scales`

**Status:** Accepted, 2026-06-11. Sprint 5.1 hardening.

## Context

Sprint 2 (WS2) wired per-layer FP8 scale propagation through the vLLM plugin via a raw
device-pointer hand-off:

```python
ptr = self._manager.fp8_scales_ptr(block_id)   # Rust read-lock acquired and dropped
# ← TOCTOU window
ctypes.memmove(ptr, scales.ctypes.data, scales.nbytes)
```

`BlockManager::evict_one` force-frees Tier B / Tier C blocks that hold `ref_count == 1`
(ADR-0010). The contract was "callers must not retain pointers across calls". The Python
plugin retained the pointer across the FFI boundary into a `ctypes.memmove`. Between the
PyO3 `fp8_scales_ptr` returning and the memmove executing, an eviction on another thread
could free the block and the allocator could recycle its physical slot to another
request. The memmove then wrote tenant A's FP8 scales into tenant B's block — silently
corrupting attention output without raising an error (audit C3).

The DevicePtr handle refactor in ADR-0025 removed the public `raw: usize` field, so the
pattern is no longer expressible. We still need a way for the plugin to install scales.

## Decision

`TesseraBlockManager::write_fp8_scales(block_id, scales: &[f32])` performs the write
inside Rust under the block manager's read lock:

```rust
let blocks = self.blocks.read();
if !blocks.contains_key(&block_id) {
    return Err(TesseraError::UnknownBlock(block_id.raw()));
}
let ptr = self.fp8_scales_base?.offset(...);
let bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
self.backend.write_bytes(ptr, &bytes)?;
drop(blocks);
```

`evict_one` takes the **write** lock on the same map, so the read lock held by
`write_fp8_scales` is sufficient to pin the block against eviction for the duration of
the write.

The PyO3 binding exposes it as `BlockManager.write_fp8_scales(block_id, list[float])`.
The Python `_write_fp8_scales` helper in `vllm_plugin.py` now calls into this. No
`ctypes` imports remain in the plugin path.

The atomic write is well-defined for the BF16 path too: when the active config has no
FP8 scale region, the method returns `Ok(())` without touching memory, so callers can
use a single code path across BF16 and FP8.

## Consequences

* The PyO3 `fp8_scales_ptr` method is removed. Existing tests in
  `tests/test_pyo3_ergonomics.py` were rewritten to exercise `write_fp8_scales`.
* `_native.pyi` updated.
* Sprint 5.1 closes audit findings C3 and H2 with one change.

## Migration

* Python code calling `BlockManager.fp8_scales_ptr(block_id)` must migrate to
  `BlockManager.write_fp8_scales(block_id, scales)`. The pointer + `ctypes.memmove`
  pattern no longer compiles — `DevicePtr` does not expose `raw`.

## Out of scope

* Per-block FP8 scale read-back. Sprint 5.1 ships write only; no Python caller needs
  read.
* On-device FP8 calibration (TD-034) — orthogonal.
