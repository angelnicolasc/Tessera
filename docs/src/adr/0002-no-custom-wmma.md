# ADR-0002 — Mount on FlashMLA; do not ship a custom WMMA kernel

**Status:** Accepted, 2026-05-21.

## Context

A production-quality absorbed MLA attention kernel requires: full WMMA / warp-level tiling,
async HBM loads (TMA on Hopper), two-level FP32 accumulation, and architecture-specific
tuning for Hopper / Ampere / Blackwell. FlashMLA does this; it is open-source (MIT),
production-proven, and achieves 3000 GB/s on H800.

We could fork FlashMLA. We don't have the engineering capacity to keep a fork performance-
competitive across NVIDIA generations, and the upstream maintainers ship on a fast cadence.

## Decision

Tessera **mounts** on upstream kernels via a thin dispatch layer:

* `FLASH_MLA` (SM ≥ 9.0)
* `FLASH_INFER` (Ampere+)
* `TRITON` (vLLM built-in fallback)
* `FLASH_ATTN4` (placeholder — see `KernelBackend` enum)

The dispatcher passes block pointers and page tables produced by Tessera's block manager to
the kernel; the kernel handles attention computation.

## Consequences

* Tessera's moat is the **block layer**, not kernel authorship. This is the differentiator
  competitors don't ship.
* Kernel updates are PRs of the form "bump the dependency". No fork to maintain.
* If a workload requires a kernel that upstream doesn't provide (e.g., DSA hierarchical
  attention pre-FA4), Tessera can add a backend variant without re-architecting the manager.
* We accept that Tessera's performance is bounded above by FlashMLA's. That is a feature: we
  inherit improvements automatically.
