# ADR-0006 — Target vLLM V1 `BlockAllocator`, not V0 `BlockSpaceManager`

**Status:** Accepted, 2026-05-21.

## Context

vLLM 0.6 (mid-2025) replaced the V0 `BlockSpaceManager` API with a cleaner V1
`BlockAllocator` protocol. V1 is the default in every vLLM release since. V0 still works
but is on a slow deprecation path.

We could target both. We don't.

## Decision

`TesseraBlockAllocator` implements the V1 protocol only:

```text
allocate_mutable_block   allocate_immutable_blocks   free
get_num_free_blocks      get_num_total_blocks
```

Plus Tessera-specific extension hooks (`post_prefill_seal`, `find_shared_prefix`) called by
Tessera-aware integration code.

Registered via `pyproject.toml` entry point: `vllm.block_allocator = tessera.vllm_plugin:TesseraBlockAllocator`.

## Consequences

* Users on vLLM ≥ 0.6 get a one-line install: `pip install tessera`. vLLM's plugin discovery
  via `importlib.metadata.entry_points` does the rest.
* V0 users get an explicit `RuntimeError` pointing at the upgrade path.
* The integration test that verifies protocol shape (`tests/test_vllm_plugin_protocol.py`)
  works without vLLM installed — it introspects method signatures only, so CI doesn't
  need to pull in the full vLLM dependency tree.
