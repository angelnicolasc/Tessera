# ADR-0005 — `trait IndexBackend` with exactly two methods (plus `remove`)

**Status:** Accepted, 2026-05-21.

## Context

The HNSW segment index is one possible implementation of approximate `c_kv` matching. The
revision document (`Research/architecture/reversion-tessera02.md`) proposes replacing it
with an Engram + Lightning Indexer dual scheme that eliminates HNSW from the hot path.

We want this replacement to be a swap-in, not a re-architecture.

Two failed alternatives:

* **Abstract the whole index pipeline.** Including descriptor computation (the mean over
  `c_kv`). That mean is specific to MLA storage; DSA will use a different descriptor. Hiding
  it behind an interface forces a leaky abstraction.
* **Hard-code usearch.** Adding Engram later requires touching every call site.

## Decision

```rust
pub trait IndexBackend: Send + Sync {
    fn add(&self, block_id: u32, descriptor: &[f32]) -> anyhow::Result<()>;
    fn query(&self, descriptor: &[f32], k: usize) -> anyhow::Result<Vec<IndexMatch>>;
    fn remove(&self, block_id: u32) -> anyhow::Result<()>;
    fn len(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

Descriptor computation is **not** part of this trait. The Python `SegmentIndex`
orchestration layer owns it (`descriptor_from_ckv`). DSA replaces that function; the trait
stays.

Sprint 0 implementation: `UsearchIndex` (chosen over FAISS specifically because usearch
supports incremental `remove` without rebuild — a hard constraint for KV-cache eviction).

## Consequences

* Engram lands as `EngramIndex: IndexBackend` — one file, no call-site churn.
* `Box<dyn IndexBackend>` adds a virtual call per query. We measured: at HNSW's expected
  cost (>10 µs / query at typical ef), the dispatch overhead is statistical noise.
* Descriptor types are locked to `&[f32]` for now. If a backend needs `u8` quantised
  descriptors (Engram does), the trait grows a second method rather than changing this one.
