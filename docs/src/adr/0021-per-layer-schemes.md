# ADR-0021 — Per-layer schemes in `MlaBlockConfig`

**Status:** Accepted, 2026-05-28.

## Context

DeepSeek-V3 (and Kimi-K2) used a single attention scheme across every layer. Tessera's
Sprint 0 `MlaBlockConfig` modelled this with a scalar `scheme: CompressionScheme`. V4
breaks the assumption: each transformer layer carries one of three schemes (CSA / HCA /
SWA), interleaved per the paper's `layer_pattern`.

Two failed alternatives:

* **Force every V4 deployment to construct one block manager per layer.** Multiplies
  bookkeeping cost by `num_layers`; share-table / segment-index semantics become
  per-layer too, which is wrong (sharing is across requests, not layers).
* **Treat the per-layer pattern as a kernel concern.** Pushes block layout decisions
  outside Tessera, which defeats the project's central thesis.

## Decision

`MlaBlockConfig` gains an optional `schemes_per_layer: Option<Arc<Vec<CompressionScheme>>>`.

* `None` → homogeneous: every layer uses `scheme`. This is the Sprint 0 path; every
  existing test, every V3 config, every MHA fallback continues to work.
* `Some(vec)` → per-layer: `vec[i]` is layer `i`'s scheme. Length must equal
  `num_layers` (enforced by `with_per_layer_schemes`). `scheme` itself becomes a
  back-compat hint (the first layer's scheme) for callers that haven't been taught the
  per-layer API.

`Arc<Vec<...>>` is the right shape because:

* Block manager + segment index + transports may all reference the same layout. `Arc`
  avoids cloning a 61-element vector per access.
* `scheme_for_layer(idx)` is O(1).
* Serialisation: `schemes_per_layer` is `#[serde(skip_serializing_if = "Option::is_none")]`
  so V3 TOMLs round-trip unchanged.

`primary_block_bytes()` and `rope_block_bytes()` are aware of the per-layer path —
they sum each layer's contribution honouring its scheme's specific accounting (V4
schemes use `bytes_per_token_per_layer`; non-V4 schemes use the legacy path).

A new helper `v4_block_size_lcm()` computes `lcm(k1, k2, …)` over the active V4 schemes;
the validator uses it to enforce that `block_size_tokens` aligns with every layer's
compression boundary.

## Consequences

* V4 hybrid layouts are fully representable in the block manager's type system.
* Pure-V3 callers see zero behaviour change.
* The Python side (`TesseraConfig`) gains a `v4: V4Config | None` field; when present
  it drives `to_native_config()` to invoke
  `MlaBlockConfig.with_per_layer_schemes(...)` instead of the homogeneous constructor.
* Cost of per-layer dispatch is a single vector index lookup — negligible on the hot
  path.

## Future work

* Per-layer dtype overrides (Sprint 6+) — V4 already varies precision per *region* via
  scheme variants; a hypothetical V5 might want per-layer dtype escalation too. The
  `Vec<CompressionScheme>` shape extends to a `Vec<LayerConfig>` if and when that
  arrives.
