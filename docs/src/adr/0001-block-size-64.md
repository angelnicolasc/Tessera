# ADR-0001 — Block size is 64 tokens

**Status:** Accepted, 2026-05-21.

## Context

PagedAttention's default block size is 16 tokens, tuned for MHA payloads. FlashMLA's
decoding kernel is built around a 64-token paged block size; the kernel hits 3000 GB/s
HBM bandwidth on H800 at that granularity and is not tuned for other values.

If Tessera chose any other block size, the kernel would either:

* Pay a cross-block boundary cost on every decode step, eroding the bandwidth gain that
  motivates FlashMLA in the first place, or
* Require Tessera to ship a custom kernel — directly contradicting [ADR-0002](0002-no-custom-wmma.md).

## Decision

`block_size_tokens` is fixed at **64** for any `CompressionScheme::MlaLatent`. The constraint
is enforced by `MlaBlockConfig::new` and surfaces as `TesseraError::InvalidConfig`.

MHA fallback (`CompressionScheme::MhaFull`) relaxes the constraint — its block size is
chosen by the deployer.

## Consequences

* Tessera and FlashMLA share a single block contract; no per-call resizing or copying.
* Configs that set the wrong block size fail loudly at construction, not at first decode.
* If FlashMLA ever changes its native size, this ADR must be superseded; the constant lives
  in `crates/tessera-core/src/config.rs::REQUIRED_BLOCK_SIZE_TOKENS` and a single search hits
  every site.
