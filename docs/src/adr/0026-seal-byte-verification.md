# ADR-0026 — `seal()` byte verification against xxh3 collisions

**Status:** Accepted, 2026-06-11. Sprint 5.1 hardening.

## Context

`TesseraBlockManager::seal()` uses xxh3-64 content addressing. On a match in the
`content_index`, the candidate block was previously dedup-collapsed into the canonical
block:

```rust
if let Some(existing) = self.content_index.get(&hash) {
    let canonical = *existing.value();
    self.increment_ref(canonical)?;       // hand the canonical to the caller
    self.free_block_internal(block_id)?;  // discard the candidate
    return Ok(SealOutcome { canonical_block: canonical, ... });
}
```

This trusts the hash. xxh3-64 is non-cryptographic and **not collision-resistant against
adversaries** — crafted-collision attacks are publicly documented. The hardening audit
(`audit C1`) traced the threat:

1. A multi-tenant deployment where two requests can submit content that hashes through
   the block manager.
2. Tenant A populates a block with content `X`. Hash `H = xxh3(X)`.
3. Tenant B crafts content `Y ≠ X` such that `xxh3(Y) = H` (the attack — non-trivial,
   but bounded by 2^64 work in the worst case and substantially less for known
   weaknesses in xxh3).
4. Tenant B calls `seal()` on `Y`. The block manager sees `H` in `content_index`,
   increments the ref-count of A's block, and frees B's candidate. **B now holds a
   reference to A's block.** Subsequent attention reads return A's `c_kv`.

Even without an adversary the 2^32 birthday bound is reachable in long-running
deployments with many blocks; organic collisions silently corrupt content.

## Decision

`seal()` byte-verifies the candidate against the canonical block before authorising
dedup. The hash continues to fast-path the lookup; the **bytes** authorise the dedup.

```rust
if let Some(existing) = self.content_index.get(&hash) {
    let canonical = *existing.value();
    if self.blocks_have_equal_primary(block_id, canonical)? {
        // genuine dedup
    } else {
        // hash collision — install fresh; bump tessera_dedup_hash_collisions_total
    }
}
```

A second hardening lands in the same change: the prior `get` / `insert` sequence had a
race where two threads with identical content would both miss the `get` and both
`insert`, silently violating the dedup invariant (audit H3). The new path uses
`DashMap::entry().or_insert()` for the install, and a hash-match on the loser of the
race triggers the same byte-equality check.

A new Prometheus counter — `tessera_dedup_hash_collisions_total` — tracks how often the
byte check rejects a hash match. Non-zero values indicate either organic collisions
(rare, expected at scale) or adversarial probing.

## Consequences

* Dedup cost on a hit grows by one extra `read_bytes` of the canonical block (`dtoh`
  copy on CUDA). For the workloads Tessera targets, dedup hits are ≪ 1% of seals; the
  added cost is negligible. On the no-hit path nothing changes.
* The `tessera_dedup_hash_collisions_total` metric is a **security signal**. Operators
  should alert on spikes.
* Future work: replace xxh3 with BLAKE3 (~3 GB/s/core, cryptographic) for content
  addressing. The byte-verify check stays regardless — defence in depth.

## Out of scope

* On-device hash kernel (TD-002) — that work is orthogonal. Whether the hash is computed
  on host or device, the byte-verify check is the authority.
* Disk-tier dedup — the disk backend stores per-region content but does not dedup
  across regions today.
