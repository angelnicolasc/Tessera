# ADR-0016 — Prefill/Decode disaggregation: `transfer_request_to_rank`

**Status:** Accepted 2026-05-21. **Superseded by**
[ADR-0018](0018-reserve-then-stream-pd-disagg.md) (Sprint 4, 2026-05-28). This ADR
introduced the push-mode hook in Sprint 3; ADR-0018 replaced its semantics with the
transactional reserve-then-stream protocol (prepare → stream → commit with full rollback
on any mid-stream failure). The method name `transfer_request_to_rank` is preserved; its
contract changed. Retained as the historical record of the push-mode design.

## Context

vLLM 0.6+ is adopting prefill/decode (PD) disaggregation as a first-class deployment
pattern: dedicated "prefill pools" of ranks run the compute-bound first-token computation,
then hand the resulting KV cache off to "decode pools" optimised for memory bandwidth. The
transfer of the KV cache between pools is the central engineering problem — and Tessera's
block manager owns exactly that state.

Two options were considered:

* **Make the application layer responsible.** Each integrator implements `export_payload`
  → network send → `import_payload` themselves. Forces every downstream user to reinvent
  the protocol; couples them to transport details (NVLink vs NCCL bytes-on-the-wire).
* **Bake a `transfer_blocks_to_rank` primitive into the block manager.** Same shape for
  every transport; tests cover correctness once.

Option 2 wins because PD is becoming a standard feature, not a niche.

## Decision

`TesseraBlockManager::transfer_request_to_rank(req_id, target, transport) -> Result<u32>`
implements **push-mode** ownership transfer:

1. Snapshot the `Vec<BlockId>` owned by `req_id` (atomic via the `req_blocks` DashMap entry).
2. For each block:
   * `export_payload(block_id)` → host-owned `BlockPayload { c_kv, k_rope, fp8_scales }`.
   * `transport.push_block(target, payload)` → destination's block manager calls
     `import_payload` and returns the new local `BlockId`.
3. On full success, `release_request(req_id)` locally to free source-side storage.
4. Records `tessera_pd_disagg_transfers_total{src, dst}` per transferred block.

The destination block manager's `import_payload`:

* Validates payload size matches `primary_block_bytes()` (caller-side debugging fast path).
* Allocates a new block under the destination's `accept_req_id`.
* Writes c_kv, k_rope, and FP8 scales via the device backend's `write_bytes`.

## Consequences

* The vLLM plugin's `TesseraBlockAllocator.transfer_request_to_rank(req_id, target)` is a
  one-line dispatch — no protocol reinvention.
* Sprint 3 doesn't ship rollback-on-target-OOM (TD-024). If `push_block` fails mid-loop,
  the source retains the partial state and the caller can retry. Sprint 4 will add
  transactional semantics with two-phase commit on the transport layer.
* `BlockPayload` currently allocates `Vec<u8>` on the heap per transfer (TD-026). For
  production NVLink P2P, the optimal path is to ship a CUDA IPC handle that the destination
  imports in place — zero-copy. The trait surface stays the same; the optimisation lives
  inside `P2pCudaTransport::push_block` once the cloud-burst session validates it.
* The `accept_req_id` in `BlockManagerPeerAdapter` is currently a constant
  (`DEFAULT_PUSH_ACCEPT_REQ_ID = 1`). Real deployments will route through the destination's
  vLLM scheduler so each incoming transfer maps to the real request.

## Test coverage (Sprint 3)

* `crates/tessera-core/tests/pd_disagg.rs` — 3 Rust cases (transfer all blocks, unknown
  request returns 0, payload roundtrip preserves bytes).
* `tests/multirank/test_e2e_tp4.py` — Python E2E across simulated TP=4 with MockTransport.
* Multi-process variant: `tests/multirank/test_mp_isolation.py` exercises the rank-aware
  constructor under spawn (CPU only).
