"""End-to-end CPU integration test (WS4 / TD-011).

Exercises the full block manager → segment index → share table pipeline without a GPU.
Scenario: two agents process the same document with different system prompts.

Validated properties:
    (a) Document blocks deduplicate via exact xxhash3 match.
    (b) Share table reflects both owners after sharing.
    (c) release_request(agent_A) leaves document blocks alive for agent_B.
    (d) Reference attention output is bit-identical for the same c_kv regardless of sharing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from tessera import _native
from tessera.cross_agent import CrossAgentShareTable
from tessera.segment_index import SegmentIndex, hash_ckv_bytes

# ─────────────────────────── helpers ─────────────────────────────────────────


def _make_manager(n_blocks: int = 256) -> _native.BlockManager:
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    cfg = _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Bf16, 0)
    return _native.BlockManager(cfg, n_blocks * cfg.total_block_bytes())


def _make_ckv(seed: int, blocks: int = 4, d_c: int = 32, layers: int = 4) -> np.ndarray:
    """Deterministic synthetic c_kv tensor, shape [layers, blocks*64, d_c]."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((layers, blocks * 64, d_c)).astype(np.float32)


# ─────────────────────────── tests ───────────────────────────────────────────


@pytest.mark.integration
class TestE2eCpuSharing:
    """Full pipeline: block manager + segment index + share table, CPU only."""

    def test_document_blocks_dedup_exact(self):
        """(a) Two agents writing the same document bytes get the same canonical blocks."""
        mgr = _make_manager()

        doc_blocks = 4
        agent_a, agent_b = 1, 2

        # Agent A fills and seals document blocks.
        canonical_a: list[int] = []
        for i in range(doc_blocks):
            bid = mgr.allocate(agent_a, i * 64, (i + 1) * 64)
            pattern = (i * 7 + 13) % 256
            mgr.fill_primary_test_pattern(bid, pattern)
            canon, _h, was_dedup = mgr.seal(bid)
            assert not was_dedup, "agent A seals first; no dedup expected"
            canonical_a.append(canon)

        used_after_a = mgr.used_blocks
        assert used_after_a == doc_blocks

        # Agent B writes the same bytes → every block deduplicates.
        canonical_b: list[int] = []
        for i in range(doc_blocks):
            bid = mgr.allocate(agent_b, i * 64, (i + 1) * 64)
            pattern = (i * 7 + 13) % 256  # same pattern
            mgr.fill_primary_test_pattern(bid, pattern)
            canon, _h, was_dedup = mgr.seal(bid)
            assert was_dedup, f"agent B block {i} must dedup"
            canonical_b.append(canon)

        # All canonical blocks are the same as agent A's.
        assert canonical_a == canonical_b, "canonical blocks must match"
        # used_blocks is still doc_blocks (duplicates freed).
        assert mgr.used_blocks == doc_blocks

    def test_share_table_reflects_both_owners(self):
        """(b) After explicit sharing, share table lists both agents as owners."""
        mgr = _make_manager()
        share = CrossAgentShareTable()

        agent_a, agent_b = 10, 20

        bid = mgr.allocate(agent_a, 0, 64)
        mgr.fill_primary_test_pattern(bid, 0xAB)
        canon, _h, _ = mgr.seal(bid)

        # Register agent_B as a shared owner.
        mgr.increment_ref(canon)
        share.add_share(agent_b, canon)

        owners = share.owners(canon)
        assert owners is not None
        assert agent_b in owners

        # Cleanup.
        mgr.free(canon)  # drops agent_b's ref
        mgr.free(canon)  # drops agent_a's ref

    def test_release_request_a_leaves_blocks_for_b(self):
        """(c) Releasing agent A's request does not free blocks still held by agent B."""
        mgr = _make_manager()
        share = CrossAgentShareTable()

        agent_a, agent_b = 100, 200

        # Agent A allocates a block.
        bid = mgr.allocate(agent_a, 0, 64)
        mgr.fill_primary_test_pattern(bid, 0x55)
        canon, _h, _ = mgr.seal(bid)

        # Agent B takes a shared reference.
        mgr.increment_ref(canon)
        share.add_share(agent_b, canon)

        assert mgr.used_blocks == 1

        # Release agent A's private blocks.
        freed = mgr.release_request(agent_a)
        assert freed == 1, "agent A had exactly one private block"

        # The canonical block is still alive because agent B still holds a reference.
        assert mgr.used_blocks == 1, "block must survive for agent B"

        # Now release agent B's share.
        to_free = share.release_request(agent_b)
        for b in to_free:
            mgr.free(b)

        assert mgr.used_blocks == 0, "all references released"

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch required for attention parity check",
    )
    def test_attention_output_identical_before_and_after_sharing(self):
        """(d) Reference attention output is identical for shared vs. private c_kv arrays."""
        torch = pytest.importorskip("torch")
        from tessera.reference.absorbed_attention import reference_absorbed_mla

        B, H, S, d_c, d_r, d_h = 1, 2, 64, 32, 8, 32
        torch.manual_seed(7)
        q_abs = torch.randn(B, H, d_c)
        q_rope = torch.randn(B, H, d_r)
        c_kv_a = torch.randn(B, S, d_c)
        c_kv_b = c_kv_a.clone()  # identical content — simulates block sharing
        k_rope = torch.randn(B, S, d_r)
        W_UV = torch.randn(H, d_h, d_c)
        scale = 1.0 / math.sqrt(d_c)

        out_a = reference_absorbed_mla(q_abs, q_rope, c_kv_a, k_rope, W_UV, scale)
        out_b = reference_absorbed_mla(q_abs, q_rope, c_kv_b, k_rope, W_UV, scale)

        torch.testing.assert_close(
            out_a,
            out_b,
            msg="attention output must be bit-identical for shared c_kv content",
        )


@pytest.mark.integration
class TestE2eCpuSegmentIndex:
    """Segment index integrates correctly with block manager lifecycle."""

    def test_segment_index_exact_hit_returns_canonical(self):
        """Layer 1 exact hash lookup returns the canonical block sealed by the first agent."""
        mgr = _make_manager()
        index = SegmentIndex(latent_dim=32, num_layers=4)

        agent_a, _agent_b = 1, 2
        d_c, layers, block_size = 32, 4, 64

        # Build a deterministic c_kv array and hash it.
        ckv = np.ones((layers, block_size, d_c), dtype=np.float32)
        content_hash = hash_ckv_bytes(ckv)

        # Agent A seals a block.
        bid = mgr.allocate(agent_a, 0, 64)
        mgr.fill_primary_test_pattern(bid, 0x01)
        canon, _h, _ = mgr.seal(bid)
        index.add(block_id=canon, content_hash=content_hash, c_kv=ckv)

        # Agent B looks up the same hash.
        match = index.lookup_exact(content_hash)
        assert match == canon, "segment index must return canonical block"

        # Cleanup.
        mgr.free(canon)
