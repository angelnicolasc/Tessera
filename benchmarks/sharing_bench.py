"""Cross-agent sharing-rate benchmark (WS8). Runs on the CPU mock backend — no GPU required.

Simulates N=16 agents that each process a unique system prompt (~512 tokens, seeded random)
followed by the same shared document (~8192 tokens, same seed). Measures:
  - total_blocks_alloc: total blocks allocated across all agents
  - unique_blocks_after_dedup: canonical blocks surviving deduplication
  - sharing_rate: fraction of total seals that hit an existing canonical block
  - doc_blocks_shared: document blocks that are shared across agents (expected ≈ doc_blocks)
  - sys_blocks: unique system-prompt blocks per agent (expected ≈ n_agents × sys_blocks_per_agent)
  - kv_cache_mb_tessera: estimated KV cache MB with dedup
  - kv_cache_mb_naive: estimated KV cache MB without dedup (N × total blocks)

Expected outcome:
  - Document blocks dedup at close to (n_agents - 1) / n_agents ≈ 93.75% for N=16.
  - unique_blocks ≈ doc_blocks + n_agents × sys_blocks_per_agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from tessera import _native  # noqa: E402


def _blocks_for_tokens(n_tokens: int, block_size: int = 64) -> int:
    """Number of blocks needed for n_tokens (ceiling division)."""
    return (n_tokens + block_size - 1) // block_size


def run(
    n_agents: int = 16,
    doc_tokens: int = 8192,
    sys_tokens_per_agent: int = 512,
    block_size: int = 64,
) -> dict[str, float]:
    """Run the cross-agent sharing simulation and return summary statistics."""
    doc_blocks = _blocks_for_tokens(doc_tokens, block_size)
    sys_blocks_per_agent = _blocks_for_tokens(sys_tokens_per_agent, block_size)
    total_blocks_budget = n_agents * (doc_blocks + sys_blocks_per_agent) + 128

    # Tiny config: small latent_dim for CPU-mock speed (dimensions don't affect dedup logic).
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    config = _native.MlaBlockConfig(scheme, 4, block_size, _native.CkvDtype.Bf16, 0)
    block_bytes_mb = config.total_block_bytes() / (1024 * 1024)
    manager = _native.BlockManager(config, total_blocks_budget * config.total_block_bytes())

    total_seals    = 0
    dedup_hits     = 0
    canonical_set: set[int] = set()

    for agent_idx in range(n_agents):
        # ── System prompt blocks (unique per agent, seed = agent_idx) ──────────
        for blk in range(sys_blocks_per_agent):
            bid = manager.allocate(agent_idx, blk * block_size, (blk + 1) * block_size)
            # Unique per agent: mix agent_idx into the pattern byte.
            pattern = ((agent_idx * 31 + blk * 7) % 255) + 1
            manager.fill_primary_test_pattern(bid, pattern)
            canon, _h, was_dedup = manager.seal(bid)
            total_seals += 1
            if was_dedup:
                dedup_hits += 1
            else:
                canonical_set.add(canon)

        # ── Document blocks (same for all agents, seed = doc block index) ──────
        base_offset = sys_blocks_per_agent
        for blk in range(doc_blocks):
            token_start = (base_offset + blk) * block_size
            bid = manager.allocate(agent_idx, token_start, token_start + block_size)
            # Same pattern for every agent — intentional dedup target.
            pattern = (blk * 13 + 17) % 256
            manager.fill_primary_test_pattern(bid, pattern)
            canon, _h, was_dedup = manager.seal(bid)
            total_seals += 1
            if was_dedup:
                dedup_hits += 1
            else:
                canonical_set.add(canon)

    unique_blocks     = len(canonical_set)
    sharing_rate      = dedup_hits / total_seals if total_seals > 0 else 0.0
    kv_mb_tessera     = unique_blocks * block_bytes_mb
    kv_mb_naive       = total_seals * block_bytes_mb

    return {
        "n_agents":             float(n_agents),
        "doc_blocks":           float(doc_blocks),
        "sys_blocks_per_agent": float(sys_blocks_per_agent),
        "total_seals":          float(total_seals),
        "dedup_hits":           float(dedup_hits),
        "sharing_rate":         sharing_rate,
        "unique_blocks":        float(unique_blocks),
        "expected_unique":      float(doc_blocks + n_agents * sys_blocks_per_agent),
        "kv_cache_mb_tessera":  kv_mb_tessera,
        "kv_cache_mb_naive":    kv_mb_naive,
        "kv_savings_ratio":     kv_mb_naive / kv_mb_tessera if kv_mb_tessera > 0 else 0.0,
    }


def main() -> None:
    print("Cross-agent sharing benchmark (N=16, doc=8192 tokens, sys=512 tokens/agent)")
    print("=" * 70)
    stats = run()
    rows = [
        ("Agents", f"{stats['n_agents']:.0f}"),
        ("Doc blocks", f"{stats['doc_blocks']:.0f}"),
        ("Sys blocks/agent", f"{stats['sys_blocks_per_agent']:.0f}"),
        ("Total seals", f"{stats['total_seals']:.0f}"),
        ("Dedup hits", f"{stats['dedup_hits']:.0f}"),
        ("Sharing rate", f"{stats['sharing_rate']:.3f}"),
        ("Unique blocks (actual)", f"{stats['unique_blocks']:.0f}"),
        ("Unique blocks (expected)", f"{stats['expected_unique']:.0f}"),
        ("KV cache MB (Tessera)", f"{stats['kv_cache_mb_tessera']:.3f}"),
        ("KV cache MB (naive)", f"{stats['kv_cache_mb_naive']:.3f}"),
        ("KV savings ratio", f"{stats['kv_savings_ratio']:.2f}×"),
    ]
    max_k = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{max_k}}  {v}")

    expected_sharing = (stats["n_agents"] - 1) / stats["n_agents"]
    print()
    print(f"Expected sharing_rate ≥ {expected_sharing:.3f} "
          f"(document blocks dedupe across all but the first agent)")
    print(f"Expected unique_blocks ≈ {stats['expected_unique']:.0f} "
          f"= doc_blocks + n_agents × sys_blocks_per_agent")


if __name__ == "__main__":
    main()
