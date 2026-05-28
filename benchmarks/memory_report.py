"""Print the MHA vs MLA BF16 vs MLA FP8 memory comparison from the playbook §11.1.

Run: ``python -m benchmarks.memory_report``. No GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from tessera.config import TesseraConfig  # noqa: E402


def report(config: TesseraConfig) -> None:
    m = config.model
    b = config.block
    print(
        f"\nModel: {m.name}  (L={m.num_layers}, H={m.num_heads}, d_h={m.head_dim}, "
        f"d_c={m.latent_dim}, d_r={m.rope_key_dim})"
    )
    print(f"Block: size_tokens={b.block_size_tokens}, ckv_dtype={b.ckv_dtype}")
    print(f"Compression vs MHA BF16: {config.compression_ratio_vs_mha_bf16():.1f}x\n")

    mha_bpt = 2 * m.num_heads * m.head_dim * m.num_layers * 2
    mla_bf16_bpt = (m.latent_dim + m.rope_key_dim) * m.num_layers * 2
    mla_fp8_bpt = m.latent_dim * m.num_layers * 1 + m.rope_key_dim * m.num_layers * 2

    print(
        f"{'Context':>8}  {'MHA BF16':>12}  {'MLA BF16':>12}  {'MLA FP8':>12}  "
        f"{'Ratio (MLA/MHA)':>18}"
    )
    for ctx in (8_192, 32_768, 131_072, 524_288, 1_048_576):
        mha_gb = mha_bpt * ctx / 1e9
        mla_bf_gb = mla_bf16_bpt * ctx / 1e9
        mla_fp_gb = mla_fp8_bpt * ctx / 1e9
        ratio = mha_gb / mla_bf_gb if mla_bf_gb > 0 else float("inf")
        print(
            f"{ctx // 1024:>6}K  {mha_gb:>10.1f} GB  {mla_bf_gb:>10.2f} GB  "
            f"{mla_fp_gb:>10.2f} GB  {ratio:>16.1f}x"
        )


def main() -> None:
    deepseek = TesseraConfig.from_toml(ROOT / "models" / "deepseek_v3.toml")
    report(deepseek)


if __name__ == "__main__":
    main()
