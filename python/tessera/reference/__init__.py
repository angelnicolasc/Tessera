"""PyTorch reference implementations for MLA attention.

This package provides numerically-exact CPU reference implementations used as ground
truth for kernel parity tests (WS3 / ADR-0011). Imports are kept local to each
function so the module loads without PyTorch installed; tests that need the oracle
should ``pytest.importorskip("torch")`` before calling anything here.
"""

from tessera.reference.absorbed_attention import reference_absorbed_mla

__all__ = ["reference_absorbed_mla"]
