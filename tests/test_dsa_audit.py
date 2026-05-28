"""WS4 — DSA audit tests (TD-017).

Verifies the Python-level DSA surface:
- CompressionScheme.dsa_hierarchical() constructs without error (the variant is accepted by PyO3).
- MlaBlockConfig with a DSA scheme raises ValueError (rejected at the config validation level).
"""

from __future__ import annotations

import pytest

from tessera import _native


def test_dsa_python_construction_succeeds() -> None:
    """CompressionScheme.dsa_hierarchical() must not raise — the variant is PyO3-exposed."""
    scheme = _native.CompressionScheme.dsa_hierarchical(128, 32, 128)
    assert scheme is not None
    r = repr(scheme)
    assert len(r) > 0  # Rust Debug representation is non-empty


def test_dsa_mla_config_raises_value_error() -> None:
    """MlaBlockConfig with a DSA scheme must raise ValueError (ADR-0004)."""
    scheme = _native.CompressionScheme.dsa_hierarchical(128, 32, 128)
    with pytest.raises(ValueError, match="(?i)DSA|not yet implemented|ADR-0004"):
        _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Bf16, 0)
