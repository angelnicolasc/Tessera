"""FP8 calibration unit tests."""

from __future__ import annotations

from tessera.fp8_calibrate import FP8_E4M3_MAX, compute_per_layer_scales


def test_scale_is_max_over_fp8_max() -> None:
    samples = {0: [10.0, 20.0, 5.0], 1: [200.0, 300.0]}
    scales = compute_per_layer_scales(samples)
    assert scales[0] == 20.0 / FP8_E4M3_MAX
    assert scales[1] == 300.0 / FP8_E4M3_MAX


def test_empty_layer_is_dropped() -> None:
    scales = compute_per_layer_scales({0: [], 1: [1.0]})
    assert 0 not in scales
    assert 1 in scales


def test_explicit_fp8_max() -> None:
    scales = compute_per_layer_scales({0: [100.0]}, fp8_max=50.0)
    assert scales[0] == 2.0
