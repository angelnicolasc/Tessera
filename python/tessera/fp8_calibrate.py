"""FP8 calibration for ``c_kv`` storage.

Calibration is required — not optional — because uncalibrated FP8 with FlashMLA produces a
documented long-context regression (vLLM, April 2026: 128K needle-in-haystack recall fell
from 91% to 13% on the FP8 path with no per-layer scaling). ADR-0007.

This module ships only the calibration harness. Loading scale factors and applying them at
seal-time is the block manager's responsibility.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


#: Max representable value of FP8 E4M3, used to scale per-layer absolute maxima down to the
#: representable range.
FP8_E4M3_MAX: float = 448.0


def compute_per_layer_scales(
    layer_maxabs: dict[int, list[float]],
    *,
    fp8_max: float = FP8_E4M3_MAX,
) -> dict[int, float]:
    """Convert per-layer max-absolute-value samples into per-layer scale factors.

    Args:
        layer_maxabs: ``{layer_idx: [max_abs_per_sample, ...]}`` collected during forward
            passes over the calibration corpus.
        fp8_max: the FP8 format's max representable value. Default is E4M3's 448.

    Returns:
        ``{layer_idx: scale_factor}`` where the scale is ``max(samples) / fp8_max`` per layer.
    """
    return {
        layer_idx: max(samples) / fp8_max
        for layer_idx, samples in layer_maxabs.items()
        if samples
    }


def save_scales(scales: dict[int, float], path: str | Path) -> None:
    """Persist per-layer FP8 scale factors to JSON.

    Args:
        scales: ``{layer_idx: scale_factor}`` produced by :func:`compute_per_layer_scales`.
        path: destination file; parent directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in sorted(scales.items())}, indent=2))


def load_scales(path: str | Path) -> dict[int, float]:
    """Load per-layer FP8 scale factors from JSON produced by :func:`save_scales`.

    Args:
        path: JSON file written by :func:`save_scales`.

    Returns:
        ``{layer_idx: scale_factor}`` with int keys.
    """
    raw: dict[str, float] = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in raw.items()}


@contextmanager
def capture_ckv_maxabs(
    model: Any,
    layer_maxabs: dict[int, list[float]],
) -> Iterator[None]:
    """Context manager that registers forward hooks on every layer's ``c_kv`` write path and
    records per-layer max-absolute-value samples into ``layer_maxabs``.

    The exact hook attachment point is model-dependent; the function below is a model-
    agnostic skeleton showing the contract. Concrete integrations override ``_install_hook``
    in a subclass.

    Args:
        model: a transformer model exposing ``layers`` as an iterable of modules with a
            named ``c_kv`` activation. The default implementation looks for an attribute
            called ``ckv_for_calibration`` and ignores the layer otherwise.
        layer_maxabs: mutable dict populated with samples in-place.
    """
    handles: list[Any] = []
    layers: Iterable[Any] = getattr(model, "layers", [])
    for idx, layer in enumerate(layers):
        capture_target = getattr(layer, "ckv_for_calibration", None)
        if capture_target is None:
            continue

        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = idx) -> None:
            value = output
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "abs"):
                value = value.abs()
            if hasattr(value, "max"):
                value = value.max()
            if hasattr(value, "item"):
                layer_maxabs[idx].append(float(value.item()))
            else:
                layer_maxabs[idx].append(float(value))

        handles.append(capture_target.register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def calibrate_ckv_fp8_scales(
    model: Any,
    calibration_dataset: Iterable[str],
    *,
    num_samples: int = 512,
    tokenize: Any | None = None,
) -> dict[int, float]:
    """End-to-end harness. Runs forward passes over ``calibration_dataset`` and returns the
    per-layer scale factors.

    The function deliberately avoids importing ``torch`` at module level — it is only required
    when this harness is actually invoked, and CI environments without torch can still type-
    check the rest of the package.

    Args:
        model: transformer with a callable ``forward`` and (optionally) a ``tokenize`` method.
        calibration_dataset: iterable of input strings.
        num_samples: cap on the number of inputs processed.
        tokenize: optional explicit tokeniser; falls back to ``model.tokenize``.

    Returns:
        ``{layer_idx: scale_factor}``.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        msg = "torch is required for FP8 calibration. `pip install tessera[torch]`"
        raise RuntimeError(msg) from exc

    layer_maxabs: dict[int, list[float]] = defaultdict(list)
    tokenize = tokenize or getattr(model, "tokenize", None)
    if tokenize is None:
        msg = "tokenize callable not provided and model has no `tokenize` method"
        raise TypeError(msg)

    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad(), capture_ckv_maxabs(model, layer_maxabs):
        for i, text in enumerate(calibration_dataset):
            if i >= num_samples:
                break
            tokens = tokenize(text)
            model.forward(tokens) if hasattr(model, "forward") else model(tokens)

    return compute_per_layer_scales(layer_maxabs)
