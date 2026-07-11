"""Shared numeric-domain validation for the Path C probe contract."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping


def normalized_probe_numeric_fields(
    probe: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, int | float | None]:
    normalized: dict[str, int | float | None] = {}
    for key in ("disagreement_threshold", "return_floor"):
        value = probe.get(key)
        if value is None:
            normalized[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{prefix}.{key} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{prefix}.{key} must be finite.")
        if key == "disagreement_threshold" and numeric < 0.0:
            raise ValueError(
                f"{prefix}.disagreement_threshold must be non-negative."
            )
        normalized[key] = numeric

    for key in ("min_selected_probes", "min_probe_opportunities"):
        value = probe.get(key)
        if value is None:
            normalized[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{prefix}.{key} must be a positive integer.")
        integer = int(value)
        if integer <= 0:
            raise ValueError(f"{prefix}.{key} must be a positive integer.")
        normalized[key] = integer

    for key in ("min_context_coverage", "min_action_coverage"):
        value = probe.get(key)
        if value is None:
            normalized[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{prefix}.{key} must lie in [0, 1].")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{prefix}.{key} must lie in [0, 1].")
        normalized[key] = numeric
    return normalized
