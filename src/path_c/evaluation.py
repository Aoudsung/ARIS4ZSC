"""Small evaluation primitives shared by DEPI applications."""

from __future__ import annotations

def br_prox(ego_return: float, approximate_best_response: float, floor: float = 1.0) -> float:
    denominator = max(abs(float(approximate_best_response)), float(floor))
    return float(approximate_best_response - ego_return) / denominator


__all__ = ["br_prox"]
