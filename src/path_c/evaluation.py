"""Evaluation metrics for held-out zero-shot partner adaptation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .statistics import (
    equal_partner_mechanism_mean,
    hierarchical_partner_bootstrap,
    one_sided_interval,
)
from .types import EvaluationRow


def br_prox(ego_return: float, approximate_best_response: float, floor: float = 1.0) -> float:
    denominator = max(abs(float(approximate_best_response)), float(floor))
    return float(approximate_best_response - ego_return) / denominator


def summarize_evaluation_rows(
    rows: Sequence[EvaluationRow],
    *,
    bootstrap_replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("Evaluation rows cannot be empty.")
    returns = [row.raw_return for row in rows]
    ego_runs = [row.ego_run_id for row in rows]
    partner_runs = [row.partner_run_id for row in rows]
    mechanisms = [row.partner_mechanism for row in rows]
    bootstrap = hierarchical_partner_bootstrap(
        values=returns,
        ego_runs=ego_runs,
        partner_runs=partner_runs,
        mechanisms=mechanisms,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    interval = one_sided_interval(bootstrap, alpha)
    by_mechanism = {}
    for mechanism in sorted(set(mechanisms)):
        current = [row for row in rows if row.partner_mechanism == mechanism]
        by_mechanism[mechanism] = {
            "partner_run_count": len({row.partner_run_id for row in current}),
            "episode_count": len(current),
            "mean_raw_return": float(np.mean([row.raw_return for row in current])),
            "negative_transfer_rate": float(
                np.mean([row.negative_transfer for row in current])
            ),
            "mean_adaptation_enabled_steps": float(
                np.mean([row.adaptation_enabled_steps for row in current])
            ),
            "mean_support_score": float(
                np.mean([row.mean_support_score for row in current])
            ),
        }
    return {
        "partner_weighted_mean_raw_return": equal_partner_mechanism_mean(
            returns, partner_runs, mechanisms
        ),
        "bootstrap": interval,
        "negative_transfer_rate": float(
            np.mean([row.negative_transfer for row in rows])
        ),
        "mean_correct_deliveries": float(
            np.mean([row.correct_deliveries for row in rows])
        ),
        "mean_wrong_deliveries": float(
            np.mean([row.wrong_deliveries for row in rows])
        ),
        "mechanisms": by_mechanism,
    }


def paired_adaptation_gain(
    adapted: Sequence[EvaluationRow],
    base: Sequence[EvaluationRow],
) -> tuple[list[float], list[str], list[str], list[str]]:
    base_index = {
        (row.ego_run_id, row.partner_run_id, row.episode_index, row.episode_seed): row
        for row in base
    }
    gains: list[float] = []
    egos: list[str] = []
    partners: list[str] = []
    mechanisms: list[str] = []
    for row in adapted:
        key = (row.ego_run_id, row.partner_run_id, row.episode_index, row.episode_seed)
        if key not in base_index:
            raise ValueError(f"Base evaluation lacks matched episode: {key}")
        gains.append(row.raw_return - base_index[key].raw_return)
        egos.append(row.ego_run_id)
        partners.append(row.partner_run_id)
        mechanisms.append(row.partner_mechanism)
    return gains, egos, partners, mechanisms


__all__ = [
    "br_prox",
    "paired_adaptation_gain",
    "summarize_evaluation_rows",
]
