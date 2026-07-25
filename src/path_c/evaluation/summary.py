"""Summaries recomputed exclusively from normalized episode rows."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .pairing import PAIRING_ROWS_SCHEMA_VERSION


def _mean_standard_error(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, standard_error


def _optional_mean_standard_error(
    values: Sequence[float | None],
) -> tuple[float | None, float | None]:
    applicable = [float(value) for value in values if value is not None]
    return (None, None) if not applicable else _mean_standard_error(applicable)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Recompute counts, means, and descriptive 95 percent intervals."""

    if not rows:
        raise ValueError("Evaluation summary requires raw episode rows.")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("schema_version") != PAIRING_ROWS_SCHEMA_VERSION:
            raise ValueError("Evaluation summary accepts only registered raw rows.")
        identity = (str(row["pairing_id"]), int(row["episode_index"]))
        if identity in identities:
            raise ValueError("Evaluation raw rows repeat a pairing episode.")
        identities.add(identity)
        grouped[identity[0]].append(row)
    pairing_summaries: dict[str, Any] = {}
    for pairing_id, group in sorted(grouped.items()):
        raw_mean, raw_se = _mean_standard_error([float(row["raw_return"]) for row in group])
        probe_mean, probe_se = _mean_standard_error([float(row["probe_count"]) for row in group])
        trigger_mean, trigger_se = _optional_mean_standard_error(
            [row["probe_trigger_rate"] for row in group]
        )
        budget_mean, budget_se = _optional_mean_standard_error(
            [row["probe_budget_usage_rate"] for row in group]
        )
        pairing_summaries[pairing_id] = {
            "episode_count": len(group),
            "mean_raw_return": raw_mean,
            "raw_return_standard_error": raw_se,
            "raw_return_descriptive_95_percent_interval": [raw_mean - 1.96 * raw_se, raw_mean + 1.96 * raw_se],
            "mean_probe_count": probe_mean,
            "probe_count_standard_error": probe_se,
            "safe_candidate_opportunity_count": sum(
                int(row["safe_candidate_opportunity_count"]) for row in group
            ),
            "mean_probe_trigger_rate": trigger_mean,
            "probe_trigger_rate_standard_error": trigger_se,
            "maximum_probe_budget": sum(
                int(row["maximum_probe_budget"]) for row in group
            ),
            "mean_probe_budget_usage_rate": budget_mean,
            "probe_budget_usage_rate_standard_error": budget_se,
            "correct_delivery_count": sum(int(row["correct_delivery_count"]) for row in group),
            "wrong_delivery_count": sum(int(row["wrong_delivery_count"]) for row in group),
            "indicator_cost": sum(float(row["indicator_cost"]) for row in group),
        }
    overall_mean, overall_se = _mean_standard_error([float(row["raw_return"]) for row in rows])
    return {
        "schema_version": "path_c_evaluation_summary_v1",
        "source": "recomputed_from_raw_episode_rows",
        "row_count": len(rows),
        "pairing_count": len(grouped),
        "overall_mean_raw_return": overall_mean,
        "overall_raw_return_standard_error": overall_se,
        "total_probe_count": sum(int(row["probe_count"]) for row in rows),
        "total_safe_candidate_opportunity_count": sum(
            int(row["safe_candidate_opportunity_count"]) for row in rows
        ),
        "total_maximum_probe_budget": sum(
            int(row["maximum_probe_budget"]) for row in rows
        ),
        "pairings": pairing_summaries,
    }
