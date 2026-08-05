"""Run-level inference utilities; episodes are never treated as partner samples."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def equal_partner_mechanism_mean(
    values: Sequence[float],
    partner_runs: Sequence[str],
    mechanisms: Sequence[str],
) -> float:
    if not (len(values) == len(partner_runs) == len(mechanisms)):
        raise ValueError("Values, partner runs, and mechanisms do not align.")
    rows = np.asarray(values, dtype=np.float64)
    run_ids = np.asarray(partner_runs, dtype=object)
    mechanism_ids = np.asarray(mechanisms, dtype=object)
    mechanism_means = []
    for mechanism in sorted(set(mechanism_ids.tolist())):
        run_means = []
        for run in sorted(set(run_ids[mechanism_ids == mechanism].tolist())):
            mask = (mechanism_ids == mechanism) & (run_ids == run)
            run_means.append(float(np.mean(rows[mask])))
        mechanism_means.append(float(np.mean(run_means)))
    return float(np.mean(mechanism_means))


def hierarchical_partner_bootstrap(
    *,
    values: Sequence[float],
    ego_runs: Sequence[str],
    partner_runs: Sequence[str],
    mechanisms: Sequence[str],
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample ego runs and partner runs within fixed mechanism strata."""

    if not (
        len(values) == len(ego_runs) == len(partner_runs) == len(mechanisms)
    ):
        raise ValueError("Bootstrap columns do not align.")
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    values_array = np.asarray(values, dtype=np.float64)
    ego_array = np.asarray(ego_runs, dtype=object)
    partner_array = np.asarray(partner_runs, dtype=object)
    mechanism_array = np.asarray(mechanisms, dtype=object)
    unique_egos = np.asarray(sorted(set(ego_array.tolist())), dtype=object)
    unique_mechanisms = sorted(set(mechanism_array.tolist()))
    runs_by_mechanism = {
        mechanism: np.asarray(
            sorted(set(partner_array[mechanism_array == mechanism].tolist())),
            dtype=object,
        )
        for mechanism in unique_mechanisms
    }
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(replicates), dtype=np.float64)
    for replica in range(int(replicates)):
        sampled_egos = rng.choice(unique_egos, size=len(unique_egos), replace=True)
        mechanism_means = []
        for mechanism in unique_mechanisms:
            source_runs = runs_by_mechanism[mechanism]
            sampled_runs = rng.choice(source_runs, size=len(source_runs), replace=True)
            run_means = []
            for partner_run in sampled_runs:
                ego_means = []
                for ego_run in sampled_egos:
                    mask = (
                        (ego_array == ego_run)
                        & (partner_array == partner_run)
                        & (mechanism_array == mechanism)
                    )
                    if np.any(mask):
                        ego_means.append(float(np.mean(values_array[mask])))
                if ego_means:
                    run_means.append(float(np.mean(ego_means)))
            if run_means:
                mechanism_means.append(float(np.mean(run_means)))
        output[replica] = (
            float(np.mean(mechanism_means)) if mechanism_means else np.nan
        )
    return output[np.isfinite(output)]


def one_sided_interval(samples: Any, alpha: float) -> Mapping[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty bootstrap distribution.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    return {
        "mean": float(np.mean(values)),
        "lcb": float(np.quantile(values, alpha, method="lower")),
        "ucb": float(np.quantile(values, 1.0 - alpha, method="higher")),
    }


__all__ = [
    "equal_partner_mechanism_mean",
    "hierarchical_partner_bootstrap",
    "one_sided_interval",
]
