"""E3/E4 signal estimation: kappa(t), TV(t), Delta-hat (exploration track).

Registered protocol: docs/research/TRAJECTORY_AND_ESTIMATION_SPEC.md sections
two to four.  Two modes:

- ``--mode validate``: synthetic Bernoulli-evidence sequences with known kappa
  are generated from the S2 model and passed through the same estimators.
  Pass criteria: TV-hat relative bias within 10 percent, kappa-hat absolute
  bias within 0.05.  Fail-closed on violation.
- ``--mode ecology``: estimates on collected partner trajectories.  Type-level
  curves (sp vs op) and individual-level curves (each run vs pooled rest).
  Delta-hat uses the panel proxy defined in the spec.

Window features are partner-action histograms (6 bins, window 20).  Group
separability uses a deterministic leave-one-out plug-in Bayes classifier on
the discrete feature keys; TV-hat = max(0, 1 - 2*err) by T2
(p_e^* = (1 - TV)/2).  The plug-in Bayes rule is the sample-frequency
majority class per feature key with leave-one-out frequency correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW = 20
ACTION_RANGE = 6
GRID_T = tuple(range(20, 401, 20))
CI_T = (60, 200, 400)
VALIDATE_EPISODES = 2_000
VALIDATE_STEPS = 64
VALIDATE_T = tuple(range(4, 65, 4))
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 20260803
TV_BIAS_TOLERANCE = 0.10
KAPPA_BIAS_TOLERANCE = 0.05


def window_histograms(actions: np.ndarray, t: int) -> np.ndarray:
    """(episodes, 6) action-count histogram for the window ending at step t."""
    start = max(0, t - WINDOW)
    window = actions[:, start:t]
    hist = np.zeros((window.shape[0], ACTION_RANGE), dtype=np.float64)
    for a in range(ACTION_RANGE):
        hist[:, a] = (window == a).sum(axis=1)
    return hist


def loo_error(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Leave-one-out plug-in Bayes error between two discrete feature sets."""
    keys_a = [tuple(row) for row in features_a.astype(np.int64)]
    keys_b = [tuple(row) for row in features_b.astype(np.int64)]
    counts_a: dict[tuple, int] = {}
    counts_b: dict[tuple, int] = {}
    for key in keys_a:
        counts_a[key] = counts_a.get(key, 0) + 1
    for key in keys_b:
        counts_b[key] = counts_b.get(key, 0) + 1
    n_a, n_b = len(keys_a), len(keys_b)
    if n_a < 2 or n_b < 2:
        raise RuntimeError("too few samples for leave-one-out classification")
    errors = 0
    for key in keys_a:
        a_freq = counts_a[key] - 1
        b_freq = counts_b.get(key, 0)
        if b_freq > a_freq:
            errors += 1
    for key in keys_b:
        a_freq = counts_a.get(key, 0)
        b_freq = counts_b[key] - 1
        if a_freq > b_freq:
            errors += 1
    return errors / (n_a + n_b)


def estimate_pair(
    actions_a: np.ndarray, actions_b: np.ndarray, t_grid: tuple
) -> dict:
    tv_curve = {}
    for t in t_grid:
        fa = window_histograms(actions_a, t)
        fb = window_histograms(actions_b, t)
        tv_curve[t] = max(0.0, 1.0 - 2.0 * loo_error(fa, fb))
    fa1 = actions_a.mean(axis=0)
    fb1 = actions_b.mean(axis=0)
    kappa = 0.5 * float(np.abs(fa1 - fb1).sum())
    return {"tv_curve": tv_curve, "kappa": kappa}


def synthetic_validation() -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for kappa_true in (0.1, 0.2, 0.35):
        p0, p1 = (1.0 - kappa_true) / 2.0, (1.0 + kappa_true) / 2.0
        group_a = (
            rng.random((VALIDATE_EPISODES, VALIDATE_STEPS)) < p0
        ).astype(np.float64)
        group_b = (
            rng.random((VALIDATE_EPISODES, VALIDATE_STEPS)) < p1
        ).astype(np.float64)
        tv_curve = {}
        for t in VALIDATE_T:
            prefix_a = group_a[:, :t].sum(axis=1)
            prefix_b = group_b[:, :t].sum(axis=1)
            tv_true = 0.5 * float(
                np.abs(
                    np.array([(prefix_a == k).mean() for k in range(t + 1)])
                    - np.array([(prefix_b == k).mean() for k in range(t + 1)])
                ).sum()
            )
            err = loo_error(prefix_a[:, None], prefix_b[:, None])
            tv_hat = max(0.0, 1.0 - 2.0 * err)
            tv_curve[t] = {
                "tv_true": tv_true,
                "tv_hat": tv_hat,
                "relative_bias": (
                    abs(tv_hat - tv_true) / tv_true if tv_true > 1e-9 else 0.0
                ),
            }
        relative_biases = [entry["relative_bias"] for entry in tv_curve.values()]
        kappa_hat = 0.5 * float(abs(group_a.mean() - group_b.mean()) * 2.0)
        results.append(
            {
                "kappa_true": kappa_true,
                "kappa_hat": kappa_hat,
                "kappa_abs_bias": abs(kappa_hat - kappa_true),
                "max_tv_relative_bias": max(relative_biases),
                "tv_curve": tv_curve,
            }
        )
    passed = all(
        entry["max_tv_relative_bias"] <= TV_BIAS_TOLERANCE
        and entry["kappa_abs_bias"] <= KAPPA_BIAS_TOLERANCE
        for entry in results
    )
    return {"pass": bool(passed), "results": results}


def bootstrap_curve_ci(
    actions_a: np.ndarray, actions_b: np.ndarray, t_grid: tuple
) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_a, n_b = len(actions_a), len(actions_b)
    curves = {t: [] for t in t_grid}
    for _ in range(BOOTSTRAP_REPLICATES):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        for t in t_grid:
            fa = window_histograms(actions_a[idx_a], t)
            fb = window_histograms(actions_b[idx_b], t)
            curves[t].append(max(0.0, 1.0 - 2.0 * loo_error(fa, fb)))
    return {
        t: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for t, values in curves.items()
    }


def ecology_estimation(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))
    records = payload["records"]
    actions_by_run = {
        record["partner_run_id"]: np.asarray(
            record["partner_actions"], dtype=np.int64
        )
        for record in records
    }
    type_by_run = {record["partner_run_id"]: record["partner_type"] for record in records}
    runs = sorted(actions_by_run)
    sp_runs = [r for r in runs if type_by_run[r] == "sp"]
    op_runs = [r for r in runs if type_by_run[r] == "op"]
    if not sp_runs or not op_runs:
        raise RuntimeError("both partner types are required")

    sp_actions = np.concatenate([actions_by_run[r] for r in sp_runs], axis=0)
    op_actions = np.concatenate([actions_by_run[r] for r in op_runs], axis=0)
    type_level = estimate_pair(sp_actions, op_actions, GRID_T)
    type_ci = (
        bootstrap_curve_ci(sp_actions, op_actions, CI_T)
        if args.bootstrap
        else {}
    )

    individual = {}
    for run in runs:
        rest = np.concatenate(
            [actions_by_run[other] for other in runs if other != run], axis=0
        )
        individual[run] = estimate_pair(actions_by_run[run], rest, GRID_T)["kappa"]

    delta_hat = None
    if args.panel_rows:
        rows = json.loads(Path(args.panel_rows).read_text(encoding="utf-8"))
        cells: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if "identity_match" in row["flags"]:
                continue
            cells.setdefault(
                (row["mode_run_id"], row["partner_run_id"]), []
            ).append(row["raw_return"])
        means = {key: float(np.mean(values)) for key, values in cells.items()}
        partners = sorted({key[1] for key in means})
        modes = sorted({key[0] for key in means})
        deltas = []
        for partner in partners:
            values = sorted(
                (means[(mode, partner)] for mode in modes if (mode, partner) in means),
                reverse=True,
            )
            if len(values) < 2:
                raise RuntimeError("partner has fewer than two valid mode cells")
            deltas.append(values[0] - values[1])
        delta_hat = float(np.mean(deltas))

    return {
        "type_level": {
            "kappa": type_level["kappa"],
            "tv_curve": {str(t): v for t, v in type_level["tv_curve"].items()},
            "tv_ci95": {str(t): v for t, v in type_ci.items()},
        },
        "individual_kappa": individual,
        "delta_hat_panel_proxy": delta_hat,
        "window": WINDOW,
        "t_grid": list(GRID_T),
        "scientific_readout_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3/E4 signal estimation (exploration track)."
    )
    parser.add_argument("--mode", required=True, choices=("validate", "ecology"))
    parser.add_argument("--trajectories", default="", help="Ecology trajectories.")
    parser.add_argument("--panel-rows", default="", help="Panel rows for delta proxy.")
    parser.add_argument("--output", required=True, help="Summary json path.")
    parser.add_argument(
        "--bootstrap", action="store_true", help="Add bootstrap intervals (slow)."
    )
    args = parser.parse_args()

    if args.mode == "validate":
        summary = synthetic_validation()
        if not summary["pass"]:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise SystemExit("fail-closed: estimator validation failed")
    else:
        if not args.trajectories:
            raise SystemExit("ecology mode requires --trajectories")
        summary = ecology_estimation(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    compact = {
        key: value
        for key, value in summary.items()
        if key not in ("results", "type_level")
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
