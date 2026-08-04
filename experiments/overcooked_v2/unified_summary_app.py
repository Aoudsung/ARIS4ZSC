"""Three-hypothesis statistical summary for unified DELTA-ZSC.

The confirmatory hierarchy is deliberately small:

H1 performance: Full DELTA exceeds the strongest external baseline.
H2 mechanism: joint response-decision training exceeds response-only training.
H3 causality: the correct belief has positive source-world decision value over
   a task-matched shuffled belief.

Every other metric is diagnostic and cannot be promoted into an additional
paper contribution after results are observed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.delta_zsc.model import (
    BASE_VARIANT,
    FULL_VARIANT,
    JOINT_VARIANT,
    RESPONSE_ONLY_VARIANT,
)
from src.path_c.storage import read_parquet, sha256_path


INTERNAL_VARIANTS = {
    BASE_VARIANT,
    RESPONSE_ONLY_VARIANT,
    JOINT_VARIANT,
    FULL_VARIANT,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parse_input(value: str) -> tuple[str, Path]:
    try:
        label, raw = value.split("=", 1)
    except ValueError as error:
        raise ValueError("Evaluation inputs use LABEL=/path/to/evaluation_dir.") from error
    label = label.strip().lower()
    path = Path(raw).resolve()
    if not label or not path.is_dir():
        raise ValueError(f"Invalid evaluation input: {value}")
    return label, path


def _seed_index(directory: Path) -> int:
    identity = json.loads((directory / "run_identity.json").read_text(encoding="utf-8"))
    deployment_ref = identity.get("deployment")
    if isinstance(deployment_ref, Mapping):
        deployment_path = Path(str(deployment_ref["path"])).resolve()
        bundle = json.loads(
            (deployment_path / "deployment_bundle.json").read_text(encoding="utf-8")
        )
        training = Path(str(bundle["source_training_run"])).resolve()
        training_identity = json.loads(
            (training / "run_identity.json").read_text(encoding="utf-8")
        )
        return int(training_identity["seed_index"])
    rows = read_parquet(directory / "episode_returns.parquet")
    candidates = {
        int(row["ego_run_index"])
        for row in rows
        if "ego_run_index" in row
    }
    if len(candidates) != 1:
        raise ValueError(f"Cannot infer one seed index from {directory}")
    return next(iter(candidates))


def _evaluation_mean(directory: Path) -> tuple[int, float, Mapping[str, Any]]:
    identity = json.loads((directory / "run_identity.json").read_text(encoding="utf-8"))
    rows = read_parquet(directory / "episode_returns.parquet")
    values = np.asarray([float(row["raw_return"]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Evaluation contains no finite raw returns: {directory}")
    return _seed_index(directory), float(np.mean(values)), identity


def _paired_bootstrap(
    difference: np.ndarray, *, replicates: int, seed: int
) -> Mapping[str, Any]:
    values = np.asarray(difference, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Paired inference needs at least two seed nodes.")
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, values.size, size=(int(replicates), values.size))
    draws = np.mean(values[indexes], axis=1)
    return {
        "seed_count": int(values.size),
        "point_estimate": float(np.mean(values)),
        "one_sided_lcb_95": float(np.quantile(draws, 0.05)),
        "interval_95": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
        "one_sided_bootstrap_p": float(
            (1.0 + np.sum(draws <= 0.0)) / (draws.size + 1.0)
        ),
    }


def _comparison(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        raise ValueError("Compared methods do not share enough paired seeds.")
    difference = np.asarray([left[index] - right[index] for index in common])
    return {
        **_paired_bootstrap(difference, replicates=replicates, seed=seed),
        "paired_seed_indexes": common,
    }


def run_summary(args: Any) -> None:
    grouped: dict[str, dict[int, float]] = {}
    sources: dict[str, list[Mapping[str, Any]]] = {}
    identities: dict[str, list[Mapping[str, Any]]] = {}
    for raw in args.evaluation:
        label, directory = _parse_input(raw)
        seed, mean, identity = _evaluation_mean(directory)
        if seed in grouped.setdefault(label, {}):
            raise ValueError(f"Duplicate {label} seed {seed} evaluation.")
        grouped[label][seed] = mean
        sources.setdefault(label, []).append(
            {"path": str(directory), "sha256": sha256_path(directory)}
        )
        identities.setdefault(label, []).append(identity)
    if not INTERNAL_VARIANTS <= set(grouped):
        raise ValueError("Summary requires base, response_only, joint, and full.")
    layouts = {
        str(identity.get("layout"))
        for rows in identities.values()
        for identity in rows
    }
    if len(layouts) != 1:
        raise ValueError("Summary inputs mix layouts.")
    replicates = int(getattr(args, "bootstrap_replicates", 9999))
    seed = int(getattr(args, "bootstrap_seed", 0))
    comparisons = {
        "joint_minus_response_only": _comparison(
            grouped[JOINT_VARIANT],
            grouped[RESPONSE_ONLY_VARIANT],
            replicates=replicates,
            seed=seed + 1,
        ),
        "full_minus_joint": _comparison(
            grouped[FULL_VARIANT],
            grouped[JOINT_VARIANT],
            replicates=replicates,
            seed=seed + 2,
        ),
        "full_minus_base": _comparison(
            grouped[FULL_VARIANT],
            grouped[BASE_VARIANT],
            replicates=replicates,
            seed=seed + 3,
        ),
    }
    external = sorted(set(grouped) - INTERNAL_VARIANTS)
    best_external = None
    if external:
        best_external = max(
            external,
            key=lambda label: np.mean(list(grouped[label].values())),
        )
        comparisons["full_minus_best_external"] = _comparison(
            grouped[FULL_VARIANT],
            grouped[best_external],
            replicates=replicates,
            seed=seed + 4,
        )
    causal_path = Path(args.causal_evaluation).resolve()
    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    if causal.get("artifact_type") != "unified_delta_causal_belief_value":
        raise ValueError("Causal source is not the unified belief-value artifact.")
    h3 = causal["summary"]

    minimum_effect = float(getattr(args, "minimum_effect", 20.0))
    h1 = (
        {
            "estimable": False,
            "passed": False,
            "reason": "no_external_baseline_supplied",
        }
        if best_external is None
        else {
            "estimable": True,
            "best_external": best_external,
            "comparison": comparisons["full_minus_best_external"],
            "passed": bool(
                comparisons["full_minus_best_external"]["one_sided_lcb_95"] > 0.0
                and comparisons["full_minus_best_external"]["point_estimate"]
                >= minimum_effect
            ),
        }
    )
    h2 = {
        "comparison": comparisons["joint_minus_response_only"],
        "passed": bool(
            comparisons["joint_minus_response_only"]["one_sided_lcb_95"] > 0.0
        ),
    }
    h3_claim = {
        "comparison": h3,
        "passed": bool(float(h3["one_sided_lcb_95"]) > 0.0),
    }
    # Ordered confirmatory hierarchy controls selective claim proliferation:
    # H2 is confirmatory only if H1 passes; H3 only if H1 and H2 pass.  All
    # measurements remain visible regardless of hierarchy status.
    hierarchy = {
        "H1_performance": h1,
        "H2_decision_emission": {
            **h2,
            "confirmatory_unlocked": bool(h1["passed"]),
            "confirmatory_passed": bool(h1["passed"] and h2["passed"]),
        },
        "H3_causal_belief_value": {
            **h3_claim,
            "confirmatory_unlocked": bool(h1["passed"] and h2["passed"]),
            "confirmatory_passed": bool(
                h1["passed"] and h2["passed"] and h3_claim["passed"]
            ),
        },
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "unified_delta_summary.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_three_hypothesis_summary",
            "layout": next(iter(layouts)),
            "method_means": {
                label: float(np.mean(list(values.values())))
                for label, values in grouped.items()
            },
            "per_seed_means": grouped,
            "comparisons": comparisons,
            "hypothesis_hierarchy": hierarchy,
            "minimum_material_effect": minimum_effect,
            "sources": {
                "evaluations": sources,
                "causal": {
                    "path": str(causal_path),
                    "sha256": sha256_path(causal_path),
                },
            },
        },
    )


__all__ = ["run_summary"]
