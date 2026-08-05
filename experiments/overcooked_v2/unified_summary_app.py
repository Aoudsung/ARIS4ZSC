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
from typing import Any, Mapping, NamedTuple

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


class EvaluationNodes(NamedTuple):
    means: Mapping[int, float]
    layout: str
    source: Mapping[str, str]
    base_fingerprints: Mapping[int, str]
    latent_fingerprints: Mapping[int, str]
    deployment_fingerprints: Mapping[int, str]
    partner_manifest_sha256: str | None
    evaluation_seed: int | None
    episodes_per_pairing: int | None


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


def _deployment_metadata(identity: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reference = identity.get("deployment")
    if not isinstance(reference, Mapping):
        return None
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Unified evaluation deployment reference is malformed.")
    deployment_path = Path(str(reference["path"])).resolve()
    if sha256_path(deployment_path) != str(reference["sha256"]):
        raise ValueError("Unified evaluation deployment hash changed.")
    bundle_path = deployment_path / "deployment_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    training_path = Path(str(bundle["source_training_run"])).resolve()
    training_identity = json.loads(
        (training_path / "run_identity.json").read_text(encoding="utf-8")
    )
    return {
        "deployment_path": deployment_path,
        "bundle": bundle,
        "training_identity": training_identity,
        "seed_index": int(training_identity["seed_index"]),
    }


def _row_seed(row: Mapping[str, Any]) -> int:
    if "ego_run_index" in row:
        return int(row["ego_run_index"])
    if "seed_index" in row:
        return int(row["seed_index"])
    raise KeyError("Raw external evaluation row has no ego_run_index/seed_index.")


def _evaluation_nodes(directory: Path, *, expected_label: str) -> EvaluationNodes:
    identity_path = directory / "run_identity.json"
    raw_path = directory / "episode_returns.parquet"
    if not identity_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(
            f"Evaluation directory lacks identity/raw rows: {directory}"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    rows = read_parquet(raw_path)
    if not rows:
        raise ValueError(f"Evaluation contains no raw episode rows: {directory}")
    deployment = _deployment_metadata(identity)
    grouped: dict[int, list[float]] = {}
    base_fingerprints: dict[int, str] = {}
    latent_fingerprints: dict[int, str] = {}
    deployment_fingerprints: dict[int, str] = {}

    if deployment is not None:
        seed = int(deployment["seed_index"])
        bundle = deployment["bundle"]
        identity_variant = str(identity.get("variant", "")).lower()
        if identity_variant != expected_label:
            raise ValueError(
                f"Evaluation label {expected_label!r} differs from identity "
                f"variant {identity_variant!r}."
            )
        if expected_label == FULL_VARIANT:
            if str(bundle["variant"]) != JOINT_VARIANT:
                raise ValueError("Full evaluation must reuse a joint deployment bundle.")
        elif str(bundle["variant"]) != expected_label:
            raise ValueError("Unified evaluation variant differs from its deployment.")
        grouped[seed] = [float(row["raw_return"]) for row in rows]
        base_fingerprints[seed] = str(bundle["base_parameter_fingerprint"])
        latent_fingerprints[seed] = str(bundle["latent_parameter_fingerprint"])
        deployment_fingerprints[seed] = str(bundle["parameter_fingerprint"])
    else:
        for row in rows:
            seed = _row_seed(row)
            grouped.setdefault(seed, []).append(float(row["raw_return"]))

    means = {}
    for seed, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(
                f"Evaluation {expected_label} seed {seed} has no finite values."
            )
        means[int(seed)] = float(np.mean(array))

    layout = str(identity.get("layout", rows[0].get("layout", "")))
    if not layout:
        raise ValueError("Evaluation layout is missing.")
    partner_ref = identity.get("partner_manifest")
    partner_sha = (
        str(partner_ref.get("sha256"))
        if isinstance(partner_ref, Mapping) and partner_ref.get("sha256")
        else None
    )
    episodes_per_pairing = identity.get("episodes_per_pairing")
    evaluation_seed = identity.get("evaluation_seed")
    return EvaluationNodes(
        means=means,
        layout=layout,
        source={"path": str(directory), "sha256": sha256_path(directory)},
        base_fingerprints=base_fingerprints,
        latent_fingerprints=latent_fingerprints,
        deployment_fingerprints=deployment_fingerprints,
        partner_manifest_sha256=partner_sha,
        evaluation_seed=(
            None if evaluation_seed is None else int(evaluation_seed)
        ),
        episodes_per_pairing=(
            None
            if episodes_per_pairing is None
            else int(episodes_per_pairing)
        ),
    )


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
        "inference_mode": "paired_seed_bootstrap",
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


def _independent_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size < 2 or right_values.size < 2:
        raise ValueError("Independent inference needs at least two runs per method.")
    rng = np.random.default_rng(int(seed))
    left_indexes = rng.integers(
        0, left_values.size, size=(int(replicates), left_values.size)
    )
    right_indexes = rng.integers(
        0, right_values.size, size=(int(replicates), right_values.size)
    )
    draws = np.mean(left_values[left_indexes], axis=1) - np.mean(
        right_values[right_indexes], axis=1
    )
    return {
        "inference_mode": "independent_run_bootstrap",
        "left_run_count": int(left_values.size),
        "right_run_count": int(right_values.size),
        "point_estimate": float(np.mean(left_values) - np.mean(right_values)),
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
    require_paired: bool,
) -> Mapping[str, Any]:
    common = sorted(set(left) & set(right))
    if len(common) >= 2:
        return {
            **_paired_bootstrap(
                np.asarray([left[index] - right[index] for index in common]),
                replicates=replicates,
                seed=seed,
            ),
            "paired_seed_indexes": common,
        }
    if require_paired:
        raise ValueError("Registered internal comparison lacks paired seed nodes.")
    return _independent_bootstrap(
        np.asarray(list(left.values())),
        np.asarray(list(right.values())),
        replicates=replicates,
        seed=seed,
    )


def _validate_internal_identity(
    nodes: Mapping[str, EvaluationNodes]
) -> Mapping[str, Any]:
    internal = {name: nodes[name] for name in INTERNAL_VARIANTS}
    seed_sets = {name: set(value.means) for name, value in internal.items()}
    if len({tuple(sorted(value)) for value in seed_sets.values()}) != 1:
        raise ValueError("Internal variants do not cover identical seed indexes.")
    common = sorted(next(iter(seed_sets.values())))
    if len(common) < 2:
        raise ValueError("Internal comparison needs at least two paired seeds.")
    for name, value in internal.items():
        if set(value.base_fingerprints) != set(common):
            raise ValueError(f"Internal variant {name} lacks base fingerprints.")
    for seed in common:
        fingerprints = {
            internal[name].base_fingerprints[seed]
            for name in INTERNAL_VARIANTS
        }
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"Base policy parameters differ across variants for seed {seed}."
            )
        if (
            internal[JOINT_VARIANT].deployment_fingerprints[seed]
            != internal[FULL_VARIANT].deployment_fingerprints[seed]
        ):
            raise RuntimeError(
                f"Full does not reuse the joint deployment for seed {seed}."
            )
    partner_hashes = {
        value.partner_manifest_sha256 for value in internal.values()
    }
    if len(partner_hashes) != 1 or None in partner_hashes:
        raise ValueError("Internal variants do not share one partner panel.")
    evaluation_seeds = {value.evaluation_seed for value in internal.values()}
    if len(evaluation_seeds) != 1:
        raise ValueError("Internal variants do not share evaluation keys.")
    episode_counts = {value.episodes_per_pairing for value in internal.values()}
    if len(episode_counts) != 1:
        raise ValueError("Internal variants use different episode budgets.")
    return {
        "paired_seed_indexes": common,
        "base_parameter_fingerprint_match": True,
        "full_reuses_joint_deployment": True,
        "shared_partner_manifest_sha256": next(iter(partner_hashes)),
        "shared_evaluation_seed": next(iter(evaluation_seeds)),
        "shared_episodes_per_pairing": next(iter(episode_counts)),
    }


def run_summary(args: Any) -> None:
    nodes: dict[str, EvaluationNodes] = {}
    for raw in args.evaluation:
        label, directory = _parse_input(raw)
        current = _evaluation_nodes(directory, expected_label=label)
        if label in nodes:
            existing = nodes[label]
            overlap = set(existing.means) & set(current.means)
            if overlap:
                raise ValueError(
                    f"Duplicate {label} seed nodes: {sorted(overlap)}"
                )
            if existing.layout != current.layout:
                raise ValueError(f"Evaluation label {label} mixes layouts.")
            nodes[label] = EvaluationNodes(
                means={**existing.means, **current.means},
                layout=existing.layout,
                source={
                    "path": "multiple",
                    "sha256": sha256_path(
                        Path(existing.source["path"])
                        if existing.source["path"] != "multiple"
                        else directory
                    ),
                },
                base_fingerprints={
                    **existing.base_fingerprints,
                    **current.base_fingerprints,
                },
                latent_fingerprints={
                    **existing.latent_fingerprints,
                    **current.latent_fingerprints,
                },
                deployment_fingerprints={
                    **existing.deployment_fingerprints,
                    **current.deployment_fingerprints,
                },
                partner_manifest_sha256=(
                    existing.partner_manifest_sha256
                    if existing.partner_manifest_sha256
                    == current.partner_manifest_sha256
                    else None
                ),
                evaluation_seed=(
                    existing.evaluation_seed
                    if existing.evaluation_seed == current.evaluation_seed
                    else None
                ),
                episodes_per_pairing=(
                    existing.episodes_per_pairing
                    if existing.episodes_per_pairing
                    == current.episodes_per_pairing
                    else None
                ),
            )
        else:
            nodes[label] = current
    if not INTERNAL_VARIANTS <= set(nodes):
        raise ValueError("Summary requires base, response_only, joint, and full.")
    layouts = {value.layout for value in nodes.values()}
    if len(layouts) != 1:
        raise ValueError("Summary inputs mix layouts.")
    internal_identity = _validate_internal_identity(nodes)
    replicates = int(getattr(args, "bootstrap_replicates", 9999))
    seed = int(getattr(args, "bootstrap_seed", 0))
    comparisons = {
        "joint_minus_response_only": _comparison(
            nodes[JOINT_VARIANT].means,
            nodes[RESPONSE_ONLY_VARIANT].means,
            replicates=replicates,
            seed=seed + 1,
            require_paired=True,
        ),
        "full_minus_joint": _comparison(
            nodes[FULL_VARIANT].means,
            nodes[JOINT_VARIANT].means,
            replicates=replicates,
            seed=seed + 2,
            require_paired=True,
        ),
        "full_minus_base": _comparison(
            nodes[FULL_VARIANT].means,
            nodes[BASE_VARIANT].means,
            replicates=replicates,
            seed=seed + 3,
            require_paired=True,
        ),
    }
    external = sorted(set(nodes) - INTERNAL_VARIANTS)
    best_external = None
    if external:
        best_external = max(
            external,
            key=lambda label: np.mean(list(nodes[label].means.values())),
        )
        comparisons["full_minus_best_external"] = _comparison(
            nodes[FULL_VARIANT].means,
            nodes[best_external].means,
            replicates=replicates,
            seed=seed + 4,
            require_paired=False,
        )

    causal_path = Path(args.causal_evaluation).resolve()
    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    if causal.get("artifact_type") != "unified_delta_causal_belief_value":
        raise ValueError("Causal source is not the unified belief-value artifact.")
    if str(causal.get("layout")) != next(iter(layouts)):
        raise ValueError("Causal source layout differs from evaluation layout.")
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
                comparisons["full_minus_best_external"]["one_sided_lcb_95"]
                > 0.0
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
            "version": 2,
            "artifact_type": "unified_delta_three_hypothesis_summary",
            "layout": next(iter(layouts)),
            "method_means": {
                label: float(np.mean(list(value.means.values())))
                for label, value in nodes.items()
            },
            "per_seed_means": {
                label: dict(value.means) for label, value in nodes.items()
            },
            "internal_identity": internal_identity,
            "comparisons": comparisons,
            "hypothesis_hierarchy": hierarchy,
            "minimum_material_effect": minimum_effect,
            "sources": {
                "evaluations": {
                    label: value.source for label, value in nodes.items()
                },
                "causal": {
                    "path": str(causal_path),
                    "sha256": sha256_path(causal_path),
                },
            },
        },
    )


__all__ = ["run_summary"]
