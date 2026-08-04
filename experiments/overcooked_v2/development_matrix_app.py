"""Executable, raw-evaluated workflow for the nested DEPI R0/B0--B3 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.training_app import run_training
from src.path_c.experiment import (
    CONFIG_VERSION,
    MECHANISM_ABLATION_VARIANTS,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    RUN_BUDGETS,
    official_training_domain_keys,
)
from src.path_c.component_diagnostics import (
    permutation_aligned_component_stability,
    validate_component_diagnostic_values,
)
from src.path_c.resources import ResourceLedger
from src.path_c.training import ANCHOR_VARIANTS
from src.path_c.storage import (
    ensure_run_identity,
    read_parquet,
    runtime_provenance,
    sha256_path,
    write_json,
    write_parquet,
)


CORE_DEVELOPMENT_VARIANTS = ("r0", "b0", "b1", "b2")
TOTAL_BUDGET_VARIANTS = ("r0_extra", "b0_extra", "b1_extra")
DEVELOPMENT_VARIANTS = (
    *CORE_DEVELOPMENT_VARIANTS,
    *TOTAL_BUDGET_VARIANTS,
    *MECHANISM_ABLATION_VARIANTS,
)


def _b2_auxiliary_transition_budget(payload: Mapping[str, Any]) -> int:
    """Return the exact registered B2 anchor/probe transition cost."""

    training = payload["training"]
    anchors = payload["anchors"]
    budget = RUN_BUDGETS["development"]
    base_updates = budget.environment_steps // (
        budget.num_envs * int(training["rollout_length"])
    )
    trigger_count = (
        base_updates + int(anchors["interval_updates"]) - 1
    ) // int(anchors["interval_updates"])
    states = int(anchors["ordinary_states"]) + 2 * int(
        anchors["matched_history_pairs"]
    )
    return trigger_count * (
        states
        * OFFICIAL_ACTION_COUNT
        * (int(anchors["fit_replicas"]) + int(anchors["evaluation_replicas"]))
        * int(anchors["continuation_horizon"])
        + 2
        * int(anchors["matched_history_pairs"])
        * int(anchors["probe_steps"])
    )


def validate_development_entry_alignment(
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless core and total-budget controls are exactly paired."""

    by_seed: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for entry in entries:
        key = (int(entry["protocol_components"]), int(entry["seed_index"]))
        by_seed.setdefault(key, []).append(entry)
    if not by_seed:
        raise ValueError("Development matrix has no run entries.")
    for (component_count, seed_index), rows in by_seed.items():
        if (
            len(rows) != len(DEVELOPMENT_VARIANTS)
            or {str(row["variant"]) for row in rows} != set(DEVELOPMENT_VARIANTS)
        ):
            raise RuntimeError(
                "Every development seed must cover core and total-budget variants exactly once."
            )
        checks = {
            "partner sampler": {
                str(row["partner_sampler_sha256"]) for row in rows
            },
            "deployable parameter capacity": {
                int(row["deployable_parameters"]) for row in rows
            },
            "episode keys": {
                json.dumps(row["episode_key_domains"], sort_keys=True)
                for row in rows
            },
        }
        for label, values in checks.items():
            if len(values) != 1:
                raise RuntimeError(
                    f"{label} differs across K={component_count} variants "
                    f"for seed {seed_index}."
                )

        indexed = {str(row["variant"]): row for row in rows}
        core_ppo = {
            int(indexed[variant]["ppo_training_steps"])
            for variant in CORE_DEVELOPMENT_VARIANTS
        }
        if len(core_ppo) != 1:
            raise RuntimeError(
                f"main PPO budget differs across K={component_count} core variants "
                f"for seed {seed_index}."
            )
        base_ppo = next(iter(core_ppo))
        b2_auxiliary = int(indexed["b2"]["auxiliary_training_steps"])
        if b2_auxiliary <= 0:
            raise RuntimeError("B2 must report a positive anchor/probe transition cost.")
        for variant in MECHANISM_ABLATION_VARIANTS:
            if int(indexed[variant]["ppo_training_steps"]) != base_ppo:
                raise RuntimeError(
                    f"main PPO budget differs for mechanism ablation {variant}."
                )
        for variant in (
            "r0",
            "b0",
            "b1",
            "deterministic_context",
            *TOTAL_BUDGET_VARIANTS,
        ):
            if int(indexed[variant]["auxiliary_training_steps"]) != 0:
                raise RuntimeError(f"{variant} must not collect B2 anchor/probe transitions.")
        for variant in ANCHOR_VARIANTS:
            if int(indexed[variant]["auxiliary_training_steps"]) != b2_auxiliary:
                raise RuntimeError(
                    f"{variant} must use the registered B2 anchor/probe transition budget."
                )
            if int(indexed[variant]["total_training_simulator_steps"]) != int(
                indexed["b2"]["total_training_simulator_steps"]
            ):
                raise RuntimeError(
                    f"{variant} and B2 total simulator budgets differ."
                )
        for variant in TOTAL_BUDGET_VARIANTS:
            if int(indexed[variant]["ppo_training_steps"]) != base_ppo + b2_auxiliary:
                raise RuntimeError(
                    f"{variant} PPO budget does not exactly replace the B2 auxiliary cost."
                )
            if int(indexed[variant]["total_training_simulator_steps"]) != int(
                indexed["b2"]["total_training_simulator_steps"]
            ):
                raise RuntimeError(
                    f"{variant} and B2 total simulator budgets differ."
                )


def _variant_config(
    source: Path, target: Path, variant: str, protocol_components: int
) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Development base config must be a mapping.")
    payload = dict(payload)
    payload["version"] = CONFIG_VERSION
    payload["method_variant"] = variant.removesuffix("_extra")
    payload["training"] = dict(payload["training"])
    payload["training"]["extra_ppo_environment_steps"] = (
        _b2_auxiliary_transition_budget(payload) if variant.endswith("_extra") else 0
    )
    payload["model"] = dict(payload["model"])
    payload["model"]["protocol_components"] = int(protocol_components)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_development_matrix(args: argparse.Namespace) -> None:
    source_config = Path(args.config).resolve()
    manifest = Path(args.partner_manifest).resolve()
    output = Path(args.output).resolve()
    seeds = tuple(int(value) for value in args.seed_index)
    requested_components = getattr(args, "protocol_components", None)
    component_counts = tuple(
        int(value)
        for value in (
            (2, 4, 8) if requested_components is None else requested_components
        )
    )
    if tuple(sorted(seeds)) != tuple(range(10)) or len(seeds) != 10:
        raise ValueError("Registered development conclusions require seeds 0--9 exactly once.")
    if set(component_counts) != {2, 4, 8} or len(component_counts) != 3:
        raise ValueError("Development sensitivity must cover K in {2,4,8} exactly once.")
    entries = []
    for component_count in component_counts:
        for variant in DEVELOPMENT_VARIANTS:
            variant_config = (
                output / "configs" / f"k-{component_count}" / f"{variant}.yaml"
            )
            _variant_config(
                source_config, variant_config, variant, component_count
            )
            for seed_index in seeds:
                run_directory = (
                    output
                    / "runs"
                    / f"k-{component_count}"
                    / variant
                    / f"seed-{seed_index}"
                )
                run_training(
                    SimpleNamespace(
                        config=str(variant_config),
                        partner_manifest=str(manifest),
                        ego_run_id=(
                            f"depi-k{component_count}-{variant}-development-"
                            f"seed-{seed_index}"
                        ),
                        seed_index=seed_index,
                        run_kind="development",
                        output=str(run_directory),
                        pair_comparator=(
                            str(args.pair_comparator)
                            if variant.removesuffix("_extra") in ANCHOR_VARIANTS
                            else None
                        ),
                        resume=bool(args.resume),
                        skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                        _execution_scope="development-matrix",
                    )
                )
                identity = json.loads(
                    (run_directory / "run_identity.json").read_text(encoding="utf-8")
                )
                ledger = ResourceLedger.from_mapping(
                    json.loads(
                        (run_directory / "resource_ledger.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
                pool_path = run_directory / "partner_pool.json"
                entries.append(
                    {
                        "variant": variant,
                        "protocol_components": component_count,
                        "seed_index": seed_index,
                        "run": str(run_directory),
                        "config": {
                            "path": str(variant_config.resolve()),
                            "sha256": sha256_path(variant_config),
                        },
                        "run_identity_sha256": sha256_path(
                            run_directory / "run_identity.json"
                        ),
                        "config_fingerprint": identity["config_fingerprint"],
                        "partner_sampler_sha256": sha256_path(pool_path),
                        "total_training_simulator_steps": (
                            ledger.total_training_simulator_steps
                        ),
                        "ppo_training_steps": ledger.ego_policy_steps,
                        "auxiliary_training_steps": (
                            ledger.counterfactual_continuation_steps
                            + ledger.matched_pair_probe_steps
                        ),
                        "deployable_parameters": ledger.deployable_parameters,
                        "gpu_hours": ledger.gpu_hours,
                        "episode_key_domains": {
                            name: list(value)
                            for name, value in official_training_domain_keys(
                                seed_index
                            ).items()
                        },
                    }
                )
    validate_development_entry_alignment(entries)
    result = {
        "version": 1,
        "artifact_type": "depi_development_matrix",
        "method": METHOD_VERSION,
        "source_config": {
            "path": str(source_config),
            "sha256": sha256_path(source_config),
        },
        "partner_manifest": {
            "path": str(manifest),
            "sha256": sha256_path(manifest),
        },
        "variants": {
            "r0": "full_history_recurrent_ppo_reference",
            "b0": "task_only_recurrence_plus_instant_partner",
            "b1": "b0_plus_exact_response_filter",
            "b2": "decision_supervision",
            "r0_extra": "r0_with_b2_cost_reallocated_to_ordinary_ppo",
            "b0_extra": "b0_with_b2_cost_reallocated_to_ordinary_ppo",
            "b1_extra": "b1_with_b2_cost_reallocated_to_ordinary_ppo",
            "b3": "not_implemented_action_conditioned_voi",
        },
        "b3_status": "not_implemented",
        "protocol_component_sensitivity": list(component_counts),
        "entries": entries,
        "budget_capacity_and_key_matching_passed": True,
    }
    ensure_run_identity(
        output,
        {
            "stage": "run-development-matrix",
            "method": METHOD_VERSION,
            "repository_runtime": runtime_provenance(),
            "source_config": result["source_config"],
            "partner_manifest": result["partner_manifest"],
            "seeds": list(seeds),
            "protocol_component_sensitivity": list(component_counts),
        },
    )
    write_json(output / "development_matrix.json", result)


def _paired_bootstrap(values: np.ndarray, *, seed: int) -> tuple[float, list[float]]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Paired development inference needs at least two seeds.")
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, values.size, size=(9_999, values.size))
    draws = values[indexes].mean(axis=1)
    interval = np.quantile(draws, (0.005, 0.995))
    return float(np.mean(values)), [float(interval[0]), float(interval[1])]


def _validate_development_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    protocol_components: int,
    seed_indexes: Sequence[int],
    schedule_sha256: str,
) -> Mapping[int, float]:
    """Validate the complete ordered-pair episode table and recompute XP."""

    seeds = tuple(int(seed) for seed in seed_indexes)
    if tuple(sorted(seeds)) != tuple(range(10)) or len(seeds) != 10:
        raise ValueError(
            "Registered development raw rows require seed indexes 0--9 exactly once."
        )
    fields = {
        "variant",
        "protocol_components",
        "left_seed_index",
        "right_seed_index",
        "left_role",
        "right_role",
        "episode_index",
        "raw_return",
        "evaluation_key_schedule_sha256",
    }
    episodes_by_pair: dict[tuple[int, int], set[int]] = {}
    returns_by_seed = {seed: [] for seed in seeds}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("Development raw episode row schema differs.")
        left = int(row["left_seed_index"])
        right = int(row["right_seed_index"])
        episode = int(row["episode_index"])
        raw_return = float(row["raw_return"])
        if (
            row["variant"] != variant
            or int(row["protocol_components"]) != int(protocol_components)
            or left not in returns_by_seed
            or right not in returns_by_seed
            or int(row["left_role"]) != 0
            or int(row["right_role"]) != 1
            or not 0 <= episode < 500
            or not np.isfinite(raw_return)
            or row["evaluation_key_schedule_sha256"] != schedule_sha256
        ):
            raise ValueError("Development raw episode identity differs.")
        pair = (left, right)
        pair_episodes = episodes_by_pair.setdefault(pair, set())
        if episode in pair_episodes:
            raise ValueError("Development raw episode row is duplicated.")
        pair_episodes.add(episode)
        if left != right:
            returns_by_seed[left].append(raw_return)
            returns_by_seed[right].append(raw_return)
    expected_pairs = {(left, right) for left in seeds for right in seeds}
    if set(episodes_by_pair) != expected_pairs or any(
        episodes != set(range(500)) for episodes in episodes_by_pair.values()
    ):
        raise ValueError("Development raw rows do not cover every ordered pairing.")
    expected_per_seed = 2 * (len(seeds) - 1) * 500
    if any(len(values) != expected_per_seed for values in returns_by_seed.values()):
        raise ValueError("Development XP role coverage differs.")
    return {
        seed: float(np.mean(values)) for seed, values in returns_by_seed.items()
    }


def evaluate_development_matrix(args: argparse.Namespace) -> None:
    """Evaluate one K/variant block and bind every score to raw episodes."""

    import hashlib
    import jax

    from experiments.overcooked_v2.deployment import load_deployment
    from experiments.overcooked_v2.official_evaluation_app import _evaluate_cell
    from experiments.overcooked_v2.official_policy import OfficialDEPIPolicy
    from src.path_c.experiment import load_config

    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("artifact_type") != "depi_development_matrix":
        raise ValueError("Development matrix artifact type differs.")
    component_count = int(args.protocol_components)
    variant = str(args.variant)
    selected = sorted(
        (
            row
            for row in matrix["entries"]
            if int(row["protocol_components"]) == component_count
            and str(row["variant"]) == variant
        ),
        key=lambda row: int(row["seed_index"]),
    )
    if variant not in DEVELOPMENT_VARIANTS or len(selected) != 10:
        raise ValueError("Development evaluation requires all ten paired ego seeds.")
    policies = []
    deployment_sources = []
    config = None
    for row in selected:
        run = Path(str(row["run"])).resolve()
        config_source = Path(str(row["config"]["path"])).resolve()
        if sha256_path(config_source) != row["config"]["sha256"]:
            raise ValueError("Development run config hash differs from the matrix.")
        identity_path = run / "run_identity.json"
        resource_path = run / "resource_ledger.json"
        partner_sampler_path = run / "partner_pool.json"
        if sha256_path(identity_path) != row["run_identity_sha256"]:
            raise ValueError("Development training run identity hash differs.")
        if sha256_path(partner_sampler_path) != row["partner_sampler_sha256"]:
            raise ValueError("Development partner sampler hash differs.")
        training_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        training_ledger = ResourceLedger.from_mapping(
            json.loads(resource_path.read_text(encoding="utf-8"))
        )
        current_config = load_config(config_source, run_kind="development")
        if (
            training_identity.get("stage") != "train"
            or training_identity.get("method") != METHOD_VERSION
            or training_identity.get("method_variant")
            != variant.removesuffix("_extra")
            or int(training_identity.get("seed_index", -1))
            != int(row["seed_index"])
            or training_identity.get("config_fingerprint")
            != current_config.fingerprint
            or current_config.fingerprint != row["config_fingerprint"]
            or training_ledger.ego_policy_steps != int(row["ppo_training_steps"])
            or training_ledger.counterfactual_continuation_steps
            + training_ledger.matched_pair_probe_steps
            != int(row["auxiliary_training_steps"])
            or training_ledger.total_training_simulator_steps
            != int(row["total_training_simulator_steps"])
        ):
            raise ValueError("Development training run lineage or resource ledger differs.")
        if config is None:
            config = current_config
        elif current_config.fingerprint != config.fingerprint:
            raise ValueError("Development evaluation block mixes config fingerprints.")
        deployment_path = run / "final_deployment"
        deployment = load_deployment(deployment_path, current_config)
        policies.append(OfficialDEPIPolicy(deployment))
        deployment_sources.append(
            {
                "seed_index": int(row["seed_index"]),
                "path": str(deployment_path),
                "sha256": sha256_path(deployment_path),
                "config_fingerprint": current_config.fingerprint,
                "training_run": str(run),
                "run_identity": {
                    "path": str(identity_path),
                    "sha256": sha256_path(identity_path),
                },
                "resource_ledger": {
                    "path": str(resource_path),
                    "sha256": sha256_path(resource_path),
                },
                "partner_sampler": {
                    "path": str(partner_sampler_path),
                    "sha256": sha256_path(partner_sampler_path),
                },
            }
        )
    schedule = {
        "root_key": [0, 0],
        "policy_count": len(policies),
        "episodes_per_ordered_pair": 500,
        "ordered_pairs": [
            [left, right]
            for left in range(len(policies))
            for right in range(len(policies))
        ],
    }
    schedule_sha = hashlib.sha256(
        json.dumps(schedule, sort_keys=True).encode("utf-8")
    ).hexdigest()
    rows = []
    root_key = jax.random.PRNGKey(0)
    for left in range(len(policies)):
        for right in range(len(policies)):
            returns = _evaluate_cell(
                left=policies[left],
                right=policies[right],
                config=config,
                root_key=root_key,
            )
            for episode_index, raw_return in enumerate(returns):
                rows.append(
                    {
                        "variant": variant,
                        "protocol_components": component_count,
                        "left_seed_index": int(selected[left]["seed_index"]),
                        "right_seed_index": int(selected[right]["seed_index"]),
                        "left_role": 0,
                        "right_role": 1,
                        "episode_index": int(episode_index),
                        "raw_return": float(raw_return),
                        "evaluation_key_schedule_sha256": schedule_sha,
                    }
                )
    output = Path(args.output).resolve()
    raw_path = output / "episode_returns.parquet"
    write_parquet(raw_path, rows)
    xp_rows = [
        row["raw_return"]
        for row in rows
        if row["left_seed_index"] != row["right_seed_index"]
    ]
    ledger = ResourceLedger(
        evaluation_steps=len(policies) * len(policies) * 500 * 400
    )
    result = {
        "version": 2,
        "artifact_type": "depi_development_evaluation",
        "method": METHOD_VERSION,
        "variant": variant,
        "protocol_components": component_count,
        "config_fingerprint": config.fingerprint,
        "matrix": {"path": str(matrix_path), "sha256": sha256_path(matrix_path)},
        "deployments": deployment_sources,
        "episode_key_schedule": schedule,
        "episode_key_schedule_sha256": schedule_sha,
        "raw_episodes": {"path": str(raw_path), "sha256": sha256_path(raw_path)},
        "episode_count": len(rows),
        "xp_mean_recomputed": float(np.mean(xp_rows)),
        "resource_ledger": ledger.to_mapping(),
    }
    ensure_run_identity(
        output,
        {
            "stage": "evaluate-development-matrix",
            "method": METHOD_VERSION,
            "matrix": result["matrix"],
            "variant": variant,
            "protocol_components": component_count,
            "deployments": deployment_sources,
            "episode_key_schedule_sha256": schedule_sha,
        },
    )
    write_json(output / "development_evaluation.json", result)
    write_json(output / "resource_ledger.json", ledger.to_mapping())


def summarize_development_matrix(args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("artifact_type") != "depi_development_matrix":
        raise ValueError("Development matrix artifact type differs.")
    matrix_index = {
        (
            int(row["protocol_components"]),
            str(row["variant"]),
            int(row["seed_index"]),
        ): row
        for row in matrix["entries"]
    }
    score_index = {}
    evaluation_sources = []
    for raw_directory in args.evaluation:
        directory = Path(raw_directory).resolve()
        artifact_path = directory / "development_evaluation.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 2
            or payload.get("artifact_type") != "depi_development_evaluation"
            or payload.get("method") != METHOD_VERSION
        ):
            raise ValueError("Development evaluation schema differs.")
        raw_path = Path(str(payload["raw_episodes"]["path"])).resolve()
        if sha256_path(raw_path) != payload["raw_episodes"]["sha256"]:
            raise ValueError("Development raw episode hash differs.")
        variant = str(payload["variant"])
        component_count = int(payload["protocol_components"])
        if variant not in DEVELOPMENT_VARIANTS:
            raise ValueError("Development evaluation variant is unknown.")
        for deployment in payload["deployments"]:
            deployment_fields = {
                "seed_index",
                "path",
                "sha256",
                "config_fingerprint",
                "training_run",
                "run_identity",
                "resource_ledger",
                "partner_sampler",
            }
            if not isinstance(deployment, Mapping) or set(deployment) != deployment_fields:
                raise ValueError("Development deployment source is malformed.")
            for ref_name in ("run_identity", "resource_ledger", "partner_sampler"):
                ref = deployment[ref_name]
                if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
                    raise ValueError("Development deployment lineage source is malformed.")
            seed = int(deployment["seed_index"])
            entry = matrix_index.get((component_count, variant, seed))
            if entry is None:
                raise ValueError("Development deployment is not a matrix entry.")
            run = Path(str(entry["run"])).resolve()
            identity_path = Path(str(deployment["run_identity"]["path"])).resolve()
            resource_path = Path(str(deployment["resource_ledger"]["path"])).resolve()
            sampler_path = Path(str(deployment["partner_sampler"]["path"])).resolve()
            training_ledger = ResourceLedger.from_mapping(
                json.loads(resource_path.read_text(encoding="utf-8"))
            )
            if (
                sha256_path(Path(deployment["path"])) != deployment["sha256"]
                or Path(str(deployment["path"])).resolve() != run / "final_deployment"
                or Path(str(deployment["training_run"])).resolve() != run
                or identity_path != run / "run_identity.json"
                or resource_path != run / "resource_ledger.json"
                or sampler_path != run / "partner_pool.json"
                or sha256_path(identity_path) != deployment["run_identity"]["sha256"]
                or sha256_path(identity_path) != entry["run_identity_sha256"]
                or sha256_path(resource_path)
                != deployment["resource_ledger"]["sha256"]
                or sha256_path(sampler_path) != deployment["partner_sampler"]["sha256"]
                or sha256_path(sampler_path) != entry["partner_sampler_sha256"]
                or deployment["config_fingerprint"] != entry["config_fingerprint"]
                or training_ledger.ego_policy_steps != int(entry["ppo_training_steps"])
                or training_ledger.counterfactual_continuation_steps
                + training_ledger.matched_pair_probe_steps
                != int(entry["auxiliary_training_steps"])
                or training_ledger.total_training_simulator_steps
                != int(entry["total_training_simulator_steps"])
            ):
                raise ValueError("Development deployment hash differs.")
        raw_rows = read_parquet(raw_path)
        policy_count = len(payload["deployments"])
        expected_count = policy_count * policy_count * 500
        if len(raw_rows) != expected_count or len(raw_rows) != int(payload["episode_count"]):
            raise ValueError("Development raw episode count differs.")
        evaluation_ledger = ResourceLedger.from_mapping(payload["resource_ledger"])
        ledger_path = directory / "resource_ledger.json"
        if (
            evaluation_ledger.evaluation_steps != expected_count * 400
            or ResourceLedger.from_mapping(
                json.loads(ledger_path.read_text(encoding="utf-8"))
            )
            != evaluation_ledger
        ):
            raise ValueError("Development evaluator resource ledger differs.")
        scores = _validate_development_raw_rows(
            raw_rows,
            variant=variant,
            protocol_components=component_count,
            seed_indexes=[
                int(deployment["seed_index"])
                for deployment in payload["deployments"]
            ],
            schedule_sha256=str(payload["episode_key_schedule_sha256"]),
        )
        for deployment in payload["deployments"]:
            seed = int(deployment["seed_index"])
            key = (component_count, variant, seed)
            if key in score_index:
                raise ValueError("Duplicate development evaluation block.")
            score_index[key] = {
                "xp_mean": scores[seed],
                "evaluation_key_schedule_sha256": payload[
                    "episode_key_schedule_sha256"
                ],
            }
        evaluation_sources.append(
            {"path": str(artifact_path), "sha256": sha256_path(artifact_path)}
        )
    expected = {
        (
            int(row["protocol_components"]),
            str(row["variant"]),
            int(row["seed_index"]),
        )
        for row in matrix["entries"]
    }
    if set(score_index) != expected:
        raise ValueError("Development scores do not cover the complete matrix.")
    seeds = sorted({seed for _, _, seed in expected})
    component_counts = sorted({count for count, _, _ in expected})
    if seeds != list(range(10)):
        raise ValueError("Development summary requires registered seeds 0--9.")
    for component_count in component_counts:
        for seed in seeds:
            schedules = {
                score_index[(component_count, variant, seed)][
                    "evaluation_key_schedule_sha256"
                ]
                for variant in DEVELOPMENT_VARIANTS
            }
            if len(schedules) != 1:
                raise ValueError("Paired variants do not share evaluation episode keys.")
    values_by_k = {
        component_count: {
            variant: np.asarray(
                [
                    float(
                        score_index[(component_count, variant, seed)]["xp_mean"]
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            for variant in DEVELOPMENT_VARIANTS
        }
        for component_count in component_counts
    }
    comparisons_by_k = {}
    for component_count, values in values_by_k.items():
        comparisons = {}
        for name, left, right, seed in (
            ("b0_minus_r0", "b0", "r0", 9),
            ("b1_minus_b0", "b1", "b0", 10),
            ("b2_minus_b1", "b2", "b1", 11),
            ("b2_minus_b0", "b2", "b0", 12),
            ("b2_minus_r0_total_budget", "b2", "r0_extra", 13),
            ("b2_minus_b0_total_budget", "b2", "b0_extra", 14),
            ("b2_minus_b1_total_budget", "b2", "b1_extra", 15),
            ("b2_minus_deterministic_context", "b2", "deterministic_context", 16),
            ("b2_minus_decision_only", "b2", "decision_only", 17),
            ("b2_minus_q_only", "b2", "q_only", 18),
            ("b2_minus_actor_only", "b2", "actor_only", 19),
            ("b2_minus_no_separation", "b2", "no_separation", 20),
            ("b2_minus_no_capability", "b2", "no_capability", 21),
        ):
            point, interval = _paired_bootstrap(
                values[left] - values[right], seed=seed + 100 * component_count
            )
            comparisons[name] = {
                "paired_mean_increment": point,
                "bootstrap_99_percent_ci": interval,
            }
        comparisons_by_k[str(component_count)] = comparisons
    component_stability = {}
    component_sources = []
    for component_count in component_counts:
        signatures = {}
        for seed in seeds:
            run = Path(
                str(matrix_index[(component_count, "b2", seed)]["run"])
            ).resolve()
            diagnostic_path = (
                run / "records" / "final_component_diagnostics.json"
            )
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if (
                payload.get("version") != 1
                or payload.get("artifact_type")
                != "depi_exchangeable_response_regime_diagnostics"
                or payload.get("method") != METHOD_VERSION
                or payload.get("method_variant") != "b2"
                or int(payload.get("seed_index", -1)) != seed
                or int(payload.get("protocol_components", -1))
                != component_count
                or payload.get("fresh_final_policy_anchors") is not True
                or payload.get("one_hot_component_intervention") is not True
            ):
                raise ValueError("Final component diagnostic identity differs.")
            validate_component_diagnostic_values(
                payload, component_count=component_count
            )
            signatures[seed] = payload["component_action_signatures"]
            component_sources.append(
                {
                    "path": str(diagnostic_path),
                    "sha256": sha256_path(diagnostic_path),
                }
            )
        component_stability[str(component_count)] = (
            permutation_aligned_component_stability(signatures)
        )
    entries = matrix["entries"]
    compute = {
        variant: {
            "total_training_simulator_steps": int(
                sum(
                    row["total_training_simulator_steps"]
                    for row in entries
                    if row["variant"] == variant
                )
            ),
            "gpu_hours": float(
                sum(row["gpu_hours"] for row in entries if row["variant"] == variant)
            ),
        }
        for variant in DEVELOPMENT_VARIANTS
    }
    result = {
        "version": 2,
        "artifact_type": "depi_development_matrix_summary",
        "method": METHOD_VERSION,
        "paired_seed_count": len(seeds),
        "xp_by_protocol_components": {
            str(component_count): {
                variant: {
                    "mean": float(np.mean(value)),
                    "per_seed": value.tolist(),
                }
                for variant, value in values.items()
            }
            for component_count, values in values_by_k.items()
        },
        "paired_increments_by_protocol_components": comparisons_by_k,
        "primary_k4_paired_increments": comparisons_by_k["4"],
        "component_permutation_aligned_stability": component_stability,
        "total_compute": compute,
        "b3_status": "not_implemented",
        "sources": {
            "matrix": {"path": str(matrix_path), "sha256": sha256_path(matrix_path)},
            "evaluations": evaluation_sources,
            "component_diagnostics": component_sources,
        },
    }
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "summarize-development-matrix",
            "method": METHOD_VERSION,
            "repository_runtime": runtime_provenance(),
            "sources": result["sources"],
        },
    )
    write_json(output / "development_matrix_summary.json", result)


__all__ = [
    "CORE_DEVELOPMENT_VARIANTS",
    "DEVELOPMENT_VARIANTS",
    "TOTAL_BUDGET_VARIANTS",
    "evaluate_development_matrix",
    "run_development_matrix",
    "summarize_development_matrix",
    "validate_development_entry_alignment",
]
