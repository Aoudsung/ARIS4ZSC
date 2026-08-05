"""Unmodified Official evaluator integration and 10x10 scoreboard output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.deployment import (
    load_deployment,
    load_self_contained_deployment,
)
from experiments.overcooked_v2.contemporary_baseline_app import (
    CONTEMPORARY_PROXY_METHODS,
)
from experiments.overcooked_v2.official_adapter import (
    _official_symbol,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_policy import (
    OfficialDEPIPolicy,
    assert_official_policy_surface,
)
from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_EPISODE_STEPS,
    OFFICIAL_CORRECT_DELIVERY_REWARD,
    OFFICIAL_EVALUATION_ROOT_SEED,
    OFFICIAL_PROTOCOL_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    RunConfig,
    load_config,
)
from src.path_c.official_statistics import (
    OFFICIAL_BOOTSTRAP_REPLICATES,
    OFFICIAL_EPISODES,
    official_node_bootstrap,
    official_pairings,
    official_scoreboard_summary,
    official_two_method_bootstrap,
    rows_to_official_cube,
    registered_bootstrap_seed,
    registered_superiority_gate,
)
from src.path_c.m1_gate import (
    M1_MINIMUM_ANCHOR_FRACTION,
    M1_SPEARMAN_THRESHOLD,
)
from src.path_c.resources import ResourceLedger, parameter_count
from src.path_c.storage import (
    ensure_run_identity,
    read_parquet,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)


FORMAL_METHODS = ("sp", "state-augmented", "op", "fcp", "depi")
EXTENDED_METHODS = ("ippo-large",)


def _validate_final_m1_payload(
    payload: Mapping[str, Any],
    *,
    layout: str,
    seed_index: int,
    model_fingerprint: str,
) -> bool:
    """Validate and recompute a final-checkpoint M1 decision."""

    expected_fields = {
        "version",
        "artifact_type",
        "method",
        "method_variant",
        "layout",
        "seed_index",
        "specification",
        "history",
        "latest",
        "final",
    }
    final_fields = {
        "update",
        "final_checkpoint_condition",
        "model_fingerprint",
        "deployment_params_fingerprint",
        "anchor_collection_policy_fingerprint",
        "continuation_policy_fingerprint",
        "partner_panel_fingerprint",
        "fit_key_domain",
        "evaluation_key_domain",
        "fresh_final_anchor",
        "bootstrap_reinitialized_after_deployment_freeze",
        "bootstrap_training",
        "m1_gate_passed",
        "m1_path_passing_fractions",
        "m1_path_mean_spearman",
        "m1_path_top_action_agreement",
        "m1_path_mean_value_regret",
        "m1_spearman_threshold",
        "m1_minimum_anchor_fraction",
    }
    final = payload.get("final")
    history = payload.get("history")
    if (
        set(payload) != expected_fields
        or payload.get("version") != 1
        or payload.get("artifact_type") != "depi_final_m1_evaluation"
        or payload.get("method") != METHOD_VERSION
        or payload.get("method_variant") != "b2"
        or payload.get("layout") != layout
        or int(payload.get("seed_index", -1)) != int(seed_index)
        or not isinstance(final, Mapping)
        or set(final) != final_fields
        or payload.get("latest") != final
        or not isinstance(history, list)
        or not history
        or final.get("final_checkpoint_condition") is not True
        or final.get("model_fingerprint") != model_fingerprint
        or final.get("deployment_params_fingerprint") != model_fingerprint
        or final.get("anchor_collection_policy_fingerprint") != model_fingerprint
        or final.get("continuation_policy_fingerprint") != model_fingerprint
        or final.get("fresh_final_anchor") is not True
        or final.get("bootstrap_reinitialized_after_deployment_freeze") is not True
        or not isinstance(final.get("partner_panel_fingerprint"), str)
        or len(final.get("partner_panel_fingerprint", "")) != 64
        or not isinstance(final.get("fit_key_domain"), Mapping)
        or not isinstance(final.get("evaluation_key_domain"), Mapping)
        or not np.isclose(
            float(final.get("m1_spearman_threshold", -1.0)),
            M1_SPEARMAN_THRESHOLD,
            atol=1.0e-6,
        )
        or not np.isclose(
            float(final.get("m1_minimum_anchor_fraction", -1.0)),
            M1_MINIMUM_ANCHOR_FRACTION,
            atol=1.0e-6,
        )
    ):
        raise ValueError("Final-checkpoint M1 artifact identity differs.")
    fit_domain = final["fit_key_domain"]
    evaluation_domain = final["evaluation_key_domain"]
    domain_fields = {
        "name", "root", "lane_key_derivation", "replica_index_range"
    }
    if (
        set(fit_domain) != domain_fields
        or set(evaluation_domain) != domain_fields
        or fit_domain["name"] == evaluation_domain["name"]
        or fit_domain["root"] != evaluation_domain["root"]
        or fit_domain["lane_key_derivation"]
        != "fold_in(root,100003),fold_in(replica_index)"
        or evaluation_domain["lane_key_derivation"]
        != fit_domain["lane_key_derivation"]
        or not isinstance(fit_domain["replica_index_range"], list)
        or not isinstance(evaluation_domain["replica_index_range"], list)
        or len(fit_domain["replica_index_range"]) != 2
        or len(evaluation_domain["replica_index_range"]) != 2
        or int(fit_domain["replica_index_range"][0]) != 0
        or int(fit_domain["replica_index_range"][1]) <= 0
        or int(fit_domain["replica_index_range"][1])
        != int(evaluation_domain["replica_index_range"][0])
        or int(evaluation_domain["replica_index_range"][1])
        <= int(evaluation_domain["replica_index_range"][0])
    ):
        raise ValueError("Final-checkpoint M1 fit/evaluation key domains overlap.")
    arrays = {
        name: final.get(name)
        for name in (
            "m1_path_passing_fractions",
            "m1_path_mean_spearman",
            "m1_path_top_action_agreement",
            "m1_path_mean_value_regret",
        )
    }
    if any(
        not isinstance(values, list)
        or len(values) != 4
        or not all(np.isfinite(float(value)) for value in values)
        for values in arrays.values()
    ):
        raise ValueError("Final-checkpoint M1 path metrics are incomplete or non-finite.")
    fractions = [float(value) for value in arrays["m1_path_passing_fractions"]]
    agreements = [float(value) for value in arrays["m1_path_top_action_agreement"]]
    regrets = [float(value) for value in arrays["m1_path_mean_value_regret"]]
    derived_pass = bool(
        all(0.0 <= value <= 1.0 for value in fractions)
        and all(0.0 <= value <= 1.0 for value in agreements)
        and all(value >= 0.0 for value in regrets)
        and all(value + 1.0e-6 >= M1_MINIMUM_ANCHOR_FRACTION for value in fractions)
    )
    last = history[-1]
    if (
        not isinstance(last, Mapping)
        or int(last.get("update", -1)) != int(final["update"])
        or last.get("passed") is not derived_pass
        or last.get("path_passing_fractions")
        != final["m1_path_passing_fractions"]
        or last.get("path_mean_spearman") != final["m1_path_mean_spearman"]
        or last.get("path_top_action_agreement")
        != final["m1_path_top_action_agreement"]
        or last.get("path_mean_value_regret")
        != final["m1_path_mean_value_regret"]
        or final.get("m1_gate_passed") is not derived_pass
    ):
        raise ValueError("Final-checkpoint M1 pass field is not metric-derived.")
    return derived_pass


def run_build_depi_policy_manifest(args: argparse.Namespace) -> None:
    import orbax.checkpoint as ocp

    validate_formal_repository_state()
    validate_registered_python_runtime()
    artifacts = tuple(Path(value).resolve() for value in args.deployments)
    if len(artifacts) != 10:
        raise ValueError("DEPI policy manifest requires exactly ten deployments.")
    runs = []
    m1_final_evaluations: list[Mapping[str, Any]] = []
    lineage_rows: list[Mapping[str, Any]] = []
    seed_indexes = set()
    layout = None
    deployment_parameter_counts = []
    for artifact in artifacts:
        bundle_path = artifact / "deployment_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        deployment_parameter_counts.append(
            parameter_count(
                ocp.PyTreeCheckpointer().restore(str(artifact / "params"))
            )
        )
        source_run = Path(str(bundle["source_training_run"])).resolve()
        training_identity = json.loads(
            (source_run / "run_identity.json").read_text(encoding="utf-8")
        )
        seed_index = int(training_identity["seed_index"])
        if seed_index in seed_indexes:
            raise ValueError(f"Duplicate DEPI seed index: {seed_index}")
        seed_indexes.add(seed_index)
        current_layout = str(training_identity["layout"])
        layout = current_layout if layout is None else layout
        if current_layout != layout:
            raise ValueError("DEPI deployments mix layouts.")
        ego_run_id = str(training_identity["ego_run_id"])
        m1_path = source_run / "m1_gate.json"
        m1_payload = json.loads(m1_path.read_text(encoding="utf-8"))
        final_m1 = m1_payload.get("final")
        derived_m1_pass = _validate_final_m1_payload(
            m1_payload,
            layout=current_layout,
            seed_index=seed_index,
            model_fingerprint=str(bundle.get("params_fingerprint", "")),
        )
        m1_final_evaluations.append(
            {
                "run_index": seed_index,
                "run_id": ego_run_id,
                "path": str(m1_path),
                "sha256": sha256_path(m1_path),
                "model_fingerprint": str(final_m1["model_fingerprint"]),
                "passed": derived_m1_pass,
            }
        )
        runs.append(
            {
                "run_index": seed_index,
                "run_id": ego_run_id,
                "parent_training_run_id": ego_run_id,
                "co_training_group_id": None,
                "checkpoint": str(artifact),
                "checkpoint_sha256": sha256_path(artifact),
            }
        )
        lineage_rows.append(
            {
                "checkpoint_sha256": sha256_path(artifact),
                "parent_training_run_id": ego_run_id,
                "co_training_group_id": None,
                "role": "formal_ego",
            }
        )
        for raw in training_identity["partner_manifest"]["runs"]:
            if raw["role"] not in {
                "owner_source",
                "development_support",
                "comparator_fit",
                "comparator_validation",
            }:
                continue
            lineage_rows.append(
                {
                    "checkpoint_sha256": str(raw["checkpoint_sha256"]),
                    "parent_training_run_id": str(raw["parent_training_run_id"]),
                    "co_training_group_id": raw["co_training_group_id"],
                    "role": f"depi_training_{raw['role']}",
                }
            )
    if seed_indexes != set(range(10)) or layout is None:
        raise ValueError("DEPI deployments must cover Official seed indexes 0..9.")
    if len(set(deployment_parameter_counts)) != 1:
        raise ValueError("DEPI deployments have inconsistent parameter counts.")
    unique_lineage = {
        (
            row["checkpoint_sha256"],
            row["parent_training_run_id"],
            row["co_training_group_id"],
            row["role"],
        ): row
        for row in lineage_rows
    }
    write_json(
        args.output,
        {
            "version": 1,
            "layout": layout,
            "method": "depi",
            "policy_kind": "depi_deployment",
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "runs": sorted(runs, key=lambda row: row["run_index"]),
            "training_lineage": list(unique_lineage.values()),
            "deployment_parameter_count": deployment_parameter_counts[0],
            "m1_final_evaluations": sorted(
                m1_final_evaluations, key=lambda row: row["run_index"]
            ),
        },
    )


def _load_policy_manifest(path: str | Path, *, expected_layout: str) -> Mapping[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "version",
        "layout",
        "method",
        "policy_kind",
        "official_source_commit",
        "runs",
        "training_lineage",
    }
    method = str(payload.get("method", "")) if isinstance(payload, Mapping) else ""
    if method in EXTENDED_METHODS:
        expected.add("capacity_match")
    if method in CONTEMPORARY_PROXY_METHODS:
        expected.add("proxy_metadata")
    if method == "depi":
        expected.update({"deployment_parameter_count", "m1_final_evaluations"})
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Policy manifest fields differ from the formal schema.")
    if int(payload["version"]) != 1 or str(payload["layout"]) != expected_layout:
        raise ValueError("Policy manifest version/layout mismatch.")
    if str(payload["method"]) not in (
        *FORMAL_METHODS,
        *EXTENDED_METHODS,
        *CONTEMPORARY_PROXY_METHODS,
    ):
        raise ValueError("Policy manifest method is not registered.")
    if method in EXTENDED_METHODS:
        match = payload["capacity_match"]
        required_match = {
            "hidden_dimension",
            "target_parameters",
            "observed_parameters",
            "absolute_mismatch",
        }
        if not isinstance(match, Mapping) or set(match) != required_match:
            raise ValueError("IPPO-Large capacity-match metadata is incomplete.")
        if abs(int(match["observed_parameters"]) - int(match["target_parameters"])) != int(
            match["absolute_mismatch"]
        ):
            raise ValueError("IPPO-Large capacity mismatch metadata is inconsistent.")
    if method == "depi" and int(payload["deployment_parameter_count"]) <= 0:
        raise ValueError("DEPI deployment parameter count must be positive.")
    if method in CONTEMPORARY_PROXY_METHODS:
        metadata = payload["proxy_metadata"]
        if (
            not isinstance(metadata, Mapping)
            or set(metadata)
            != {
                "is_proxy",
                "not_a_published_method_reproduction",
                "closest_in_repository_variant",
                "description",
                "legal_information_boundary",
                "partner_manifest_sha256",
                "ego_policy_steps_per_run",
                "deployable_parameter_count",
                "depi_reference_manifest",
                "same_training_partner_pool",
                "same_ego_interaction_budget",
                "same_deployable_capacity",
            }
            or metadata["is_proxy"] is not True
            or metadata["not_a_published_method_reproduction"] is not True
            or metadata["same_training_partner_pool"] is not True
            or metadata["same_ego_interaction_budget"] is not True
            or metadata["same_deployable_capacity"] is not True
            or int(metadata["ego_policy_steps_per_run"]) <= 0
            or int(metadata["deployable_parameter_count"]) <= 0
        ):
            raise ValueError("Contemporary baseline proxy metadata differs.")
        reference = metadata["depi_reference_manifest"]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or sha256_path(Path(str(reference["path"])).resolve())
            != reference["sha256"]
        ):
            raise ValueError("Contemporary baseline DEPI reference differs.")
    if str(payload["policy_kind"]) not in {"official_ppo", "depi_deployment"}:
        raise ValueError("Policy manifest has an unknown policy kind.")
    if str(payload["official_source_commit"]) != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("Policy manifest is not bound to the fixed Official commit.")
    runs = payload["runs"]
    if not isinstance(runs, Sequence) or len(runs) != 10:
        raise ValueError("Formal policy manifest must contain exactly ten runs.")
    indexes = set()
    normalized = []
    for raw in runs:
        if not isinstance(raw, Mapping) or set(raw) != {
            "run_index",
            "run_id",
            "parent_training_run_id",
            "co_training_group_id",
            "checkpoint",
            "checkpoint_sha256",
        }:
            raise ValueError("Formal policy-run manifest fields differ.")
        index = int(raw["run_index"])
        artifact = Path(str(raw["checkpoint"])).resolve()
        if index in indexes or not 0 <= index < 10:
            raise ValueError("Policy run indexes must be exactly 0..9.")
        indexes.add(index)
        observed = sha256_path(artifact)
        if observed != str(raw["checkpoint_sha256"]):
            raise ValueError(f"Policy artifact hash mismatch: {artifact}")
        normalized.append({**dict(raw), "checkpoint": str(artifact)})
    if indexes != set(range(10)):
        raise ValueError("Policy run indexes must be exactly 0..9.")
    if method == "depi":
        m1_rows = payload["m1_final_evaluations"]
        m1_fields = {
            "run_index",
            "run_id",
            "path",
            "sha256",
            "model_fingerprint",
            "passed",
        }
        if not isinstance(m1_rows, Sequence) or len(m1_rows) != 10:
            raise ValueError("DEPI manifest must bind ten final-checkpoint M1 evaluations.")
        m1_by_index: dict[int, Mapping[str, Any]] = {}
        run_by_index = {int(row["run_index"]): row for row in normalized}
        for raw in m1_rows:
            if not isinstance(raw, Mapping) or set(raw) != m1_fields:
                raise ValueError("Final-checkpoint M1 manifest fields differ.")
            index = int(raw["run_index"])
            if index in m1_by_index or index not in run_by_index:
                raise ValueError("Final-checkpoint M1 indexes must be exactly 0..9.")
            m1_path = Path(str(raw["path"])).resolve()
            if sha256_path(m1_path) != str(raw["sha256"]):
                raise ValueError(f"Final-checkpoint M1 source hash mismatch: {m1_path}")
            m1_payload = json.loads(m1_path.read_text(encoding="utf-8"))
            final_m1 = m1_payload.get("final")
            bundle = json.loads(
                (Path(str(run_by_index[index]["checkpoint"])) / "deployment_bundle.json")
                .read_text(encoding="utf-8")
            )
            derived_pass = _validate_final_m1_payload(
                m1_payload,
                layout=expected_layout,
                seed_index=index,
                model_fingerprint=str(bundle.get("params_fingerprint", "")),
            )
            if (
                str(raw["model_fingerprint"]) != bundle.get("params_fingerprint")
                or str(raw["run_id"]) != str(run_by_index[index]["run_id"])
                or raw["passed"] is not derived_pass
            ):
                raise ValueError("Final-checkpoint M1 evidence differs from its deployment.")
            m1_by_index[index] = {**dict(raw), "path": str(m1_path)}
        if set(m1_by_index) != set(range(10)):
            raise ValueError("Final-checkpoint M1 indexes must be exactly 0..9.")
    lineage = payload["training_lineage"]
    if not isinstance(lineage, Sequence) or not lineage:
        raise ValueError("Policy manifest must disclose non-empty training lineage.")
    lineage_fields = {
        "checkpoint_sha256",
        "parent_training_run_id",
        "co_training_group_id",
        "role",
    }
    if any(not isinstance(row, Mapping) or set(row) != lineage_fields for row in lineage):
        raise ValueError("Policy training-lineage rows differ from the formal schema.")
    normalized_payload = {
        **dict(payload),
        "runs": sorted(normalized, key=lambda row: row["run_index"]),
    }
    if method == "depi":
        normalized_payload["m1_final_evaluations"] = [
            m1_by_index[index] for index in range(10)
        ]
    return normalized_payload


def _load_policies(manifest: Mapping[str, Any], config: RunConfig) -> tuple[Any, ...]:
    policies = []
    kind = str(manifest["policy_kind"])
    for run in manifest["runs"]:
        path = Path(str(run["checkpoint"]))
        if kind == "official_ppo":
            official_config, params = restore_official_checkpoint(path)
            policy = official_policy(params, official_config)
        else:
            deployment = (
                load_self_contained_deployment(path)
                if str(manifest["method"]) in CONTEMPORARY_PROXY_METHODS
                else load_deployment(path, config)
            )
            policy = OfficialDEPIPolicy(deployment)
            assert_official_policy_surface(policy)
        policies.append(policy)
    return tuple(policies)


def _env_kwargs(config: RunConfig) -> Mapping[str, Any]:
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType

    return {
        # The locked Official environment accepts a scalar observation type for
        # the homogeneous two-policy evaluation used by Table 2.  Passing the
        # same value as a two-element list exposes an upstream JaxMARL bug:
        # ``OvercookedV2.obs_shape`` then becomes a list of shapes, which is not
        # a legal ``lax.dynamic_slice`` size when partial observations are on.
        "observation_type": ObservationType.DEFAULT,
        "agent_view_size": config.environment.agent_view_size,
        "negative_rewards": config.environment.negative_rewards,
        "random_agent_positions": config.environment.random_agent_positions,
        "sample_recipe_on_delivery": config.environment.sample_recipe_on_delivery,
        "indicate_successful_delivery": config.environment.indicate_successful_delivery,
        "max_steps": OFFICIAL_EPISODE_STEPS,
    }


def _evaluate_cell(
    *, left: Any, right: Any, config: RunConfig, root_key: Any
) -> np.ndarray:
    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    eval_pairing = _official_symbol(
        "overcooked_v2_experiments.eval.evaluate", "eval_pairing"
    )
    outputs = eval_pairing(
        PolicyPairing(left, right),
        config.environment.layout,
        root_key,
        env_kwargs=dict(_env_kwargs(config)),
        num_seeds=OFFICIAL_EPISODES,
        no_viz=True,
    )
    values = np.asarray(
        [outputs[f"seed-{index}"].total_reward for index in range(OFFICIAL_EPISODES)],
        dtype=np.float64,
    )
    if values.shape != (OFFICIAL_EPISODES,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Official evaluator returned an invalid episode vector.")
    return values


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left\\right", *range(matrix.shape[1])])
        for index, row in enumerate(matrix):
            writer.writerow([index, *[float(value) for value in row]])


def _write_cell_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("left_run_index", "right_run_index", "cell_mean")
        )
        writer.writeheader()
        for left, right in official_pairings():
            writer.writerow(
                {
                    "left_run_index": left,
                    "right_run_index": right,
                    "cell_mean": float(matrix[left, right]),
                }
            )


def run_official_evaluation(args: argparse.Namespace) -> None:
    import jax

    validate_formal_repository_state()
    validate_registered_python_runtime()
    runtime = validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifest_path = Path(args.policy_manifest).resolve()
    manifest = _load_policy_manifest(
        manifest_path, expected_layout=config.environment.layout
    )
    output = Path(args.output).resolve()
    identity = {
        "stage": "official-10x10-evaluate",
        "layout": config.environment.layout,
        "method": manifest["method"],
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_runtime": runtime,
        "repository_runtime": runtime_provenance(),
        "config_fingerprint": config.fingerprint,
        "policy_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
            "content": manifest,
        },
        "evaluation_root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
        "episodes_per_cell": OFFICIAL_EPISODES,
    }
    ensure_run_identity(output, identity)
    policies = _load_policies(manifest, config)
    root_key = jax.random.PRNGKey(OFFICIAL_EVALUATION_ROOT_SEED)
    rows = []
    for left, right in official_pairings():
        returns = _evaluate_cell(
            left=policies[left], right=policies[right], config=config, root_key=root_key
        )
        for episode_index, raw_return in enumerate(returns):
            rows.append(
                {
                    "layout": config.environment.layout,
                    "method": str(manifest["method"]),
                    "left_run_index": left,
                    "right_run_index": right,
                    "episode_index": episode_index,
                    "raw_return": float(raw_return),
                }
            )
    cube = rows_to_official_cube(rows)
    summary = official_scoreboard_summary(cube)
    write_parquet(output / "episode_returns.parquet", rows)
    _write_cell_csv(output / "cell_means.csv", np.asarray(summary["cell_means"]))
    _write_matrix_csv(output / "matrix.csv", np.asarray(summary["cell_means"]))
    write_json(output / "summary.json", summary)
    write_json(
        output / "official_protocol.json",
        {
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "layout": config.environment.layout,
            "root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
            "pairings": 100,
            "ordered_cross_play_pairings": 90,
            "self_play_pairings": 10,
            "episodes_per_pairing": 500,
            "episode_steps": 400,
            "return": "sum of raw agent_0 reward",
        },
    )
    ledger = ResourceLedger(evaluation_steps=100 * 500 * 400)
    write_json(output / "budget_ledger.json", ledger.to_mapping())
    report = (
        f"# Official-Protocol Scoreboard\n\n"
        f"- Layout: `{config.environment.layout}`\n"
        f"- Method: `{manifest['method']}`\n"
        f"- SP: {summary['sp_mean']:.6f} ± {summary['sp_population_sd']:.6f}\n"
        f"- XP: {summary['xp_mean']:.6f} ± {summary['xp_population_sd']:.6f}\n"
        f"- Gap (point only): {summary['gap_point']:.6f}\n\n"
        "No Gap standard deviation is reported because the fixed Official source "
        "does not define one.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def _parse_result(value: str) -> tuple[str, str, Path]:
    try:
        label, raw_path = value.split("=", 1)
        layout, method = label.split(":", 1)
    except ValueError as error:
        raise ValueError("Results must use LAYOUT:METHOD=/path syntax.") from error
    if layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError(f"Unknown result layout: {layout}")
    if method not in FORMAL_METHODS:
        raise ValueError(f"Unknown result method: {method}")
    return layout, method, Path(raw_path).resolve()


def _summary_configs(values: Sequence[str]) -> Mapping[str, tuple[RunConfig, Path]]:
    parsed: dict[str, tuple[RunConfig, Path]] = {}
    for value in values:
        try:
            layout, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("Summary configs must use LAYOUT=/path syntax.") from error
        path = Path(raw_path).resolve()
        config = load_config(path, run_kind="formal")
        if config.environment.layout != layout:
            raise ValueError("Summary config label disagrees with its layout.")
        if layout in parsed:
            raise ValueError(f"Duplicate summary config for {layout}.")
        parsed[layout] = (config, path)
    required = {"test_time_simple", "test_time_wide"}
    if set(parsed) != required:
        raise ValueError("Official summary requires one frozen config per layout.")
    return parsed


def _validate_raw_summary_source(
    directory: Path,
    *,
    layout: str,
    method: str,
    config: RunConfig,
) -> Mapping[str, Any]:
    """Validate raw nodes, lineage, key schedule, and evaluation resources."""

    identity = json.loads(
        (directory / "run_identity.json").read_text(encoding="utf-8")
    )
    if (
        identity.get("stage") != "official-10x10-evaluate"
        or identity.get("layout") != layout
        or identity.get("method") != method
        or identity.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
        or identity.get("evaluation_root_key")
        != [0, OFFICIAL_EVALUATION_ROOT_SEED]
        or int(identity.get("episodes_per_cell", -1)) != OFFICIAL_EPISODES
        or identity.get("config_fingerprint") != config.fingerprint
    ):
        raise ValueError("Official evaluation identity disagrees with preregistration.")
    manifest = identity.get("policy_manifest", {}).get("content")
    if not isinstance(manifest, Mapping) or manifest.get("method") != method:
        raise ValueError("Official result has no complete raw policy-node manifest.")
    if len(manifest.get("runs", ())) != 10:
        raise ValueError("Official result does not contain ten raw training nodes.")
    runs = list(manifest["runs"])
    if {int(row["run_index"]) for row in runs} != set(range(10)):
        raise ValueError("Official result does not contain run nodes 0..9 exactly once.")
    if len({str(row["run_id"]) for row in runs}) != 10:
        raise ValueError("Official policy nodes do not have ten independent run ids.")
    if len({str(row["parent_training_run_id"]) for row in runs}) != 10:
        raise ValueError("Official policy nodes reuse a parent training run.")
    if method != "depi" and manifest.get("policy_kind") != "official_ppo":
        raise ValueError("A baseline must be raw evaluated runs, not a published point estimate.")
    lineage = manifest.get("training_lineage")
    if not isinstance(lineage, Sequence) or not lineage:
        raise ValueError("Official policy lineage is incomplete.")
    ledger = ResourceLedger.from_mapping(
        json.loads((directory / "budget_ledger.json").read_text(encoding="utf-8"))
    )
    expected_steps = 100 * OFFICIAL_EPISODES * OFFICIAL_EPISODE_STEPS
    if ledger.evaluation_steps != expected_steps:
        raise ValueError("Official evaluation resource ledger is incomplete.")
    return identity


def run_official_summary(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    parsed = [_parse_result(value) for value in args.result]
    configs = _summary_configs(args.config)
    index = {(layout, method): path for layout, method, path in parsed}
    expected = {
        (layout, method)
        for layout in ("test_time_simple", "test_time_wide")
        for method in FORMAL_METHODS
    }
    if set(index) != expected:
        raise ValueError(
            "Formal summary requires Simple and Wide results for SP, SA, OP, FCP, "
            f"and DEPI; missing={sorted(expected - set(index))}."
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {
        "version": 2,
        "artifact_type": "depi_official_scoreboard_summary",
        "method": METHOD_VERSION,
        "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "layouts": {},
        "source_results": {},
        "statistical_preregistration": {},
    }
    both_pass = True
    for layout in ("test_time_simple", "test_time_wide"):
        config, config_path = configs[layout]
        cubes = {}
        summaries = {}
        identities = {}
        for method in FORMAL_METHODS:
            directory = index[(layout, method)]
            rows = read_parquet(directory / "episode_returns.parquet")
            if {str(row["layout"]) for row in rows} != {layout} or {
                str(row["method"]) for row in rows
            } != {method}:
                raise ValueError("Official result rows disagree with their labels.")
            cubes[method] = rows_to_official_cube(rows)
            summaries[method] = official_scoreboard_summary(cubes[method])
            identity = _validate_raw_summary_source(
                directory, layout=layout, method=method, config=config
            )
            identities[method] = identity
            combined["source_results"][f"{layout}:{method}"] = {
                "path": str(directory),
                "sha256": sha256_path(directory),
            }
        lineage_sets = {}
        for method in FORMAL_METHODS:
            rows = identities[method]["policy_manifest"]["content"][
                "training_lineage"
            ]
            lineage_sets[method] = {
                "parent": {str(row["parent_training_run_id"]) for row in rows},
                "checkpoint": {str(row["checkpoint_sha256"]) for row in rows},
                "co_training_group": {
                    str(row["co_training_group_id"])
                    for row in rows
                    if row["co_training_group_id"] is not None
                },
            }
        for left_index, left in enumerate(FORMAL_METHODS):
            for right in FORMAL_METHODS[left_index + 1 :]:
                collisions = {
                    name: lineage_sets[left][name] & lineage_sets[right][name]
                    for name in lineage_sets[left]
                }
                collisions = {
                    name: values for name, values in collisions.items() if values
                }
                if collisions:
                    raise ValueError(
                        "Formal method lineages are not run/family-disjoint: "
                        f"{left}/{right}, collisions={sorted(collisions)}."
                    )
        bootstrap = official_node_bootstrap(
            cubes,
            target_method="depi",
            baseline_methods=("sp", "state-augmented", "op", "fcp"),
            fcp_method="fcp",
            replicates=OFFICIAL_BOOTSTRAP_REPLICATES,
            seed=registered_bootstrap_seed(layout, "official-scoreboard"),
            alpha=float(config.evaluation.one_sided_alpha),
            inference_mode=config.evaluation.inference_mode,
        )
        delivery_margin = OFFICIAL_CORRECT_DELIVERY_REWARD
        comparison = bootstrap["depi_vs_best_baseline"]
        superiority = registered_superiority_gate(
            comparison,
            lcb_threshold=float(config.evaluation.superiority_lcb_threshold),
            minimum_effect=float(config.evaluation.minimum_effect),
            minimum_effect_rule=config.evaluation.minimum_effect_rule,
        )
        xp_pass = superiority["passed"]
        competence_pass = (
            bootstrap["depi_sp_minus_fcp_sp"]["one_sided_lcb"]
            > -delivery_margin
        )
        both_pass = both_pass and xp_pass and competence_pass
        combined["layouts"][layout] = {
            "methods": summaries,
            "bootstrap": bootstrap,
            "xp_gate_passed": xp_pass,
            "superiority_lcb_passed": superiority["superiority_lcb_passed"],
            "minimum_effect_passed": superiority["minimum_effect_passed"],
            "competence_gate_passed": competence_pass,
            "correct_delivery_margin": delivery_margin,
        }
        combined["statistical_preregistration"][layout] = {
            "config": str(config_path),
            "config_sha256": sha256_path(config_path),
            "config_fingerprint": config.fingerprint,
            "inference_mode": config.evaluation.inference_mode,
            "one_sided_alpha": config.evaluation.one_sided_alpha,
            "superiority_lcb_threshold": config.evaluation.superiority_lcb_threshold,
            "minimum_effect": config.evaluation.minimum_effect,
            "minimum_effect_rule": config.evaluation.minimum_effect_rule,
        }
    combined["primary_benchmark_gate_passed"] = both_pass
    combined["mechanism_claims_unlocked"] = False
    combined["mechanism_claims_reason"] = (
        "Common-Partner Scoreboard and preregistered component ablations are "
        "separate mandatory gates."
    )
    ensure_run_identity(
        output,
        {
            "stage": "official-scoreboard-summary",
            "repository_runtime": runtime_provenance(),
            "bootstrap_replicates": OFFICIAL_BOOTSTRAP_REPLICATES,
            "one_sided_alpha": 0.05,
            "source_results": combined["source_results"],
            "statistical_preregistration": combined[
                "statistical_preregistration"
            ],
        },
    )
    write_json(output / "official_scoreboard_summary.json", combined)
    table_rows = []
    for layout in ("test_time_simple", "test_time_wide"):
        layout_result = combined["layouts"][layout]
        for method in FORMAL_METHODS:
            summary = layout_result["methods"][method]
            table_rows.append(
                {
                    "layout": layout,
                    "method": method,
                    "sp_mean": summary["sp_mean"],
                    "sp_population_sd": summary["sp_population_sd"],
                    "xp_mean": summary["xp_mean"],
                    "xp_population_sd": summary["xp_population_sd"],
                    "gap_point": summary["gap_point"],
                    "depi_vs_best_baseline_one_sided_lcb": (
                        layout_result["bootstrap"]["depi_vs_best_baseline"][
                            "one_sided_lcb"
                        ]
                        if method == "depi"
                        else ""
                    ),
                }
            )
    with (output / "official_scoreboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "# Official Table 2 Scoreboard",
        "",
        "| Layout | Method | SP | XP | Gap point | DEPI LCB vs best baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lcb = row["depi_vs_best_baseline_one_sided_lcb"]
        lines.append(
            f"| {row['layout']} | {row['method']} | "
            f"{row['sp_mean']:.6f} ± {row['sp_population_sd']:.6f} | "
            f"{row['xp_mean']:.6f} ± {row['xp_population_sd']:.6f} | "
            f"{row['gap_point']:.6f} | "
            f"{'' if lcb == '' else f'{float(lcb):.6f}'} |"
        )
    lines.extend(
        (
            "",
            "Gap is a point estimate only; the fixed Official source does not "
            "register a Gap standard-deviation formula. The DEPI lower bound "
            "uses 9,999 registered run-node bootstrap replicates.",
        )
    )
    (output / "official_scoreboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_capacity_result(value: str) -> tuple[str, str, Path]:
    try:
        label, raw_path = value.split("=", 1)
        layout, method = label.split(":", 1)
    except ValueError as error:
        raise ValueError("Capacity results use LAYOUT:METHOD=/path syntax.") from error
    if layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError(f"Unknown capacity layout: {layout}")
    if method not in {"depi", "ippo-large"}:
        raise ValueError(f"Unknown capacity method: {method}")
    return layout, method, Path(raw_path).resolve()


def run_capacity_summary(args: argparse.Namespace) -> None:
    validate_formal_repository_state()
    validate_registered_python_runtime()
    parsed = [_parse_capacity_result(value) for value in args.result]
    index = {(layout, method): path for layout, method, path in parsed}
    expected = {
        (layout, method)
        for layout in ("test_time_simple", "test_time_wide")
        for method in ("depi", "ippo-large")
    }
    if set(index) != expected:
        raise ValueError("Capacity summary requires DEPI and IPPO-Large on both layouts.")
    result: dict[str, Any] = {
        "version": 2,
        "artifact_type": "depi_capacity_control_summary",
        "method": METHOD_VERSION,
        "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "layouts": {},
        "source_results": {},
    }
    both_pass = True
    for layout in ("test_time_simple", "test_time_wide"):
        cubes = {}
        manifests = {}
        for method in ("depi", "ippo-large"):
            directory = index[(layout, method)]
            rows = read_parquet(directory / "episode_returns.parquet")
            if {str(row["layout"]) for row in rows} != {layout} or {
                str(row["method"]) for row in rows
            } != {method}:
                raise ValueError("Capacity result rows disagree with their labels.")
            cubes[method] = rows_to_official_cube(rows)
            identity = json.loads(
                (directory / "run_identity.json").read_text(encoding="utf-8")
            )
            if (
                identity.get("stage") != "official-10x10-evaluate"
                or identity.get("layout") != layout
                or identity.get("method") != method
            ):
                raise ValueError("Capacity result identity disagrees with its label.")
            manifests[method] = identity["policy_manifest"]["content"]
            result["source_results"][f"{layout}:{method}"] = {
                "path": str(directory),
                "sha256": sha256_path(directory),
            }
        target = int(manifests["depi"]["deployment_parameter_count"])
        capacity_match = manifests["ippo-large"]["capacity_match"]
        if int(capacity_match["target_parameters"]) != target:
            raise ValueError("IPPO-Large was not matched to this DEPI deployment size.")
        comparison = official_two_method_bootstrap(
            cubes["depi"],
            cubes["ippo-large"],
            left_name="depi",
            right_name="ippo-large",
            replicates=OFFICIAL_BOOTSTRAP_REPLICATES,
            seed=registered_bootstrap_seed(layout, "capacity-control"),
            alpha=0.05,
        )
        passed = comparison["one_sided_lcb"] > 0.0
        both_pass = both_pass and passed
        result["layouts"][layout] = {
            "depi": official_scoreboard_summary(cubes["depi"]),
            "ippo_large": official_scoreboard_summary(cubes["ippo-large"]),
            "capacity_match": capacity_match,
            "bootstrap": comparison,
            "capacity_explanation_rejected": passed,
        }
    result["capacity_control_gate_passed"] = both_pass
    result["scope"] = (
        "Capacity control only; IPPO-Large is not an Official Table 2 baseline."
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "capacity-control-summary",
            "repository_runtime": runtime_provenance(),
            "bootstrap_replicates": OFFICIAL_BOOTSTRAP_REPLICATES,
            "one_sided_alpha": 0.05,
            "source_results": result["source_results"],
        },
    )
    write_json(output / "capacity_control_summary.json", result)
    lines = [
        "# Parameter-matched capacity control",
        "",
        "| Layout | DEPI XP | IPPO-Large XP | DEPI−IPPO-Large LCB | Passed |",
        "|---|---:|---:|---:|---:|",
    ]
    for layout in ("test_time_simple", "test_time_wide"):
        current = result["layouts"][layout]
        lines.append(
            f"| {layout} | {current['depi']['xp_mean']:.6f} | "
            f"{current['ippo_large']['xp_mean']:.6f} | "
            f"{current['bootstrap']['one_sided_lcb']:.6f} | "
            f"{current['capacity_explanation_rejected']} |"
        )
    (output / "capacity_control_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = [
    "run_build_depi_policy_manifest",
    "run_capacity_summary",
    "run_official_evaluation",
    "run_official_summary",
]
