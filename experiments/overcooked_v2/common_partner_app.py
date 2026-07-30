"""Shared mechanism-disjoint partner evaluation for every formal method."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.official_adapter import (
    official_delivery_counts,
    official_pairing_rollouts,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_evaluation_app import (
    FORMAL_METHODS,
    _env_kwargs,
    _load_policies,
    _load_policy_manifest,
)
from experiments.overcooked_v2.official_policy import OfficialDeltaPolicy
from src.path_c.experiment import OFFICIAL_EVALUATION_ROOT_SEED, load_config, load_partner_manifest
from src.path_c.official_statistics import (
    OFFICIAL_BOOTSTRAP_REPLICATES,
    common_partner_bootstrap,
    common_partner_summary,
    registered_bootstrap_seed,
)
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    read_parquet,
    read_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)


COMMON_PARTNER_MECHANISMS = ("rnn-sp", "state-augmented", "rnn-op", "fcp")


def _validate_common_panel(panel: Any, manifests: Mapping[str, Mapping[str, Any]]) -> Any:
    runs = panel.by_role("confirmatory")
    if len(runs) != 16:
        raise ValueError("Formal Common-Partner panel must contain exactly 16 runs.")
    counts = Counter(run.generation_mechanism for run in runs)
    if set(counts) != set(COMMON_PARTNER_MECHANISMS) or set(counts.values()) != {4}:
        raise ValueError(
            "Common panel must contain four fresh runs from each registered "
            f"mechanism {COMMON_PARTNER_MECHANISMS}."
        )
    if len({run.parent_training_run_id for run in runs}) != 16:
        raise ValueError("Every Common-Partner policy needs an independent parent run.")
    if len({tuple(run.jax_prng_key or ()) for run in runs}) != 16:
        raise ValueError("Every Common-Partner policy needs an independent JAX key.")
    if any(run.owner_seed_index is not None for run in runs):
        raise ValueError("Common-Partner runs cannot belong to a DELTA outer run.")

    panel_hashes = {run.checkpoint_sha256 for run in runs}
    panel_parents = {run.parent_training_run_id for run in runs}
    panel_groups = {
        run.co_training_group_id for run in runs if run.co_training_group_id is not None
    }
    for method, manifest in manifests.items():
        for row in manifest["training_lineage"]:
            collisions = []
            if row["checkpoint_sha256"] in panel_hashes:
                collisions.append("checkpoint")
            if row["parent_training_run_id"] in panel_parents:
                collisions.append("parent")
            group = row["co_training_group_id"]
            if group is not None and group in panel_groups:
                collisions.append("co-training-group")
            if collisions:
                raise ValueError(
                    f"Common panel leaks into {method} training lineage via {collisions}."
                )
    return runs


def _official_environment(config: Any) -> Any:
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

    return OvercookedV2(layout=config.environment.layout, **dict(_env_kwargs(config)))


def _parse_manifests(values: list[str], *, layout: str) -> Mapping[str, Mapping[str, Any]]:
    output = {}
    for value in values:
        try:
            method, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("Policy manifests must use METHOD=/path syntax.") from error
        if method in output:
            raise ValueError(f"Duplicate policy manifest method: {method}")
        manifest = _load_policy_manifest(raw_path, expected_layout=layout)
        if manifest["method"] != method:
            raise ValueError(f"Policy manifest method label differs for {method}.")
        output[method] = manifest
    if set(output) != set(FORMAL_METHODS):
        raise ValueError("Common evaluation requires SP, SA, OP, FCP, and DELTA manifests.")
    return output


def run_common_partner_evaluation(args: argparse.Namespace) -> None:
    import jax

    validate_formal_repository_state()
    validate_registered_python_runtime()
    runtime = validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifests = _parse_manifests(args.policy_manifest, layout=config.environment.layout)
    panel_path = Path(args.partner_manifest).resolve()
    panel = load_partner_manifest(
        panel_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    partners = _validate_common_panel(panel, manifests)
    br_prox_path = Path(args.br_prox_result).resolve()
    br_identity = read_run_identity(br_prox_path)
    if (
        br_identity.get("stage") != "common-partner-local-br-prox"
        or br_identity.get("layout") != config.environment.layout
        or br_identity.get("config_fingerprint") != config.fingerprint
        or br_identity.get("panel_sha256") != sha256_path(panel_path)
        or br_identity.get("policy_manifests") != manifests
    ):
        raise ValueError("Common BR-Prox identity differs from this frozen evaluation.")
    br_prox_rows = read_parquet(br_prox_path / "common_partner_br_prox.parquet")
    expected_br_rows = (
        len(FORMAL_METHODS)
        * 10
        * 16
        * 2
        * config.evaluation.br_prox_anchors_per_pairing
    )
    if len(br_prox_rows) != expected_br_rows:
        raise ValueError(
            f"Common BR-Prox artifact has {len(br_prox_rows)} rows; expected "
            f"{expected_br_rows}."
        )
    expected_methods = set(FORMAL_METHODS)
    if {str(row["method"]) for row in br_prox_rows} != expected_methods:
        raise ValueError("Common BR-Prox artifact does not cover all five methods.")
    if {str(row["partner_run_id"]) for row in br_prox_rows} != {
        run.run_id for run in partners
    }:
        raise ValueError("Common BR-Prox artifact belongs to another partner panel.")
    if any(
        row.get("scope")
        != "one_action_deviation_with_frozen_deployed_continuation"
        for row in br_prox_rows
    ):
        raise ValueError("Common BR-Prox artifact overstates or changes its scope.")
    output = Path(args.output).resolve()
    identity = {
        "stage": "common-partner-evaluate",
        "layout": config.environment.layout,
        "official_runtime": runtime,
        "repository_runtime": runtime_provenance(),
        "config_fingerprint": config.fingerprint,
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_path(panel_path),
            "content": panel.to_mapping(),
        },
        "policy_manifests": manifests,
        "br_prox_artifact": {
            "path": str(br_prox_path),
            "sha256": sha256_path(br_prox_path),
        },
        "evaluation_root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
    }
    ensure_run_identity(output, identity)
    partner_policies = []
    for run in partners:
        official_config, params = restore_official_checkpoint(run.checkpoint)
        partner_policies.append(official_policy(params, official_config))
    environment = _official_environment(config)
    root_key = jax.random.PRNGKey(OFFICIAL_EVALUATION_ROOT_SEED)
    all_rows = []
    summaries = {}
    for method in FORMAL_METHODS:
        ego_policies = _load_policies(manifests[method], config)
        method_rows = []
        for ego_index, ego in enumerate(ego_policies):
            ego_run_id = str(manifests[method]["runs"][ego_index]["run_id"])
            for partner_index, (partner_run, partner) in enumerate(
                zip(partners, partner_policies, strict=True)
            ):
                for ego_role in (0, 1):
                    left, right = (ego, partner) if ego_role == 0 else (partner, ego)
                    rollouts, episode_keys = official_pairing_rollouts(
                        left_policy=left,
                        right_policy=right,
                        environment=environment,
                        root_key=root_key,
                        episodes=500,
                    )
                    returns = np.asarray(rollouts.total_reward, dtype=np.float64)
                    correct, wrong = official_delivery_counts(
                        environment=environment,
                        rollouts=rollouts,
                        episode_keys=episode_keys,
                    )
                    correct = np.asarray(correct, dtype=np.int64)
                    wrong = np.asarray(wrong, dtype=np.int64)
                    if method == "delta":
                        assert isinstance(ego, OfficialDeltaPolicy)
                        base_ego = OfficialDeltaPolicy(ego.deployment, force_base=True)
                        base_left, base_right = (
                            (base_ego, partner)
                            if ego_role == 0
                            else (partner, base_ego)
                        )
                        base_rollouts, unused_keys = official_pairing_rollouts(
                            left_policy=base_left,
                            right_policy=base_right,
                            environment=environment,
                            root_key=root_key,
                            episodes=500,
                        )
                        del unused_keys
                        base_returns = np.asarray(
                            base_rollouts.total_reward, dtype=np.float64
                        )
                    for episode_index in range(500):
                        row = {
                            "layout": config.environment.layout,
                            "method": method,
                            "ego_run_id": ego_run_id,
                            "ego_run_index": ego_index,
                            "partner_run_id": partner_run.run_id,
                            "partner_run_index": partner_index,
                            "partner_mechanism": partner_run.generation_mechanism,
                            "ego_role": ego_role,
                            "episode_index": episode_index,
                            "raw_return": float(returns[episode_index]),
                            "correct_deliveries": int(correct[episode_index]),
                            "wrong_deliveries": int(wrong[episode_index]),
                        }
                        if method == "delta":
                            row["base_raw_return"] = float(
                                base_returns[episode_index]
                            )
                        method_rows.append(row)
                        all_rows.append(row)
        summaries[method] = {
            **common_partner_summary(method_rows),
            "mean_correct_deliveries": float(
                np.mean([row["correct_deliveries"] for row in method_rows])
            ),
            "mean_wrong_deliveries": float(
                np.mean([row["wrong_deliveries"] for row in method_rows])
            ),
            "br_prox": float(
                np.mean(
                    [
                        row["br_prox"]
                        for row in br_prox_rows
                        if row["method"] == method
                    ]
                )
            ),
        }
    rows_by_method = {
        method: [row for row in all_rows if row["method"] == method]
        for method in FORMAL_METHODS
    }
    bootstrap = common_partner_bootstrap(
        rows_by_method,
        delta_method="delta",
        baseline_methods=("sp", "state-augmented", "op", "fcp"),
        replicates=OFFICIAL_BOOTSTRAP_REPLICATES,
        seed=registered_bootstrap_seed(
            config.environment.layout, "common-partner-scoreboard"
        ),
        alpha=0.05,
    )
    worst_mechanism_margin = min(
        summaries["delta"]["mechanism_means"][mechanism]
        - max(
            summaries[baseline]["mechanism_means"][mechanism]
            for baseline in ("sp", "state-augmented", "op", "fcp")
        )
        for mechanism in summaries["delta"]["mechanism_means"]
    )
    result = {
        "methods": summaries,
        "bootstrap": bootstrap,
        "common_partner_gate_passed": bootstrap["one_sided_lcb"] > 0.0,
        "worst_mechanism_margin_point": float(worst_mechanism_margin),
        "br_prox_complete": True,
        "mechanism_claims_unlocked": False,
        "mechanism_claims_reason": (
            "Performance gates and preregistered component-ablation gates must "
            "be combined before any mechanism claim."
        ),
    }
    write_parquet(output / "common_partner_episodes.parquet", all_rows)
    write_json(output / "common_partner_summary.json", result)
    table_rows = []
    for method in FORMAL_METHODS:
        current = summaries[method]
        table_rows.append(
            {
                "method": method,
                "mean_common_partner_return": current[
                    "mean_common_partner_return"
                ],
                "worst_mechanism_return": current["worst_mechanism_return"],
                "partner_cvar_10": current["partner_cvar_10"],
                "negative_transfer_rate": current["negative_transfer_rate"],
                "br_prox": current["br_prox"],
                "mean_correct_deliveries": current["mean_correct_deliveries"],
                "mean_wrong_deliveries": current["mean_wrong_deliveries"],
            }
        )
    with (output / "common_partner_scoreboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "# Common-Partner Scoreboard",
        "",
        "| Method | Mean | Worst mechanism | Partner-CVaR10 | Negative transfer | BR-Prox | Correct deliveries | Wrong deliveries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        negative = row["negative_transfer_rate"]
        lines.append(
            f"| {row['method']} | {row['mean_common_partner_return']:.6f} | "
            f"{row['worst_mechanism_return']:.6f} | "
            f"{row['partner_cvar_10']:.6f} | "
            f"{'N/A' if negative is None else f'{float(negative):.6f}'} | "
            f"{row['br_prox']:.6f} | "
            f"{row['mean_correct_deliveries']:.6f} | "
            f"{row['mean_wrong_deliveries']:.6f} |"
        )
    lines.extend(
        (
            "",
            f"DELTA−best-baseline one-sided LCB: "
            f"{bootstrap['one_sided_lcb']:.6f}.",
            "Negative transfer is defined only for DELTA's paired calibrated "
            "conditional policy versus its own robust base.",
        )
    )
    (output / "common_partner_scoreboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    # Five primary method panels plus one paired DELTA base-only panel.
    write_json(
        output / "budget_ledger.json",
        ResourceLedger(
            evaluation_steps=(5 + 1) * 10 * 16 * 2 * 500 * 400
        ).to_mapping(),
    )


__all__ = ["run_common_partner_evaluation"]
