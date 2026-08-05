"""Shared mechanism-disjoint partner evaluation for every formal method."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
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
from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_EVALUATION_ROOT_SEED,
    OFFICIAL_PROTOCOL_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
)
from src.path_c.heuristic_partners import HEURISTIC_RUN_NAMES, official_heuristic_panel
from src.path_c.official_statistics import (
    OFFICIAL_BOOTSTRAP_REPLICATES,
    common_partner_bootstrap,
    common_partner_summary,
    registered_bootstrap_seed,
    registered_superiority_gate,
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
COMMON_PARTNER_COUNT = 18
COMMON_PARTNER_COUNTS = {
    "rnn-sp": 4,
    "state-augmented": 4,
    "rnn-op": 4,
    "fcp": 4,
    "heuristic": 2,
}
VIRTUAL_PARTNER_PANEL = {
    "family": "heuristic",
    "test_only": True,
    "run_ids": list(HEURISTIC_RUN_NAMES),
}


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
        raise ValueError("Common-Partner runs cannot belong to a DEPI outer run.")

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


def _load_common_panel_policies(partners: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Load 16 frozen checkpoints and append two virtual heuristics."""

    descriptors = list(partners)
    policies = []
    for run in partners:
        official_config, params = restore_official_checkpoint(run.checkpoint)
        policies.append(official_policy(params, official_config))
    for descriptor, policy in official_heuristic_panel():
        descriptors.append(descriptor)
        policies.append(policy)
    if len(descriptors) != COMMON_PARTNER_COUNT:
        raise AssertionError("The Common-Partner panel must contain 18 policies.")
    return tuple(descriptors), tuple(policies)


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
        raise ValueError("Common evaluation requires SP, SA, OP, FCP, and DEPI manifests.")
    return output


def _manifest_sources(values: list[str]) -> Mapping[str, Mapping[str, str]]:
    sources: dict[str, Mapping[str, str]] = {}
    for value in values:
        method, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        sources[method] = {"path": str(path), "sha256": sha256_path(path)}
    return sources


def _episode_key_schedule_sha256(keys: Any) -> str:
    values = np.ascontiguousarray(np.asarray(keys))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def run_common_partner_evaluation(args: argparse.Namespace) -> None:
    import jax

    validate_formal_repository_state()
    validate_registered_python_runtime()
    runtime = validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifests = _parse_manifests(args.policy_manifest, layout=config.environment.layout)
    policy_manifest_sources = _manifest_sources(args.policy_manifest)
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
        or br_identity.get("virtual_partner_panel") != VIRTUAL_PARTNER_PANEL
    ):
        raise ValueError("Common BR-Prox identity differs from this frozen evaluation.")
    br_prox_rows = read_parquet(br_prox_path / "common_partner_br_prox.parquet")
    expected_br_rows = (
        len(FORMAL_METHODS)
        * 10
        * COMMON_PARTNER_COUNT
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
    } | set(HEURISTIC_RUN_NAMES):
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
        "virtual_partner_panel": VIRTUAL_PARTNER_PANEL,
        "br_prox_artifact": {
            "path": str(br_prox_path),
            "sha256": sha256_path(br_prox_path),
        },
        "evaluation_root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
    }
    ensure_run_identity(output, identity)
    partners, partner_policies = _load_common_panel_policies(partners)
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
                    schedule_sha256 = _episode_key_schedule_sha256(episode_keys)
                    correct, wrong = official_delivery_counts(
                        environment=environment,
                        rollouts=rollouts,
                        episode_keys=episode_keys,
                    )
                    correct = np.asarray(correct, dtype=np.int64)
                    wrong = np.asarray(wrong, dtype=np.int64)
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
                            "episode_key_schedule_sha256": schedule_sha256,
                            "raw_return": float(returns[episode_index]),
                            "correct_deliveries": int(correct[episode_index]),
                            "wrong_deliveries": int(wrong[episode_index]),
                        }
                        method_rows.append(row)
                        all_rows.append(row)
        summaries[method] = {
            **common_partner_summary(
                method_rows,
                expected_partner_counts=COMMON_PARTNER_COUNTS,
            ),
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
        target_method="depi",
        baseline_methods=("sp", "state-augmented", "op", "fcp"),
        replicates=OFFICIAL_BOOTSTRAP_REPLICATES,
        seed=registered_bootstrap_seed(
            config.environment.layout, "common-partner-scoreboard"
        ),
        alpha=0.05,
    )
    worst_mechanism_margin = min(
        summaries["depi"]["mechanism_means"][mechanism]
        - max(
            summaries[baseline]["mechanism_means"][mechanism]
            for baseline in ("sp", "state-augmented", "op", "fcp")
        )
        for mechanism in summaries["depi"]["mechanism_means"]
    )
    superiority = registered_superiority_gate(
        bootstrap,
        lcb_threshold=float(config.evaluation.superiority_lcb_threshold),
        minimum_effect=float(config.evaluation.minimum_effect),
        minimum_effect_rule=config.evaluation.minimum_effect_rule,
    )
    schedule_hashes = sorted(
        {str(row["episode_key_schedule_sha256"]) for row in all_rows}
    )
    if len(schedule_hashes) != 1:
        raise RuntimeError("Common-Partner pairings do not share one CRN key schedule.")
    ledger = ResourceLedger(
        evaluation_steps=5 * 10 * COMMON_PARTNER_COUNT * 2 * 500 * 400
    )
    raw_episode_path = output / "common_partner_episodes.parquet"
    write_parquet(raw_episode_path, all_rows)
    result = {
        "version": 2,
        "artifact_type": "depi_common_partner_evaluation",
        "method": METHOD_VERSION,
        "method_variant": "b2",
        "layout": config.environment.layout,
        "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "paired_crn": True,
        "episode_key_schedule_sha256": schedule_hashes[0],
        "sources": {
            "config": {
                "path": str(Path(args.config).resolve()),
                "sha256": sha256_path(Path(args.config).resolve()),
            },
            "partner_manifest": {
                "path": str(panel_path),
                "sha256": sha256_path(panel_path),
            },
            "policy_manifests": policy_manifest_sources,
            "br_prox": {
                "path": str(br_prox_path),
                "sha256": sha256_path(br_prox_path),
            },
            "raw_episodes": {
                "path": str(raw_episode_path),
                "sha256": sha256_path(raw_episode_path),
            },
        },
        "resource_ledger": ledger.to_mapping(),
        "methods": summaries,
        "bootstrap": bootstrap,
        "common_partner_gate_passed": superiority["passed"],
        "superiority_lcb_passed": superiority["superiority_lcb_passed"],
        "minimum_effect_passed": superiority["minimum_effect_passed"],
        "statistical_preregistration": {
            "superiority_lcb_threshold": config.evaluation.superiority_lcb_threshold,
            "minimum_effect": config.evaluation.minimum_effect,
            "minimum_effect_rule": config.evaluation.minimum_effect_rule,
        },
        "worst_mechanism_margin_point": float(worst_mechanism_margin),
        "br_prox_complete": True,
        "mechanism_claims_unlocked": False,
        "mechanism_claims_reason": (
            "Performance gates and preregistered component-ablation gates must "
            "be combined before any mechanism claim."
        ),
    }
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
            f"DEPI−best-baseline one-sided LCB: "
            f"{bootstrap['one_sided_lcb']:.6f}.",
            "Negative transfer is not a primary DEPI field because the method has "
            "one actor and no deployment fallback branch.",
        )
    )
    (output / "common_partner_scoreboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(output / "budget_ledger.json", ledger.to_mapping())


__all__ = [
    "COMMON_PARTNER_COUNT",
    "VIRTUAL_PARTNER_PANEL",
    "_load_common_panel_policies",
    "run_common_partner_evaluation",
]
