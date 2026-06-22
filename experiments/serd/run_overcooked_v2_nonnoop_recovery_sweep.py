"""Sweep real OvercookedV2 non-noop semantic branches for recovery evidence.

This runner is a direct oral-evidence repair after the forced-noop positive
control: it enumerates every one-step non-noop action replacement for both
agents at sampled real-policy states, compares each semantic replacement
against matched one-action control families, and writes a full M4 bundle.
It is exploratory evidence, not a gate or fallback path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
from typing import Any

from .fixture_env import CONTROL_FAMILIES
from .m4_bundle import write_m4_bundle
from .run_overcooked_v2_policy_m4 import (
    ACTION_ID_BY_NAME,
    ACTION_NAMES_BY_ID,
    LoadedPolicyPair,
    PolicySpec,
    _action_divergence,
    _advance_policy_prefix,
    _compute_actions,
    _install_remote_paths,
    _load_policy_pair,
    _make_env,
    _parse_ints,
    _parse_policy_spec,
    _rollout_return,
    _sequence_distances,
    _summarize_variant,
    _trace_row,
    _write_csv,
)
from .jaxmarl_overcooked_v2_adapter import state_phi_pre_h
from .serd_core import BranchRecord


CHANNELS = ("action", "position", "state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg-ssa-root",
        type=Path,
        default=Path("/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA"),
    )
    parser.add_argument(
        "--jaxmarl-src",
        type=Path,
        default=Path("/apps/users/cxw/Document/CodeSpace/Selfs/TG-SSA/external/JaxMARL"),
    )
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument(
        "--policy-run",
        action="append",
        required=True,
        help="Policy spec in label:method:run_dir form.",
    )
    parser.add_argument("--run-nums", type=str, required=True)
    parser.add_argument("--pairing", choices=("self", "cross_next"), default="cross_next")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--warmup-horizon", type=int, default=8)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shaped-reward-weight", type=float, default=1.0)
    parser.add_argument("--epsilon-shock", type=float, default=0.001)
    parser.add_argument("--epsilon-phi", type=float, default=0.0)
    parser.add_argument("--delta-serd", type=float, default=0.05)
    parser.add_argument("--delta-variant", type=float, default=0.05)
    parser.add_argument(
        "--suppress-policy-stdout",
        action="store_true",
        help="Suppress noisy policy forward stdout while preserving stderr and final summary.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _replacement_actions(base_action: int) -> list[int]:
    return [
        action_id
        for action_id in sorted(ACTION_NAMES_BY_ID)
        if action_id != base_action
    ]


def _replace_actor_action(
    base_joint: tuple[int, int],
    actor_index: int,
    action_id: int,
) -> tuple[int, int]:
    if action_id == base_joint[actor_index]:
        raise ValueError("semantic replacement must differ from base action")
    branch = list(base_joint)
    branch[actor_index] = action_id
    return (branch[0], branch[1])


def _control_joint_for_template(
    base_joint: tuple[int, int],
    semantic_joint: tuple[int, int],
    actor_index: int,
    family: str,
    probe_index: int,
) -> tuple[int, int]:
    preferred_by_family = {
        "random_lag": ACTION_ID_BY_NAME["interact"],
        "state_block": ACTION_ID_BY_NAME["up"],
        "reward_shaping": ACTION_ID_BY_NAME["down"],
        "naive_replanning": ACTION_ID_BY_NAME["left"],
    }
    ordered = [
        preferred_by_family[family],
        ACTION_ID_BY_NAME["stay"],
        ACTION_ID_BY_NAME["right"],
        ACTION_ID_BY_NAME["down"],
        ACTION_ID_BY_NAME["left"],
        ACTION_ID_BY_NAME["up"],
        ACTION_ID_BY_NAME["interact"],
    ]
    offset = probe_index % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    for action_id in rotated:
        if action_id == base_joint[actor_index]:
            continue
        control = _replace_actor_action(base_joint, actor_index, action_id)
        if control != semantic_joint:
            return control
    raise RuntimeError(
        f"could not build distinct control joint for family={family}, "
        f"base_joint={base_joint}, semantic_joint={semantic_joint}"
    )


def _semantic_label(actor_index: int, action_id: int) -> str:
    return f"actor{actor_index}_to_{ACTION_NAMES_BY_ID[action_id]}"


def _operator_note(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# Operator Note",
        "",
        "No passwords, tokens, shell history, or private credentials are recorded",
        "in this artifact.",
        "",
        "Command class:",
        "",
        "```text",
        "python -m experiments.serd.run_overcooked_v2_nonnoop_recovery_sweep",
        "```",
        "",
        f"Decision/status: `{summary['m4_status']}`",
        f"Output directory: `{output_dir}`",
        f"Policy specs: `{args.policy_run}`",
        f"Run nums: `{args.run_nums}`",
        f"Pairing: `{args.pairing}`",
        "",
        "This run enumerates all one-step non-noop action replacements for both",
        "agents at sampled real-policy states. It is intended to find genuine",
        "non-noop recovery/discrimination evidence, not to validate a gate or",
        "reuse the forced-noop sensitivity control.",
        "",
    ]
    (output_dir / "operator_note.md").write_text("\n".join(lines), encoding="utf-8")


def _build_records_for_pair(
    *,
    env,
    jax,
    pair: LoadedPolicyPair,
    domain: str,
    probes: int,
    warmup_horizon: int,
    rollout_horizon: int,
    shaped_reward_weight: float,
    seed: int,
) -> tuple[
    list[BranchRecord],
    list[BranchRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []
    variant_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []

    for probe_index in range(probes):
        probe_seed = seed + pair.ego_run_num * 10_000 + probe_index
        recovery_seed = seed + 100_000 + pair.partner_run_num * 10_000 + probe_index
        obs, state, done, hstates = _advance_policy_prefix(
            env=env,
            jax=jax,
            pair=pair,
            seed=probe_seed,
            warmup_horizon=warmup_horizon + probe_index,
        )
        phi = state_phi_pre_h(state)
        key = jax.random.PRNGKey(recovery_seed)
        _actions, base_joint, _next_hstates = _compute_actions(
            env,
            pair,
            obs,
            done,
            hstates,
            key,
        )
        no_shock = _rollout_return(
            env=env,
            jax=jax,
            pair=pair,
            obs=obs,
            state=state,
            done=done,
            hstates=hstates,
            seed=recovery_seed,
            horizon=rollout_horizon,
            shaped_reward_weight=shaped_reward_weight,
        )
        for actor_index in (0, 1):
            for action_id in _replacement_actions(base_joint[actor_index]):
                semantic_joint = _replace_actor_action(base_joint, actor_index, action_id)
                disruption = _semantic_label(actor_index, action_id)
                probe_id = f"{pair.label}_{disruption}_{probe_index:03d}"
                semantic = _rollout_return(
                    env=env,
                    jax=jax,
                    pair=pair,
                    obs=obs,
                    state=state,
                    done=done,
                    hstates=hstates,
                    seed=recovery_seed,
                    horizon=rollout_horizon,
                    shaped_reward_weight=shaped_reward_weight,
                    joint_override_sequence=[semantic_joint],
                )
                semantic_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=pair.label,
                        domain=domain,
                        disruption=disruption,
                        family="semantic",
                        no_shock_return=no_shock.total_return,
                        branch_return=semantic.total_return,
                        shock_magnitude=_action_divergence(base_joint, semantic_joint),
                        phi_pre_h=dict(phi),
                    )
                )
                template_rows.append(
                    {
                        "policy": pair.label,
                        "domain": domain,
                        "ego_run_num": pair.ego_run_num,
                        "partner_run_num": pair.partner_run_num,
                        "probe_index": probe_index,
                        "probe_id": probe_id,
                        "base_joint": json.dumps(base_joint),
                        "actor_index": actor_index,
                        "replacement_action": ACTION_NAMES_BY_ID[action_id],
                        "semantic_joint": json.dumps(semantic_joint),
                        "no_shock_return": no_shock.total_return,
                        "semantic_return": semantic.total_return,
                        "semantic_loss": no_shock.total_return - semantic.total_return,
                    }
                )
                trace_rows.extend(
                    [
                        _trace_row(pair, domain, disruption, probe_id, "no_shock", no_shock),
                        _trace_row(pair, domain, disruption, probe_id, "semantic", semantic),
                    ]
                )

                semantic_distances = _sequence_distances(no_shock, semantic)
                for family in CONTROL_FAMILIES:
                    control_joint = _control_joint_for_template(
                        base_joint,
                        semantic_joint,
                        actor_index,
                        family,
                        probe_index,
                    )
                    control = _rollout_return(
                        env=env,
                        jax=jax,
                        pair=pair,
                        obs=obs,
                        state=state,
                        done=done,
                        hstates=hstates,
                        seed=recovery_seed,
                        horizon=rollout_horizon,
                        shaped_reward_weight=shaped_reward_weight,
                        joint_override_sequence=[control_joint],
                    )
                    control_records.append(
                        BranchRecord(
                            probe_id=probe_id,
                            policy=pair.label,
                            domain=domain,
                            disruption=disruption,
                            family=family,
                            no_shock_return=no_shock.total_return,
                            branch_return=control.total_return,
                            shock_magnitude=_action_divergence(base_joint, control_joint),
                            phi_pre_h=dict(phi),
                        )
                    )
                    trace_rows.append(
                        _trace_row(pair, domain, disruption, probe_id, family, control)
                    )
                    control_distances = _sequence_distances(no_shock, control)
                    channel_values = [
                        control_distances[channel] - semantic_distances[channel]
                        for channel in CHANNELS
                    ]
                    variant_rows.append(
                        {
                            "policy": pair.label,
                            "domain": domain,
                            "ego_run_num": pair.ego_run_num,
                            "partner_run_num": pair.partner_run_num,
                            "probe_id": probe_id,
                            "disruption": disruption,
                            "probe_index": probe_index,
                            "actor_index": actor_index,
                            "replacement_action": ACTION_NAMES_BY_ID[action_id],
                            "control_family": family,
                            "control_joint": json.dumps(control_joint),
                            "return_delta": control.total_return - semantic.total_return,
                            "return_independent_serd": mean(channel_values),
                            "semantic_action_distance_to_no_shock": semantic_distances["action"],
                            "control_action_distance_to_no_shock": control_distances["action"],
                            "action_variant_serd": channel_values[0],
                            "semantic_position_distance_to_no_shock": semantic_distances["position"],
                            "control_position_distance_to_no_shock": control_distances["position"],
                            "position_variant_serd": channel_values[1],
                            "semantic_state_distance_to_no_shock": semantic_distances["state"],
                            "control_state_distance_to_no_shock": control_distances["state"],
                            "state_variant_serd": channel_values[2],
                        }
                    )
    return semantic_records, control_records, variant_rows, trace_rows, template_rows


def _summarize_survival_rows(
    standard_worst: list[dict[str, Any]],
    variant_worst: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, source_rows in (
        ("standard_serd_worst", standard_worst),
        ("return_independent_action_position_state_serd", variant_worst),
    ):
        for row in source_rows:
            if row.get("classification") == "survival":
                rows.append(
                    {
                        "metric": metric_name,
                        "policy": row["policy"],
                        "domain": row["domain"],
                        "disruption": row["disruption"],
                        "mean_serd_worst": row["mean_serd_worst"],
                        "ci95_low": row["ci95_low"],
                        "ci95_high": row["ci95_high"],
                        "n": row["n"],
                        "limiting_family": row["limiting_family"],
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    _install_remote_paths(args.tg_ssa_root, args.jaxmarl_src)

    import jax

    policy_specs: list[PolicySpec] = [_parse_policy_spec(item) for item in args.policy_run]
    run_nums = _parse_ints(args.run_nums)
    env_kwargs = {
        "agent_view_size": 2,
        "negative_rewards": True,
        "random_agent_positions": True,
        "sample_recipe_on_delivery": True,
    }
    env = _make_env(args.layout, env_kwargs, args.jaxmarl_src)

    all_semantic: list[BranchRecord] = []
    all_controls: list[BranchRecord] = []
    all_variant_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    all_template_rows: list[dict[str, Any]] = []
    policy_matrix_rows: list[dict[str, Any]] = []

    for spec in policy_specs:
        for run_num in run_nums:
            pair = _load_policy_pair(spec, run_num, run_nums, args.pairing)
            if args.suppress_policy_stdout:
                with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
                    semantic, controls, variant_rows, trace_rows, template_rows = _build_records_for_pair(
                        env=env,
                        jax=jax,
                        pair=pair,
                        domain=args.domain,
                        probes=args.probes,
                        warmup_horizon=args.warmup_horizon,
                        rollout_horizon=args.rollout_horizon,
                        shaped_reward_weight=args.shaped_reward_weight,
                        seed=args.seed,
                    )
            else:
                semantic, controls, variant_rows, trace_rows, template_rows = _build_records_for_pair(
                    env=env,
                    jax=jax,
                    pair=pair,
                    domain=args.domain,
                    probes=args.probes,
                    warmup_horizon=args.warmup_horizon,
                    rollout_horizon=args.rollout_horizon,
                    shaped_reward_weight=args.shaped_reward_weight,
                    seed=args.seed,
                )
            all_semantic.extend(semantic)
            all_controls.extend(controls)
            all_variant_rows.extend(variant_rows)
            all_trace_rows.extend(trace_rows)
            all_template_rows.extend(template_rows)
            policy_matrix_rows.append(
                {
                    "policy": pair.label,
                    "method": spec.method,
                    "domain": args.domain,
                    "layout": args.layout,
                    "ego_run_num": pair.ego_run_num,
                    "partner_run_num": pair.partner_run_num,
                    "run_dir": str(spec.run_dir),
                    "pairing": args.pairing,
                    "status": "RESTORED_AND_SCORED",
                }
            )

    if not all_semantic or not all_controls:
        raise RuntimeError("non-noop sweep produced no semantic/control records")

    run_config = {
        "adapter": "experiments.serd.run_overcooked_v2_nonnoop_recovery_sweep",
        "tg_ssa_root": str(args.tg_ssa_root),
        "jaxmarl_src": str(args.jaxmarl_src),
        "layout": args.layout,
        "domain": args.domain,
        "policy_run": args.policy_run,
        "run_nums": run_nums,
        "pairing": args.pairing,
        "probes": args.probes,
        "warmup_horizon": args.warmup_horizon,
        "rollout_horizon": args.rollout_horizon,
        "shaped_reward_weight": args.shaped_reward_weight,
        "epsilon_shock": args.epsilon_shock,
        "epsilon_phi": args.epsilon_phi,
        "delta_serd": args.delta_serd,
        "delta_variant": args.delta_variant,
        "control_families": list(CONTROL_FAMILIES),
        "semantic_template": "all_single_step_nonnoop_action_replacements",
    }
    provenance = {
        "route": "overcookedv2_real_policy_nonnoop_recovery_sweep",
        "policy_source": "official OvercookedV2 RNN checkpoint directories",
        "not_policy_source": "handcoded M2 adapter",
        "run_command_or_queue_manifest": "python -m experiments.serd.run_overcooked_v2_nonnoop_recovery_sweep",
        "policy_matrix_rows": policy_matrix_rows,
        "claim_boundary": (
            "exploratory full-sweep real-policy non-noop recovery/discrimination "
            "evidence; interpret all rows, not cherry-picked cases"
        ),
    }
    write_m4_bundle(
        output_dir=args.output_dir,
        semantic_records=all_semantic,
        control_records=all_controls,
        provenance=provenance,
        run_config=run_config,
        acceptance_notes=[
            "Real OvercookedV2 PPO checkpoints were restored and used for policy calls.",
            "Semantic rows enumerate all one-step non-noop action replacements for both agents at sampled states.",
            "Matched controls replace the same actor with distinct one-step control actions and preserve pre-H identity.",
            "This artifact is a full sweep intended to find real non-noop recovery evidence; it is not a gate or forced-noop sensitivity control.",
        ],
    )

    family_variant, worst_variant = _summarize_variant(all_variant_rows, args.delta_variant)
    _write_csv(args.output_dir / "domain_policy_matrix.csv", policy_matrix_rows, list(policy_matrix_rows[0]))
    _write_csv(args.output_dir / "semantic_template_index.csv", all_template_rows, list(all_template_rows[0]))
    _write_csv(args.output_dir / "return_independent_variant.csv", all_variant_rows, list(all_variant_rows[0]))
    _write_csv(args.output_dir / "return_independent_family_serd.csv", family_variant, list(family_variant[0]))
    _write_csv(args.output_dir / "return_independent_worst_serd.csv", worst_variant, list(worst_variant[0]))
    _write_csv(args.output_dir / "branch_trace_summary.csv", all_trace_rows, list(all_trace_rows[0]))

    with (args.output_dir / "worst_serd.csv").open(newline="", encoding="utf-8") as handle:
        standard_worst = list(csv.DictReader(handle))
    survival_rows = _summarize_survival_rows(standard_worst, worst_variant)
    if survival_rows:
        _write_csv(args.output_dir / "nonnoop_survival_rows.csv", survival_rows, list(survival_rows[0]))
    else:
        _write_csv(
            args.output_dir / "nonnoop_survival_rows.csv",
            [],
            [
                "metric",
                "policy",
                "domain",
                "disruption",
                "mean_serd_worst",
                "ci95_low",
                "ci95_high",
                "n",
                "limiting_family",
            ],
        )

    summary_path = args.output_dir / "summary.json"
    enriched_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    enriched_summary.update(
        {
            "route_decision": "OVERCOOKEDV2_NONNOOP_RECOVERY_SWEEP_SCORED",
            "n_policy_pairs": len(policy_matrix_rows),
            "n_semantic_templates": len(
                {
                    (row["actor_index"], row["replacement_action"])
                    for row in all_template_rows
                }
            ),
            "n_semantic_template_rows": len(all_template_rows),
            "n_return_independent_rows": len(all_variant_rows),
            "n_return_independent_worst_rows": len(worst_variant),
            "n_survival_rows_total": len(survival_rows),
            "standard_any_survival_worst": any(
                row.get("classification") == "survival"
                for row in standard_worst
            ),
            "return_independent_any_survival_worst": any(
                row.get("classification") == "survival"
                for row in worst_variant
            ),
            "output_files": {
                "domain_policy_matrix": "domain_policy_matrix.csv",
                "semantic_template_index": "semantic_template_index.csv",
                "return_independent_variant": "return_independent_variant.csv",
                "return_independent_family": "return_independent_family_serd.csv",
                "return_independent_worst": "return_independent_worst_serd.csv",
                "branch_trace_summary": "branch_trace_summary.csv",
                "nonnoop_survival_rows": "nonnoop_survival_rows.csv",
                "operator_note": "operator_note.md",
            },
        }
    )
    summary_path.write_text(
        json.dumps(enriched_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _operator_note(args.output_dir, args, enriched_summary)
    print(json.dumps(enriched_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
