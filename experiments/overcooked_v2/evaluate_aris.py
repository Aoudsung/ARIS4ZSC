from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.factor_belief import FactorLocalBeliefModel
from src.aris_bellman.replay import EvidenceBuffer
from src.aris_bellman.specs import GraphSpec

from experiments.overcooked_v2.diagnostics import (
    belief_swap_top_pairs,
    diagnostic_cost,
    factor_deletion_return_drop,
    graph_with_deleted_factor,
    mutual_information_proxy,
    realized_delta_info,
    reference_gap_closure,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.evidence_router import D_EVID, OCV2EvidenceRouter
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer
from experiments.overcooked_v2.obs_encoder import infer_obs_dim
from experiments.overcooked_v2.option_termination import OptionRuntime, option_success
from experiments.overcooked_v2.option_executor import option_primitive_step
from experiments.overcooked_v2.option_inferencer import (
    PartnerOptionInferencer,
    make_behavior_option_inferencer,
)
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.provenance import (
    reward_config_payload,
    runtime_provenance,
    sha256_file,
    sha256_json,
)
from experiments.overcooked_v2.reward_design import ContributionLedger
from experiments.overcooked_v2.state_utils import (
    agent_facing_pos, get_agent_pos, get_inventory,
    get_pot_contents, is_pot_cooking, is_pot_ready_for_plate,
)
from jaxmarl.environments.overcooked_v2.common import Actions as _OCActions
from experiments.overcooked_v2.train_aris import (
    _advance_persistent_belief,
    _belief_persistence_enabled,
    _build_belief_model,
    _build_env,
    _build_option_lib,
    _build_q_network,
    _graph_objective_metadata_status,
    _graph_tensors,
    _initialise_persistent_belief,
    _obs_vector,
    _partner_id_tensor,
    _q_forward_kwargs,
    _select_train_partners,
    _state_repr,
    _tensor,
    _training_reward,
)


@dataclass
class EvalContext:
    checkpoint_path: Path
    config: dict[str, Any]
    graph: GraphSpec
    method: str
    graph_variant: str
    seed_name: str
    q_net: torch.nn.Module
    belief_model: FactorLocalBeliefModel
    layout_graph: LayoutGraph
    option_lib: OCV2OptionLibrary
    obs_dim: int
    # Diagnostic-only hooks (default off → normal eval/selection unchanged). RC-1: when
    # `qaudit` is a list, _select_option appends a read-only (q_total, q_base) decomposition
    # per greedy decision. RC-2: when `scripted_priority` is a kind-priority list, selection
    # follows that priority over valid options instead of argmax-Q (bypasses the network).
    qaudit: list | None = None
    scripted_priority: list[str] | None = None
    # RC-2b per-step option-execution trace (diagnostic; default off). When trace_steps is a
    # list, _execute_eval_option appends one record per primitive step for options whose kind
    # equals trace_kind (or all kinds if trace_kind is None).
    trace_steps: list | None = None
    trace_kind: str | None = None
    # RC-2b state-aware scripted oracle: callable(ctx, state, valid_ids) -> option_id, used by the
    # reachability probe to drive the full pipeline with stage/inventory awareness (bypasses Q).
    scripted_fsm: object | None = None


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(result["summary"]), indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    anchor = _resolve_checkpoint_path(Path(args.checkpoint))
    variants = _parse_csv(args.graph_variants)
    contexts = [
        _load_context(_sibling_checkpoint(anchor, variant), variant)
        for variant in variants
    ]
    partner_names = _resolve_partner_names(
        contexts[0].option_lib,
        args.partners,
        partner_set=str(contexts[0].config.get("training", {}).get("partner_set", "standard7")),
    )
    reward_scale_status = {
        ctx.graph_variant: _graph_objective_metadata_status(
            ctx.graph,
            ctx.config,
            layout_graph=ctx.layout_graph,
            option_lib=ctx.option_lib,
            partners=_select_train_partners(ctx.option_lib, ctx.config),
        )
        for ctx in contexts
    }
    seed = int(args.seed)
    max_episode_options = int(
        args.max_episode_options
        or contexts[0].config["training"].get("max_episode_options", 20)
    )

    fast = bool(getattr(args, "fast", False))
    allow_diag_skip = bool(getattr(args, "allow_diag_skip", False))
    random_policy_only = bool(getattr(args, "random_policy_only", False))
    factor_deletion_episodes = _factor_deletion_episode_count(args, fast)
    results = []
    for ctx in contexts:
        for partner_name in partner_names:
            aggregate, episodes = _evaluate_partner(
                ctx,
                partner_name,
                episodes=int(args.episodes),
                seed=seed,
                max_episode_options=max_episode_options,
                graph_override=ctx.graph,
                random_policy=random_policy_only,
                collect_diagnostics=not fast,
                allow_diag_skip=allow_diag_skip,
            )
            # P5: formal evaluation metadata must not inspect true partner
            # terminal_policy/role/protocol and derive role-match returns. Partner
            # names select the requested evaluation partner only; they are not routed
            # into decision evidence or return conditioning.
            aggregate["partner_protocol"] = "not_recorded_on_formal_main_path"
            q_proxy_factor_mask = [] if fast else _factor_deletion_q_proxy_diagnostics(ctx)
            result = {
                "method": "random_policy" if random_policy_only else ctx.method,
                "graph_variant": ctx.graph_variant,
                "partner": partner_name,
                "checkpoint": str(ctx.checkpoint_path),
                "reward_scale_verified": bool(
                    reward_scale_status[ctx.graph_variant]["reward_scale_verified"]
                ),
                "event_semantics_version": reward_scale_status[ctx.graph_variant][
                    "event_semantics_version"
                ],
                "aggregate": aggregate,
                "episodes": episodes,
                "q_proxy_factor_mask": q_proxy_factor_mask,
                "rollout_factor_mask": [],
                "factor_deletion_q_proxy": q_proxy_factor_mask,
            }
            if factor_deletion_episodes > 0 and not random_policy_only:
                result["rollout_factor_mask"] = _factor_deletion_rollout_diagnostics(
                    ctx,
                    partner_name,
                    float(aggregate["mean_return"]),
                    episodes=factor_deletion_episodes,
                    seed=seed + 400_000,
                    max_episode_options=max_episode_options,
                    allow_diag_skip=allow_diag_skip,
                )
                result["factor_deletion_return_drop"] = result["rollout_factor_mask"]
            results.append(result)

    baseline_cache_dir = _resolve_baseline_cache_dir(args)
    baselines = _random_baselines(
        contexts[0],
        partner_names,
        episodes=int(args.episodes),
        seed=seed + 200_000,
        max_episode_options=max_episode_options,
        cache_dir=baseline_cache_dir,
    )
    external_references = None
    if args.reference_base_checkpoint or args.reference_ref_checkpoint:
        if not args.reference_base_checkpoint or not args.reference_ref_checkpoint:
            raise ValueError(
                "reference_gap_closure requires both --reference_base_checkpoint and "
                "--reference_ref_checkpoint."
            )
        external_references = _external_reference_baselines(
            args,
            partner_names,
            episodes=int(args.episodes),
            seed=seed + 600_000,
            max_episode_options=max_episode_options,
            cache_dir=baseline_cache_dir,
        )
        _attach_external_reference_gaps(results, external_references)
    else:
        _attach_within_run_relative_returns(results, baselines)

    return {
        "schema_version": "ocv2_eval_v1",
        "anchor_checkpoint": str(anchor),
        "graph_variants": variants,
        "partners": partner_names,
        "episodes_per_partner": int(args.episodes),
        "max_episode_options": max_episode_options,
        "diagnostic_granularity": "option",
        "allow_diag_skip": allow_diag_skip,
        "factor_deletion_episodes": factor_deletion_episodes,
        "episode_return_kind": "reward_sum_minus_cost_coef_realized_cost",
        "reward_scale_verified": all(
            bool(item["reward_scale_verified"])
            for item in reward_scale_status.values()
        ),
        "graph_event_semantics_versions": {
            variant: status["event_semantics_version"]
            for variant, status in reward_scale_status.items()
        },
        "graph_reward_scale_status": reward_scale_status,
        "graph_provenance": {
            ctx.graph_variant: (ctx.graph.metadata or {}).get("provenance", {})
            for ctx in contexts
        },
        "eval_provenance": {
            ctx.graph_variant: _eval_provenance(ctx)
            for ctx in contexts
        },
        "reference_gap_semantics": _reference_semantics(external_references is not None),
        "results": results,
        "reference_baselines": baselines,
        "external_reference_baselines": external_references,
        "summary": _summary(results, baselines, time.time() - start),
    }


def _load_context(checkpoint_path: Path, variant: str) -> EvalContext:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for graph variant {variant}: {checkpoint_path}"
        )
    checkpoint = _torch_load(checkpoint_path)
    config = checkpoint["config"]
    graph = GraphSpec.from_json_dict(checkpoint["graph"])
    method = str(checkpoint["method"])

    env = _build_env(graph.layout_name, config)
    layout_graph = parse_layout(env, graph.layout_name)
    env.set_featurizer(NumpyFeaturizer(layout_graph))
    obs, _ = env.reset(0)
    option_lib = _build_option_lib(layout_graph, config)
    obs_dim = infer_obs_dim(env, obs)
    q_net = _build_q_network(method, obs_dim, graph, config).to(torch.device("cpu"))
    q_net.load_state_dict(checkpoint["q_net"])
    q_net.eval()
    belief_model = _build_belief_model(graph, config).to(torch.device("cpu"))
    belief_model.load_state_dict(checkpoint["belief_model"])
    belief_model.eval()
    return EvalContext(
        checkpoint_path=checkpoint_path,
        config=config,
        graph=graph,
        method=method,
        graph_variant=variant,
        seed_name=checkpoint_path.parent.name,
        q_net=q_net,
        belief_model=belief_model,
        layout_graph=layout_graph,
        option_lib=option_lib,
        obs_dim=obs_dim,
    )


def _evidence_policy_for_config(config: dict[str, Any]) -> str:
    """E2: derive the evidence-policy string the router stamps and the gate checks.

    `evidence.partner_option_inference.mode == "zeroed"` ⇒ the zeroed-channel ablation
    policy (METHOD_LOCK sec18.8); anything else ⇒ the formal behavior-inferred policy.
    Kept in one place so the router-stamped string and the gate's admitted set cannot
    drift apart.
    """
    mode = str(
        ((config or {}).get("evidence", {}) or {})
        .get("partner_option_inference", {})
        .get("mode", "inferred")
    )
    return (
        "behavior_inferred_v1_zeroed_ablation"
        if mode == "zeroed"
        else "behavior_inferred_v1"
    )


def _evaluate_partner(
    ctx: EvalContext,
    partner_name: str,
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    graph_override: GraphSpec,
    random_policy: bool,
    collect_diagnostics: bool,
    allow_diag_skip: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    env = _build_env(graph_override.layout_name, ctx.config)
    env.set_featurizer(NumpyFeaturizer(ctx.layout_graph))
    router = OCV2EvidenceRouter(
        graph_override,
        ctx.layout_graph.cell_to_entity,
        ctx.layout_graph.region_cells,
        evidence_policy=_evidence_policy_for_config(ctx.config),
    )
    partners = {partner.name: partner for partner in make_training_partners(
        ctx.option_lib,
        partner_set=str(ctx.config.get("training", {}).get("partner_set", "standard7")),
    )}
    if partner_name not in partners:
        raise KeyError(f"Unknown partner {partner_name!r}; choices={sorted(partners)}")
    partner = partners[partner_name]
    episode_rows: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        row = _run_episode(
            ctx,
            env,
            partner,
            router,
            graph_override,
            rng,
            seed + episode_idx,
            max_episode_options,
            random_policy=random_policy,
            collect_diagnostics=collect_diagnostics,
            allow_diag_skip=allow_diag_skip,
        )
        row["episode_id"] = int(episode_idx)
        episode_rows.append(row)

    aggregate = _aggregate_episodes(episode_rows)
    aggregate["partner_option_evidence"] = router.partner_option_evidence_summary()
    _validate_eval_integrity(
        aggregate,
        collect_diagnostics=collect_diagnostics,
        allow_diag_skip=allow_diag_skip,
    )
    return aggregate, episode_rows


def _run_episode(
    ctx: EvalContext,
    env: OCV2Adapter,
    partner: Any,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    rng: np.random.Generator,
    seed: int,
    max_episode_options: int,
    *,
    random_policy: bool,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
) -> dict[str, Any]:
    evidence_buffer = EvidenceBuffer(
        num_factors=graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    evidence_buffer.reset()
    router.reset()
    obs, state0 = env.reset(seed)
    partner.reset(seed)
    partner_option_inferencer = make_behavior_option_inferencer(ctx.option_lib, ctx.config)
    partner_option_inferencer.reset(state0)
    _initialise_persistent_belief(
        evidence_buffer, ctx.method, ctx.belief_model, graph, torch.device("cpu"),
        _belief_persistence_enabled(ctx.config),
    )
    contribution_ledger = ContributionLedger.from_config(
        ctx.config.get("training", {})
    )

    episode_return = 0.0
    primitive_steps = 0
    option_count = 0
    blocking_events = 0
    delivery_seen = False
    ego_completion_seen = False
    partner_delivery_seen = False
    wrong_delivery_seen = False
    delta_values: list[float] = []
    mi_values: list[float] = []
    diagnostic_cost_values: list[float] = []
    swap_values: list[dict[str, Any]] = []
    belief_influence_values: list[dict[str, float]] = []
    termination_counts: dict[str, int] = {}
    option_kind_stats: dict[str, dict[str, Any]] = {}
    diagnostic_status_counts: dict[str, int] = {}
    selection_stats: dict[str, int] = {
        "option_selection_count": 0,
        "forced_noop_count": 0,
        "no_valid_option_count": 0,
    }
    delivery_counts = _empty_delivery_counts()
    done = False

    while not done and option_count < max_episode_options:
        option_id = _select_option(
            ctx,
            obs,
            env.state,
            evidence_buffer,
            graph,
            rng,
            random_policy,
            partner_id=int(getattr(partner, "partner_id", 0)),
            selection_stats=selection_stats,
        )
        option_return, done, obs, info = _execute_eval_option(
            ctx,
            env,
            obs,
            partner,
            router,
            graph,
            evidence_buffer,
            option_id,
            collect_diagnostics,
            allow_diag_skip,
            rng,
            contribution_ledger=contribution_ledger,
            partner_option_inferencer=partner_option_inferencer,
        )
        episode_return += option_return
        option_count += 1
        primitive_steps += int(info["primitive_steps"])
        blocking_events += int(info["blocking_events"])
        delivery_seen = bool(delivery_seen or info["delivery_seen"])
        _merge_counts(delivery_counts, info["delivery_counts"])
        ego_completion_seen = bool(
            ego_completion_seen or int(info["delivery_counts"].get("ego_sole_correct_delivery", 0)) > 0
        )
        partner_delivery_seen = bool(
            partner_delivery_seen or int(info["delivery_counts"].get("partner_delivery_event", 0)) > 0
        )
        wrong_delivery_seen = bool(
            wrong_delivery_seen or int(info["delivery_counts"].get("wrong_delivery_event", 0)) > 0
        )
        _merge_counts(diagnostic_status_counts, info["diagnostic_status_counts"])
        delta_values.extend(info["delta_info"])
        mi_values.extend(info["mi"])
        diagnostic_cost_values.extend(info["diagnostic_cost"])
        swap_values.extend(info["belief_swap"])
        belief_influence_values.extend(info["belief_influence"])
        _increment(termination_counts, str(info["termination_reason"]))
        _update_option_kind_stats(
            option_kind_stats,
            graph.options[int(option_id)].kind,
            str(info["termination_reason"]),
        )

    return {
        "return": float(episode_return),
        # S20: headline completion is ego-owned success; team delivery is separate.
        "completed": bool(ego_completion_seen),
        "team_completed": bool(delivery_seen),
        "partner_delivery_episode": bool(partner_delivery_seen),
        "wrong_delivery_episode": bool(wrong_delivery_seen),
        "ego_correct_completed": bool(ego_completion_seen),
        "partner_correct_completed": bool(delivery_counts.get("partner_correct_delivery", 0) > 0),
        "wrong_delivery_completed": bool(wrong_delivery_seen),
        "delivery_counts": delivery_counts,
        "primitive_steps": int(primitive_steps),
        "option_count": int(option_count),
        "blocking_events": int(blocking_events),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "delta_info_mean": _mean_or_nan(delta_values),
        "mi_mean": _mean_or_nan(mi_values),
        "diagnostic_cost_mean": _mean_or_nan(diagnostic_cost_values),
        "diagnostic_count": len(delta_values),
        "diagnostic_status_counts": diagnostic_status_counts,
        **selection_stats,
        "forced_noop_fraction": float(
            selection_stats["forced_noop_count"]
            / max(1, selection_stats["option_selection_count"])
        ),
        "no_valid_option_fraction": float(
            selection_stats["no_valid_option_count"]
            / max(1, selection_stats["option_selection_count"])
        ),
        "belief_swap": _aggregate_swap(swap_values),
        "belief_influence": _aggregate_belief_influence(belief_influence_values),
        "belief_influence_count": len(belief_influence_values),
        "termination_counts": termination_counts,
        "option_kind_stats": option_kind_stats,
    }


def _execute_eval_option(
    ctx: EvalContext,
    env: OCV2Adapter,
    obs: dict[str, np.ndarray],
    partner: Any,
    router: OCV2EvidenceRouter,
    graph: GraphSpec,
    evidence_buffer: EvidenceBuffer,
    option_id: int,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
    rng: np.random.Generator,
    *,
    contribution_ledger: ContributionLedger | None = None,
    partner_option_inferencer: PartnerOptionInferencer | None = None,
) -> tuple[float, bool, dict[str, np.ndarray], dict[str, Any]]:
    opt = ctx.option_lib.options[int(option_id)]
    runtime = OptionRuntime(
        option_id=int(option_id),
        start_pos=get_agent_pos(env.state, 0),
    )
    reward_sum = 0.0
    realized_cost = 0.0
    duration = 0
    blocking_events = 0
    delivery_seen = False
    termination_reason = "running"
    delta_values: list[float] = []
    mi_values: list[float] = []
    diagnostic_cost_values: list[float] = []
    swap_values: list[dict[str, Any]] = []
    belief_influence_values: list[dict[str, float]] = []
    diagnostic_status_counts: dict[str, int] = {}
    delivery_counts = _empty_delivery_counts()
    done = False
    belief_before_option = _current_belief(ctx, evidence_buffer, graph)

    # F2 (RC-2b): executor patience. When the agent makes no progress toward the option's target
    # interaction cell for `block_patience` steps (livelock / partner camping the only stand cell),
    # terminate the option as blocked so the policy can re-decide, instead of thrashing/colliding
    # for the whole budget. Default 0 = off (current behavior).
    _patience = int((ctx.config.get("options") or {}).get("block_patience", 0))
    _spd = ctx.option_lib.layout_graph.shortest_path_dist
    _opt_targets = tuple(ctx.option_lib._target_cells(opt))

    def _dist_to_target(_st: Any) -> int | None:
        if not _opt_targets:
            return None
        _a = get_agent_pos(_st, 0)
        _ds = [_spd.get((_a, _t)) for _t in _opt_targets]
        _ds = [d for d in _ds if d is not None]
        return min(_ds) if _ds else None

    _best_dist = _dist_to_target(env.state)
    _stuck = 0
    _pot_positions = [e.pos for e in ctx.layout_graph.entities.values() if e.kind == "pot"]

    _budget = ctx.option_lib.option_budget(env.state, 0, int(option_id))
    while duration < _budget:
        _ostep = option_primitive_step(
            env,
            ctx.option_lib,
            int(option_id),
            partner,
            obs,
            rng,
            partner_option_inferencer=partner_option_inferencer,
        )
        ego_action = _ostep.ego_action
        partner_action = _ostep.partner_action
        prev_state = _ostep.prev_state
        step = _ostep.step
        event = _ostep.event
        if contribution_ledger is not None:
            contribution_ledger.update(event, ego_option_kind=str(opt.kind))
        ego_contributed = False
        if contribution_ledger is not None:
            ego_contributed = contribution_ledger.query_and_reset_on_delivery(event)
        reward_sum += _training_reward(
            step,
            ctx.config,
            "agent_0",
            event,
            ego_contributed=ego_contributed,
        )
        realized_cost += float(ctx.config["training"].get("cost_per_step", 1.0))
        duration += 1
        x_f = router.route(
            event,
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_f)
        _advance_persistent_belief(
            evidence_buffer,
            ctx.method,
            ctx.belief_model,
            graph,
            torch.device("cpu"),
            x_f,
            _belief_persistence_enabled(ctx.config),
        )
        blocking_events += int(bool(event.collision_or_block))
        if getattr(ctx, "trace_steps", None) is not None and (
            ctx.trace_kind is None or opt.kind == ctx.trace_kind
        ):
            ctx.trace_steps.append({
                "kind": opt.kind,
                "step": int(duration),
                "agent": list(get_agent_pos(prev_state, 0)),
                "partner": list(get_agent_pos(prev_state, 1)),
                "target_pos": list(opt.target_pos) if opt.target_pos is not None else None,
                "facing": list(agent_facing_pos(prev_state, 0)),
                "action": int(ego_action),
                "is_interact": bool(int(ego_action) == int(_OCActions.interact)),
                "inv_before": int(get_inventory(prev_state, 0)),
                "inv_after": int(get_inventory(step.state, 0)),
                "blocked": bool(event.collision_or_block),
                "pots": [
                    [list(p), int(get_pot_contents(step.state, p)),
                     bool(is_pot_cooking(step.state, p)),
                     bool(is_pot_ready_for_plate(step.state, p, require_correct_recipe=False))]
                    for p in _pot_positions
                ],
            })
        _accumulate_delivery_counts(delivery_counts, event)
        delivery_seen = bool(delivery_seen or event.delivery_event)
        done = bool(step.dones.get("__all__", False))
        if _patience and _best_dist is not None and not done:
            _cur = _dist_to_target(step.state)
            if _cur is not None and _cur < _best_dist:
                _best_dist = _cur
                _stuck = 0
            elif _cur is not None and _cur > 0:
                _stuck += 1
            if _stuck >= _patience:
                obs = step.obs
                termination_reason = "blocked_no_progress"
                break
        terminated, termination_reason = ctx.option_lib.option_terminated(
            opt,
            prev_state,
            step.state,
            event,
            agent_id=0,
            elapsed=duration,
            runtime=runtime,
        )
        if done and not terminated:
            termination_reason = "env_max_steps"
        obs = step.obs
        if done or terminated:
            break

    # Diagnostic label fix (mirrors train_aris._execute_option): relabel a budget
    # exhaustion so option stats don't record "running" as a terminal reason.
    if not done and termination_reason == "running":
        termination_reason = "budget_exhausted"

    # Push option-level FAILURE signal to evidence when the option did not succeed
    # (mirrors train_aris). Belief must see failure events at inference too.
    _failed = termination_reason in {"budget_exhausted", "max_steps", "env_max_steps", "blocked_no_progress"}
    if _failed and duration > 0:
        x_fail = router.route_failure_boundary(
            ego_option_id=int(option_id),
            ego_option_elapsed=duration,
            ego_option_max_steps=opt.max_steps,
        )
        evidence_buffer.append(x_fail)
        _advance_persistent_belief(
            evidence_buffer,
            ctx.method,
            ctx.belief_model,
            graph,
            torch.device("cpu"),
            x_fail,
            _belief_persistence_enabled(ctx.config),
        )

    if collect_diagnostics:
        belief_after_option = _current_belief(ctx, evidence_buffer, graph)
        diag = _option_diagnostics(
            ctx,
            obs,
            int(option_id),
            belief_before_option,
            belief_after_option,
            graph,
            allow_diag_skip=allow_diag_skip,
        )
        _increment(diagnostic_status_counts, str(diag.get("status", "unknown")))
        if diag.get("status") == "ok":
            delta_values.append(diag["delta_info"])
            mi_values.append(diag["mi"])
            diagnostic_cost_values.append(diag["diagnostic_cost"])
            swap_values.append(diag["belief_swap"])
            belief_influence_values.append(diag["belief_influence"])

    option_return = reward_sum - float(ctx.config["training"]["cost_coef"]) * realized_cost
    return (
        float(option_return),
        bool(done),
        obs,
        {
            "primitive_steps": int(duration),
            "blocking_events": int(blocking_events),
            "delivery_seen": bool(delivery_seen),
            "delivery_counts": delivery_counts,
            "termination_reason": termination_reason,
            "delta_info": delta_values,
            "mi": mi_values,
            "diagnostic_cost": diagnostic_cost_values,
            "belief_swap": swap_values,
            "belief_influence": belief_influence_values,
            "diagnostic_status_counts": diagnostic_status_counts,
        },
    )


_DIAG_SKIP = {
    "status": "shape_mismatch",
    "delta_info": float("nan"),
    "mi": float("nan"),
    "diagnostic_cost": float("nan"),
    "belief_swap": {"status": "shape_mismatch"},
    "belief_influence": {
        "belief_zero_delta": 0.0,
        "belief_uniform_delta": 0.0,
        "relevance_zero_delta": 0.0,
    },
}

_DIAG_UNSUPPORTED = {
    "status": "unsupported_method",
    "delta_info": float("nan"),
    "mi": float("nan"),
    "diagnostic_cost": float("nan"),
    "belief_swap": {"status": "unsupported_method"},
    "belief_influence": {
        "belief_zero_delta": 0.0,
        "belief_uniform_delta": 0.0,
        "relevance_zero_delta": 0.0,
    },
}


def _option_diagnostics(
    ctx: EvalContext,
    obs_next: dict[str, np.ndarray],
    option_id: int,
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    graph: GraphSpec,
    *,
    allow_diag_skip: bool = False,
) -> dict[str, Any]:
    method = getattr(ctx, "method", "aris_bellman")
    if method not in {"aris_bellman", "flat_factor"}:
        return dict(_DIAG_UNSUPPORTED)
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    mode_mask = graph_batch["mode_mask"]
    if belief_before.shape != mode_mask.shape:
        if not allow_diag_skip:
            raise RuntimeError(
                "Diagnostic belief/mode-mask shape mismatch: "
                f"belief={tuple(belief_before.shape)} mode_mask={tuple(mode_mask.shape)}. "
                "Pass --allow_diag_skip only for non-formal smoke runs."
            )
        return dict(_DIAG_SKIP)
    obs_tensor = _tensor(_obs_vector(obs_next, "agent_0")[None, ...], torch.device("cpu"))
    with torch.no_grad():
        delta = realized_delta_info(
            ctx.q_net,
            obs_tensor,
            belief_before,
            belief_after,
            graph_batch,
            gamma=float(ctx.config["training"]["gamma"]),
        )
        mi = mutual_information_proxy(belief_before, belief_after, mode_mask)
        q_base = _base_q_values(ctx, obs_tensor, graph_batch, belief_after)
        _, cost = diagnostic_cost(q_base, int(option_id), delta, tau=0.0)
        swap = belief_swap_top_pairs(ctx.q_net, obs_tensor, belief_after, graph_batch, graph)
        q_actual = ctx.q_net(
            obs_tensor,
            belief_after,
            **_q_forward_kwargs(graph_batch),
        ).squeeze(0)
        belief_influence = _belief_influence_decomposition(
            ctx,
            obs_tensor,
            belief_after,
            _q_forward_kwargs(graph_batch),
            q_actual,
        )
        return {
            "status": "ok",
            "delta_info": float(delta.mean().item()),
            "mi": float(mi.mean().item()),
            "diagnostic_cost": float(cost.mean().item()),
            "belief_swap": swap,
            "belief_influence": belief_influence,
        }


def _belief_influence_decomposition(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    belief_actual: torch.Tensor,
    graph_kwargs: dict[str, Any],
    q_actual: torch.Tensor,
) -> dict[str, float]:
    if not hasattr(ctx.q_net, "forward_with_belief_override"):
        return {
            "belief_zero_delta": 0.0,
            "belief_uniform_delta": 0.0,
            "relevance_zero_delta": 0.0,
        }
    with torch.no_grad():
        belief_zero = torch.zeros_like(belief_actual)
        q_zero = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_zero,
            graph_kwargs=graph_kwargs,
        ).squeeze(0)

        mode_mask = graph_kwargs.get("mode_mask")
        if mode_mask is not None:
            mm = mode_mask.to(dtype=belief_actual.dtype)
            n_valid = mm.sum(dim=-1, keepdim=True).clamp(min=1.0)
            belief_unif = mm / n_valid
        else:
            belief_unif = torch.ones_like(belief_actual) / max(1, belief_actual.shape[-1])
        q_unif = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_unif,
            graph_kwargs=graph_kwargs,
        ).squeeze(0)

        graph_kwargs_no_rel = dict(graph_kwargs)
        rm = graph_kwargs_no_rel.get("relevance_mask")
        if rm is not None:
            graph_kwargs_no_rel["relevance_mask"] = torch.zeros_like(rm)
        q_no_rel = ctx.q_net.forward_with_belief_override(
            obs_tensor,
            belief_actual,
            graph_kwargs=graph_kwargs_no_rel,
        ).squeeze(0)
    return {
        "belief_zero_delta": float((q_actual - q_zero).abs().max().item()),
        "belief_uniform_delta": float((q_actual - q_unif).abs().max().item()),
        "relevance_zero_delta": float((q_actual - q_no_rel).abs().max().item()),
    }


def _current_belief(
    ctx: EvalContext,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
) -> torch.Tensor:
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    evidence = _tensor(evidence_buffer.snapshot()[None, ...], torch.device("cpu"))
    evidence_mask = torch.as_tensor(
        evidence_buffer.snapshot_mask()[None, ...], dtype=torch.bool, device=torch.device("cpu")
    )
    evidence_lengths = torch.as_tensor([evidence_buffer.length()], dtype=torch.float32)
    hidden_np = evidence_buffer.belief_window_base_snapshot()
    belief_hidden = _tensor(hidden_np[None, ...], torch.device("cpu")) if hidden_np is not None else None
    with torch.no_grad():
        return _state_repr(
            ctx.method,
            ctx.belief_model,
            evidence,
            graph_batch,
            evidence_lengths=evidence_lengths,
            evidence_mask=evidence_mask,
            belief_hidden=belief_hidden,
        )


def _select_option(
    ctx: EvalContext,
    obs: dict[str, np.ndarray],
    state: Any,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
    rng: np.random.Generator,
    random_policy: bool,
    partner_id: int | None = None,
    selection_stats: dict[str, int] | None = None,
) -> int:
    _record_selection_attempt(selection_stats)
    valid = ctx.option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        _record_forced_noop(selection_stats)
        return _noop_option_id(ctx.option_lib)
    if random_policy:
        return int(rng.choice(valid_ids))

    if getattr(ctx, "scripted_fsm", None) is not None:
        chosen = ctx.scripted_fsm(ctx, state, valid_ids)
        if chosen is not None:
            return int(chosen)

    if getattr(ctx, "scripted_priority", None) is not None:
        scripted = _scripted_priority_select(ctx, graph, valid_ids)
        if scripted is not None:
            return scripted

    with torch.no_grad():
        graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
        obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, ...], torch.device("cpu"))
        belief = _current_belief(ctx, evidence_buffer, graph)
        q_values = ctx.q_net(
            obs_tensor,
            belief,
            **_q_forward_kwargs(graph_batch),
            partner_id=_partner_id_tensor(partner_id, 1, torch.device("cpu")),
        ).squeeze(0)
        if getattr(ctx, "qaudit", None) is not None:
            _record_qaudit(ctx, obs_tensor, graph_batch, q_values, valid)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
        q_values = q_values.masked_fill(~valid_tensor, -1e9)
        return int(torch.argmax(q_values).item())


def _scripted_priority_select(
    ctx: EvalContext,
    graph: GraphSpec,
    valid_ids: np.ndarray,
) -> int | None:
    """RC-2 diagnostic: pick the valid option of the highest-priority kind, bypassing Q.

    Used to confirm whether the full fetch->cook->plate->serve pipeline is reachable and
    whether serve_soup success / ego delivery / completion ever fire under deliberate play.
    Returns None if no valid option matches any listed kind (caller falls back to Q).
    """
    kinds = {int(i): str(graph.options[int(i)].kind) for i in valid_ids}
    for target in ctx.scripted_priority:
        for vid in valid_ids:
            if kinds[int(vid)] == target:
                return int(vid)
    return None


def _record_qaudit(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    graph_batch: dict[str, Any],
    q_values: torch.Tensor,
    valid: np.ndarray,
) -> None:
    """RC-1 diagnostic: log the (q_total, q_base) decomposition for the current decision.

    Read-only: it does NOT change the returned argmax. adv_sum is recovered downstream as
    q_total - q_base (the network defines q_values = q_base + sum_f A_f). Skips silently if
    the wrapped net does not expose a factor-local base head.
    """
    if not (hasattr(ctx.q_net, "q_net") and hasattr(ctx.q_net.q_net, "q_base_values")):
        return
    encoded = ctx.q_net.encoder(obs_tensor)
    q_base = ctx.q_net.q_net.q_base_values(encoded, graph_batch["option_mask"]).squeeze(0)
    ctx.qaudit.append(
        {
            "q_full": [float(x) for x in q_values.detach().tolist()],
            "q_base": [float(x) for x in q_base.detach().tolist()],
            "valid": [bool(x) for x in valid.tolist()],
        }
    )


def _factor_deletion_q_proxy_diagnostics(ctx: EvalContext) -> list[dict[str, Any]]:
    if ctx.graph.num_factors == 0:
        return []

    q_drops = _factor_deletion_q_proxy(ctx)
    rows = []
    for factor in ctx.graph.factors:
        rows.append(
            {
                "factor_id": int(factor.id),
                "factor_kind": factor.factor_kind,
                "option_i": int(factor.option_i),
                "option_j": int(factor.option_j),
                "q_proxy_drop": float(q_drops[int(factor.id)]),
                "ablation_mode": "q_proxy_factor_mask",
                "ablation_scope": "single_initial_state_empty_evidence",
                "rollout_episodes": 0,
            }
        )
    rows.sort(key=lambda row: (-row["q_proxy_drop"], row["factor_id"]))
    return rows


def _factor_deletion_rollout_diagnostics(
    ctx: EvalContext,
    partner_name: str,
    base_return: float,
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    allow_diag_skip: bool = False,
) -> list[dict[str, Any]]:
    deleted_returns: dict[int, float] = {}
    for factor in ctx.graph.factors:
        deleted_graph = graph_with_deleted_factor(ctx.graph, int(factor.id))
        aggregate, _ = _evaluate_partner(
            ctx,
            partner_name,
            episodes=episodes,
            seed=seed + int(factor.id),
            max_episode_options=max_episode_options,
            graph_override=deleted_graph,
            random_policy=False,
            collect_diagnostics=False,
            allow_diag_skip=allow_diag_skip,
        )
        deleted_returns[int(factor.id)] = float(aggregate["mean_return"])

    rows = factor_deletion_return_drop(
        ctx.graph,
        base_return,
        lambda factor_id: deleted_returns[int(factor_id)],
    )
    for row in rows:
        row["ablation_mode"] = "rollout_factor_mask"
        row["ablation_scope"] = "factor_mask_relevance_mode_disabled"
        row["rollout_episodes"] = int(episodes)
    return rows


def _factor_deletion_q_proxy(ctx: EvalContext) -> dict[int, float]:
    env = _build_env(ctx.graph.layout_name, ctx.config)
    env.set_featurizer(NumpyFeaturizer(ctx.layout_graph))
    obs, _ = env.reset(0)
    evidence_buffer = EvidenceBuffer(
        num_factors=ctx.graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, ...], torch.device("cpu"))
    base_graph_batch = _graph_tensors(ctx.graph, 1, torch.device("cpu"))
    base_belief = _current_belief(ctx, evidence_buffer, ctx.graph)
    with torch.no_grad():
        base_value = ctx.q_net(
            obs_tensor,
            base_belief,
            **_q_forward_kwargs(base_graph_batch),
        ).max(dim=-1).values.item()

    drops = {}
    for factor in ctx.graph.factors:
        deleted_graph = graph_with_deleted_factor(ctx.graph, int(factor.id))
        deleted_batch = _graph_tensors(deleted_graph, 1, torch.device("cpu"))
        deleted_belief = _current_belief(ctx, evidence_buffer, deleted_graph)
        with torch.no_grad():
            deleted_value = ctx.q_net(
                obs_tensor,
                deleted_belief,
                **_q_forward_kwargs(deleted_batch),
            ).max(dim=-1).values.item()
        drops[int(factor.id)] = float(base_value - deleted_value)
    return drops


_ENV_CACHE_KEYS = (
    "max_steps",
    "agent_view_size",
    "negative_rewards",
    "sample_recipe_on_delivery",
    "random_reset",
    "random_agent_positions",
    "force_path_planning",
)


# Option-runtime config that changes valid options / budgets / termination and
# therefore rollout returns (review BLOCK fix — was missing from the key).
_OPTIONS_CACHE_KEYS = (
    "max_option_steps",
    "strict_preconditions",
    "dynamic_budget",
    "block_patience",
)
# Bump when the key scheme or entry shape changes so stale valid-JSON entries from
# an older code version are rejected rather than trusted (review MEDIUM fix).
_BASELINE_CACHE_SCHEMA = 2


def _baseline_env_payload(config: dict[str, Any]) -> dict[str, Any]:
    """The subset of config that changes a reference/random rollout's outcome."""
    env_cfg = (config or {}).get("env", {}) or {}
    opt_cfg = (config or {}).get("options", {}) or {}
    return {
        "env": {k: env_cfg.get(k) for k in _ENV_CACHE_KEYS},
        "options": {k: opt_cfg.get(k) for k in _OPTIONS_CACHE_KEYS},
        "reward_config": reward_config_payload(config or {}),
        "partner_set": str(
            ((config or {}).get("training", {}) or {}).get("partner_set", "standard7")
        ),
    }


def _baseline_cache_target(
    cache_dir: Path | None,
    *,
    kind: str,
    layout: str,
    config: dict[str, Any],
    partner: str,
    episodes: int,
    seed: int,
    max_episode_options: int,
    extra: Any = None,
) -> tuple[Path | None, str | None]:
    """(path, key) for a checkpoint-independent baseline rollout.

    Key covers everything that determines the result: kind, layout, the env/options/
    reward config subset, partner_set, partner, episodes, max_episode_options, seed.
    `extra` carries reference-checkpoint hashes for the external variant (which IS
    checkpoint-dependent). Returns (None, None) when caching is disabled.
    """
    if cache_dir is None:
        return None, None
    key = sha256_json(
        {
            "schema": _BASELINE_CACHE_SCHEMA,
            "kind": kind,
            "layout": layout,
            "config": _baseline_env_payload(config),
            "partner": partner,
            "episodes": int(episodes),
            "max_episode_options": int(max_episode_options),
            "seed": int(seed),
            "extra": extra,
        }
    )
    return cache_dir / f"{kind}_{key}.json", key


def _resolve_baseline_cache_dir(args: argparse.Namespace) -> Path | None:
    """Cache dir for baseline rollouts: explicit --baseline_cache_dir, 'none' to
    disable, or default <output_dir>/.baseline_cache."""
    raw = getattr(args, "baseline_cache_dir", None)
    if raw is not None and str(raw).lower() == "none":
        return None
    cache_dir = Path(raw) if raw else (Path(args.output).parent / ".baseline_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _read_baseline_cache(path: Path | None, expected_key: str | None) -> dict[str, Any] | None:
    """Return the cached value only if the file parses, matches the current schema,
    and its embedded key equals the expected key (defends against corrupt/partial
    files and stale entries from an older key scheme)."""
    if path is None or expected_key is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # corrupted/partial → recompute
    if not isinstance(raw, dict):
        return None
    if raw.get("_cache_schema") != _BASELINE_CACHE_SCHEMA:
        return None
    if raw.get("_cache_key") != expected_key:
        return None
    value = raw.get("value")
    return value if isinstance(value, dict) else None


def _write_baseline_cache(
    path: Path | None, expected_key: str | None, value: dict[str, Any]
) -> None:
    if path is None or expected_key is None:
        return
    payload = {
        "_cache_schema": _BASELINE_CACHE_SCHEMA,
        "_cache_key": expected_key,
        "value": value,
    }
    # Atomic on POSIX: write a pid-tagged temp then rename. Parallel eval runs as
    # independent subprocesses; the key now covers every result-affecting input, so a
    # same-path collision means identical inputs and last-writer-wins is harmless.
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)


def _random_baselines(
    ctx: EvalContext,
    partner_names: list[str],
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    baselines = {}
    for idx, partner_name in enumerate(partner_names):
        cache_path, cache_key = _baseline_cache_target(
            cache_dir,
            kind="random_policy",
            layout=ctx.graph.layout_name,
            config=ctx.config,
            partner=partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
        )
        cached = _read_baseline_cache(cache_path, cache_key)
        if cached is not None:
            baselines[partner_name] = cached
            continue
        aggregate, _ = _evaluate_partner(
            ctx,
            partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            graph_override=ctx.graph,
            random_policy=True,
            collect_diagnostics=False,
        )
        entry = {
            "base_kind": "random_policy",
            "mean_return": aggregate["mean_return"],
            "completion_rate": aggregate["completion_rate"],
            "blocking_rate": aggregate["blocking_rate"],
        }
        _write_baseline_cache(cache_path, cache_key, entry)
        baselines[partner_name] = entry
    return baselines


def _external_reference_baselines(
    args: argparse.Namespace,
    partner_names: list[str],
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    base_ctx = _load_context(
        _resolve_checkpoint_path(Path(args.reference_base_checkpoint)),
        "reference_base",
    )
    ref_ctx = _load_context(
        _resolve_checkpoint_path(Path(args.reference_ref_checkpoint)),
        "reference_ref",
    )
    # These baselines DO depend on the two reference checkpoints, so the cache key
    # includes their content hashes (guards against a checkpoint being swapped).
    ref_extra = {
        "base_ckpt_sha256": sha256_file(base_ctx.checkpoint_path),
        "ref_ckpt_sha256": sha256_file(ref_ctx.checkpoint_path),
    }
    baselines: dict[str, Any] = {}
    for idx, partner_name in enumerate(partner_names):
        cache_path, cache_key = _baseline_cache_target(
            cache_dir,
            kind="external_reference",
            layout=base_ctx.graph.layout_name,
            config=base_ctx.config,
            partner=partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            extra=ref_extra,
        )
        cached = _read_baseline_cache(cache_path, cache_key)
        if cached is not None:
            baselines[partner_name] = cached
            continue
        base_aggregate, _ = _evaluate_partner(
            base_ctx,
            partner_name,
            episodes=episodes,
            seed=seed + idx,
            max_episode_options=max_episode_options,
            graph_override=base_ctx.graph,
            random_policy=False,
            collect_diagnostics=False,
        )
        ref_aggregate, _ = _evaluate_partner(
            ref_ctx,
            partner_name,
            episodes=episodes,
            seed=seed + 10_000 + idx,
            max_episode_options=max_episode_options,
            graph_override=ref_ctx.graph,
            random_policy=False,
            collect_diagnostics=False,
        )
        entry = {
            "base_checkpoint": str(base_ctx.checkpoint_path),
            "ref_checkpoint": str(ref_ctx.checkpoint_path),
            "base_method": base_ctx.method,
            "ref_method": ref_ctx.method,
            "base_mean_return": base_aggregate["mean_return"],
            "ref_mean_return": ref_aggregate["mean_return"],
            "episodes": int(episodes),
        }
        _write_baseline_cache(cache_path, cache_key, entry)
        baselines[partner_name] = entry
    return baselines


def _attach_external_reference_gaps(
    results: list[dict[str, Any]],
    references: dict[str, Any],
) -> None:
    for row in results:
        reference = references[row["partner"]]
        row["aggregate"]["reference_gap_closure"] = reference_gap_closure(
            float(row["aggregate"]["mean_return"]),
            float(reference["base_mean_return"]),
            float(reference["ref_mean_return"]),
        )


def _attach_within_run_relative_returns(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
) -> None:
    by_partner: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_partner.setdefault(result["partner"], []).append(result)
    for partner, rows in by_partner.items():
        r_base = float(baselines[partner]["mean_return"])
        r_ref = max(float(row["aggregate"]["mean_return"]) for row in rows)
        for row in rows:
            metric = reference_gap_closure(
                float(row["aggregate"]["mean_return"]),
                r_base,
                r_ref,
            )
            if metric.get("status") == "ok" and float(row["aggregate"]["mean_return"]) == r_ref:
                metric.pop("raw_value", None)
                metric["status"] = "within_run_reference_variant"
            row["aggregate"]["within_run_relative_return"] = metric


def _reference_semantics(has_external_references: bool) -> dict[str, str]:
    if has_external_references:
        return {
            "r_base": "external_reference_base_checkpoint",
            "r_ref": "external_reference_ref_checkpoint",
            "reference_type": "external_checkpoint_reference_gap_closure",
        }
    return {
        "r_base": "random_policy_rollout",
        "r_ref": "best_mean_return_among_requested_graph_variants_per_partner",
        "reference_type": "within_run_relative_return_not_reference_gap_closure",
    }


def _aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["return"]) for row in episodes]
    primitive_steps = sum(int(row["primitive_steps"]) for row in episodes)
    blocking_events = sum(int(row["blocking_events"]) for row in episodes)
    term_counts: dict[str, int] = {}
    option_kind_stats: dict[str, dict[str, Any]] = {}
    diagnostic_status_counts: dict[str, int] = {}
    delivery_counts = _empty_delivery_counts()
    option_selection_count = 0
    forced_noop_count = 0
    no_valid_option_count = 0
    for row in episodes:
        _merge_counts(delivery_counts, row.get("delivery_counts", {}))
        _merge_counts(diagnostic_status_counts, row.get("diagnostic_status_counts", {}))
        option_selection_count += int(row.get("option_selection_count", 0))
        forced_noop_count += int(row.get("forced_noop_count", 0))
        no_valid_option_count += int(row.get("no_valid_option_count", 0))
        for key, value in row["termination_counts"].items():
            term_counts[key] = term_counts.get(key, 0) + int(value)
        for kind, item in row.get("option_kind_stats", {}).items():
            aggregate = option_kind_stats.setdefault(
                kind,
                {
                    "attempt_count": 0,
                    "success_count": 0,
                    "timeout_count": 0,
                    "success_rate": 0.0,
                    "termination_reason_histogram": {},
                },
            )
            aggregate["attempt_count"] = int(aggregate["attempt_count"]) + int(
                item.get("attempt_count", 0)
            )
            aggregate["success_count"] = int(aggregate["success_count"]) + int(
                item.get("success_count", 0)
            )
            aggregate["timeout_count"] = int(aggregate["timeout_count"]) + int(
                item.get("timeout_count", 0)
            )
            histogram = aggregate["termination_reason_histogram"]
            for reason, count in item.get("termination_reason_histogram", {}).items():
                histogram[reason] = int(histogram.get(reason, 0)) + int(count)
    for item in option_kind_stats.values():
        item["success_rate"] = float(
            int(item["success_count"]) / max(1, int(item["attempt_count"]))
        )
    return {
        "mean_return": _mean_or_nan(returns),
        "return_std": float(np.std(returns)) if returns else float("nan"),
        "completion_rate": _mean_or_nan([float(row["completed"]) for row in episodes]),
        "headline_success_metric": "ego_correct_completion_rate",
        "ego_correct_completion_rate": _mean_or_nan([float(row.get("ego_correct_completed", False)) for row in episodes]),
        "team_completion_rate": _mean_or_nan([float(row.get("team_completed", False)) for row in episodes]),
        "partner_correct_completion_rate": _mean_or_nan([float(row.get("partner_correct_completed", False)) for row in episodes]),
        "wrong_delivery_rate": _mean_or_nan([float(row.get("wrong_delivery_completed", False)) for row in episodes]),
        "partner_delivery_episode_rate": _mean_or_nan([float(row.get("partner_delivery_episode", False)) for row in episodes]),
        "wrong_delivery_episode_rate": _mean_or_nan([float(row.get("wrong_delivery_episode", False)) for row in episodes]),
        "delivery_counts": delivery_counts,
        "ego_delivery_count": int(delivery_counts["ego_delivery_event"]),
        "partner_delivery_count": int(delivery_counts["partner_delivery_event"]),
        "correct_delivery_count": int(delivery_counts["correct_delivery"]),
        "wrong_delivery_count": int(delivery_counts["wrong_delivery_event"]),
        "ego_correct_delivery_count": int(delivery_counts["ego_correct_delivery"]),
        "ego_sole_correct_delivery_count": int(delivery_counts.get("ego_sole_correct_delivery", 0)),
        "partner_correct_delivery_count": int(delivery_counts["partner_correct_delivery"]),
        "ego_wrong_delivery_count": int(delivery_counts["ego_wrong_delivery_event"]),
        "partner_wrong_delivery_count": int(delivery_counts["partner_wrong_delivery_event"]),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "mean_duration": _mean_or_nan([float(row["primitive_steps"]) for row in episodes]),
        "primitive_steps": int(primitive_steps),
        "option_selection_count": int(option_selection_count),
        "forced_noop_count": int(forced_noop_count),
        "no_valid_option_count": int(no_valid_option_count),
        "forced_noop_fraction": float(forced_noop_count / max(1, option_selection_count)),
        "no_valid_option_fraction": float(no_valid_option_count / max(1, option_selection_count)),
        "diagnostic_status_counts": diagnostic_status_counts,
        "termination_counts": term_counts,
        "option_kind_stats": option_kind_stats,
        "delta_info": _weighted_episode_summary(episodes, "delta_info_mean"),
        "mi": _weighted_episode_summary(episodes, "mi_mean"),
        "diagnostic_cost": _weighted_episode_summary(episodes, "diagnostic_cost_mean"),
        "belief_swap_delta": _aggregate_swap([row["belief_swap"] for row in episodes]),
        "belief_influence": _weighted_belief_influence_summary(episodes),
    }


def _base_q_values(
    ctx: EvalContext,
    obs_tensor: torch.Tensor,
    graph_batch: dict[str, Any],
    belief: torch.Tensor,
) -> torch.Tensor:
    if hasattr(ctx.q_net, "q_net") and hasattr(ctx.q_net.q_net, "q_base_values"):
        encoded = ctx.q_net.encoder(obs_tensor)
        return ctx.q_net.q_net.q_base_values(encoded, graph_batch["option_mask"])
    return ctx.q_net(obs_tensor, belief, **_q_forward_kwargs(graph_batch))


def _resolve_partner_names(
    option_lib: OCV2OptionLibrary,
    selector: str,
    *,
    partner_set: str = "standard7",
) -> list[str]:
    partners = [
        partner.name
        for partner in make_training_partners(option_lib, partner_set=partner_set)
    ]
    if selector == "all":
        return partners
    requested = _parse_csv(selector)
    missing = sorted(set(requested) - set(partners))
    if missing:
        raise KeyError(f"Unknown partners {missing}; choices={partners}")
    return requested


def _sibling_checkpoint(anchor: Path, variant: str) -> Path:
    seed_dir = anchor.parent
    method_dir = seed_dir.parent.parent
    return method_dir / variant / seed_dir.name / "checkpoint.pt"


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _summary(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
    wall: float,
) -> dict[str, Any]:
    def _mean_or_zero(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    closures = _relative_values(results, "reference_gap_closure", "value")
    raw_closures = _relative_values(results, "reference_gap_closure", "raw_value")
    within_run = _relative_values(results, "within_run_relative_return", "value")
    raw_within_run = _relative_values(results, "within_run_relative_return", "raw_value")
    summary = {
        "num_results": len(results),
        "partners": sorted(baselines),
        "mean_return": _mean_or_nan(
            [float(row["aggregate"]["mean_return"]) for row in results]
        ),
        "mean_completion_rate": _mean_or_nan(
            [float(row["aggregate"]["completion_rate"]) for row in results]
        ),
        "mean_blocking_rate": _mean_or_nan(
            [float(row["aggregate"]["blocking_rate"]) for row in results]
        ),
        "wall_time_sec": float(wall),
    }
    if closures:
        summary["mean_reference_gap_closure"] = _mean_or_nan(closures)
    if raw_closures:
        summary["mean_reference_gap_closure_raw"] = _mean_or_nan(raw_closures)
        summary["num_negative_raw_reference_gaps"] = int(
            sum(1 for value in raw_closures if float(value) < 0.0)
        )
    if within_run:
        summary["mean_within_run_relative_return"] = _mean_or_nan(within_run)
    if raw_within_run:
        summary["mean_within_run_relative_return_raw"] = _mean_or_nan(raw_within_run)
    summary["mean_role_match_rate"] = None
    summary["role_match_status"] = "removed_from_formal_main_path_p5"
    summary["mean_ego_correct_completion_rate"] = _mean_or_zero([
        float(r["aggregate"].get("ego_correct_completion_rate", 0.0)) for r in results
    ])
    summary["mean_partner_correct_completion_rate"] = _mean_or_zero([
        float(r["aggregate"].get("partner_correct_completion_rate", 0.0)) for r in results
    ])
    summary["mean_wrong_delivery_rate"] = _mean_or_zero([
        float(r["aggregate"].get("wrong_delivery_rate", 0.0)) for r in results
    ])
    summary["mean_ego_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("ego_correct_delivery_count", 0.0)) for r in results
    ])
    summary["mean_partner_delivery_count"] = _mean_or_zero([
        float(r["aggregate"].get("partner_correct_delivery_count", 0.0)) for r in results
    ])
    return summary


def _relative_values(
    results: list[dict[str, Any]],
    key: str,
    value_key: str,
) -> list[float]:
    values = []
    for row in results:
        metric = row["aggregate"].get(key)
        if not isinstance(metric, dict):
            continue
        value = metric.get(value_key)
        if value is not None:
            values.append(float(value))
    return values


def _finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return {
        "mean": _mean_or_nan(finite),
        "count": len(finite),
        "status": "ok" if finite else "no_values",
    }


def _aggregate_swap(values: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [value for value in values if value.get("status") == "ok"]
    if not ok:
        statuses = sorted({str(value.get("status", "unknown")) for value in values})
        return {"status": "no_valid_swaps", "value": None, "input_statuses": statuses}
    pair_rows = [
        pair
        for value in ok
        for pair in value.get("pairs", [])
        if pair.get("status") == "ok"
    ]
    return {
        "status": "ok",
        "num_option_diagnostics": len(ok),
        "num_pair_rows": len(pair_rows),
        "pairs": pair_rows,
        "mean_abs_maxq_delta": _mean_or_nan(
            [float(value["mean_abs_maxq_delta"]) for value in ok]
        ),
        "mean_abs_q_delta": _mean_or_nan(
            [float(value["mean_abs_q_delta"]) for value in ok]
        ),
        "action_flip_rate": _mean_or_nan(
            [float(value["action_flip_rate"]) for value in ok]
        ),
    }


def _aggregate_belief_influence(values: list[dict[str, float]]) -> dict[str, Any]:
    keys = ("belief_zero_delta", "belief_uniform_delta", "relevance_zero_delta")
    finite_by_key: dict[str, list[float]] = {key: [] for key in keys}
    for row in values:
        for key in keys:
            value = float(row.get(key, 0.0))
            if np.isfinite(value):
                finite_by_key[key].append(value)
    return {
        "mean_belief_zero_delta": _mean_or_nan(finite_by_key["belief_zero_delta"]),
        "mean_belief_uniform_delta": _mean_or_nan(finite_by_key["belief_uniform_delta"]),
        "mean_relevance_zero_delta": _mean_or_nan(finite_by_key["relevance_zero_delta"]),
        "count": max((len(v) for v in finite_by_key.values()), default=0),
        "status": "ok" if any(finite_by_key.values()) else "no_values",
    }


def _weighted_belief_influence_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        ("mean_belief_zero_delta", "belief_zero_delta"),
        ("mean_belief_uniform_delta", "belief_uniform_delta"),
        ("mean_relevance_zero_delta", "relevance_zero_delta"),
    )
    totals = {out_key: 0.0 for out_key, _ in keys}
    count = 0
    for row in episodes:
        influence = row.get("belief_influence", {})
        if not isinstance(influence, dict):
            continue
        row_count = int(row.get("belief_influence_count", row.get("diagnostic_count", 0)))
        if row_count <= 0:
            continue
        row_values: dict[str, float] = {}
        valid = True
        for out_key, _raw_key in keys:
            value = float(influence.get(out_key, float("nan")))
            if not np.isfinite(value):
                valid = False
                break
            row_values[out_key] = value
        if not valid:
            continue
        for out_key in totals:
            totals[out_key] += row_values[out_key] * row_count
        count += row_count
    if count <= 0:
        return {
            "mean_belief_zero_delta": 0.0,
            "mean_belief_uniform_delta": 0.0,
            "mean_relevance_zero_delta": 0.0,
            "count": 0,
            "status": "no_values",
        }
    return {
        "mean_belief_zero_delta": float(totals["mean_belief_zero_delta"] / count),
        "mean_belief_uniform_delta": float(totals["mean_belief_uniform_delta"] / count),
        "mean_relevance_zero_delta": float(totals["mean_relevance_zero_delta"] / count),
        "count": int(count),
        "status": "ok",
    }


def _weighted_episode_summary(
    episodes: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    total = 0.0
    count = 0
    for row in episodes:
        row_count = int(row.get("diagnostic_count", 0))
        value = float(row.get(key, 0.0))
        if row_count <= 0 or not np.isfinite(value):
            continue
        total += value * row_count
        count += row_count
    return {
        "mean": float(total / count) if count else float("nan"),
        "count": count,
        "status": "ok" if count else "no_values",
    }


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _empty_delivery_counts() -> dict[str, int]:
    return {
        "delivery_event": 0,
        "ego_delivery_event": 0,
        "partner_delivery_event": 0,
        "correct_delivery": 0,
        "wrong_delivery_event": 0,
        "ego_correct_delivery": 0,
        "ego_sole_correct_delivery": 0,
        "partner_correct_delivery": 0,
        "ego_wrong_delivery_event": 0,
        "partner_wrong_delivery_event": 0,
    }


def _accumulate_delivery_counts(counts: dict[str, int], event: Any) -> None:
    for key in counts:
        counts[key] = int(counts.get(key, 0)) + int(bool(getattr(event, key, False)))


def _merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _record_selection_attempt(selection_stats: dict[str, int] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["option_selection_count"] = int(
        selection_stats.get("option_selection_count", 0)
    ) + 1


def _record_forced_noop(selection_stats: dict[str, int] | None) -> None:
    if selection_stats is None:
        return
    selection_stats["no_valid_option_count"] = int(
        selection_stats.get("no_valid_option_count", 0)
    ) + 1
    selection_stats["forced_noop_count"] = int(
        selection_stats.get("forced_noop_count", 0)
    ) + 1


def _validate_eval_integrity(
    aggregate: dict[str, Any],
    *,
    collect_diagnostics: bool,
    allow_diag_skip: bool,
) -> None:
    if int(aggregate.get("forced_noop_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval encountered no-valid-option forced noop events: "
            f"{aggregate['forced_noop_count']}. This hard integrity check is not "
            "bypassed by --allow_diag_skip."
        )
    evidence = aggregate.get("partner_option_evidence", {})
    # E2 (METHOD_LOCK sec18.8): admit the explicit zeroed-channel ablation policy
    # alongside the formal one. The oracle_source/observed_dist/missing hard checks
    # below still run unconditionally, so admitting this string does NOT weaken the
    # real-path guarantees — a zeroed run routes withheld steps to `zeroed_count`, so
    # missing_count stays 0, while any true oracle leak would still hard-fail.
    _allowed_evidence_policies = {
        "behavior_inferred_v1",
        "behavior_inferred_v1_zeroed_ablation",
    }
    if str(evidence.get("evidence_policy")) not in _allowed_evidence_policies:
        raise RuntimeError(f"Formal eval has wrong partner-option evidence policy: {evidence!r}")
    if int(evidence.get("observed_dist_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed non-behavior partner-option distributions: "
            f"{evidence['observed_dist_count']} events."
        )
    if int(evidence.get("missing_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed missing partner-option evidence: "
            f"{evidence['missing_count']} events."
        )
    if int(evidence.get("oracle_source_count", 0)) > 0:
        raise RuntimeError(
            "Formal eval observed oracle partner-option evidence source: "
            f"{evidence['oracle_source_count']} events."
        )
    if collect_diagnostics:
        bad = {
            key: value
            for key, value in (
                ("delta_info", aggregate.get("delta_info", {})),
                ("mi", aggregate.get("mi", {})),
                ("diagnostic_cost", aggregate.get("diagnostic_cost", {})),
            )
            if isinstance(value, dict) and value.get("status") not in {"ok", "unsupported_method"}
        }
        if bad and not allow_diag_skip:
            raise RuntimeError(
                "Formal eval diagnostics produced no valid values: "
                f"{json.dumps(_jsonable(bad), sort_keys=True)}. "
                "Pass --allow_diag_skip only for smoke runs."
            )


def _factor_deletion_episode_count(args: argparse.Namespace, fast: bool) -> int:
    requested = getattr(args, "factor_deletion_episodes", None)
    if requested is not None:
        return int(requested)
    return 0 if fast else 3


def _eval_provenance(ctx: EvalContext) -> dict[str, Any]:
    return runtime_provenance(
        config=ctx.config,
        layout_graph=ctx.layout_graph,
        option_lib=ctx.option_lib,
        partners=_select_train_partners(ctx.option_lib, ctx.config),
        ce_path=ctx.config.get("graph", {}).get("ce_path"),
        replay_path=ctx.config.get("graph", {}).get("replay_path"),
        graph_path=ctx.config.get("graph", {}).get("graph_path"),
    )


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    return 0


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _update_option_kind_stats(
    stats: dict[str, dict[str, Any]],
    option_kind: str,
    termination_reason: str,
) -> None:
    item = stats.setdefault(
        str(option_kind),
        {
            "attempt_count": 0,
            "success_count": 0,
            "timeout_count": 0,
            "success_rate": 0.0,
            "termination_reason_histogram": {},
        },
    )
    item["attempt_count"] = int(item["attempt_count"]) + 1
    item["success_count"] = int(item["success_count"]) + int(
        option_success(option_kind, termination_reason)
    )
    item["timeout_count"] = int(item["timeout_count"]) + int(
        str(termination_reason) in {"max_steps", "env_max_steps"}
    )
    histogram = item["termination_reason_histogram"]
    reason = str(termination_reason)
    histogram[reason] = int(histogram.get(reason, 0)) + 1
    item["success_rate"] = float(
        int(item["success_count"]) / max(1, int(item["attempt_count"]))
    )


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate OvercookedV2 ARIS checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph_variants", required=True)
    parser.add_argument("--partners", default="all")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_options", type=int, default=0)
    parser.add_argument("--factor_deletion_episodes", type=int, default=None)
    parser.add_argument(
        "--allow_diag_skip",
        action="store_true",
        help="Smoke/debug escape hatch: record diagnostic skips instead of failing formal eval.",
    )
    parser.add_argument(
        "--random_policy_only",
        action="store_true",
        help="Use the checkpoint only to load env/graph context and evaluate random valid options.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Gate-only mode: skip per-option diagnostics and factor q-proxy "
        "(mean_return unaffected). Used for fast verification matrices.",
    )
    parser.add_argument("--reference_base_checkpoint", default=None)
    parser.add_argument("--reference_ref_checkpoint", default=None)
    parser.add_argument(
        "--baseline_cache_dir",
        default=None,
        help="Directory to cache checkpoint-independent reference/random baseline "
        "rollouts across eval invocations (E1 speedup). Defaults to "
        "<output_dir>/.baseline_cache. Pass 'none' to disable.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
