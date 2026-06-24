from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aris_bellman.factor_belief import FactorLocalBeliefModel
from src.aris_bellman.replay import EvidenceBuffer
from src.aris_bellman.specs import FactorSpec, GraphSpec, OptionSpec

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
from experiments.overcooked_v2.event_extractor import extract_event
from experiments.overcooked_v2.evidence_router import D_EVID, OCV2EvidenceRouter
from experiments.overcooked_v2.layout_parser import LayoutGraph, parse_layout
from experiments.overcooked_v2.obs_encoder import infer_obs_dim
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.train_aris import (
    _build_belief_model,
    _build_env,
    _build_q_network,
    _graph_tensors,
    _obs_vector,
    _q_forward_kwargs,
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
    partner_names = _resolve_partner_names(contexts[0].option_lib, args.partners)
    seed = int(args.seed)
    max_episode_options = int(
        args.max_episode_options
        or contexts[0].config["training"].get("max_episode_options", 20)
    )

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
                random_policy=False,
                collect_diagnostics=True,
            )
            deletion = _factor_deletion_q_proxy_diagnostics(
                ctx,
                aggregate["mean_return"],
            )
            results.append(
                {
                    "method": ctx.method,
                    "graph_variant": ctx.graph_variant,
                    "partner": partner_name,
                    "checkpoint": str(ctx.checkpoint_path),
                    "aggregate": aggregate,
                    "episodes": episodes,
                    "factor_deletion_return_drop": deletion,
                }
            )

    baselines = _random_baselines(
        contexts[0],
        partner_names,
        episodes=int(args.episodes),
        seed=seed + 200_000,
        max_episode_options=max_episode_options,
    )
    _attach_reference_gaps(results, baselines)

    return {
        "schema_version": "ocv2_eval_v1",
        "anchor_checkpoint": str(anchor),
        "graph_variants": variants,
        "partners": partner_names,
        "episodes_per_partner": int(args.episodes),
        "max_episode_options": max_episode_options,
        "diagnostic_granularity": "option",
        "episode_return_kind": "reward_sum_minus_cost_coef_realized_cost",
        "reference_gap_semantics": {
            "r_base": "random_policy_rollout",
            "r_ref": "best_mean_return_among_requested_graph_variants_per_partner",
            "reference_type": "within_run_relative_not_external_oracle",
        },
        "results": results,
        "reference_baselines": baselines,
        "summary": _summary(results, baselines, time.time() - start),
    }


def _load_context(checkpoint_path: Path, variant: str) -> EvalContext:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for graph variant {variant}: {checkpoint_path}"
        )
    checkpoint = _torch_load(checkpoint_path)
    config = checkpoint["config"]
    graph = graph_from_json(checkpoint["graph"])
    method = str(checkpoint["method"])

    env = _build_env(graph.layout_name, config)
    obs, _ = env.reset(0)
    layout_graph = parse_layout(env, graph.layout_name)
    option_lib = OCV2OptionLibrary(
        layout_graph,
        max_option_steps=int(config["options"]["max_option_steps"]),
    )
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    env = _build_env(graph_override.layout_name, ctx.config)
    router = OCV2EvidenceRouter(
        graph_override,
        ctx.layout_graph.cell_to_entity,
        ctx.layout_graph.region_cells,
    )
    partners = {partner.name: partner for partner in make_training_partners(ctx.option_lib)}
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
        )
        row["episode_id"] = int(episode_idx)
        episode_rows.append(row)

    return _aggregate_episodes(episode_rows), episode_rows


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
) -> dict[str, Any]:
    evidence_buffer = EvidenceBuffer(
        num_factors=graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    evidence_buffer.reset()
    obs, _ = env.reset(seed)
    partner.reset(seed)

    episode_return = 0.0
    primitive_steps = 0
    option_count = 0
    blocking_events = 0
    delivery_seen = False
    delta_values: list[float] = []
    mi_values: list[float] = []
    diagnostic_cost_values: list[float] = []
    swap_values: list[dict[str, Any]] = []
    termination_counts: dict[str, int] = {}
    done = False

    while not done and option_count < max_episode_options:
        option_id = _select_option(ctx, obs, env.state, evidence_buffer, graph, rng, random_policy)
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
            rng,
        )
        episode_return += option_return
        option_count += 1
        primitive_steps += int(info["primitive_steps"])
        blocking_events += int(info["blocking_events"])
        delivery_seen = bool(delivery_seen or info["delivery_seen"])
        delta_values.extend(info["delta_info"])
        mi_values.extend(info["mi"])
        diagnostic_cost_values.extend(info["diagnostic_cost"])
        swap_values.extend(info["belief_swap"])
        _increment(termination_counts, str(info["termination_reason"]))

    return {
        "return": float(episode_return),
        "completed": bool(delivery_seen),
        "primitive_steps": int(primitive_steps),
        "option_count": int(option_count),
        "blocking_events": int(blocking_events),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "delta_info_mean": _mean_or_zero(delta_values),
        "mi_mean": _mean_or_zero(mi_values),
        "diagnostic_cost_mean": _mean_or_zero(diagnostic_cost_values),
        "diagnostic_count": len(delta_values),
        "belief_swap": _aggregate_swap(swap_values),
        "termination_counts": termination_counts,
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
    rng: np.random.Generator,
) -> tuple[float, bool, dict[str, np.ndarray], dict[str, Any]]:
    opt = ctx.option_lib.options[int(option_id)]
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
    done = False
    belief_before_option = _current_belief(ctx, evidence_buffer, graph)

    while duration < opt.max_steps:
        ego_action = ctx.option_lib.primitive_action(env.state, 0, int(option_id))
        partner_action = partner.act(obs.get("agent_1"), env.state, rng)
        prev_state = env.state
        step = env.step(ego_action, partner_action.primitive_action)
        event = extract_event(
            prev_state,
            ego_action,
            partner_action.primitive_action,
            step.state,
            step.info,
            partner_action.option_id,
            partner_action.option_dist,
        )
        x_f = router.route(event, ego_option_id=int(option_id))
        evidence_buffer.append(x_f)

        reward_sum += _training_reward(step, ctx.config, "agent_0")
        realized_cost += float(ctx.config["training"].get("cost_per_step", 1.0))
        duration += 1
        blocking_events += int(bool(event.collision_or_block))
        delivery_seen = bool(
            delivery_seen
            or event.delivery_event
            or float(step.rewards.get("agent_0", 0.0)) > 0.0
        )
        done = bool(step.dones.get("__all__", False))
        terminated, termination_reason = ctx.option_lib.option_terminated(
            opt,
            prev_state,
            step.state,
            event,
            agent_id=0,
            elapsed=duration,
        )
        obs = step.obs
        if done or terminated:
            break

    if collect_diagnostics:
        belief_after_option = _current_belief(ctx, evidence_buffer, graph)
        diag = _option_diagnostics(
            ctx,
            obs,
            int(option_id),
            belief_before_option,
            belief_after_option,
            graph,
        )
        delta_values.append(diag["delta_info"])
        mi_values.append(diag["mi"])
        diagnostic_cost_values.append(diag["diagnostic_cost"])
        swap_values.append(diag["belief_swap"])

    option_return = reward_sum - float(ctx.config["training"]["cost_coef"]) * realized_cost
    return (
        float(option_return),
        bool(done),
        obs,
        {
            "primitive_steps": int(duration),
            "blocking_events": int(blocking_events),
            "delivery_seen": bool(delivery_seen),
            "termination_reason": termination_reason,
            "delta_info": delta_values,
            "mi": mi_values,
            "diagnostic_cost": diagnostic_cost_values,
            "belief_swap": swap_values,
        },
    )


def _option_diagnostics(
    ctx: EvalContext,
    obs_next: dict[str, np.ndarray],
    option_id: int,
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    graph: GraphSpec,
) -> dict[str, Any]:
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    obs_tensor = _tensor(_obs_vector(obs_next, "agent_0")[None, :], torch.device("cpu"))
    with torch.no_grad():
        delta = realized_delta_info(
            ctx.q_net,
            obs_tensor,
            belief_before,
            belief_after,
            graph_batch,
            gamma=float(ctx.config["training"]["gamma"]),
        )
        mi = mutual_information_proxy(belief_before, belief_after, graph_batch["mode_mask"])
        q_base = _base_q_values(ctx, obs_tensor, graph_batch, belief_after)
        _, cost = diagnostic_cost(q_base, int(option_id), delta, tau=0.0)
        swap = belief_swap_top_pairs(ctx.q_net, obs_tensor, belief_after, graph_batch, graph)
        return {
            "delta_info": float(delta.mean().item()),
            "mi": float(mi.mean().item()),
            "diagnostic_cost": float(cost.mean().item()),
            "belief_swap": swap,
        }


def _current_belief(
    ctx: EvalContext,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
) -> torch.Tensor:
    graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
    evidence = _tensor(evidence_buffer.snapshot()[None, ...], torch.device("cpu"))
    with torch.no_grad():
        return _state_repr(ctx.method, ctx.belief_model, evidence, graph_batch)


def _select_option(
    ctx: EvalContext,
    obs: dict[str, np.ndarray],
    state: Any,
    evidence_buffer: EvidenceBuffer,
    graph: GraphSpec,
    rng: np.random.Generator,
    random_policy: bool,
) -> int:
    valid = ctx.option_lib.valid_options(state, 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return _noop_option_id(ctx.option_lib)
    if random_policy:
        return int(rng.choice(valid_ids))

    with torch.no_grad():
        graph_batch = _graph_tensors(graph, 1, torch.device("cpu"))
        obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, :], torch.device("cpu"))
        belief = _current_belief(ctx, evidence_buffer, graph)
        q_values = ctx.q_net(obs_tensor, belief, **_q_forward_kwargs(graph_batch)).squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
        q_values = q_values.masked_fill(~valid_tensor, -1e9)
        return int(torch.argmax(q_values).item())


def _factor_deletion_q_proxy_diagnostics(
    ctx: EvalContext,
    base_return: float,
) -> list[dict[str, Any]]:
    if ctx.graph.num_factors == 0:
        return []

    q_drops = _factor_deletion_q_proxy(ctx)

    def eval_deleted(factor_id: int) -> float:
        return float(base_return - q_drops[int(factor_id)])

    rows = factor_deletion_return_drop(ctx.graph, base_return, eval_deleted)
    for row in rows:
        row["ablation_mode"] = "q_proxy_factor_mask"
        row["ablation_scope"] = "single_initial_state_empty_evidence"
        row["rollout_episodes"] = 0
    return rows


def _factor_deletion_q_proxy(ctx: EvalContext) -> dict[int, float]:
    env = _build_env(ctx.graph.layout_name, ctx.config)
    obs, _ = env.reset(0)
    evidence_buffer = EvidenceBuffer(
        num_factors=ctx.graph.num_factors,
        window=int(ctx.config["training"]["evidence_window"]),
        evidence_dim=D_EVID,
    )
    obs_tensor = _tensor(_obs_vector(obs, "agent_0")[None, :], torch.device("cpu"))
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


def _random_baselines(
    ctx: EvalContext,
    partner_names: list[str],
    *,
    episodes: int,
    seed: int,
    max_episode_options: int,
) -> dict[str, Any]:
    baselines = {}
    for idx, partner_name in enumerate(partner_names):
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
        baselines[partner_name] = {
            "base_kind": "random_policy",
            "mean_return": aggregate["mean_return"],
            "completion_rate": aggregate["completion_rate"],
            "blocking_rate": aggregate["blocking_rate"],
        }
    return baselines


def _attach_reference_gaps(
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
            row["aggregate"]["reference_gap_closure"] = reference_gap_closure(
                float(row["aggregate"]["mean_return"]),
                r_base,
                r_ref,
            )


def _aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["return"]) for row in episodes]
    primitive_steps = sum(int(row["primitive_steps"]) for row in episodes)
    blocking_events = sum(int(row["blocking_events"]) for row in episodes)
    term_counts: dict[str, int] = {}
    for row in episodes:
        for key, value in row["termination_counts"].items():
            term_counts[key] = term_counts.get(key, 0) + int(value)
    return {
        "mean_return": _mean_or_zero(returns),
        "return_std": float(np.std(returns)) if returns else 0.0,
        "completion_rate": _mean_or_zero([float(row["completed"]) for row in episodes]),
        "blocking_rate": float(blocking_events / max(1, primitive_steps)),
        "mean_duration": _mean_or_zero([float(row["primitive_steps"]) for row in episodes]),
        "primitive_steps": int(primitive_steps),
        "termination_counts": term_counts,
        "delta_info": _weighted_episode_summary(episodes, "delta_info_mean"),
        "mi": _weighted_episode_summary(episodes, "mi_mean"),
        "diagnostic_cost": _weighted_episode_summary(episodes, "diagnostic_cost_mean"),
        "belief_swap_delta": _aggregate_swap([row["belief_swap"] for row in episodes]),
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


def _resolve_partner_names(option_lib: OCV2OptionLibrary, selector: str) -> list[str]:
    partners = [partner.name for partner in make_training_partners(option_lib)]
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


def graph_from_json(data: dict[str, Any]) -> GraphSpec:
    options = [
        OptionSpec(
            id=int(item["id"]),
            name=str(item["name"]),
            kind=str(item["kind"]),
            target_id=item.get("target_id"),
            target_pos=_pos_or_none(item.get("target_pos")),
            entity_ids=tuple(item.get("entity_ids", ())),
            region_ids=tuple(item.get("region_ids", ())),
            max_steps=int(item["max_steps"]),
            interruptible=bool(item.get("interruptible", True)),
            terminal_event=item.get("terminal_event"),
            metadata=item.get("metadata"),
        )
        for item in data["options"]
    ]
    factors = [
        FactorSpec(
            id=int(item["id"]),
            option_i=int(item["option_i"]),
            option_j=int(item["option_j"]),
            ce_score=float(item["ce_score"]),
            num_modes=int(item["num_modes"]),
            entity_ids=tuple(item.get("entity_ids", ())),
            region_ids=tuple(item.get("region_ids", ())),
            factor_kind=str(item.get("factor_kind", "generic_option_pair")),
            metadata=item.get("metadata"),
        )
        for item in data["factors"]
    ]
    return GraphSpec(
        layout_name=str(data["layout_name"]),
        options=options,
        factors=factors,
        relevance=np.asarray(data["relevance"], dtype=bool),
        option_mask=np.asarray(data["option_mask"], dtype=bool),
        factor_mask=np.asarray(data["factor_mask"], dtype=bool),
        mode_mask=np.asarray(data["mode_mask"], dtype=bool),
        route_map={
            int(key): tuple(int(value) for value in values)
            for key, values in data.get("route_map", {}).items()
        },
        option_features=_array_or_none(data.get("option_features")),
        factor_features=_array_or_none(data.get("factor_features")),
        metadata=data.get("metadata") or {},
    )


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _pos_or_none(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return int(value[0]), int(value[1])


def _summary(
    results: list[dict[str, Any]],
    baselines: dict[str, Any],
    wall: float,
) -> dict[str, Any]:
    closures = [
        row["aggregate"]["reference_gap_closure"]["value"]
        for row in results
        if row["aggregate"]["reference_gap_closure"]["value"] is not None
    ]
    raw_closures = [
        row["aggregate"]["reference_gap_closure"]["raw_value"]
        for row in results
        if row["aggregate"]["reference_gap_closure"].get("raw_value") is not None
    ]
    return {
        "num_results": len(results),
        "partners": sorted(baselines),
        "mean_return": _mean_or_zero(
            [float(row["aggregate"]["mean_return"]) for row in results]
        ),
        "mean_completion_rate": _mean_or_zero(
            [float(row["aggregate"]["completion_rate"]) for row in results]
        ),
        "mean_blocking_rate": _mean_or_zero(
            [float(row["aggregate"]["blocking_rate"]) for row in results]
        ),
        "mean_reference_gap_closure": _mean_or_zero(closures),
        "mean_reference_gap_closure_raw": _mean_or_zero(raw_closures),
        "num_negative_raw_reference_gaps": int(
            sum(1 for value in raw_closures if float(value) < 0.0)
        ),
        "wall_time_sec": float(wall),
    }


def _finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return {
        "mean": _mean_or_zero(finite),
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
        "mean_abs_maxq_delta": _mean_or_zero(
            [float(value["mean_abs_maxq_delta"]) for value in ok]
        ),
        "mean_abs_q_delta": _mean_or_zero(
            [float(value["mean_abs_q_delta"]) for value in ok]
        ),
        "action_flip_rate": _mean_or_zero(
            [float(value["action_flip_rate"]) for value in ok]
        ),
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
        "mean": float(total / count) if count else 0.0,
        "count": count,
        "status": "ok" if count else "no_values",
    }


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _noop_option_id(option_lib: OCV2OptionLibrary) -> int:
    for opt in option_lib.options:
        if opt.kind == "noop":
            return int(opt.id)
    return 0


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


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
    return parser


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
