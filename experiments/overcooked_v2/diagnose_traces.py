"""Per-decision Q-rank + belief traces for a trained cramped-room ARIS checkpoint.

For each option decision in an eval episode, log:
  - state summary (agent inv, pot state)
  - valid option set + kinds
  - selected option
  - top-5 valid options with Q values
  - Q_base per option
  - belief entropy per factor + relevance to the selected option
  - belief-swap ΔQ for each factor swapped to uniform (proxy for factor deletion)

Standalone: does not touch evaluate_aris. Uses the same context loader so the
model + graph + option lib exactly match training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.evaluate_aris import (
    _load_context,
    _execute_eval_option,
    _current_belief,
    _q_forward_kwargs,
    _graph_tensors,
    _tensor,
)
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.reward_design import ContributionLedger
from experiments.overcooked_v2.state_utils import (
    get_agent_pos, get_inventory, get_pot_contents,
    is_pot_cooking, is_pot_ready_for_plate,
)
from src.aris_bellman.replay import EvidenceBuffer
from src.aris_bellman.metrics import masked_entropy


def _obs_vec(obs, key="agent_0"):
    v = obs.get(key) if isinstance(obs, dict) else obs
    return np.asarray(v, dtype=np.float32)


def _top_k_valid(q_values, valid_ids, k, options):
    pairs = [(int(i), float(q_values[int(i)].item())) for i in valid_ids]
    pairs.sort(key=lambda x: -x[1])
    return [{"rank": r, "option_id": vid, "kind": options[vid].kind, "q": q}
            for r, (vid, q) in enumerate(pairs[:k])]


def _state_summary(state, lg, agent_id=0):
    pots = []
    for e in lg.entities.values():
        if e.kind == "pot":
            pots.append({
                "pos": list(e.pos),
                "contents": int(get_pot_contents(state, e.pos)),
                "cooking": bool(is_pot_cooking(state, e.pos)),
                "ready": bool(is_pot_ready_for_plate(state, e.pos, require_correct_recipe=False)),
            })
    return {
        "ego_pos": list(get_agent_pos(state, 0)),
        "partner_pos": list(get_agent_pos(state, 1)),
        "ego_inv": int(get_inventory(state, 0)),
        "partner_inv": int(get_inventory(state, 1)),
        "pots": pots,
    }


def _belief_swap_delta_q(q_net, obs_tensor, belief, mode_mask_t, gb_kw, factor_idx):
    """Replace one factor's belief with uniform (over valid modes) and re-forward.
       Returns the resulting Q vector delta (Q_swapped - Q_original) per option."""
    b2 = belief.clone()
    mask_f = mode_mask_t[0, factor_idx].to(dtype=b2.dtype)
    n_valid = mask_f.sum().clamp(min=1.0)
    b2[0, factor_idx] = mask_f / n_valid
    with torch.no_grad():
        q2 = q_net(obs_tensor, b2, **gb_kw).squeeze(0)
    return q2


def diagnose_episode(ctx, env, lg, lib, partner_name, seed, max_options=40):
    partner = next(p for p in make_training_partners(lib) if p.name == partner_name)
    obs, _ = env.reset(seed)
    partner.reset(seed)
    contribution_ledger = ContributionLedger.from_config(ctx.config.get("training", {}))
    rng = np.random.default_rng(seed + 999)

    device = torch.device("cpu")
    gb = _graph_tensors(ctx.graph, 1, device)
    gb_kw = _q_forward_kwargs(gb)
    mode_mask_t = gb["mode_mask"]
    evidence = EvidenceBuffer(
        num_factors=ctx.graph.num_factors,
        window=ctx.config["training"]["evidence_window"],
        evidence_dim=6,  # D_EVID default in this codebase
    )
    decisions = []
    done = False
    nopt = 0
    total_correct = ego_deliv = prt_deliv = 0
    while not done and nopt < max_options:
        state = env.state
        valid = lib.valid_options(state, 0)
        valid_ids = list(np.flatnonzero(valid).tolist())
        if not valid_ids:
            break
        obs_tensor = _tensor(_obs_vec(obs)[None, ...], device)
        belief = _current_belief(ctx, evidence, ctx.graph)
        # Q_full
        with torch.no_grad():
            q_full = ctx.q_net(obs_tensor, belief, **gb_kw).squeeze(0)
        # Q_base (encoder + base head only; the ARIS model exposes q_net.q_base_values)
        q_base_vec = None
        try:
            encoded = ctx.q_net.encoder(obs_tensor)
            q_base_vec = ctx.q_net.q_net.q_base_values(encoded, gb["option_mask"]).squeeze(0)
        except AttributeError:
            pass  # non-factor model
        # Argmax over valid
        mask_bool = torch.zeros_like(q_full, dtype=torch.bool)
        for i in valid_ids:
            mask_bool[i] = True
        q_masked = q_full.masked_fill(~mask_bool, -1e9)
        selected_id = int(torch.argmax(q_masked).item())

        # Belief entropy per factor
        h_per = []
        for f in range(belief.shape[1]):
            h = float(masked_entropy(belief[:, f, :], mode_mask_t[:, f, :]).item())
            h_per.append(h)
        # Relevance of the selected option to each factor
        rel_selected = [bool(ctx.graph.relevance[f, selected_id])
                        for f in range(ctx.graph.num_factors)]

        # Belief-swap ΔQ on the selected option for each factor (proxy for factor deletion)
        swap_delta_selected = []
        for f in range(ctx.graph.num_factors):
            q_swap = _belief_swap_delta_q(ctx.q_net, obs_tensor, belief, mode_mask_t, gb_kw, f)
            swap_delta_selected.append(float((q_swap[selected_id] - q_full[selected_id]).item()))

        decisions.append({
            "opt#": nopt,
            "state": _state_summary(state, lg),
            "valid_ids": valid_ids,
            "valid_kinds": [ctx.graph.options[i].kind for i in valid_ids],
            "selected_id": selected_id,
            "selected_kind": ctx.graph.options[selected_id].kind,
            "top5": _top_k_valid(q_full, valid_ids, 5, ctx.graph.options),
            "q_base_selected": (float(q_base_vec[selected_id].item())
                                if q_base_vec is not None else None),
            "q_full_selected": float(q_full[selected_id].item()),
            "adv_selected": (float((q_full[selected_id] - q_base_vec[selected_id]).item())
                             if q_base_vec is not None else None),
            "belief_entropy_per_factor": h_per,
            "rel_selected_per_factor": rel_selected,
            "swap_delta_q_selected_per_factor": swap_delta_selected,
        })

        # Execute the selected option
        option_return, done, obs, info = _execute_eval_option(
            ctx, env, obs, partner, ctx.evidence_router, ctx.graph, evidence,
            selected_id, collect_diagnostics=False, allow_diag_skip=True,
            rng=np.random.default_rng(seed + nopt + 1),
            contribution_ledger=contribution_ledger,
        )
        ev = (info or {}).get("event", {})
        total_correct += int((ev or {}).get("correct_delivery", False))
        ego_deliv += int((ev or {}).get("ego_correct_delivery", False))
        prt_deliv += int((ev or {}).get("partner_correct_delivery", False))
        nopt += 1

    return {
        "partner": partner_name,
        "seed": int(seed),
        "num_option_decisions": len(decisions),
        "total_correct": int(total_correct),
        "ego_correct": int(ego_deliv),
        "partner_correct": int(prt_deliv),
        "decisions": decisions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--graph_variant", default="full_support")
    ap.add_argument("--partners", required=True, help="comma-separated partner names")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ctx = _load_context(Path(args.checkpoint), args.graph_variant)
    lg = ctx.layout_graph
    lib = OCV2OptionLibrary(
        lg,
        max_option_steps=int(ctx.config.get("options", {}).get("max_option_steps", 16)),
        strict_preconditions=bool(ctx.config.get("options", {}).get("strict_preconditions", False)),
        dynamic_budget=bool(ctx.config.get("options", {}).get("dynamic_budget", False)),
    )
    env_cfg = ctx.config.get("env", {})
    env = OCV2Adapter(
        ctx.graph.layout_name,
        max_steps=int(env_cfg.get("max_steps", 400)),
        observation_type=str(env_cfg.get("observation_type", "default")),
        force_path_planning=bool(env_cfg.get("force_path_planning", False)),
    )

    all_episodes = []
    for partner in args.partners.split(","):
        for e in range(int(args.episodes)):
            all_episodes.append(diagnose_episode(ctx, env, lg, lib, partner.strip(), seed=e))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_episodes, indent=2), encoding="utf-8")
    print("wrote %s (%d episodes)" % (args.output, len(all_episodes)))


if __name__ == "__main__":
    main()
