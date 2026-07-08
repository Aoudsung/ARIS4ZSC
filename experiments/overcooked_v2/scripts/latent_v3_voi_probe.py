"""Link-A C-5 value-of-information probe for latent_v3 partners.

This is a diagnostic probe, not a method run. The dynamic ``mode_oracle`` ego
uses the partner's diagnostic current mode to choose between a terminal-taking
script and a deferential prep script. Mode-blind arms paired by episode seed:
``fullchain`` (always take), ``prepchain`` (always defer), and ``reactive``
(wait-and-see: defer, observe the open opportunity for W option-decisions, take
over iff the partner has not committed — plate OR soup acquisition counts as
commitment). ``reactive`` is the binding blind baseline for the north-star
question — emergent belief only pays if anticipation beats reaction.

C-5 aggregation (amended 2026-07-08, user-signed after cert r1 overturn): the
judged statistic is the POOLED mode_oracle lift over the best SINGLE blind ego
across the certified partner set. Per-partner min-vs-post-hoc-best-static is
retained only as ``legacy_min_relative_lift`` (it is ~0 by construction for
2-policy modes and judges nothing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.partner_modes import (  # noqa: E402
    _carries_soup,
    _gate_main,
)
from experiments.overcooked_v2.state_utils import get_inventory, has_plate  # noqa: E402

KNOWN_ORDER = [
    "serve_soup", "plate_soup", "pick_plate", "deliver_ingredient_to_pot",
    "fetch_ingredient", "wait_duration_after_arrival", "wait_duration",
    "wait_at_bottleneck", "cross_bottleneck", "handoff_counter",
    "press_recipe_button", "clear_interaction_cell", "drop_item_to_counter",
]
TERMINAL_KINDS = ("serve_soup", "plate_soup", "pick_plate")


def _full_priority(ctx) -> list[str]:
    present = list(dict.fromkeys(str(o.kind) for o in ctx.graph.options))
    ordered = [k for k in KNOWN_ORDER if k in present]
    extras = [k for k in present if k not in KNOWN_ORDER and k != "noop"]
    return ordered + extras + (["noop"] if "noop" in present else [])


def _prep_priority(ctx) -> list[str]:
    return [kind for kind in _full_priority(ctx) if kind not in TERMINAL_KINDS]


def _resolve_ckpt(path_str: str) -> Path:
    ckpt = Path(path_str)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    if ckpt.is_dir():
        for name in ("checkpoint.pt", "checkpoint_final.pt"):
            if (ckpt / name).exists():
                return ckpt / name
        raise FileNotFoundError(f"no checkpoint.pt/checkpoint_final.pt under {ckpt}")
    return ckpt


def _reactive_fsm(ctx, full_priority: list[str], prep_priority: list[str],
                  wait_options: int):
    """Mode-blind wait-and-see ego: defer by default; within an open opportunity
    window (onset gate), take over after W option-decisions iff the partner has
    not initiated (soup acquisition observed between boundary samples; boundary
    aliasing accepted — returns are measured, not assumed)."""
    pots = tuple(
        entity.pos
        for entity in ctx.layout_graph.entities.values()
        if entity.kind == "pot"
    )
    st: dict[str, Any] = {}

    def _reset() -> None:
        st.update(last=None, in_window=False, waited=0, partner_took=False)

    _reset()

    def _fsm(fsm_ctx, state, valid_ids):
        if st["last"] is not None:
            before = int(get_inventory(st["last"], 1))
            after = int(get_inventory(state, 1))
            # plate OR soup acquisition = commit signal: in the round-2 substrate
            # only claim-mode partners ever convert terminal interacts (codex r2)
            soup_gained = _carries_soup(after) and not _carries_soup(before)
            plate_gained = has_plate(after) and not has_plate(before)
            if soup_gained or plate_gained:
                st["partner_took"] = True
        gate = _gate_main(state, pots)
        if gate and not st["in_window"]:
            st["in_window"] = True
            st["waited"] = 0
            st["partner_took"] = False
        if not gate:
            st["in_window"] = False
        ego_committed = _carries_soup(int(get_inventory(state, 0)))
        take_over = ego_committed or (
            st["in_window"]
            and not st["partner_took"]
            and st["waited"] >= int(wait_options)
        )
        if st["in_window"] and not st["partner_took"]:
            st["waited"] += 1
        st["last"] = state
        priority = full_priority if take_over else prep_priority
        old = getattr(fsm_ctx, "scripted_priority", None)
        fsm_ctx.scripted_priority = priority
        try:
            return E._scripted_priority_select(fsm_ctx, fsm_ctx.graph, valid_ids)
        finally:
            fsm_ctx.scripted_priority = old

    return _fsm, _reset


def _mode_oracle_fsm(partner: Any, full_priority: list[str], prep_priority: list[str]):
    def _fsm(ctx, state, valid_ids):
        sync_public_state = getattr(partner, "sync_public_state", None)
        if callable(sync_public_state):
            sync_public_state(state)
        diag_fn = getattr(partner, "diagnostic_mode_state", None)
        if not callable(diag_fn):
            return None
        diag = diag_fn()
        policy_id = int(diag.get("mode_policy_id", -1))
        priority = full_priority if policy_id == 0 else prep_priority
        old = getattr(ctx, "scripted_priority", None)
        ctx.scripted_priority = priority
        try:
            return E._scripted_priority_select(ctx, ctx.graph, valid_ids)
        finally:
            ctx.scripted_priority = old

    return _fsm


def _build_eval_objects(ctx, partner_name: str, partner_set: str):
    env = E._build_env(ctx.graph.layout_name, ctx.config)
    env.set_featurizer(E.NumpyFeaturizer(ctx.layout_graph))
    router = E.OCV2EvidenceRouter(
        ctx.graph,
        ctx.layout_graph.cell_to_entity,
        ctx.layout_graph.region_cells,
        evidence_policy=E._evidence_policy_for_config(ctx.config),
    )
    partners = {
        partner.name: partner
        for partner in E.make_training_partners(ctx.option_lib, partner_set=partner_set)
    }
    if partner_name not in partners:
        raise KeyError(f"unknown partner {partner_name!r}; choices={sorted(partners)}")
    return env, router, partners[partner_name]


def _run_arm(
    ctx,
    partner_name: str,
    *,
    arm: str,
    partner_set: str,
    episodes: int,
    eval_seed: int,
    max_episode_options: int,
    reactive_wait: int = 2,
) -> dict[str, Any]:
    env, router, partner = _build_eval_objects(ctx, partner_name, partner_set)
    full = _full_priority(ctx)
    prep = _prep_priority(ctx)
    fsm_reset = None
    if arm == "mode_oracle":
        ctx.scripted_fsm = _mode_oracle_fsm(partner, full, prep)
        ctx.scripted_priority = None
    elif arm == "reactive":
        ctx.scripted_fsm, fsm_reset = _reactive_fsm(ctx, full, prep, reactive_wait)
        ctx.scripted_priority = None
    elif arm == "fullchain":
        ctx.scripted_fsm = None
        ctx.scripted_priority = full
    elif arm == "prepchain":
        ctx.scripted_fsm = None
        ctx.scripted_priority = prep
    else:
        raise ValueError(f"unknown arm {arm!r}")
    ctx.qaudit = None
    rows = []
    for i in range(int(episodes)):
        seed = int(eval_seed) + i
        rng = np.random.default_rng(seed)
        if fsm_reset is not None:
            fsm_reset()
        row = E._run_episode(
            ctx,
            env,
            partner,
            router,
            ctx.graph,
            rng,
            seed,
            int(max_episode_options),
            random_policy=False,
            collect_diagnostics=False,
            allow_diag_skip=True,
        )
        row["episode_id"] = int(i)
        rows.append(row)
    agg = E._aggregate_episodes(rows)
    agg["partner_option_evidence"] = router.partner_option_evidence_summary()
    agg["return_mean"] = float(np.mean([float(r["return"]) for r in rows])) if rows else float("nan")
    agg["episode_returns"] = [float(r["return"]) for r in rows]
    return agg


def _mean_seed_returns(seed_blocks: list[dict[str, Any]], arm: str) -> float:
    vals = [float(block["arms"][arm]["return_mean"]) for block in seed_blocks]
    return float(np.mean(vals)) if vals else float("nan")


def run(args: argparse.Namespace) -> None:
    ckpt = _resolve_ckpt(args.checkpoint)
    partners = [p for p in args.partners.split(",") if p]
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s]
    out: dict[str, Any] = {
        "diag": "Link-A C-5 latent_v3 value of information",
        "checkpoint": str(ckpt),
        "partner_set": str(args.partner_set),
        "episodes": int(args.episodes),
        "eval_seeds": eval_seeds,
        "max_episode_options": int(args.max_episode_options),
        "partners": {},
    }
    arms_order = ("mode_oracle", "fullchain", "prepchain", "reactive")
    blind_arms = ("fullchain", "prepchain", "reactive")
    per_partner_lifts = []
    for partner_name in partners:
        seed_blocks = []
        for eval_seed in eval_seeds:
            arms = {}
            for arm in arms_order:
                ctx = E._load_context(ckpt, f"link_a_voi_{arm}")
                arms[arm] = _run_arm(
                    ctx,
                    partner_name,
                    arm=arm,
                    partner_set=str(args.partner_set),
                    episodes=int(args.episodes),
                    eval_seed=int(eval_seed),
                    max_episode_options=int(args.max_episode_options),
                    reactive_wait=int(args.reactive_wait),
                )
            seed_blocks.append({"eval_seed": int(eval_seed), "arms": arms})
        means = {arm: _mean_seed_returns(seed_blocks, arm) for arm in arms_order}
        best_blind = max(means[a] for a in blind_arms)
        lift = float(means["mode_oracle"] - best_blind)
        rel = float(lift / max(abs(best_blind), 1e-6))
        # legacy co-report keeps its original STATIC-only definition (codex r2)
        best_static = max(means["fullchain"], means["prepchain"])
        per_partner_lifts.append(
            float((means["mode_oracle"] - best_static) / max(abs(best_static), 1e-6)))
        out["partners"][partner_name] = {
            "per_eval_seed": seed_blocks,
            "return_mean": {**means, "best_blind": best_blind},
            "absolute_lift": lift,
            "relative_lift": rel,
        }
    # C-5 judged statistic (amended 2026-07-08): pooled lift over the best SINGLE
    # blind ego across the certified set; per-partner min is co-report only.
    pooled = {
        arm: float(np.mean(
            [out["partners"][p]["return_mean"][arm] for p in partners]))
        for arm in arms_order
    }
    best_blind_arm = max(blind_arms, key=lambda a: pooled[a])
    pooled_lift = float(pooled["mode_oracle"] - pooled[best_blind_arm])
    pooled_rel = float(pooled_lift / max(abs(pooled[best_blind_arm]), 1e-6))
    out["thresholds"] = {"pooled_relative_lift_min": 0.15, "reactive_wait": int(args.reactive_wait)}
    out["criteria"] = {
        "pooled_return_means": pooled,
        "best_blind_arm": best_blind_arm,
        "pooled_absolute_lift": pooled_lift,
        "pooled_relative_lift": pooled_rel,
        "legacy_min_relative_lift": float(np.min(per_partner_lifts))
        if per_partner_lifts else float("nan"),
        "PASS": bool(pooled_rel >= 0.15),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["criteria"], indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partner_set", default="latent_v3_dev")
    ap.add_argument(
        "--partners",
        default=",".join([
            "blind-cert-ingnear-titfortat3",
            "blind-cert-ingfar-escalate2",
            "blind-cert-prepnear-patience5",
            "blind-cert-prepfar-block20",
            "blind-cert-bneck-patience3",
            "blind-cert-flex-block16",
        ]),
    )
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--eval-seeds", default="0")
    ap.add_argument("--max-episode-options", type=int, default=60)
    ap.add_argument("--reactive-wait", type=int, default=2,
                    help="option-decisions the reactive arm observes an open "
                         "opportunity before taking over")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
