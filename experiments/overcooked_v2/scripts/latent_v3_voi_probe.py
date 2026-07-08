"""Link-A C-5 value-of-information probe for latent_v3 partners.

This is a diagnostic probe, not a method run. The dynamic ``mode_oracle`` ego
uses the partner's diagnostic current mode to choose between a terminal-taking
script and a deferential prep script. Fixed ``fullchain`` and ``prepchain`` arms
are paired by episode seed.
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
) -> dict[str, Any]:
    env, router, partner = _build_eval_objects(ctx, partner_name, partner_set)
    full = _full_priority(ctx)
    prep = _prep_priority(ctx)
    if arm == "mode_oracle":
        ctx.scripted_fsm = _mode_oracle_fsm(partner, full, prep)
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
    per_partner_lifts = []
    for partner_name in partners:
        seed_blocks = []
        for eval_seed in eval_seeds:
            arms = {}
            for arm in ("mode_oracle", "fullchain", "prepchain"):
                ctx = E._load_context(ckpt, f"link_a_voi_{arm}")
                arms[arm] = _run_arm(
                    ctx,
                    partner_name,
                    arm=arm,
                    partner_set=str(args.partner_set),
                    episodes=int(args.episodes),
                    eval_seed=int(eval_seed),
                    max_episode_options=int(args.max_episode_options),
                )
            seed_blocks.append({"eval_seed": int(eval_seed), "arms": arms})
        oracle = _mean_seed_returns(seed_blocks, "mode_oracle")
        full = _mean_seed_returns(seed_blocks, "fullchain")
        prep = _mean_seed_returns(seed_blocks, "prepchain")
        best_blind = max(full, prep)
        lift = float(oracle - best_blind)
        rel = float(lift / max(abs(best_blind), 1e-6))
        per_partner_lifts.append(rel)
        out["partners"][partner_name] = {
            "per_eval_seed": seed_blocks,
            "return_mean": {
                "mode_oracle": oracle,
                "fullchain": full,
                "prepchain": prep,
                "best_blind": best_blind,
            },
            "absolute_lift": lift,
            "relative_lift": rel,
            "pass": bool(rel >= 0.15),
        }
    out["thresholds"] = {"relative_lift_min": 0.15}
    out["criteria"] = {
        "min_relative_lift": float(np.min(per_partner_lifts)) if per_partner_lifts else float("nan"),
        "PASS": bool(per_partner_lifts and min(per_partner_lifts) >= 0.15),
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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
