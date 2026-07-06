"""Behavior-fingerprint probe (override-gate follow-up, seed-heterogeneity forensics).

One (checkpoint, partner, mode) per invocation; prints a single JSON blob:
  mode=argmax    : the trained policy as-is — what is the seed actually doing?
  mode=fullchain : rc2-style FULL task-chain forced play (serve > plate > pick_plate >
                   deliver > fetch > ...) — can deliberate play complete against this
                   partner at all, i.e. is the earliest-stage deadlock breakable by
                   control, or does the partner/env block solo completion?

Fingerprint = option_kind_stats (attempts/successes per kind), termination reasons,
ego/partner delivery counts, completion, return. Shared-RNG pairing is NOT claimed
here (this is a per-arm fingerprint, not a paired contrast).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    """Deferential scripted ego: full task pipeline MINUS the terminal chain.

    Used by the sec18.14 interference-sensitivity certification (checkpoint-free
    A/B): fullchain = bulldozer ego, prepchain = prep-and-defer ego. Both keep a
    wait/noop tail so selection never falls back to argmax(Q) — trained weights
    have no influence on either arm.
    """
    return [k for k in _full_priority(ctx) if k not in TERMINAL_KINDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--partner", required=True)
    ap.add_argument("--mode", choices=("argmax", "fullchain", "prepchain"), required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--max-episode-options", type=int, default=60)
    ap.add_argument("--partner_set", default=None,
                    help="eval-only partner-registry override (e.g. blind_v1)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / args.checkpoint).resolve()
    ctx = E._load_context(ckpt, "probe")
    ctx.scripted_fsm = None
    if args.mode == "fullchain":
        ctx.scripted_priority = _full_priority(ctx)
    elif args.mode == "prepchain":
        ctx.scripted_priority = _prep_priority(ctx)
    else:
        ctx.scripted_priority = None
    ctx.qaudit = None
    if args.partner_set:
        ctx.partner_set_override = str(args.partner_set)

    agg, rows = E._evaluate_partner(
        ctx, args.partner,
        episodes=int(args.episodes), seed=int(args.eval_seed),
        max_episode_options=int(args.max_episode_options),
        graph_override=ctx.graph, random_policy=False,
        collect_diagnostics=False, allow_diag_skip=True,
    )
    oks = agg.get("option_kind_stats") or {}
    out = {
        "checkpoint": str(ckpt),
        "partner": args.partner,
        "mode": args.mode,
        "episodes": int(args.episodes),
        "return_mean": float(np.mean([float(r["return"]) for r in rows])),
        "completion_rate": agg.get("completion_rate"),
        "ego_delivery_count": agg.get("ego_delivery_count"),
        "partner_delivery_count": agg.get("partner_delivery_count"),
        "termination_counts": agg.get("termination_counts"),
        "option_kind_stats": {
            k: {"att": v.get("attempt_count"), "ok": v.get("success_count")}
            for k, v in sorted(oks.items())
        },
        "forced_noop_fraction": agg.get("forced_noop_fraction"),
        "no_valid_option_fraction": agg.get("no_valid_option_fraction"),
    }
    print(json.dumps(out, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
