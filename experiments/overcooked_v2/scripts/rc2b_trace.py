"""RC-2b per-step trace of a single option kind (default pick_plate).

Drives the scripted pipeline and dumps every primitive step of the traced option so we can
see whether the agent reaches the stand cell, faces the entity, interacts, and whether the
interaction changes inventory -- pinpointing the 0%-success cause (reach / interact / block).

Usage:
    python experiments/overcooked_v2/scripts/rc2b_trace.py \
        --checkpoint results/ocv2_rc1_bounded/cramped_room/aris_bellman/full_support/seed0 \
        --partner dish-server --kind pick_plate --max-option-steps 16 --force-path-planning true
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.options import OCV2OptionLibrary  # noqa: E402
from experiments.overcooked_v2.state_utils import (  # noqa: E402
    get_inventory, has_plate, is_ingredient, is_plated_cooked_soup,
    is_pot_cooking, is_pot_ready_for_plate,
)

VARIANT = "full_support"


def make_fsm(graph):
    def fsm(ctx, state, valid_ids):
        kinds = {int(i): str(graph.options[int(i)].kind) for i in valid_ids}

        def pick(k):
            for i in valid_ids:
                if kinds[int(i)] == k:
                    return int(i)
            return None

        pots = [e.pos for e in ctx.layout_graph.entities.values() if e.kind == "pot"]
        soup_in_progress = any(
            is_pot_cooking(state, p) or is_pot_ready_for_plate(state, p, require_correct_recipe=False)
            for p in pots
        )
        inv = get_inventory(state, 0)
        if is_plated_cooked_soup(inv):
            return pick("serve_soup")
        if has_plate(inv):
            return pick("plate_soup")
        if is_ingredient(inv):
            d = pick("deliver_ingredient_to_pot")
            return d if d is not None else pick("drop_item_to_counter")
        if soup_in_progress:
            pp = pick("pick_plate")
            if pp is not None:
                return pp
        f = pick("fetch_ingredient")
        return f if f is not None else pick("pick_plate")

    return fsm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--partner", default="dish-server")
    ap.add_argument("--kind", default="pick_plate")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-option-steps", type=int, default=16)
    ap.add_argument("--force-path-planning", choices=("true", "false"), default="true")
    ap.add_argument("--block-patience", type=int, default=0)
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()

    ckpt = (REPO_ROOT / args.checkpoint).resolve()
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoint.pt"
    ctx = E._load_context(ckpt, VARIANT)
    ctx.config.setdefault("env", {})["force_path_planning"] = (args.force_path_planning == "true")
    ctx.config["options"]["max_option_steps"] = int(args.max_option_steps)
    ctx.config["options"]["block_patience"] = int(args.block_patience)
    ctx.option_lib = OCV2OptionLibrary(ctx.layout_graph, max_option_steps=int(args.max_option_steps))

    ctx.scripted_fsm = make_fsm(ctx.graph)
    ctx.trace_steps = []
    ctx.trace_kind = None if args.kind == "all" else args.kind

    print(f"=== RC-2b trace: kind={args.kind} partner={args.partner} "
          f"budget={args.max_option_steps} fpp={args.force_path_planning} ===")
    try:
        E._evaluate_partner(
            ctx, args.partner, episodes=int(args.episodes), seed=0,
            max_episode_options=40, graph_override=ctx.graph,
            random_policy=False, collect_diagnostics=False, allow_diag_skip=True,
        )
    except Exception as ex:
        print(f"(rollout raised post-hoc: {type(ex).__name__}: {ex})")

    rec = ctx.trace_steps
    print(f"total traced steps for kind={args.kind}: {len(rec)}\n")
    print(f"{'step':>4} {'agent':>8} {'partner':>8} {'target':>8} {'facing':>8} "
          f"{'act':>4} {'intr':>5} {'invB':>5} {'invA':>5} {'blk':>4}")
    for r in rec[: args.show]:
        _tp = str(tuple(r['target_pos'])) if r['target_pos'] is not None else "-"
        _pots = " ".join(f"c{p[1]}{'C' if p[2] else ''}{'R' if p[3] else ''}" for p in r.get("pots", []))
        print(f"{r['step']:>3} {r['kind'][:12]:12s} {str(tuple(r['agent'])):>7} {str(tuple(r['partner'])):>7} "
              f"{_tp:>7} {r['action']:>3} {str(r['is_interact'])[0]:>2} "
              f"{r['inv_before']:>4}->{r['inv_after']:<4} blk={str(r['blocked'])[0]} pots[{_pots}]")

    out = ckpt.parent.parent.parent.parent.parent / "rc2b_trace.json"
    try:
        Path(out).write_text(json.dumps({"kind": args.kind, "partner": args.partner,
                                         "steps": rec}, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
