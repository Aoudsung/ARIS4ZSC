"""RC-2b option-execution inspector.

Localizes the pick_plate 0%-success bug: for each pipeline option kind, print the
navigation target cells + interaction cells + reachability (shortest-path distance from the
agent), so we can tell "can't reach the target cell" from "reaches but interaction never
fires". deliver_ingredient_to_pot works (10/10 for some partners) and serves as the control.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_inspect.py \
        --checkpoint results/ocv2_rc1_bounded/cramped_room/aris_bellman/full_support/seed0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer  # noqa: E402
from experiments.overcooked_v2.state_utils import get_agent_pos  # noqa: E402

PIPELINE = ["fetch_ingredient", "deliver_ingredient_to_pot", "pick_plate", "plate_soup", "serve_soup"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    ckpt = (REPO_ROOT / args.checkpoint).resolve()
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoint.pt"
    ctx = E._load_context(ckpt, "full_support")
    ol = ctx.option_lib
    lg = ctx.layout_graph
    spd = lg.shortest_path_dist

    env = E._build_env(ctx.graph.layout_name, ctx.config)
    env.set_featurizer(NumpyFeaturizer(lg))
    env.reset(0)
    a0 = get_agent_pos(env.state, 0)
    a1 = get_agent_pos(env.state, 1)
    passable = set()
    for (u, v) in spd.keys():
        passable.add(u)
        passable.add(v)
    print(f"agent0={a0} agent1={a1}  | #passable_cells={len(passable)}")
    print(f"entities: {[(eid, e.kind, e.pos) for eid, e in lg.entities.items()]}")
    print()

    for kind in PIPELINE:
        opts = [o for o in ol.options if o.kind == kind]
        if not opts:
            print(f"{kind:28s} (NO OPTION OF THIS KIND)")
            continue
        for opt in opts:
            tcells = list(ol._target_cells(opt))
            icells = (opt.metadata or {}).get("interaction_cells")
            t_passable = [t in passable for t in tcells]
            d_from_a0 = [spd.get((a0, t)) for t in tcells]
            print(f"{kind:28s} name={opt.name}")
            print(f"   target_pos={opt.target_pos}  entity_ids={opt.entity_ids}")
            print(f"   interaction_cells(meta)={icells}")
            print(f"   _target_cells={tcells}  passable={t_passable}  dist_from_a0={d_from_a0}")


if __name__ == "__main__":
    main()
