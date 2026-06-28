"""RC-2b F4: layout reachability scan (standalone option-level oracle).

cramped_room is provably NOT reachable at the option level (two agents livelock in 6 cells;
oracle completion 0 for all partners). This scans candidate layouts with the same state-aware
FSM + executor patience to find the ADMITTED set (layouts where the option-level oracle actually
completes the pipeline). Only admitted layouts may host the claim experiments.

No checkpoint / CE / Q / belief needed: the oracle is scripted and drives options directly.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_layout_scan.py \
        --layouts cramped_room,asymm_advantages,coord_ring,counter_circuit,forced_coord,long_room,two_rooms,cramped_room_v2 \
        --partners 2 --episodes 4 --out results/ocv2_rc1_bounded/rc2b_layout_scan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.event_extractor import extract_event  # noqa: E402
from experiments.overcooked_v2.layout_parser import parse_layout  # noqa: E402
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer  # noqa: E402
from experiments.overcooked_v2.option_termination import OptionRuntime  # noqa: E402
from experiments.overcooked_v2.options import OCV2OptionLibrary  # noqa: E402
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402
from experiments.overcooked_v2.state_utils import (  # noqa: E402
    get_agent_pos, get_inventory, has_plate, is_ingredient, is_plated_cooked_soup,
)

CONFIG = REPO_ROOT / "experiments/overcooked_v2/configs/ocv2_step4.yaml"


def fsm(option_lib, state, valid_ids):
    kinds = {int(i): str(option_lib.options[int(i)].kind) for i in valid_ids}

    def pick(k):
        for i in valid_ids:
            if kinds[int(i)] == k:
                return int(i)
        return None

    inv = get_inventory(state, 0)
    if is_plated_cooked_soup(inv):
        return pick("serve_soup")
    if has_plate(inv):
        return pick("plate_soup")
    if is_ingredient(inv):
        return pick("deliver_ingredient_to_pot")
    f = pick("fetch_ingredient")
    return f if f is not None else pick("pick_plate")


def _noop_id(option_lib):
    for o in option_lib.options:
        if o.kind == "noop":
            return o.id
    return 0


def _dist(option_lib, state, opt, spd):
    cells = option_lib._target_cells(opt)
    if not cells:
        return None
    a = get_agent_pos(state, 0)
    ds = [spd.get((a, c)) for c in cells]
    ds = [d for d in ds if d is not None]
    return min(ds) if ds else None


def run_episode(env, option_lib, partner, seed, max_options, patience, spd):
    obs, _ = env.reset(seed)
    partner.reset(seed)
    rng = np.random.default_rng(seed)
    completed = False
    reasons = defaultdict(int)
    kind_attempt = defaultdict(int)
    noop_id = _noop_id(option_lib)
    for _ in range(max_options):
        state = env.state
        valid = option_lib.valid_options(state, 0)
        valid_ids = np.flatnonzero(valid)
        oid = noop_id if valid_ids.size == 0 else fsm(option_lib, state, valid_ids)
        if oid is None:
            oid = noop_id
        opt = option_lib.options[int(oid)]
        kind_attempt[opt.kind] += 1
        runtime = OptionRuntime(option_id=int(oid), start_pos=get_agent_pos(state, 0))
        best = _dist(option_lib, state, opt, spd)
        stuck = 0
        done = False
        reason = "max_steps"
        _budget = option_lib.option_budget(state, 0, int(oid))
        for step_i in range(_budget):
            ego_a = option_lib.primitive_action(env.state, 0, int(oid))
            pa = partner.act(obs.get("agent_1"), env.state, rng)
            prev = env.state
            step = env.step(ego_a, pa.primitive_action)
            event = extract_event(prev, ego_a, pa.primitive_action, step.state, step.info,
                                  pa.option_id, pa.option_dist)
            obs = step.obs
            done = bool(step.dones.get("__all__", False))
            if getattr(event, "delivery_event", False):
                completed = True
            cur = _dist(option_lib, step.state, opt, spd)
            if cur is not None and best is not None and cur < best:
                best = cur
                stuck = 0
            elif cur is not None and cur > 0:
                stuck += 1
            terminated, reason = option_lib.option_terminated(opt, prev, step.state, event, 0,
                                                              step_i + 1, runtime)
            if patience and best is not None and stuck >= patience and not (done or terminated):
                reason = "blocked_no_progress"
                break
            if done or terminated:
                break
        reasons[reason] += 1
        if done:
            break
    return completed, dict(reasons), dict(kind_attempt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--partners", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-option-steps", type=int, default=16)
    ap.add_argument("--max-options", type=int, default=60)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--strict-preconditions", action="store_true")
    ap.add_argument("--dynamic-budget", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config.setdefault("env", {})["force_path_planning"] = True
    config.setdefault("options", {})["max_option_steps"] = int(args.max_option_steps)

    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
    results = {}
    print(f"=== RC-2b layout reachability scan ({len(layouts)} layouts) ===")
    print(f"{'layout':28s} {'admit':>6} {'best_compl':>10} {'#options':>9}  top_reasons")
    for lay in layouts:
        try:
            env = E._build_env(lay, config)
            layout_graph = parse_layout(env, lay)
            env.set_featurizer(NumpyFeaturizer(layout_graph))
            option_lib = OCV2OptionLibrary(
                layout_graph, max_option_steps=int(args.max_option_steps),
                strict_preconditions=args.strict_preconditions, dynamic_budget=args.dynamic_budget,
            )
            spd = layout_graph.shortest_path_dist
            partners = make_training_partners(option_lib)[: args.partners]
            n_pass = layout_graph.passable_count if hasattr(layout_graph, "passable_count") else None
            best_compl = 0.0
            agg_reasons = defaultdict(int)
            for partner in partners:
                comps = []
                for ep in range(args.episodes):
                    done_compl, reasons, _ = run_episode(env, option_lib, partner, ep,
                                                         args.max_options, args.patience, spd)
                    comps.append(1.0 if done_compl else 0.0)
                    for r, c in reasons.items():
                        agg_reasons[r] += c
                best_compl = max(best_compl, sum(comps) / len(comps))
            admit = best_compl > 0.0
            top = sorted(agg_reasons.items(), key=lambda kv: -kv[1])[:3]
            results[lay] = {"admit": admit, "best_completion": best_compl,
                            "n_options": len(option_lib.options), "reasons": dict(agg_reasons)}
            print(f"{lay:28s} {str(admit):>6} {best_compl:>10.2f} {len(option_lib.options):>9}  {top}")
        except Exception as ex:
            results[lay] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"{lay:28s} ERROR {type(ex).__name__}: {ex}")

    admitted = [k for k, v in results.items() if v.get("admit")]
    print(f"\nADMITTED (option-level reachable): {admitted}")
    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"admitted": admitted, "results": results}, indent=2, default=str),
                   encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
