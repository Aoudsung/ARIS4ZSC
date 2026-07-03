"""R2.1 throughput-lens admissibility certificate (METHOD_LOCK sec18.9).

Sibling of rc2b_substrate_certificate.py. Uses THROUGHPUT (ego_deliv + partner_deliv
per episode), not completion, as the lens — because sec9 already established that
completion saturates on OvercookedV2 and throughput is the discriminative metric.

For each (layout, partner) computes over N seeds x E episodes:
  fsm_throughput / rand_throughput / ponly_throughput  (mean total deliveries/ep)
  ego_serve_share  (ego_deliv / (ego_deliv+partner_deliv))
  fsm_ttfs         (time to first serve under fsm, +inf if none)

A cell is ADMITTED iff:
  fsm_tp >= --fsm-tp-min  AND  rand_tp <= --rand-tp-max  AND
  ponly_tp <= --ponly-tp-max  AND  (fsm_tp - rand_tp) >= --gap-min

FSM-ego probe hygiene (sec18.9.5): averages over --seeds (default 3) so
single-seed geometry-lock artifacts (like R2.1's ingredient-far fsm=0.0 vs
rand=1.0) do not flip the verdict.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_throughput_certificate.py \\
        --config experiments/overcooked_v2/configs/ocv2_step4_asymm_role_v1.yaml \\
        --layouts asymm_advantages,cramped_room,forced_coord,coord_ring \\
        --held-out-partners heldout-handoff-alternate-yield,heldout-resource-server-claim \\
        --seeds 3 --episodes 5 --max-options 80 \\
        --out results/ocv2_asymm/throughput_certificate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from rc2b_layout_scan import fsm, _noop_id  # noqa: E402
from experiments.overcooked_v2.train_aris import _build_env, _build_option_lib  # noqa: E402
from experiments.overcooked_v2.layout_parser import parse_layout  # noqa: E402
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer  # noqa: E402
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402
from experiments.overcooked_v2.event_extractor import extract_event  # noqa: E402
from experiments.overcooked_v2.option_termination import OptionRuntime  # noqa: E402
from experiments.overcooked_v2.state_utils import get_agent_pos  # noqa: E402
from experiments.overcooked_v2.provenance import option_library_hash, sha256_json  # noqa: E402


def _episode_stats(env, option_lib, partner, seed, max_options, spd, policy):
    """One episode; return (ego_deliv, partner_deliv, ttfs_option_step)."""
    obs, _ = env.reset(seed)
    partner.reset(seed)
    rng = np.random.default_rng(seed)
    noop_id = _noop_id(option_lib)
    ego = par = 0
    ttfs = math.inf
    option_step = 0
    for _ in range(max_options):
        state = env.state
        valid = option_lib.valid_options(state, 0)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0 or policy == "partner_only":
            oid = noop_id
        elif policy == "random":
            oid = int(rng.choice(valid_ids))
        else:
            oid = fsm(option_lib, state, valid_ids) or noop_id
        opt = option_lib.options[int(oid)]
        runtime = OptionRuntime(option_id=int(oid), start_pos=get_agent_pos(state, 0))
        budget = option_lib.option_budget(state, 0, int(oid))
        option_step += 1
        for step_i in range(budget):
            ego_a = option_lib.primitive_action(env.state, 0, int(oid))
            pa = partner.act(obs.get("agent_1"), env.state, rng)
            prev = env.state
            step = env.step(ego_a, pa.primitive_action)
            event = extract_event(
                prev, ego_a, pa.primitive_action, step.state, step.info,
                partner_option=None, partner_option_dist=None,
                partner_option_source="diagnostic_behavior_only",
            )
            obs = step.obs
            if getattr(event, "ego_delivery_event", False):
                ego += 1
                if ttfs == math.inf:
                    ttfs = option_step
            if getattr(event, "partner_delivery_event", False):
                par += 1
                if ttfs == math.inf:
                    ttfs = option_step
            done = bool(step.dones.get("__all__", False))
            terminated, _ = option_lib.option_terminated(opt, prev, step.state, event, 0, step_i + 1, runtime)
            if done or terminated:
                break
        if bool(step.dones.get("__all__", False)):
            break
    return ego, par, ttfs


def _agg(env, option_lib, partner, seed0, seeds, episodes, max_options, spd, policy):
    ego_totals = []
    par_totals = []
    ttfs_vals = []
    for s in range(seeds):
        for ep in range(episodes):
            e, p, t = _episode_stats(env, option_lib, partner, seed0 + 1000 * s + ep,
                                     max_options, spd, policy)
            ego_totals.append(e)
            par_totals.append(p)
            ttfs_vals.append(t)
    total_tp = [e + p for e, p in zip(ego_totals, par_totals)]
    denoms = [e + p for e, p in zip(ego_totals, par_totals)]
    ego_shares = [e / d for e, d in zip(ego_totals, denoms) if d > 0]
    finite_ttfs = [t for t in ttfs_vals if math.isfinite(t)]
    return {
        "throughput_mean": float(np.mean(total_tp)) if total_tp else 0.0,
        "throughput_std": float(np.std(total_tp)) if total_tp else 0.0,
        "ego_serve_share": float(np.mean(ego_shares)) if ego_shares else 0.0,
        "ttfs_mean": float(np.mean(finite_ttfs)) if finite_ttfs else float("inf"),
        "n_samples": len(total_tp),
        "n_serve_episodes": int(sum(1 for d in denoms if d > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layouts", default="")
    ap.add_argument("--held-out-partners", default="")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-options", type=int, default=80)
    ap.add_argument("--fsm-tp-min", type=float, default=0.8)
    ap.add_argument("--rand-tp-max", type=float, default=0.4)
    ap.add_argument("--ponly-tp-max", type=float, default=0.4)
    ap.add_argument("--gap-min", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(
        (cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).read_text(encoding="utf-8")
    )
    env_cfg = config.get("env", {}) or {}
    opt_cfg = config.get("options", {}) or {}
    partner_set = str((config.get("training", {}) or {}).get("partner_set", "standard7"))
    held_out = [s.strip() for s in args.held_out_partners.split(",") if s.strip()]
    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()] or [str(config["layout"])]
    executor_flags = {
        "force_path_planning": bool(env_cfg.get("force_path_planning", False)),
        "strict_preconditions": bool(opt_cfg.get("strict_preconditions", False)),
        "dynamic_budget": bool(opt_cfg.get("dynamic_budget", False)),
        "max_option_steps": int(opt_cfg.get("max_option_steps", 6)),
    }

    print(f"=== R2.1 throughput-lens certificate (sec18.9) | {len(layouts)} layouts x {partner_set} ===")
    print(f"seeds={args.seeds} episodes={args.episodes} max-opt={args.max_options}")
    print(f"thresholds: fsm_tp>={args.fsm_tp_min} rand_tp<={args.rand_tp_max} "
          f"ponly_tp<={args.ponly_tp_max} gap>={args.gap_min}")
    hdr = f"{'layout/partner':46s} {'held':>5} {'fsm_tp':>7} {'rand_tp':>8} {'ponly_tp':>9} {'gap':>5} {'admit':>6}"
    print(hdr)

    per_layout: dict[str, dict] = {}
    admitted = []
    option_kinds: list[str] = []
    for lay in layouts:
        try:
            env = _build_env(lay, config)
            lg = parse_layout(env, lay)
            env.set_featurizer(NumpyFeaturizer(lg))
            ol = _build_option_lib(lg, config)
            spd = lg.shortest_path_dist
            partners = make_training_partners(ol, partner_set=partner_set)
            if not option_kinds:
                option_kinds = sorted({o.kind for o in ol.options})
            rows = {}
            for p in partners:
                is_held = p.name in held_out
                fsm_r = _agg(env, ol, p, 100, args.seeds, args.episodes, args.max_options, spd, "fsm")
                rand_r = _agg(env, ol, p, 1000, args.seeds, args.episodes, args.max_options, spd, "random")
                pon_r = _agg(env, ol, p, 2000, args.seeds, args.episodes, args.max_options, spd, "partner_only")
                fsm_tp, rand_tp, pon_tp = fsm_r["throughput_mean"], rand_r["throughput_mean"], pon_r["throughput_mean"]
                gap = fsm_tp - rand_tp
                admit = bool(
                    fsm_tp >= args.fsm_tp_min
                    and rand_tp <= args.rand_tp_max
                    and pon_tp <= args.ponly_tp_max
                    and gap >= args.gap_min
                )
                rows[p.name] = {
                    "held_out": is_held,
                    "fsm": fsm_r, "random": rand_r, "partner_only": pon_r,
                    "gap_fsm_minus_rand": gap,
                    "admit_throughput": admit,
                    "admit_as_held_out": bool(admit and is_held),
                }
                if admit and is_held:
                    admitted.append({"layout": lay, "partner": p.name})
                print(f"{lay+'/'+p.name:46s} {str(is_held):>5} {fsm_tp:>7.2f} {rand_tp:>8.2f} "
                      f"{pon_tp:>9.2f} {gap:>5.2f} {str(admit):>6}")
            per_layout[lay] = {
                "n_options": ol.num_options,
                "option_library_hash": option_library_hash(ol),
                "partners": rows,
            }
        except Exception as ex:  # noqa: BLE001
            per_layout[lay] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"{lay:46s} ERROR {type(ex).__name__}: {ex}")

    executor_semantics_hash = sha256_json({**executor_flags, "option_kinds": option_kinds})
    status = "PASS" if admitted else "FAIL_STRUCTURAL"
    cert = {
        "probe": "throughput_certificate_v1",
        "sec_ref": "METHOD_LOCK sec18.9",
        "status": status,
        "partner_set": partner_set,
        "thresholds": {
            "fsm_tp_min": args.fsm_tp_min,
            "rand_tp_max": args.rand_tp_max,
            "ponly_tp_max": args.ponly_tp_max,
            "gap_min": args.gap_min,
        },
        "config": {"seeds": args.seeds, "episodes": args.episodes, "max_options": args.max_options,
                   "executor_flags": executor_flags},
        "per_layout": per_layout,
        "admitted_held_out": admitted,
        "executor_semantics_hash": executor_semantics_hash,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSTATUS={status} | admitted held-out cells: {admitted}")
    print(f"executor_semantics_hash={executor_semantics_hash}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
