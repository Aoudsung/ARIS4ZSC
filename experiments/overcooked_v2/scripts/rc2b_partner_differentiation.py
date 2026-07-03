"""R2.1 partner-differentiation certificate (EXPERIMENT_CHAIN_PLAN §10.1, METHOD_LOCK sec18.4).

Question: are the scripted partners in a registry BEHAVIORALLY DISTINGUISHABLE along
value-critical factors on the target layout? If they collapse to one behavior (the
sec15/16 finding for standard partners on tight layouts), an ARIS-vs-baseline ZSC test
on them is degenerate and the substrate must be rejected (→ FCP/MEP line).

This is a SUBSTRATE probe: it drives a deterministic, state-based FSM ego (checkpoint-
free, reusing rc2b_layout_scan.fsm) against each partner and records the PARTNER's own
behavior — primitive-action histogram, macro-option-kind histogram, terminal role
(who delivers), and bottleneck-camping rate. Partner option labels are read ONLY for
this diagnostic and never enter extract_event (P1 boundary preserved).

Preregistered verdict (sec18.4): the registry is DIFFERENTIATED iff, across the probed
partners, at least two value-critical FACTORS each show >=2 well-separated modes:
  - serving axis   : ego_serve_rate  = ego_deliv / (ego_deliv + partner_deliv)
  - bottleneck axis: camping_rate    = fraction of partner option-steps in wait_at_bottleneck
A pair of partners is DISTINGUISHABLE if the L1 distance between their behavior
signatures exceeds --distinct-l1.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_partner_differentiation.py \
        --config experiments/overcooked_v2/configs/ocv2_step4_asymm_role_v1.yaml \
        --episodes 8 --max-options 40 --out results/ocv2_asymm/partner_differentiation.json
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from experiments.overcooked_v2.provenance import sha256_json  # noqa: E402


def _probe_partner(env, option_lib, partner, seed, max_options, patience, spd):
    """One FSM-ego episode; record the PARTNER's own behavior (diagnostic only)."""
    obs, _ = env.reset(seed)
    partner.reset(seed)
    rng = np.random.default_rng(seed)
    noop_id = _noop_id(option_lib)
    prim_actions: list[int] = []
    partner_option_kinds: list[str] = []
    ego_deliv = 0
    partner_deliv = 0
    completed = False
    for _ in range(max_options):
        state = env.state
        valid = option_lib.valid_options(state, 0)
        valid_ids = np.flatnonzero(valid)
        oid = noop_id if valid_ids.size == 0 else (fsm(option_lib, state, valid_ids) or noop_id)
        opt = option_lib.options[int(oid)]
        runtime = OptionRuntime(option_id=int(oid), start_pos=get_agent_pos(state, 0))
        budget = option_lib.option_budget(state, 0, int(oid))
        for step_i in range(budget):
            ego_a = option_lib.primitive_action(env.state, 0, int(oid))
            pa = partner.act(obs.get("agent_1"), env.state, rng)
            # DIAGNOSTIC ONLY: read the partner's own chosen macro-option to label its
            # behavior. This is NOT routed into extract_event (partner_option=None), so
            # the P1 oracle-free evidence boundary is preserved.
            cur_opt = getattr(partner, "current_option", None)
            if cur_opt is not None and 0 <= int(cur_opt) < len(option_lib.options):
                partner_option_kinds.append(str(option_lib.options[int(cur_opt)].kind))
            prim_actions.append(int(pa.primitive_action))
            prev = env.state
            step = env.step(ego_a, pa.primitive_action)
            event = extract_event(
                prev, ego_a, pa.primitive_action, step.state, step.info,
                partner_option=None, partner_option_dist=None,
                partner_option_source="diagnostic_behavior_only",
            )
            obs = step.obs
            if getattr(event, "delivery_event", False):
                completed = True
            if getattr(event, "ego_delivery_event", False):
                ego_deliv += 1
            if getattr(event, "partner_delivery_event", False):
                partner_deliv += 1
            done = bool(step.dones.get("__all__", False))
            terminated, _ = option_lib.option_terminated(opt, prev, step.state, event, 0, step_i + 1, runtime)
            if done or terminated:
                break
        if bool(step.dones.get("__all__", False)):
            break
    return {
        "prim_actions": prim_actions,
        "partner_option_kinds": partner_option_kinds,
        "ego_deliv": ego_deliv,
        "partner_deliv": partner_deliv,
        "completed": completed,
    }


def _signature(agg_prim: dict[int, int], agg_kind: dict[str, int], n_prim: int,
               all_kinds: list[str], n_prim_actions: int) -> np.ndarray:
    """Behavior signature = normalized primitive-action histogram ++ option-kind histogram."""
    prim = np.array([agg_prim.get(a, 0) for a in range(n_prim_actions)], dtype=np.float64)
    kind = np.array([agg_kind.get(k, 0) for k in all_kinds], dtype=np.float64)
    prim = prim / prim.sum() if prim.sum() > 0 else prim
    kind = kind / kind.sum() if kind.sum() > 0 else kind
    return np.concatenate([prim, kind])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layout", default="")
    ap.add_argument("--partners", default="", help="comma list; default = all in the config partner_set")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-options", type=int, default=40)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--n-prim-actions", type=int, default=6)
    ap.add_argument("--distinct-l1", type=float, default=0.30, help="pair distinguishable if L1(sig) > this")
    ap.add_argument("--serve-spread-min", type=float, default=0.50, help="serving-axis PASS: ego_serve_rate range >= this")
    ap.add_argument("--camp-spread-min", type=float, default=0.30, help="bottleneck-axis PASS: camping_rate range >= this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load((cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).read_text(encoding="utf-8"))
    layout = args.layout or str(config["layout"])
    partner_set = str((config.get("training", {}) or {}).get("partner_set", "standard7"))

    env = _build_env(layout, config)
    lg = parse_layout(env, layout)
    env.set_featurizer(NumpyFeaturizer(lg))
    option_lib = _build_option_lib(lg, config)
    spd = lg.shortest_path_dist
    all_partners = make_training_partners(option_lib, partner_set=partner_set)
    want = [s.strip() for s in args.partners.split(",") if s.strip()]
    partners = [p for p in all_partners if (not want or p.name in want)]
    all_kinds = sorted({o.kind for o in option_lib.options})

    print(f"=== R2.1 partner-differentiation | layout={layout} set={partner_set} "
          f"partners={len(partners)} episodes={args.episodes} ===")
    per_partner: dict[str, dict] = {}
    sigs: dict[str, np.ndarray] = {}
    for p in partners:
        agg_prim: dict[int, int] = {}
        agg_kind: dict[str, int] = {}
        ego_d = part_d = n_prim = camp = 0
        completions = 0
        for ep in range(args.episodes):
            r = _probe_partner(env, option_lib, p, 100 + ep, args.max_options, args.patience, spd)
            for a in r["prim_actions"]:
                agg_prim[a] = agg_prim.get(a, 0) + 1
                n_prim += 1
            for k in r["partner_option_kinds"]:
                agg_kind[k] = agg_kind.get(k, 0) + 1
                if k == "wait_at_bottleneck":
                    camp += 1
            ego_d += r["ego_deliv"]
            part_d += r["partner_deliv"]
            completions += int(r["completed"])
        total_serves = ego_d + part_d
        n_kind = sum(agg_kind.values())
        row = {
            "protocol": {
                "role": getattr(p.protocol, "role", None),
                "bottleneck_policy": getattr(p.protocol, "bottleneck_policy", None),
                "terminal_policy": getattr(p.protocol, "terminal_policy", None),
            },
            "ego_deliv": ego_d, "partner_deliv": part_d,
            "ego_serve_rate": float(ego_d / total_serves) if total_serves else 0.0,
            "camping_rate": float(camp / n_kind) if n_kind else 0.0,
            "completion_rate": float(completions / args.episodes),
            "n_partner_option_steps": n_kind,
            "option_kind_hist": {k: agg_kind.get(k, 0) for k in all_kinds},
        }
        per_partner[p.name] = row
        sigs[p.name] = _signature(agg_prim, agg_kind, n_prim, all_kinds, args.n_prim_actions)
        print(f"  {p.name:34s} ego_serve={row['ego_serve_rate']:.2f} camp={row['camping_rate']:.2f} "
              f"compl={row['completion_rate']:.2f} bneck={row['protocol']['bottleneck_policy']} "
              f"term={row['protocol']['terminal_policy']}")

    names = list(per_partner)
    pairwise = {}
    n_distinct = 0
    n_pairs = 0
    for a, b in itertools.combinations(names, 2):
        l1 = float(np.abs(sigs[a] - sigs[b]).sum())
        pairwise[f"{a}|{b}"] = l1
        n_pairs += 1
        n_distinct += int(l1 > args.distinct_l1)

    serve_rates = [per_partner[n]["ego_serve_rate"] for n in names]
    camp_rates = [per_partner[n]["camping_rate"] for n in names]
    serve_spread = float(max(serve_rates) - min(serve_rates)) if serve_rates else 0.0
    camp_spread = float(max(camp_rates) - min(camp_rates)) if camp_rates else 0.0

    def _modes(vals, lo=0.34, hi=0.66):
        """>=2 well-separated modes if some vals are clearly low and some clearly high."""
        low = sum(1 for v in vals if v <= lo)
        high = sum(1 for v in vals if v >= hi)
        return low >= 1 and high >= 1, {"low": low, "high": high}

    serve_ok, serve_modes = _modes(serve_rates)
    camp_ok, camp_modes = _modes(camp_rates)
    serving_axis_pass = bool(serve_spread >= args.serve_spread_min and serve_ok)
    bottleneck_axis_pass = bool(camp_spread >= args.camp_spread_min and camp_ok)
    factors_passed = int(serving_axis_pass) + int(bottleneck_axis_pass)
    verdict = "PASS" if factors_passed >= 2 else ("PARTIAL" if factors_passed == 1 else "FAIL")

    cert = {
        "probe": "partner_differentiation_v1",
        "layout": layout,
        "partner_set": partner_set,
        "episodes": args.episodes,
        "thresholds": {
            "distinct_l1": args.distinct_l1,
            "serve_spread_min": args.serve_spread_min,
            "camp_spread_min": args.camp_spread_min,
        },
        "per_partner": per_partner,
        "pairwise_l1": pairwise,
        "distinguishable_pairs": f"{n_distinct}/{n_pairs}",
        "serving_axis": {"spread": serve_spread, "modes": serve_modes, "pass": serving_axis_pass},
        "bottleneck_axis": {"spread": camp_spread, "modes": camp_modes, "pass": bottleneck_axis_pass},
        "factors_passed": factors_passed,
        "verdict": verdict,
        "config_hash": sha256_json({"layout": layout, "partner_set": partner_set}),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nserving_axis pass={serving_axis_pass} (spread={serve_spread:.2f} modes={serve_modes})")
    print(f"bottleneck_axis pass={bottleneck_axis_pass} (spread={camp_spread:.2f} modes={camp_modes})")
    print(f"distinguishable pairs: {n_distinct}/{n_pairs}")
    print(f"VERDICT: {verdict}  (factors_passed={factors_passed}/2)")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
