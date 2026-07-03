"""RC-2b Phase A: SubstrateAdmissibilityCertificate for the held-out ZSC condition.

Unlike the layout scan (which maxed over partners), this measures PER-PARTNER fsm-oracle vs random vs
partner-only completion, so it can certify the *held-out* claim condition:

    a (layout, held-out partner) pair is ADMITTED iff
        fsm_completion   >= --oracle-min   (a skilled ego CAN complete)   AND
        random_completion <= --random-max  (a random ego FAILS -> skill matters)  AND
        ponly_completion  <= --ponly-max   (the partner CANNOT solo -> ego is necessary)

It stamps the executor_semantics_hash (fpp + strict + dynamic + max_option_steps + option-kind set) so
downstream CE/train/eval artifacts can be checked to share the exact substrate this certificate covers.

Reuses the layout-scan rollout machinery (run_episode/fsm) and the corrected executor factories
(_build_env / _build_option_lib) so the certificate runs on the SAME substrate train/eval will use.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_substrate_certificate.py \
        --config experiments/overcooked_v2/configs/ocv2_step4_asymm.yaml \
        --layouts asymm_advantages,asymm_advantages_recipes_center,asymm_advantages_recipes_left,asymm_advantages_recipes_right \
        --held-out-partners bottleneck-yield,flexible-balanced \
        --episodes 5 --out results/ocv2_asymm/substrate_certificate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import (scripts is not a package)

from rc2b_layout_scan import run_episode  # noqa: E402
from experiments.overcooked_v2.train_aris import _build_env, _build_option_lib  # noqa: E402
from experiments.overcooked_v2.layout_parser import parse_layout  # noqa: E402
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer  # noqa: E402
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402
from experiments.overcooked_v2.provenance import option_library_hash, sha256_json  # noqa: E402


def _completion(env, option_lib, partner, spd, episodes, max_options, patience, policy, seed0):
    done = []
    for ep in range(episodes):
        dc, _, _, _, _ = run_episode(env, option_lib, partner, seed0 + ep, max_options, patience, spd, policy=policy)
        done.append(1.0 if dc else 0.0)
    return sum(done) / len(done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layouts", default="")
    ap.add_argument("--held-out-partners", default="bottleneck-yield,flexible-balanced")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-options", type=int, default=60)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--oracle-min", type=float, default=0.8)
    ap.add_argument("--random-max", type=float, default=0.4)
    ap.add_argument("--ponly-max", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8")) \
        if not Path(args.config).is_absolute() else yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env_cfg = config.get("env", {}) or {}
    opt_cfg = config.get("options", {}) or {}
    held_out = [s.strip() for s in args.held_out_partners.split(",") if s.strip()]
    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()] or [str(config["layout"])]

    executor_flags = {
        "force_path_planning": bool(env_cfg.get("force_path_planning", False)),
        "strict_preconditions": bool(opt_cfg.get("strict_preconditions", False)),
        "dynamic_budget": bool(opt_cfg.get("dynamic_budget", False)),
        "max_option_steps": int(opt_cfg.get("max_option_steps", 6)),
    }

    per_layout: dict[str, dict] = {}
    admitted_conditions: list[dict] = []
    option_kinds: list[str] = []
    print(f"=== RC-2b SubstrateAdmissibilityCertificate ({len(layouts)} layouts) ===")
    print(f"held-out partners: {held_out} | thresholds oracle>={args.oracle_min} random<={args.random_max} ponly<={args.ponly_max}")
    print(f"{'layout/partner':46s} {'held':>5} {'fsm':>5} {'rand':>5} {'ponly':>6} {'admit':>6}")

    for lay in layouts:
        try:
            env = _build_env(lay, config)
            lg = parse_layout(env, lay)
            env.set_featurizer(NumpyFeaturizer(lg))
            ol = _build_option_lib(lg, config)
            spd = lg.shortest_path_dist
            partners = make_training_partners(
                ol, partner_set=str((config.get("training", {}) or {}).get("partner_set", "standard7"))
            )
            if not option_kinds:
                option_kinds = sorted({o.kind for o in ol.options})
            rows = {}
            for p in partners:
                is_held = p.name in held_out
                fsm_c = _completion(env, ol, p, spd, args.episodes, args.max_options, args.patience, "fsm", 100)
                rnd_c = _completion(env, ol, p, spd, args.episodes, args.max_options, args.patience, "random", 1000)
                po_c = _completion(env, ol, p, spd, args.episodes, args.max_options, args.patience, "partner_only", 2000)
                admit = bool(is_held and fsm_c >= args.oracle_min and rnd_c <= args.random_max and po_c <= args.ponly_max)
                rows[p.name] = {
                    "held_out": is_held, "fsm_completion": fsm_c, "random_completion": rnd_c,
                    "partner_only_completion": po_c, "ego_necessary": po_c <= args.ponly_max,
                    "hard_for_random": rnd_c <= args.random_max, "completable": fsm_c >= args.oracle_min,
                    "admit_held_out": admit,
                }
                if admit:
                    admitted_conditions.append({"layout": lay, "partner": p.name})
                print(f"{lay+'/'+p.name:46s} {str(is_held):>5} {fsm_c:>5.2f} {rnd_c:>5.2f} {po_c:>6.2f} {str(admit):>6}")
            per_layout[lay] = {
                "n_options": ol.num_options, "option_library_hash": option_library_hash(ol),
                "partners": rows,
            }
        except Exception as ex:  # noqa: BLE001
            per_layout[lay] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"{lay:46s} ERROR {type(ex).__name__}: {ex}")

    executor_semantics_hash = sha256_json({**executor_flags, "option_kinds": option_kinds})
    status = "PASS" if admitted_conditions else "FAIL_STRUCTURAL"
    cert = {
        "status": status,
        "executor_semantics_hash": executor_semantics_hash,
        "executor_flags": executor_flags,
        "option_kinds": option_kinds,
        "held_out_partners": held_out,
        "thresholds": {"oracle_min": args.oracle_min, "random_max": args.random_max, "ponly_max": args.ponly_max},
        "episodes": args.episodes,
        "admitted_conditions": admitted_conditions,
        "per_layout": per_layout,
    }
    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2, default=str), encoding="utf-8")
    print(f"\nSTATUS={status} | admitted held-out conditions: {[(c['layout'], c['partner']) for c in admitted_conditions]}")
    print(f"executor_semantics_hash={executor_semantics_hash}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
