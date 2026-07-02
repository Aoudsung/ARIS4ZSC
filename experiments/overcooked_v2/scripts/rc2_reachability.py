"""RC-2: completion-reachability probe.

Drives the ego agent with a deliberate kind-priority pipeline (serve > plate > pick_plate
> deliver > fetch > wait > ... > noop) that bypasses the learned Q, to distinguish:

  (a) counter / termination bug  -- scripted optimal play STILL never registers
      serve_soup success / ego_delivery / completion  =>  fix option_termination /
      event_extractor / attribution.
  (b) learning / value failure   -- scripted play DOES complete (counters fire)  =>
      completion=0 under the learned policy is because serve options never out-rank noop
      (same root as RC-1), not a measurement bug.

The checkpoint only supplies env / option library / graph; selection is fully scripted, so
the trained weights do not affect the rollout.

Usage:
    python experiments/overcooked_v2/scripts/rc2_reachability.py \
        --checkpoint results/ocv2_rc1/cramped_room/aris_bellman/full_support/seed0 \
        --episodes 5 --out results/ocv2_rc1/rc2_reachability.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.options import OCV2OptionLibrary  # noqa: E402

VARIANT = "full_support"
KNOWN_ORDER = [
    "serve_soup", "plate_soup", "pick_plate", "deliver_ingredient_to_pot",
    "fetch_ingredient", "wait_duration_after_arrival", "wait_duration",
    "wait_at_bottleneck", "cross_bottleneck", "handoff_counter",
    "press_recipe_button", "clear_interaction_cell", "drop_item_to_counter",
]


def _build_priority(ctx) -> list[str]:
    present = list(dict.fromkeys(str(o.kind) for o in ctx.graph.options))
    ordered = [k for k in KNOWN_ORDER if k in present]
    extras = [k for k in present if k not in KNOWN_ORDER and k != "noop"]
    tail = ["noop"] if "noop" in present else []
    return ordered + extras + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="seed0 dir or a checkpoint .pt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-episode-options", type=int, default=40)
    ap.add_argument("--max-option-steps", type=int, default=None,
                    help="RC-2b executor test: override per-option primitive-step budget.")
    ap.add_argument("--force-path-planning", choices=("true", "false"), default=None,
                    help="RC-2b executor test: override env force_path_planning.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = (REPO_ROOT / args.checkpoint).resolve()
    if ckpt.is_dir():
        for name in ("checkpoint.pt", "checkpoint_final.pt"):
            if (ckpt / name).exists():
                ckpt = ckpt / name
                break
    ctx = E._load_context(ckpt, VARIANT)
    # RC-2b executor overrides (diagnostic): test whether step budget / path planning is the wall.
    if args.force_path_planning is not None:
        ctx.config.setdefault("env", {})["force_path_planning"] = (args.force_path_planning == "true")
    if args.max_option_steps:
        ctx.config["options"]["max_option_steps"] = int(args.max_option_steps)
        ctx.option_lib = OCV2OptionLibrary(ctx.layout_graph, max_option_steps=int(args.max_option_steps))
    priority = _build_priority(ctx)
    print(f"=== RC-2 reachability probe (scripted pipeline) ===")
    print(f"checkpoint: {ckpt}")
    print(f"overrides: max_option_steps={ctx.config['options']['max_option_steps']} "
          f"force_path_planning={ctx.config.get('env', {}).get('force_path_planning')}")
    print(f"priority: {priority}\n")

    partners = E._resolve_partner_names(ctx.option_lib, "all")
    per_partner = {}
    any_serve = any_ego = any_completion = False
    for partner in partners:
        ctx.scripted_priority = priority  # active for this rollout
        ctx.qaudit = None
        try:
            agg, _rows = E._evaluate_partner(
                ctx, partner,
                episodes=int(args.episodes), seed=0,
                max_episode_options=int(args.max_episode_options),
                graph_override=ctx.graph, random_policy=False,
                collect_diagnostics=False, allow_diag_skip=True,
            )
        except Exception as ex:
            per_partner[partner] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"  {partner:28s} ERROR {type(ex).__name__}: {ex}")
            continue
        oks = agg.get("option_kind_stats", {})
        serve = oks.get("serve_soup", {})
        plate = oks.get("plate_soup", {})
        rec = {
            "completion_rate": agg.get("completion_rate"),
            "ego_delivery_count": agg.get("ego_delivery_count"),
            "partner_delivery_count": agg.get("partner_delivery_count"),
            "serve_soup_attempt": int(serve.get("attempt_count", 0)),
            "serve_soup_success": int(serve.get("success_count", 0)),
            "plate_soup_attempt": int(plate.get("attempt_count", 0)),
            "plate_soup_success": int(plate.get("success_count", 0)),
            "termination_counts": agg.get("termination_counts", {}),
            "option_kind_stats": {k: oks[k] for k in sorted(oks)},
        }
        per_partner[partner] = rec
        any_serve = any_serve or rec["serve_soup_success"] > 0
        any_ego = any_ego or (rec["ego_delivery_count"] or 0) > 0
        any_completion = any_completion or (rec["completion_rate"] or 0) > 0
        print(f"  {partner:28s} completion={rec['completion_rate']} "
              f"ego_deliv={rec['ego_delivery_count']} part_deliv={rec['partner_delivery_count']} "
              f"serve(att/ok)={rec['serve_soup_attempt']}/{rec['serve_soup_success']} "
              f"plate(att/ok)={rec['plate_soup_attempt']}/{rec['plate_soup_success']}")

    if any_serve or any_ego or any_completion:
        verdict = ("REACHABLE: scripted optimal play registers serve/ego-delivery/completion "
                   "=> counters & termination WORK; completion=0 under the learned policy is a "
                   "value/learning failure (serve options never out-rank noop), not a bug.")
    else:
        verdict = ("NOT REACHABLE under scripted optimal play: serve_soup success / ego delivery "
                   "/ completion never fire => termination/precondition/attribution BUG "
                   "(inspect option_termination.serve_soup + event_extractor delivery attribution).")
    print(f"\nVERDICT: {verdict}")

    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(ckpt), "priority": priority,
        "any_serve_success": any_serve, "any_ego_delivery": any_ego,
        "any_completion": any_completion, "verdict": verdict,
        "per_partner": per_partner,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
