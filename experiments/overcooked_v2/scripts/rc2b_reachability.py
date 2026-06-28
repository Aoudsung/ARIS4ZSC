"""RC-2b F4: state-aware reachability oracle.

Drives the full pipeline with a stage/inventory FSM (not a static kind-priority, which confounded
the earlier RC-2 probe) and executor patience (F2: blocked options yield instead of thrashing), to
answer the binding M0 question: is the task COMPLETABLE with a partner present?

FSM (per decision, using env preconditions to sequence fill -> cook -> plate -> serve):
    holding plated cooked soup -> serve_soup
    holding plate             -> plate_soup (else wait for the pot to cook)
    holding ingredient        -> deliver_ingredient_to_pot
    empty-handed              -> fetch_ingredient while a pot still needs one, else pick_plate

VERDICT:
  REACHABLE      -> some partner completes  => task is solvable; learning problem is real.
  NOT REACHABLE  -> no partner ever completes even with optimal patient play => the (layout, partner)
                    is pathological (single-stand-cell chokepoint permanently camped) -> reject it
                    in the reachability preflight gate, or add yield options / pick a better layout.

Usage:
    python experiments/overcooked_v2/scripts/rc2b_reachability.py \
        --checkpoint results/ocv2_rc1_bounded/cramped_room/aris_bellman/full_support/seed0 \
        --episodes 10 --block-patience 3 --max-option-steps 16 --force-path-planning true \
        --out results/ocv2_rc1_bounded/rc2b_reachability_oracle.json
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
    """State-aware oracle. Key fix (RC-2b): FINISH the current soup (plate->serve) before
    pre-fetching the next ingredient, and DROP an ingredient that can no longer be delivered
    (pot full/cooking) -- otherwise the agent occupies its hands with a 4th ingredient and can
    never pick_plate (which needs empty hands), so the cooked soup is never served."""
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
            return pick("plate_soup")  # else None -> wait for the pot to finish cooking
        if is_ingredient(inv):
            d = pick("deliver_ingredient_to_pot")
            return d if d is not None else pick("drop_item_to_counter")  # recover: free hands
        # empty-handed: finish the current soup before pre-fetching the next ingredient.
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
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-episode-options", type=int, default=60)
    ap.add_argument("--max-option-steps", type=int, default=16)
    ap.add_argument("--force-path-planning", choices=("true", "false"), default="true")
    ap.add_argument("--block-patience", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = (REPO_ROOT / args.checkpoint).resolve()
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoint.pt"
    ctx = E._load_context(ckpt, VARIANT)
    ctx.config.setdefault("env", {})["force_path_planning"] = (args.force_path_planning == "true")
    ctx.config.setdefault("options", {})["max_option_steps"] = int(args.max_option_steps)
    ctx.config["options"]["block_patience"] = int(args.block_patience)
    ctx.option_lib = OCV2OptionLibrary(ctx.layout_graph, max_option_steps=int(args.max_option_steps))
    ctx.scripted_fsm = make_fsm(ctx.graph)

    print(f"=== RC-2b reachability oracle (state-aware FSM) ===")
    print(f"checkpoint={ckpt}")
    print(f"overrides: max_option_steps={args.max_option_steps} fpp={args.force_path_planning} "
          f"block_patience={args.block_patience} episodes={args.episodes}\n")

    partners = E._resolve_partner_names(ctx.option_lib, "all")
    per_partner = {}
    any_completion = any_serve = any_ego = False
    for partner in partners:
        try:
            agg, _ = E._evaluate_partner(
                ctx, partner, episodes=int(args.episodes), seed=0,
                max_episode_options=int(args.max_episode_options),
                graph_override=ctx.graph, random_policy=False,
                collect_diagnostics=False, allow_diag_skip=True,
            )
        except Exception as ex:
            per_partner[partner] = {"error": f"{type(ex).__name__}: {ex}"}
            print(f"  {partner:24s} ERROR {type(ex).__name__}: {ex}")
            continue
        oks = agg.get("option_kind_stats", {})
        def ok(k):
            s = oks.get(k, {})
            return f"{s.get('success_count', 0)}/{s.get('attempt_count', 0)}"
        rec = {
            "completion_rate": agg.get("completion_rate"),
            "ego_delivery": agg.get("ego_delivery_count"),
            "partner_delivery": agg.get("partner_delivery_count"),
            "serve": ok("serve_soup"), "plate": ok("plate_soup"),
            "pick_plate": ok("pick_plate"), "deliver": ok("deliver_ingredient_to_pot"),
            "termination_counts": agg.get("termination_counts", {}),
        }
        per_partner[partner] = rec
        any_completion = any_completion or (rec["completion_rate"] or 0) > 0
        any_serve = any_serve or int(oks.get("serve_soup", {}).get("success_count", 0)) > 0
        any_ego = any_ego or (rec["ego_delivery"] or 0) > 0
        print(f"  {partner:24s} compl={rec['completion_rate']} ego_deliv={rec['ego_delivery']} "
              f"serve={rec['serve']} plate={rec['plate']} pick={rec['pick_plate']} deliver={rec['deliver']} "
              f"blocked={rec['termination_counts'].get('blocked_no_progress', 0)}")

    reachable = any_completion or any_serve or any_ego
    verdict = ("REACHABLE: optimal patient play completes the pipeline => task is solvable, "
               "learning problem is real."
               if reachable else
               "NOT REACHABLE: no partner completes even with patient optimal play => pathological "
               "(layout, partner) -> reject in reachability gate / add yield options / change layout.")
    print(f"\nVERDICT: {verdict}")
    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reachable": reachable, "verdict": verdict,
                               "per_partner": per_partner}, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
