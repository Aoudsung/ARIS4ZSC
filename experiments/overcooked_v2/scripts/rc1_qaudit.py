"""RC-1: per-checkpoint Q-ranking / advantage decomposition audit.

Confirms the collapse mechanism behind the noop attractor. For each saved
checkpoint_u<N>.pt (one training run with --save_all_checkpoints), runs a greedy rollout
with the read-only qaudit hook and records, per decision, the (q_total, q_base)
decomposition. The decisive signal is the **initial-state** decision (deterministic
env.reset, empty evidence -> identical state across all checkpoints):

    delta = Q(noop) - max_{valid option != noop} Q(option)

delta crossing 0 (noop out-ranks every productive option) marks the ranking collapse.
We overlay the per-update td_loss series + greedy validation return so the decisive
question can be answered: does delta cross 0 BEFORE td_loss explodes (=> ranking collapse
is primary, loss-scale is downstream) or only together with it?

adv_sum = q_total - q_base isolates whether the base head or the summed factor-advantage
drives noop dominance.

Usage:
    python experiments/overcooked_v2/scripts/rc1_qaudit.py \
        --run-dir results/ocv2_rc1/cramped_room/aris_bellman/full_support/seed0 \
        --episodes 3 --out results/ocv2_rc1/rc1_qaudit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402

VARIANT = "full_support"


def _analyze(rec: dict[str, Any], noop_id: int, kinds: list[str]) -> dict[str, Any] | None:
    q = rec["q_full"]
    qb = rec["q_base"]
    valid = rec["valid"]
    n = len(q)
    valid_idx = [i for i in range(n) if i < len(valid) and valid[i]]
    if noop_id not in valid_idx:
        return None
    others = [i for i in valid_idx if i != noop_id]
    if not others:
        return None
    best_other = max(others, key=lambda i: q[i])
    argmax_valid = max(valid_idx, key=lambda i: q[i])

    def kind(i: int) -> str:
        return kinds[i] if i < len(kinds) else f"opt{i}"

    return {
        "delta": float(q[noop_id] - q[best_other]),
        "argmax_kind": kind(argmax_valid),
        "argmax_is_noop": bool(argmax_valid == noop_id),
        "q_noop": float(q[noop_id]),
        "q_best_other": float(q[best_other]),
        "best_other_kind": kind(best_other),
        "qbase_noop": float(qb[noop_id]),
        "qbase_best_other": float(qb[best_other]),
        "advsum_noop": float(q[noop_id] - qb[noop_id]),
        "advsum_best_other": float(q[best_other] - qb[best_other]),
        "n_valid": len(valid_idx),
    }


def _td_at(td_losses: list[float], update: int, log_interval: int, total: int) -> float | None:
    if not td_losses:
        return None
    # td_losses is logged every log_interval updates; map update -> nearest index.
    idx = min(len(td_losses) - 1, max(0, round(update / max(1, log_interval)) - 1))
    try:
        return float(td_losses[idx])
    except (IndexError, TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="seed0 dir with checkpoint_u<N>.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-episode-options", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    ckpts = sorted(
        run_dir.glob("checkpoint_u*.pt"),
        key=lambda p: int(re.search(r"checkpoint_u(\d+)\.pt", p.name).group(1)),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoint_u*.pt under {run_dir}")

    metrics = {}
    mpath = run_dir / "metrics.json"
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    td_losses = metrics.get("td_losses") or []
    log_interval = 100
    greedy_val = {int(v.get("update", -1)): v.get("mean_return") for v in metrics.get("greedy_validation", [])}

    rows = []
    print(f"=== RC-1 Q-decomposition audit: {len(ckpts)} checkpoints from {run_dir.name} ===")
    print(f"{'upd':>5} {'val_ret':>8} {'init_dlt':>9} {'argmax@init':>22} "
          f"{'qb_noop':>8} {'qb_opt':>8} {'adv_noop':>9} {'adv_opt':>9} {'noopwin':>7} {'td~':>10}")
    for ckpt in ckpts:
        update = int(re.search(r"checkpoint_u(\d+)\.pt", ckpt.name).group(1))
        ctx = E._load_context(ckpt, VARIANT)
        ctx.qaudit = []
        noop_id = E._noop_option_id(ctx.option_lib)
        kinds = [str(o.kind) for o in ctx.graph.options]
        partner = E._resolve_partner_names(ctx.option_lib, "all")[0]
        err = None
        try:
            E._evaluate_partner(
                ctx, partner,
                episodes=int(args.episodes), seed=0,
                max_episode_options=int(args.max_episode_options),
                graph_override=ctx.graph, random_policy=False,
                collect_diagnostics=False, allow_diag_skip=True,
            )
        except Exception as ex:  # integrity gate may raise post-rollout; qaudit already filled
            err = f"{type(ex).__name__}: {ex}"

        analyzed = [a for a in (_analyze(r, noop_id, kinds) for r in ctx.qaudit) if a is not None]
        initial = analyzed[0] if analyzed else None
        n = len(analyzed)
        mean_delta = sum(a["delta"] for a in analyzed) / n if n else None
        noop_win = sum(1 for a in analyzed if a["argmax_is_noop"]) / n if n else None
        td_approx = _td_at([float(x) for x in td_losses if isinstance(x, (int, float))], update, log_interval, len(td_losses))
        row = {
            "update": update,
            "val_return": greedy_val.get(update),
            "n_decisions": n,
            "initial": initial,
            "mean_delta": mean_delta,
            "noop_win_frac": noop_win,
            "td_approx": td_approx,
            "error": err,
        }
        rows.append(row)
        vr = f"{row['val_return']:.2f}" if isinstance(row['val_return'], (int, float)) else "NA"
        if initial:
            print(f"{update:>5} {vr:>8} {initial['delta']:>9.3f} {initial['argmax_kind']:>22} "
                  f"{initial['qbase_noop']:>8.2f} {initial['qbase_best_other']:>8.2f} "
                  f"{initial['advsum_noop']:>9.2f} {initial['advsum_best_other']:>9.2f} "
                  f"{(noop_win if noop_win is not None else 0):>7.2f} "
                  f"{(td_approx if td_approx is not None else float('nan')):>10.2g}")
        else:
            print(f"{update:>5} {vr:>8}  (no analyzable decision; err={err})")

    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_dir": str(run_dir), "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("Read: init_dlt>0 => noop out-ranks best productive option at the initial state "
          "(ranking collapse). Compare the update where init_dlt crosses 0 vs where td~ explodes.")


if __name__ == "__main__":
    main()
