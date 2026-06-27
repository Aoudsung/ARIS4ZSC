"""RC-1 q-stability gate (plan section 2.6).

Judges a training run's Q-stability from the per-checkpoint decomposition audit
(rc1_qaudit.json) + training metrics. Crucially it evaluates EVERY checkpoint, including
the FINAL one -- so greedy checkpoint-selection cannot hide a diverged final model
("TRAINING_DIVERGED_BEST_CHECKPOINT_MASKED").

Hard criteria (must all pass for overall PASS):
  H1 q_hard_bound       max |Q| over all checkpoints <= vmax * (1 + eps)
  H2 final_not_collapsed final greedy return > collapse_floor (not the ~-0.4 noop floor)
  H3 no_ranking_collapse final initial-state argmax != noop (init delta < 0) AND
                         noop_win_frac is not 1.0 on the last two checkpoints

Advisory (reported, do not fail the gate -- they flag remaining plan items):
  A1 noop_win_final <= noop_win_target   (else: needs section 2.4 wait-options)
  A2 final >= final_best_ratio * best     (else: greedy variance / mild degradation)

Usage:
    python experiments/overcooked_v2/scripts/rc_qstability_gate.py \
        --audit results/ocv2_rc1_bounded/rc1_qaudit.json \
        --metrics results/ocv2_rc1_bounded/rc1_bounded_train_metrics.json \
        --vmax 20 --out results/ocv2_rc1_bounded/q_stability_gate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--vmax", type=float, default=20.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--collapse-floor", type=float, default=1.0)
    ap.add_argument("--noop-win-target", type=float, default=0.2)
    ap.add_argument("--final-best-ratio", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    if not rows:
        raise SystemExit("audit has no rows")

    def q_abs(row):
        ini = row.get("initial") or {}
        return max(abs(ini.get("q_noop", 0.0)), abs(ini.get("q_best_other", 0.0)))

    max_abs_q = max(q_abs(r) for r in rows)
    final = rows[-1]
    fin_init = final.get("initial") or {}
    val_returns = [r.get("val_return") for r in rows if isinstance(r.get("val_return"), (int, float))]
    final_ret = final.get("val_return")
    best_ret = max(val_returns) if val_returns else None
    noop_win = [r.get("noop_win_frac") for r in rows]
    last2_noop = [w for w in noop_win[-2:] if isinstance(w, (int, float))]

    # Hard criteria
    H1 = max_abs_q <= args.vmax * (1.0 + args.eps)
    H2 = isinstance(final_ret, (int, float)) and final_ret > args.collapse_floor
    H3 = (
        isinstance(fin_init.get("delta"), (int, float)) and fin_init["delta"] < 0.0
        and not fin_init.get("argmax_is_noop", False)
        and not any(abs(w - 1.0) < 1e-9 for w in last2_noop)
    )
    hard = {"H1_q_hard_bound": bool(H1), "H2_final_not_collapsed": bool(H2),
            "H3_no_ranking_collapse": bool(H3)}
    overall = all(hard.values())

    # Advisory
    noop_win_final = noop_win[-1] if isinstance(noop_win[-1], (int, float)) else None
    A1 = (noop_win_final is not None) and (noop_win_final <= args.noop_win_target)
    A2 = (
        isinstance(final_ret, (int, float)) and isinstance(best_ret, (int, float)) and best_ret > 0
        and final_ret >= args.final_best_ratio * best_ret
    )
    advisory = {
        "A1_noop_win_final_le_target": {"pass": bool(A1), "value": noop_win_final,
                                        "target": args.noop_win_target,
                                        "note": "" if A1 else "needs section 2.4 (noop -> task-valid wait options)"},
        "A2_final_ge_ratio_best": {"pass": bool(A2), "final": final_ret, "best": best_ret,
                                   "note": "" if A2 else "final below best*ratio (greedy 3-ep variance / mild degradation)"},
    }

    status = "PASS" if overall else "TRAINING_DIVERGED_BEST_CHECKPOINT_MASKED"
    gate = {
        "status": status,
        "overall_hard_pass": overall,
        "vmax": args.vmax,
        "max_abs_q_bounded": max_abs_q,
        "final_update": final.get("update"),
        "final_greedy_return": final_ret,
        "best_greedy_return": best_ret,
        "final_init_delta": fin_init.get("delta"),
        "final_argmax_kind": fin_init.get("argmax_kind"),
        "noop_win_series": noop_win,
        "hard_criteria": hard,
        "advisory_criteria": advisory,
        "raw_base_growth": "not_audited (re-run rc1_qaudit after adding raw_base capture)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print(f"=== RC-1 q-stability gate: {status} ===")
    print(f"  max|Q|={max_abs_q:.3f} (Vmax={args.vmax})  "
          f"final@{final.get('update')}={final_ret}  best={best_ret}  "
          f"final_init_delta={fin_init.get('delta')}  final_argmax={fin_init.get('argmax_kind')}")
    for k, v in hard.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    for k, v in advisory.items():
        print(f"  [{'ok ' if v['pass'] else 'warn'}] {k}: {v.get('note') or 'ok'}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
