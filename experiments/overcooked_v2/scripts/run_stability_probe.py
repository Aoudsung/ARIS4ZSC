"""P0-step1 stability probe: confirm the divergence/collapse root cause WITHOUT touching
train_aris.py (so it cannot collide with the P1 honesty-fix edits).

For aris_bellman/full_support, retrain at increasing update budgets, greedy-eval each, and
record: greedy mean_return, final td_loss + whether it decreased, and the greedy noop fraction.
If the deadly-triad/overtraining hypothesis holds, greedy return peaks early then collapses to
a noop policy while td_loss grows monotonically.

Usage (8-GPU host; CPU-bound fast eval runs single-job):
    python experiments/overcooked_v2/scripts/run_stability_probe.py --budgets 500,1000,2000,3000,5000 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = "experiments/overcooked_v2/configs/ocv2_step4.yaml"
PROBE_DIR = "results/ocv2_stability"
LAYOUT = "cramped_room"
METHOD = "aris_bellman"
VARIANT = "full_support"
SEED = 0


def _ckpt_dir(updates: int) -> Path:
    return (REPO_ROOT / PROBE_DIR / f"u{updates}" / LAYOUT / METHOD / VARIANT / f"seed{SEED}")


def _train(updates: int, gpu: int, timeout: int) -> dict[str, Any]:
    out_dir = REPO_ROOT / PROBE_DIR / f"u{updates}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    cmd = [sys.executable, str(REPO_ROOT / "experiments/overcooked_v2/train_aris.py"),
           "--config", str(REPO_ROOT / CONFIG), "--graph_variant", VARIANT,
           "--method", METHOD, "--seed", str(SEED), "--updates", str(updates),
           "--output_dir", str(out_dir)]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout)
    return {"rc": r.returncode, "elapsed": time.time() - t0,
            "stderr_tail": r.stderr[-400:] if r.returncode else "",
            "ckpt": (_ckpt_dir(updates) / "checkpoint.pt").exists()}


def _eval(updates: int, gpu: int, timeout: int) -> Path | None:
    out = REPO_ROOT / PROBE_DIR / f"eval_u{updates}.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    cmd = [sys.executable, str(REPO_ROOT / "experiments/overcooked_v2/evaluate_aris.py"),
           "--checkpoint", str(_ckpt_dir(updates)), "--graph_variants", VARIANT,
           "--partners", "all", "--episodes", "3", "--seed", str(SEED),
           "--output", str(out), "--fast"]
    if out.exists():
        out.unlink()
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout)
    return out if (r.returncode == 0 and out.exists()) else None


def _train_signals(updates: int) -> dict[str, Any]:
    m = _ckpt_dir(updates) / "metrics.json"
    if not m.exists():
        return {}
    d = json.loads(m.read_text(encoding="utf-8"))
    return {"td_first": d.get("td_loss_first_window"), "td_last": d.get("td_loss_last_window"),
            "td_decreased": d.get("td_loss_decreased"),
            "train_reward_mean": d.get("reward_mean"),
            "served_soup": d.get("served_soup_count")}


def _eval_signals(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"greedy_return": None, "noop_frac": None, "completion": None}
    d = json.loads(path.read_text(encoding="utf-8"))
    results = d.get("results", [])
    rets = [e.get("aggregate", {}).get("mean_return") for e in results
            if e.get("aggregate", {}).get("mean_return") is not None]
    greedy = float(sum(rets) / len(rets)) if rets else None
    # noop fraction across partners (option_kind_stats)
    noop_att = tot_att = 0
    comp = []
    for e in results:
        ks = e.get("aggregate", {}).get("option_kind_stats", {})
        for kind, st in ks.items():
            tot_att += int(st.get("attempt_count", 0))
            if kind == "noop":
                noop_att += int(st.get("attempt_count", 0))
        c = e.get("aggregate", {}).get("completion_rate")
        if c is not None:
            comp.append(float(c))
    return {"greedy_return": greedy,
            "noop_frac": (noop_att / tot_att) if tot_att else None,
            "completion": (sum(comp) / len(comp)) if comp else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="500,1000,2000,3000,5000")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-timeout", type=int, default=3600)
    ap.add_argument("--eval-timeout", type=int, default=7200)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    print(f"=== P0 stability probe: {METHOD}/{VARIANT} seed{SEED}, budgets={budgets}, gpu={args.gpu} ===")
    rows = []
    for u in budgets:
        print(f"\n--- updates={u} ---", flush=True)
        tr = _train(u, args.gpu, args.train_timeout)
        print(f"  train rc={tr['rc']} ckpt={tr['ckpt']} ({tr['elapsed']:.0f}s)", flush=True)
        if tr["rc"] != 0:
            print(f"  stderr: {tr['stderr_tail']}", flush=True)
        ev = _eval(u, args.gpu, args.eval_timeout) if tr["ckpt"] else None
        row = {"updates": u, **_train_signals(u), **_eval_signals(ev)}
        rows.append(row)
        print(f"  greedy_return={row.get('greedy_return')} noop_frac={row.get('noop_frac')} "
              f"completion={row.get('completion')} td={row.get('td_first')}→{row.get('td_last')} "
              f"dec={row.get('td_decreased')}", flush=True)

    out = REPO_ROOT / PROBE_DIR / "stability_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"method": METHOD, "variant": VARIANT, "seed": SEED, "rows": rows},
                              indent=2, default=str), encoding="utf-8")
    print("\n=== SUMMARY (updates → greedy_return / noop_frac / td_last / dec) ===")
    for r in rows:
        gr = f"{r['greedy_return']:.3f}" if r.get("greedy_return") is not None else "NA"
        nf = f"{r['noop_frac']:.2f}" if r.get("noop_frac") is not None else "NA"
        print(f"  {r['updates']:5d} → return={gr:>8}  noop={nf:>5}  td_last={r.get('td_last')}  dec={r.get('td_decreased')}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
