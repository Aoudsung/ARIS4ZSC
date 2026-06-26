"""Step 4: Micro-train across (method × graph_variant × seed) and build evaluation matrix."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = "experiments/overcooked_v2/configs/ocv2_step4.yaml"
OUTPUT_DIR = "results/ocv2_step4"
LAYOUT = "cramped_room"

METHODS = (
    "aris_bellman",
    "base_only",
    "flat_factor",
    "global_gru",
    "partner_id_q",
    "random_policy",
)
VARIANTS = ("full_support", "minus_high_ce", "overcomplete", "shuffled_relevance")
SEEDS = (0, 1, 2)
GPUS = [0, 1, 2, 4, 5, 6, 7]

EVAL_EPISODES = 3


def _job_key(method: str, variant: str, seed: int) -> str:
    return f"{method}/{variant}/seed{seed}"


def _checkpoint_dir(method: str, variant: str, seed: int) -> Path:
    return REPO_ROOT / OUTPUT_DIR / LAYOUT / method / variant / f"seed{seed}"


def _run_train_job(gpu: int, method: str, variant: str, seed: int) -> dict[str, Any]:
    key = _job_key(method, variant, seed)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "overcooked_v2" / "train_aris.py"),
        "--config", str(REPO_ROOT / CONFIG),
        "--graph_variant", variant,
        "--method", method,
        "--seed", str(seed),
        "--output_dir", str(REPO_ROOT / OUTPUT_DIR),
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.time() - t0
    ckpt = _checkpoint_dir(method, variant, seed) / "checkpoint.pt"
    return {
        "key": key,
        "method": method,
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "checkpoint_exists": ckpt.exists(),
        "stderr_tail": result.stderr[-500:] if result.returncode != 0 else "",
    }


def run_training_phase() -> list[dict[str, Any]]:
    jobs = [
        (method, variant, seed)
        for method in METHODS
        for variant in VARIANTS
        for seed in SEEDS
    ]
    print(f"=== Phase 2: Training {len(jobs)} jobs on {len(GPUS)} GPUs ===")

    results: list[dict[str, Any]] = []
    gpu_cycle = 0

    with ProcessPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {}
        for method, variant, seed in jobs:
            ckpt = _checkpoint_dir(method, variant, seed) / "checkpoint.pt"
            if ckpt.exists():
                results.append({
                    "key": _job_key(method, variant, seed),
                    "method": method, "variant": variant, "seed": seed,
                    "gpu": -1, "returncode": 0, "elapsed": 0,
                    "checkpoint_exists": True, "stderr_tail": "",
                })
                print(
                    f"  [{len(results):2d}/{len(jobs)}] "
                    f"{_job_key(method, variant, seed):45s} CACHED"
                )
                continue
            gpu = GPUS[gpu_cycle % len(GPUS)]
            gpu_cycle += 1
            future = pool.submit(_run_train_job, gpu, method, variant, seed)
            futures[future] = _job_key(method, variant, seed)

        for future in as_completed(futures):
            info = future.result()
            results.append(info)
            status = "OK" if info["checkpoint_exists"] else "FAIL"
            print(
                f"  [{len(results):2d}/{len(jobs)}] {info['key']:45s} "
                f"GPU:{info['gpu']} {status} ({info['elapsed']:.0f}s)"
            )
            if info["returncode"] != 0:
                print(f"    stderr: {info['stderr_tail']}")

    ok = sum(1 for r in results if r["checkpoint_exists"])
    print(f"\nTraining complete: {ok}/{len(jobs)} checkpoints saved")
    return results


def _run_eval_job(method: str, seed: int) -> dict[str, Any]:
    ckpt_dir = _checkpoint_dir(method, "full_support", seed)
    out_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "overcooked_v2" / "evaluate_aris.py"),
        "--checkpoint", str(ckpt_dir),
        "--graph_variants", ",".join(VARIANTS),
        "--partners", "all",
        "--episodes", str(EVAL_EPISODES),
        "--seed", str(seed),
        "--output", str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=7200,
        )
        return {
            "method": method,
            "seed": seed,
            "returncode": result.returncode,
            "output_path": str(out_path),
            "output_exists": out_path.exists(),
            "stderr_tail": result.stderr[-500:] if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "method": method,
            "seed": seed,
            "returncode": -1,
            "output_path": str(out_path),
            "output_exists": out_path.exists(),
            "stderr_tail": "TIMEOUT after 7200s",
        }


def run_evaluation_phase() -> list[dict[str, Any]]:
    eval_jobs = [(m, s) for m in METHODS if m != "random_policy" for s in SEEDS]
    print(f"\n=== Phase 4: Evaluation ({len(eval_jobs)} runs) ===")

    results: list[dict[str, Any]] = []
    for method, seed in eval_jobs:
        ckpt = _checkpoint_dir(method, "full_support", seed) / "checkpoint.pt"
        if not ckpt.exists():
            print(f"  SKIP eval {method}/seed{seed} — no checkpoint")
            continue
        info = _run_eval_job(method, seed)
        results.append(info)
        status = "OK" if info["output_exists"] else "FAIL"
        print(f"  [{len(results):2d}/{len(eval_jobs)}] {method}/seed{seed} {status}")
        if info["returncode"] != 0:
            print(f"    stderr: {info['stderr_tail']}")

    return results


def _extract_random_baseline_from_eval(seed: int) -> float | None:
    for method in METHODS:
        if method == "random_policy":
            continue
        eval_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}.json"
        if not eval_path.exists():
            continue
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        baselines = data.get("reference_baselines", {})
        if not isinstance(baselines, dict):
            continue
        returns = [
            float(baseline["mean_return"])
            for baseline in baselines.values()
            if isinstance(baseline, dict) and baseline.get("mean_return") is not None
        ]
        if returns:
            return float(np.mean(returns))
    return None


def build_matrix() -> dict[str, Any]:
    print("\n=== Phase 5: Evaluation Matrix ===")
    matrix: dict[str, dict[str, list[float]]] = {
        method: {variant: [] for variant in VARIANTS}
        for method in METHODS
    }

    for method in METHODS:
        for seed in SEEDS:
            if method == "random_policy":
                ret = _extract_random_baseline_from_eval(seed)
                if ret is not None:
                    matrix[method]["full_support"].append(ret)
                continue
            eval_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}.json"
            if not eval_path.exists():
                continue
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            per_variant: dict[str, list[float]] = {}
            for entry in data.get("results", []):
                variant = entry.get("graph_variant", "")
                ret = entry.get("aggregate", {}).get("mean_return")
                if variant and ret is not None:
                    per_variant.setdefault(variant, []).append(float(ret))
            for variant in VARIANTS:
                if variant in per_variant:
                    matrix[method][variant].append(np.mean(per_variant[variant]))

    print(f"\n{'Method':25s}", end="")
    for v in VARIANTS:
        print(f"  {v:22s}", end="")
    print()
    print("-" * (25 + 24 * len(VARIANTS)))

    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for method in METHODS:
        summary[method] = {}
        print(f"{method:25s}", end="")
        for variant in VARIANTS:
            vals = matrix[method][variant]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                summary[method][variant] = {"mean": mean, "std": std, "n": len(vals)}
                print(f"  {mean:8.3f} ± {std:5.3f}     ", end="")
            else:
                summary[method][variant] = {"mean": None, "std": None, "n": 0}
                print(f"  {'N/A':>22s}", end="")
        print()

    return summary


def check_gates(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    print("\n=== Validation Gates ===")
    gates: dict[str, dict[str, Any]] = {}

    def _mean(method: str, variant: str) -> float | None:
        cell = summary.get(method, {}).get(variant, {})
        return cell.get("mean") if cell else None

    ab_fs = _mean("aris_bellman", "full_support")
    bo_fs = _mean("base_only", "full_support")
    rp_vals = [
        ret
        for seed in SEEDS
        if (ret := _extract_random_baseline_from_eval(seed)) is not None
    ]
    rp_fs = float(np.mean(rp_vals)) if rp_vals else None
    ab_mh = _mean("aris_bellman", "minus_high_ce")
    ab_oc = _mean("aris_bellman", "overcomplete")

    def _gate(name: str, left: float | None, right: float | None, desc: str) -> None:
        if left is None or right is None:
            passed = False
            status = "FAIL"
        else:
            passed = left > right
            status = "PASS" if passed else "FAIL"
        gates[name] = {"status": status, "left": left, "right": right, "passed": passed}
        lbl = f"{left:.3f}" if left is not None else "N/A"
        rbl = f"{right:.3f}" if right is not None else "N/A"
        print(f"  {name}: {status}  ({lbl} > {rbl})  — {desc}")

    _gate("G1_method_superiority", ab_fs, bo_fs,
          "aris_bellman/full_support > base_only/full_support")
    _gate("G2_minus_high_ce", ab_fs, ab_mh,
          "aris_bellman/full_support > aris_bellman/minus_high_ce")
    _gate("G3_above_random", ab_fs, rp_fs,
          "aris_bellman/full_support > random_policy/full_support")
    _gate("G4_overcomplete", ab_fs, ab_oc,
          "aris_bellman/full_support > aris_bellman/overcomplete")

    all_pass = all(g["passed"] for g in gates.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return gates


def main() -> None:
    print(f"Step 4: Micro-Train & Evaluation Matrix")
    print(f"Methods: {METHODS}")
    print(f"Variants: {VARIANTS}")
    print(f"Seeds: {SEEDS}")
    print(f"GPUs: {GPUS}")
    print(f"Total jobs: {len(METHODS) * len(VARIANTS) * len(SEEDS)}")
    print()

    train_results = run_training_phase()
    eval_results = run_evaluation_phase()
    summary = build_matrix()
    gates = check_gates(summary)

    output = {
        "matrix": summary,
        "gates": gates,
        "train_results": train_results,
        "eval_results": eval_results,
        "config": {
            "methods": METHODS,
            "variants": VARIANTS,
            "seeds": SEEDS,
            "gpus": GPUS,
            "eval_episodes": EVAL_EPISODES,
        },
    }
    out_path = REPO_ROOT / OUTPUT_DIR / "step4_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
