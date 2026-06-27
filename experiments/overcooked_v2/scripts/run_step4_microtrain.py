"""Step 4: Micro-train across (method × graph_variant × seed) and build evaluation matrix.

Phase-separable + host-configurable orchestrator. The two execution hosts are the
same physical machine sharing /apps/users/cxw/.../CPR_REPO, so checkpoints written by
the training phase are visible to the eval phase on the other container:

  - Training is GPU-bound  → run on the 8-GPU container (CPU-capped at 8 cores):
        run_step4 --phase train --gpus 0,2,4,5,6,7 --updates 5000 --seeds 0
  - Eval (esp. full diagnostics) is CPU-bound → run on the 1-GPU / 144-core container:
        run_step4 --phase eval --eval-workers 24 --full-diagnostics --seeds 0
  - Build the matrix + gates from the eval JSONs on either host:
        run_step4 --phase matrix --seeds 0

Per-job values (gpu, updates, episodes, full_diagnostics, timeout) are passed as
function arguments so they survive ProcessPoolExecutor workers under any start method.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
# Only aris_bellman is evaluated across all graph variants (graph-causality row);
# the other trainable methods are only needed at full_support for the method gates.
EVAL_VARIANTS: dict[str, tuple[str, ...]] = {
    "aris_bellman": VARIANTS,
}


def _job_key(method: str, variant: str, seed: int) -> str:
    return f"{method}/{variant}/seed{seed}"


def _checkpoint_dir(method: str, variant: str, seed: int) -> Path:
    return REPO_ROOT / OUTPUT_DIR / LAYOUT / method / variant / f"seed{seed}"


# --------------------------------------------------------------------------- train

def _run_train_job(
    gpu: int, method: str, variant: str, seed: int, updates: int, timeout: int
) -> dict[str, Any]:
    key = _job_key(method, variant, seed)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "overcooked_v2" / "train_aris.py"),
        "--config", str(REPO_ROOT / CONFIG),
        "--graph_variant", variant,
        "--method", method,
        "--seed", str(seed),
        "--updates", str(updates),
        "--output_dir", str(REPO_ROOT / OUTPUT_DIR),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        rc, stderr_tail = result.returncode, result.stderr[-500:]
    except subprocess.TimeoutExpired:
        rc, stderr_tail = -1, f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0
    ckpt = _checkpoint_dir(method, variant, seed) / "checkpoint.pt"
    return {
        "key": key,
        "method": method,
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "returncode": rc,
        "elapsed": elapsed,
        "checkpoint_exists": ckpt.exists(),
        "stderr_tail": stderr_tail if rc != 0 else "",
    }


def run_training_phase(
    gpus: list[int], seeds: tuple[int, ...], updates: int, timeout: int
) -> list[dict[str, Any]]:
    all_jobs = [
        (method, variant, seed)
        for method in METHODS
        for variant in VARIANTS
        for seed in seeds
    ]
    print(f"=== Phase 2: Training {len(all_jobs)} jobs on {len(gpus)} GPU slots "
          f"(updates={updates}) ===")

    results: list[dict[str, Any]] = []
    pending: list[tuple[str, str, int]] = []
    for method, variant, seed in all_jobs:
        ckpt = _checkpoint_dir(method, variant, seed) / "checkpoint.pt"
        if ckpt.exists():
            results.append({
                "key": _job_key(method, variant, seed),
                "method": method, "variant": variant, "seed": seed,
                "gpu": -1, "returncode": 0, "elapsed": 0,
                "checkpoint_exists": True, "stderr_tail": "",
            })
            print(f"  [{len(results):2d}/{len(all_jobs)}] "
                  f"{_job_key(method, variant, seed):45s} CACHED")
        else:
            pending.append((method, variant, seed))

    if pending:
        gpu_pool = list(gpus)
        with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
            active: dict[Any, int] = {}
            job_iter = iter(pending)

            def _submit_train() -> bool:
                if not gpu_pool:
                    return False
                try:
                    method, variant, seed = next(job_iter)
                except StopIteration:
                    return False
                gpu = gpu_pool.pop(0)
                future = pool.submit(
                    _run_train_job, gpu, method, variant, seed, updates, timeout
                )
                active[future] = gpu
                return True

            for _ in range(min(len(gpus), len(pending))):
                _submit_train()

            while active:
                done_futures = [f for f in active if f.done()]
                if not done_futures:
                    time.sleep(0.5)
                    continue
                for future in done_futures:
                    freed_gpu = active.pop(future)
                    gpu_pool.append(freed_gpu)
                    info = future.result()
                    results.append(info)
                    status = "OK" if info["checkpoint_exists"] else "FAIL"
                    print(f"  [{len(results):2d}/{len(all_jobs)}] {info['key']:45s} "
                          f"GPU:{info['gpu']} {status} ({info['elapsed']:.0f}s)")
                    if info["returncode"] != 0:
                        print(f"    stderr: {info['stderr_tail']}")
                    _submit_train()

    ok = sum(1 for r in results if r["checkpoint_exists"])
    print(f"\nTraining complete: {ok}/{len(all_jobs)} checkpoints saved")
    return results


# ---------------------------------------------------------------------------- eval

def _run_eval_job(
    gpu: int, method: str, seed: int, variant: str,
    episodes: int, full_diagnostics: bool, timeout: int,
) -> dict[str, Any]:
    context_method = "aris_bellman" if method == "random_policy" else method
    context_variant = "full_support" if method == "random_policy" else variant
    ckpt_dir = _checkpoint_dir(context_method, context_variant, seed)
    suffix = f"_{variant}" if variant != "full_support" else ""
    out_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}{suffix}.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "overcooked_v2" / "evaluate_aris.py"),
        "--checkpoint", str(ckpt_dir),
        "--graph_variants", variant,
        "--partners", "all",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", str(out_path),
    ]
    if method == "random_policy":
        cmd.append("--random_policy_only")
    if not full_diagnostics:
        cmd.append("--fast")
    if out_path.exists():
        out_path.unlink()
    eval_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=eval_env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        rc, stderr_tail = result.returncode, result.stderr[-500:]
    except subprocess.TimeoutExpired:
        rc, stderr_tail = -1, f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0
    ok = rc == 0 and out_path.exists()
    return {
        "method": method,
        "seed": seed,
        "gpu": gpu,
        "variant": variant,
        "returncode": rc,
        "elapsed": elapsed,
        "output_path": str(out_path),
        "command": cmd,
        "eval_args": {
            "method": method,
            "context_method": context_method,
            "variant": variant,
            "context_variant": context_variant,
            "seed": seed,
            "episodes": episodes,
            "full_diagnostics": full_diagnostics,
            "timeout": timeout,
        },
        "ok": ok,
        "stderr_tail": stderr_tail if rc != 0 else "",
    }


def run_evaluation_phase(
    eval_gpus: list[int], seeds: tuple[int, ...],
    episodes: int, full_diagnostics: bool, timeout: int,
) -> list[dict[str, Any]]:
    pending: list[tuple[str, int, str]] = []
    for m in METHODS:
        for v in EVAL_VARIANTS.get(m, ("full_support",)):
            for s in seeds:
                ckpt_method = "aris_bellman" if m == "random_policy" else m
                ckpt_variant = "full_support" if m == "random_policy" else v
                if (_checkpoint_dir(ckpt_method, ckpt_variant, s) / "checkpoint.pt").exists():
                    pending.append((m, s, v))
    mode = "full-diagnostics" if full_diagnostics else "fast"
    print(f"\n=== Phase 4: Evaluation ({len(pending)} runs, {len(eval_gpus)} workers, "
          f"{mode}) ===")

    results: list[dict[str, Any]] = []
    gpu_pool = list(eval_gpus)
    with ProcessPoolExecutor(max_workers=len(eval_gpus)) as pool:
        active: dict[Any, int] = {}
        job_iter = iter(pending)

        def _submit_next() -> bool:
            if not gpu_pool:
                return False
            try:
                method, seed, variant = next(job_iter)
            except StopIteration:
                return False
            gpu = gpu_pool.pop(0)
            future = pool.submit(
                _run_eval_job, gpu, method, seed, variant,
                episodes, full_diagnostics, timeout,
            )
            active[future] = gpu
            return True

        for _ in range(min(len(eval_gpus), len(pending))):
            _submit_next()

        while active:
            done_futures = [f for f in active if f.done()]
            if not done_futures:
                time.sleep(1)
                continue
            for future in done_futures:
                freed_gpu = active.pop(future)
                gpu_pool.append(freed_gpu)
                info = future.result()
                results.append(info)
                status = "OK" if info["ok"] else "FAIL"
                print(f"  [{len(results):2d}/{len(pending)}] "
                      f"{info['method']}/{info['variant']}/seed{info['seed']} "
                      f"GPU:{info['gpu']} {status} ({info['elapsed']:.0f}s)")
                if not info["ok"]:
                    print(f"    stderr: {info['stderr_tail']}")
                _submit_next()

    ok = sum(1 for r in results if r["ok"])
    print(f"\nEvaluation complete: {ok}/{len(pending)} succeeded")
    return results


# -------------------------------------------------------------------------- matrix

def _manifest_output_paths(eval_manifest: dict[str, Any] | None) -> set[Path]:
    if not eval_manifest:
        return set()
    return {
        Path(str(result["output_path"])).resolve()
        for result in eval_manifest.get("results", [])
        if isinstance(result, dict) and result.get("output_path")
    }


def _assert_eval_path_manifested(
    eval_path: Path,
    eval_manifest: dict[str, Any] | None,
) -> None:
    manifested = _manifest_output_paths(eval_manifest)
    if manifested and eval_path.resolve() not in manifested:
        raise RuntimeError(
            f"Matrix wants {eval_path}, but it is absent from eval_results.json."
        )


def _extract_random_baseline_from_eval(
    seed: int,
    eval_manifest: dict[str, Any] | None = None,
) -> float | None:
    for method in METHODS:
        if method == "random_policy":
            continue
        eval_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}.json"
        if not eval_path.exists():
            continue
        _assert_eval_path_manifested(eval_path, eval_manifest)
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


def _extract_direct_random_policy_eval(
    seed: int,
    eval_manifest: dict[str, Any] | None = None,
) -> float | None:
    eval_path = REPO_ROOT / OUTPUT_DIR / f"eval_random_policy_seed{seed}.json"
    if not eval_path.exists():
        return None
    _assert_eval_path_manifested(eval_path, eval_manifest)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    returns = [
        float(entry.get("aggregate", {}).get("mean_return"))
        for entry in data.get("results", [])
        if entry.get("method") == "random_policy"
        and entry.get("aggregate", {}).get("mean_return") is not None
    ]
    return float(np.mean(returns)) if returns else None


def build_matrix(
    seeds: tuple[int, ...],
    eval_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print("\n=== Phase 5: Evaluation Matrix ===")
    matrix: dict[str, dict[str, list[float]]] = {
        method: {variant: [] for variant in VARIANTS}
        for method in METHODS
    }

    for method in METHODS:
        for seed in seeds:
            if method == "random_policy":
                ret = _extract_direct_random_policy_eval(seed, eval_manifest)
                if ret is None:
                    ret = _extract_random_baseline_from_eval(seed, eval_manifest)
                if ret is not None:
                    matrix[method]["full_support"].append(ret)
                continue
            for ev in EVAL_VARIANTS.get(method, ("full_support",)):
                suffix = f"_{ev}" if ev != "full_support" else ""
                eval_path = REPO_ROOT / OUTPUT_DIR / f"eval_{method}_seed{seed}{suffix}.json"
                if not eval_path.exists():
                    continue
                _assert_eval_path_manifested(eval_path, eval_manifest)
                data = json.loads(eval_path.read_text(encoding="utf-8"))
                for entry in data.get("results", []):
                    variant = entry.get("graph_variant", "")
                    ret = entry.get("aggregate", {}).get("mean_return")
                    if variant in VARIANTS and ret is not None:
                        matrix[method][variant].append(float(ret))

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


def check_gates(
    summary: dict[str, Any],
    seeds: tuple[int, ...],
    eval_manifest: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    print("\n=== Validation Gates ===")
    gates: dict[str, dict[str, Any]] = {}

    def _mean(method: str, variant: str) -> float | None:
        cell = summary.get(method, {}).get(variant, {})
        return cell.get("mean") if cell else None

    ab_fs = _mean("aris_bellman", "full_support")
    bo_fs = _mean("base_only", "full_support")
    rp_vals = []
    for seed in seeds:
        ret = _extract_direct_random_policy_eval(seed, eval_manifest)
        if ret is None:
            ret = _extract_random_baseline_from_eval(seed, eval_manifest)
        if ret is not None:
            rp_vals.append(ret)
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


# ---------------------------------------------------------------------------- main

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["train", "eval", "matrix", "all"], default="all")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    p.add_argument("--gpus", default="0,0,0",
                   help="comma-separated GPU ids = training worker slots")
    p.add_argument("--eval-workers", type=int, default=6,
                   help="number of parallel eval workers (CPU-bound)")
    p.add_argument("--eval-gpu", type=int, default=0,
                   help="GPU id all eval workers share (eval is CPU-bound)")
    p.add_argument("--full-diagnostics", action="store_true",
                   help="run eval WITHOUT --fast (G-TVOI/MI/belief-swap/factor-deletion)")
    p.add_argument("--updates", type=int, default=5000,
                   help="train_aris total_updates override")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--train-timeout", type=int, default=3600)
    p.add_argument("--eval-timeout", type=int, default=14400)
    return p.parse_args(argv)


def _write_json(name: str, payload: Any) -> Path:
    out_path = REPO_ROOT / OUTPUT_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def _load_eval_manifest_for_matrix(seeds: tuple[int, ...]) -> dict[str, Any] | None:
    path = REPO_ROOT / OUTPUT_DIR / "eval_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest_seeds = tuple(int(seed) for seed in data.get("seeds", ()))
    if manifest_seeds != tuple(seeds):
        raise RuntimeError(
            "eval_results.json seed manifest does not match matrix seeds: "
            f"manifest={manifest_seeds}, matrix={tuple(seeds)}"
        )
    data["manifest_path"] = str(path)
    return data


def _matrix_eval_config(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    eval_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    config = {
        "methods": METHODS,
        "variants": VARIANTS,
        "seeds": seeds,
        "updates": args.updates,
    }
    if eval_manifest is not None:
        config.update(
            {
                "eval_results_manifest": eval_manifest.get("manifest_path"),
                "eval_episodes": eval_manifest.get("eval_episodes"),
                "full_diagnostics": eval_manifest.get("full_diagnostics"),
                "eval_commands": [
                    result.get("command")
                    for result in eval_manifest.get("results", [])
                    if isinstance(result, dict) and result.get("command")
                ],
                "eval_args": [
                    result.get("eval_args")
                    for result in eval_manifest.get("results", [])
                    if isinstance(result, dict) and result.get("eval_args")
                ],
            }
        )
    return config


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    eval_gpus = [int(args.eval_gpu)] * int(args.eval_workers)

    print("Step 4: Micro-Train & Evaluation Matrix")
    print(f"Phase: {args.phase}")
    print(f"Methods: {METHODS}")
    print(f"Variants: {VARIANTS}")
    print(f"Seeds: {seeds}")
    print(f"Train GPU slots: {gpus} (updates={args.updates})")
    print(f"Eval workers: {len(eval_gpus)} on GPU {args.eval_gpu} "
          f"({'full-diagnostics' if args.full_diagnostics else 'fast'})")
    print()

    if args.phase in ("train", "all"):
        train_results = run_training_phase(gpus, seeds, args.updates, args.train_timeout)
        _write_json("train_results.json", {"seeds": seeds, "updates": args.updates,
                                           "results": train_results})

    if args.phase in ("eval", "all"):
        eval_results = run_evaluation_phase(
            eval_gpus, seeds, args.eval_episodes, args.full_diagnostics, args.eval_timeout
        )
        _write_json("eval_results.json", {"seeds": seeds,
                                          "eval_episodes": args.eval_episodes,
                                          "full_diagnostics": args.full_diagnostics,
                                          "results": eval_results})

    if args.phase in ("matrix", "all"):
        eval_manifest = _load_eval_manifest_for_matrix(seeds)
        summary = build_matrix(seeds, eval_manifest)
        gates = check_gates(summary, seeds, eval_manifest)
        out_path = _write_json("step4_matrix.json", {
            "matrix": summary,
            "gates": gates,
            "config": _matrix_eval_config(args, seeds, eval_manifest),
        })
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
