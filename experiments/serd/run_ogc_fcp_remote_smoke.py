"""Remote-only OGC/FCP source smoke runner for SERD M3.

This script is intended to be copied or invoked on the SSH backend that has the
OGC checkout and FCP-labeled population files. It does not produce claim-bearing
SERD BranchRecord rows. Its purpose is to separate three M3 gates:

1. inventory: population JSON/file metadata exists;
2. checkpoint-load: one OGC checkpoint/config loads under the OGC environment;
3. eval-smoke: OGC's own population evaluator can run a tiny rollout.

Do not treat a passing smoke result as FCP/PECAN scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUCCESS_STATUSES = {
    "inventory_complete",
    "checkpoint_load_passed_no_rollout",
    "eval_smoke_passed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inventory", "checkpoint-load", "eval-smoke"),
        default="inventory",
    )
    parser.add_argument(
        "--ogc-src",
        type=Path,
        default=Path("/apps/users/cxw/ZSC_coordinator/external/OGC/src"),
        help="Remote OGC src directory containing minimax/ and populations/.",
    )
    parser.add_argument(
        "--population-json",
        type=Path,
        default=Path("populations/fcp/Overcooked-CounterCircuit6_9/population.json"),
    )
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--band", choices=("low", "mid", "high"), default="low")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/apps/users/cxw/logs/minimax"),
        help="Remote OGC training log directory used by eval-smoke.",
    )
    parser.add_argument(
        "--ego-xpid",
        type=str,
        default=None,
        help="Required for eval-smoke: OGC xpid for the evaluated ego policy.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint")
    parser.add_argument("--env-names", type=str, default="Overcooked-CounterCircuit6_9")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--agent-idxs", type=str, default="0")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/serd_ogc_fcp_remote_smoke"),
    )
    parser.add_argument("--results-fname", type=str, default="ogc_fcp_eval_smoke")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/serd_ogc_fcp_remote_smoke/smoke.json"),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_population(ogc_src: Path, population_json: Path) -> tuple[Path, dict[str, Any]]:
    population_path = resolve_under(ogc_src, population_json)
    with population_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return population_path, payload


def checkpoint_entry(
    population: dict[str, Any],
    agent_id: int,
    band: str,
) -> tuple[int, str, str]:
    offset = {"low": 1, "mid": 2, "high": 3}[band]
    population_index = (agent_id - 1) * 3 + offset
    key = str(population_index)
    meta_key = f"{population_index}_meta"
    if key not in population:
        raise KeyError(f"population entry missing: {key}")
    if meta_key not in population:
        raise KeyError(f"population meta entry missing: {meta_key}")
    return population_index, str(population[key]), str(population[meta_key])


def infer_xpid_from_population_entry(
    ogc_src: Path,
    population: dict[str, Any],
    agent_id: int,
    band: str,
) -> tuple[str, Path]:
    _, checkpoint_rel, _ = checkpoint_entry(population, agent_id, band)
    checkpoint_path = resolve_under(ogc_src, Path(checkpoint_rel))
    xpid_path = checkpoint_path.parent / "xpid.txt"
    if not xpid_path.exists():
        raise FileNotFoundError(
            f"--ego-xpid was not provided and xpid.txt is missing: {xpid_path}"
        )
    xpid = xpid_path.read_text(encoding="utf-8").strip()
    if not xpid:
        raise ValueError(f"xpid.txt is empty: {xpid_path}")
    return xpid, xpid_path


def patched_population_evaluator(ogc_src: Path, results_path: Path) -> Path:
    source_path = ogc_src / "minimax" / "evaluate_against_population.py"
    if not source_path.exists():
        raise FileNotFoundError(f"OGC evaluator not found: {source_path}")

    patched_dir = results_path / "_patched_ogc"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_path = patched_dir / "evaluate_against_population_csv_fixed.py"
    shutil.copyfile(source_path, patched_path)

    original = "df = pd.DataFrame.from_dict(all_eval_stats)"
    replacement = (
        "df = pd.DataFrame.from_dict(all_eval_stats, "
        'orient="index").transpose()'
    )
    text = patched_path.read_text(encoding="utf-8")
    if original not in text:
        raise RuntimeError(
            "OGC evaluator CSV construction did not match the expected source line"
        )
    patched_path.write_text(text.replace(original, replacement), encoding="utf-8")
    return patched_path


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    ogc_src = args.ogc_src.expanduser().resolve()
    population_path, population = load_population(ogc_src, args.population_json)
    population_size = int(population.get("population_size", 0))

    checkpoint_entries = []
    missing = []
    band_counts = {"low": 0, "mid": 0, "high": 0, "other": 0}
    for idx in range(1, population_size + 1):
        raw_path = str(population.get(str(idx), ""))
        if not raw_path:
            missing.append({"entry": idx, "reason": "missing population key"})
            continue
        checkpoint_path = resolve_under(ogc_src, Path(raw_path))
        band = checkpoint_path.stem if checkpoint_path.stem in band_counts else "other"
        band_counts[band] += 1
        checkpoint_entries.append(
            {
                "entry": idx,
                "relative_path": raw_path,
                "exists": checkpoint_path.exists(),
                "size_bytes": checkpoint_path.stat().st_size
                if checkpoint_path.exists()
                else None,
                "band": band,
            }
        )
        if not checkpoint_path.exists():
            missing.append({"entry": idx, "path": str(checkpoint_path)})

    sample_index, sample_checkpoint, sample_meta = checkpoint_entry(
        population, args.agent_id, args.band
    )
    sample_checkpoint_path = resolve_under(ogc_src, Path(sample_checkpoint))
    sample_meta_path = resolve_under(ogc_src, Path(sample_meta))

    return {
        "status": "inventory_complete" if not missing else "inventory_incomplete",
        "mode": args.mode,
        "ogc_src": str(ogc_src),
        "population_json": str(population_path),
        "population_json_sha256": sha256_file(population_path),
        "population_size": population_size,
        "band_counts": band_counts,
        "missing": missing,
        "sample": {
            "population_index": sample_index,
            "checkpoint": str(sample_checkpoint_path),
            "checkpoint_exists": sample_checkpoint_path.exists(),
            "checkpoint_sha256": sha256_file(sample_checkpoint_path)
            if sample_checkpoint_path.exists()
            else None,
            "meta": str(sample_meta_path),
            "meta_exists": sample_meta_path.exists(),
        },
        "entries_preview": checkpoint_entries[:12],
    }


def checkpoint_load(args: argparse.Namespace) -> dict[str, Any]:
    ogc_src = args.ogc_src.expanduser().resolve()
    if str(ogc_src) not in sys.path:
        sys.path.insert(0, str(ogc_src))
    population_path, population = load_population(ogc_src, args.population_json)
    population_index, checkpoint_rel, meta_rel = checkpoint_entry(
        population, args.agent_id, args.band
    )
    checkpoint_path = resolve_under(ogc_src, Path(checkpoint_rel))
    meta_path = resolve_under(ogc_src, Path(meta_rel))

    from minimax.util.checkpoint import load_config, load_pkl_object

    obj = load_pkl_object(str(checkpoint_path))
    cfg = load_config(str(meta_path))
    state_keys = None
    if isinstance(obj, (list, tuple)) and len(obj) > 1 and hasattr(obj[1], "keys"):
        state_keys = sorted(str(key) for key in obj[1].keys())

    return {
        "status": "checkpoint_load_passed_no_rollout",
        "mode": args.mode,
        "ogc_src": str(ogc_src),
        "population_json": str(population_path),
        "population_index": population_index,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "meta": str(meta_path),
        "loaded_checkpoint_type": type(obj).__name__,
        "loaded_checkpoint_len": len(obj) if hasattr(obj, "__len__") else None,
        "state_keys": state_keys,
        "config": {
            "env_name": getattr(cfg, "env_name", None),
            "layout": getattr(getattr(cfg, "env_args", None), "fix_to_single_layout", None),
            "student_model_name": getattr(cfg, "student_model_name", None),
            "student_agent_kind": getattr(cfg, "student_agent_kind", None),
            "train_runner": getattr(cfg, "train_runner", None),
        },
    }


def eval_smoke(args: argparse.Namespace) -> dict[str, Any]:
    ogc_src = args.ogc_src.expanduser().resolve()
    population_path = resolve_under(ogc_src, args.population_json)
    _, population = load_population(ogc_src, args.population_json)
    ego_xpid = args.ego_xpid
    ego_xpid_source = "argument"
    if not ego_xpid:
        ego_xpid, xpid_path = infer_xpid_from_population_entry(
            ogc_src, population, args.agent_id, args.band
        )
        ego_xpid_source = str(xpid_path)
    results_path = args.results_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser()
    output_csv = results_path / f"{args.results_fname}.csv"
    results_path.mkdir(parents=True, exist_ok=True)
    evaluator_path = patched_population_evaluator(ogc_src, results_path)

    cmd = [
        sys.executable,
        str(evaluator_path),
        "--population_json",
        str(population_path),
        "--log_dir",
        str(log_dir),
        "--xpid",
        ego_xpid,
        "--checkpoint_name",
        args.checkpoint_name,
        "--env_names",
        args.env_names,
        "--n_episodes",
        str(args.n_episodes),
        "--agent_idxs",
        args.agent_idxs,
        "--results_path",
        str(results_path),
        "--results_fname",
        args.results_fname,
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ogc_src)
        if not existing_pythonpath
        else f"{ogc_src}{os.pathsep}{existing_pythonpath}"
    )
    completed = subprocess.run(
        cmd,
        cwd=str(ogc_src),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_sec,
        check=False,
    )
    output_csv_exists = output_csv.exists()
    output_csv_size_bytes = output_csv.stat().st_size if output_csv_exists else None
    failure_reasons = []
    if completed.returncode != 0:
        failure_reasons.append(f"returncode={completed.returncode}")
    if not output_csv_exists:
        failure_reasons.append("output_csv_missing")
    elif output_csv_size_bytes <= 1:
        failure_reasons.append("output_csv_empty_or_headerless")
    status = "eval_smoke_passed" if not failure_reasons else "eval_smoke_failed"
    return {
        "status": status,
        "mode": args.mode,
        "command": cmd,
        "cwd": str(ogc_src),
        "patched_evaluator": str(evaluator_path),
        "patched_evaluator_source": str(ogc_src / "minimax" / "evaluate_against_population.py"),
        "patched_evaluator_change": (
            "CSV output uses orient='index' plus transpose so OGC's global and "
            "band-specific statistic lists can have different lengths."
        ),
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "ego_xpid": ego_xpid,
        "ego_xpid_source": ego_xpid_source,
        "failure_reasons": failure_reasons,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_csv": str(output_csv),
        "output_csv_exists": output_csv_exists,
        "output_csv_size_bytes": output_csv_size_bytes,
        "notes": [
            "This is an OGC rollout/API smoke only.",
            "It does not emit SERD BranchRecord rows and is not claim-bearing.",
        ],
    }


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "claim_bearing": False,
    }
    try:
        if args.mode == "inventory":
            payload.update(inventory(args))
        elif args.mode == "checkpoint-load":
            payload.update(checkpoint_load(args))
        elif args.mode == "eval-smoke":
            payload.update(eval_smoke(args))
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        exit_code = 0 if str(payload.get("status", "")) in SUCCESS_STATUSES else 1
    except Exception as exc:  # Visible failure is more useful than silent fallback.
        payload.update(
            {
                "status": "failed",
                "mode": args.mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        exit_code = 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
