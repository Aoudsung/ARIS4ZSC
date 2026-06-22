from __future__ import annotations

import argparse
import os
import sys

import yaml

from .canonical import dump_json
from .sweep import run_one, run_sweep


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML object")
    return cfg


def _failure_report(out_dir: str, status: str, exc: BaseException) -> None:
    os.makedirs(out_dir, exist_ok=True)
    failure = {
        "status": status,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "claim_boundary": {
            "matched_control_pool_certified": False,
            "valid_semantic_recovery_probe_claimed": False,
            "pilot_or_benchmark_claimed": False,
        },
    }
    dump_json(os.path.join(out_dir, "failure_report.json"), failure)
    with open(os.path.join(out_dir, "GATE3_RUN_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# TCASO-CRITIC Gate 3 Run Report\n\n")
        f.write("```text\n" + status + "\n```\n\n")
        f.write(f"Failure: `{type(exc).__name__}`: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TCASO-CRITIC Gate 3 depth-2/3 vectorized certifier sweeps")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one layout/depth/tau config")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser("sweep", help="run layout × depth × tau-family sweep")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.cmd == "run":
            row = run_one(cfg, args.out)
            print(os.path.join(args.out, "GATE3_RUN_REPORT.md"))
            return 0
        if args.cmd == "sweep":
            run_sweep(cfg, args.out)
            print(os.path.join(args.out, "AGGREGATE_GATE3_SWEEP_REPORT.md"))
            return 0
        raise AssertionError(args.cmd)
    except Exception as exc:  # noqa: BLE001 - top-level report only; no certificate writing
        _failure_report(args.out, "GATE3_DEPTH23_VECTOR_SWEEP_FAILED_BEFORE_CERTIFICATE_WRITE", exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
