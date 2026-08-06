#!/usr/bin/env python3
"""Run every repository pytest function in a fresh bounded process.

JAX/XLA executables and thread pools are intentionally process-scoped.  A
single long-lived pytest process can retain compiled programs and make a clean
suite appear to hang or exhaust memory.  This runner is the authoritative local
CPU regression entrypoint: it discovers top-level test functions, runs each in
its own process, and writes incremental machine-readable results.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "experiments" / "overcooked_v2" / "tests"
RESULT_JSON = ROOT / "validation" / "ISOLATED_TEST_RESULTS.json"
RESULT_TEXT = ROOT / "validation" / "LOCAL_TEST_RESULTS.txt"
TIMEOUT_SECONDS = int(os.environ.get("DELTA_TEST_TIMEOUT_SECONDS", "600"))


def discover() -> list[str]:
    nodes: list[str] = []
    for source in sorted(TEST_ROOT.glob("test_delta_*.py")):
        module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        relative = source.relative_to(ROOT)
        for item in module.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                nodes.append(f"{relative}::{item.name}")
    return nodes


def write_results(records: list[dict[str, object]], *, started: float, complete: bool) -> None:
    passed = sum(row["status"] == "passed" for row in records)
    failed = sum(row["status"] == "failed" for row in records)
    timed_out = sum(row["status"] == "timeout" for row in records)
    package_version = lambda name: importlib.metadata.version(name)
    payload = {
        "schema_version": 1,
        "runner": "validation/run_all_isolated_tests.py",
        "platform": "cpu",
        "python_version": platform.python_version(),
        "jax_version": package_version("jax"),
        "numpy_version": package_version("numpy"),
        "started_unix": started,
        "finished_unix": time.time() if complete else None,
        "duration_seconds": (time.time() - started) if complete else None,
        "complete": complete,
        "discovered": len(discover()),
        "executed": len(records),
        "passed": passed,
        "failed": failed,
        "timed_out": timed_out,
        "records": records,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "DELTA v4 isolated local CPU regression",
        f"python={payload['python_version']}",
        f"jax={payload['jax_version']}",
        f"numpy={payload['numpy_version']}",
        f"platform={payload['platform']}",
        f"complete={str(complete).lower()}",
        f"discovered={payload['discovered']}",
        f"executed={payload['executed']}",
        f"passed={passed}",
        f"failed={failed}",
        f"timed_out={timed_out}",
    ]
    if records:
        lines.append("")
    for row in records:
        lines.append(f"[{row['status']}] {row['nodeid']} ({row['seconds']:.3f}s)")
        output = str(row.get("output", "")).strip()
        if row["status"] != "passed" and output:
            lines.extend(f"    {line}" for line in output.splitlines()[-40:])
    RESULT_TEXT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_one(nodeid: str) -> dict[str, object]:
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [sys.executable, "-m", "pytest", "-q", nodeid]
    began = time.time()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=TIMEOUT_SECONDS)
        status = "passed" if process.returncode == 0 else "failed"
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        status = "timeout"
        returncode = None
    return {
        "nodeid": nodeid,
        "status": status,
        "returncode": returncode,
        "seconds": time.time() - began,
        "output": output,
    }


def _resume_records(nodes: list[str]) -> tuple[float, list[dict[str, object]]]:
    """Load only previously passed records in current discovery order."""

    if not RESULT_JSON.is_file():
        return time.time(), []
    try:
        payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return time.time(), []
    prior = {
        str(row.get("nodeid")): row
        for row in payload.get("records", [])
        if isinstance(row, dict) and row.get("status") == "passed"
    }
    records = [prior[node] for node in nodes if node in prior]
    started = float(payload.get("started_unix") or time.time())
    return started, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an incomplete run, retaining only previously passed nodes",
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help="run at most this many pending nodes and leave an incremental result",
    )
    args = parser.parse_args()
    if args.max_tests is not None and args.max_tests <= 0:
        parser.error("--max-tests must be positive")

    nodes = discover()
    if args.resume:
        started, records = _resume_records(nodes)
    else:
        started, records = time.time(), []
    completed_nodes = {str(row["nodeid"]) for row in records}
    pending = [node for node in nodes if node not in completed_nodes]
    selected = pending if args.max_tests is None else pending[: args.max_tests]
    write_results(records, started=started, complete=False)
    for nodeid in selected:
        index = nodes.index(nodeid) + 1
        print(f"[{index}/{len(nodes)}] {nodeid}", flush=True)
        record = run_one(nodeid)
        records.append(record)
        records.sort(key=lambda row: nodes.index(str(row["nodeid"])))
        print(f"  -> {record['status']} ({record['seconds']:.2f}s)", flush=True)
        write_results(records, started=started, complete=False)
        if record["status"] != "passed":
            print(str(record["output"]), flush=True)

    complete = len(records) == len(nodes) and all(
        row["status"] == "passed" for row in records
    )
    write_results(records, started=started, complete=complete)
    failures = [row for row in records if row["status"] != "passed"]
    if complete:
        print(f"complete: {len(records)}/{len(nodes)} passed", flush=True)
    else:
        print(
            f"incremental: {len(records)}/{len(nodes)} executed, "
            f"{len(records) - len(failures)} passed",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
