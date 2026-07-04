"""Aggregate E1-rev wave results into the preregistered tables (PM-7).

Two modes:

* ``train``: reads every run's train ``metrics.json`` under ``--results_dir``
  matching ``--tag``, HARD-validates run finality first (LDS-C1 consumer-side
  defense: ``final=True``, ``run_status="ok"``, ``updates_done == --expected_updates``),
  then emits the per-arm terminal-competence table (METHOD_LOCK sec18.12.3
  co-primary: guard-fail seeds COUNT in the denominator) and a performance table
  restricted to guard-pass runs.

* ``eval``: reads held-out eval JSONs matched by ``--eval_glob``, accesses the
  preregistered headline/throughput fields FAIL-CLOSED (a missing field is a
  schema error, never an imputed zero — LDS-C4 lesson), and emits per-arm means
  with seed-level bootstrap 95% CIs (EXPERIMENT_CHAIN_PLAN §9.2: 10k resamples).

Run/eval file paths must contain ``<tag>_<method>_s<seed>`` (the wave-orchestrator
convention), e.g. ``results_phase3/e1rev_aris_bellman_s3/.../metrics.json``.

Output: a JSON document (``--output``) plus a Markdown rendering next to it.
All formulas and exclusion rules are recorded in the output metadata so the
tables are self-describing.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0  # fixed + recorded: deterministic CIs

# Frozen predicate strings (recorded verbatim in output metadata).
COMPETENCE_PREDICATE = (
    'free_rider_guard == "pass" and checkpoint_selection.deployable_checkpoint '
    "is non-empty (train_aris NEW-2 eligibility gate)"
)
PERFORMANCE_INCLUSION = 'free_rider_guard == "pass" (guard-fail runs excluded)'
THROUGHPUT_FORMULA = (
    "eval aggregate fields team/ego_correct_delivery_throughput_per_episode "
    "(denominator = episodes per partner, frozen in evaluate_aris; LDS-C3)"
)


class AggregationError(RuntimeError):
    """A run/eval artifact failed hard validation; nothing was aggregated."""


def _run_id_from_path(path: str, tag: str) -> tuple[str, int]:
    match = re.search(rf"{re.escape(tag)}_([a-z_]+)_s(\d+)", path)
    if not match:
        raise AggregationError(
            f"Cannot parse '<{tag}>_<method>_s<seed>' from path: {path}"
        )
    return match.group(1), int(match.group(2))


def _require(record: dict[str, Any], key: str, path: str) -> Any:
    """Fail-closed field access: missing keys are schema errors, not zeros."""
    if key not in record:
        raise AggregationError(f"{path}: required field '{key}' is missing")
    return record[key]


def _validate_final(metrics: dict[str, Any], path: str, expected_updates: int) -> list[str]:
    """LDS-C1: reject non-final artifacts (periodic metrics.json writes)."""
    problems = []
    if metrics.get("final") is not True:
        problems.append(f"{path}: final={metrics.get('final')!r} (expected True)")
    if metrics.get("run_status") != "ok":
        problems.append(f"{path}: run_status={metrics.get('run_status')!r} (expected 'ok')")
    if int(metrics.get("updates_done", -1)) != int(expected_updates):
        problems.append(
            f"{path}: updates_done={metrics.get('updates_done')!r} "
            f"(expected {expected_updates})"
        )
    return problems


def _bootstrap_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    idx = rng.randint(0, len(arr), size=(BOOTSTRAP_RESAMPLES, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_lo": float(np.percentile(means, 2.5)),
        "ci95_hi": float(np.percentile(means, 97.5)),
        "n": int(len(arr)),
    }


def aggregate_train(results_dir: str, tag: str, expected_updates: int) -> dict[str, Any]:
    pattern = str(Path(results_dir) / f"{tag}_*" / "**" / "metrics.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise AggregationError(f"No metrics.json found under {pattern}")

    problems: list[str] = []
    runs: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as fh:
            metrics = json.load(fh)
        problems.extend(_validate_final(metrics, path, expected_updates))
        method, seed = _run_id_from_path(path, tag)
        runs.append({"method": method, "seed": seed, "path": path, "metrics": metrics})
    if problems:
        raise AggregationError(
            "Hard validation failed (LDS-C1); refusing to aggregate:\n  "
            + "\n  ".join(problems)
        )
    dupes = {}
    for run in runs:
        key = (run["method"], run["seed"])
        if key in dupes:
            raise AggregationError(
                f"Duplicate run for {key}: {dupes[key]} and {run['path']}"
            )
        dupes[key] = run["path"]

    arms: dict[str, dict[str, Any]] = {}
    for method in sorted({run["method"] for run in runs}):
        arm_runs = [run for run in runs if run["method"] == method]
        competent, guard_pass, per_seed = [], [], []
        for run in sorted(arm_runs, key=lambda r: r["seed"]):
            m = run["metrics"]
            guard = str(_require(m, "free_rider_guard", run["path"]))
            is_pass = guard == "pass"
            # Deployability comes from the NEW-2 eligibility gate's own record
            # (checkpoint_selection.deployable_checkpoint: "checkpoint.pt" or
            # null). Truthiness, not is-not-None: "" must not count.
            selection = _require(m, "checkpoint_selection", run["path"])
            is_competent = is_pass and bool(selection.get("deployable_checkpoint"))
            guard_pass.append(is_pass)
            competent.append(is_competent)
            per_seed.append({
                "seed": run["seed"],
                "free_rider_guard": guard,
                "competent": is_competent,
                "selected_ego_correct_completion_rate": m.get(
                    "selected_ego_correct_completion_rate"
                ),
                "selected_ego_sole_correct_delivery_count": m.get(
                    "selected_ego_sole_correct_delivery_count"
                ),
                "best_greedy_return": m.get("best_greedy_return"),
            })
        # Performance means over guard-pass runs ONLY; guard-fail seeds still
        # count in the competence denominator (sec18.12.3).
        pass_rows = [row for row in per_seed if row["free_rider_guard"] == "pass"]

        def _mean_over_pass(key: str) -> float | None:
            vals = [row[key] for row in pass_rows]
            if any(v is None for v in vals) or not vals:
                return None
            return float(np.mean([float(v) for v in vals]))

        arms[method] = {
            "n_seeds": len(per_seed),
            "guard_pass_count": int(sum(guard_pass)),
            "terminal_competence_rate": float(sum(competent)) / len(per_seed),
            "mean_selected_ego_correct_completion_rate_guardpass": _mean_over_pass(
                "selected_ego_correct_completion_rate"
            ),
            "mean_selected_ego_sole_correct_delivery_count_guardpass": _mean_over_pass(
                "selected_ego_sole_correct_delivery_count"
            ),
            "mean_best_greedy_return_guardpass": _mean_over_pass("best_greedy_return"),
            "per_seed": per_seed,
        }
    return {
        "mode": "train",
        "tag": tag,
        "expected_updates": expected_updates,
        "n_runs": len(runs),
        "competence_predicate": COMPETENCE_PREDICATE,
        "performance_inclusion": PERFORMANCE_INCLUSION,
        "note": "train-phase table; NOT the decisive read (held-out eval is)",
        "arms": arms,
    }


# Preregistered eval fields (sec18.6 headline + LDS-C3 throughput). A null
# value here can only come from a 0-episode aggregate ⇒ fail-closed on null.
EVAL_FIELDS = (
    "ego_correct_completion_rate",
    "team_correct_delivery_throughput_per_episode",
    "ego_correct_delivery_throughput_per_episode",
)
# ego_serve_share (LDS-C3) is fail-closed on ABSENCE of the key, but a null
# VALUE is a legitimate measurement state ("no correct delivery by anyone" ⇒
# share undefined): such runs are excluded from the CI with the count recorded.
NULLABLE_EVAL_FIELDS = ("ego_serve_share",)


def aggregate_eval(eval_glob: str, tag: str) -> dict[str, Any]:
    paths = sorted(glob.glob(eval_glob, recursive=True))
    if not paths:
        raise AggregationError(f"No eval JSON matched {eval_glob}")

    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, int], str] = {}
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        method, seed = _run_id_from_path(path, tag)
        if (method, seed) in seen:
            raise AggregationError(
                f"Duplicate eval artifact for ({method}, s{seed}): "
                f"{seen[(method, seed)]} and {path}"
            )
        seen[(method, seed)] = path
        results = _require(doc, "results", path)
        if not results:
            raise AggregationError(f"{path}: empty results list")
        # Per-run value = mean over held-out partners of each preregistered field.
        row: dict[str, Any] = {"method": method, "seed": seed, "path": path}
        for field in EVAL_FIELDS:
            values = []
            for result in results:
                aggregate = _require(result, "aggregate", path)
                value = _require(aggregate, field, path)
                if value is None:
                    raise AggregationError(
                        f"{path}: field '{field}' is null — 0-episode or corrupt "
                        "aggregate must be fixed, not averaged (LDS-C4)"
                    )
                values.append(float(value))
            row[field] = float(np.mean(values))
        for field in NULLABLE_EVAL_FIELDS:
            values = []
            for result in results:
                aggregate = _require(result, "aggregate", path)
                value = _require(aggregate, field, path)  # absence still fails
                if value is not None:
                    values.append(float(value))
            row[field] = float(np.mean(values)) if values else None
        rows.append(row)

    arms: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        arm_rows = sorted(
            (row for row in rows if row["method"] == method), key=lambda r: r["seed"]
        )
        nullable_stats = {}
        for field in NULLABLE_EVAL_FIELDS:
            defined = [row[field] for row in arm_rows if row[field] is not None]
            nullable_stats[field] = {
                **_bootstrap_ci(defined),
                "n_undefined": len(arm_rows) - len(defined),
            }
        arms[method] = {
            "n_runs": len(arm_rows),
            "seeds": [row["seed"] for row in arm_rows],
            **{
                field: _bootstrap_ci([row[field] for row in arm_rows])
                for field in EVAL_FIELDS
            },
            **nullable_stats,
        }
    return {
        "mode": "eval",
        "tag": tag,
        "n_eval_files": len(paths),
        "headline": "ego_correct_completion_rate",
        "throughput_formula": THROUGHPUT_FORMULA,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "arms": arms,
    }


def _render_markdown(doc: dict[str, Any]) -> str:
    lines = [f"# aggregate_e1rev — mode={doc['mode']} tag={doc['tag']}", ""]
    if doc["mode"] == "train":
        lines += [
            f"- runs: {doc['n_runs']} · expected_updates: {doc['expected_updates']}",
            f"- competence predicate: `{doc['competence_predicate']}`",
            f"- performance inclusion: `{doc['performance_inclusion']}`",
            f"- NOTE: {doc['note']}",
            "",
            "| arm | seeds | guard pass | competence rate | selCCR (pass) | ego_sole (pass) | best_ret (pass) |",
            "|---|---|---|---|---|---|---|",
        ]
        for method, arm in doc["arms"].items():
            def _fmt(value: Any) -> str:
                return "-" if value is None else f"{value:.3g}"
            lines.append(
                f"| {method} | {arm['n_seeds']} | {arm['guard_pass_count']} "
                f"| {arm['terminal_competence_rate']:.2f} "
                f"| {_fmt(arm['mean_selected_ego_correct_completion_rate_guardpass'])} "
                f"| {_fmt(arm['mean_selected_ego_sole_correct_delivery_count_guardpass'])} "
                f"| {_fmt(arm['mean_best_greedy_return_guardpass'])} |"
            )
    else:
        lines += [
            f"- eval files: {doc['n_eval_files']} · headline: {doc['headline']}",
            f"- bootstrap: {doc['bootstrap']['resamples']} resamples, seed {doc['bootstrap']['seed']}",
            "",
            "| arm | n | headline mean [95% CI] | team tp/ep | ego tp/ep | serve share |",
            "|---|---|---|---|---|---|",
        ]

        def _num(value: Any) -> str:
            return "-" if value is None else f"{value:.3f}"

        for method, arm in doc["arms"].items():
            head = arm["ego_correct_completion_rate"]
            team = arm["team_correct_delivery_throughput_per_episode"]
            ego = arm["ego_correct_delivery_throughput_per_episode"]
            share = arm["ego_serve_share"]
            share_cell = _num(share["mean"])
            if share.get("n_undefined"):
                share_cell += f" ({share['n_undefined']} undef)"
            lines.append(
                f"| {method} | {arm['n_runs']} "
                f"| {_num(head['mean'])} [{_num(head['ci95_lo'])}, {_num(head['ci95_hi'])}] "
                f"| {_num(team['mean'])} | {_num(ego['mean'])} | {share_cell} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--tag", default="e1rev")
    parser.add_argument("--results_dir", default="results_phase3")
    parser.add_argument("--expected_updates", type=int, default=5000)
    parser.add_argument("--eval_glob", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.mode == "train":
        doc = aggregate_train(args.results_dir, args.tag, args.expected_updates)
    else:
        if not args.eval_glob:
            parser.error("--eval_glob is required in eval mode")
        doc = aggregate_eval(args.eval_glob, args.tag)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2, sort_keys=True))
    markdown = output.with_suffix(".md")
    markdown.write_text(_render_markdown(doc))
    print(f"[aggregate_e1rev] wrote {output} and {markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
