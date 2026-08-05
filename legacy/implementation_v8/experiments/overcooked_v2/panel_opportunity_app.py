"""S1 run-disjoint opportunity panel (full-episode version one).

Implements ``docs/status/S1_EXECUTION_SPEC.md``.  The tool only evaluates
existing official checkpoints; it never trains.  Every pairing runs the
unmodified Official episode contract through the registered evaluator.  Any
invalid evaluator output is fail-closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.official_adapter import (
    _official_symbol,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_evaluation_app import _env_kwargs
from src.path_c.experiment import (
    OFFICIAL_CORRECT_DELIVERY_REWARD,
    OFFICIAL_EVALUATION_ROOT_SEED,
    RunConfig,
    load_config,
)
from src.path_c.official_statistics import OFFICIAL_EPISODES
from src.path_c.storage import write_json

MANIFEST_VERSION = 1
PANEL_TYPES = ("sp", "op", "state-augmented", "fcp")
RUN_FIELDS = {"run_id", "checkpoint"}
ENTRY_FIELDS = {"type", "runs"}
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260803


def load_panel_manifest(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "version",
        "layout",
        "entries",
    }:
        raise ValueError("Panel manifest fields differ from the registered schema.")
    if int(payload["version"]) != MANIFEST_VERSION:
        raise ValueError("Panel manifest version mismatch.")
    entries = payload["entries"]
    if not isinstance(entries, Sequence) or not entries:
        raise ValueError("Panel manifest must list at least one entry.")
    seen_types = set()
    normalized = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != ENTRY_FIELDS:
            raise ValueError("Panel entry fields differ from the registered schema.")
        panel_type = str(entry["type"])
        if panel_type not in PANEL_TYPES or panel_type in seen_types:
            raise ValueError(f"Panel entry type is unknown or duplicated: {panel_type}")
        seen_types.add(panel_type)
        runs = entry["runs"]
        if not isinstance(runs, Sequence) or not runs:
            raise ValueError(f"Panel entry {panel_type} lists no runs.")
        seen_run_ids = set()
        for run in runs:
            if not isinstance(run, Mapping) or set(run) != RUN_FIELDS:
                raise ValueError("Panel run fields differ from the registered schema.")
            run_id = str(run["run_id"])
            checkpoint = Path(str(run["checkpoint"])).resolve()
            if run_id in seen_run_ids or not checkpoint.exists():
                raise ValueError(f"Panel run is duplicated or missing: {run_id}")
            seen_run_ids.add(run_id)
        normalized.append({"type": panel_type, "runs": list(runs)})
    return {"layout": str(payload["layout"]), "entries": normalized}


def build_pairings(manifest: Mapping[str, Any]) -> list[dict]:
    """Every mode-run x partner-run pairing with the registered flags."""
    pairings = []
    for mode_entry in manifest["entries"]:
        for mode_run in mode_entry["runs"]:
            for partner_entry in manifest["entries"]:
                for partner_run in partner_entry["runs"]:
                    identity = mode_run["run_id"] == partner_run["run_id"]
                    same_type = mode_entry["type"] == partner_entry["type"]
                    if same_type and identity:
                        flags = {"identity_match", "same_type"}
                    elif same_type:
                        flags = {"same_type_disjoint"}
                    else:
                        flags = set()
                    pairings.append(
                        {
                            "mode_type": mode_entry["type"],
                            "mode_run_id": str(mode_run["run_id"]),
                            "mode_checkpoint": str(mode_run["checkpoint"]),
                            "partner_type": partner_entry["type"],
                            "partner_run_id": str(partner_run["run_id"]),
                            "partner_checkpoint": str(partner_run["checkpoint"]),
                            "flags": sorted(flags),
                        }
                    )
    return pairings


def load_policy(checkpoint: str) -> Any:
    official_config, params = restore_official_checkpoint(Path(checkpoint))
    return official_policy(params, official_config)


def evaluate_pairing(
    *,
    mode_policy: Any,
    partner_policy: Any,
    config: RunConfig,
    root_key: Any,
    episodes: int,
) -> np.ndarray:
    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    eval_pairing = _official_symbol(
        "overcooked_v2_experiments.eval.evaluate", "eval_pairing"
    )
    outputs = eval_pairing(
        PolicyPairing(mode_policy, partner_policy),
        config.environment.layout,
        root_key,
        env_kwargs=dict(_env_kwargs(config)),
        num_seeds=episodes,
        no_viz=True,
    )
    values = np.asarray(
        [outputs[f"seed-{index}"].total_reward for index in range(episodes)],
        dtype=np.float64,
    )
    if values.shape != (episodes,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Official evaluator returned an invalid episode vector.")
    return values


def cell_mean_grid(pairings: list[dict], cells: dict[tuple[str, str], np.ndarray]) -> dict:
    grid: dict[str, dict[str, float]] = {}
    for pairing in pairings:
        key = (pairing["mode_run_id"], pairing["partner_run_id"])
        grid.setdefault(pairing["mode_run_id"], {})[
            pairing["partner_run_id"]
        ] = float(cells[key].mean())
    return grid


def gamma_compat(pairings: list[dict], cells: dict[tuple[str, str], np.ndarray]) -> dict:
    """Full-episode Gamma_compat = V_Z - V_fix with partner-type granularity."""
    means = {
        (p["mode_run_id"], p["partner_run_id"]): float(cells[k].mean())
        for p, k in zip(
            pairings,
            ((p["mode_run_id"], p["partner_run_id"]) for p in pairings),
        )
    }
    partner_types = sorted({p["partner_type"] for p in pairings})
    mode_runs = sorted({p["mode_run_id"] for p in pairings})

    per_partner_best = {}
    for partner_type in partner_types:
        type_pairs = [p for p in pairings if p["partner_type"] == partner_type]
        grouped: dict[str, list[float]] = {}
        for pairing in type_pairs:
            grouped.setdefault(pairing["partner_run_id"], []).append(
                means[(pairing["mode_run_id"], pairing["partner_run_id"])]
            )
        per_partner_best[partner_type] = {
            partner_run: max(values) for partner_run, values in grouped.items()
        }
    v_z = float(np.mean([v for d in per_partner_best.values() for v in d.values()]))

    all_means = [
        means[(p["mode_run_id"], p["partner_run_id"])] for p in pairings
    ]
    v_fix = max(
        float(np.mean([m for m, p in zip(all_means, pairings) if p["mode_run_id"] == run]))
        for run in mode_runs
    )
    return {
        "v_z": v_z,
        "v_fix": v_fix,
        "gamma_compat": v_z - v_fix,
        "per_partner_type_best": per_partner_best,
    }


def contamination_contrast(pairings: list[dict], cells: dict[tuple[str, str], np.ndarray]) -> dict:
    identity = []
    disjoint = []
    for pairing in pairings:
        values = cells[(pairing["mode_run_id"], pairing["partner_run_id"])]
        if "identity_match" in pairing["flags"]:
            identity.append(float(values.mean()))
        else:
            disjoint.append(float(values.mean()))
    return {
        "identity_match_mean": float(np.mean(identity)) if identity else None,
        "run_disjoint_mean": float(np.mean(disjoint)) if disjoint else None,
        "contrast": (
            float(np.mean(identity) - np.mean(disjoint))
            if identity and disjoint
            else None
        ),
    }


def run_level_bootstrap(values: np.ndarray, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_panel(args: argparse.Namespace) -> None:
    import jax

    validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifest = load_panel_manifest(args.manifest)
    if manifest["layout"] != config.environment.layout:
        raise RuntimeError("Panel manifest layout differs from the config layout.")
    pairings = build_pairings(manifest)
    if args.merge_shards:
        merge_shards(pairings, Path(args.output))
        return
    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            raise RuntimeError("shard-index out of range")
        block = (len(pairings) + args.shard_count - 1) // args.shard_count
        pairings = pairings[args.shard_index * block : (args.shard_index + 1) * block]
        if not pairings:
            raise RuntimeError("empty shard slice")
    episodes = int(args.pilot_episodes) if args.pilot_episodes else OFFICIAL_EPISODES
    root_key = jax.random.PRNGKey(OFFICIAL_EVALUATION_ROOT_SEED)

    policies: dict[str, Any] = {}
    cells: dict[tuple[str, str], np.ndarray] = {}
    for pairing in pairings:
        for role in ("mode", "partner"):
            run_id = pairing[f"{role}_run_id"]
            if run_id not in policies:
                policies[run_id] = load_policy(pairing[f"{role}_checkpoint"])
        key = (pairing["mode_run_id"], pairing["partner_run_id"])
        cells[key] = evaluate_pairing(
            mode_policy=policies[pairing["mode_run_id"]],
            partner_policy=policies[pairing["partner_run_id"]],
            config=config,
            root_key=root_key,
            episodes=episodes,
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pairing in pairings:
        values = cells[(pairing["mode_run_id"], pairing["partner_run_id"])]
        for index, value in enumerate(values):
            rows.append({**pairing, "episode_index": index, "raw_return": float(value)})
    (output / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    write_summary(pairings, cells, episodes, output)


def write_summary(
    pairings: list[dict],
    cells: dict[tuple[str, str], np.ndarray],
    episodes: int,
    output: Path,
) -> None:
    gamma = gamma_compat(pairings, cells)
    contrast = contamination_contrast(pairings, cells)
    all_values = np.concatenate(list(cells.values()))
    summary = {
        "episodes_per_pairing": episodes,
        "pairings": len(pairings),
        "delta_min": OFFICIAL_CORRECT_DELIVERY_REWARD,
        **gamma,
        "contamination": contrast,
        "overall_mean_ci95": list(run_level_bootstrap(all_values, BOOTSTRAP_REPLICATES)),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def merge_shards(pairings: list[dict], output: Path) -> None:
    """Combine shard row files into the final rows.json and summary.json."""
    shard_rows = []
    for shard_file in sorted(output.glob("shard_*/rows.json")):
        shard_rows.extend(json.loads(shard_file.read_text(encoding="utf-8")))
    if not shard_rows:
        raise RuntimeError("no shard rows found to merge")
    episodes = max(int(row["episode_index"]) for row in shard_rows) + 1
    seen = set()
    for row in shard_rows:
        key = (
            row["mode_run_id"],
            row["partner_run_id"],
            int(row["episode_index"]),
        )
        if key in seen:
            raise RuntimeError("duplicate row detected across shards")
        seen.add(key)
    expected_rows = len(pairings) * episodes
    if len(shard_rows) != expected_rows:
        raise RuntimeError(
            f"merged row count {len(shard_rows)} differs from expected {expected_rows}"
        )
    (output / "rows.json").write_text(json.dumps(shard_rows), encoding="utf-8")
    cells: dict[tuple[str, str], np.ndarray] = {}
    for pairing in pairings:
        key = (pairing["mode_run_id"], pairing["partner_run_id"])
        cells[key] = np.array(
            [
                row["raw_return"]
                for row in shard_rows
                if row["mode_run_id"] == key[0] and row["partner_run_id"] == key[1]
            ],
            dtype=np.float64,
        )
        if cells[key].shape != (episodes,):
            raise RuntimeError(f"cell {key} has an invalid episode count")
    write_summary(pairings, cells, episodes, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S1 run-disjoint opportunity panel (exploration track)."
    )
    parser.add_argument("--config", required=True, help="Formal run config yaml.")
    parser.add_argument("--manifest", required=True, help="Panel manifest json.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--pilot-episodes",
        type=int,
        default=0,
        help="Override episodes per pairing for the pilot stage.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shards.")
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard_*/rows.json under --output into final outputs.",
    )
    args = parser.parse_args()
    if args.pilot_episodes and args.pilot_episodes <= 0:
        raise SystemExit("pilot-episodes must be positive")
    run_panel(args)


if __name__ == "__main__":
    main()
