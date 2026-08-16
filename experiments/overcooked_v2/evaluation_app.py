"""Common-partner and population-matrix evaluation for CETR-ZSC."""

from __future__ import annotations

import argparse
import csv
from itertools import permutations
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.cetr_zsc.config import (
    FORMAL_PEAK_MEMORY_LIMIT_BYTES,
    FORMAL_METHOD_LABEL,
    METHOD_VERSION,
    OFFICIAL_BASELINE_METHODS,
    OFFICIAL_TRAINING_RUN_COUNT,
    load_config,
)
from src.cetr_zsc.manifest import load_partner_manifest, normalized_mechanism
from src.cetr_zsc.resources import ResourceLedger, peak_device_memory_bytes
from src.cetr_zsc.storage import ensure_run_identity, read_json, write_json

from .deployment import load_deployment
from .official_policy import OfficialCetrPolicy


POLICY_MANIFEST_VERSION = 2
EVALUATION_SCHEMA_VERSION = 3
PAPER_MATRIX_ROOT_SEED = 42
FORMAL_COMMON_EGO_RUNS = OFFICIAL_TRAINING_RUN_COUNT
FORMAL_COMMON_PARTNER_RUNS = 16
FORMAL_EVALUATION_EPISODES = 500
FORMAL_COMMON_MECHANISM_COUNTS = {"sp": 4, "sa": 4, "op": 4, "fcp": 4}
COMMON_PARTNER_KEY_SCHEDULE = (
    "fold_in_root_by_ego_partner_then_role_then_split_episode_keys"
)
POPULATION_KEY_SCHEDULE = "split_root_sp_xp_then_split_fixed_ordered_cells"
PAPER_MATRIX_METHODS = (
    "sp",
    "state-augmented",
    "op",
    "fcp",
    FORMAL_METHOD_LABEL,
)
PAPER_TABLE_VALUES = {
    "grounded_coord_ring": {
        "sp": ("164±10", "-12±75", "175±28"),
        "state-augmented": ("165±7", "-16±65", "183±26"),
        "op": ("121±36", "-9±21", "131±37"),
        "fcp": ("86±34", "6±46", "79±29"),
    },
    "test_time_simple": {
        "sp": ("145±22", "-81±99", "220±26"),
        "state-augmented": ("161±18", "-55±103", "230±53"),
        "op": ("121±37", "-3±51", "131±50"),
        "fcp": ("35±44", "6±29", "25±47"),
    },
    "test_time_wide": {
        "sp": ("194±12", "-30±96", "220±31"),
        "state-augmented": ("137±82", "-12±72", "142±81"),
        "op": ("175±54", "-11±42", "193±61"),
        "fcp": ("95±56", "23±40", "68±50"),
    },
}


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_policy_manifest(path: str | Path, *, layout: str) -> Mapping[str, Any]:
    source = Path(path).resolve()
    manifest = read_json(source)
    if (
        manifest.get("version") != POLICY_MANIFEST_VERSION
        or manifest.get("layout") != layout
        or manifest.get("policy_kind")
        not in {"cetr_deployment", "official_checkpoint"}
    ):
        raise ValueError("Policy manifest identity differs.")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Policy manifest has no runs.")
    if [int(row.get("run_index", -1)) for row in runs] != list(range(len(runs))):
        raise ValueError("Policy manifest run indexes must be contiguous from zero.")
    return manifest


def build_policy_manifest(args: argparse.Namespace) -> None:
    kind = str(args.policy_kind)
    paths = tuple(Path(value).resolve() for value in args.policy)
    if len(paths) != int(args.run_count):
        raise ValueError("Policy manifest run count differs.")

    source_manifest = None
    lineage: list[Mapping[str, Any]] = []
    if kind == "official_checkpoint":
        source_value = getattr(args, "training_lineage_manifest", None)
        if source_value is None:
            raise ValueError(
                "Official policy populations require --training-lineage-manifest."
            )
        source_path = Path(source_value).resolve()
        source_manifest = _load_policy_manifest(source_path, layout=str(args.layout))
        if (
            source_manifest.get("method") != str(args.method)
            or source_manifest.get("policy_kind") != kind
            or len(source_manifest["runs"]) != len(paths)
            or [Path(row["policy"]).resolve() for row in source_manifest["runs"]]
            != list(paths)
        ):
            raise ValueError("Baseline training lineage does not match the policy population.")
        lineage = list(source_manifest.get("training_lineage", ()))

    runs = []
    for index, path in enumerate(paths):
        if kind == "cetr_deployment":
            deployment = load_deployment(path)
            if deployment.config.environment.layout != args.layout:
                raise ValueError("CETR deployment layout differs.")
            identity = read_json(path / "deployment_bundle.json")
            run_id = deployment.ego_run_id
            resource_ledger = None
        else:
            source_run = source_manifest["runs"][index]
            identity = source_run["identity"]
            run_id = source_run["run_id"]
            resource_ledger = source_run.get("resource_ledger")
        runs.append(
            {
                "run_index": index,
                "run_id": run_id,
                "policy": str(path),
                "identity": identity,
                "resource_ledger": resource_ledger,
            }
        )

    write_json(
        args.output,
        {
            "version": POLICY_MANIFEST_VERSION,
            "method": str(args.method),
            "layout": str(args.layout),
            "policy_kind": kind,
            "runs": runs,
            "training_lineage": lineage,
            "source_training_manifest": (
                None
                if source_manifest is None
                else {"path": str(Path(args.training_lineage_manifest).resolve())}
            ),
        },
    )


def _load_policy(row: Mapping[str, Any], kind: str) -> Any:
    if kind == "cetr_deployment":
        return OfficialCetrPolicy(load_deployment(row["policy"]))
    from .official_adapter import official_policy, restore_official_checkpoint

    config, params = restore_official_checkpoint(row["policy"])
    return official_policy(params, config)


def _resource_ledger(*, rows: int, episode_steps: int, started: float) -> ResourceLedger:
    import jax

    elapsed = time.perf_counter() - started
    return ResourceLedger(
        evaluation_steps=int(rows) * int(episode_steps),
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
        peak_memory_bytes=peak_device_memory_bytes(),
    )


def _run_common_partner(
    args: argparse.Namespace,
    *,
    config: Any,
    started: float,
    policy_manifest: Mapping[str, Any],
    policy_manifest_path: Path,
) -> None:
    import jax

    from .official_adapter import (
        VectorEnvironment,
        official_pairing_rollouts,
        official_policy,
        restore_official_checkpoint,
    )

    partner_manifest_path = Path(args.partner_manifest).resolve()
    partner_manifest = load_partner_manifest(
        partner_manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_file_check),
    )
    partners = partner_manifest.by_role(str(args.partner_role))
    if not partners:
        raise ValueError("Evaluation partner role is empty.")
    if config.run_kind == "formal":
        if str(args.partner_role) != "confirmatory":
            raise ValueError("Formal common-partner evaluation uses the confirmatory panel.")
        counts: dict[str, int] = {}
        for run in partners:
            mechanism = normalized_mechanism(run.generation_mechanism)
            counts[mechanism] = counts.get(mechanism, 0) + 1
        if counts != FORMAL_COMMON_MECHANISM_COUNTS:
            raise ValueError(
                "Formal confirmatory evaluation requires exactly four SP, SA, "
                "OP, and FCP parents."
            )

    environment = VectorEnvironment.create(config).environment
    ego_policies = [
        _load_policy(row, policy_manifest["policy_kind"])
        for row in policy_manifest["runs"]
    ]
    partner_policies = []
    for run in partners:
        partner_config, partner_params = restore_official_checkpoint(run.checkpoint)
        partner_policies.append(official_policy(partner_params, partner_config))

    root_seed = (
        config.evaluation.evaluation_seed
        if getattr(args, "seed", None) is None
        else int(args.seed)
    )
    root = jax.random.PRNGKey(root_seed)
    rows = []
    roles = (0, 1) if config.evaluation.evaluate_both_roles else (0,)
    for ego_index, ego in enumerate(ego_policies):
        for partner_index, (partner_run, partner) in enumerate(
            zip(partners, partner_policies, strict=True)
        ):
            for role in roles:
                left, right = (ego, partner) if role == 0 else (partner, ego)
                pairing_key = jax.random.fold_in(
                    jax.random.fold_in(root, ego_index * 10_000 + partner_index), role
                )
                rollouts, episode_keys = official_pairing_rollouts(
                    left_policy=left,
                    right_policy=right,
                    environment=environment,
                    root_key=pairing_key,
                    episodes=config.evaluation.episodes_per_pairing,
                )
                returns = np.asarray(rollouts.total_reward, dtype=np.float64)
                keys = np.asarray(episode_keys, dtype=np.uint32)
                for episode, (value, episode_key) in enumerate(
                    zip(returns, keys, strict=True)
                ):
                    rows.append(
                        {
                            "evaluation_mode": "common_partner",
                            "layout": config.environment.layout,
                            "method": policy_manifest["method"],
                            "ego_run_index": ego_index,
                            "ego_run_id": policy_manifest["runs"][ego_index]["run_id"],
                            "partner_run_index": partner_index,
                            "partner_run_id": partner_run.run_id,
                            "partner_mechanism": partner_run.generation_mechanism,
                            "ego_role": role,
                            "episode_index": episode,
                            "environment_key": [int(word) for word in episode_key],
                            "raw_return": float(value),
                        }
                    )

    output = Path(args.output).resolve()
    identity = {
        "stage": "official-evaluation",
        "evaluation_mode": "common_partner",
        "layout": config.environment.layout,
        "method": policy_manifest["method"],
        "policy_manifest": {"path": str(policy_manifest_path)},
        "partner_manifest": {"path": str(partner_manifest_path)},
        "partner_role": str(args.partner_role),
        "root_seed": root_seed,
        "key_schedule": COMMON_PARTNER_KEY_SCHEDULE,
        "observation_protocol": "default_non_permuted",
    }
    ensure_run_identity(output, identity)
    raw = output / "episode_returns.jsonl"
    _write_jsonl(raw, rows)
    ledger = _resource_ledger(
        rows=len(rows), episode_steps=config.environment.episode_steps, started=started
    )
    if (
        config.run_kind == "formal"
        and ledger.peak_memory_bytes >= FORMAL_PEAK_MEMORY_LIMIT_BYTES
    ):
        raise RuntimeError("Formal evaluation peak device memory must remain below 40,000 MiB.")
    write_json(
        output / "evaluation_summary.json",
        {
            "version": EVALUATION_SCHEMA_VERSION,
            "artifact_type": "cetr_raw_evaluation",
            "evaluation_mode": "common_partner",
            "layout": config.environment.layout,
            "method": policy_manifest["method"],
            "mean_return": float(np.mean([row["raw_return"] for row in rows])),
            "episode_count": len(rows),
            "episodes_per_pairing": config.evaluation.episodes_per_pairing,
            "evaluate_both_roles": config.evaluation.evaluate_both_roles,
            "root_seed": root_seed,
            "key_schedule": COMMON_PARTNER_KEY_SCHEDULE,
            "observation_protocol": "default_non_permuted",
            "policy_manifest": {"path": str(policy_manifest_path)},
            "partner_manifest": {"path": str(partner_manifest_path)},
            "partner_role": str(args.partner_role),
            "raw": {"path": str(raw)},
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


def _population_statistics(cube: np.ndarray) -> Mapping[str, Any]:
    cell = np.mean(cube, axis=2)
    diagonal = np.diag(cell)
    xp_rows = np.asarray(
        [np.mean(np.delete(cell[index], index)) for index in range(cell.shape[0])]
    )
    gaps = diagonal - xp_rows
    off_diagonal = cell[~np.eye(cell.shape[0], dtype=bool)]
    return {
        "sp_point": float(np.mean(diagonal)),
        "xp_point": float(np.mean(xp_rows)),
        "gap_point": float(np.mean(diagonal) - np.mean(xp_rows)),
        "sd_sp_diag10": float(np.std(diagonal, ddof=1)),
        "sd_xp_rows10": float(np.std(xp_rows, ddof=1)),
        "sd_gap_rows10": float(np.std(gaps, ddof=1)),
        "sd_xp_cells90": float(np.std(off_diagonal, ddof=1)),
        "sp_diagonal": diagonal.tolist(),
        "xp_rows": xp_rows.tolist(),
        "gap_rows": gaps.tolist(),
        "cell_means": cell.tolist(),
    }


def _run_population_matrix(
    args: argparse.Namespace,
    *,
    config: Any,
    started: float,
    left_manifest: Mapping[str, Any],
    left_manifest_path: Path,
    right_manifest: Mapping[str, Any],
    right_manifest_path: Path,
) -> None:
    import jax

    from .official_adapter import VectorEnvironment, official_pairing_rollouts

    if config.run_kind != "formal":
        raise ValueError("Population-matrix evaluation uses the formal config.")
    left_runs = left_manifest["runs"]
    right_runs = right_manifest["runs"]
    if (
        len(left_runs) != 10
        or len(right_runs) != 10
        or left_manifest["method"] != right_manifest["method"]
        or [row["run_id"] for row in left_runs]
        != [row["run_id"] for row in right_runs]
        or [Path(row["policy"]).resolve() for row in left_runs]
        != [Path(row["policy"]).resolve() for row in right_runs]
    ):
        raise ValueError("Paper population matrix requires one aligned ten-run method.")
    root_seed = int(args.seed)
    if root_seed != PAPER_MATRIX_ROOT_SEED:
        raise ValueError("Paper population matrix uses evaluation root seed 42.")

    environment = VectorEnvironment.create(config).environment
    left_policies = [
        _load_policy(row, left_manifest["policy_kind"])
        for row in left_runs
    ]
    right_policies = [
        _load_policy(row, right_manifest["policy_kind"])
        for row in right_runs
    ]
    sp_root, xp_root = jax.random.split(jax.random.PRNGKey(root_seed), 2)
    xp_pairs = tuple(permutations(range(10), 2))
    sp_pairs = tuple((index, index) for index in range(10))
    xp_keys = jax.random.split(xp_root, len(xp_pairs))
    sp_keys = jax.random.split(sp_root, len(sp_pairs))

    rows = []
    for cell_type, pairs, keys in (
        ("xp", xp_pairs, xp_keys),
        ("sp", sp_pairs, sp_keys),
    ):
        for cell_index, ((left_index, right_index), key) in enumerate(
            zip(pairs, keys, strict=True)
        ):
            rollouts, episode_keys = official_pairing_rollouts(
                left_policy=left_policies[left_index],
                right_policy=right_policies[right_index],
                environment=environment,
                root_key=key,
                episodes=config.evaluation.episodes_per_pairing,
            )
            returns = np.asarray(rollouts.total_reward, dtype=np.float64)
            keys_for_episodes = np.asarray(episode_keys, dtype=np.uint32)
            for episode, (value, episode_key) in enumerate(
                zip(returns, keys_for_episodes, strict=True)
            ):
                rows.append(
                    {
                        "evaluation_mode": "population_matrix",
                        "layout": config.environment.layout,
                        "method": left_manifest["method"],
                        "cell_type": cell_type,
                        "cell_index": cell_index,
                        "left_run_index": left_index,
                        "left_run_id": left_runs[left_index]["run_id"],
                        "right_run_index": right_index,
                        "right_run_id": right_runs[right_index]["run_id"],
                        "episode_index": episode,
                        "environment_key": [int(word) for word in episode_key],
                        "raw_return": float(value),
                    }
                )

    cube = np.empty((10, 10, config.evaluation.episodes_per_pairing), dtype=np.float64)
    for row in rows:
        cube[
            int(row["left_run_index"]),
            int(row["right_run_index"]),
            int(row["episode_index"]),
        ] = float(row["raw_return"])
    statistics = _population_statistics(cube)

    output = Path(args.output).resolve()
    identity = {
        "stage": "official-evaluation",
        "evaluation_mode": "population_matrix",
        "layout": config.environment.layout,
        "method": left_manifest["method"],
        "left_policy_manifest": {"path": str(left_manifest_path)},
        "right_policy_manifest": {"path": str(right_manifest_path)},
        "root_seed": root_seed,
        "key_schedule": POPULATION_KEY_SCHEDULE,
        "observation_protocol": "default_non_permuted",
    }
    ensure_run_identity(output, identity)
    raw = output / "episode_returns.jsonl"
    _write_jsonl(raw, rows)
    ledger = _resource_ledger(
        rows=len(rows), episode_steps=config.environment.episode_steps, started=started
    )
    if ledger.peak_memory_bytes >= FORMAL_PEAK_MEMORY_LIMIT_BYTES:
        raise RuntimeError("Formal evaluation peak device memory must remain below 40,000 MiB.")
    population_summary = output / "population_matrix_summary.json"
    write_json(
        population_summary,
        {
            "version": 1,
            "artifact_type": "cetr_population_matrix_summary",
            "layout": config.environment.layout,
            "method": left_manifest["method"],
            "cube_shape": list(cube.shape),
            "episodes_per_cell": config.evaluation.episodes_per_pairing,
            "root_seed": root_seed,
            "key_schedule": POPULATION_KEY_SCHEDULE,
            "observation_protocol": "default_non_permuted",
            **statistics,
            "raw": {"path": str(raw)},
        },
    )
    write_json(
        output / "evaluation_summary.json",
        {
            "version": EVALUATION_SCHEMA_VERSION,
            "artifact_type": "cetr_raw_evaluation",
            "evaluation_mode": "population_matrix",
            "layout": config.environment.layout,
            "method": left_manifest["method"],
            "mean_return": float(np.mean(cube)),
            "episode_count": len(rows),
            "episodes_per_pairing": config.evaluation.episodes_per_pairing,
            "root_seed": root_seed,
            "key_schedule": POPULATION_KEY_SCHEDULE,
            "observation_protocol": "default_non_permuted",
            "left_policy_manifest": {"path": str(left_manifest_path)},
            "right_policy_manifest": {"path": str(right_manifest_path)},
            "raw": {"path": str(raw)},
            "population_summary": {"path": str(population_summary)},
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


def run_evaluation(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    from .official_adapter import validate_official_runtime

    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal":
        import jax

        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if len(devices) != 1:
            raise RuntimeError("Formal evaluation requires exactly one visible CUDA device.")
        validate_official_runtime()
    mode = str(getattr(args, "evaluation_mode", "common_partner"))
    if mode == "common_partner":
        if not getattr(args, "policy_manifest", None) or not getattr(
            args, "partner_manifest", None
        ):
            raise ValueError(
                "Common-partner evaluation requires policy and partner manifests."
            )
        policy_manifest_path = Path(args.policy_manifest).resolve()
        policy_manifest = _load_policy_manifest(
            policy_manifest_path, layout=config.environment.layout
        )
        if (
            config.run_kind == "formal"
            and len(policy_manifest["runs"]) != config.evaluation.minimum_ego_runs
        ):
            raise ValueError("Formal evaluation requires ten independent ego runs.")
        _run_common_partner(
            args,
            config=config,
            started=started,
            policy_manifest=policy_manifest,
            policy_manifest_path=policy_manifest_path,
        )
        return
    if mode == "population_matrix":
        if (
            not getattr(args, "left_policy_manifest", None)
            or not getattr(args, "right_policy_manifest", None)
            or getattr(args, "seed", None) is None
        ):
            raise ValueError(
                "Population-matrix evaluation requires left/right manifests and seed."
            )
        left_path = Path(args.left_policy_manifest).resolve()
        right_path = Path(args.right_policy_manifest).resolve()
        _run_population_matrix(
            args,
            config=config,
            started=started,
            left_manifest=_load_policy_manifest(
                left_path, layout=config.environment.layout
            ),
            left_manifest_path=left_path,
            right_manifest=_load_policy_manifest(
                right_path, layout=config.environment.layout
            ),
            right_manifest_path=right_path,
        )
        return
    raise ValueError(f"Unknown evaluation mode: {mode}")


def _node_matrix(
    rows: list[Mapping[str, Any]],
) -> tuple[np.ndarray, list[int], list[str]]:
    egos = sorted({int(row["ego_run_index"]) for row in rows})
    partners = sorted({str(row["partner_run_id"]) for row in rows})
    matrix = np.empty((len(egos), len(partners)), dtype=np.float64)
    for i, ego in enumerate(egos):
        for j, partner in enumerate(partners):
            values = [
                float(row["raw_return"])
                for row in rows
                if int(row["ego_run_index"]) == ego
                and str(row["partner_run_id"]) == partner
            ]
            matrix[i, j] = np.mean(values)
    return matrix, egos, partners


def _bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, replicates: int, seed: int
) -> Mapping[str, Any]:
    if left.shape != right.shape:
        raise ValueError("Paired node matrices differ.")
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        ego = rng.integers(0, left.shape[0], size=left.shape[0])
        partner = rng.integers(0, left.shape[1], size=left.shape[1])
        draws[index] = np.mean(
            left[np.ix_(ego, partner)] - right[np.ix_(ego, partner)]
        )
    return {
        "estimate": float(np.mean(left - right)),
        "interval_95": [float(v) for v in np.quantile(draws, (0.025, 0.975))],
        "one_sided_lcb": float(np.quantile(draws, 0.05)),
    }


def _validate_common_rows(
    rows: list[Mapping[str, Any]],
    *,
    method: str,
    layout: str,
    ego_count: int,
    partner_ids: set[str],
    episodes: int,
) -> None:
    keys = {
        (
            int(row["ego_run_index"]),
            str(row["partner_run_id"]),
            int(row["ego_role"]),
            int(row["episode_index"]),
        )
        for row in rows
    }
    expected = ego_count * len(partner_ids) * 2 * episodes
    if (
        len(rows) != expected
        or len(keys) != expected
        or {int(row["ego_run_index"]) for row in rows} != set(range(ego_count))
        or {str(row["partner_run_id"]) for row in rows} != partner_ids
        or {int(row["ego_role"]) for row in rows} != {0, 1}
        or {int(row["episode_index"]) for row in rows} != set(range(episodes))
        or any(len(row.get("environment_key", ())) != 2 for row in rows)
        or any(
            row.get("evaluation_mode") != "common_partner"
            or row.get("method") != method
            or row.get("layout") != layout
            for row in rows
        )
    ):
        raise ValueError("Common-partner raw matrix is incomplete or duplicated.")


def _lineage_sets(policy_manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    rows = list(policy_manifest.get("training_lineage", ()))
    rows.extend(
        {
            "parent_training_run_id": run["identity"].get("parent_training_run_id"),
            "co_training_group_id": run["identity"].get("co_training_group_id"),
        }
        for run in policy_manifest["runs"]
        if isinstance(run.get("identity"), Mapping)
    )
    parents = {
        str(row["parent_training_run_id"])
        for row in rows
        if row.get("parent_training_run_id") is not None
    }
    groups = {
        str(row["co_training_group_id"])
        for row in rows
        if row.get("co_training_group_id") is not None
    }
    return parents, groups


def summarize_evaluations(args: argparse.Namespace) -> None:
    sources = {}
    matrices = {}
    ego_indexes = partner_ids = None
    partner_manifest_path = None
    layout = None
    root_seed = None
    key_schedule = None
    observation_protocol = None
    environment_keys = None
    for value in args.evaluation:
        method, raw_path = value.split("=", 1)
        if method in matrices:
            raise ValueError(f"Duplicate evaluation method: {method}")
        directory = Path(raw_path).resolve()
        summary = read_json(directory / "evaluation_summary.json")
        if (
            summary.get("artifact_type") != "cetr_raw_evaluation"
            or summary.get("evaluation_mode") != "common_partner"
            or summary.get("method") != method
        ):
            raise ValueError("Evaluation summary identity differs.")
        current_partner_manifest_path = Path(
            summary["partner_manifest"]["path"]
        ).resolve()
        partner_manifest = load_partner_manifest(
            current_partner_manifest_path,
            expected_layout=str(summary["layout"]),
            verify_files=False,
        )
        confirmatory = partner_manifest.by_role("confirmatory")
        if len(confirmatory) != FORMAL_COMMON_PARTNER_RUNS:
            raise ValueError("Common-partner summary requires 16 confirmatory partners.")
        current_partner_ids = {row.run_id for row in confirmatory}
        policy_manifest = _load_policy_manifest(
            summary["policy_manifest"]["path"], layout=str(summary["layout"])
        )
        rows = _read_jsonl(Path(summary["raw"]["path"]))
        _validate_common_rows(
            rows,
            method=method,
            layout=str(summary["layout"]),
            ego_count=FORMAL_COMMON_EGO_RUNS,
            partner_ids=current_partner_ids,
            episodes=FORMAL_EVALUATION_EPISODES,
        )
        matrix, current_egos, current_partners = _node_matrix(rows)
        current_layout = str(summary["layout"])
        current_root_seed = int(summary["root_seed"])
        current_key_schedule = str(summary["key_schedule"])
        current_observation_protocol = str(summary["observation_protocol"])
        current_environment_keys = {
            (
                int(row["ego_run_index"]),
                str(row["partner_run_id"]),
                int(row["ego_role"]),
                int(row["episode_index"]),
            ): tuple(int(word) for word in row["environment_key"])
            for row in rows
        }
        if ego_indexes is None:
            ego_indexes = current_egos
            partner_ids = current_partners
            partner_manifest_path = current_partner_manifest_path
            layout = current_layout
            root_seed = current_root_seed
            key_schedule = current_key_schedule
            observation_protocol = current_observation_protocol
            environment_keys = current_environment_keys
        elif (
            current_egos != ego_indexes
            or current_partners != partner_ids
            or current_partner_manifest_path != partner_manifest_path
            or current_layout != layout
            or current_root_seed != root_seed
            or current_key_schedule != key_schedule
            or current_observation_protocol != observation_protocol
            or current_environment_keys != environment_keys
        ):
            raise ValueError(
                "Common-partner run, partner, layout, or key schedule differs across methods."
            )

        if method in OFFICIAL_BASELINE_METHODS:
            policy_parents, policy_groups = _lineage_sets(policy_manifest)
            confirmatory_parents = {
                row.parent_training_run_id for row in confirmatory
            }
            confirmatory_groups = {
                row.co_training_group_id
                for row in confirmatory
                if row.co_training_group_id is not None
            }
            if policy_parents & confirmatory_parents or policy_groups & confirmatory_groups:
                raise ValueError(
                    "Baseline ego training lineage overlaps the confirmatory panel."
                )
        matrices[method] = matrix
        sources[method] = {
            "path": str(directory),
            "policy_manifest": str(Path(summary["policy_manifest"]["path"]).resolve()),
        }

    if FORMAL_METHOD_LABEL not in matrices:
        raise ValueError(f"Summary requires {FORMAL_METHOD_LABEL}.")
    observed_baselines = set(matrices) - {FORMAL_METHOD_LABEL}
    if observed_baselines != set(OFFICIAL_BASELINE_METHODS):
        raise ValueError("Common-partner summary requires all five registered baselines.")
    baseline_methods = list(OFFICIAL_BASELINE_METHODS)
    strongest = max(
        baseline_methods, key=lambda name: float(np.mean(matrices[name]))
    )
    contrasts = {
        name: _bootstrap_difference(
            matrices[FORMAL_METHOD_LABEL],
            matrices[name],
            replicates=int(args.bootstrap_replicates),
            seed=int(args.seed) + index,
        )
        for index, name in enumerate(baseline_methods)
    }
    all_baselines_gate = bool(
        all(float(value["one_sided_lcb"]) > 0.0 for value in contrasts.values())
        and all(float(value["estimate"]) >= 20.0 for value in contrasts.values())
    )
    write_json(
        args.output,
        {
            "version": 3,
            "artifact_type": "cetr_official_summary",
            "method": METHOD_VERSION,
            "layout": layout,
            "ego_run_indexes": ego_indexes,
            "partner_run_ids": partner_ids,
            "partner_manifest": {"path": str(partner_manifest_path)},
            "evaluation_root_seed": root_seed,
            "key_schedule": key_schedule,
            "observation_protocol": observation_protocol,
            "means": {name: float(np.mean(value)) for name, value in matrices.items()},
            "strongest_baseline": strongest,
            "cetr_active_vs_strongest": contrasts[strongest],
            "cetr_active_vs_each_baseline": contrasts,
            "all_baselines_material_superiority_gate": all_baselines_gate,
            "testing_rule": (
                "intersection_union: every registered baseline contrast must have "
                "one-sided LCB>0 and estimate>=20"
            ),
            "bootstrap_seed": int(args.seed),
            "sources": sources,
        },
    )


def _validated_population_cube(
    directory: Path, method: str
) -> tuple[Any, np.ndarray, Mapping[tuple[int, int, int], tuple[int, int]]]:
    summary = read_json(directory / "evaluation_summary.json")
    population = read_json(Path(summary["population_summary"]["path"]))
    rows = _read_jsonl(Path(summary["raw"]["path"]))
    keys = {
        (
            int(row["left_run_index"]),
            int(row["right_run_index"]),
            int(row["episode_index"]),
        )
        for row in rows
    }
    expected = 10 * 10 * 500
    if (
        summary.get("evaluation_mode") != "population_matrix"
        or summary.get("method") != method
        or summary.get("key_schedule") != POPULATION_KEY_SCHEDULE
        or summary.get("observation_protocol") != "default_non_permuted"
        or population.get("cube_shape") != [10, 10, 500]
        or len(rows) != expected
        or len(keys) != expected
        or {int(row["left_run_index"]) for row in rows} != set(range(10))
        or {int(row["right_run_index"]) for row in rows} != set(range(10))
        or {int(row["episode_index"]) for row in rows} != set(range(500))
        or any(len(row.get("environment_key", ())) != 2 for row in rows)
    ):
        raise ValueError("Population-matrix raw cube is incomplete or duplicated.")
    cube = np.empty((10, 10, 500), dtype=np.float64)
    for row in rows:
        cube[
            int(row["left_run_index"]),
            int(row["right_run_index"]),
            int(row["episode_index"]),
        ] = float(row["raw_return"])
    environment_keys = {
        (
            int(row["left_run_index"]),
            int(row["right_run_index"]),
            int(row["episode_index"]),
        ): tuple(int(word) for word in row["environment_key"])
        for row in rows
    }
    return summary, cube, environment_keys


def summarize_population_matrices(args: argparse.Namespace) -> None:
    directories = {}
    for value in args.evaluation:
        method, raw_path = value.split("=", 1)
        directories[method] = Path(raw_path).resolve()
    supplied_methods = set(directories)
    if supplied_methods not in (
        {FORMAL_METHOD_LABEL},
        set(PAPER_MATRIX_METHODS),
    ):
        raise ValueError(
            "Paper matrix summary requires CETR-active alone or the complete "
            "SP, SA, OP, FCP, and CETR-active reproduction."
        )

    cell_rows = []
    statistics_by_method = {}
    layout = None
    root_seed = None
    environment_keys = None
    for method in (method for method in PAPER_MATRIX_METHODS if method in directories):
        summary, cube, current_environment_keys = _validated_population_cube(
            directories[method], method
        )
        current_layout = str(summary["layout"])
        current_seed = int(summary["root_seed"])
        if layout is None:
            layout, root_seed = current_layout, current_seed
            environment_keys = current_environment_keys
        elif (
            current_layout != layout
            or current_seed != root_seed
            or current_environment_keys != environment_keys
        ):
            raise ValueError(
                "Population matrices differ in layout, seed, or key schedule."
            )
        statistics = _population_statistics(cube)
        statistics_by_method[method] = statistics
        for left_index, current in enumerate(statistics["cell_means"]):
            for right_index, value in enumerate(current):
                cell_rows.append(
                    {
                        "method": method,
                        "left_run_index": left_index,
                        "right_run_index": right_index,
                        "cell_type": "sp" if left_index == right_index else "xp",
                        "mean_raw_return": value,
                    }
                )

    paper_values = PAPER_TABLE_VALUES[str(layout)]
    rows = []
    for method in PAPER_MATRIX_METHODS:
        statistics = statistics_by_method.get(method)
        paper_sp, paper_xp, paper_gap = paper_values.get(
            method, ("n/a", "n/a", "n/a")
        )
        rows.append(
            {
                "method": method,
                "paper_sp_verbatim": paper_sp,
                "reproduced_sp_point": (
                    None if statistics is None else statistics["sp_point"]
                ),
                "paper_xp_verbatim": paper_xp,
                "reproduced_xp_point": (
                    None if statistics is None else statistics["xp_point"]
                ),
                "paper_gap_verbatim": paper_gap,
                "reproduced_gap_point": (
                    None if statistics is None else statistics["gap_point"]
                ),
                "sd_sp_diag10": (
                    None if statistics is None else statistics["sd_sp_diag10"]
                ),
                "sd_xp_rows10": (
                    None if statistics is None else statistics["sd_xp_rows10"]
                ),
                "sd_gap_rows10": (
                    None if statistics is None else statistics["sd_gap_rows10"]
                ),
                "sd_xp_cells90": (
                    None if statistics is None else statistics["sd_xp_cells90"]
                ),
            }
        )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "table2_extension.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells_path = output / "population_cell_means.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)

    markdown = [
        f"# Table 2 extension: {layout}",
        "",
        "| Method | Paper SP | Reproduced SP | Paper XP | Reproduced XP | Paper Gap | Reproduced Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reproduced_sp = row["reproduced_sp_point"]
        reproduced_xp = row["reproduced_xp_point"]
        reproduced_gap = row["reproduced_gap_point"]
        markdown.append(
            "| {method} | {paper_sp} | {reproduced_sp} | {paper_xp} | "
            "{reproduced_xp} | {paper_gap} | {reproduced_gap} |".format(
                method=row["method"],
                paper_sp=row["paper_sp_verbatim"],
                reproduced_sp=(
                    "—" if reproduced_sp is None else f"{reproduced_sp:.6f}"
                ),
                paper_xp=row["paper_xp_verbatim"],
                reproduced_xp=(
                    "—" if reproduced_xp is None else f"{reproduced_xp:.6f}"
                ),
                paper_gap=row["paper_gap_verbatim"],
                reproduced_gap=(
                    "—" if reproduced_gap is None else f"{reproduced_gap:.6f}"
                ),
            )
        )
    markdown.extend(
        (
            "",
            "Paper error terms are copied verbatim; their public aggregation "
            "definition is unavailable. Reproduced dispersion columns are named "
            "explicitly in the CSV. Common-partner results are a different estimand.",
        )
    )
    markdown_path = output / "table2_extension.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(
        output / "population_matrix_summary.json",
        {
            "version": 1,
            "artifact_type": "cetr_population_matrix_comparison_summary",
            "layout": layout,
            "root_seed": root_seed,
            "methods": rows,
            "evaluated_methods": [
                method for method in PAPER_MATRIX_METHODS if method in directories
            ],
            "comparison_mode": (
                "paper_reference"
                if supplied_methods == {FORMAL_METHOD_LABEL}
                else "full_local_reproduction"
            ),
            "table_csv": {"path": str(table_path)},
            "table_markdown": {"path": str(markdown_path)},
            "cell_means_csv": {"path": str(cells_path)},
            "scope": "single_layout_no_full_h1_h2_h3_or_sota_claim",
        },
    )


__all__ = [
    "build_policy_manifest",
    "run_evaluation",
    "summarize_evaluations",
    "summarize_population_matrices",
]
