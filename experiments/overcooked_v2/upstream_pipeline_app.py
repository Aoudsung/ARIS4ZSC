"""Continuous Official-upstream DAG selected entirely by experiment config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from src.delta_zsc.config import (
    OFFICIAL_OP_TOTAL_TIMESTEPS,
    OFFICIAL_SP_TOTAL_TIMESTEPS,
    load_config,
)
from src.delta_zsc.resources import ResourceLedger, parameter_count
from src.delta_zsc.storage import ensure_run_identity, read_json, write_json

from .delta_manifest_app import build_partner_manifest
from .evaluation_app import POLICY_MANIFEST_VERSION
from .official_adapter import restore_official_checkpoint, validate_official_runtime


STATE_AUGMENTED_NUM_ENVS = 128
"""Parallel environments for a state-augmented upstream population.

The Official default is 256 (model/rnn.yaml).  State-augmented is the only
method whose population trains as one ``pmap(vmap(train_jit))`` over all ten
runs at once, so its peak scales with run_count * NUM_ENVS rather than NUM_ENVS
alone -- and its run_count cannot be lowered, because the state-collection pass
asserts that run_count**2 divides ten.  At 256 the wide layout asked for a
single 17.4 GiB block and died on an otherwise empty 46 GiB L40; simple fits at
256 because it carries 39 observation channels against wide's 43.

Halving the environments halves that peak and leaves everything else -- total
timesteps, minibatch count, learning rate schedule -- at the Official values.
It applies to every layout driven through this pipeline, so it is a disclosed
deviation from the Official state-augmented configuration wherever it is used.
Registered 2026-08-12 by user decision, after the panel count itself was raised
from four to ten for the assertion above.
"""

UPSTREAM_ROOT_SEEDS = {
    "support_sp": 11_042,
    "support_op": 11_043,
    "panel_sp": 12_042,
    "panel_op": 12_043,
    "coverage_sa": 13_042,
    "confirmatory_sa": 14_042,
    "baseline_fcp_sources": 15_000,
    "panel_fcp_sources": 16_000,
    "panel_fcp": 17_042,
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _gpu_environment() -> Mapping[str, str]:
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cuda"
    environment["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["WANDB_MODE"] = "disabled"
    environment["HYDRA_FULL_ERROR"] = "1"
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ptxas = (
        Path(sys.prefix)
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
        / "bin"
    )
    if ptxas.is_dir():
        environment["PATH"] = f"{ptxas}:{environment.get('PATH', '')}"
    return environment


def _run_logged(command: Sequence[str], *, output: Path, name: str) -> float:
    write_json(output / f"{name}_command.json", {"argv": list(command)})
    started = time.perf_counter()
    with (output / f"{name}.log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            list(command),
            cwd=_repository_root(),
            env=_gpu_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - started


def _population_keys(root_seed: int, run_count: int) -> list[list[int]]:
    import jax

    keys = jax.random.split(jax.random.PRNGKey(int(root_seed)), int(run_count))
    return [[int(word) for word in key] for key in keys]


def _official_run_root(hydra_root: Path) -> Path:
    roots = tuple(
        path
        for path in (hydra_root / "runs").iterdir()
        if path.is_dir()
    )
    if len(roots) != 1:
        raise RuntimeError(
            f"Official job at {hydra_root} produced {len(roots)} run roots."
        )
    return roots[0]


def _link_directory(link: Path, target: Path) -> None:
    """Create one movable directory alias, preserving a matching existing one."""

    resolved = target.resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() != resolved:
            raise ValueError(f"Directory alias {link} points to a different target.")
        return
    relative = Path(os.path.relpath(resolved, start=link.parent.resolve()))
    link.symlink_to(relative, target_is_directory=True)


def _checkpoint_rows(run_directory: Path, checkpoint_count: int) -> list[Mapping[str, Any]]:
    if checkpoint_count == 1:
        return [{"checkpoint_stage": 1.0, "path": str(run_directory / "ckpt_final")}]
    return [
        {"checkpoint_stage": 0.0, "path": str(run_directory / "ckpt_0")},
        {"checkpoint_stage": 0.5, "path": str(run_directory / "ckpt_1")},
        {"checkpoint_stage": 1.0, "path": str(run_directory / "ckpt_final")},
    ]


def _training_steps(method: str) -> int:
    return (
        OFFICIAL_OP_TOTAL_TIMESTEPS
        if method == "op"
        else OFFICIAL_SP_TOTAL_TIMESTEPS
    )


def _run_official_population(
    *,
    method: str,
    layout: str,
    root_seed: int,
    run_count: int,
    checkpoint_count: int,
    parent_prefix: str,
    output: Path,
    co_training_groups: Sequence[str | None] | None = None,
    fcp_population: Path | None = None,
    shared_ledger: Path | None = None,
    external_lineage: Sequence[Mapping[str, Any]] = (),
) -> Path:
    experiment = {
        "sp": "rnn-sp",
        "op": "rnn-op",
        "state-augmented": "rnn-sa",
        "fcp": "rnn-fcp",
    }[method]
    groups = (
        list(co_training_groups)
        if co_training_groups is not None
        else [None] * int(run_count)
    )
    identity = {
        "stage": "official-population",
        "method": method,
        "layout": layout,
        "root_seed": int(root_seed),
        "run_count": int(run_count),
        "checkpoint_count": int(checkpoint_count),
        "parent_prefix": parent_prefix,
        "co_training_groups": groups,
        "fcp_population": None if fcp_population is None else str(fcp_population),
        "shared_ledger": None if shared_ledger is None else str(shared_ledger),
        "external_training_lineage": list(external_lineage),
    }
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(output, identity)
    policy_manifest = output / "policy_manifest.json"
    if (output / "population_training.json").is_file():
        return policy_manifest

    if method == "fcp":
        population_dirs = [
            path
            for path in fcp_population.iterdir()
            if path.is_dir() and path.name.startswith("fcp_")
        ]
        indexed_groups = {
            f"fcp_{index:02d}": group for index, group in enumerate(groups)
        }
        groups = [indexed_groups[path.name] for path in population_dirs]

    attempts = output / "official_hydra"
    attempt_index = len(tuple(attempts.glob("attempt-*")))
    checkpoint_names = (
        ("ckpt_final",)
        if checkpoint_count == 1
        else ("ckpt_0", "ckpt_1", "ckpt_final")
    )
    completed_aliases = all(
        (output / f"run-{index}" / name).is_dir()
        for index in range(int(run_count))
        for name in checkpoint_names
    )
    run_root = None
    if completed_aliases:
        completed_attempt = attempt_index - 1
        command_record = output / f"training-attempt-{completed_attempt:02d}_command.json"
        training_log = output / f"training-attempt-{completed_attempt:02d}.log"
        elapsed = training_log.stat().st_mtime - command_record.stat().st_mtime
    else:
        hydra_root = attempts / f"attempt-{attempt_index:02d}"
        command = [
            sys.executable,
            "-m",
            "experiments.overcooked_v2.official_training",
            f"+experiment={experiment}",
            f"+env={layout}",
            "++env.ENV_KWARGS.indicate_successful_delivery=true",
            f"SEED={int(root_seed)}",
            f"NUM_CHECKPOINTS={int(checkpoint_count)}",
            "VISUALIZE=false",
            "TUNE=false",
            "wandb.WANDB_MODE=disabled",
            f"hydra.run.dir={hydra_root}",
            "hydra.job.chdir=true",
        ]
        if method == "fcp":
            command.append(f"+FCP={fcp_population}")
        else:
            command.append(f"NUM_SEEDS={int(run_count)}")
        if method == "state-augmented":
            command.append(
                f"++model.NUM_ENVS={int(STATE_AUGMENTED_NUM_ENVS)}"
            )
        elapsed = _run_logged(
            command, output=output, name=f"training-attempt-{attempt_index:02d}"
        )
        run_root = _official_run_root(hydra_root)
    keys = _population_keys(root_seed, run_count)
    records = []
    final_paths = []
    for index in range(int(run_count)):
        run_directory = (
            output / f"run-{index}"
            if run_root is None
            else run_root / f"run_{index}"
        )
        # Expose the seed-index path consumed by the DELTA initializer mapping.
        # The Official trainer nests its run directories below a timestamped
        # Hydra root, while the experiment workflow addresses the same
        # parents as ``run-<seed>/ckpt_final``.  One directory link keeps those
        # two views of the same checkpoint aligned without copying artifacts.
        seed_directory = output / f"run-{index}"
        if run_root is not None:
            _link_directory(seed_directory, run_directory)
        checkpoints = _checkpoint_rows(seed_directory, checkpoint_count)
        for row in checkpoints:
            if not Path(row["path"]).is_dir():
                raise FileNotFoundError(row["path"])
        final_path = Path(checkpoints[-1]["path"]).resolve()
        final_paths.append(final_path)
        records.append(
            {
                "run_index": index,
                "parent_training_run_id": f"{parent_prefix}-{index:02d}",
                "co_training_group_id": groups[index],
                "root_seed": int(root_seed),
                "jax_prng_key": keys[index],
                "checkpoints": checkpoints,
            }
        )

    unused_config, first_params = restore_official_checkpoint(final_paths[0])
    del unused_config
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_count = len([item for item in visible.split(",") if item.strip()]) or 1
    shared_steps = (
        40_000 * int(run_count) * int(run_count)
        if method == "state-augmented"
        else 0
    )
    total_ledger = ResourceLedger(
        ego_policy_steps=int(run_count) * _training_steps(method),
        upstream_partner_steps=(
            ResourceLedger.from_mapping(read_json(shared_ledger)).total_training_simulator_steps
            if method == "fcp"
            else shared_steps
        ),
        training_gpu_hours=elapsed * gpu_count / 3600.0,
        training_wall_clock_hours=elapsed / 3600.0,
        deployable_parameters=parameter_count(first_params),
    )
    write_json(output / "resource_ledger.json", total_ledger.to_mapping())
    run_ledger_root = output / "resource_ledgers"
    run_ledger_root.mkdir(parents=True, exist_ok=True)
    manifest_runs = []
    for record, final_path in zip(records, final_paths, strict=True):
        run_ledger = ResourceLedger(
            ego_policy_steps=_training_steps(method),
            upstream_partner_steps=total_ledger.upstream_partner_steps,
            training_gpu_hours=total_ledger.training_gpu_hours / int(run_count),
            training_wall_clock_hours=total_ledger.training_wall_clock_hours / int(run_count),
            deployable_parameters=total_ledger.deployable_parameters,
        )
        ledger_path = run_ledger_root / f"run-{record['run_index']:02d}.json"
        write_json(ledger_path, run_ledger.to_mapping())
        manifest_runs.append(
            {
                "run_index": record["run_index"],
                "run_id": record["parent_training_run_id"],
                "policy": str(final_path),
                "identity": {
                    "parent_training_run_id": record["parent_training_run_id"],
                    "co_training_group_id": record["co_training_group_id"],
                },
                "resource_ledger": {"path": str(ledger_path)},
            }
        )
    training_lineage = [
        *list(external_lineage),
        *(
            {
                "checkpoint": row["policy"],
                "parent_training_run_id": row["identity"]["parent_training_run_id"],
                "co_training_group_id": row["identity"]["co_training_group_id"],
                "role": "partner_or_ego",
            }
            for row in manifest_runs
        ),
    ]
    write_json(
        policy_manifest,
        {
            "version": POLICY_MANIFEST_VERSION,
            "method": method,
            "layout": layout,
            "policy_kind": "official_checkpoint",
            "runs": manifest_runs,
            "training_lineage": training_lineage,
        },
    )
    write_json(
        output / "population_training.json",
        {
            "version": 1,
            "method": method,
            "layout": layout,
            "root_seed": int(root_seed),
            "runs": records,
            "policy_manifest": {"path": str(policy_manifest)},
            "resource_ledger": {"path": str(output / "resource_ledger.json")},
        },
    )
    return policy_manifest


def _aggregate_ledgers(paths: Sequence[Path]) -> ResourceLedger:
    ledgers = [ResourceLedger.from_mapping(read_json(path)) for path in paths]
    return ResourceLedger(
        ego_policy_steps=sum(item.ego_policy_steps for item in ledgers),
        upstream_partner_steps=sum(item.upstream_partner_steps for item in ledgers),
        training_gpu_hours=sum(item.training_gpu_hours for item in ledgers),
        training_wall_clock_hours=sum(item.training_wall_clock_hours for item in ledgers),
        peak_memory_bytes=max(item.peak_memory_bytes for item in ledgers),
        deployable_parameters=max(item.deployable_parameters for item in ledgers),
    )


def _build_fcp_population(
    *, purpose: str, count: int, layout: str, output: Path
) -> tuple[Path, Path, Path]:
    training_root = output / "source_training"
    population_root = output / "population"
    lineage = []
    ledgers = []
    for population_index in range(int(count)):
        source = training_root / f"population-{population_index:02d}"
        root_seed = UPSTREAM_ROOT_SEEDS[f"{purpose}_fcp_sources"] + population_index
        _run_official_population(
            method="sp",
            layout=layout,
            root_seed=root_seed,
            run_count=8,
            checkpoint_count=3,
            parent_prefix=f"{layout}-{purpose}-fcp-source-{population_index:02d}",
            output=source,
        )
        training = read_json(source / "population_training.json")
        target = population_root / f"fcp_{population_index:02d}"
        target.mkdir(parents=True, exist_ok=True)
        group = f"{layout}-{purpose}-fcp-population-{population_index:02d}"
        for record in training["runs"]:
            run_index = int(record["run_index"])
            source_run = Path(record["checkpoints"][0]["path"]).resolve().parent
            target_run = target / f"run_{run_index}"
            _link_directory(target_run, source_run)
            for checkpoint in record["checkpoints"]:
                lineage.append(
                    {
                        "checkpoint": str(
                            target_run / Path(checkpoint["path"]).name
                        ),
                        "parent_training_run_id": record[
                            "parent_training_run_id"
                        ],
                        "co_training_group_id": group,
                        "role": "fcp_population_checkpoint",
                        "jax_prng_key": record["jax_prng_key"],
                    }
                )
        ledgers.append(source / "resource_ledger.json")
    lineage_path = output / "training_lineage.json"
    ledger_path = output / "resource_ledger.json"
    write_json(lineage_path, lineage)
    write_json(ledger_path, _aggregate_ledgers(ledgers).to_mapping())
    return population_root, lineage_path, ledger_path


def _baseline_command(
    *,
    method: str,
    layout: str,
    output: Path,
    shared_ledger: Path | None = None,
    fcp_population: Path | None = None,
    fcp_lineage: Path | None = None,
    fcp_ledger: Path | None = None,
) -> None:
    if (output / "policy_manifest.json").is_file():
        return
    command = [
        sys.executable,
        "-m",
        "experiments.overcooked_v2.delta_zsc",
        "train-baseline",
        "--method",
        method,
        "--layout",
        layout,
        "--output",
        str(output),
    ]
    if shared_ledger is not None:
        command.extend(("--shared-training-ledger", str(shared_ledger)))
    if fcp_population is not None:
        command.extend(
            (
                "--fcp-population",
                str(fcp_population),
                "--fcp-population-ledger",
                str(fcp_ledger),
                "--training-lineage-manifest",
                str(fcp_lineage),
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    _run_logged(command, output=output, name="wrapper")


def _partner_rows(
    population: Path,
    *,
    role: str,
    run_indexes: Sequence[int],
    all_checkpoints: bool,
) -> list[Mapping[str, Any]]:
    payload = read_json(population / "population_training.json")
    method = str(payload["method"])
    mechanism = {
        "sp": "rnn-sp",
        "op": "rnn-op",
        "state-augmented": "rnn-sa",
        "fcp": "rnn-fcp",
    }[method]
    family = {
        "sp": "official-sp-default",
        "op": "official-op-symmetry",
        "state-augmented": "official-sa-default",
        "fcp": "official-fcp-default",
    }[method]
    records = {int(row["run_index"]): row for row in payload["runs"]}
    rows = []
    for run_index in run_indexes:
        record = records[int(run_index)]
        checkpoints = record["checkpoints"] if all_checkpoints else [record["checkpoints"][-1]]
        for checkpoint in checkpoints:
            stage = float(checkpoint["checkpoint_stage"])
            rows.append(
                {
                    "run_id": f"{record['parent_training_run_id']}-stage-{stage:.1f}",
                    "role": role,
                    "checkpoint": checkpoint["path"],
                    "parent_training_run_id": record["parent_training_run_id"],
                    "generation_mechanism": mechanism,
                    "checkpoint_stage": stage,
                    "hyperparameter_family": family,
                    "seed": int(payload["root_seed"]),
                    "seed_index": int(run_index),
                    "jax_prng_key": record["jax_prng_key"],
                    "owner_seed_index": None,
                    "co_training_group_id": record["co_training_group_id"],
                    "partner_type_id": None,
                }
            )
    return rows


def run_upstream(args: argparse.Namespace) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    layout = config.environment.layout
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "upstream-dag",
            "layout": layout,
            "root_seeds": UPSTREAM_ROOT_SEEDS,
            "config": config.to_mapping(),
        },
    )

    support_sp = output / "development_support" / "sp"
    support_op = output / "development_support" / "op"
    panel_sp = output / "partner_panels" / "sp"
    panel_op = output / "partner_panels" / "op"
    coverage_sa = output / "development_coverage" / "state-augmented"
    confirmatory_sa = output / "confirmatory" / "state-augmented"
    _run_official_population(
        method="sp", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["support_sp"], run_count=10,
        checkpoint_count=3, parent_prefix=f"{layout}-support-sp", output=support_sp,
    )
    _run_official_population(
        method="op", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["support_op"], run_count=10,
        checkpoint_count=3, parent_prefix=f"{layout}-support-op", output=support_op,
    )
    _run_official_population(
        method="sp", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["panel_sp"], run_count=10,
        checkpoint_count=1, parent_prefix=f"{layout}-panel-sp", output=panel_sp,
    )
    _run_official_population(
        method="op", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["panel_op"], run_count=10,
        checkpoint_count=1, parent_prefix=f"{layout}-panel-op", output=panel_op,
    )
    # The Official state-augmented trainer collects states over every ordered
    # pairing of the population and hardcodes ten mini-batches for that pass
    # (state_sample_run.py: num_rollouts=10, state_step_size=10, and
    # scanned_mini_batch_map(..., 10)).  The outer dimension is run_count
    # squared, so run_count=4 gives 16 and trips "outer_dim 16 must be divisible
    # by num_mini_batches 10".  No panel of four can satisfy that assertion --
    # the count has to make run_count**2 divisible by ten, and ten is the
    # smallest such value as well as what the Official configs use themselves.
    #
    # Registered at four until 2026-08-12, when the wide upstream first
    # exercised this path.  Simple never did: its state-augmented checkpoints
    # were trained standalone with NUM_SEEDS=10 and the partner manifest then
    # drew four of them.  Consequence to disclose: wide's two state-augmented
    # panels hold ten runs where simple's hold four.
    _run_official_population(
        method="state-augmented", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["coverage_sa"], run_count=10,
        checkpoint_count=1, parent_prefix=f"{layout}-coverage-sa", output=coverage_sa,
        co_training_groups=[f"{layout}-coverage-sa-shared-population"] * 10,
    )
    _run_official_population(
        method="state-augmented", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["confirmatory_sa"], run_count=10,
        checkpoint_count=1, parent_prefix=f"{layout}-confirmatory-sa", output=confirmatory_sa,
        co_training_groups=[f"{layout}-confirmatory-sa-shared-population"] * 10,
    )

    baseline_fcp_population, baseline_fcp_lineage, baseline_fcp_ledger = (
        _build_fcp_population(
            purpose="baseline", count=10, layout=layout,
            output=output / "fcp_sources" / "baseline",
        )
    )
    panel_fcp_population, panel_fcp_lineage, panel_fcp_ledger = (
        _build_fcp_population(
            purpose="panel", count=8, layout=layout,
            output=output / "fcp_sources" / "panel",
        )
    )
    baselines = output / "baselines"
    _baseline_command(
        method="sp", layout=layout, output=baselines / "sp"
    )
    _baseline_command(
        method="op", layout=layout, output=baselines / "op"
    )
    sa_shared = baselines / "state-augmented" / "shared_training_ledger.json"
    write_json(
        sa_shared,
        ResourceLedger(upstream_partner_steps=4_000_000).to_mapping(),
    )
    _baseline_command(
        method="state-augmented", layout=layout,
        output=baselines / "state-augmented", shared_ledger=sa_shared,
    )
    _baseline_command(
        method="fcp", layout=layout, output=baselines / "fcp",
        fcp_population=baseline_fcp_population,
        fcp_lineage=baseline_fcp_lineage, fcp_ledger=baseline_fcp_ledger,
    )

    panel_fcp = output / "partner_panels" / "fcp"
    panel_lineage = json.loads(panel_fcp_lineage.read_text(encoding="utf-8"))
    _run_official_population(
        method="fcp", layout=layout,
        root_seed=UPSTREAM_ROOT_SEEDS["panel_fcp"], run_count=8,
        checkpoint_count=1, parent_prefix=f"{layout}-panel-fcp", output=panel_fcp,
        co_training_groups=[f"{layout}-panel-fcp-population-{index:02d}" for index in range(8)],
        fcp_population=panel_fcp_population, shared_ledger=panel_fcp_ledger,
        external_lineage=panel_lineage,
    )

    rows = []
    rows.extend(
        _partner_rows(
            support_sp, role="development_support", run_indexes=range(10),
            all_checkpoints=True,
        )
    )
    rows.extend(
        _partner_rows(
            support_op, role="development_support", run_indexes=range(10),
            all_checkpoints=True,
        )
    )
    for source in (panel_sp, panel_op):
        rows.extend(
            _partner_rows(
                source, role="calibration", run_indexes=range(2),
                all_checkpoints=False,
            )
        )
        rows.extend(
            _partner_rows(
                source, role="development_coverage", run_indexes=range(2, 6),
                all_checkpoints=False,
            )
        )
        rows.extend(
            _partner_rows(
                source, role="confirmatory", run_indexes=range(6, 10),
                all_checkpoints=False,
            )
        )
    rows.extend(
        _partner_rows(
            coverage_sa, role="development_coverage", run_indexes=range(4),
            all_checkpoints=False,
        )
    )
    rows.extend(
        _partner_rows(
            confirmatory_sa, role="confirmatory", run_indexes=range(4),
            all_checkpoints=False,
        )
    )
    rows.extend(
        _partner_rows(
            panel_fcp, role="development_coverage", run_indexes=range(4),
            all_checkpoints=False,
        )
    )
    rows.extend(
        _partner_rows(
            panel_fcp, role="confirmatory", run_indexes=range(4, 8),
            all_checkpoints=False,
        )
    )
    plan = output / "plans" / "partner_plan.json"
    write_json(
        plan,
        {"version": 2, "layout": layout, "runs": rows},
    )
    manifest = output / "manifests" / "partners.json"
    build_partner_manifest(
        argparse.Namespace(
            plan=str(plan), expected_layout=layout,
            output=str(manifest),
        )
    )
    write_json(
        output / "upstream_summary.json",
        {
            "version": 1,
            "layout": layout,
            "partner_manifest": {"path": str(manifest)},
            "baseline_policy_manifests": {
                method: {"path": str(baselines / method / "policy_manifest.json")}
                for method in ("sp", "state-augmented", "op", "fcp")
            },
            "root_seeds": UPSTREAM_ROOT_SEEDS,
        },
    )


__all__ = ["UPSTREAM_ROOT_SEEDS", "run_upstream"]
