#!/usr/bin/env python3
"""Run one frozen OvercookedV2 Test Time Path C standard experiment.

The script owns only orchestration and evidence assembly.  Every learned policy
is still produced by the registered official or Path C runner, and evaluation
is delegated to the independent ten-policy matrix entrypoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.path_c.contracts.config import CONDITION_CONTROLLERS
from src.path_c.contracts.outer_units import (
    FAMILY_POOL_FORMAL_SEED_ROOT,
    FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
    FORMAL_OUTER_UNIT_COUNT,
    OTHER_PLAY_FAMILY,
    SELF_PLAY_FAMILY,
    derive_family_pool_seed,
    load_outer_units_manifest,
)
from src.path_c.evaluation.standard import (
    EPISODES_PER_PAIRING,
    POPULATION_MANIFEST_SCHEMA_VERSION,
    PopulationManifest,
    file_sha256,
    summarize_project_xp_difference,
    summarize_standard_rows,
    validate_standard_summary_payload,
)
from src.path_c.evaluation.response_contrast import (
    RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION,
    RESPONSE_CONTRAST_ROW_COUNT,
    ResponseContrastManifest,
    summarize_response_contrast_rows,
)
from src.path_c.pipeline.run import source_records
from experiments.overcooked_v2.official.path_c_family_pool_training import (
    ABILITY_ADMISSION_RULE_ID,
    ABILITY_OBSERVATION_SELECTION_EFFECT,
    evaluate_final_policy_ability_admission,
    validate_checkpoint_history,
)
ORCHESTRATOR_SCHEMA_VERSION = "path_c_family_pool_formal_orchestrator_v1"
EXECUTION_CONTRACT_SCHEMA_VERSION = "path_c_family_pool_execution_contract_v2"
EXECUTION_CONTRACT_STATUS = (
    "result_aware_protocol_revision_frozen_before_path_c_and_wide_training"
)
STAGES = (
    "prepare",
    "test",
    "mechanical_smoke",
    "freeze",
    "upstream",
    "outer_manifest",
    "path_c",
    "populations",
    "evaluation",
    "response_contrast",
    "audit",
)
CONDITIONS = (
    "decision_focused",
    "no_probe",
    "random_safe_probe",
    "generic_response_information",
)
POPULATIONS = ("pre_adaptation_backbone", *CONDITIONS)
JAX_MEMORY_FRACTION = "0.90"
TEST_FILES = tuple(
    f"experiments/overcooked_v2/tests/test_path_c_model_{suffix}.py"
    for suffix in (
        "contracts",
        "model",
        "belief_probe",
        "training",
        "evaluation",
        "pipeline",
    )
)
PRACTICAL_EFFECT_THRESHOLD = 20.0
BOOTSTRAP_SAMPLES = 9_999
REGISTERED_LAYOUTS = ("test_time_simple", "test_time_wide")


@dataclass(frozen=True, slots=True)
class OfficialTask:
    task_id: str
    layout: str
    outer_unit_id: int
    member_id: str
    family_id: str
    experiment_variant: str
    training_seed: int
    launch_config_path: Path
    output_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "run_0" / "ckpt_final"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "official_artifact_manifest.json"

    @property
    def ability_admission_path(self) -> Path:
        return self.output_dir / "ability_admission.json"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "layout": self.layout,
            "outer_unit_id": self.outer_unit_id,
            "member_id": self.member_id,
            "family_id": self.family_id,
            "experiment_variant": self.experiment_variant,
            "training_seed": self.training_seed,
            "launch_config_path": str(self.launch_config_path),
            "launch_config_sha256": file_sha256(self.launch_config_path),
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
        }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _write_identical_or_new(path: Path, content: str, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite an existing formal file: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing formal file differs from the frozen content: {path}")
        return
    _atomic_text(path, content)


def _resolve_template_path(template: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else template.parent / path).resolve()


def _member_specs() -> tuple[tuple[str, str, str, int], ...]:
    return (
        ("backbone", SELF_PLAY_FAMILY, "official_backbone", 0),
        ("self_play_0", SELF_PLAY_FAMILY, "official_partner_self_play", 0),
        ("self_play_1", SELF_PLAY_FAMILY, "official_partner_self_play", 1),
        ("other_play_0", OTHER_PLAY_FAMILY, "official_partner_other_play", 0),
        ("other_play_1", OTHER_PLAY_FAMILY, "official_partner_other_play", 1),
    )


def expected_official_task_identities(
    layout: str = "test_time_simple",
) -> tuple[Mapping[str, Any], ...]:
    """Return the complete data-before seed allocation without touching files."""

    if layout not in REGISTERED_LAYOUTS:
        raise ValueError("The formal orchestrator received an unregistered layout.")
    rows: list[Mapping[str, Any]] = []
    for unit in range(FORMAL_OUTER_UNIT_COUNT):
        for member_id, family_id, role, member_index in _member_specs():
            rows.append(
                {
                    "task_id": f"outer_unit_{unit:02d}__{member_id}",
                    "layout": layout,
                    "outer_unit_id": unit,
                    "member_id": member_id,
                    "family_id": family_id,
                    "experiment_variant": (
                        "rnn-sp" if family_id == SELF_PLAY_FAMILY else "rnn-op"
                    ),
                    "training_seed": derive_family_pool_seed(
                        layout=layout,
                        outer_unit_id=unit,
                        role=role,
                        member_index=member_index,
                    ),
                }
            )
    return tuple(rows)


def _formal_paths(
    repository_root: Path, output_root: Path | None, *, layout: str
) -> Mapping[str, Path]:
    formal_root = (
        output_root.resolve()
        if output_root is not None
        else repository_root
        / "results"
        / "path_c_family_pool_formal_standard"
        / layout
    )
    control = formal_root / "control"
    simple_control = formal_root.parent / "test_time_simple" / "control"
    return {
        "formal_root": formal_root,
        "control": control,
        "launches": control / "launch_configs",
        "logs": control / "logs",
        "schedule": control / "official_task_schedule.json",
        "preflight": control / "preflight.json",
        "test_receipt": control / "test_receipt.json",
        "mechanical_smoke": control / "mechanical_smoke.json",
        "reference_acceptance_report": (
            control / "reference_acceptance_report.json"
            if layout == "test_time_simple"
            else simple_control / "reference_acceptance_report.json"
        ),
        "reference_acceptance_log": (
            control / "logs" / "reference_acceptance.log"
            if layout == "test_time_simple"
            else simple_control / "logs" / "reference_acceptance.log"
        ),
        "contract": control / "formal_execution_contract.json",
        "outer_manifest": control / f"path_c_outer_units_{layout}.json",
        "populations": control / "populations",
        "audit": control / "formal_audit_summary.json",
        "response_manifest": control / "response_contrast_manifest.json",
        "response_contrast": control / "response_contrast_receipt.json",
        "state": control / "orchestrator_state.json",
    }


def _template_paths(
    repository_root: Path, *, layout: str
) -> Mapping[str, Path]:
    if layout not in REGISTERED_LAYOUTS:
        raise ValueError("The formal template layout is not registered.")
    configs = repository_root / "experiments" / "overcooked_v2" / "configs"
    layout_name = "simple" if layout == "test_time_simple" else "wide"
    return {
        "sp": configs / f"path_c_official_sp_{layout_name}_seed100.yaml",
        "op": configs / f"path_c_official_op_{layout_name}_seed201.yaml",
        **{
            condition: configs
            / (
                f"path_c_model_v3_formal_{condition}.yaml"
                if layout == "test_time_simple"
                else f"path_c_model_v3_formal_wide_{condition}.yaml"
            )
            for condition in CONDITIONS
        },
    }


def _launch_payload(
    *,
    template_path: Path,
    seed: int,
    output_dir: Path,
    reference_acceptance_report: Path,
) -> Mapping[str, Any]:
    payload = dict(_load_mapping(template_path))
    payload["seed"] = int(seed)
    payload["output_dir"] = str(output_dir)
    checkpoint = dict(payload["checkpoint"])
    checkpoint["expected_path"] = str(output_dir / "run_0" / "ckpt_final")
    payload["checkpoint"] = checkpoint
    acceptance = dict(payload["type_a_acceptance"])
    acceptance["reference_acceptance_config"] = str(
        _resolve_template_path(
            template_path, acceptance["reference_acceptance_config"]
        )
    )
    acceptance["reference_acceptance_report"] = str(
        reference_acceptance_report.resolve()
    )
    payload["type_a_acceptance"] = acceptance
    payload["environment_config"] = str(
        _resolve_template_path(template_path, payload["environment_config"])
    )
    return payload


def prepare_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    *,
    layout: str,
    resume: bool,
) -> tuple[OfficialTask, ...]:
    templates = _template_paths(repository_root, layout=layout)
    for template in templates.values():
        if not template.is_file():
            raise FileNotFoundError(f"Formal template is missing: {template}")
    control = paths["control"]
    if control.exists() and not resume:
        raise FileExistsError(
            f"Formal control directory already exists; use --resume only for the same run: {control}"
        )
    paths["launches"].mkdir(parents=True, exist_ok=True)
    tasks: list[OfficialTask] = []
    for identity in expected_official_task_identities(layout):
        unit = int(identity["outer_unit_id"])
        member_id = str(identity["member_id"])
        variant = str(identity["experiment_variant"])
        launch_path = paths["launches"] / f"outer_unit_{unit:02d}__{member_id}.yaml"
        output_dir = (
            paths["formal_root"]
            / "upstream"
            / f"outer_unit_{unit:02d}"
            / member_id
        )
        payload = _launch_payload(
            template_path=templates["sp" if variant == "rnn-sp" else "op"],
            seed=int(identity["training_seed"]),
            output_dir=output_dir,
            reference_acceptance_report=paths["reference_acceptance_report"],
        )
        content = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        _write_identical_or_new(launch_path, content, resume=resume)
        tasks.append(
            OfficialTask(
                task_id=str(identity["task_id"]),
                layout=layout,
                outer_unit_id=unit,
                member_id=member_id,
                family_id=str(identity["family_id"]),
                experiment_variant=variant,
                training_seed=int(identity["training_seed"]),
                launch_config_path=launch_path.resolve(),
                output_dir=output_dir.resolve(),
            )
        )
    schedule = {
        "schema_version": "path_c_formal_official_task_schedule_v1",
        "layout": layout,
        "seed_root": FAMILY_POOL_FORMAL_SEED_ROOT,
        "task_count": len(tasks),
        "tasks": [task.to_mapping() for task in tasks],
    }
    schedule_content = json.dumps(schedule, indent=2, sort_keys=True) + "\n"
    _write_identical_or_new(paths["schedule"], schedule_content, resume=resume)
    return tuple(tasks)


def _selected_gpu_rows(gpus: Sequence[int]) -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,"
            "ecc.errors.uncorrected.volatile.total,"
            "ecc.errors.uncorrected.aggregate.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError("nvidia-smi returned an unexpected health row.")
        index = int(fields[0])
        rows[index] = {
            "index": index,
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "volatile_uncorrectable_ecc": int(fields[4]),
            "aggregate_uncorrectable_ecc": int(fields[5]),
        }
    selected = []
    for index in gpus:
        if index not in rows:
            raise ValueError(f"Selected GPU does not exist: {index}")
        row = rows[index]
        if row["volatile_uncorrectable_ecc"] != 0 or row[
            "aggregate_uncorrectable_ecc"
        ] != 0:
            raise RuntimeError(f"Selected GPU {index} has an uncorrectable ECC record.")
        if row["memory_used_mib"] > 4_096:
            raise RuntimeError(f"Selected GPU {index} is already in material use.")
        selected.append(row)
    return selected


def _run_logged(
    command: Sequence[str],
    *,
    repository_root: Path,
    log_path: Path,
    gpu: int | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = _cuda_environment(gpu)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=repository_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Remote command failed; inspect {log_path}.")


def _cuda_environment(gpu: int | None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cuda,cpu",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": JAX_MEMORY_FRACTION,
        }
    )
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _parallel_commands(
    jobs: Sequence[tuple[str, Sequence[str], Path]],
    *,
    repository_root: Path,
    gpus: Sequence[int],
) -> None:
    pending = list(jobs)
    active: dict[int, tuple[str, subprocess.Popen[str], Any, Path]] = {}
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job_id, command, log_path = pending.pop(0)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("w", encoding="utf-8")
                environment = _cuda_environment(gpu)
                process = subprocess.Popen(
                    list(command),
                    cwd=repository_root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                active[gpu] = (job_id, process, handle, log_path)
                print(f"started {job_id} on GPU {gpu}", flush=True)
            if not active:
                continue
            time.sleep(5.0)
            for gpu, (job_id, process, handle, log_path) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                del active[gpu]
                if return_code != 0:
                    raise RuntimeError(f"{job_id} failed; inspect {log_path}.")
                print(f"completed {job_id} on GPU {gpu}", flush=True)
    except BaseException:
        for unused_gpu, (unused_id, process, handle, unused_log) in active.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            handle.close()
        raise


def _source_roots(
    repository_root: Path, tasks: Sequence[OfficialTask], *, layout: str
) -> tuple[Path, ...]:
    overcooked = repository_root / "experiments" / "overcooked_v2"
    configs = overcooked / "configs"
    templates = _template_paths(repository_root, layout=layout)
    return (
        Path(__file__).resolve(),
        repository_root / "src" / "path_c",
        overcooked / "model_dock",
        overcooked / "official" / "overcooked_v2_experiments_adapter.py",
        overcooked / "official" / "path_c_family_pool_training.py",
        overcooked / "path_c_official_artifact.py",
        overcooked / "path_c_flax_policy.py",
        overcooked / "path_c_official_evidence.py",
        overcooked / "path_c_pool_admission.py",
        overcooked / "path_c_response_summary.py",
        overcooked / "path_c_seed.py",
        overcooked / "path_c_standard_training.py",
        overcooked / "scripts" / "run_path_c_model.py",
        overcooked / "scripts" / "evaluate_path_c_model.py",
        overcooked / "scripts" / "run_path_c_family_pool_smoke.py",
        overcooked / "scripts" / "run_path_c_official_reference_acceptance.py",
        overcooked / "scripts" / "with_jax_cuda12.sh",
        *(
            (
                overcooked
                / "official"
                / "overcooked_v2_experiments_wide_adapter.py",
                overcooked / "scripts" / "run_path_c_official_wide_training.py",
            )
            if layout == "test_time_wide"
            else (
                overcooked / "scripts" / "run_path_c_official_training.py",
                overcooked
                / "scripts"
                / "run_path_c_official_op_training.py",
            )
        ),
        configs / f"ocv2_{layout}_standard.yaml",
        configs / "path_c_official_sp_simple_seed999_acceptance.yaml",
        *(templates[name] for name in ("sp", "op", *CONDITIONS)),
        *(task.launch_config_path for task in tasks),
        *(repository_root / path for path in TEST_FILES),
    )


def _records_match(records: Sequence[Mapping[str, str]]) -> bool:
    return all(
        Path(str(record["path"])).is_file()
        and file_sha256(record["path"]) == record["sha256"]
        for record in records
    )


def test_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
    gpus: Sequence[int],
    resume: bool,
) -> Mapping[str, Any]:
    sources = source_records(
        _source_roots(repository_root, tasks, layout=layout)
    )
    if resume and paths["test_receipt"].is_file():
        receipt = _load_mapping(paths["test_receipt"])
        if receipt.get("source_files") == sources and _records_match(sources):
            return receipt
    health = _selected_gpu_rows(gpus)
    wrapper = repository_root / "experiments" / "overcooked_v2" / "scripts" / "with_jax_cuda12.sh"
    python = repository_root / ".venv" / "bin" / "python"
    compile_logs: dict[str, Mapping[str, str]] = {}
    for gpu in gpus:
        log = paths["logs"] / "preflight" / f"gpu_{gpu}_jax_compile.log"
        _run_logged(
            (
                "bash",
                str(wrapper),
                str(python),
                "-c",
                "import jax; import jax.numpy as jnp; "
                "y=jax.jit(lambda x:x+1)(jnp.ones((8,),dtype=jnp.float32)); "
                "jax.block_until_ready(y); "
                "assert jax.default_backend()=='gpu'; print(jax.__version__, jax.devices())",
            ),
            repository_root=repository_root,
            log_path=log,
            gpu=gpu,
        )
        compile_logs[str(gpu)] = {"path": str(log), "sha256": file_sha256(log)}
    disk = shutil.disk_usage(repository_root)
    preflight = {
        "schema_version": "path_c_formal_remote_preflight_v1",
        "gpu_health": health,
        "cuda_compile_logs": compile_logs,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }
    _atomic_json(paths["preflight"], preflight)
    junit = paths["control"] / "tests" / "path_c_model_tests.xml"
    test_log = paths["control"] / "tests" / "path_c_model_tests.log"
    command = (
        "bash",
        str(wrapper),
        str(python),
        "-m",
        "pytest",
        "-q",
        *TEST_FILES,
        f"--junitxml={junit}",
    )
    _run_logged(
        command,
        repository_root=repository_root,
        log_path=test_log,
        gpu=gpus[0],
    )
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if not counts["tests"] or any(counts[name] for name in ("failures", "errors", "skipped")):
        raise RuntimeError("The six Path C model test files did not pass without skips.")
    receipt = {
        "schema_version": "path_c_formal_test_receipt_v1",
        "source_files": sources,
        "counts": counts,
        "preflight": {"path": str(paths["preflight"]), "sha256": file_sha256(paths["preflight"])},
        "log": {"path": str(test_log), "sha256": file_sha256(test_log)},
        "junit": {"path": str(junit), "sha256": file_sha256(junit)},
    }
    _atomic_json(paths["test_receipt"], receipt)
    return receipt


def _archive_stale_mechanical_smoke(paths: Mapping[str, Path]) -> Path:
    receipt = paths["mechanical_smoke"]
    receipt_sha256 = file_sha256(receipt)
    archive = (
        paths["control"]
        / "stale_mechanical_smoke"
        / receipt_sha256
    )
    if archive.exists():
        raise RuntimeError(
            "The stale mechanical-check archive already exists; refusing to overwrite it."
        )
    archive.mkdir(parents=True)
    archived: list[dict[str, str]] = []
    candidates = (
        receipt,
        paths["control"] / "mechanical_smoke_artifacts",
        paths["logs"] / "mechanical_smoke.log",
    )
    for source in candidates:
        if not source.exists():
            continue
        destination = archive / source.name
        shutil.move(str(source), str(destination))
        record = {"path": str(destination)}
        if destination.is_file():
            record["sha256"] = file_sha256(destination)
        archived.append(record)
    _atomic_json(
        archive / "archive_receipt.json",
        {
            "schema_version": "path_c_stale_mechanical_smoke_archive_v1",
            "previous_receipt_sha256": receipt_sha256,
            "archived": archived,
        },
    )
    return archive


def _archive_stale_formal_contract(paths: Mapping[str, Path]) -> Path:
    contract = paths["contract"]
    contract_sha256 = file_sha256(contract)
    archive = paths["control"] / "stale_formal_contract" / contract_sha256
    if archive.exists():
        raise RuntimeError(
            "The stale formal-contract archive already exists; refusing to overwrite it."
        )
    archive.mkdir(parents=True)
    destination = archive / contract.name
    shutil.move(str(contract), str(destination))
    _atomic_json(
        archive / "archive_receipt.json",
        {
            "schema_version": "path_c_stale_formal_contract_archive_v1",
            "previous_contract_sha256": contract_sha256,
            "path": str(destination),
            "sha256": file_sha256(destination),
        },
    )
    return archive


def mechanical_smoke_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
    gpus: Sequence[int],
    resume: bool,
) -> Mapping[str, Any]:
    """Run wiring checks and refresh the registered preproduction acceptance."""

    if layout == "test_time_wide":
        simple_receipt = (
            paths["formal_root"].parent
            / "test_time_simple"
            / "control"
            / "mechanical_smoke.json"
        )
        if not simple_receipt.is_file():
            raise FileNotFoundError(
                "Test Time Wide requires the completed Test Time Simple wiring check."
            )
        receipt = dict(_load_mapping(simple_receipt))
        reference_report = paths["reference_acceptance_report"]
        reference_record = receipt.get("reference_acceptance")
        if (
            receipt.get("passed") is not True
            or receipt.get("scientific_readout_allowed") is not False
            or not isinstance(reference_record, Mapping)
            or reference_record.get("path") != str(reference_report)
            or reference_record.get("sha256") != file_sha256(reference_report)
        ):
            raise RuntimeError("The Test Time Simple wiring check is not reusable.")
        copied = {
            "schema_version": "path_c_family_pool_mechanical_smoke_reuse_v1",
            "layout": layout,
            "source_receipt": str(simple_receipt),
            "source_receipt_sha256": file_sha256(simple_receipt),
            "reference_acceptance": dict(reference_record),
            "scientific_readout_allowed": False,
            "passed": True,
        }
        _atomic_json(paths["mechanical_smoke"], copied)
        return copied
    sources = source_records(
        _source_roots(repository_root, tasks, layout=layout)
    )
    if resume and paths["mechanical_smoke"].is_file():
        receipt = _load_mapping(paths["mechanical_smoke"])
        reference_record = receipt.get("reference_acceptance")
        if (
            receipt.get("passed") is True
            and receipt.get("source_files") == sources
            and _records_match(sources)
            and isinstance(reference_record, Mapping)
            and reference_record.get("path")
            == str(paths["reference_acceptance_report"])
            and reference_record.get("sha256")
            == file_sha256(paths["reference_acceptance_report"])
        ):
            return receipt
        _archive_stale_mechanical_smoke(paths)
    wrapper = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "with_jax_cuda12.sh"
    )
    python = repository_root / ".venv" / "bin" / "python"
    script = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "run_path_c_family_pool_smoke.py"
    )
    smoke_root = paths["control"] / "mechanical_smoke_artifacts"
    log = paths["logs"] / "mechanical_smoke.log"
    _run_logged(
        (
            "bash",
            str(wrapper),
            str(python),
            str(script),
            "--output-root",
            str(smoke_root),
        ),
        repository_root=repository_root,
        log_path=log,
        gpu=int(gpus[0]),
    )
    generated = _load_mapping(smoke_root / "summary.json")
    if (
        generated.get("passed") is not True
        or generated.get("performance_readout_generated") is not False
        or generated.get("scientific_readout_allowed") is not False
    ):
        raise RuntimeError("The family-pool mechanical wiring check did not pass.")
    acceptance_config = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "path_c_official_sp_simple_seed999_acceptance.yaml"
    )
    acceptance_script = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "run_path_c_official_reference_acceptance.py"
    )
    if paths["reference_acceptance_report"].is_file():
        validation_code = (
            "from experiments.overcooked_v2.official."
            "overcooked_v2_experiments_adapter import "
            "validate_official_reference_acceptance_report as validate;"
            f"result=validate({str(acceptance_config)!r},"
            f"{str(paths['reference_acceptance_report'])!r});"
            "print('completed_environment_steps='"
            "+str(result['completed_environment_steps']));"
            "print('acceptance_pass='+str(result['acceptance_pass']).lower())"
        )
        acceptance_command = (
            "bash",
            str(wrapper),
            str(python),
            "-c",
            validation_code,
        )
    else:
        acceptance_command = (
            "bash",
            str(wrapper),
            str(python),
            str(acceptance_script),
            "--config",
            str(acceptance_config),
            "--report",
            str(paths["reference_acceptance_report"]),
        )
    _run_logged(
        acceptance_command,
        repository_root=repository_root,
        log_path=paths["reference_acceptance_log"],
        gpu=int(gpus[0]),
    )
    # The CUDA-wrapped acceptance entrypoint validates this report before
    # returning zero.  Keep the unwrapped orchestrator free of JAX imports.
    acceptance_report = _load_mapping(paths["reference_acceptance_report"])
    if acceptance_report.get("acceptance_pass") is not True:
        raise RuntimeError("The current official reference acceptance did not pass.")
    receipt = {
        **dict(generated),
        "layout": layout,
        "source_files": sources,
        "reference_acceptance": {
            "config_path": str(acceptance_config),
            "config_sha256": file_sha256(acceptance_config),
            "path": str(paths["reference_acceptance_report"]),
            "sha256": file_sha256(paths["reference_acceptance_report"]),
            "completed_environment_steps": int(
                acceptance_report["completed_environment_steps"]
            ),
            "acceptance_pass": True,
            "log_path": str(paths["reference_acceptance_log"]),
            "log_sha256": file_sha256(paths["reference_acceptance_log"]),
        },
        "summary_path": str(smoke_root / "summary.json"),
        "summary_sha256": file_sha256(smoke_root / "summary.json"),
        "log_path": str(log),
        "log_sha256": file_sha256(log),
    }
    _atomic_json(paths["mechanical_smoke"], receipt)
    return receipt


def freeze_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
    resume: bool,
) -> Mapping[str, Any]:
    test_receipt = _load_mapping(paths["test_receipt"])
    if any(test_receipt["counts"][name] for name in ("failures", "errors", "skipped")):
        raise RuntimeError("Formal execution cannot freeze a failed test receipt.")
    smoke_receipt = _load_mapping(paths["mechanical_smoke"])
    reference_record = smoke_receipt.get("reference_acceptance")
    if (
        smoke_receipt.get("passed") is not True
        or not isinstance(reference_record, Mapping)
        or reference_record.get("path")
        != str(paths["reference_acceptance_report"])
        or reference_record.get("sha256")
        != file_sha256(paths["reference_acceptance_report"])
    ):
        raise RuntimeError("Formal execution cannot freeze a failed wiring check.")
    sources = source_records(
        _source_roots(repository_root, tasks, layout=layout)
    )
    frozen_at = datetime.now(timezone.utc).isoformat()
    if paths["contract"].is_file():
        existing = _load_mapping(paths["contract"])
        frozen_at = str(existing.get("frozen_at_utc", frozen_at))
    templates = _template_paths(repository_root, layout=layout)
    contract = {
        "schema_version": EXECUTION_CONTRACT_SCHEMA_VERSION,
        "status": EXECUTION_CONTRACT_STATUS,
        "authorization": (
            "user_authorized_ability_threshold_audit_only_and_continue_2026_07_25"
        ),
        "frozen_at_utc": frozen_at,
        "layout": layout,
        "run_kind": "formal",
        "scientific_readout_allowed": False,
        "final_type_b_human_adjudication_required": True,
        "result_aware_protocol_revision": {
            "reason": (
                "three_fixed_upstream_runs_fell_below_the_registered_ability_"
                "threshold_after_all_simple_upstream_training_completed"
            ),
            "revision": (
                "retain_all_fixed_seeds_and_checkpoints_without_retraining_or_"
                "replacement_and_treat_the_threshold_as_audit_only"
            ),
            "simple_and_wide_status": "exploratory_result_aware_revision",
        },
        "seed_root": FAMILY_POOL_FORMAL_SEED_ROOT,
        "outer_training_unit_count": FORMAL_OUTER_UNIT_COUNT,
        "official_task_schedule_path": str(paths["schedule"]),
        "official_task_schedule_sha256": file_sha256(paths["schedule"]),
        "condition_templates": {
            condition: {
                "path": str(templates[condition]),
                "sha256": file_sha256(templates[condition]),
                "controller": CONDITION_CONTROLLERS[condition],
            }
            for condition in CONDITIONS
        },
        "budgets": {
            "preproduction_reference_acceptance_environment_steps_shared_across_layouts": 29_949_952,
            "official_self_play_environment_steps_each": 29_949_952,
            "official_other_play_environment_steps_each": 29_999_104,
            "official_ability_admission_additional_environment_steps_each": 0,
            "prefit_environment_steps_per_outer_unit": 1_000_000,
            "training_calibration_episodes_per_outer_unit": 500,
            "training_calibration_environment_steps_per_outer_unit": 200_000,
            "adaptation_environment_steps_per_condition_and_outer_unit": 10_000_000,
            "deployment_calibration_episodes_per_condition_and_outer_unit": 500,
            "deployment_calibration_environment_steps_per_condition_and_outer_unit": 200_000,
            "evaluation_episodes_per_pairing": EPISODES_PER_PAIRING,
            "evaluation_episode_steps": 400,
        },
        "official_ability_observation": {
            "rule_id": ABILITY_ADMISSION_RULE_ID,
            "evidence": "completed_official_training_raw_return_series",
            "history_selection_consults_return": False,
            "selection_effect": ABILITY_OBSERVATION_SELECTION_EFFECT,
            "threshold_failure_action": (
                "record_and_include_without_seed_replacement_or_retraining"
            ),
            "all_fixed_seed_tasks_included": True,
        },
        "evaluation": {
            "population_ids": list(POPULATIONS),
            "self_play_pairings_per_population": 10,
            "directed_cross_play_pairings_per_population": 90,
            "episodes_per_pairing": EPISODES_PER_PAIRING,
            "condition_independent_episode_seed_schedule": "path_c_standard_evaluation_seed_v2",
            "response_contrast": {
                "population": "decision_focused",
                "directed_cross_play_pairings": 90,
                "matched_episode_blocks_per_pairing": 500,
                "branches": ["A1", "A2-mask", "A2-use"],
            },
        },
        "primary_project_comparison": {
            "left": "decision_focused_xp",
            "right": "no_probe_xp",
            "outer_unit_node_resampling_samples": BOOTSTRAP_SAMPLES,
            "practical_effect_threshold_raw_return_per_episode": PRACTICAL_EFFECT_THRESHOLD,
            "claim_decision_not_automatic": True,
        },
        "source_files": sources,
        "test_receipt": {
            "path": str(paths["test_receipt"]),
            "sha256": file_sha256(paths["test_receipt"]),
        },
        "mechanical_smoke_receipt": {
            "path": str(paths["mechanical_smoke"]),
            "sha256": file_sha256(paths["mechanical_smoke"]),
        },
        "reference_acceptance": dict(reference_record),
    }
    content = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if (
        resume
        and paths["contract"].is_file()
        and paths["contract"].read_text(encoding="utf-8") != content
    ):
        _archive_stale_formal_contract(paths)
        contract["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    _write_identical_or_new(paths["contract"], content, resume=resume)
    return contract


def _assert_frozen(
    paths: Mapping[str, Path], *, layout: str
) -> Mapping[str, Any]:
    contract = _load_mapping(paths["contract"])
    if (
        contract.get("schema_version") != EXECUTION_CONTRACT_SCHEMA_VERSION
        or contract.get("status") != EXECUTION_CONTRACT_STATUS
        or contract.get("layout") != layout
        or contract.get("scientific_readout_allowed") is not False
        or not _records_match(contract.get("source_files", ()))
        or contract.get("official_task_schedule_sha256") != file_sha256(paths["schedule"])
        or not isinstance(contract.get("reference_acceptance"), Mapping)
        or contract["reference_acceptance"].get("path")
        != str(paths["reference_acceptance_report"])
        or contract["reference_acceptance"].get("sha256")
        != file_sha256(paths["reference_acceptance_report"])
    ):
        raise RuntimeError("The formal execution contract or its source files drifted.")
    return contract


def _validate_official_completion(task: OfficialTask) -> Mapping[str, Any]:
    manifest = _load_mapping(task.artifact_manifest_path)
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Official task lacks checkpoint metadata: {task.task_id}")
    if (
        manifest.get("schema_version") != "path_c_official_training_artifact_v2"
        or manifest.get("run_status") != "completed"
        or manifest.get("layout") != task.layout
        or manifest.get("seed") != task.training_seed
        or manifest.get("experiment_variant") != task.experiment_variant
        or manifest.get("training_config_sha256") != file_sha256(task.launch_config_path)
        or Path(str(checkpoint.get("path", ""))).resolve() != task.checkpoint_path
        or not task.checkpoint_path.is_dir()
    ):
        raise ValueError(f"Official task completion changed identity: {task.task_id}")
    dependencies = manifest.get("training_implementation_dependencies", {}).get("files")
    if not isinstance(dependencies, Mapping) or not _records_match(
        tuple({"path": path, "sha256": digest} for path, digest in dependencies.items())
    ):
        raise ValueError(f"Official task source closure drifted: {task.task_id}")
    expected_steps = 29_949_952 if task.experiment_variant == "rnn-sp" else 29_999_104
    if manifest.get("effective_environment_steps") != expected_steps:
        raise ValueError(f"Official task effective budget changed: {task.task_id}")
    validate_checkpoint_history(
        manifest.get("checkpoint_history", ()),
        final_weights_sha256=str(checkpoint.get("model_weights_sha256", "")),
    )
    return manifest


def _write_or_validate_official_ability_admission(
    task: OfficialTask,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    metrics_path = task.output_dir / "training_metrics.json"
    metrics = _load_mapping(metrics_path)
    if (
        metrics.get("experiment_variant") != task.experiment_variant
        or metrics.get("seed") != task.training_seed
        or metrics.get("effective_environment_steps")
        != manifest.get("effective_environment_steps")
    ):
        raise ValueError(
            f"Official ability metrics changed identity: {task.task_id}"
        )
    checkpoint = manifest["checkpoint"]
    report = {
        **dict(
            evaluate_final_policy_ability_admission(
                metrics,
                training_run_id=str(manifest["training_run_id"]),
                launch_config_sha256=file_sha256(task.launch_config_path),
                checkpoint_path=task.checkpoint_path,
                checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
                model_weights_sha256=str(checkpoint["model_weights_sha256"]),
            )
        ),
        "training_metrics_path": str(metrics_path),
        "training_metrics_sha256": file_sha256(metrics_path),
    }
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if task.ability_admission_path.is_file():
        if task.ability_admission_path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Official ability-admission evidence changed: {task.task_id}"
            )
    else:
        _atomic_text(task.ability_admission_path, content)
    return report


def upstream_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
    gpus: Sequence[int],
) -> Mapping[str, Any]:
    _assert_frozen(paths, layout=layout)
    wrapper = repository_root / "experiments" / "overcooked_v2" / "scripts" / "with_jax_cuda12.sh"
    python = repository_root / ".venv" / "bin" / "python"
    pending = []
    for task in tasks:
        if task.artifact_manifest_path.is_file():
            manifest = _validate_official_completion(task)
            _write_or_validate_official_ability_admission(task, manifest)
            continue
        if task.output_dir.exists():
            raise RuntimeError(f"Incomplete official output requires inspection: {task.output_dir}")
        runner_name = (
            "run_path_c_official_wide_training.py"
            if layout == "test_time_wide"
            else (
                "run_path_c_official_training.py"
                if task.experiment_variant == "rnn-sp"
                else "run_path_c_official_op_training.py"
            )
        )
        runner = (
            repository_root
            / "experiments"
            / "overcooked_v2"
            / "scripts"
            / runner_name
        )
        pending.append(
            (
                task.task_id,
                ("bash", str(wrapper), str(python), str(runner), "--config", str(task.launch_config_path)),
                paths["logs"] / "upstream" / f"{task.task_id}.log",
            )
        )
    _parallel_commands(pending, repository_root=repository_root, gpus=gpus)
    manifests = [_validate_official_completion(task) for task in tasks]
    admissions = [
        _write_or_validate_official_ability_admission(task, manifest)
        for task, manifest in zip(tasks, manifests, strict=True)
    ]
    failed_task_ids = [
        task.task_id
        for task, admission in zip(tasks, admissions, strict=True)
        if admission["passed"] is not True
    ]
    receipt = {
        "schema_version": "path_c_formal_upstream_receipt_v1",
        "task_count": len(tasks),
        "completed_environment_steps": sum(
            int(manifest["effective_environment_steps"]) for manifest in manifests
        ),
        "completed_episode_count": sum(
            int(_load_mapping(task.output_dir / "run_descriptor.json")["completed_episode_count"])
            for task in tasks
        ),
        "artifact_manifests": {
            task.task_id: {
                "path": str(task.artifact_manifest_path),
                "sha256": file_sha256(task.artifact_manifest_path),
            }
            for task in tasks
        },
        "ability_admissions": {
            task.task_id: {
                "path": str(task.ability_admission_path),
                "sha256": file_sha256(task.ability_admission_path),
                "passed": bool(admission["passed"]),
            }
            for task, admission in zip(tasks, admissions, strict=True)
        },
        "ability_observation_summary": {
            "selection_effect": ABILITY_OBSERVATION_SELECTION_EFFECT,
            "threshold_passed_count": len(tasks) - len(failed_task_ids),
            "threshold_failed_count": len(failed_task_ids),
            "threshold_failed_task_ids": failed_task_ids,
            "all_fixed_seed_tasks_included": True,
            "seed_replacement_performed": False,
            "retraining_until_pass_performed": False,
        },
    }
    target = paths["control"] / "upstream_receipt.json"
    _atomic_json(target, receipt)
    return receipt


def outer_manifest_stage(
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
) -> Mapping[str, Any]:
    _assert_frozen(paths, layout=layout)
    by_coordinate = {(task.outer_unit_id, task.member_id): task for task in tasks}
    units = []
    for unit_id in range(FORMAL_OUTER_UNIT_COUNT):
        def source(member_id: str) -> Mapping[str, Any]:
            task = by_coordinate[(unit_id, member_id)]
            artifact = _validate_official_completion(task)
            _write_or_validate_official_ability_admission(task, artifact)
            checkpoint = artifact["checkpoint"]
            history = artifact.get("checkpoint_history")
            if not isinstance(history, Sequence) or len(history) != 3:
                raise ValueError(
                    f"Official task lacks its three-checkpoint history: {task.task_id}"
                )
            return {
                "member_id": member_id,
                "family_id": task.family_id,
                "training_seed": task.training_seed,
                "checkpoint_path": str(task.checkpoint_path),
                "flax_weights_sha256": checkpoint["model_weights_sha256"],
                "training_run_id": artifact["training_run_id"],
                "launch_config_path": str(task.launch_config_path),
                "launch_config_sha256": file_sha256(task.launch_config_path),
                "ability_admission_path": str(task.ability_admission_path),
                "ability_admission_sha256": file_sha256(
                    task.ability_admission_path
                ),
                "checkpoint_history": [
                    {
                        "checkpoint_index": int(item["checkpoint_index"]),
                        "update_step": int(item["update_step"]),
                        "effective_environment_steps": int(
                            item["effective_environment_steps"]
                        ),
                        "checkpoint_path": str(item["path"]),
                        "checkpoint_sha256": str(item["checkpoint_sha256"]),
                        "flax_weights_sha256": str(
                            item["model_weights_sha256"]
                        ),
                    }
                    for item in history
                ],
            }

        units.append(
            {
                "outer_unit_id": unit_id,
                "backbone": source("backbone"),
                "partners": [
                    source(member)
                    for member in (
                        "self_play_0",
                        "self_play_1",
                        "other_play_0",
                        "other_play_1",
                    )
                ],
                "seeds": {
                    name: derive_family_pool_seed(
                        layout=layout,
                        outer_unit_id=unit_id,
                        role=role,
                    )
                    for name, role in (
                        ("model_seed", "model"),
                        ("environment_seed", "environment"),
                        ("evaluation_seed", "evaluation"),
                    )
                },
            }
        )
    payload = {
        "schema_version": FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
        "layout": layout,
        "seed_root": FAMILY_POOL_FORMAL_SEED_ROOT,
        "units": units,
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if paths["outer_manifest"].is_file() and paths["outer_manifest"].read_text(encoding="utf-8") != content:
        raise RuntimeError("Existing outer-unit manifest differs from completed upstream runs.")
    if not paths["outer_manifest"].is_file():
        _atomic_text(paths["outer_manifest"], content)
    manifest = load_outer_units_manifest(paths["outer_manifest"])
    return {
        "schema_version": "path_c_formal_outer_manifest_receipt_v1",
        "path": str(manifest.path),
        "sha256": manifest.sha256,
        "unit_count": len(manifest.units),
    }


def _path_c_command(
    repository_root: Path,
    *,
    condition: str,
    layout: str,
    unit_id: int,
    outer_manifest: Path,
) -> tuple[str, ...]:
    wrapper = repository_root / "experiments" / "overcooked_v2" / "scripts" / "with_jax_cuda12.sh"
    python = repository_root / ".venv" / "bin" / "python"
    runner = repository_root / "experiments" / "overcooked_v2" / "scripts" / "run_path_c_model.py"
    template = _template_paths(repository_root, layout=layout)[condition]
    return (
        "bash",
        str(wrapper),
        str(python),
        str(runner),
        "--config",
        str(template),
        "--outer-units-manifest",
        str(outer_manifest),
        "--outer-unit",
        str(unit_id),
        "--resume",
        "--through",
        "deployment_calibration",
    )


def _pipeline_state(paths: Mapping[str, Path], unit_id: int, condition: str) -> Mapping[str, Any]:
    path = (
        paths["formal_root"]
        / f"outer_unit_{unit_id:02d}"
        / condition
        / "pipeline_state.json"
    )
    state = _load_mapping(path)
    stages = state.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {
        "pool_check",
        "prefit",
        "training_calibration",
        "adaptation",
        "deployment_calibration",
    }:
        raise ValueError(f"Incomplete Path C pipeline state: {path}")
    summary_path = Path(stages["adaptation"]["artifacts"]["summary"]["path"])
    summary = _load_mapping(summary_path)
    if (
        summary.get("run_kind") != "formal"
        or summary.get("effective_environment_steps") != 10_000_000
        or summary.get("completed_episodes") != 25_000
        or summary.get("decision_recording") != "probed_steps_only_gzip"
    ):
        raise ValueError(f"Path C adaptation budget or recording mode changed: {path}")
    deployment_summary = _load_mapping(
        Path(stages["deployment_calibration"]["artifacts"]["summary"]["path"])
    )
    if (
        deployment_summary.get("completed_episodes") != 500
        or deployment_summary.get("effective_environment_steps") != 200_000
        or deployment_summary.get("parameter_artifact") != "adaptation.manifest"
    ):
        raise ValueError(f"Path C deployment calibration changed: {path}")
    return state


def _shared_stage_receipts(
    states: Mapping[str, Mapping[str, Any]],
) -> Mapping[int, Mapping[str, Mapping[str, str]]]:
    receipts = {
        unit: {
            stage: {
                condition: str(
                    states[f"outer_unit_{unit:02d}__{condition}"]["stages"][stage][
                        "receipt_sha256"
                    ]
                )
                for condition in CONDITIONS
            }
            for stage in ("pool_check", "prefit", "training_calibration")
        }
        for unit in range(FORMAL_OUTER_UNIT_COUNT)
    }
    if any(
        len(set(condition_receipts.values())) != 1
        for unit_receipts in receipts.values()
        for condition_receipts in unit_receipts.values()
    ):
        raise RuntimeError(
            "Four conditions did not reuse one shared stage within an outer unit."
        )
    return receipts


def path_c_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    *,
    layout: str,
    gpus: Sequence[int],
) -> Mapping[str, Any]:
    _assert_frozen(paths, layout=layout)
    load_outer_units_manifest(paths["outer_manifest"])
    decision_jobs = [
        (
            f"path_c_outer_unit_{unit:02d}__decision_focused",
            _path_c_command(
                repository_root,
                condition="decision_focused",
                layout=layout,
                unit_id=unit,
                outer_manifest=paths["outer_manifest"],
            ),
            paths["logs"] / "path_c" / f"outer_unit_{unit:02d}__decision_focused.log",
        )
        for unit in range(FORMAL_OUTER_UNIT_COUNT)
    ]
    _parallel_commands(decision_jobs, repository_root=repository_root, gpus=gpus)
    remaining_jobs = [
        (
            f"path_c_outer_unit_{unit:02d}__{condition}",
            _path_c_command(
                repository_root,
                condition=condition,
                layout=layout,
                unit_id=unit,
                outer_manifest=paths["outer_manifest"],
            ),
            paths["logs"] / "path_c" / f"outer_unit_{unit:02d}__{condition}.log",
        )
        for unit in range(FORMAL_OUTER_UNIT_COUNT)
        for condition in CONDITIONS[1:]
    ]
    _parallel_commands(remaining_jobs, repository_root=repository_root, gpus=gpus)
    states = {
        f"outer_unit_{unit:02d}__{condition}": _pipeline_state(
            paths, unit, condition
        )
        for unit in range(FORMAL_OUTER_UNIT_COUNT)
        for condition in CONDITIONS
    }
    shared_receipts = _shared_stage_receipts(states)
    return {
        "schema_version": "path_c_formal_training_receipt_v1",
        "condition_count": len(CONDITIONS),
        "outer_unit_count": FORMAL_OUTER_UNIT_COUNT,
        "adaptation_environment_steps": 400_000_000,
        "adaptation_completed_episodes": 1_000_000,
        "shared_receipts": shared_receipts,
    }


def _adapted_entry(
    paths: Mapping[str, Path], unit: int, condition: str
) -> Mapping[str, Any]:
    state = _pipeline_state(paths, unit, condition)
    adaptation = state["stages"]["adaptation"]["artifacts"]
    calibration = state["stages"]["deployment_calibration"]["artifacts"]
    checkpoint_manifest = Path(adaptation["manifest"]["path"])
    calibration_summary = Path(calibration["summary"]["path"])
    return {
        "outer_unit_id": unit,
        "policy_id": f"outer_unit_{unit:02d}",
        "checkpoint_path": str(checkpoint_manifest.parent),
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
        "calibration_summary_path": str(calibration_summary),
        "calibration_summary_sha256": file_sha256(calibration_summary),
    }


def populations_stage(
    repository_root: Path, paths: Mapping[str, Path], *, layout: str
) -> Mapping[str, Any]:
    del repository_root
    _assert_frozen(paths, layout=layout)
    outer = load_outer_units_manifest(paths["outer_manifest"])
    population_paths: dict[str, Mapping[str, str]] = {}
    for population in POPULATIONS:
        backbone = population == "pre_adaptation_backbone"
        policies = []
        for unit in outer.units:
            if backbone:
                artifact_path = unit.backbone.checkpoint_path.parent.parent / "official_artifact_manifest.json"
                policies.append(
                    {
                        "outer_unit_id": unit.outer_unit_id,
                        "policy_id": f"outer_unit_{unit.outer_unit_id:02d}",
                        "checkpoint_path": str(unit.backbone.checkpoint_path),
                        "checkpoint_manifest_sha256": file_sha256(artifact_path),
                    }
                )
            else:
                policies.append(_adapted_entry(paths, unit.outer_unit_id, population))
        payload = {
            "schema_version": POPULATION_MANIFEST_SCHEMA_VERSION,
            "population_id": population,
            "source_type": "official_backbone" if backbone else "adaptation_checkpoint",
            "condition_id": None if backbone else population,
            "controller": None if backbone else CONDITION_CONTROLLERS[population],
            "layout": layout,
            "run_kind": "formal",
            "scientific_readout_allowed": False,
            "episodes_per_pairing": EPISODES_PER_PAIRING,
            "outer_units_manifest_path": str(paths["outer_manifest"]),
            "outer_units_manifest_sha256": outer.sha256,
            "policies": policies,
            "output_root": str(paths["formal_root"] / "standard_evaluation"),
        }
        target = paths["populations"] / f"{population}.json"
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if target.is_file() and target.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing population manifest changed: {target}")
        if not target.is_file():
            _atomic_text(target, content)
        PopulationManifest.load(target)
        population_paths[population] = {"path": str(target), "sha256": file_sha256(target)}
    decision_population = paths["populations"] / "decision_focused.json"
    response_manifest = {
        "schema_version": RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION,
        "population_manifest_path": str(decision_population),
        "population_manifest_sha256": file_sha256(decision_population),
        "output_root": str(paths["formal_root"] / "response_contrast"),
        "run_kind": "formal",
        "scientific_readout_allowed": False,
    }
    response_content = json.dumps(response_manifest, indent=2, sort_keys=True) + "\n"
    if (
        paths["response_manifest"].is_file()
        and paths["response_manifest"].read_text(encoding="utf-8")
        != response_content
    ):
        raise RuntimeError("Existing response-contrast manifest changed.")
    if not paths["response_manifest"].is_file():
        _atomic_text(paths["response_manifest"], response_content)
    ResponseContrastManifest.load(paths["response_manifest"])
    return {
        "schema_version": "path_c_formal_population_receipt_v1",
        "populations": population_paths,
        "response_contrast_manifest": {
            "path": str(paths["response_manifest"]),
            "sha256": file_sha256(paths["response_manifest"]),
        },
    }


def evaluation_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    *,
    layout: str,
    gpus: Sequence[int],
) -> Mapping[str, Any]:
    _assert_frozen(paths, layout=layout)
    wrapper = repository_root / "experiments" / "overcooked_v2" / "scripts" / "with_jax_cuda12.sh"
    python = repository_root / ".venv" / "bin" / "python"
    evaluator = repository_root / "experiments" / "overcooked_v2" / "scripts" / "evaluate_path_c_model.py"
    jobs = []
    for population in POPULATIONS:
        manifest = paths["populations"] / f"{population}.json"
        PopulationManifest.load(manifest)
        jobs.append(
            (
                f"standard_evaluation__{population}",
                ("bash", str(wrapper), str(python), str(evaluator), "--manifest", str(manifest), "--resume"),
                paths["logs"] / "evaluation" / f"{population}.log",
            )
        )
    _parallel_commands(jobs, repository_root=repository_root, gpus=gpus)
    summaries = {}
    for population in POPULATIONS:
        root = paths["formal_root"] / "standard_evaluation" / population
        summary_path = root / "summary.json"
        receipt_path = root / "evaluation_receipt.json"
        summary = validate_standard_summary_payload(_load_mapping(summary_path))
        if (
            summary.get("effective_environment_steps") != 20_000_000
            or summary.get("scientific_readout_allowed") is not False
        ):
            raise ValueError(f"Standard evaluation budget changed: {population}")
        summaries[population] = {
            "summary_path": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
        }
    return {
        "schema_version": "path_c_formal_evaluation_receipt_v1",
        "population_count": len(POPULATIONS),
        "episode_count": 250_000,
        "environment_steps": 100_000_000,
        "summaries": summaries,
    }


def response_contrast_stage(
    repository_root: Path,
    paths: Mapping[str, Path],
    *,
    layout: str,
    gpus: Sequence[int],
) -> Mapping[str, Any]:
    """Run the separately registered response-masking XP contrast."""

    _assert_frozen(paths, layout=layout)
    manifest = ResponseContrastManifest.load(paths["response_manifest"])
    wrapper = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "with_jax_cuda12.sh"
    )
    python = repository_root / ".venv" / "bin" / "python"
    evaluator = (
        repository_root
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "evaluate_path_c_model.py"
    )
    log = paths["logs"] / "response_contrast" / "decision_focused.log"
    _run_logged(
        (
            "bash",
            str(wrapper),
            str(python),
            str(evaluator),
            "--manifest",
            str(manifest.path),
            "--resume",
        ),
        repository_root=repository_root,
        log_path=log,
        gpu=int(gpus[0]),
    )
    root = (
        paths["formal_root"]
        / "response_contrast"
        / "decision_focused_response_contrast"
    )
    summary_path = root / "summary.json"
    receipt_path = root / "evaluation_receipt.json"
    summary = _load_mapping(summary_path)
    if (
        summary.get("pairing_count") != 90
        or summary.get("matched_block_count") != 45_000
        or summary.get("raw_row_count") != RESPONSE_CONTRAST_ROW_COUNT
        or summary.get("executed_environment_steps") != 54_000_000
        or summary.get("scientific_readout_allowed") is not False
        or abs(float(summary.get("identity_residual", float("inf")))) > 1.0e-10
    ):
        raise ValueError("The response contrast failed its formal accounting.")
    payload = {
        "schema_version": "path_c_formal_response_contrast_receipt_v1",
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "evaluation_receipt_path": str(receipt_path),
        "evaluation_receipt_sha256": file_sha256(receipt_path),
        "log_path": str(log),
        "log_sha256": file_sha256(log),
    }
    _atomic_json(paths["response_contrast"], payload)
    return payload


def _jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_stage(
    paths: Mapping[str, Path],
    tasks: Sequence[OfficialTask],
    *,
    layout: str,
) -> Mapping[str, Any]:
    contract = _assert_frozen(paths, layout=layout)
    population_rows: dict[str, list[Mapping[str, Any]]] = {}
    summaries: dict[str, Mapping[str, Any]] = {}
    seed_reference: dict[tuple[int, int, int], int] | None = None
    for population in POPULATIONS:
        root = paths["formal_root"] / "standard_evaluation" / population
        rows_path = root / "rows.jsonl"
        rows = _jsonl_rows(rows_path)
        recomputed = validate_standard_summary_payload(summarize_standard_rows(rows))
        saved = validate_standard_summary_payload(_load_mapping(root / "summary.json"))
        for key, value in recomputed.items():
            if saved.get(key) != value:
                raise RuntimeError(f"Saved standard summary changed {population}.{key}.")
        seeds = {
            (int(row["outer_unit_0"]), int(row["outer_unit_1"]), int(row["episode_index"])): int(row["episode_seed"])
            for row in rows
        }
        if seed_reference is None:
            seed_reference = seeds
        elif seeds != seed_reference:
            raise RuntimeError("Formal populations did not share matrix episode seeds.")
        if population in {"pre_adaptation_backbone", "no_probe"} and any(
            int(row["probe_count"]) != 0 for row in rows
        ):
            raise RuntimeError(f"A zero-probe population recorded a probe: {population}")
        population_rows[population] = rows
        summaries[population] = recomputed
    comparison = dict(
        summarize_project_xp_difference(
            population_rows["decision_focused"],
            population_rows["no_probe"],
            bootstrap_seed=derive_family_pool_seed(
                layout=layout,
                outer_unit_id=0,
                role="project_comparison_bootstrap",
            ),
            bootstrap_samples=BOOTSTRAP_SAMPLES,
        )
    )
    comparison.update(
        {
            "practical_effect_threshold_raw_return_per_episode": PRACTICAL_EFFECT_THRESHOLD,
            "final_type_b_human_adjudication_required": True,
            "scientific_readout_allowed": False,
        }
    )
    official_steps = sum(
        int(_validate_official_completion(task)["effective_environment_steps"])
        for task in tasks
    )
    prefit_steps = 0
    training_calibration_steps = 0
    deployment_calibration_steps = 0
    adaptation_steps = 0
    for unit in range(FORMAL_OUTER_UNIT_COUNT):
        reference = _pipeline_state(paths, unit, "decision_focused")
        prefit_steps += int(
            _load_mapping(Path(reference["stages"]["prefit"]["artifacts"]["summary"]["path"]))[
                "effective_environment_steps"
            ]
        )
        training_calibration_steps += int(
            _load_mapping(
                Path(
                    reference["stages"]["training_calibration"]["artifacts"][
                        "summary"
                    ]["path"]
                )
            )["effective_environment_steps"]
        )
        for condition in CONDITIONS:
            state = _pipeline_state(paths, unit, condition)
            adaptation_steps += int(
                _load_mapping(Path(state["stages"]["adaptation"]["artifacts"]["summary"]["path"]))[
                    "effective_environment_steps"
                ]
            )
            deployment_calibration_steps += int(
                _load_mapping(
                    Path(
                        state["stages"]["deployment_calibration"]["artifacts"][
                            "summary"
                        ]["path"]
                    )
                )["effective_environment_steps"]
            )
    evaluation_steps = sum(int(summary["effective_environment_steps"]) for summary in summaries.values())
    response_rows_path = (
        paths["formal_root"]
        / "response_contrast"
        / "decision_focused_response_contrast"
        / "rows.jsonl"
    )
    response_rows = _jsonl_rows(response_rows_path)
    response_summary = dict(summarize_response_contrast_rows(response_rows))
    response_saved = _load_mapping(response_rows_path.with_name("summary.json"))
    for key, value in response_summary.items():
        if response_saved.get(key) != value:
            raise RuntimeError(f"Saved response summary changed at {key}.")
    response_steps = int(response_summary["executed_environment_steps"])
    upstream_receipt = _load_mapping(paths["control"] / "upstream_receipt.json")
    ability_observations = upstream_receipt.get("ability_admissions")
    ability_summary = upstream_receipt.get("ability_observation_summary")
    if not isinstance(ability_observations, Mapping) or not isinstance(
        ability_summary, Mapping
    ):
        raise RuntimeError("The upstream ability observations are missing.")
    recomputed_failed = sorted(
        task_id
        for task_id, record in ability_observations.items()
        if isinstance(record, Mapping) and record.get("passed") is not True
    )
    if (
        len(ability_observations) != len(tasks)
        or ability_summary.get("selection_effect")
        != ABILITY_OBSERVATION_SELECTION_EFFECT
        or ability_summary.get("threshold_passed_count")
        != len(tasks) - len(recomputed_failed)
        or ability_summary.get("threshold_failed_count") != len(recomputed_failed)
        or sorted(ability_summary.get("threshold_failed_task_ids", ()))
        != recomputed_failed
        or ability_summary.get("all_fixed_seed_tasks_included") is not True
        or ability_summary.get("seed_replacement_performed") is not False
        or ability_summary.get("retraining_until_pass_performed") is not False
    ):
        raise RuntimeError("The upstream ability-observation summary changed.")
    audit = {
        "schema_version": "path_c_formal_standard_audit_v1",
        "layout": layout,
        "run_kind": "formal",
        "scientific_readout_allowed": False,
        "final_type_b_human_adjudication_required": True,
        "result_aware_protocol_revision": (
            "ability_threshold_audit_only_include_all_fixed_seeds_2026_07_25"
        ),
        "confirmatory_status": "exploratory_result_aware_revision",
        "execution_contract_path": str(paths["contract"]),
        "execution_contract_sha256": file_sha256(paths["contract"]),
        "outer_units_manifest_path": str(paths["outer_manifest"]),
        "outer_units_manifest_sha256": file_sha256(paths["outer_manifest"]),
        "ability_observation_summary": dict(ability_summary),
        "actual_environment_steps": {
            "official_upstream": official_steps,
            "path_c_prefit": prefit_steps,
            "path_c_training_calibration": training_calibration_steps,
            "path_c_adaptation": adaptation_steps,
            "path_c_deployment_calibration": deployment_calibration_steps,
            "standard_evaluation": evaluation_steps,
            "response_contrast": response_steps,
            "total": (
                official_steps
                + prefit_steps
                + training_calibration_steps
                + adaptation_steps
                + deployment_calibration_steps
                + evaluation_steps
                + response_steps
            ),
        },
        "population_summaries": summaries,
        "primary_project_comparison": comparison,
        "response_contrast_summary": response_summary,
        "contract_sha256": _canonical_sha256(contract),
    }
    _atomic_json(paths["audit"], audit)
    return audit


def _write_state(paths: Mapping[str, Path], stage: str, payload: Mapping[str, Any]) -> None:
    state = (
        dict(_load_mapping(paths["state"]))
        if paths["state"].is_file()
        else {"schema_version": ORCHESTRATOR_SCHEMA_VERSION, "stages": {}}
    )
    state["stages"][stage] = {
        "payload_sha256": _canonical_sha256(payload),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(paths["state"], state)


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU identifiers must be comma-separated integers.") from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("GPU identifiers must be distinct integers.")
    if any(value not in set(range(8)) for value in values):
        raise argparse.ArgumentTypeError(
            "GPU identifiers must be between 0 and 7."
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=REGISTERED_LAYOUTS,
        default="test_time_simple",
    )
    parser.add_argument("--through", choices=STAGES, default="audit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", type=_parse_gpus, default=(4, 6, 7))
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[3]
    paths = _formal_paths(
        repository_root, arguments.output_root, layout=arguments.layout
    )
    terminal_index = STAGES.index(arguments.through)
    tasks = prepare_stage(
        repository_root,
        paths,
        layout=arguments.layout,
        resume=arguments.resume,
    )
    _write_state(paths, "prepare", _load_mapping(paths["schedule"]))
    if terminal_index == 0:
        return
    tested = test_stage(
        repository_root,
        paths,
        tasks,
        layout=arguments.layout,
        gpus=arguments.gpus,
        resume=arguments.resume,
    )
    _write_state(paths, "test", tested)
    if terminal_index == 1:
        return
    smoke = mechanical_smoke_stage(
        repository_root,
        paths,
        tasks,
        layout=arguments.layout,
        gpus=arguments.gpus,
        resume=arguments.resume,
    )
    _write_state(paths, "mechanical_smoke", smoke)
    if terminal_index == 2:
        return
    frozen = freeze_stage(
        repository_root,
        paths,
        tasks,
        layout=arguments.layout,
        resume=arguments.resume,
    )
    _write_state(paths, "freeze", frozen)
    if terminal_index == 3:
        return
    upstream = upstream_stage(
        repository_root,
        paths,
        tasks,
        layout=arguments.layout,
        gpus=arguments.gpus,
    )
    _write_state(paths, "upstream", upstream)
    if terminal_index == 4:
        return
    outer = outer_manifest_stage(paths, tasks, layout=arguments.layout)
    _write_state(paths, "outer_manifest", outer)
    if terminal_index == 5:
        return
    path_c = path_c_stage(
        repository_root,
        paths,
        layout=arguments.layout,
        gpus=arguments.gpus,
    )
    _write_state(paths, "path_c", path_c)
    if terminal_index == 6:
        return
    populations = populations_stage(
        repository_root, paths, layout=arguments.layout
    )
    _write_state(paths, "populations", populations)
    if terminal_index == 7:
        return
    evaluation = evaluation_stage(
        repository_root,
        paths,
        layout=arguments.layout,
        gpus=arguments.gpus,
    )
    _write_state(paths, "evaluation", evaluation)
    if terminal_index == 8:
        return
    response = response_contrast_stage(
        repository_root,
        paths,
        layout=arguments.layout,
        gpus=arguments.gpus,
    )
    _write_state(paths, "response_contrast", response)
    if terminal_index == 9:
        return
    audit = audit_stage(paths, tasks, layout=arguments.layout)
    _write_state(paths, "audit", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
