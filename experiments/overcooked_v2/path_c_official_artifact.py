"""Static contracts for official OvercookedV2 trainer outputs.

The official trainer packages live on the remote host and are intentionally
read-only.  This module records the interface expected from one completed run,
hashes the complete declared source dependency set, and binds the serialized
checkpoint to its Flax parameter tree.  It does not start training or evaluate
policies.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.path_c_standard_training import (
    canonical_mapping_sha256,
    partner_training_run_id,
)


OFFICIAL_TRAINER_LAUNCH_SCHEMA_VERSION = "path_c_official_trainer_launch_v1"
OFFICIAL_RUN_DESCRIPTOR_SCHEMA_VERSION = "path_c_official_run_descriptor_v1"
OFFICIAL_ARTIFACT_SCHEMA_VERSION = "path_c_official_training_artifact_v2"
OFFICIAL_SOURCE_CLOSURE_SCHEMA_VERSION = "path_c_official_source_closure_v1"
FLAX_WEIGHTS_HASH_DOMAIN = "path_c_flax_weights_sha256_v1"
OFFICIAL_ACTION_RULE_ID = "official_flax_categorical_actor_v1"
OFFICIAL_MODEL_CLASS = "OfficialFlaxRecurrentActor"

OFFICIAL_SP_FAMILY_ID = "official_rnn_sp_ippo_v1"
OFFICIAL_OP_FAMILY_ID = "official_rnn_op_other_play_v1"
OFFICIAL_REFERENCE_ID = "overcooked_v2_experiments_actor_critic_rnn_v1"
OFFICIAL_REFERENCE_ACCEPTANCE_RULE_ID = (
    "official_rnn_sp_seed999_final_quarter_return_gate_v1"
)

# 名义总步数是官方配置意图；有效环境步是整数更新日程实际能够执行并须由产物回读的步数。
OFFICIAL_NOMINAL_TOTAL_TIMESTEPS = 30_000_000
OFFICIAL_EFFECTIVE_STEP_CONTRACTS = {
    "rnn-sp": {
        "nominal_total_timesteps": OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
        "num_updates": 457,
        "num_envs": 256,
        "num_steps": 256,
        "effective_environment_steps": 29_949_952,
    },
    "rnn-op": {
        "nominal_total_timesteps": OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
        "num_updates": 1_831,
        "num_envs": 64,
        "num_steps": 256,
        "effective_environment_steps": 29_999_104,
    },
}

_COMMON_FAMILY_FIELDS = {
    "training_algorithm": "official_recurrent_independent_proximal_policy_optimization",
    "training_objective": "official_clipped_policy_and_value_objective_with_entropy",
    "reward_objective": (
        "raw_team_reward_plus_official_environment_shaping_linear_to_zero_by_15000000_steps"
    ),
    "observation_contract": "official_local_5x5",
    "action_space": "six_primitive_actions",
    "model_class": OFFICIAL_MODEL_CLASS,
    "checkpoint_phase": "completed_official_training_run",
    "action_rule": OFFICIAL_ACTION_RULE_ID,
}

OFFICIAL_SP_FAMILY_DEFINITION = {
    "family_id": OFFICIAL_SP_FAMILY_ID,
    **_COMMON_FAMILY_FIELDS,
    "convention_generation": "official_rnn_sp_parameter_shared_self_play",
}
OFFICIAL_OP_FAMILY_DEFINITION = {
    "family_id": OFFICIAL_OP_FAMILY_ID,
    **_COMMON_FAMILY_FIELDS,
    "convention_generation": "official_rnn_op_symmetry_randomized_other_play",
}
OFFICIAL_FAMILY_DEFINITIONS = {
    OFFICIAL_SP_FAMILY_ID: OFFICIAL_SP_FAMILY_DEFINITION,
    OFFICIAL_OP_FAMILY_ID: OFFICIAL_OP_FAMILY_DEFINITION,
}
OFFICIAL_FAMILY_VARIANTS = {
    OFFICIAL_SP_FAMILY_ID: "rnn-sp",
    OFFICIAL_OP_FAMILY_ID: "rnn-op",
}
OFFICIAL_FAMILY_HASHES = {
    family_id: canonical_mapping_sha256(definition)
    for family_id, definition in OFFICIAL_FAMILY_DEFINITIONS.items()
}

_OFFICIAL_R015_ENVIRONMENT_KWARGS = {
    "layout": "test_time_simple",
    "max_steps": 400,
    "observation_type": "DEFAULT",
    "agent_view_size": 2,
    "negative_rewards": True,
    "random_agent_positions": True,
    "sample_recipe_on_delivery": True,
    "indicate_successful_delivery": True,
    "force_path_planning": False,
    "random_reset": False,
}

_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_framed_bytes(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _parameter_leaves(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
    """Yield deterministic string-key paths and dense array leaves."""

    if isinstance(value, Mapping):
        if not value:
            raise ValueError("A Flax parameter mapping cannot be empty.")
        normalized: list[tuple[str, Any]] = []
        for raw_key, child in value.items():
            key = str(raw_key)
            if not key or raw_key != key:
                raise ValueError("Flax parameter mapping keys must be non-empty strings.")
            normalized.append((key, child))
        for key, child in sorted(normalized, key=lambda item: item[0]):
            yield from _parameter_leaves(child, (*path, key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("A Flax parameter sequence cannot be empty.")
        for index, child in enumerate(value):
            yield from _parameter_leaves(child, (*path, f"[{index}]"))
        return
    if not path:
        raise ValueError("A Flax parameter tree must contain named leaves.")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Flax parameter hashing does not accept object arrays.")
    yield path, np.ascontiguousarray(array)


def flax_weights_sha256(params: Mapping[str, Any]) -> str:
    """Hash a Flax parameter tree independently of checkpoint metadata.

    Leaves are ordered by their complete key path.  Each leaf contributes the
    path components, dtype, shape, and contiguous bytes through length-framed
    fields.  The domain tag keeps this identity separate from the Torch weight
    hash used by the retired in-repository trainer path.
    """

    if not isinstance(params, Mapping) or not params:
        raise ValueError("Flax params must be a non-empty mapping.")
    leaves = list(_parameter_leaves(params))
    if not leaves:
        raise ValueError("Flax params contain no tensor leaves.")
    leaves.sort(key=lambda item: item[0])
    digest = hashlib.sha256(FLAX_WEIGHTS_HASH_DOMAIN.encode("ascii") + b"\x00")
    for key_path, value in leaves:
        _hash_framed_bytes(
            digest,
            json.dumps(list(key_path), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
        )
        _hash_framed_bytes(digest, str(value.dtype).encode("ascii"))
        _hash_framed_bytes(
            digest,
            ",".join(str(int(dimension)) for dimension in value.shape).encode("ascii"),
        )
        _hash_framed_bytes(digest, value.tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_artifact_sha256(path: str | Path) -> str:
    """Hash one checkpoint file or an Orbax-style checkpoint directory."""

    target = Path(path)
    if target.is_file():
        return _file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Official checkpoint is missing: {target}")
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError("An official checkpoint directory cannot be empty.")
    digest = hashlib.sha256(b"path_c_checkpoint_directory_sha256_v1\x00")
    for item in files:
        relative = item.relative_to(target).as_posix()
        _hash_framed_bytes(digest, relative.encode("utf-8"))
        _hash_framed_bytes(digest, bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def locate_editable_package(package_name: str) -> dict[str, str]:
    """Resolve an installed package to its source directory without importing it."""

    if not package_name or "." in package_name:
        raise ValueError("The editable package name must be one top-level package.")
    specification = importlib.util.find_spec(package_name)
    if specification is None or specification.origin is None:
        raise FileNotFoundError(f"Installed package is unavailable: {package_name}")
    origin = Path(specification.origin).resolve()
    if not origin.is_file():
        raise ValueError(f"Installed package has no source file: {package_name}")
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "package_name": package_name,
        "version": version,
        "package_source_root": str(origin.parent),
        "finder_origin": str(origin),
    }


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def validate_official_launch_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one static official-trainer launch registration."""

    if payload.get("schema_version") != OFFICIAL_TRAINER_LAUNCH_SCHEMA_VERSION:
        raise ValueError("Unsupported official-trainer launch schema.")
    if payload.get("registration_status") != "static_registered_not_frozen" or (
        payload.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("Official launch registration changed its static status.")
    family = payload.get("partner_family")
    if not isinstance(family, Mapping):
        raise ValueError("Official launch registration lacks a partner family.")
    family_id = str(family.get("family_id", ""))
    if dict(family) != OFFICIAL_FAMILY_DEFINITIONS.get(family_id):
        raise ValueError("Official launch family differs from its registered semantics.")
    variant = str(payload.get("experiment_variant", ""))
    if variant != OFFICIAL_FAMILY_VARIANTS[family_id]:
        raise ValueError("Official launch variant differs from its registered family.")
    if payload.get("layout") != "test_time_simple":
        raise ValueError("R015 official partner production requires test_time_simple.")
    environment = payload.get("environment_kwargs")
    if not isinstance(environment, Mapping) or dict(environment) != (
        _OFFICIAL_R015_ENVIRONMENT_KWARGS
    ):
        raise ValueError("Official launch changed the registered environment kwargs.")
    total_timesteps = float(payload.get("TOTAL_TIMESTEPS", float("nan")))
    if (
        not math.isfinite(total_timesteps)
        or total_timesteps != float(OFFICIAL_NOMINAL_TOTAL_TIMESTEPS)
    ):
        raise ValueError("Official launch changed the nominal 30-million-step intent.")
    effective_contract = payload.get("effective_environment_steps_contract")
    if not isinstance(effective_contract, Mapping) or dict(effective_contract) != (
        OFFICIAL_EFFECTIVE_STEP_CONTRACTS[variant]
    ):
        raise ValueError(
            "Official launch changed its attainable effective-environment-step contract."
        )
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Official launch seed must be a non-negative integer.")
    acceptance = payload.get("type_a_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get(
        "reference_id"
    ) != OFFICIAL_REFERENCE_ID or acceptance.get(
        "reference_acceptance_rule_id"
    ) != OFFICIAL_REFERENCE_ACCEPTANCE_RULE_ID:
        raise ValueError("Official launch changed its registered reference acceptance.")
    if seed == 999:
        if acceptance.get("status") != "registered_pending_run" or payload.get(
            "production_status"
        ) != "not_a_production_candidate":
            raise ValueError("Reference seed 999 cannot be a production candidate.")
    elif acceptance.get("status") != "amended_pending_registered_reference_run" or (
        acceptance.get("remaining_gate") != "seed999_reference_curve_not_yet_run"
    ) or payload.get("production_status") != "pending_separate_remote_authorization":
        raise ValueError("Official production launch bypassed the seed-999 gate.")
    elif not str(acceptance.get("reference_acceptance_config", "")) or not str(
        acceptance.get("reference_acceptance_report", "")
    ):
        raise ValueError("Official production launch lacks its seed-999 evidence paths.")
    if payload.get("WANDB_MODE") != "disabled":
        raise ValueError("Official production registration keeps WANDB_MODE disabled.")
    if not isinstance(payload.get("output_dir"), str) or not payload["output_dir"]:
        raise ValueError("Official launch requires a registered output directory.")
    if not isinstance(payload.get("environment_config"), str) or not payload[
        "environment_config"
    ]:
        raise ValueError("Official launch requires a registered environment config.")
    official_package = payload.get("official_package")
    if (
        not isinstance(official_package, Mapping)
        or official_package.get("distribution_name")
        != "overcooked_v2_experiments"
        or official_package.get("package_version") != "0.0.1"
    ):
        raise ValueError("Official launch changed the registered experiments package.")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("JAX_PLATFORMS") != "cuda,cpu":
        raise ValueError("Official launch requires JAX_PLATFORMS=cuda,cpu.")
    if runtime.get("wrapper") != (
        "experiments/overcooked_v2/scripts/with_jax_cuda12.sh"
    ):
        raise ValueError("Official launch must use the CUDA 12 JAX wrapper.")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("model_class") != OFFICIAL_MODEL_CLASS:
        raise ValueError("Official launch changed its Flax policy class.")
    if model.get("action_rule") != OFFICIAL_ACTION_RULE_ID:
        raise ValueError("Official launch changed its registered action rule.")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not str(
        checkpoint.get("expected_path", "")
    ):
        raise ValueError("Official launch lacks its expected checkpoint path.")
    wiring = payload.get("wiring_verification")
    if not isinstance(wiring, Mapping) or wiring.get("status") not in {
        "pending_wiring_verification",
        "verified",
    }:
        raise ValueError("Official launch wiring status is invalid.")
    if wiring.get("status") == "verified":
        required_verified_values = (
            None
            if not isinstance(official_package, Mapping)
            else (
                official_package.get("editable_source_root"),
                official_package.get("hydra_entrypoint"),
                official_package.get("repository_adapter"),
                model.get("network_definition"),
                model.get("recurrent_state_initializer"),
                model.get("apply_interface"),
                checkpoint.get("format"),
                checkpoint.get("native_save_mechanism"),
            )
        )
        if required_verified_values is None or any(
            not isinstance(value, str)
            or not value
            or value.startswith("pending")
            for value in required_verified_values
        ):
            raise ValueError(
                "Verified official launch wiring cannot retain a pending package, "
                "network, or checkpoint field."
            )
        if checkpoint.get("format") not in {
            "flax_msgpack_file_v1",
            "orbax_pytree_directory_v1",
        }:
            raise ValueError("Verified official checkpoint format is unsupported.")
        if model.get("action_order") != [
            "right",
            "down",
            "left",
            "up",
            "stay",
            "interact",
        ]:
            raise ValueError("Verified official action order changed.")
        if checkpoint.get("parameter_tree_path") != ["params"]:
            raise ValueError("Verified official checkpoint parameter path changed.")
        if wiring.get("official_entrypoint_requires_repository_adapter") is not True:
            raise ValueError("Verified launch omitted its required repository adapter.")
    return payload


def load_official_launch_config(path: str | Path) -> Mapping[str, Any]:
    return validate_official_launch_config(_load_mapping(Path(path)))


def _select_tree_path(tree: Any, path: Sequence[str]) -> Any:
    selected = tree
    for component in path:
        if not isinstance(selected, Mapping) or component not in selected:
            raise ValueError("Official checkpoint parameter_tree_path is invalid.")
        selected = selected[component]
    return selected


def _normalize_parameter_tree_path(raw_path: Any) -> tuple[str, ...]:
    if not isinstance(raw_path, Sequence) or isinstance(
        raw_path, (str, bytes, bytearray)
    ) or not raw_path:
        raise ValueError("Official checkpoint parameter_tree_path must be non-empty.")
    path = tuple(str(component) for component in raw_path)
    if any(not component for component in path):
        raise ValueError("Official checkpoint parameter_tree_path has an empty component.")
    return path


def load_flax_parameter_tree(
    checkpoint_path: str | Path,
    *,
    checkpoint_format: str,
    parameter_tree_path: Sequence[str],
) -> Mapping[str, Any]:
    """Load the parameter subtree through the registered serialization adapter."""

    path = Path(checkpoint_path)
    if checkpoint_format == "flax_msgpack_file_v1":
        if not path.is_file():
            raise FileNotFoundError("The registered Flax msgpack checkpoint is missing.")
        try:
            from flax import serialization
        except ImportError as error:  # pragma: no cover - remote runtime dependency
            raise RuntimeError("Flax is required to restore the official checkpoint.") from error
        restored = serialization.msgpack_restore(path.read_bytes())
    elif checkpoint_format == "orbax_pytree_directory_v1":
        if not path.is_dir():
            raise FileNotFoundError("The registered Orbax checkpoint directory is missing.")
        try:
            import orbax.checkpoint as ocp
        except ImportError as error:  # pragma: no cover - remote runtime dependency
            raise RuntimeError("Orbax is required to restore the official checkpoint.") from error
        restored = ocp.PyTreeCheckpointer().restore(str(path))
    else:
        raise ValueError("The official checkpoint serialization format is unsupported.")
    params = _select_tree_path(
        restored,
        _normalize_parameter_tree_path(parameter_tree_path),
    )
    if not isinstance(params, Mapping) or not params:
        raise ValueError("The official checkpoint does not expose a Flax parameter tree.")
    return params


def _source_dependency_closure(
    raw_dependencies: Any,
) -> Mapping[str, Any]:
    if not isinstance(raw_dependencies, Sequence) or isinstance(
        raw_dependencies, (str, bytes)
    ) or not raw_dependencies:
        raise ValueError("Official run descriptor requires source dependencies.")
    files: dict[str, str] = {}
    for raw_entry in raw_dependencies:
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "sha256"}:
            raise ValueError("An official source dependency has the wrong schema.")
        path = Path(str(raw_entry["path"])).resolve()
        declared_sha256 = str(raw_entry["sha256"])
        if not path.is_file() or not _is_sha256(declared_sha256):
            raise ValueError("An official source dependency is missing or unbound.")
        actual_sha256 = _file_sha256(path)
        mirror_root_value = os.environ.get(
            "PATH_C_OFFICIAL_SOURCE_MIRROR_ROOT", ""
        ).strip()
        if actual_sha256 != declared_sha256 and mirror_root_value:
            mirror_root = Path(mirror_root_value).resolve()
            candidate = None
            for anchor in ("experiments", "src"):
                if anchor in path.parts:
                    candidate = mirror_root.joinpath(
                        *path.parts[path.parts.index(anchor) :]
                    )
                    break
            if (
                candidate is not None
                and candidate.is_file()
                and _file_sha256(candidate) == declared_sha256
            ):
                actual_sha256 = declared_sha256
        if actual_sha256 != declared_sha256:
            raise ValueError("Official source dependency closure hash mismatch.")
        normalized = str(path)
        if normalized in files:
            raise ValueError("Official source dependency closure repeats a path.")
        files[normalized] = actual_sha256
    return {
        "schema_version": OFFICIAL_SOURCE_CLOSURE_SCHEMA_VERSION,
        "files": dict(sorted(files.items())),
    }


def _resolve_declared_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def build_official_artifact_manifest(
    launch_config_path: str | Path,
    run_descriptor_path: str | Path,
    *,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest-v2-equivalent record from one completed official run."""

    launch_path = Path(launch_config_path).resolve()
    descriptor_path = Path(run_descriptor_path).resolve()
    launch = load_official_launch_config(launch_path)
    wiring = launch["wiring_verification"]
    if wiring.get("status") != "verified":
        raise ValueError("Official checkpoint wiring must be verified before wrapping a run.")
    descriptor = _load_mapping(descriptor_path)
    if descriptor.get("schema_version") != OFFICIAL_RUN_DESCRIPTOR_SCHEMA_VERSION or (
        descriptor.get("run_status") != "completed"
    ):
        raise ValueError("Official run descriptor is not a completed run.")
    family = dict(launch["partner_family"])
    family_id = family["family_id"]
    effective_environment_steps = int(
        launch["effective_environment_steps_contract"][
            "effective_environment_steps"
        ]
    )
    for field, expected in (
        ("experiment_variant", launch["experiment_variant"]),
        ("layout", launch["layout"]),
        ("seed", launch["seed"]),
        ("effective_environment_steps", effective_environment_steps),
    ):
        if descriptor.get(field) != expected:
            raise ValueError(f"Official run descriptor changed {field}.")
    if descriptor.get("environment_kwargs") != launch["environment_kwargs"]:
        raise ValueError("Official run descriptor changed the environment kwargs.")
    architecture = descriptor.get("architecture")
    if not isinstance(architecture, Mapping) or dict(architecture) != dict(
        launch["model"]
    ):
        raise ValueError("Official run descriptor architecture differs from launch config.")
    checkpoint = descriptor.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "format",
        "parameter_tree_path",
    }:
        raise ValueError("Official run descriptor checkpoint has the wrong schema.")
    checkpoint_path = _resolve_declared_path(
        checkpoint["path"], base=descriptor_path.parent
    )
    expected_checkpoint = _resolve_declared_path(
        launch["checkpoint"]["expected_path"], base=launch_path.parent
    )
    if checkpoint_path != expected_checkpoint:
        raise ValueError("Official run descriptor identifies another checkpoint.")
    if checkpoint["format"] != launch["checkpoint"]["format"]:
        raise ValueError("Official checkpoint format differs from launch registration.")
    parameter_tree_path = _normalize_parameter_tree_path(
        checkpoint["parameter_tree_path"]
    )
    source_dependencies = _source_dependency_closure(
        descriptor.get("training_source_dependencies")
    )
    implementation_sha256 = canonical_mapping_sha256(source_dependencies)
    environment_config_path = _resolve_declared_path(
        launch["environment_config"], base=launch_path.parent
    )
    environment_config_sha256 = _file_sha256(environment_config_path)
    if params is None:
        params = load_flax_parameter_tree(
            checkpoint_path,
            checkpoint_format=str(checkpoint["format"]),
            parameter_tree_path=parameter_tree_path,
        )
    weights_sha256 = flax_weights_sha256(params)
    checkpoint_sha256 = checkpoint_artifact_sha256(checkpoint_path)
    launch_sha256 = _file_sha256(launch_path)
    family_sha256 = OFFICIAL_FAMILY_HASHES[family_id]
    training_run_id = partner_training_run_id(
        family_spec_sha256=family_sha256,
        training_seed=int(launch["seed"]),
        training_config_sha256=launch_sha256,
        environment_config_sha256=environment_config_sha256,
        training_implementation_sha256=implementation_sha256,
    )
    return {
        "schema_version": OFFICIAL_ARTIFACT_SCHEMA_VERSION,
        "run_status": "completed",
        "scientific_readout_allowed": False,
        "experiment_variant": launch["experiment_variant"],
        "layout": launch["layout"],
        "seed": int(launch["seed"]),
        "nominal_total_timesteps": OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
        "effective_environment_steps": effective_environment_steps,
        "partner_family": family,
        "partner_family_spec_sha256": family_sha256,
        "training_config_path": str(launch_path),
        "training_config_sha256": launch_sha256,
        "environment_config_path": str(environment_config_path),
        "environment_config_sha256": environment_config_sha256,
        "training_implementation_dependencies": source_dependencies,
        "training_implementation_sha256": implementation_sha256,
        "architecture": dict(architecture),
        "architecture_sha256": canonical_mapping_sha256(architecture),
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": str(checkpoint["format"]),
            "parameter_tree_path": list(parameter_tree_path),
            "checkpoint_sha256": checkpoint_sha256,
            "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
            "model_weights_sha256": weights_sha256,
        },
        "training_run_id": training_run_id,
    }


def validate_official_artifact_manifest(
    launch_config_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_checkpoint_path: str | Path,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a wrapped official artifact and return support-provenance fields."""

    launch_path = Path(launch_config_path).resolve()
    artifact_path = Path(manifest_path).resolve()
    reported = _load_mapping(artifact_path)
    if reported.get("schema_version") != OFFICIAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Official training manifest has the wrong schema.")
    if reported.get("scientific_readout_allowed") is not False:
        raise ValueError("Official training artifacts cannot authorize scientific readout.")
    if int(reported.get("nominal_total_timesteps", -1)) != (
        OFFICIAL_NOMINAL_TOTAL_TIMESTEPS
    ):
        raise ValueError("Official training manifest changed the nominal step intent.")
    checkpoint = reported.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "format",
        "parameter_tree_path",
        "checkpoint_sha256",
        "model_weights_hash_domain",
        "model_weights_sha256",
    }:
        raise ValueError("Official training manifest lacks its checkpoint binding.")
    parameter_tree_path = _normalize_parameter_tree_path(
        checkpoint.get("parameter_tree_path")
    )
    expected_path = Path(expected_checkpoint_path).resolve()
    if Path(str(checkpoint.get("path", ""))).resolve() != expected_path:
        raise ValueError("Official training manifest identifies another checkpoint.")
    reported_dependencies = reported.get("training_implementation_dependencies")
    if not isinstance(reported_dependencies, Mapping) or not isinstance(
        reported_dependencies.get("files"), Mapping
    ):
        raise ValueError("Official training manifest lacks its source dependency closure.")

    descriptor = {
        "schema_version": OFFICIAL_RUN_DESCRIPTOR_SCHEMA_VERSION,
        "run_status": reported.get("run_status"),
        "experiment_variant": reported.get("experiment_variant"),
        "layout": reported.get("layout"),
        "seed": reported.get("seed"),
        "effective_environment_steps": reported.get("effective_environment_steps"),
        "environment_kwargs": load_official_launch_config(launch_path)[
            "environment_kwargs"
        ],
        "architecture": reported.get("architecture"),
        "checkpoint": {
            "path": checkpoint.get("path"),
            "format": checkpoint.get("format"),
            "parameter_tree_path": list(parameter_tree_path),
        },
        "training_source_dependencies": [
            {"path": path, "sha256": sha256}
            for path, sha256 in dict(reported_dependencies["files"]).items()
        ],
    }
    # Do not write a sidecar merely to reuse the builder.  The following mirrors
    # its checks directly through a small in-memory validation path.
    launch = load_official_launch_config(launch_path)
    if launch["wiring_verification"].get("status") != "verified":
        raise ValueError("Official checkpoint wiring is not verified.")
    if descriptor["run_status"] != "completed":
        raise ValueError("Official training manifest is not complete.")
    expected_effective_environment_steps = int(
        launch["effective_environment_steps_contract"][
            "effective_environment_steps"
        ]
    )
    if descriptor["experiment_variant"] != launch["experiment_variant"] or (
        descriptor["layout"] != launch["layout"]
    ) or int(descriptor["seed"]) != int(launch["seed"]) or int(
        descriptor["effective_environment_steps"]
    ) != expected_effective_environment_steps:
        raise ValueError("Official training manifest changed the registered run.")
    if descriptor["architecture"] != launch["model"]:
        raise ValueError("Official training manifest architecture changed.")
    registered_checkpoint_path = _resolve_declared_path(
        launch["checkpoint"]["expected_path"], base=launch_path.parent
    )
    if expected_path != registered_checkpoint_path:
        raise ValueError("Official support registration names another checkpoint.")
    if checkpoint.get("format") != launch["checkpoint"].get("format"):
        raise ValueError("Official checkpoint format changed after wiring verification.")
    dependencies = _source_dependency_closure(
        descriptor["training_source_dependencies"]
    )
    if dependencies != reported_dependencies:
        raise ValueError("Official training dependency closure changed.")
    implementation_sha256 = canonical_mapping_sha256(dependencies)
    if reported.get("training_implementation_sha256") != implementation_sha256:
        raise ValueError("Official training implementation hash changed.")
    actual_checkpoint_sha256 = checkpoint_artifact_sha256(expected_path)
    if checkpoint.get("checkpoint_sha256") != actual_checkpoint_sha256:
        raise ValueError("Official checkpoint content hash changed.")
    if params is None:
        params = load_flax_parameter_tree(
            expected_path,
            checkpoint_format=str(checkpoint.get("format", "")),
            parameter_tree_path=parameter_tree_path,
        )
    weights_sha256 = flax_weights_sha256(params)
    if checkpoint.get("model_weights_hash_domain") != FLAX_WEIGHTS_HASH_DOMAIN or (
        checkpoint.get("model_weights_sha256") != weights_sha256
    ):
        raise ValueError("Official checkpoint Flax weight hash changed.")
    environment_config_path = _resolve_declared_path(
        launch["environment_config"], base=launch_path.parent
    )
    environment_config_sha256 = _file_sha256(environment_config_path)
    training_config_sha256 = _file_sha256(launch_path)
    if Path(str(reported.get("training_config_path", ""))).resolve() != launch_path or (
        reported.get("training_config_sha256") != training_config_sha256
    ):
        raise ValueError("Official training configuration binding changed.")
    if Path(str(reported.get("environment_config_path", ""))).resolve() != (
        environment_config_path
    ) or reported.get("environment_config_sha256") != environment_config_sha256:
        raise ValueError("Official environment configuration binding changed.")
    family = dict(launch["partner_family"])
    family_sha256 = OFFICIAL_FAMILY_HASHES[family["family_id"]]
    expected_run_id = partner_training_run_id(
        family_spec_sha256=family_sha256,
        training_seed=int(launch["seed"]),
        training_config_sha256=training_config_sha256,
        environment_config_sha256=environment_config_sha256,
        training_implementation_sha256=implementation_sha256,
    )
    if reported.get("training_run_id") != expected_run_id:
        raise ValueError("Official training run identifier changed.")
    architecture = dict(launch["model"])
    if reported.get("architecture_sha256") != canonical_mapping_sha256(architecture):
        raise ValueError("Official training architecture hash changed.")
    if reported.get("partner_family") != family or reported.get(
        "partner_family_spec_sha256"
    ) != family_sha256:
        raise ValueError("Official training family binding changed.")
    return {
        "artifact_verified": True,
        "family_id": family["family_id"],
        "family_spec_sha256": family_sha256,
        "training_seed": int(launch["seed"]),
        "snapshot_environment_steps": expected_effective_environment_steps,
        "checkpoint_path": str(expected_path),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "model_weights_sha256": weights_sha256,
        "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
        "training_config_path": str(launch_path),
        "training_config_sha256": training_config_sha256,
        "training_manifest_path": str(artifact_path),
        "training_manifest_sha256": _file_sha256(artifact_path),
        "environment_config_sha256": environment_config_sha256,
        "training_implementation_sha256": implementation_sha256,
        "training_implementation_dependencies": dependencies,
        "training_run_id": expected_run_id,
        "architecture": architecture,
        "architecture_sha256": canonical_mapping_sha256(architecture),
        "layout": str(launch["layout"]),
    }
