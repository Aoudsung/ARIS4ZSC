"""Content-addressed checkpoint and exact reference-policy ownership checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import VQBCCheckpointRef, VQBCConfig
from .integrity import file_sha256, tree_sha256
from .types import CheckpointMetadataV3


CHECKPOINT_SCHEMA_VERSION_V3 = "path_c_flax_checkpoint_v3"
TRAIN_STATE_FORMAT = "flax_msgpack_pytree_leaves_v1"
STATE_FILE = "train_state.msgpack"
DEPLOYMENT_FILE = "deployment.msgpack"
MANIFEST_FILE = "manifest.json"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def state_sha256(serialized_state: bytes) -> str:
    return hashlib.sha256(b"path_c_vqbc_train_state_v2\x00" + serialized_state).hexdigest()


def _serialize_train_state(train_state: Any) -> bytes:
    from flax import serialization
    import jax

    leaves, unused_tree = jax.tree_util.tree_flatten(train_state)
    del unused_tree
    return serialization.msgpack_serialize(
        {
            "leaf_count": len(leaves),
            "leaves": {
                str(index): leaf for index, leaf in enumerate(leaves)
            },
        }
    )


def _restore_train_state(target_state: Any, serialized: bytes) -> Any:
    from flax import serialization
    import jax

    payload = serialization.msgpack_restore(serialized)
    target_leaves, tree = jax.tree_util.tree_flatten(target_state)
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"leaf_count", "leaves"}
        or int(payload["leaf_count"]) != len(target_leaves)
        or not isinstance(payload["leaves"], Mapping)
        or set(payload["leaves"])
        != {str(index) for index in range(len(target_leaves))}
    ):
        raise ValueError("Fourth-model checkpoint train-state tree changed.")
    restored_leaves = [
        payload["leaves"][str(index)]
        for index in range(len(target_leaves))
    ]
    return jax.tree_util.tree_unflatten(tree, restored_leaves)


def checkpoint_metadata(
    *,
    config: VQBCConfig,
    train_state: Any,
    serialized_state: bytes,
) -> CheckpointMetadataV3:
    model_hash = tree_sha256(train_state.online_params)
    return CheckpointMetadataV3(
        schema_version="path_c_model_checkpoint_metadata_v3",
        run_kind=config.run_kind,
        scientific_readout_allowed=False,
        outer_unit_id=(
            None if config.outer_unit is None else config.outer_unit.outer_unit_id
        ),
        reference_checkpoint_path=str(config.backbone_init.checkpoint_path),
        reference_training_run_id=config.backbone_init.training_run_id,
        reference_weights_sha256=config.backbone_init.flax_weights_sha256,
        reference_launch_config_path=str(config.backbone_init.launch_config_path),
        reference_launch_config_sha256=(
            config.backbone_init.launch_config_sha256
        ),
        config_sha256=config.config_sha256,
        model_weights_sha256=model_hash,
        state_sha256=state_sha256(serialized_state),
        effective_environment_steps=int(train_state.effective_environment_steps),
        completed_episodes=int(train_state.completed_episodes),
        update_count=int(train_state.update_count),
        extra={
            "target_model_weights_sha256": tree_sha256(train_state.target_params),
            "codebook_replacement_count": int(
                train_state.codebook.replacement_count
            ),
        },
    )


def save_vqbc_checkpoint(
    directory: str | Path,
    *,
    config: VQBCConfig,
    train_state: Any,
) -> Mapping[str, Any]:
    try:
        from flax import serialization
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Fourth-model checkpoint writing requires Flax.") from error
    root = Path(directory).resolve()
    serialized = _serialize_train_state(train_state)
    deployment_serialized = serialization.msgpack_serialize(
        {
            "online_params": train_state.online_params,
            "codebook": {
                name: getattr(train_state.codebook, name)
                for name in train_state.codebook._fields
            },
            "kl_state": {
                name: getattr(train_state.kl_state, name)
                for name in train_state.kl_state._fields
            },
        }
    )
    metadata = checkpoint_metadata(
        config=config, train_state=train_state, serialized_state=serialized
    )
    state_path = root / STATE_FILE
    deployment_path = root / DEPLOYMENT_FILE
    _atomic_bytes(state_path, serialized)
    _atomic_bytes(deployment_path, deployment_serialized)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
        "format": TRAIN_STATE_FORMAT,
        "state_file": STATE_FILE,
        "state_file_sha256": file_sha256(state_path),
        "deployment_file": DEPLOYMENT_FILE,
        "deployment_file_sha256": file_sha256(deployment_path),
        "metadata": metadata.to_mapping(),
    }
    _atomic_json(root / MANIFEST_FILE, manifest)
    return manifest


def assert_reference_ownership(
    metadata: CheckpointMetadataV3,
    expected_reference: VQBCCheckpointRef,
    *,
    expected_outer_unit_id: int | None,
) -> None:
    observed = (
        Path(metadata.reference_checkpoint_path).resolve(),
        metadata.reference_training_run_id,
        metadata.reference_weights_sha256,
        Path(metadata.reference_launch_config_path).resolve(),
        metadata.reference_launch_config_sha256,
        metadata.outer_unit_id,
    )
    expected = (
        expected_reference.checkpoint_path.resolve(),
        expected_reference.training_run_id,
        expected_reference.flax_weights_sha256,
        expected_reference.launch_config_path.resolve(),
        expected_reference.launch_config_sha256,
        expected_outer_unit_id,
    )
    if observed != expected:
        raise ValueError(
            "The VQBC checkpoint is not anchored to this side's own official reference."
        )


def load_vqbc_checkpoint(
    directory: str | Path,
    *,
    target_state: Any,
    expected_config: VQBCConfig,
) -> tuple[Any, CheckpointMetadataV3, Mapping[str, Any]]:
    try:
        from flax import serialization
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Fourth-model checkpoint loading requires Flax.") from error
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError("Fourth-model checkpoint manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "format",
        "state_file",
        "state_file_sha256",
        "deployment_file",
        "deployment_file_sha256",
        "metadata",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != fields
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION_V3
        or manifest.get("format") != TRAIN_STATE_FORMAT
        or manifest.get("state_file") != STATE_FILE
        or manifest.get("deployment_file") != DEPLOYMENT_FILE
    ):
        raise ValueError("Fourth-model checkpoint manifest changed.")
    state_path = root / STATE_FILE
    deployment_path = root / DEPLOYMENT_FILE
    if (
        not state_path.is_file()
        or file_sha256(state_path) != manifest["state_file_sha256"]
    ):
        raise ValueError("Fourth-model checkpoint state bytes changed.")
    if (
        not deployment_path.is_file()
        or file_sha256(deployment_path)
        != manifest["deployment_file_sha256"]
    ):
        raise ValueError("Fourth-model checkpoint deployment bytes changed.")
    serialized = state_path.read_bytes()
    metadata = CheckpointMetadataV3.from_mapping(manifest["metadata"])
    if metadata.state_sha256 != state_sha256(serialized):
        raise ValueError("Fourth-model checkpoint state hash changed.")
    if metadata.config_sha256 != expected_config.config_sha256:
        raise ValueError("Fourth-model checkpoint belongs to another config.")
    expected_outer = (
        None
        if expected_config.outer_unit is None
        else expected_config.outer_unit.outer_unit_id
    )
    assert_reference_ownership(
        metadata,
        expected_config.backbone_init,
        expected_outer_unit_id=expected_outer,
    )
    restored = _restore_train_state(target_state, serialized)
    if tree_sha256(restored.online_params) != metadata.model_weights_sha256:
        raise ValueError("Restored online parameters changed their content hash.")
    return restored, metadata, manifest


def load_vqbc_deployment(
    directory: str | Path,
    *,
    expected_config: VQBCConfig,
) -> tuple[Mapping[str, Any], CheckpointMetadataV3, Mapping[str, Any]]:
    """Load only policy parameters, codebook, and two deployment temperatures."""

    try:
        from flax import serialization
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Fourth-model deployment loading requires Flax.") from error
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError("Fourth-model checkpoint manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION_V3
        or manifest.get("deployment_file") != DEPLOYMENT_FILE
    ):
        raise ValueError("Fourth-model deployment manifest changed.")
    deployment_path = root / DEPLOYMENT_FILE
    if (
        not deployment_path.is_file()
        or file_sha256(deployment_path)
        != manifest.get("deployment_file_sha256")
    ):
        raise ValueError("Fourth-model deployment bytes changed.")
    metadata = CheckpointMetadataV3.from_mapping(manifest.get("metadata", {}))
    if metadata.config_sha256 != expected_config.config_sha256:
        raise ValueError("Fourth-model deployment belongs to another config.")
    expected_outer = (
        None
        if expected_config.outer_unit is None
        else expected_config.outer_unit.outer_unit_id
    )
    assert_reference_ownership(
        metadata,
        expected_config.backbone_init,
        expected_outer_unit_id=expected_outer,
    )
    payload = serialization.msgpack_restore(deployment_path.read_bytes())
    if not isinstance(payload, Mapping) or set(payload) != {
        "online_params",
        "codebook",
        "kl_state",
    }:
        raise ValueError("Fourth-model deployment payload changed fields.")
    if tree_sha256(payload["online_params"]) != metadata.model_weights_sha256:
        raise ValueError("Deployment parameters changed their content hash.")
    return payload, metadata, manifest


def assert_pairing_reference_ownership(
    *,
    left_metadata: CheckpointMetadataV3,
    left_reference: VQBCCheckpointRef,
    left_outer_unit_id: int,
    right_metadata: CheckpointMetadataV3,
    right_reference: VQBCCheckpointRef,
    right_outer_unit_id: int,
) -> None:
    assert_reference_ownership(
        left_metadata,
        left_reference,
        expected_outer_unit_id=left_outer_unit_id,
    )
    assert_reference_ownership(
        right_metadata,
        right_reference,
        expected_outer_unit_id=right_outer_unit_id,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION_V3",
    "assert_pairing_reference_ownership",
    "assert_reference_ownership",
    "checkpoint_metadata",
    "load_vqbc_deployment",
    "load_vqbc_checkpoint",
    "save_vqbc_checkpoint",
    "state_sha256",
]
