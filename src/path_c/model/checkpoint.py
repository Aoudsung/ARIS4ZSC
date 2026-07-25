"""Content-addressed Flax checkpoint read and write."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..contracts.records import CheckpointMetadata


CHECKPOINT_SCHEMA_VERSION = "path_c_flax_checkpoint_v1"
WEIGHTS_FILE = "weights.msgpack"
MANIFEST_FILE = "manifest.json"


def _leaves(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _leaves(value[key], (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, str(index)))
        return
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("Checkpoint parameter leaves cannot use object dtype.")
    yield path, array


def tree_sha256(params: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"path_c_model_weights_v1\x00")
    leaves = list(_leaves(params))
    if not leaves:
        raise ValueError("Parameter tree has no leaves.")
    for path, array in leaves:
        for field in (
            "/".join(path).encode("utf-8"),
            str(array.dtype).encode("ascii"),
            ",".join(str(value) for value in array.shape).encode("ascii"),
            array.tobytes(order="C"),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def save_checkpoint(
    directory: str | Path,
    *,
    params: Mapping[str, Any],
    metadata: CheckpointMetadata,
) -> Mapping[str, Any]:
    try:
        from flax import serialization
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Checkpoint writing requires Flax.") from error
    root = Path(directory).resolve()
    weights_path = root / WEIGHTS_FILE
    model_weights_sha256 = tree_sha256(params)
    declared_model_hash = metadata.extra.get("model_weights_sha256")
    if declared_model_hash is not None and declared_model_hash != model_weights_sha256:
        raise ValueError("Checkpoint metadata model-weight hash does not match the parameters.")
    _atomic_bytes(weights_path, serialization.msgpack_serialize(dict(params)))
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": "flax_msgpack_file_v1",
        "weights_file": WEIGHTS_FILE,
        "weights_file_sha256": file_sha256(weights_path),
        "model_weights_sha256": model_weights_sha256,
        "metadata": metadata.to_mapping(),
    }
    _atomic_json(root / MANIFEST_FILE, manifest)
    return manifest


def load_checkpoint(directory: str | Path) -> tuple[dict[str, Any], CheckpointMetadata, Mapping[str, Any]]:
    try:
        from flax import serialization
    except ImportError as error:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Checkpoint loading requires Flax.") from error
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError("Checkpoint manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "format",
        "weights_file",
        "weights_file_sha256",
        "model_weights_sha256",
        "metadata",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("Checkpoint manifest fields changed.")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version changed.")
    if (
        manifest.get("format") != "flax_msgpack_file_v1"
        or manifest.get("weights_file") != WEIGHTS_FILE
    ):
        raise ValueError("Checkpoint storage format changed.")
    weights_path = root / str(manifest.get("weights_file"))
    if not weights_path.is_file() or file_sha256(weights_path) != manifest.get("weights_file_sha256"):
        raise ValueError("Checkpoint weight bytes do not match the manifest.")
    restored = serialization.msgpack_restore(weights_path.read_bytes())
    params = {str(key): value for key, value in restored.items()}
    if tree_sha256(params) != manifest.get("model_weights_sha256"):
        raise ValueError("Restored parameter values do not match the manifest.")
    metadata = CheckpointMetadata.from_mapping(manifest.get("metadata", {}))
    declared_model_hash = metadata.extra.get("model_weights_sha256")
    if declared_model_hash is not None and declared_model_hash != manifest.get(
        "model_weights_sha256"
    ):
        raise ValueError("Checkpoint metadata and manifest bind different model weights.")
    return params, metadata, manifest
