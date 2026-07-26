"""Content hashes and explicit parameter copying for model version four."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leaves(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _leaves(value[key], (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, str(index)))
        return
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("Checkpoint parameter leaves cannot use object dtype.")
    yield path, array


def tree_sha256(params: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"path_c_vqbc_model_weights_v2\x00")
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


def _plain(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {str(key): _plain(value) for key, value in tree.items()}
    return tree


def get_parameter_leaf(tree: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = tree
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Parameter tree is missing {'/'.join(path)}.")
        value = value[part]
    return value


def _set_parameter_leaf(
    tree: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    target = tree
    for part in path[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Target parameter tree is missing {'/'.join(path)}.")
        target = child
    if not path or path[-1] not in target:
        raise ValueError(f"Target parameter tree is missing {'/'.join(path)}.")
    target[path[-1]] = value


def copy_explicit_parameter_leaves(
    initial_params: Mapping[str, Any],
    official_params: Mapping[str, Any],
    leaf_mapping: Mapping[tuple[str, ...], tuple[str, ...]],
) -> dict[str, Any]:
    """Copy named leaves only after checking exact source and target shapes."""

    target = _plain(initial_params)
    source = _plain(official_params)
    if not isinstance(target, dict) or not isinstance(source, Mapping):
        raise TypeError("Both parameter trees must be mappings.")
    if not leaf_mapping:
        raise ValueError("Official initialization requires an explicit mapping.")
    targets = tuple(tuple(path) for path in leaf_mapping)
    if len(targets) != len(set(targets)):
        raise ValueError("Official parameter mapping cannot repeat a target leaf.")
    for target_path, source_path in leaf_mapping.items():
        target_value = get_parameter_leaf(target, tuple(target_path))
        source_value = get_parameter_leaf(source, tuple(source_path))
        if np.shape(target_value) != np.shape(source_value):
            raise ValueError(
                f"Official leaf {'/'.join(source_path)} and target "
                f"{'/'.join(target_path)} have different shapes."
            )
        _set_parameter_leaf(target, tuple(target_path), source_value)
    return target


__all__ = [
    "canonical_sha256",
    "copy_explicit_parameter_leaves",
    "file_sha256",
    "get_parameter_leaf",
    "tree_sha256",
]
