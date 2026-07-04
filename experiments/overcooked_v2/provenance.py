from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

PROVENANCE_SCHEMA_VERSION = "ocv2_provenance_v1"
GRAPH_HASH_FIELD = "graph_json_sha256"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_numpy(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        )
    )
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str | None:
    command = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command,
            cwd=None if repo_root is None else str(repo_root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def reward_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "layout": str(config["layout"]),
        "cost_coef": float(training["cost_coef"]),
        "cost_per_step": float(training["cost_per_step"]),
        "shaped_reward_coef": float(training["shaped_reward_coef"]),
    }


def layout_parse_hash(layout_graph: Any) -> str:
    return sha256_json(_layout_payload(layout_graph))


def option_library_hash(option_lib: Any) -> str:
    return sha256_json([_jsonable(option) for option in option_lib.options])


def partner_pool_hash(partners: list[Any]) -> str:
    payload = []
    for partner in partners:
        payload.append(
            {
                "name": str(getattr(partner, "name", "")),
                "partner_id": int(getattr(partner, "partner_id", -1)),
                "protocol": _jsonable(getattr(partner, "protocol", None)),
            }
        )
    return sha256_json(payload)


def graph_json_hash(graph_json: dict[str, Any]) -> str:
    return sha256_json(_without_graph_self_hash(graph_json))


# S28: metadata keys stamped at RUNTIME (train start / checkpoint save) that are
# audit trail, not graph semantics. The checkpoint-embedded graph legitimately
# carries them while the on-disk graph.json does not, so graph-IDENTITY
# comparisons must ignore them — otherwise the eval-side reward-scale gate
# (LDS-B2) false-positives on every checkpoint (first caught 2026-07-04 on the
# E1-rev stage-1 launch; verified content-identical, d945aa… both sides).
GRAPH_RUNTIME_METADATA_KEYS = frozenset(
    {"provenance", "formal_experiment", "graph_source", "preflight_gate"}
)


def graph_content_hash(graph_json: dict[str, Any]) -> str:
    """Canonical hash of a graph's CONTENT: factors/CE plus semantic metadata,
    with runtime bookkeeping keys stripped (S28). Factor/CE drift and semantic
    metadata drift still change this hash; runtime stamps do not."""
    copied = json.loads(json.dumps(_jsonable(graph_json)))
    metadata = copied.get("metadata")
    if isinstance(metadata, dict):
        copied["metadata"] = {
            key: item
            for key, item in metadata.items()
            if key not in GRAPH_RUNTIME_METADATA_KEYS
        }
    return sha256_json(copied)


def stamp_graph_hash(graph_json: dict[str, Any]) -> dict[str, Any]:
    metadata = graph_json.setdefault("metadata", {})
    provenance = metadata.setdefault("provenance", {})
    provenance["schema_version"] = PROVENANCE_SCHEMA_VERSION
    provenance[GRAPH_HASH_FIELD] = graph_json_hash(graph_json)
    return graph_json


def graph_hash_from_spec(graph: Any) -> str:
    return graph_json_hash(graph.to_json_dict())


def runtime_provenance(
    *,
    config: dict[str, Any],
    layout_graph: Any,
    option_lib: Any,
    partners: list[Any],
    repo_root: str | Path | None = None,
    ce_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    commit = git_commit(repo_root)
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "reward_config_sha256": sha256_json(reward_config_payload(config)),
        "git_commit": commit,
        "git_commit_sha256": None if commit is None else sha256_json(commit),
    }
    if layout_graph is not None:
        payload["layout_parse_sha256"] = layout_parse_hash(layout_graph)
    if option_lib is not None:
        payload["option_library_sha256"] = option_library_hash(option_lib)
    if partners:
        payload["partner_pool_sha256"] = partner_pool_hash(partners)
    for key, path in (
        ("ce_matrix_sha256", ce_path),
        ("replay_sha256", replay_path),
        (GRAPH_HASH_FIELD, graph_path),
    ):
        if path:
            path_obj = Path(path)
            if path_obj.exists():
                payload[key] = (
                    graph_json_hash(json.loads(path_obj.read_text(encoding="utf-8")))
                    if key == GRAPH_HASH_FIELD
                    else sha256_file(path_obj)
                )
    return payload


def _without_graph_self_hash(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(_jsonable(value)))
    provenance = copied.get("metadata", {}).get("provenance", {})
    if isinstance(provenance, dict):
        provenance.pop(GRAPH_HASH_FIELD, None)
    return copied


def _layout_payload(layout_graph: Any) -> dict[str, Any]:
    return {
        "layout_name": str(layout_graph.layout_name),
        "width": int(layout_graph.width),
        "height": int(layout_graph.height),
        "passable": np.asarray(layout_graph.passable, dtype=bool).astype(int).tolist(),
        "entities": {
            key: _jsonable(value)
            for key, value in sorted(layout_graph.entities.items())
        },
        "entities_by_kind": {
            key: list(value)
            for key, value in sorted(layout_graph.entities_by_kind.items())
        },
        "interaction_cells": {
            key: [list(cell) for cell in value]
            for key, value in sorted(layout_graph.interaction_cells.items())
        },
        "bottlenecks": [list(cell) for cell in layout_graph.bottlenecks],
        "region_cells": {
            key: [list(cell) for cell in value]
            for key, value in sorted(layout_graph.region_cells.items())
        },
        "cell_to_entity": {
            f"{cell[0]},{cell[1]}": entity
            for cell, entity in sorted(layout_graph.cell_to_entity.items())
        },
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
