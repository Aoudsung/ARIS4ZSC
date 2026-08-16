"""Deterministic run identities and dependency-light checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from .config import CHECKPOINT_SCHEMA_VERSION


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def ensure_run_identity(directory: str | Path, identity: Mapping[str, Any]) -> Path:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "run_identity.json"
    normalized = json.loads(json.dumps(identity, sort_keys=True, default=str))
    if target.is_file():
        observed = read_json(target)
        if observed != normalized:
            raise RuntimeError("Existing run identity differs from the requested run.")
        return target
    return write_json(target, normalized)


def save_checkpoint(
    directory: str | Path,
    *,
    step: int,
    state: Any,
    identity: Mapping[str, Any],
) -> Path:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"checkpoint_{int(step):012d}.pkl"
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    descriptor = {
        "version": CHECKPOINT_SCHEMA_VERSION,
        "step": int(step),
        "checkpoint": target.name,
        "identity": json.loads(json.dumps(identity, sort_keys=True, default=str)),
    }
    write_json(root / "latest.json", descriptor)
    return target


def load_latest_checkpoint(
    directory: str | Path,
    *,
    expected_identity: Mapping[str, Any],
) -> tuple[int, Any] | None:
    root = Path(directory).resolve()
    latest = root / "latest.json"
    if not latest.is_file():
        return None
    descriptor = read_json(latest)
    required = {"version", "step", "checkpoint", "identity"}
    if not isinstance(descriptor, Mapping) or set(descriptor) != required:
        raise ValueError("Checkpoint descriptor schema differs.")
    if int(descriptor["version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint descriptor version must be {CHECKPOINT_SCHEMA_VERSION}."
        )
    expected = json.loads(json.dumps(expected_identity, sort_keys=True, default=str))
    if descriptor["identity"] != expected:
        raise ValueError("Checkpoint identity differs from the requested run.")
    checkpoint = root / str(descriptor["checkpoint"])
    with checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    return int(descriptor["step"]), state


__all__ = [
    "ensure_run_identity",
    "load_latest_checkpoint",
    "read_json",
    "save_checkpoint",
    "write_json",
]
