from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert arrays, dataclasses, enums, and numpy/JAX scalars to canonical JSON.

    This function is deliberately strict: unknown Python objects are rejected
    instead of stringified. Silent stringification would create unstable hashes
    and would permit invalid certifier state to enter persisted artifacts.
    """

    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    # JAX arrays expose tolist through numpy conversion, but importing jax here
    # would make validators depend on optional runtime dependencies.
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _to_jsonable(value.item())
        except Exception:  # noqa: BLE001 - rejected below if unsupported
            pass
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                k = str(k)
            clean[k] = _to_jsonable(v)
        return clean
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not canonical JSON encodable")


def canonical_json(value: Any) -> str:
    """Return deterministic RFC-style JSON used for hashes and record ids."""

    return json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any, prefix: str | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}" if prefix else digest


def dump_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(value), f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: str, rows: list[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row))
            f.write("\n")
