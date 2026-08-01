"""Immutable continuous-generator snapshot archive."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def immutable_parameter_snapshot(params: Any) -> Any:
    """Return a device-resident copy that remains valid after live donation.

    Generator parameters continue to be updated with donated buffers.  An
    archive entry therefore must never retain the live parameter leaves: a
    later donated update invalidates those leaves and makes the next archive
    stack attempt access a deleted PjRt buffer.  Materialising and awaiting a
    distinct copy preserves the exact values without changing any random key,
    model operation, or training sample.
    """

    import jax
    import jax.numpy as jnp

    snapshot = jax.tree_util.tree_map(jnp.copy, params)
    return jax.block_until_ready(snapshot)


def stack_parameter_trees(trees: Sequence[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    if not trees:
        raise ValueError("Cannot stack an empty snapshot archive.")
    signatures = [jax.tree_util.tree_structure(tree) for tree in trees]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Generator snapshot parameter trees have different structure.")
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *trees)


def save_generator_snapshot(path: str | Path, params: Any) -> Path:
    import orbax.checkpoint as ocp

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(target), params, force=True)
    return target


def load_generator_snapshot(path: str | Path) -> Any:
    import orbax.checkpoint as ocp

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    return ocp.PyTreeCheckpointer().restore(str(source))


def load_snapshot_archive(paths: Sequence[str | Path]) -> tuple[Any, ...]:
    return tuple(load_generator_snapshot(path) for path in paths)


__all__ = [
    "immutable_parameter_snapshot",
    "load_generator_snapshot",
    "load_snapshot_archive",
    "save_generator_snapshot",
    "stack_parameter_trees",
]
