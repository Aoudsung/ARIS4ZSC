"""Explicit DEPI parameter ownership for the single combined transaction.

Every objective is still evaluated inside one ``value_and_grad`` call.  Each
objective receives the same numerical parameter tree with disallowed leaves
wrapped in ``stop_gradient``; gradients therefore add once without allowing a
control loss to distort the probabilistic response decoder.
"""

from __future__ import annotations

from typing import Any


LOSS_OWNERSHIP: dict[str, tuple[str, ...]] = {
    "task_encoder": ("ppo", "signature", "decision", "owner_distillation"),
    "capability_encoder": (
        "ppo", "response", "signature", "decision", "separation"
    ),
    "protocol_component_embeddings": (
        "ppo",
        "response",
        "signature",
        "decision",
        "separation",
    ),
    "universal_actor": ("ppo", "decision", "owner_distillation"),
    "universal_critic": ("ppo", "signature"),
    "response_decoder": ("response",),
}


def tree_path_string(path: Any) -> str:
    parts = []
    for entry in path:
        parts.append(
            str(
                getattr(
                    entry,
                    "key",
                    getattr(entry, "name", getattr(entry, "idx", entry)),
                )
            )
        )
    return "/".join(parts)


def _parameter_owns(name: str, loss_name: str) -> bool:
    for prefix, owners in LOSS_OWNERSHIP.items():
        if name.startswith(prefix + "/") or name == prefix:
            return loss_name in owners
    return False


def keep_owned_gradients(gradients: Any, *, loss_name: str) -> Any:
    """Zero every leaf whose owning loss set excludes ``loss_name``."""

    import jax
    import jax.numpy as jnp

    def owned(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value if _parameter_owns(name, loss_name) else jnp.zeros_like(value)

    return jax.tree_util.tree_map_with_path(owned, gradients)


def parameters_for_loss(parameters: Any, *, loss_name: str) -> Any:
    """Preserve parameter values while stopping disallowed gradient paths."""

    import jax

    def owned(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value if _parameter_owns(name, loss_name) else jax.lax.stop_gradient(value)

    return jax.tree_util.tree_map_with_path(owned, parameters)


__all__ = [
    "LOSS_OWNERSHIP",
    "keep_owned_gradients",
    "parameters_for_loss",
    "tree_path_string",
]
