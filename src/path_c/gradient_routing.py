"""DEPI parameter ownership for the combined four-loss update.

METHOD_SPEC §3.3/§3.4: the eight-objective PCGrad machinery
(``combine_belief_gradients``, ``project_conflicting_gradient``,
``route_response_with_pcgrad``, ``BELIEF_OBJECTIVE_ORDER``) is abolished.
All four losses are evaluated at the current parameters in a single
``jax.value_and_grad`` and one gradient step updates every trainable
parameter; the ownership table below is retained for the raw-Q Retrace
sub-step (critic Q heads only) and as a structural audit of the combined
gradient.

Ownership table (METHOD_SPEC §3.4):

* ``task_encoder``                                   <- {ppo, signature}
* ``capability_encoder`` / ``protocol_encoder`` /
  ``protocol_component_embeddings``                  <- {ppo, response, signature, separation}
* ``universal_actor``                                <- {ppo}
* ``universal_critic``: ``shaped_*`` heads           <- {ppo}
* ``universal_critic``: ``raw_q1_*`` / ``raw_q2_*``  <- {signature}
* ``response_decoder``                               <- {response}
"""

from __future__ import annotations

from typing import Any


LOSS_OWNERSHIP: dict[str, tuple[str, ...]] = {
    "task_encoder": ("ppo", "signature"),
    "capability_encoder": ("ppo", "response", "signature", "separation"),
    "protocol_encoder": ("ppo", "response", "signature", "separation"),
    "protocol_component_embeddings": ("ppo", "response", "signature", "separation"),
    "universal_actor": ("ppo",),
    "universal_critic": ("ppo", "signature"),  # split by head below
    "response_decoder": ("response",),
}


def tree_path_string(path: Any) -> str:
    parts = []
    for entry in path:
        parts.append(str(getattr(entry, "key", getattr(entry, "name", getattr(entry, "idx", entry)))))
    return "/".join(parts)


def tree_inner_product(first: Any, second: Any) -> Any:
    import jax
    import jax.numpy as jnp

    left = jax.tree_util.tree_leaves(first)
    right = jax.tree_util.tree_leaves(second)
    if len(left) != len(right):
        raise ValueError("Gradient trees differ.")
    return sum(
        (jnp.vdot(jnp.asarray(a), jnp.asarray(b)) for a, b in zip(left, right)),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def tree_l2_norm(tree: Any) -> Any:
    import jax.numpy as jnp

    return jnp.sqrt(jnp.maximum(tree_inner_product(tree, tree), 0.0))


def sum_gradient_trees(*gradients: Any) -> Any:
    import jax

    if not gradients:
        raise ValueError("At least one gradient tree is required.")
    return jax.tree_util.tree_map(lambda *values: sum(values), *gradients)


def select_gradient_prefixes(gradients: Any, prefixes: tuple[str, ...]) -> Any:
    import jax
    import jax.numpy as jnp

    def select(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value if any(name.startswith(prefix + "/") for prefix in prefixes) else jnp.zeros_like(value)

    return jax.tree_util.tree_map_with_path(select, gradients)


def remove_gradient_prefixes(gradients: Any, prefixes: tuple[str, ...]) -> Any:
    import jax
    import jax.numpy as jnp

    def remove(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return jnp.zeros_like(value) if any(name.startswith(prefix + "/") for prefix in prefixes) else value

    return jax.tree_util.tree_map_with_path(remove, gradients)


def _parameter_owns(name: str, loss_name: str) -> bool:
    if name.startswith("universal_critic/"):
        raw_head = "/raw_q1_" in name or "/raw_q2_" in name or name.startswith(
            "universal_critic/raw_q"
        )
        return loss_name in ({"signature"} if raw_head else {"ppo"})
    for prefix, owners in LOSS_OWNERSHIP.items():
        if name.startswith(prefix + "/") or name == prefix:
            return loss_name in owners
    return False


def keep_owned_gradients(gradients: Any, *, loss_name: str) -> Any:
    """Zero every leaf whose owning loss set does not contain ``loss_name``."""

    import jax
    import jax.numpy as jnp

    def owned(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value if _parameter_owns(name, loss_name) else jnp.zeros_like(value)

    return jax.tree_util.tree_map_with_path(owned, gradients)


def keep_gradients_for_losses(gradients: Any, *, loss_names: tuple[str, ...]) -> Any:
    return sum_gradient_trees(*(keep_owned_gradients(gradients, loss_name=name) for name in loss_names))


def keep_raw_q_head_gradients(gradients: Any) -> Any:
    """Raw-Q Retrace sub-step restriction: critic Q heads only (§3.3)."""

    return keep_owned_gradients(gradients, loss_name="signature")


def scale_gradient_prefixes(gradients: Any, prefixes: tuple[str, ...], scale: Any) -> Any:
    import jax

    def scale_leaf(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value * scale if any(name.startswith(prefix + "/") for prefix in prefixes) else value

    return jax.tree_util.tree_map_with_path(scale_leaf, gradients)


__all__ = [
    "LOSS_OWNERSHIP",
    "keep_gradients_for_losses",
    "keep_owned_gradients",
    "keep_raw_q_head_gradients",
    "remove_gradient_prefixes",
    "scale_gradient_prefixes",
    "select_gradient_prefixes",
    "sum_gradient_trees",
    "tree_inner_product",
    "tree_l2_norm",
    "tree_path_string",
]
