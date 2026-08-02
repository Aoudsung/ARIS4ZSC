"""Explicit r3 optimizer ownership and response PCGrad routing."""

from __future__ import annotations

from typing import Any, Mapping


PARAMETER_GRADIENT_OWNERS: Mapping[str, frozenset[str]] = {
    "task_encoder": frozenset(("ppo", "raw_q", "base_distillation")),
    "belief_encoder": frozenset(("raw_q", "counterfactual", "response")),
    "belief_set_encoder": frozenset(("raw_q", "counterfactual", "response")),
    "universal_actor/base": frozenset(("base_ppo", "base_distillation")),
    "universal_actor/residual": frozenset(("conditional_ppo", "q_policy")),
    "universal_critic/shaped_value": frozenset(("ppo_value",)),
    "universal_critic/raw_q": frozenset(("raw_q", "counterfactual")),
    "response_decoder": frozenset(("response",)),
    "partner_generator": frozenset(("generator_ppo", "br_diversity")),
}


def tree_inner_product(first: Any, second: Any) -> Any:
    import jax
    import jax.numpy as jnp

    leaves_a = jax.tree_util.tree_leaves(first)
    leaves_b = jax.tree_util.tree_leaves(second)
    if len(leaves_a) != len(leaves_b):
        raise ValueError("Gradient trees differ.")
    if not leaves_a:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return sum(
        (jnp.vdot(jnp.asarray(a), jnp.asarray(b)) for a, b in zip(leaves_a, leaves_b)),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def tree_l2_norm(tree: Any) -> Any:
    import jax.numpy as jnp

    return jnp.sqrt(jnp.maximum(tree_inner_product(tree, tree), 0.0))


def project_response_gradient(
    response_gradient: Any,
    decision_gradient: Any,
    *,
    epsilon: float = 1e-12,
) -> tuple[Any, Mapping[str, Any]]:
    """Project conflicting response gradients and cap them at decision norm."""

    import jax
    import jax.numpy as jnp

    dot = tree_inner_product(response_gradient, decision_gradient)
    decision_sq = tree_inner_product(decision_gradient, decision_gradient)
    coefficient = jnp.minimum(dot, 0.0) / (decision_sq + float(epsilon))
    projected = jax.tree_util.tree_map(
        lambda response, decision: response - coefficient * decision,
        response_gradient,
        decision_gradient,
    )
    decision_norm = jnp.sqrt(jnp.maximum(decision_sq, 0.0))
    projected_norm = tree_l2_norm(projected)
    scale = jnp.minimum(1.0, decision_norm / (projected_norm + float(epsilon)))
    projected = jax.tree_util.tree_map(lambda value: value * scale, projected)
    return projected, {
        "pre_projection_dot": dot,
        "projection_coefficient": coefficient,
        "decision_norm": decision_norm,
        "response_norm_before_cap": projected_norm,
        "response_cap_scale": scale,
        "post_projection_dot": tree_inner_product(projected, decision_gradient),
    }


def assert_gradient_owner(parameter_path: str, loss_name: str) -> None:
    matching = [
        owners
        for prefix, owners in PARAMETER_GRADIENT_OWNERS.items()
        if parameter_path == prefix or parameter_path.startswith(prefix + "/")
    ]
    if not matching or loss_name not in matching[0]:
        raise ValueError(f"Loss {loss_name!r} cannot update {parameter_path!r}.")


def tree_path_string(path: Any) -> str:
    parts = []
    for entry in path:
        value = getattr(entry, "key", getattr(entry, "name", getattr(entry, "idx", entry)))
        parts.append(str(value))
    return "/".join(parts)


def keep_owned_gradients(gradients: Any, *, loss_name: str) -> Any:
    """Zero every parameter leaf not owned by one r3 optimizer.

    Flax module paths are translated to the semantic ownership table here so a
    newly added subtree fails closed instead of receiving a global gradient.
    """

    import jax
    import jax.numpy as jnp

    def owned(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        allowed = False
        if name.startswith("task_encoder/"):
            allowed = loss_name in {"ppo", "raw_q", "base_distillation"}
        elif name.startswith("belief_encoder/") or name.startswith("belief_set_encoder/"):
            allowed = loss_name in {"raw_q", "counterfactual", "response"}
        elif name.startswith("universal_actor/"):
            if "/context_residual/" in f"/{name}/":
                allowed = loss_name in {"conditional_ppo", "q_policy"}
            else:
                allowed = loss_name in {"base_ppo", "base_distillation"}
        elif name.startswith("universal_critic/"):
            if "/raw_q" in f"/{name}":
                allowed = loss_name in {"raw_q", "counterfactual"}
            else:
                allowed = loss_name == "ppo_value"
        elif name.startswith("response_decoder/"):
            allowed = loss_name == "response"
        return value if allowed else jnp.zeros_like(value)

    return jax.tree_util.tree_map_with_path(owned, gradients)


def keep_gradients_for_losses(gradients: Any, *, loss_names: tuple[str, ...]) -> Any:
    import jax

    routed = tuple(
        keep_owned_gradients(gradients, loss_name=name) for name in loss_names
    )
    # Registered ownership sets for one combined auxiliary update are
    # disjoint; summing is therefore the exact union and remains JIT-safe.
    return jax.tree_util.tree_map(lambda *values: sum(values), *routed)


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


def scale_gradient_prefixes(
    gradients: Any, prefixes: tuple[str, ...], scale: Any
) -> Any:
    """Apply a dynamic learning-rate multiplier without changing Optax state."""

    import jax

    def scale_leaf(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value * scale if any(
            name.startswith(prefix + "/") for prefix in prefixes
        ) else value

    return jax.tree_util.tree_map_with_path(scale_leaf, gradients)


def scale_base_policy_gradients(
    gradients: Any,
    *,
    task_trunk_scale: Any,
    base_actor_scale: Any,
) -> Any:
    """Scale the task trunk and base actor independently.

    Before C0, the task trunk is frozen while the base actor remains trainable
    against the exclusive qualified external support.  After C0 both receive
    the registered 0.1 multiplier.  The conditional residual is never touched
    by this helper.
    """

    import jax

    def scale_leaf(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        base_actor = name.startswith("universal_actor/") and (
            "/context_residual/" not in f"/{name}/"
        )
        if name.startswith("task_encoder/"):
            return value * task_trunk_scale
        if base_actor:
            return value * base_actor_scale
        return value

    return jax.tree_util.tree_map_with_path(scale_leaf, gradients)


def route_response_with_pcgrad(
    response_gradients: Any,
    decision_gradients: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Project/cap only shared belief gradients; decoder gradients stay separate."""

    belief_prefixes = ("belief_encoder", "belief_set_encoder")
    response_belief = select_gradient_prefixes(response_gradients, belief_prefixes)
    decision_belief = select_gradient_prefixes(decision_gradients, belief_prefixes)
    projected, metrics = project_response_gradient(response_belief, decision_belief)
    decoder = select_gradient_prefixes(response_gradients, ("response_decoder",))
    return sum_gradient_trees(projected, decoder), metrics


__all__ = [
    "PARAMETER_GRADIENT_OWNERS",
    "assert_gradient_owner",
    "keep_owned_gradients",
    "keep_gradients_for_losses",
    "project_response_gradient",
    "route_response_with_pcgrad",
    "scale_gradient_prefixes",
    "scale_base_policy_gradients",
    "select_gradient_prefixes",
    "sum_gradient_trees",
    "tree_path_string",
    "tree_inner_product",
    "tree_l2_norm",
]
