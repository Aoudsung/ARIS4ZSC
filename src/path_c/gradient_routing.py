"""V6 parameter ownership and normalized multi-objective belief gradients."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


BELIEF_OBJECTIVE_ORDER = (
    "ppo",
    "raw_q",
    "counterfactual",
    "response",
    "decision_equivalence",
    "information_bottleneck",
    "q_policy",
    "robust",
)


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


def keep_owned_gradients(gradients: Any, *, loss_name: str) -> Any:
    """Route head gradients while excluding belief from all head optimizers."""

    import jax
    import jax.numpy as jnp

    def owned(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        if name.startswith("belief_encoder/"):
            allowed = False
        elif name.startswith("task_encoder/"):
            allowed = loss_name in {
                "ppo",
                "raw_q",
                "counterfactual",
                "owner_distillation",
            }
        elif name.startswith("universal_actor/"):
            allowed = loss_name in {"ppo", "q_policy", "robust", "owner_distillation"}
        elif name.startswith("universal_critic/"):
            raw_head = "/raw_q" in f"/{name}"
            allowed = loss_name in ({"raw_q", "counterfactual"} if raw_head else {"ppo"})
        elif name.startswith("response_decoder/"):
            allowed = loss_name == "response"
        else:
            allowed = False
        return value if allowed else jnp.zeros_like(value)

    return jax.tree_util.tree_map_with_path(owned, gradients)


def keep_gradients_for_losses(gradients: Any, *, loss_names: tuple[str, ...]) -> Any:
    return sum_gradient_trees(*(keep_owned_gradients(gradients, loss_name=name) for name in loss_names))


def scale_gradient_prefixes(gradients: Any, prefixes: tuple[str, ...], scale: Any) -> Any:
    import jax

    def scale_leaf(path: Any, value: Any) -> Any:
        name = tree_path_string(path)
        return value * scale if any(name.startswith(prefix + "/") for prefix in prefixes) else value

    return jax.tree_util.tree_map_with_path(scale_leaf, gradients)


def project_conflicting_gradient(
    gradient: Any, reference: Any, *, epsilon: float = 1.0e-8
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    dot = tree_inner_product(gradient, reference)
    reference_square = tree_inner_product(reference, reference)
    coefficient = jnp.minimum(dot, 0.0) / (reference_square + float(epsilon))
    projected = jax.tree_util.tree_map(
        lambda value, base: value - coefficient * base, gradient, reference
    )
    return projected, dot


def combine_belief_gradients(
    objective_gradients: Mapping[str, Any],
    norm_ema: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    decay: float = 0.99,
    epsilon: float = 1.0e-8,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """EMA-normalize, cap, deterministic-PCGrad and combine one snapshot.

    The EMA removes persistent scale differences, while the instantaneous cap
    prevents a newly spiking objective from dominating during the EMA's
    adaptation window.  After weighting, every objective has L2 norm at most
    the absolute value of its registered weight.
    """

    import jax
    import jax.numpy as jnp

    missing = set(BELIEF_OBJECTIVE_ORDER) - set(objective_gradients)
    if missing:
        raise ValueError(f"Missing V6 belief objectives: {sorted(missing)}")
    normalized: list[Any] = []
    next_ema: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for name in BELIEF_OBJECTIVE_ORDER:
        belief_gradient = select_gradient_prefixes(
            objective_gradients[name], ("belief_encoder",)
        )
        norm = tree_l2_norm(belief_gradient)
        previous = jnp.asarray(norm_ema.get(name, 0.0), dtype=jnp.float32)
        ema = float(decay) * previous + (1.0 - float(decay)) * norm
        next_ema[name] = ema
        ema_scaled = jax.tree_util.tree_map(
            lambda value: value * float(weights[name]) / (ema + float(epsilon)),
            belief_gradient,
        )
        ema_scaled_norm = tree_l2_norm(ema_scaled)
        registered_norm = abs(float(weights[name]))
        cap_scale = jnp.minimum(
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(registered_norm, dtype=jnp.float32)
            / (ema_scaled_norm + float(epsilon)),
        )
        scaled = jax.tree_util.tree_map(lambda value: value * cap_scale, ema_scaled)
        metrics[f"belief_gradient_raw_norm/{name}"] = norm
        metrics[f"belief_gradient_ema_normalized_norm/{name}"] = ema_scaled_norm
        metrics[f"belief_gradient_normalized_norm/{name}"] = tree_l2_norm(scaled)
        normalized.append(scaled)

    # Report cosines between the unmodified, EMA-normalized objectives.  The
    # subsequent deterministic PCGrad pass may change its reference vectors;
    # those projected vectors must not be mislabeled as the original conflict
    # evidence.
    for index, gradient in enumerate(normalized):
        for previous_index in range(index):
            reference = normalized[previous_index]
            denominator = (
                tree_l2_norm(gradient) * tree_l2_norm(reference) + float(epsilon)
            )
            metrics[
                f"belief_gradient_cosine/{BELIEF_OBJECTIVE_ORDER[index]}_vs_"
                f"{BELIEF_OBJECTIVE_ORDER[previous_index]}"
            ] = tree_inner_product(gradient, reference) / denominator

    projected: list[Any] = []
    for index, gradient in enumerate(normalized):
        candidate = gradient
        for reference in projected:
            candidate, unused_dot = project_conflicting_gradient(
                candidate, reference, epsilon=epsilon
            )
            del unused_dot
        projected.append(candidate)
    combined = sum_gradient_trees(*projected)
    metrics["belief_gradient_combined_norm"] = tree_l2_norm(combined)
    return combined, next_ema, metrics


def project_response_gradient(response_gradient: Any, decision_gradient: Any, *, epsilon: float = 1.0e-8):
    """Compatibility helper implemented by the same V6 projection primitive."""

    projected, dot = project_conflicting_gradient(response_gradient, decision_gradient, epsilon=epsilon)
    return projected, {
        "pre_projection_dot": dot,
        "post_projection_dot": tree_inner_product(projected, decision_gradient),
        "decision_norm": tree_l2_norm(decision_gradient),
        "response_norm_before_cap": tree_l2_norm(response_gradient),
    }


def route_response_with_pcgrad(response_gradients: Any, decision_gradients: Any):
    response_belief = select_gradient_prefixes(response_gradients, ("belief_encoder",))
    decision_belief = select_gradient_prefixes(decision_gradients, ("belief_encoder",))
    projected, metrics = project_response_gradient(response_belief, decision_belief)
    decoder = select_gradient_prefixes(response_gradients, ("response_decoder",))
    return sum_gradient_trees(projected, decoder), metrics


__all__ = [
    "BELIEF_OBJECTIVE_ORDER",
    "combine_belief_gradients",
    "keep_owned_gradients",
    "keep_gradients_for_losses",
    "project_conflicting_gradient",
    "project_response_gradient",
    "remove_gradient_prefixes",
    "route_response_with_pcgrad",
    "scale_gradient_prefixes",
    "select_gradient_prefixes",
    "sum_gradient_trees",
    "tree_inner_product",
    "tree_l2_norm",
    "tree_path_string",
]
