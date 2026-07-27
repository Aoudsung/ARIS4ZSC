"""Path C V4.3 belief, value quotient, response, and policy mathematics.

The module contains only mathematical operations consumed by the model, trainer,
or evaluator.  Latent slot assignment is based on Bellman evidence; response and
outcome models never define the slot semantics.
"""

from __future__ import annotations

from typing import Any, NamedTuple


INFORMATION_TRIGGER_FLOAT32_ULPS = 256.0
DEPLOYMENT_MODES = (
    "posterior_use",
    "prior_only",
    "reference_only",
    "generic_response_information",
)


class PolicyState(NamedTuple):
    """Complete recurrent state consumed by one deployed policy."""

    reference_carry: Any
    trainable_carry: Any
    control_carry: Any
    slot_log_belief: Any
    previous_action: Any
    previous_team_reward: Any
    episode_start: Any
    log_temperature: Any
    generic_log_temperature: Any


class CodebookState(NamedTuple):
    embeddings: Any
    exponential_counts: Any
    exponential_sums: Any
    unused_rollouts: Any
    initialized: Any
    replacement_count: Any
    last_replaced_codes: Any


class ControlValues(NamedTuple):
    j_use_by_estimator: Any
    j_mask_by_estimator: Any
    j_use: Any
    j_mask: Any
    value_mask: Any
    per_action_response_value: Any
    information_net_value: Any
    response_marginal: Any
    updated_belief: Any
    physical_joint: Any
    next_use_policy_probabilities: Any
    next_mask_policy_probabilities: Any
    per_action_expected_next_policy_tv: Any
    policy_gain_by_estimator: Any
    per_action_policy_mediated_gain: Any


class PolicyValues(NamedTuple):
    logits: Any
    probabilities: Any
    kl_divergence: Any


class PolicyEffectValues(NamedTuple):
    use_policy: PolicyValues
    mask_policy: PolicyValues
    raw_response_effect: Any
    raw_policy_cost: Any
    raw_net_effect: Any
    regularized_net_effect: Any
    total_variation: Any


def uniform_slot_log_belief(batch_shape: tuple[int, ...], slot_count: int) -> Any:
    import jax.numpy as jnp

    if slot_count <= 1:
        raise ValueError("A slot belief requires at least two slots.")
    return jnp.full(
        (*batch_shape, slot_count),
        -jnp.log(jnp.asarray(slot_count, dtype=jnp.float32)),
        dtype=jnp.float32,
    )


def normalized_log_belief(log_belief: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(log_belief, dtype=jnp.float32)
    return values - jax.scipy.special.logsumexp(
        values, axis=-1, keepdims=True
    )


def deployment_belief_after_response(
    *, mode: str, current_log_belief: Any, updated_log_belief: Any
) -> Any:
    if mode not in DEPLOYMENT_MODES:
        raise ValueError("Unknown deployment mode.")
    if mode == "prior_only":
        current = normalized_log_belief(current_log_belief)
        return uniform_slot_log_belief(
            current.shape[:-1], int(current.shape[-1])
        )
    return normalized_log_belief(updated_log_belief)


def _temperature_axes(temperature: Any, action_values: Any) -> tuple[Any, Any]:
    """Broadcast a positive temperature over all non-action axes."""

    import jax.numpy as jnp

    score = jnp.asarray(action_values)
    alpha = jnp.asarray(temperature, dtype=score.dtype)
    if alpha.ndim == score.ndim and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    try:
        scalar = jnp.broadcast_to(alpha, score.shape[:-1])
    except ValueError as error:
        raise ValueError("Temperature cannot broadcast to action-value prefixes.") from error
    if jnp.issubdtype(score.dtype, jnp.floating):
        scalar = jnp.maximum(scalar, jnp.finfo(score.dtype).tiny)
    return scalar[..., None], scalar


def regularized_policy(
    reference_logits: Any, score: Any, temperature: Any
) -> PolicyValues:
    """KL-regularized policy with the reference distribution as the prior."""

    import jax
    import jax.numpy as jnp

    reference_raw = jnp.asarray(reference_logits)
    score_values = jnp.asarray(score, dtype=reference_raw.dtype)
    if reference_raw.shape != score_values.shape:
        raise ValueError("Reference logits and policy scores must share shape.")
    reference_log = jax.nn.log_softmax(reference_raw, axis=-1)
    alpha, unused_scalar = _temperature_axes(temperature, score_values)
    del unused_scalar
    logits = reference_log + score_values / alpha
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    kl = jnp.sum(
        probabilities * (log_probabilities - reference_log), axis=-1
    )
    return PolicyValues(
        logits=logits,
        probabilities=probabilities,
        kl_divergence=jnp.maximum(kl, 0.0),
    )


def regularized_objective_value(
    reference_logits: Any, score: Any, temperature: Any
) -> Any:
    """Return E[score] - alpha KL under the optimal regularized policy."""

    import jax
    import jax.numpy as jnp

    reference = jax.nn.log_softmax(jnp.asarray(reference_logits), axis=-1)
    score_values = jnp.asarray(score, dtype=reference.dtype)
    alpha, scalar_alpha = _temperature_axes(temperature, score_values)
    return scalar_alpha * jax.scipy.special.logsumexp(
        reference + score_values / alpha, axis=-1
    )


def mean_regularized_policy_kl(
    reference_logits: Any, score: Any, temperature: Any
) -> Any:
    import jax.numpy as jnp

    return jnp.mean(
        regularized_policy(reference_logits, score, temperature).kl_divergence
    )


def solve_temperature_for_target_kl(
    *,
    reference_logits: Any,
    score: Any,
    target_kl: float,
    minimum_temperature: float,
    maximum_temperature: float,
    iterations: int,
) -> tuple[Any, Any]:
    """Solve the scalar KL temperature by deterministic log-space bisection."""

    import jax
    import jax.numpy as jnp

    if target_kl < 0.0:
        raise ValueError("Target KL must be non-negative.")
    if minimum_temperature <= 0.0 or maximum_temperature < minimum_temperature:
        raise ValueError("Temperature bounds are invalid.")
    if iterations <= 0:
        raise ValueError("Temperature bisection needs at least one iteration.")

    lower = jnp.log(jnp.asarray(minimum_temperature, dtype=jnp.float32))
    upper = jnp.log(jnp.asarray(maximum_temperature, dtype=jnp.float32))
    target = jnp.asarray(target_kl, dtype=jnp.float32)

    def kl_at(log_alpha: Any) -> Any:
        return mean_regularized_policy_kl(
            reference_logits, score, jnp.exp(log_alpha)
        )

    kl_lower = kl_at(lower)
    kl_upper = kl_at(upper)

    def body(unused_index: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        del unused_index
        lo, hi = bounds
        middle = 0.5 * (lo + hi)
        middle_kl = kl_at(middle)
        # KL decreases monotonically as temperature increases.
        return (
            jnp.where(middle_kl > target, middle, lo),
            jnp.where(middle_kl > target, hi, middle),
        )

    solved_lower, solved_upper = jax.lax.fori_loop(
        0, int(iterations), body, (lower, upper)
    )
    interior = 0.5 * (solved_lower + solved_upper)
    chosen = jnp.where(
        kl_lower <= target,
        lower,
        jnp.where(kl_upper >= target, upper, interior),
    )
    temperature = jnp.exp(chosen)
    return temperature, kl_at(chosen)


def bellman_control_values(
    *,
    slot_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
    next_q_use_mean: Any,
    next_q_mask_mean: Any,
    next_reference_logits_mean: Any,
    temperature: Any,
    gamma: float,
    probability_floor: float = 1.0e-8,
) -> ControlValues:
    """Compute behavior-consistent use/mask values with common physical weights.

    Shapes are ``belief [...,M]``, ``response [...,M,A,Y]``, reward
    ``[...,M,A]``, next Q ``[...,E,M,A,Y,U]`` and next reference logits
    ``[...,A,Y,U]``.  Use and mask share the physical conditional posterior;
    only the controller belief used to construct the next action policy differs.
    """

    import jax.numpy as jnp

    belief = jnp.asarray(slot_belief, dtype=jnp.float32)
    response = jnp.asarray(response_probabilities, dtype=jnp.float32)
    rewards = jnp.asarray(reward_mean, dtype=jnp.float32)
    use_q = jnp.asarray(next_q_use_mean, dtype=jnp.float32)
    mask_q = jnp.asarray(next_q_mask_mean, dtype=jnp.float32)
    next_reference = jnp.asarray(next_reference_logits_mean, dtype=jnp.float32)
    if use_q.shape != mask_q.shape:
        raise ValueError("Use and mask next-Q predictions must share shape.")
    if use_q.shape[-5] != 2:
        raise ValueError("Behavior-consistent control requires two estimators.")
    if response.shape[-3] != belief.shape[-1]:
        raise ValueError("Response slots and belief slots differ.")
    if rewards.shape != response.shape[:-1]:
        raise ValueError("Reward means require [...,slot,action] axes.")
    expected_q_shape = (
        *response.shape[:-3],
        2,
        response.shape[-3],
        response.shape[-2],
        response.shape[-1],
        next_reference.shape[-1],
    )
    if use_q.shape != expected_q_shape:
        raise ValueError(
            "Next-Q means require [...,estimator,slot,action,response,next_action]."
        )
    if next_reference.shape[:-3] != response.shape[:-3] or (
        next_reference.shape[-3] != response.shape[-2]
        or next_reference.shape[-2] != response.shape[-1]
    ):
        raise ValueError(
            "Next reference logits require [...,action,response,next_action]."
        )

    response = jnp.clip(response, float(probability_floor), 1.0)
    response = response / jnp.sum(response, axis=-1, keepdims=True)
    belief = belief / jnp.maximum(
        jnp.sum(belief, axis=-1, keepdims=True), float(probability_floor)
    )
    physical_joint = belief[..., :, None, None] * response
    response_marginal = jnp.sum(physical_joint, axis=-3)
    response_marginal = response_marginal / jnp.maximum(
        jnp.sum(response_marginal, axis=-1, keepdims=True),
        float(probability_floor),
    )
    updated = physical_joint / jnp.maximum(
        response_marginal[..., None, :, :], float(probability_floor)
    )

    use_controller_by_estimator = jnp.einsum(
        "...may,...emayu->...eayu", updated, use_q
    )
    mask_controller_by_estimator = jnp.einsum(
        "...m,...emayu->...eayu", belief, mask_q
    )
    use_controller_score = jnp.min(use_controller_by_estimator, axis=-4)
    mask_controller_score = jnp.min(mask_controller_by_estimator, axis=-4)
    alpha = jnp.asarray(temperature)
    next_alpha = alpha if alpha.ndim == 0 else alpha[..., None, None]
    use_policy = regularized_policy(
        next_reference, use_controller_score, next_alpha
    )
    mask_policy = regularized_policy(
        next_reference, mask_controller_score, next_alpha
    )

    use_slot_value = jnp.einsum(
        "...ayu,...emayu->...emay", use_policy.probabilities, use_q
    )
    mask_slot_value = jnp.einsum(
        "...ayu,...emayu->...emay", mask_policy.probabilities, mask_q
    )
    use_conditional = jnp.einsum(
        "...may,...emay->...eay", updated, use_slot_value
    )
    mask_conditional = jnp.einsum(
        "...may,...emay->...eay", updated, mask_slot_value
    )
    use_expected = jnp.einsum(
        "...ay,...eay->...ea", response_marginal, use_conditional
    )
    mask_expected = jnp.einsum(
        "...ay,...eay->...ea", response_marginal, mask_conditional
    )

    immediate = jnp.einsum("...m,...ma->...a", belief, rewards)
    j_use_by_estimator = immediate[..., None, :] + float(gamma) * use_expected
    j_mask_by_estimator = immediate[..., None, :] + float(gamma) * mask_expected
    j_use = jnp.min(j_use_by_estimator, axis=-2)
    j_mask = jnp.min(j_mask_by_estimator, axis=-2)
    value_mask = jnp.max(j_mask, axis=-1)

    delta_policy = use_policy.probabilities - mask_policy.probabilities
    common_q = 0.5 * (use_q + mask_q)
    gain_by_slot = jnp.einsum(
        "...ayu,...emayu->...emay", delta_policy, common_q
    )
    gain_conditional = jnp.einsum(
        "...may,...emay->...eay", updated, gain_by_slot
    )
    policy_gain_by_estimator = float(gamma) * jnp.einsum(
        "...ay,...eay->...ea", response_marginal, gain_conditional
    )
    policy_mediated_gain = jnp.min(policy_gain_by_estimator, axis=-2)
    next_policy_tv_by_response = 0.5 * jnp.sum(
        jnp.abs(delta_policy), axis=-1
    )
    expected_next_policy_tv = jnp.sum(
        response_marginal * next_policy_tv_by_response, axis=-1
    )

    return ControlValues(
        j_use_by_estimator=j_use_by_estimator,
        j_mask_by_estimator=j_mask_by_estimator,
        j_use=j_use,
        j_mask=j_mask,
        value_mask=value_mask,
        per_action_response_value=j_use - j_mask,
        information_net_value=j_use - value_mask[..., None],
        response_marginal=response_marginal,
        updated_belief=updated,
        physical_joint=physical_joint,
        next_use_policy_probabilities=use_policy.probabilities,
        next_mask_policy_probabilities=mask_policy.probabilities,
        per_action_expected_next_policy_tv=expected_next_policy_tv,
        policy_gain_by_estimator=policy_gain_by_estimator,
        per_action_policy_mediated_gain=policy_mediated_gain,
    )


def generic_response_information(
    *,
    slot_belief: Any,
    response_probabilities: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    import jax.numpy as jnp

    belief = jnp.asarray(slot_belief, dtype=jnp.float32)
    response = jnp.maximum(
        jnp.asarray(response_probabilities, dtype=jnp.float32),
        float(probability_floor),
    )
    response = response / jnp.sum(response, axis=-1, keepdims=True)
    mixture = jnp.einsum("...m,...may->...ay", belief, response)
    mixture_entropy = -jnp.sum(mixture * jnp.log(mixture), axis=-1)
    slot_entropy = -jnp.sum(response * jnp.log(response), axis=-1)
    expected_slot_entropy = jnp.einsum(
        "...m,...ma->...a", belief, slot_entropy
    )
    return jnp.maximum(mixture_entropy - expected_slot_entropy, 0.0)


def policy_effect_decomposition(
    *, reference_logits: Any, j_use: Any, j_mask: Any, temperature: Any
) -> PolicyEffectValues:
    """Predict raw-return effects for the exact current-step policies."""

    import jax.numpy as jnp

    use_policy = regularized_policy(reference_logits, j_use, temperature)
    mask_policy = regularized_policy(reference_logits, j_mask, temperature)
    use_values = jnp.asarray(j_use)
    mask_values = jnp.asarray(j_mask)
    response_effect = jnp.sum(
        use_policy.probabilities * (use_values - mask_values), axis=-1
    )
    policy_cost = jnp.sum(
        (mask_policy.probabilities - use_policy.probabilities) * mask_values,
        axis=-1,
    )
    raw_net = (
        jnp.sum(use_policy.probabilities * use_values, axis=-1)
        - jnp.sum(mask_policy.probabilities * mask_values, axis=-1)
    )
    regularized_net = regularized_objective_value(
        reference_logits, use_values, temperature
    ) - regularized_objective_value(reference_logits, mask_values, temperature)
    total_variation = 0.5 * jnp.sum(
        jnp.abs(use_policy.probabilities - mask_policy.probabilities), axis=-1
    )
    return PolicyEffectValues(
        use_policy=use_policy,
        mask_policy=mask_policy,
        raw_response_effect=response_effect,
        raw_policy_cost=policy_cost,
        raw_net_effect=raw_net,
        regularized_net_effect=regularized_net,
        total_variation=total_variation,
    )


def deployment_policy(
    *,
    mode: str,
    reference_logits: Any,
    control_values: ControlValues,
    information_gain: Any,
    temperature: Any,
    generic_temperature: Any,
) -> PolicyValues:
    import jax

    if mode == "reference_only":
        reference = jax.numpy.asarray(reference_logits)
        return PolicyValues(
            logits=jax.nn.log_softmax(reference, axis=-1),
            probabilities=jax.nn.softmax(reference, axis=-1),
            kl_divergence=jax.numpy.zeros(reference.shape[:-1]),
        )
    if mode == "generic_response_information":
        return regularized_policy(
            reference_logits, information_gain, generic_temperature
        )
    if mode not in {"posterior_use", "prior_only"}:
        raise ValueError("Unknown deployment mode.")
    return regularized_policy(reference_logits, control_values.j_use, temperature)


def policy_effect_trigger_tolerance(j_use: Any, j_mask: Any) -> Any:
    """Float32 numerical floor for a predicted raw-return effect."""

    import jax.numpy as jnp

    use = jnp.asarray(j_use, dtype=jnp.float32)
    mask = jnp.asarray(j_mask, dtype=jnp.float32)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(use), axis=-1),
            jnp.max(jnp.abs(mask), axis=-1),
        ),
    )
    return (
        jnp.asarray(INFORMATION_TRIGGER_FLOAT32_ULPS, dtype=jnp.float32)
        * jnp.finfo(jnp.float32).eps
        * scale
    )

VALUE_EQUIVALENCE_TOLERANCE = 1.0e-3
POSTERIOR_SUPPORT_FLOOR = 1.0e-3


def normalized_advantages(q_values: Any) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    return values - jnp.max(values, axis=-1, keepdims=True)


def value_signatures(q_values: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    if values.shape[-3] != 2:
        raise ValueError("Value signatures require exactly two Q estimators.")
    signatures = normalized_advantages(values)
    signature = 0.5 * (signatures[..., 0, :, :] + signatures[..., 1, :, :])
    radius = 0.5 * jnp.max(
        jnp.abs(signatures[..., 0, :, :] - signatures[..., 1, :, :]),
        axis=-1,
    )
    return signature, radius


def complete_link_value_class_ids(
    signatures: Any,
    radii: Any,
    *,
    tolerance: float = VALUE_EQUIVALENCE_TOLERANCE,
) -> Any:
    """Merge only when the upper confidence bound certifies equivalence."""

    import jax
    import jax.numpy as jnp

    if not 0.0 <= float(tolerance) < float("inf"):
        raise ValueError("Value-class tolerance must be finite and non-negative.")
    signature_values = jnp.asarray(signatures)
    radius_values = jnp.asarray(radii)
    if radius_values.shape != signature_values.shape[:-1]:
        raise ValueError("Signatures and radii have incompatible shapes.")
    slot_count = int(signature_values.shape[-2])
    flat_signatures = signature_values.reshape(
        (-1, slot_count, signature_values.shape[-1])
    )
    flat_radii = radius_values.reshape((-1, slot_count))

    def one(signatures_at_state: Any, radii_at_state: Any) -> Any:
        distances = jnp.max(
            jnp.abs(
                signatures_at_state[:, None, :]
                - signatures_at_state[None, :, :]
            ),
            axis=-1,
        )
        compatible = (
            distances
            + radii_at_state[:, None]
            + radii_at_state[None, :]
        ) <= float(tolerance)
        initial = (
            jnp.full((slot_count,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def assign(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            class_ids, class_count = carry
            indexes = jnp.arange(slot_count, dtype=jnp.int32)
            members = (class_ids[None, :] == indexes[:, None]) & (
                jnp.arange(slot_count) < index
            )[None, :]
            eligible = (
                jnp.any(members, axis=-1)
                & jnp.all(
                    jnp.where(members, compatible[index][None, :], True),
                    axis=-1,
                )
                & (indexes < class_count)
            )
            first = jnp.min(
                jnp.where(eligible, indexes, jnp.asarray(slot_count, jnp.int32))
            )
            selected = jnp.where(first < slot_count, first, class_count)
            return class_ids.at[index].set(selected), jnp.maximum(
                class_count, selected + 1
            )

        return jax.lax.fori_loop(0, slot_count, assign, initial)[0]

    flat = jax.vmap(one)(flat_signatures, flat_radii)
    return flat.reshape(signature_values.shape[:-2] + (slot_count,))


def class_membership(class_ids: Any, slot_count: int) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    membership = jax.nn.one_hot(
        jnp.asarray(class_ids, dtype=jnp.int32), slot_count, dtype=jnp.float32
    )
    counts = jnp.sum(membership, axis=-2)
    return membership, counts, counts > 0.0


def normalized_slot_probabilities(slot_log_belief: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(slot_log_belief, dtype=jnp.float32)
    return jnp.exp(
        values - jax.scipy.special.logsumexp(values, axis=-1, keepdims=True)
    )


def class_sum(values: Any, membership: Any) -> Any:
    import jax.numpy as jnp

    return jnp.einsum("...mc,...m->...c", membership, values)


def posterior_supported_value_class_count(
    *,
    class_ids: Any,
    slot_log_belief: Any,
    probability_floor: float = POSTERIOR_SUPPORT_FLOOR,
) -> Any:
    import jax.numpy as jnp

    floor = float(probability_floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("Posterior support floor must lie in [0, 1).")
    probabilities = normalized_slot_probabilities(slot_log_belief)
    slot_count = int(probabilities.shape[-1])
    membership, unused_counts, class_mask = class_membership(
        class_ids, slot_count
    )
    del unused_counts
    class_belief = class_sum(probabilities, membership)
    return jnp.sum(
        class_mask & (class_belief >= floor), axis=-1, dtype=jnp.int32
    )


def slot_bayes_update(
    *,
    slot_log_belief: Any,
    slot_response_probabilities: Any,
    action: Any,
    response_code: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    import jax
    import jax.numpy as jnp

    probabilities = jnp.asarray(slot_response_probabilities)
    action_index = jnp.asarray(action, dtype=jnp.int32)
    response_index = jnp.asarray(response_code, dtype=jnp.int32)
    selected_action = jnp.take_along_axis(
        probabilities, action_index[..., None, None, None], axis=-2
    )[..., 0, :]
    likelihood = jnp.take_along_axis(
        selected_action, response_index[..., None, None], axis=-1
    )[..., 0]
    posterior = jnp.asarray(slot_log_belief) + jnp.log(
        jnp.clip(likelihood, float(probability_floor), 1.0)
    )
    return posterior - jax.scipy.special.logsumexp(
        posterior, axis=-1, keepdims=True
    )


class CodebookUpdate(NamedTuple):
    state: CodebookState
    assignments: Any
    quantization_error: Any
    replaced_codes: Any


def empty_codebook(*, code_count: int, signature_dim: int) -> CodebookState:
    import jax.numpy as jnp

    if code_count <= 1 or signature_dim <= 0:
        raise ValueError("Codebook dimensions must be positive.")
    return CodebookState(
        embeddings=jnp.zeros((code_count, signature_dim), dtype=jnp.float32),
        exponential_counts=jnp.zeros((code_count,), dtype=jnp.float32),
        exponential_sums=jnp.zeros(
            (code_count, signature_dim), dtype=jnp.float32
        ),
        unused_rollouts=jnp.zeros((code_count,), dtype=jnp.int32),
        initialized=jnp.asarray(False),
        replacement_count=jnp.asarray(0, dtype=jnp.int32),
        last_replaced_codes=jnp.zeros((code_count,), dtype=jnp.bool_),
    )


def farthest_point_initialization(
    signatures: Any, key: Any, code_count: int, valid_mask: Any | None = None
) -> Any:
    """Choose one keyed center and then the farthest signature repeatedly."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[0] < code_count:
        raise ValueError("Codebook initialization needs at least one row per code.")
    valid = (
        jnp.ones((values.shape[0],), dtype=jnp.bool_)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.bool_)
    )
    if valid.shape != (values.shape[0],):
        raise ValueError("Codebook valid mask has the wrong shape.")
    probabilities = valid.astype(jnp.float32)
    probabilities = probabilities / jnp.maximum(jnp.sum(probabilities), 1.0)
    first = jax.random.choice(key, values.shape[0], shape=(), p=probabilities)
    centers = jnp.zeros((code_count, values.shape[-1]), dtype=values.dtype)
    centers = centers.at[0].set(values[first])
    minimum_distance = jnp.where(
        valid,
        jnp.sum(jnp.square(values - centers[0]), axis=-1),
        -jnp.inf,
    )

    def add_center(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        current_centers, distances = carry
        selected = jnp.argmax(distances)
        center = values[selected]
        current_centers = current_centers.at[index].set(center)
        candidate_distance = jnp.where(
            valid,
            jnp.sum(jnp.square(values - center), axis=-1),
            -jnp.inf,
        )
        return current_centers, jnp.where(
            valid, jnp.minimum(distances, candidate_distance), -jnp.inf
        )

    return jax.lax.fori_loop(
        1, code_count, add_center, (centers, minimum_distance)
    )[0]


def nearest_codes(signatures: Any, embeddings: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    codes = jnp.asarray(embeddings, dtype=jnp.float32)
    distances = jnp.sum(
        jnp.square(values[..., None, :] - codes), axis=-1
    )
    assignments = jnp.argmin(distances, axis=-1)
    error = jnp.take_along_axis(distances, assignments[..., None], axis=-1)[..., 0]
    return assignments, error


def update_codebook(
    state: CodebookState,
    *,
    signatures: Any,
    key: Any,
    valid_mask: Any | None = None,
    decay: float = 0.99,
    replacement_after_rollouts: int = 10,
    epsilon: float = 1.0e-5,
) -> CodebookUpdate:
    """Apply initialization, exponential moving averages, and stale-code repair."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32).reshape(
        (-1, state.embeddings.shape[-1])
    )
    valid = (
        jnp.ones((values.shape[0],), dtype=jnp.float32)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.float32).reshape((-1,))
    )
    if valid.shape != (values.shape[0],):
        raise ValueError("Codebook valid mask must align with signatures.")
    code_count = int(state.embeddings.shape[0])
    initialized_embeddings = jax.lax.cond(
        state.initialized,
        lambda unused: state.embeddings,
        lambda unused: farthest_point_initialization(
            values, key, code_count, valid > 0.0
        ),
        operand=None,
    )
    assignments, error = nearest_codes(values, initialized_embeddings)
    one_hot = jax.nn.one_hot(assignments, code_count, dtype=jnp.float32)
    weighted_one_hot = one_hot * valid[:, None]
    batch_counts = jnp.sum(weighted_one_hot, axis=0)
    batch_sums = jnp.einsum("nc,nd->cd", weighted_one_hot, values)
    old_counts = jnp.where(
        state.initialized, state.exponential_counts, jnp.zeros_like(batch_counts)
    )
    old_sums = jnp.where(
        state.initialized,
        state.exponential_sums,
        jnp.zeros_like(batch_sums),
    )
    exponential_counts = float(decay) * old_counts + (1.0 - float(decay)) * batch_counts
    exponential_sums = float(decay) * old_sums + (1.0 - float(decay)) * batch_sums
    candidate_embeddings = exponential_sums / jnp.maximum(
        exponential_counts[:, None], epsilon
    )
    candidate_embeddings = jnp.where(
        (exponential_counts > epsilon)[:, None],
        candidate_embeddings,
        initialized_embeddings,
    )
    unused = jnp.where(
        batch_counts > 0.0,
        jnp.zeros_like(state.unused_rollouts),
        state.unused_rollouts + 1,
    )
    stale = unused >= int(replacement_after_rollouts)
    descending_error_indexes = jnp.argsort(
        jnp.where(valid > 0.0, error, -jnp.inf)
    )[::-1]
    replacement_rows = values[
        descending_error_indexes[
            jnp.arange(code_count, dtype=jnp.int32) % values.shape[0]
        ]
    ]
    embeddings = jnp.where(stale[:, None], replacement_rows, candidate_embeddings)
    exponential_counts = jnp.where(stale, jnp.ones_like(exponential_counts), exponential_counts)
    exponential_sums = jnp.where(
        stale[:, None], replacement_rows, exponential_sums
    )
    unused = jnp.where(stale, jnp.zeros_like(unused), unused)
    next_state = CodebookState(
        embeddings=embeddings,
        exponential_counts=exponential_counts,
        exponential_sums=exponential_sums,
        unused_rollouts=unused,
        initialized=jnp.asarray(True),
        replacement_count=state.replacement_count + jnp.sum(stale.astype(jnp.int32)),
        last_replaced_codes=stale,
    )
    return CodebookUpdate(
        state=next_state,
        assignments=assignments.reshape(jnp.asarray(signatures).shape[:-1]),
        quantization_error=error.reshape(jnp.asarray(signatures).shape[:-1]),
        replaced_codes=stale,
    )


def target_response_signatures(
    *,
    current_centered_advantages: Any,
    next_centered_advantages: Any,
) -> Any:
    """Encode the value-relevant change induced by one physical transition.

    Both tensors have tail axes ``[estimator, slot, action]`` and are evaluated
    at the same canonical uniform belief.  The response signature concatenates
    the mean and standard deviation of the centered-advantage delta across the
    permutation-symmetric estimator/slot bank.  No responsibility or identity
    label enters the response alphabet.
    """

    import jax.numpy as jnp

    current = jnp.asarray(current_centered_advantages)
    following = jnp.asarray(next_centered_advantages)
    if current.shape != following.shape or current.shape[-3] != 2:
        raise ValueError(
            "Response targets require matching [..,2,slot,action] tensors."
        )
    delta = following - current
    mean = jnp.mean(delta, axis=(-3, -2))
    standard_deviation = jnp.std(delta, axis=(-3, -2))
    return jnp.concatenate((mean, standard_deviation), axis=-1)


__all__ = [
    "CodebookState",
    "CodebookUpdate",
    "ControlValues",
    "DEPLOYMENT_MODES",
    "INFORMATION_TRIGGER_FLOAT32_ULPS",
    "PolicyEffectValues",
    "PolicyState",
    "PolicyValues",
    "POSTERIOR_SUPPORT_FLOOR",
    "VALUE_EQUIVALENCE_TOLERANCE",
    "bellman_control_values",
    "class_membership",
    "class_sum",
    "complete_link_value_class_ids",
    "deployment_belief_after_response",
    "deployment_policy",
    "empty_codebook",
    "farthest_point_initialization",
    "generic_response_information",
    "mean_regularized_policy_kl",
    "nearest_codes",
    "normalized_advantages",
    "normalized_log_belief",
    "normalized_slot_probabilities",
    "policy_effect_decomposition",
    "policy_effect_trigger_tolerance",
    "posterior_supported_value_class_count",
    "regularized_objective_value",
    "regularized_policy",
    "slot_bayes_update",
    "solve_temperature_for_target_kl",
    "target_response_signatures",
    "uniform_slot_log_belief",
    "update_codebook",
    "value_signatures",
]
