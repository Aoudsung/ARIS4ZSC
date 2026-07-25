"""Four matched probe controllers with one structured output."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, NamedTuple

from .scores import SequentialScores


class ControllerDecision(NamedTuple):
    chosen_action: Any
    probed: Any
    candidate: Any
    j_use: Any
    j_mask: Any
    v_base: Any
    v_mask: Any
    s_seq: Any
    response_information: Any
    budget_remaining: Any
    actor_owned_action: Any

    @property
    def score_components(self) -> dict[str, Any]:
        """Expose one common named score record for logging and review."""

        return {
            "j_use": self.j_use,
            "j_mask": self.j_mask,
            "v_base": self.v_base,
            "v_mask": self.v_mask,
            "s_seq": self.s_seq,
            "response_information": self.response_information,
        }


def _common_decision(
    *,
    scores: Any,
    candidate_mask: Any,
    base_action: Any,
    budget_remaining: Any,
    threshold: float,
) -> tuple[Any, Any, Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(scores)
    mask = jnp.asarray(candidate_mask, dtype=jnp.bool_)
    budget_available = jnp.asarray(budget_remaining, dtype=jnp.int32) > 0
    mask = mask & budget_available[:, None]
    safe_values = jnp.where(mask, values, -jnp.inf)
    candidate = jnp.argmax(safe_values, axis=-1)
    best = jnp.max(safe_values, axis=-1)
    has_candidate = jnp.any(mask, axis=-1)
    if isinstance(threshold, Real) and not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite.")
    threshold_value = jnp.asarray(threshold, dtype=values.dtype)
    probed = has_candidate & (best > threshold_value)
    chosen = jnp.where(probed, candidate, jnp.asarray(base_action, dtype=jnp.int32))
    remaining = jnp.maximum(jnp.asarray(budget_remaining, dtype=jnp.int32) - probed.astype(jnp.int32), 0)
    return chosen, probed, candidate, remaining


def sequential_controller(
    *, scores: SequentialScores, candidate_mask: Any, base_action: Any,
    budget_remaining: Any, threshold: float
) -> ControllerDecision:
    import jax.numpy as jnp

    chosen, probed, candidate, remaining = _common_decision(
        scores=scores.s_seq,
        candidate_mask=candidate_mask,
        base_action=base_action,
        budget_remaining=budget_remaining,
        threshold=threshold,
    )
    gather = candidate[:, None]
    chosen_score = jnp.take_along_axis(scores.s_seq, gather, axis=-1)[:, 0]
    nan = jnp.full(chosen_score.shape, jnp.nan, dtype=jnp.float32)
    return ControllerDecision(
        chosen_action=chosen,
        probed=probed,
        candidate=candidate,
        j_use=jnp.take_along_axis(scores.j_use, gather, axis=-1)[:, 0],
        j_mask=jnp.take_along_axis(scores.j_mask, gather, axis=-1)[:, 0],
        v_base=scores.v_base,
        v_mask=scores.v_mask,
        s_seq=chosen_score,
        response_information=nan,
        budget_remaining=remaining,
        actor_owned_action=~probed,
    )


def information_controller(
    *, information_scores: Any, candidate_mask: Any, base_action: Any,
    budget_remaining: Any, threshold: float
) -> ControllerDecision:
    import jax.numpy as jnp

    chosen, probed, candidate, remaining = _common_decision(
        scores=information_scores,
        candidate_mask=candidate_mask,
        base_action=base_action,
        budget_remaining=budget_remaining,
        threshold=threshold,
    )
    score = jnp.take_along_axis(information_scores, candidate[:, None], axis=-1)[:, 0]
    available = jnp.any(
        jnp.asarray(candidate_mask, dtype=jnp.bool_)
        & (jnp.asarray(budget_remaining, dtype=jnp.int32) > 0)[:, None],
        axis=-1,
    )
    score = jnp.where(available, score, -jnp.inf)
    nan = jnp.full(score.shape, jnp.nan, dtype=jnp.float32)
    return ControllerDecision(
        chosen, probed, candidate, nan, nan, nan, nan, nan, score, remaining, ~probed
    )


def random_safe_controller(
    *, key: Any, candidate_mask: Any, base_action: Any, budget_remaining: Any,
    trigger_probability: float
) -> ControllerDecision:
    import jax
    import jax.numpy as jnp

    if isinstance(trigger_probability, Real) and not 0.0 <= float(
        trigger_probability
    ) <= 1.0:
        raise ValueError("trigger_probability must lie in [0, 1].")
    probability = jnp.asarray(trigger_probability, dtype=jnp.float32)
    mask = jnp.asarray(candidate_mask, dtype=jnp.bool_)
    budget_available = jnp.asarray(budget_remaining, dtype=jnp.int32) > 0
    mask = mask & budget_available[:, None]
    key_array = jnp.asarray(key)
    if key_array.shape == (2,):
        trigger_key, action_key = jax.random.split(key_array)
        trigger_values = jax.random.uniform(trigger_key, (mask.shape[0],))
        sample_candidate = lambda logits: jax.random.categorical(action_key, logits, axis=-1)
    elif key_array.shape == (mask.shape[0], 2):
        split_keys = jax.vmap(lambda value: jax.random.split(value, 2))(key_array)
        trigger_values = jax.vmap(lambda value: jax.random.uniform(value, ()))(
            split_keys[:, 0]
        )
        sample_candidate = lambda logits: jax.vmap(
            lambda value, row: jax.random.categorical(value, row, axis=-1)
        )(split_keys[:, 1], logits)
    else:
        raise ValueError("Random-safe control requires one root key or one key per batch row.")
    has_candidate = jnp.any(mask, axis=-1)
    probed = (trigger_values < probability) & has_candidate
    choosable = jnp.where(
        has_candidate[:, None], mask, jnp.ones_like(mask, dtype=jnp.bool_)
    )
    logits = jnp.where(choosable, 0.0, -jnp.inf)
    candidate = sample_candidate(logits)
    chosen = jnp.where(probed, candidate, jnp.asarray(base_action, dtype=jnp.int32))
    remaining = jnp.maximum(jnp.asarray(budget_remaining, dtype=jnp.int32) - probed.astype(jnp.int32), 0)
    nan = jnp.full((mask.shape[0],), jnp.nan, dtype=jnp.float32)
    return ControllerDecision(
        chosen, probed, candidate, nan, nan, nan, nan, nan, nan, remaining, ~probed
    )


def off_controller(*, base_action: Any, budget_remaining: Any) -> ControllerDecision:
    import jax.numpy as jnp

    base = jnp.asarray(base_action, dtype=jnp.int32)
    false = jnp.zeros(base.shape, dtype=jnp.bool_)
    nan = jnp.full(base.shape, jnp.nan, dtype=jnp.float32)
    return ControllerDecision(
        base,
        false,
        base,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        jnp.asarray(budget_remaining),
        ~false,
    )


def select_action(
    controller: str,
    *,
    key: Any,
    sequential: SequentialScores,
    information_scores: Any,
    candidate_mask: Any,
    base_action: Any,
    budget_remaining: Any,
    decision_threshold: float,
    information_threshold: float,
    random_trigger_probability: float,
) -> ControllerDecision:
    """Dispatch one registered controller before entering a compiled rollout."""

    if controller == "registered_response_sequential_branch_v1":
        return sequential_controller(
            scores=sequential, candidate_mask=candidate_mask, base_action=base_action,
            budget_remaining=budget_remaining, threshold=decision_threshold,
        )
    if controller == "generic_response_information":
        return information_controller(
            information_scores=information_scores, candidate_mask=candidate_mask,
            base_action=base_action, budget_remaining=budget_remaining,
            threshold=information_threshold,
        )
    if controller == "registered_random_safe_probe_v1":
        return random_safe_controller(
            key=key, candidate_mask=candidate_mask, base_action=base_action,
            budget_remaining=budget_remaining, trigger_probability=random_trigger_probability,
        )
    if controller == "off":
        return off_controller(base_action=base_action, budget_remaining=budget_remaining)
    raise ValueError(f"Unknown controller {controller!r}.")
