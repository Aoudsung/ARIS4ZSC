"""Safe candidate enumeration with window and episode-budget controls."""

from __future__ import annotations

from typing import Any, NamedTuple


class CandidateSet(NamedTuple):
    mask: Any
    probe_allowed: Any
    any_candidate: Any


def enumerate_candidates(
    *,
    safe_action_mask: Any,
    base_action: Any,
    episode_step: Any,
    budget_remaining: Any,
    candidate_window: int,
) -> CandidateSet:
    """Return candidates that are safe, early, budgeted, and not the base action."""

    import jax
    import jax.numpy as jnp

    safe = jnp.asarray(safe_action_mask, dtype=jnp.bool_)
    base = jnp.asarray(base_action, dtype=jnp.int32)
    step = jnp.asarray(episode_step, dtype=jnp.int32)
    budget = jnp.asarray(budget_remaining, dtype=jnp.int32)
    if safe.ndim != 2 or base.shape != safe.shape[:1]:
        raise ValueError("safe_action_mask and base_action must have shapes [batch, action] and [batch].")
    if step.shape != base.shape or budget.shape != base.shape:
        raise ValueError("episode_step and budget_remaining must have shape [batch].")
    if candidate_window <= 0:
        raise ValueError("candidate_window must be positive.")
    differs = ~jax.nn.one_hot(base, safe.shape[-1], dtype=jnp.bool_)
    allowed = (step < int(candidate_window)) & (budget > 0)
    mask = safe & differs & allowed[:, None]
    return CandidateSet(mask=mask, probe_allowed=allowed, any_candidate=jnp.any(mask, axis=-1))
