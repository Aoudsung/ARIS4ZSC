"""One low-rank continuously context-conditioned coordination actor.

METHOD_SPEC §1.3: the actor interface is ``(task_features, context)`` with
``context = concat(r_t, u, c_t)``.  Low-rank modulation is kept:
``task_basis(x) ⊙ context_gain(concat(r, u, c))``.  Still a single actor for all
partner types/sources.
"""

from __future__ import annotations

from typing import Any

_ACTOR: Any | None = None


def universal_actor_class() -> Any:
    global _ACTOR
    if _ACTOR is not None:
        return _ACTOR

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class UniversalCoordinationActor(nn.Module):
        action_count: int
        hidden_dim: int
        modulation_rank: int

        @nn.compact
        def __call__(self, task_features: Any, context: Any) -> Any:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            joined_context = jnp.asarray(context, dtype=jnp.float32)
            if task.shape[:-1] != joined_context.shape[:-1]:
                raise ValueError("Actor task and context batch axes differ.")
            trunk = nn.tanh(nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=zeros,
                name="task_hidden",
            )(task))
            task_basis = nn.Dense(
                self.modulation_rank,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="task_low_rank_basis",
            )(trunk)
            context_gain = nn.tanh(nn.Dense(
                self.modulation_rank,
                kernel_init=orthogonal(0.5),
                bias_init=zeros,
                name="context_low_rank_gain",
            )(joined_context))
            interaction = nn.tanh(nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="interaction_hidden",
            )(task_basis * context_gain))
            joined = trunk + interaction
            return nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="policy_logits",
            )(joined)

    _ACTOR = UniversalCoordinationActor
    return UniversalCoordinationActor


__all__ = ["universal_actor_class"]
