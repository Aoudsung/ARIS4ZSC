"""Single fixed-capacity actor with low-rank belief modulation."""

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

    class LowRankContextResidual(nn.Module):
        output_dim: int
        rank: int
        hidden_dim: int

        @nn.compact
        def __call__(self, features: Any, context: Any) -> Any:
            x = jnp.asarray(features, dtype=jnp.float32)
            eta = jnp.asarray(context, dtype=jnp.float32)
            if x.shape[:-1] != eta.shape[:-1]:
                raise ValueError("Actor task and belief batch axes differ.")
            task_basis = nn.Dense(
                self.rank,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="task_low_rank_basis",
            )(x)
            context_gain = nn.tanh(
                nn.Dense(
                    self.rank,
                    kernel_init=orthogonal(0.5),
                    bias_init=zeros,
                    name="context_low_rank_gain",
                )(eta)
            )
            interaction = task_basis * context_gain
            interaction = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=zeros,
                    name="interaction_hidden",
                )(interaction)
            )
            return nn.Dense(
                self.output_dim,
                # C0 requires exact base equivalence before any conditional
                # coupling.  Zeroing the last layer makes gate=0 and gate=1
                # identical at initialization without disabling upstream
                # representation learning.
                kernel_init=zeros,
                bias_init=zeros,
                name="residual_logits",
            )(interaction)

    class UniversalCoordinationActor(nn.Module):
        action_count: int
        hidden_dim: int
        modulation_rank: int

        @nn.compact
        def __call__(
            self,
            task_features: Any,
            belief_embedding: Any,
            gate: Any,
        ) -> tuple[Any, Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            belief = jnp.asarray(belief_embedding, dtype=jnp.float32)
            gate_value = jnp.asarray(gate, dtype=jnp.float32)
            if gate_value.ndim == task.ndim and gate_value.shape[-1] == 1:
                gate_value = gate_value[..., 0]
            if gate_value.shape != task.shape[:-1]:
                gate_value = jnp.broadcast_to(gate_value, task.shape[:-1])
            base_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="base_hidden",
                )(task)
            )
            base_logits = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="base_logits",
            )(base_hidden)
            residual = LowRankContextResidual(
                output_dim=self.action_count,
                rank=self.modulation_rank,
                hidden_dim=self.hidden_dim,
                name="context_residual",
            )(task, belief)
            execution = base_logits + gate_value[..., None] * residual
            return base_logits, residual, execution

    _ACTOR = UniversalCoordinationActor
    return UniversalCoordinationActor


__all__ = ["universal_actor_class"]
