"""Continuous multi-modal partner belief encoders."""

from __future__ import annotations

from typing import Any

_BELIEF_CELL: Any | None = None
_BELIEF_SCAN: Any | None = None


def belief_encoder_classes() -> tuple[Any, Any]:
    global _BELIEF_CELL, _BELIEF_SCAN
    if _BELIEF_CELL is not None:
        return _BELIEF_CELL, _BELIEF_SCAN

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class PartnerBeliefCell(nn.Module):
        hidden_dim: int
        latent_dim: int
        mixture_components: int
        action_count: int
        action_embedding_dim: int
        log_variance_minimum: float
        log_variance_maximum: float

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any, Any, Any, Any],
        ) -> tuple[Any, tuple[Any, Any, Any, Any]]:
            (
                previous_observation,
                observation,
                previous_action,
                previous_reward,
                episode_start,
            ) = inputs
            action = jnp.asarray(previous_action, dtype=jnp.int32)
            reward = jnp.asarray(previous_reward, dtype=jnp.float32)
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            previous = jnp.asarray(previous_observation, dtype=jnp.float32).reshape(
                action.shape + (-1,)
            )
            current = jnp.asarray(observation, dtype=jnp.float32).reshape(
                action.shape + (-1,)
            )
            if previous.shape != current.shape:
                raise ValueError("Belief encoder observation shapes differ.")
            sentinel = jnp.where(start, self.action_count, action)
            action_embedding = nn.Embed(
                num_embeddings=self.action_count + 1,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="previous_action_embedding",
            )(sentinel)
            evidence = jnp.concatenate(
                (
                    previous,
                    current,
                    current - previous,
                    action_embedding,
                    reward[..., None],
                    start.astype(jnp.float32)[..., None],
                ),
                axis=-1,
            )
            evidence = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="evidence_projection",
                )(evidence)
            )
            evidence = nn.LayerNorm(name="evidence_layer_norm")(evidence)
            carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
            next_carry, hidden = nn.GRUCell(
                features=self.hidden_dim,
                name="belief_gru",
            )(carry, evidence)
            mixture_logits = nn.Dense(
                self.mixture_components,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="mixture_logits",
            )(hidden)
            means = nn.Dense(
                self.mixture_components * self.latent_dim,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="mixture_means",
            )(hidden).reshape(
                action.shape + (self.mixture_components, self.latent_dim)
            )
            log_variances = nn.Dense(
                self.mixture_components * self.latent_dim,
                kernel_init=zeros,
                bias_init=zeros,
                name="mixture_log_variances",
            )(hidden).reshape(
                action.shape + (self.mixture_components, self.latent_dim)
            )
            log_variances = jnp.clip(
                log_variances,
                self.log_variance_minimum,
                self.log_variance_maximum,
            )
            weights = nn.softmax(mixture_logits, axis=-1)
            mixture_entropy = -jnp.sum(
                weights * jnp.log(jnp.maximum(weights, 1.0e-8)), axis=-1
            )
            normalized_entropy = mixture_entropy / jnp.log(
                jnp.asarray(max(self.mixture_components, 2), dtype=jnp.float32)
            )
            expected_variance = jnp.sum(
                weights[..., :, None] * jnp.exp(log_variances), axis=(-2, -1)
            ) / float(self.latent_dim)
            # This is an uncertainty confidence used only as a training
            # diagnostic.  The confirmatory OOD support gate is the frozen
            # Mahalanobis model stored in CalibrationArtifact.
            support_score = jnp.exp(-0.5 * expected_variance) * (1.0 - normalized_entropy)
            support_score = jnp.clip(support_score, 0.0, 1.0)
            return next_carry, (
                mixture_logits,
                means,
                log_variances,
                support_score,
            )

    ScannedPartnerBelief = nn.scan(
        PartnerBeliefCell,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )

    _BELIEF_CELL = PartnerBeliefCell
    _BELIEF_SCAN = ScannedPartnerBelief
    return PartnerBeliefCell, ScannedPartnerBelief


def initial_belief_carry(batch_size: int, hidden_dim: int) -> Any:
    import jax.numpy as jnp

    return jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32)


__all__ = ["belief_encoder_classes", "initial_belief_carry"]
