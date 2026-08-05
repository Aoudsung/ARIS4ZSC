"""Capability state and exact categorical protocol filtering for DEPI.

``CapabilityEncoderCell`` is the sole learned recurrent partner-history
encoder.  The protocol state has no recognition GRU or hidden dimension: its
complete carry is the categorical posterior itself, updated by the registered
transition matrix and the shared response likelihood.

The visible-evidence transition prior uses a registered stay probability and
equal mass on the other components.  When no partner evidence is visible the
transition is exactly the identity, so occlusion cannot erase the posterior.
(METHOD_SPEC §2.1).  The predictive step and the log-sticky correction applied
to the posterior logits are parameter-free constant mappings.  Posterior
entropy is a report-only uncertainty reading and never enters a loss weight.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_COMPONENTS = 4
STICKY_SELF_PROBABILITY = 0.97
CAPABILITY_UPDATE_PERIOD = 16

_CAPABILITY_CELL: Any | None = None


def sticky_predictive(
    previous_probabilities: Any, *, stay: float = STICKY_SELF_PROBABILITY
) -> Any:
    """Exact one-step predictive distribution of the sticky HMM prior.

    This is an explicit multiplication by the registered row-stochastic
    transition matrix.  The off-diagonal contribution excludes the current
    component; no later softmax repairs an invalid probability mass.
    """

    import jax.numpy as jnp

    previous = jnp.asarray(previous_probabilities, dtype=jnp.float32)
    transition = sticky_transition_matrix(
        previous.shape[-1], stay=float(stay)
    )
    return previous @ transition


def sticky_transition_matrix(
    component_count: int = PROTOCOL_COMPONENTS,
    *,
    stay: float = STICKY_SELF_PROBABILITY,
) -> Any:
    """Return the symmetric row-stochastic sticky transition matrix."""

    import jax.numpy as jnp

    count = int(component_count)
    stay_probability = float(stay)
    if count < 2:
        raise ValueError("A sticky transition requires at least two components.")
    if not 0.0 <= stay_probability <= 1.0:
        raise ValueError("Sticky self-transition probability must lie in [0, 1].")
    jump = (1.0 - stay_probability) / float(count - 1)
    matrix = jnp.full((count, count), jump, dtype=jnp.float32)
    return matrix.at[jnp.diag_indices(count)].set(stay_probability)


def sticky_predictive_log_probabilities(
    previous_probabilities: Any, *, stay: float = STICKY_SELF_PROBABILITY
) -> Any:
    import jax.numpy as jnp

    return jnp.log(
        jnp.maximum(sticky_predictive(previous_probabilities, stay=stay), 1.0e-12)
    )


def uniform_prior(component_count: int = PROTOCOL_COMPONENTS) -> Any:
    import jax.numpy as jnp

    return jnp.full((int(component_count),), 1.0 / float(component_count), dtype=jnp.float32)


def exact_bayes_filter_step(
    previous_probabilities: Any,
    log_likelihood: Any,
    *,
    stay: float = STICKY_SELF_PROBABILITY,
    evidence_valid: Any = True,
) -> Any:
    """Reference exact filtering recursion for the synthetic-HMM unit test.

    posterior ∝ (T^T pi_{t-1}) ⊙ likelihood.  Both training and deployment
    call this exact log-domain update (METHOD_SPEC §2.3).
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    previous = jnp.asarray(previous_probabilities, dtype=jnp.float32)
    sticky = sticky_predictive(previous, stay=float(stay))
    valid = jnp.asarray(evidence_valid, dtype=jnp.bool_)
    predictive = jnp.where(valid[..., None], sticky, previous)
    evidence = jnp.where(
        valid[..., None], jnp.asarray(log_likelihood, dtype=jnp.float32), 0.0
    )
    return jnn.softmax(
        jnp.log(jnp.maximum(predictive, 1.0e-30))
        + evidence,
        axis=-1,
    )


def posterior_entropy(probabilities: Any) -> Any:
    """Bounded posterior entropy H(q_t); METHOD_SPEC §2.5 item 1."""

    import jax.numpy as jnp

    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    return -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1.0e-12)), axis=-1)


def _evidence_vector(
    *,
    module: Any,
    previous_observation: Any,
    observation: Any,
    previous_action: Any,
    episode_start: Any,
    action_count: int,
    action_embedding_dim: int,
    embedding_name: str,
) -> Any:
    """Build evidence from partner-plane deltas, own action, and episode start."""

    import flax.linen as nn
    import jax.numpy as jnp

    action = jnp.asarray(previous_action, dtype=jnp.int32)
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    from .task_encoder import instantaneous_partner_observation

    previous = jnp.asarray(
        instantaneous_partner_observation(previous_observation), dtype=jnp.float32
    ).reshape(
        action.shape + (-1,)
    )
    current = jnp.asarray(
        instantaneous_partner_observation(observation), dtype=jnp.float32
    ).reshape(
        action.shape + (-1,)
    )
    if previous.shape != current.shape:
        raise ValueError("Capability/protocol evidence observations differ.")
    sentinel = jnp.where(start, int(action_count), action)
    action_embedding = nn.Embed(
        num_embeddings=int(action_count) + 1,
        features=int(action_embedding_dim),
        embedding_init=nn.initializers.normal(0.02),
        name=embedding_name,
    )(sentinel)
    return jnp.concatenate(
        (
            current - previous,
            action_embedding,
            start.astype(jnp.float32)[..., None],
        ),
        axis=-1,
    )


def capability_encoder_classes() -> Any:
    global _CAPABILITY_CELL
    if _CAPABILITY_CELL is not None:
        return _CAPABILITY_CELL

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class CapabilityEncoderCell(nn.Module):
        """GRU-64 stable semantic partner-capability encoder; output u ∈ R^4."""

        hidden_dim: int
        output_dim: int
        action_count: int
        action_embedding_dim: int

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any, Any, Any],
        ) -> tuple[Any, Any]:
            previous_observation, observation, previous_action, episode_start = inputs
            action = jnp.asarray(previous_action, dtype=jnp.int32)
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            evidence = _evidence_vector(
                module=self,
                previous_observation=previous_observation,
                observation=observation,
                previous_action=action,
                episode_start=start,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                embedding_name="capability_action_embedding",
            )
            projected = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="evidence_projection",
                )(evidence)
            )
            projected = nn.LayerNorm(name="evidence_layer_norm")(projected)
            from .types import CapabilityCarry

            hidden_carry, published, steps = carry
            hidden_carry = jnp.where(
                start[..., None], jnp.zeros_like(hidden_carry), hidden_carry
            )
            published = jnp.where(
                start[..., None], jnp.zeros_like(published), published
            )
            steps = jnp.where(start, jnp.zeros_like(steps), steps)
            next_carry, hidden = nn.GRUCell(
                features=self.hidden_dim, name="capability_gru"
            )(hidden_carry, projected)
            # The published representation is bounded because its first four
            # coordinates are directly supervised against registered rolling
            # observable rates in [-1, 1].  This is not a free prediction head:
            # the deployable coordinates themselves carry the semantic target.
            candidate = jnp.tanh(
                nn.Dense(
                    self.output_dim,
                    kernel_init=orthogonal(0.1),
                    bias_init=zeros,
                    name="capability_output",
                )(hidden)
            )
            next_steps = steps + jnp.asarray(1, dtype=steps.dtype)
            publish = (next_steps % int(CAPABILITY_UPDATE_PERIOD)) == 0
            capability = jnp.where(publish[..., None], candidate, published)
            return CapabilityCarry(next_carry, capability, next_steps), capability

    _CAPABILITY_CELL = CapabilityEncoderCell
    return CapabilityEncoderCell


def initial_capability_carry(
    batch_size: int, hidden_dim: int, output_dim: int = 4
) -> Any:
    import jax.numpy as jnp

    from .types import CapabilityCarry

    return CapabilityCarry(
        hidden=jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32),
        published=jnp.zeros((int(batch_size), int(output_dim)), dtype=jnp.float32),
        steps=jnp.zeros((int(batch_size),), dtype=jnp.int32),
    )


def initial_protocol_probabilities(
    batch_shape: tuple[int, ...], component_count: int = PROTOCOL_COMPONENTS
) -> Any:
    import jax.numpy as jnp

    return jnp.full(
        tuple(int(value) for value in batch_shape) + (int(component_count),),
        1.0 / float(component_count),
        dtype=jnp.float32,
    )


__all__ = [
    "CAPABILITY_UPDATE_PERIOD",
    "PROTOCOL_COMPONENTS",
    "STICKY_SELF_PROBABILITY",
    "capability_encoder_classes",
    "exact_bayes_filter_step",
    "initial_capability_carry",
    "initial_protocol_probabilities",
    "posterior_entropy",
    "sticky_predictive",
    "sticky_predictive_log_probabilities",
    "sticky_transition_matrix",
    "uniform_prior",
]
