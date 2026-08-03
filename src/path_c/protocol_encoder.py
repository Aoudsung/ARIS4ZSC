"""Capability and protocol encoders for DEPI (METHOD_SPEC §1.1/§2).

Implements the two partner-side legal-history pathways:

* ``CapabilityEncoderCell`` -- GRU-64 recurrent encoder producing the stable
  partner capability/tendency embedding ``u`` (16 dimensions).
* ``ProtocolEncoderCell`` -- GRU-128 amortized recursive Bayes filter
  producing the categorical posterior ``pi_t`` over ``K=4`` protocol regimes.

Both cells consume the same evidence vector ``e_t = (o_t - o_{t-1},
a^{ego}_{t-1} embedding, episode_start)``.  The observation difference is the
only legal carrier of partner behaviour inside the ego view (METHOD_SPEC §1.1
decision record); the task pathway never reads it.

The sticky transition prior P(z_{t+1}|z_t) with P(stay)=0.9 and
P(jump to each other)=0.1/3 is a registered constant; it is never learned
(METHOD_SPEC §2.1).  The predictive step and the log-sticky correction applied
to the posterior logits are parameter-free constant mappings.

METHOD_SPEC §2.5: uncertainty comes from (i) the bounded posterior entropy
H(q_t) and (ii) a B=3 bootstrap history-encoder ensemble whose epistemic
readout is the mean total-variation distance of the member posteriors.  Both
are report-only quantities and never enter any loss weight.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_COMPONENTS = 4
STICKY_SELF_PROBABILITY = 0.9
STICKY_JUMP_PROBABILITY = 0.1 / 3.0
BOOTSTRAP_MEMBER_COUNT = 3
BOOTSTRAP_DROPOUT_RATE = 0.1

_CAPABILITY_CELL: Any | None = None
_PROTOCOL_CELL: Any | None = None


def sticky_predictive(previous_probabilities: Any) -> Any:
    """Exact one-step predictive distribution of the sticky HMM prior.

    p_pred,k = 0.9 * pi_{t-1,k} + (0.1/3) * sum_j pi_{t-1,j}.  This is a
    parameter-free constant mixing (METHOD_SPEC §2.1/§2.3).
    """

    import jax.numpy as jnp

    previous = jnp.asarray(previous_probabilities, dtype=jnp.float32)
    mass = jnp.sum(previous, axis=-1, keepdims=True)
    return (
        float(STICKY_SELF_PROBABILITY) * previous
        + float(STICKY_JUMP_PROBABILITY) * mass
    )


def sticky_predictive_log_probabilities(previous_probabilities: Any) -> Any:
    import jax.numpy as jnp

    return jnp.log(jnp.maximum(sticky_predictive(previous_probabilities), 1.0e-12))


def uniform_prior(component_count: int = PROTOCOL_COMPONENTS) -> Any:
    import jax.numpy as jnp

    return jnp.full((int(component_count),), 1.0 / float(component_count), dtype=jnp.float32)


def exact_bayes_filter_step(
    previous_probabilities: Any, log_likelihood: Any
) -> Any:
    """Reference exact filtering recursion for the synthetic-HMM unit test.

    posterior ∝ (T^T pi_{t-1}) ⊙ likelihood.  The amortized ProtocolEncoderCell
    reproduces this recursion with the same constant predictive step; this
    helper is the ground-truth comparator (METHOD_SPEC §2.3).
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    predictive = sticky_predictive(previous_probabilities)
    unnormalized = predictive * jnp.exp(
        jnp.asarray(log_likelihood, dtype=jnp.float32)
    )
    return jnn.softmax(
        jnp.log(jnp.maximum(unnormalized, 1.0e-30)), axis=-1
    )


def posterior_entropy(probabilities: Any) -> Any:
    """Bounded posterior entropy H(q_t); METHOD_SPEC §2.5 item 1."""

    import jax.numpy as jnp

    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    return -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1.0e-12)), axis=-1)


def ensemble_total_variation(member_probabilities: Any) -> Any:
    """Epistemic readout of the B-member bootstrap ensemble.

    ``member_probabilities`` has shape (B, ..., K); the readout is the mean
    pairwise total-variation distance, report-only (METHOD_SPEC §2.5 item 2).
    """

    import jax.numpy as jnp

    members = jnp.asarray(member_probabilities, dtype=jnp.float32)
    count = members.shape[0]
    if count < 2:
        return jnp.zeros(members.shape[1:-1], dtype=jnp.float32)
    distances = []
    for index in range(count):
        for other in range(index + 1, count):
            distances.append(
                0.5 * jnp.sum(jnp.abs(members[index] - members[other]), axis=-1)
            )
    stacked = jnp.stack(distances, axis=0)
    return jnp.mean(stacked, axis=0)


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
    """Build e_t = (Δo, a^ego_{t-1} embedding, episode_start) (METHOD_SPEC §1.1)."""

    import flax.linen as nn
    import jax.numpy as jnp

    action = jnp.asarray(previous_action, dtype=jnp.int32)
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    previous = jnp.asarray(previous_observation, dtype=jnp.float32).reshape(
        action.shape + (-1,)
    )
    current = jnp.asarray(observation, dtype=jnp.float32).reshape(
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
        """GRU-64 stable partner-capability encoder; output u ∈ R^16."""

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
            carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
            next_carry, hidden = nn.GRUCell(
                features=self.hidden_dim, name="capability_gru"
            )(carry, projected)
            capability = nn.Dense(
                self.output_dim,
                kernel_init=orthogonal(0.1),
                bias_init=zeros,
                name="capability_output",
            )(hidden)
            return next_carry, capability

    _CAPABILITY_CELL = CapabilityEncoderCell
    return CapabilityEncoderCell


def protocol_encoder_classes() -> Any:
    global _PROTOCOL_CELL
    if _PROTOCOL_CELL is not None:
        return _PROTOCOL_CELL

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class ProtocolEncoderCell(nn.Module):
        """Amortized recursive Bayes filter over K protocol regimes.

        The GRU input carries the evidence plus the previous posterior; the
        posterior logits receive the parameter-free log-sticky predictive
        correction (METHOD_SPEC §2.3).  ``h_0`` maps to the prior because the
        initial carry is zero and the initial previous posterior is uniform.
        """

        hidden_dim: int
        component_count: int
        action_count: int
        action_embedding_dim: int

        @nn.compact
        def __call__(
            self,
            carry: Any,
            inputs: tuple[Any, Any, Any, Any, Any],
        ) -> tuple[Any, Any]:
            (
                previous_observation,
                observation,
                previous_action,
                episode_start,
                previous_probabilities,
            ) = inputs
            action = jnp.asarray(previous_action, dtype=jnp.int32)
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            previous_probs = jnp.asarray(previous_probabilities, dtype=jnp.float32)
            evidence = _evidence_vector(
                module=self,
                previous_observation=previous_observation,
                observation=observation,
                previous_action=action,
                episode_start=start,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                embedding_name="protocol_action_embedding",
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
            gru_input = jnp.concatenate((projected, previous_probs), axis=-1)
            carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
            next_carry, hidden = nn.GRUCell(
                features=self.hidden_dim, name="protocol_gru"
            )(carry, gru_input)
            logits = nn.Dense(
                self.component_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="posterior_logits",
            )(hidden)
            # Constant log-sticky predictive correction; no learned parameters
            # (METHOD_SPEC §2.1/§2.3 decision record).
            correction = sticky_predictive_log_probabilities(previous_probs)
            return next_carry, logits + correction

    _PROTOCOL_CELL = ProtocolEncoderCell
    return ProtocolEncoderCell


def bootstrap_protocol_ensemble_class() -> Any:
    """B=3 independently initialized protocol encoders (METHOD_SPEC §2.5)."""

    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class BootstrapProtocolEnsemble(nn.Module):
        hidden_dim: int
        component_count: int
        action_count: int
        action_embedding_dim: int
        member_count: int = BOOTSTRAP_MEMBER_COUNT
        dropout_rate: float = BOOTSTRAP_DROPOUT_RATE

        @nn.compact
        def __call__(
            self,
            carries: Any,
            inputs: tuple[Any, Any, Any, Any, Any],
            training: bool = True,
        ) -> tuple[Any, Any]:
            ProtocolCell = protocol_encoder_classes()
            next_carries = []
            posteriors = []
            for member in range(int(self.member_count)):
                cell = ProtocolCell(
                    hidden_dim=self.hidden_dim,
                    component_count=self.component_count,
                    action_count=self.action_count,
                    action_embedding_dim=self.action_embedding_dim,
                    name=f"bootstrap_member_{member}",
                )
                next_carry, logits = cell(carries[member], inputs)
                logits = nn.Dropout(
                    rate=float(self.dropout_rate), deterministic=not training
                )(logits)
                next_carries.append(next_carry)
                posteriors.append(logits)
            probabilities = [
                jax.nn.softmax(logits, axis=-1) for logits in posteriors
            ]
            return tuple(next_carries), jnp.stack(probabilities, axis=0)

    return BootstrapProtocolEnsemble


def initial_capability_carry(batch_size: int, hidden_dim: int) -> Any:
    import jax.numpy as jnp

    return jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32)


def initial_protocol_carry(batch_size: int, hidden_dim: int) -> Any:
    import jax.numpy as jnp

    return jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32)


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
    "BOOTSTRAP_DROPOUT_RATE",
    "BOOTSTRAP_MEMBER_COUNT",
    "PROTOCOL_COMPONENTS",
    "STICKY_JUMP_PROBABILITY",
    "STICKY_SELF_PROBABILITY",
    "bootstrap_protocol_ensemble_class",
    "capability_encoder_classes",
    "ensemble_total_variation",
    "exact_bayes_filter_step",
    "initial_capability_carry",
    "initial_protocol_carry",
    "initial_protocol_probabilities",
    "posterior_entropy",
    "protocol_encoder_classes",
    "sticky_predictive",
    "sticky_predictive_log_probabilities",
    "uniform_prior",
]
