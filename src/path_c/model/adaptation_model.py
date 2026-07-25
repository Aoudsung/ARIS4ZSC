"""Assembly and gradient routing for the complete Path C Flax model."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts.config import ModelConfig
from .backbone import backbone_class, copy_explicit_parameter_leaves
from .heads import head_classes
from .transition import transition_head_class


_MODEL_CACHE: dict[tuple[int, ...], Any] = {}


def _model_class() -> Any:
    cached = _MODEL_CACHE.get((1,))
    if cached is not None:
        return cached
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    Backbone = backbone_class()
    SharedActorCritic, PrototypeValueHeads, PrototypeResponseHeads = head_classes()
    PrototypeTransitionHeads = transition_head_class()

    class PathCFlaxAdaptationModel(nn.Module):
        num_prototypes: int
        encoder_dim: int
        recurrent_dim: int
        head_hidden_dim: int
        response_hidden_dim: int
        value_hidden_dim: int
        transition_hidden_dim: int
        next_feature_dim: int
        action_count: int
        response_count: int
        activation: str = "relu"

        def initial_carry(self, batch_size: int) -> Any:
            if batch_size <= 0:
                raise ValueError("The recurrent batch size must be positive.")
            # Flax submodules can only be instantiated while a module is bound
            # or inside setup/compact.  The official GRU carry is an all-zero
            # array, so construct that value directly for unbound callers.
            return jnp.zeros(
                (int(batch_size), self.recurrent_dim), dtype=jnp.float32
            )

        @nn.compact
        def __call__(self, carry: Any, observations: Any, episode_start: Any) -> tuple[Any, Mapping[str, Any]]:
            backbone = Backbone(
                encoder_dim=self.encoder_dim,
                hidden_dim=self.recurrent_dim,
                activation=self.activation,
                name="backbone",
            )
            next_carry, features = backbone(carry, observations, episode_start)
            actor_logits, shared_value = SharedActorCritic(
                hidden_dim=self.head_hidden_dim,
                action_count=self.action_count,
                activation=self.activation,
                name="shared_heads",
            )(features)
            detached = jax.lax.stop_gradient(features)
            value_heads = PrototypeValueHeads(
                num_prototypes=self.num_prototypes,
                hidden_dim=self.value_hidden_dim,
                activation=self.activation,
                name="prototype_values",
            )
            prototype_values = value_heads(detached)
            response_logits, response_probs = PrototypeResponseHeads(
                num_prototypes=self.num_prototypes,
                hidden_dim=self.response_hidden_dim,
                action_count=self.action_count,
                response_count=self.response_count,
                activation=self.activation,
                name="prototype_responses",
            )(detached)
            transition_logits, transition_probs, reward_estimates, next_features = PrototypeTransitionHeads(
                num_prototypes=self.num_prototypes,
                hidden_dim=self.transition_hidden_dim,
                action_count=self.action_count,
                response_count=self.response_count,
                next_feature_dim=self.next_feature_dim,
                activation=self.activation,
                name="prototype_transition",
            )(detached)

            # Evaluate every generated next feature with every value head.  The
            # resulting axes are [time, batch, generating prototype, action,
            # response, value-head prototype].
            generated_values = []
            for prototype in range(self.num_prototypes):
                generated_values.append(value_heads(next_features[..., prototype, :, :, :]))
            next_values = jnp.stack(generated_values, axis=-4)
            return next_carry, {
                "features": features,
                "actor_logits": actor_logits,
                "shared_value": shared_value,
                "prototype_values": prototype_values,
                "response_logits": response_logits,
                "response_probabilities": response_probs,
                "transition_response_logits": transition_logits,
                "transition_response_probabilities": transition_probs,
                "reward_estimates": reward_estimates,
                "next_feature_summaries": next_features,
                "next_values": next_values,
            }

    _MODEL_CACHE[(1,)] = PathCFlaxAdaptationModel
    return PathCFlaxAdaptationModel


def build_model(
    *,
    num_prototypes: int,
    action_count: int,
    response_count: int,
    official_dimensions: Mapping[str, Any],
    model_config: ModelConfig,
) -> Any:
    """Construct a model using official dimensions unless explicitly overridden."""

    if num_prototypes <= 1 or action_count <= 1 or response_count <= 1:
        raise ValueError("Model counts must include multiple prototypes, actions, and responses.")
    encoder_dim = model_config.encoder_dim or int(official_dimensions["encoder_dim"])
    recurrent_dim = model_config.gru_hidden_dim or int(official_dimensions["gru_hidden_dim"])
    next_dim = model_config.next_feature_summary_dim or recurrent_dim
    if encoder_dim != recurrent_dim:
        raise ValueError("The official convolutional encoder output must equal the recurrent width.")
    if next_dim != recurrent_dim:
        raise ValueError("The first implementation requires next features to match the recurrent width.")
    model_class = _model_class()
    return model_class(
        num_prototypes=num_prototypes,
        encoder_dim=encoder_dim,
        recurrent_dim=recurrent_dim,
        head_hidden_dim=int(official_dimensions["actor_critic_hidden_dim"]),
        response_hidden_dim=model_config.response_hidden_dim,
        value_hidden_dim=model_config.value_hidden_dim,
        transition_hidden_dim=model_config.transition_hidden_dim,
        next_feature_dim=next_dim,
        action_count=action_count,
        response_count=response_count,
        activation=str(official_dimensions.get("activation", "relu")),
    )


def initialize_from_official(
    model: Any,
    *,
    random_key: Any,
    official_params: Mapping[str, Any],
    explicit_leaf_mapping: Mapping[tuple[str, ...], tuple[str, ...]],
    example_observations: Any,
    example_episode_start: Any,
) -> dict[str, Any]:
    """Randomly initialize new heads and copy exact official backbone leaves."""

    batch_size = int(example_observations.shape[1])
    variables = model.init(
        random_key,
        model.initial_carry(batch_size),
        example_observations,
        example_episode_start,
    )
    return copy_explicit_parameter_leaves(
        variables["params"], official_params, explicit_leaf_mapping
    )


def __getattr__(name: str) -> Any:
    if name == "PathCFlaxAdaptationModel":
        return _model_class()
    raise AttributeError(name)
