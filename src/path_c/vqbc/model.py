"""Flax implementation of the value-quotient belief-conditioned controller."""

from __future__ import annotations

from typing import Any, Mapping

from .config import VQBCModelConfig
from .integrity import copy_explicit_parameter_leaves
from .policy import (
    bellman_control_values,
    deployment_policy,
    generic_response_information,
)
from .quotient import aggregate_slots, complete_link_class_ids, value_signatures
from .types import VQBCOutput


_MODEL_CACHE: dict[tuple[int, ...], Any] = {}


def _model_class() -> Any:
    cached = _MODEL_CACHE.get((4,))
    if cached is not None:
        return cached

    import functools

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import constant, normal, orthogonal, zeros

    class OfficialCNN(nn.Module):
        output_size: int
        activation: str = "relu"

        @nn.compact
        def __call__(self, observations: Any) -> Any:
            activation = nn.relu if self.activation == "relu" else nn.tanh
            values = jnp.asarray(observations, dtype=jnp.float32)
            for index, (features, kernel_size) in enumerate(
                (
                    (128, (1, 1)),
                    (128, (1, 1)),
                    (8, (1, 1)),
                    (16, (3, 3)),
                    (32, (3, 3)),
                    (32, (3, 3)),
                )
            ):
                values = nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=constant(0.0),
                    name=f"Conv_{index}",
                )(values)
                values = activation(values)
            values = values.reshape((values.shape[0], -1))
            values = nn.Dense(
                self.output_size,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="Dense_0",
            )(values)
            return activation(values)

    class ScannedRNN(nn.Module):
        @functools.partial(
            nn.scan,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        @nn.compact
        def __call__(self, carry: Any, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
            embedding, episode_start = inputs
            reset_cell = nn.GRUCell(features=embedding.shape[1])
            reset_carry = reset_cell.initialize_carry(
                jax.random.PRNGKey(0),
                (embedding.shape[0], embedding.shape[1]),
            )
            carry = jnp.where(
                jnp.asarray(episode_start, dtype=jnp.bool_)[:, None],
                reset_carry,
                carry,
            )
            return nn.GRUCell(features=embedding.shape[1])(carry, embedding)

    class HistoryBackbone(nn.Module):
        hidden_dim: int
        action_count: int
        action_embedding_dim: int
        activation: str = "relu"

        @nn.compact
        def __call__(
            self,
            carry: Any,
            observations: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            episode_start: Any,
        ) -> tuple[Any, Any]:
            values = jnp.asarray(observations, dtype=jnp.float32)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            rewards = jnp.asarray(previous_team_rewards, dtype=jnp.float32)
            starts = jnp.asarray(episode_start, dtype=jnp.bool_)
            if (
                values.ndim != 5
                or actions.shape != values.shape[:2]
                or rewards.shape != values.shape[:2]
                or starts.shape != values.shape[:2]
            ):
                raise ValueError(
                    "History backbone inputs must share [time, batch] leading axes."
                )
            encoded = jax.vmap(
                OfficialCNN(
                    output_size=self.hidden_dim,
                    activation=self.activation,
                    name="CNN_0",
                )
            )(values)
            sentinel_actions = jnp.where(starts, self.action_count, actions)
            action_embedding = nn.Embed(
                num_embeddings=self.action_count + 1,
                features=self.action_embedding_dim,
                embedding_init=normal(0.02),
                name="PreviousActionEmbedding",
            )(sentinel_actions)
            action_projection = nn.Dense(
                self.hidden_dim,
                kernel_init=zeros,
                bias_init=zeros,
                name="PreviousActionProjection",
            )(action_embedding)
            reward_projection = nn.Dense(
                self.hidden_dim,
                kernel_init=zeros,
                bias_init=zeros,
                name="PreviousRewardProjection",
            )(rewards[..., None])
            joined = encoded + action_projection + reward_projection
            joined = nn.LayerNorm(name="LayerNorm_0")(joined)
            return ScannedRNN(name="ScannedRNN_0")(carry, (joined, starts))

    class DuelingEstimator(nn.Module):
        slot_count: int
        action_count: int
        hidden_dim: int
        slot_embedding_dim: int
        output_scale: float
        prefix: str

        @nn.compact
        def __call__(self, features: Any) -> tuple[Any, Any]:
            shared_hidden = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(2.0),
                    bias_init=zeros,
                    name=f"{self.prefix}_shared_hidden",
                )(features)
            )
            shared_value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name=f"{self.prefix}_shared_value",
            )(shared_hidden)[..., 0]
            slot_embedding = self.param(
                f"{self.prefix}_slot_embedding",
                normal(0.02),
                (self.slot_count, self.slot_embedding_dim),
            )
            feature_grid = jnp.broadcast_to(
                features[..., None, :],
                features.shape[:-1] + (self.slot_count, features.shape[-1]),
            )
            embedding_grid = jnp.broadcast_to(
                slot_embedding,
                features.shape[:-1] + slot_embedding.shape,
            )
            joined = jnp.concatenate((feature_grid, embedding_grid), axis=-1)
            hidden = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name=f"{self.prefix}_slot_hidden",
                )(joined)
            )
            offset = nn.Dense(
                1,
                kernel_init=orthogonal(self.output_scale),
                bias_init=zeros,
                name=f"{self.prefix}_slot_offset",
            )(hidden)[..., 0]
            advantages = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(self.output_scale),
                bias_init=zeros,
                name=f"{self.prefix}_advantages",
            )(hidden)
            centered = advantages - jnp.mean(advantages, axis=-1, keepdims=True)
            return shared_value[..., None, None] + offset[..., :, None] + centered, centered

    class TwinDuelingQ(nn.Module):
        slot_count: int
        action_count: int
        hidden_dim: int
        slot_embedding_dim: int
        prior_scale: float

        @nn.compact
        def __call__(self, features: Any) -> tuple[Any, Any, Any, Any]:
            learned = []
            learned_advantages = []
            priors = []
            for estimator in range(2):
                learned_value, learned_advantage = DuelingEstimator(
                    slot_count=self.slot_count,
                    action_count=self.action_count,
                    hidden_dim=self.hidden_dim,
                    slot_embedding_dim=self.slot_embedding_dim,
                    output_scale=1.0e-3,
                    prefix=f"learned_estimator_{estimator}",
                    name=f"LearnedEstimator_{estimator}",
                )(features)
                prior_value, unused_prior_advantage = DuelingEstimator(
                    slot_count=self.slot_count,
                    action_count=self.action_count,
                    hidden_dim=self.hidden_dim,
                    slot_embedding_dim=self.slot_embedding_dim,
                    output_scale=1.0,
                    prefix=f"prior_estimator_{estimator}",
                    name=f"PriorEstimator_{estimator}",
                )(features)
                del unused_prior_advantage
                learned.append(learned_value)
                learned_advantages.append(learned_advantage)
                priors.append(jax.lax.stop_gradient(prior_value))
            learned_values = jnp.stack(learned, axis=-3)
            prior_values = jnp.stack(priors, axis=-3)
            q_values = learned_values + self.prior_scale * prior_values
            return (
                q_values,
                learned_values,
                prior_values,
                q_values - jnp.mean(q_values, axis=-1, keepdims=True),
            )

    class OutcomeModel(nn.Module):
        slot_count: int
        action_count: int
        response_count: int
        hidden_dim: int
        slot_embedding_dim: int
        action_embedding_dim: int
        log_standard_deviation_minimum: float
        log_standard_deviation_maximum: float

        @nn.compact
        def __call__(self, features: Any) -> Mapping[str, Any]:
            detached = jax.lax.stop_gradient(features)
            slot_embedding = self.param(
                "slot_embedding",
                normal(0.02),
                (self.slot_count, self.slot_embedding_dim),
            )
            action_embedding = self.param(
                "action_embedding",
                normal(0.02),
                (self.action_count, self.action_embedding_dim),
            )
            prefix = detached.shape[:-1]
            feature_grid = jnp.broadcast_to(
                detached[..., None, None, :],
                prefix
                + (
                    self.slot_count,
                    self.action_count,
                    detached.shape[-1],
                ),
            )
            slot_grid = jnp.broadcast_to(
                slot_embedding[:, None, :],
                prefix
                + (
                    self.slot_count,
                    self.action_count,
                    self.slot_embedding_dim,
                ),
            )
            action_grid = jnp.broadcast_to(
                action_embedding[None, :, :],
                prefix
                + (
                    self.slot_count,
                    self.action_count,
                    self.action_embedding_dim,
                ),
            )
            joined = jnp.concatenate((feature_grid, slot_grid, action_grid), axis=-1)
            hidden = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="hidden",
                )(joined)
            )
            response_logits = nn.Dense(
                self.response_count,
                kernel_init=zeros,
                bias_init=zeros,
                name="response_logits",
            )(hidden)
            reward_mean = nn.Dense(
                1, kernel_init=zeros, bias_init=zeros, name="reward_mean"
            )(hidden)[..., 0]
            reward_log_std = nn.Dense(
                1,
                kernel_init=zeros,
                bias_init=zeros,
                name="reward_log_standard_deviation",
            )(hidden)[..., 0]
            continuation_size = 2 * self.response_count * self.action_count
            next_q_mean = nn.Dense(
                continuation_size,
                kernel_init=zeros,
                bias_init=zeros,
                name="next_q_mean",
            )(hidden).reshape(
                prefix
                + (
                    self.slot_count,
                    self.action_count,
                    2,
                    self.response_count,
                    self.action_count,
                )
            )
            next_q_log_std = nn.Dense(
                continuation_size,
                kernel_init=zeros,
                bias_init=zeros,
                name="next_q_log_standard_deviation",
            )(hidden).reshape(next_q_mean.shape)
            next_q_mean = jnp.moveaxis(next_q_mean, -3, -5)
            next_q_log_std = jnp.moveaxis(next_q_log_std, -3, -5)
            next_q_mean = next_q_mean.at[
                ..., self.response_count - 1, :
            ].set(0.0)
            lower = self.log_standard_deviation_minimum
            upper = self.log_standard_deviation_maximum
            return {
                "response_logits": response_logits,
                "response_probabilities": jax.nn.softmax(
                    response_logits, axis=-1
                ),
                "reward_mean": reward_mean,
                "reward_log_standard_deviation": jnp.clip(
                    reward_log_std, lower, upper
                ),
                "next_q_mean": next_q_mean,
                "next_q_log_standard_deviation": jnp.clip(
                    next_q_log_std, lower, upper
                ),
            }

    class ResponseEncoder(nn.Module):
        action_count: int
        hidden_dim: int

        @nn.compact
        def __call__(
            self,
            observations: Any,
            actions: Any,
            next_observations: Any,
            dones: Any,
        ) -> Any:
            current = jnp.asarray(observations, dtype=jnp.float32)
            following = jnp.asarray(next_observations, dtype=jnp.float32)
            if current.shape != following.shape:
                raise ValueError("Response encoder observations must have equal shapes.")
            flat_current = current.reshape(current.shape[:2] + (-1,))
            flat_following = following.reshape(following.shape[:2] + (-1,))
            action_one_hot = jax.nn.one_hot(
                jnp.asarray(actions, dtype=jnp.int32),
                self.action_count,
                dtype=jnp.float32,
            )
            joined = jnp.concatenate(
                (
                    flat_current,
                    flat_following,
                    flat_following - flat_current,
                    action_one_hot,
                    jnp.asarray(dones, dtype=jnp.float32)[..., None],
                ),
                axis=-1,
            )
            hidden = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="hidden",
                )(joined)
            )
            return nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="signature",
            )(hidden)

    class VQBCFlaxModel(nn.Module):
        hidden_dim: int
        head_hidden_dim: int
        action_embedding_dim: int
        slot_embedding_dim: int
        slot_count: int
        action_count: int
        response_count: int
        prior_scale: float
        log_standard_deviation_minimum: float
        log_standard_deviation_maximum: float
        activation: str = "relu"

        def setup(self) -> None:
            self.backbone = HistoryBackbone(
                hidden_dim=self.hidden_dim,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                activation=self.activation,
            )
            self.q_heads = TwinDuelingQ(
                slot_count=self.slot_count,
                action_count=self.action_count,
                hidden_dim=self.head_hidden_dim,
                slot_embedding_dim=self.slot_embedding_dim,
                prior_scale=self.prior_scale,
            )
            self.outcome = OutcomeModel(
                slot_count=self.slot_count,
                action_count=self.action_count,
                response_count=self.response_count,
                hidden_dim=self.head_hidden_dim,
                slot_embedding_dim=self.slot_embedding_dim,
                action_embedding_dim=self.action_embedding_dim,
                log_standard_deviation_minimum=(
                    self.log_standard_deviation_minimum
                ),
                log_standard_deviation_maximum=(
                    self.log_standard_deviation_maximum
                ),
            )
            self.response_encoder = ResponseEncoder(
                action_count=self.action_count,
                hidden_dim=self.head_hidden_dim,
            )

        def initial_carry(self, batch_size: int) -> Any:
            if batch_size <= 0:
                raise ValueError("The recurrent batch size must be positive.")
            return jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)

        def __call__(
            self,
            carry: Any,
            observations: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            episode_start: Any,
        ) -> tuple[Any, Mapping[str, Any]]:
            next_carry, features = self.backbone(
                carry,
                observations,
                previous_actions,
                previous_team_rewards,
                episode_start,
            )
            q_values, learned_q, prior_q, centered = self.q_heads(features)
            outcome = self.outcome(features)
            return next_carry, {
                "features": features,
                "q_values": q_values,
                "learned_q_values": learned_q,
                "prior_q_values": prior_q,
                "centered_advantages": centered,
                **outcome,
            }

        def control_only(
            self,
            carry: Any,
            observations: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            episode_start: Any,
        ) -> tuple[Any, Mapping[str, Any]]:
            """Evaluate the backbone and Q heads without materializing outcome axes."""

            next_carry, features = self.backbone(
                carry,
                observations,
                previous_actions,
                previous_team_rewards,
                episode_start,
            )
            q_values, learned_q, prior_q, centered = self.q_heads(features)
            return next_carry, {
                "features": features,
                "q_values": q_values,
                "learned_q_values": learned_q,
                "prior_q_values": prior_q,
                "centered_advantages": centered,
            }

        def encode_response(
            self,
            observations: Any,
            actions: Any,
            next_observations: Any,
            dones: Any,
        ) -> Any:
            return self.response_encoder(
                observations, actions, next_observations, dones
            )

    _MODEL_CACHE[(4,)] = VQBCFlaxModel
    return VQBCFlaxModel


def build_vqbc_model(
    *, model_config: VQBCModelConfig, official_dimensions: Mapping[str, Any]
) -> Any:
    if int(official_dimensions["gru_hidden_dim"]) != model_config.hidden_dim:
        raise ValueError("The fourth-model recurrent width must match the official model.")
    return _model_class()(
        hidden_dim=model_config.hidden_dim,
        head_hidden_dim=model_config.head_hidden_dim,
        action_embedding_dim=model_config.action_embedding_dim,
        slot_embedding_dim=model_config.slot_embedding_dim,
        slot_count=model_config.slot_count,
        action_count=model_config.action_count,
        response_count=model_config.response_count,
        prior_scale=model_config.prior_scale,
        log_standard_deviation_minimum=(
            model_config.log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=(
            model_config.log_standard_deviation_maximum
        ),
        activation=str(official_dimensions.get("activation", "relu")),
    )


def response_logits_from_codebook(
    response_signature: Any, codebook_embeddings: Any
) -> Any:
    import jax.numpy as jnp

    signature = jnp.asarray(response_signature)
    codebook = jnp.asarray(codebook_embeddings)
    if signature.shape[-1] != codebook.shape[-1]:
        raise ValueError("Response signatures and codebook entries differ in width.")
    return -jnp.sum(
        jnp.square(signature[..., None, :] - codebook), axis=-1
    )


def encode_response_codes(
    *,
    model: Any,
    params: Mapping[str, Any],
    observations: Any,
    actions: Any,
    next_observations: Any,
    dones: Any,
    codebook_embeddings: Any,
    terminal_response: int = 15,
) -> tuple[Any, Any, Any]:
    import jax.numpy as jnp

    signatures = model.apply(
        {"params": params},
        observations,
        actions,
        next_observations,
        dones,
        method=model.encode_response,
    )
    logits = response_logits_from_codebook(signatures, codebook_embeddings)
    codes = jnp.argmax(logits, axis=-1)
    codes = jnp.where(jnp.asarray(dones, dtype=jnp.bool_), terminal_response, codes)
    return codes, logits, signatures


def vqbc_forward(
    *,
    raw_output: Mapping[str, Any],
    reference_logits: Any,
    slot_log_belief: Any,
    temperature: Any,
    generic_temperature: Any,
    deployment_mode: str,
    gamma: float,
) -> VQBCOutput:
    """Apply value quotienting and the registered deployment policy."""

    q_values = raw_output["q_values"]
    signatures, radii = value_signatures(q_values)
    class_ids = complete_link_class_ids(signatures, radii)
    aggregated = aggregate_slots(
        class_ids=class_ids,
        slot_log_belief=slot_log_belief,
        response_probabilities=raw_output["response_probabilities"],
        reward_mean=raw_output["reward_mean"],
        next_q_mean=raw_output["next_q_mean"],
    )
    control = bellman_control_values(
        class_belief=aggregated.class_belief,
        response_probabilities=aggregated.response_probabilities,
        reward_mean=aggregated.reward_mean,
        next_q_mean=aggregated.next_q_mean,
        gamma=gamma,
    )
    information = generic_response_information(
        class_belief=aggregated.class_belief,
        response_probabilities=aggregated.response_probabilities,
    )
    policy = deployment_policy(
        mode=deployment_mode,
        reference_logits=reference_logits,
        control_values=control,
        information_gain=information,
        temperature=temperature,
        generic_temperature=generic_temperature,
    )
    return VQBCOutput(
        features=raw_output["features"],
        q_values=q_values,
        learned_q_values=raw_output["learned_q_values"],
        prior_q_values=raw_output["prior_q_values"],
        centered_advantages=raw_output["centered_advantages"],
        response_logits=raw_output["response_logits"],
        response_probabilities=raw_output["response_probabilities"],
        reward_mean=raw_output["reward_mean"],
        reward_log_standard_deviation=raw_output[
            "reward_log_standard_deviation"
        ],
        next_q_mean=raw_output["next_q_mean"],
        next_q_log_standard_deviation=raw_output[
            "next_q_log_standard_deviation"
        ],
        quotient_ids=class_ids,
        j_use=control.j_use,
        j_mask=control.j_mask,
        information_gain=information,
        execution_logits=policy.logits,
    )


def explicit_official_parameter_mapping() -> dict[tuple[str, ...], tuple[str, ...]]:
    convolution = {
        ("backbone", "CNN_0", f"Conv_{index}", leaf): (
            "params",
            "CNN_0",
            f"Conv_{index}",
            leaf,
        )
        for index in range(6)
        for leaf in ("kernel", "bias")
    }
    encoder = {
        ("backbone", "CNN_0", "Dense_0", leaf): (
            "params",
            "CNN_0",
            "Dense_0",
            leaf,
        )
        for leaf in ("kernel", "bias")
    }
    normalization = {
        ("backbone", "LayerNorm_0", leaf): ("params", "LayerNorm_0", leaf)
        for leaf in ("scale", "bias")
    }
    recurrent = {
        ("backbone", "ScannedRNN_0", "GRUCell_1", gate, leaf): (
            "params",
            "ScannedRNN_0",
            "GRUCell_1",
            gate,
            leaf,
        )
        for gate, leaves in (
            ("ir", ("kernel", "bias")),
            ("hr", ("kernel",)),
            ("iz", ("kernel", "bias")),
            ("hz", ("kernel",)),
            ("in", ("kernel", "bias")),
            ("hn", ("kernel", "bias")),
        )
        for leaf in leaves
    }
    critic: dict[tuple[str, ...], tuple[str, ...]] = {}
    for estimator in range(2):
        target = f"LearnedEstimator_{estimator}"
        prefix = f"learned_estimator_{estimator}"
        for leaf in ("kernel", "bias"):
            critic[
                ("q_heads", target, f"{prefix}_shared_hidden", leaf)
            ] = ("params", "Dense_2", leaf)
            critic[
                ("q_heads", target, f"{prefix}_shared_value", leaf)
            ] = ("params", "Dense_3", leaf)
    return {**convolution, **encoder, **normalization, **recurrent, **critic}


def initialize_vqbc_from_official(
    model: Any,
    *,
    random_key: Any,
    official_params: Mapping[str, Any],
    example_observations: Any,
    example_previous_actions: Any,
    example_previous_team_rewards: Any,
    example_episode_start: Any,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    batch_size = int(example_observations.shape[1])
    model_key, encoder_key = jax.random.split(random_key)
    variables = model.init(
        model_key,
        model.initial_carry(batch_size),
        example_observations,
        example_previous_actions,
        example_previous_team_rewards,
        example_episode_start,
    )
    encoder_variables = model.init(
        encoder_key,
        example_observations,
        jnp.zeros(example_observations.shape[:2], dtype=jnp.int32),
        example_observations,
        jnp.zeros(example_observations.shape[:2], dtype=jnp.bool_),
        method=model.encode_response,
    )
    initialized = {
        str(name): value for name, value in variables["params"].items()
    }
    initialized["response_encoder"] = encoder_variables["params"][
        "response_encoder"
    ]
    return copy_explicit_parameter_leaves(
        initialized,
        official_params,
        explicit_official_parameter_mapping(),
    )


def __getattr__(name: str) -> Any:
    if name == "VQBCFlaxModel":
        return _model_class()
    raise AttributeError(name)


__all__ = [
    "build_vqbc_model",
    "encode_response_codes",
    "explicit_official_parameter_mapping",
    "initialize_vqbc_from_official",
    "response_logits_from_codebook",
    "vqbc_forward",
]
