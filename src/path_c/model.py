"""Path C V4.2 heads attached to the official recurrent feature.

The official OvercookedV2 network remains the sole observation encoder and
recurrent model.  This module receives the recurrent feature returned by that
public network interface and adds only the research heads.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .method import (
    bellman_control_values,
    complete_link_value_class_ids,
    deployment_policy,
    generic_response_information,
    normalized_log_belief,
    policy_effect_decomposition,
    posterior_supported_value_class_count,
    value_signatures,
)


class ModelOutput(NamedTuple):
    features: Any
    q_values: Any
    learned_q_values: Any
    prior_q_values: Any
    centered_advantages: Any
    response_logits: Any
    response_probabilities: Any
    reward_mean: Any
    reward_log_standard_deviation: Any
    continuation_use_mean: Any
    continuation_use_log_standard_deviation: Any
    continuation_mask_mean: Any
    continuation_mask_log_standard_deviation: Any
    value_class_ids: Any
    supported_value_class_count: Any
    j_use: Any
    j_mask: Any
    per_action_response_value: Any
    per_action_net_value: Any
    information_gain: Any
    execution_logits: Any
    mask_execution_logits: Any
    predicted_response_effect: Any
    predicted_policy_cost: Any
    predicted_net_effect: Any
    predicted_regularized_net_effect: Any
    predicted_policy_total_variation: Any


_MODEL_CLASS: Any | None = None


def _head_classes() -> tuple[Any, Any, Any, Any]:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import normal, orthogonal, zeros

    class IndependentDuelingEstimator(nn.Module):
        slot_count: int
        action_count: int
        hidden_dim: int
        output_scale: float
        prefix: str
        condition_on_belief: bool

        @nn.compact
        def __call__(
            self, features: Any, slot_log_belief: Any
        ) -> tuple[Any, Any]:
            belief = jax.lax.stop_gradient(
                jax.nn.softmax(jnp.asarray(slot_log_belief), axis=-1)
            )
            if belief.shape != features.shape[:-1] + (self.slot_count,):
                raise ValueError("Q heads require [..., slot] belief axes.")
            values = []
            advantages = []
            for slot in range(self.slot_count):
                base_hidden = nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(2.0),
                    bias_init=zeros,
                    name=f"{self.prefix}_slot_{slot}_hidden",
                )(features)
                if self.condition_on_belief:
                    base_hidden = base_hidden + nn.Dense(
                        self.hidden_dim,
                        kernel_init=zeros,
                        bias_init=zeros,
                        name=f"{self.prefix}_slot_{slot}_belief_to_hidden",
                    )(belief)
                hidden = nn.relu(base_hidden)
                value = nn.Dense(
                    1,
                    kernel_init=orthogonal(1.0),
                    bias_init=zeros,
                    name=f"{self.prefix}_slot_{slot}_value",
                )(hidden)[..., 0]
                raw_advantage = nn.Dense(
                    self.action_count,
                    kernel_init=orthogonal(self.output_scale),
                    bias_init=zeros,
                    name=f"{self.prefix}_slot_{slot}_advantages",
                )(hidden)
                centered = raw_advantage - jnp.mean(
                    raw_advantage, axis=-1, keepdims=True
                )
                values.append(value[..., None] + centered)
                advantages.append(centered)
            return jnp.stack(values, axis=-2), jnp.stack(advantages, axis=-2)

    class TwinDuelingQ(nn.Module):
        slot_count: int
        action_count: int
        hidden_dim: int
        prior_scale: float

        @nn.compact
        def __call__(
            self, features: Any, slot_log_belief: Any
        ) -> tuple[Any, Any, Any, Any]:
            learned = []
            priors = []
            for estimator in range(2):
                learned_value, unused = IndependentDuelingEstimator(
                    slot_count=self.slot_count,
                    action_count=self.action_count,
                    hidden_dim=self.hidden_dim,
                    output_scale=1.0e-2,
                    prefix=f"learned_estimator_{estimator}",
                    condition_on_belief=True,
                    name=f"LearnedEstimator_{estimator}",
                )(features, slot_log_belief)
                del unused
                prior_value, unused = IndependentDuelingEstimator(
                    slot_count=self.slot_count,
                    action_count=self.action_count,
                    hidden_dim=self.hidden_dim,
                    output_scale=1.0,
                    prefix=f"prior_estimator_{estimator}",
                    condition_on_belief=False,
                    name=f"PriorEstimator_{estimator}",
                )(features, slot_log_belief)
                del unused
                learned.append(learned_value)
                priors.append(jax.lax.stop_gradient(prior_value))
            learned_values = jnp.stack(learned, axis=-3)
            prior_values = jnp.stack(priors, axis=-3)
            q_values = learned_values + self.prior_scale * prior_values
            signatures = q_values - jnp.max(q_values, axis=-1, keepdims=True)
            return q_values, learned_values, prior_values, signatures

    class OutcomeHead(nn.Module):
        """Physical outcomes and behavior-consistent continuation scalars."""

        slot_count: int
        action_count: int
        response_count: int
        hidden_dim: int
        action_embedding_dim: int
        log_standard_deviation_minimum: float
        log_standard_deviation_maximum: float

        @nn.compact
        def __call__(
            self, features: Any, slot_log_belief: Any
        ) -> Mapping[str, Any]:
            detached = jax.lax.stop_gradient(features)
            belief = jax.lax.stop_gradient(
                jax.nn.softmax(jnp.asarray(slot_log_belief), axis=-1)
            )
            if belief.shape != detached.shape[:-1] + (self.slot_count,):
                raise ValueError("Outcome model requires one posterior per feature.")
            action_embedding = self.param(
                "action_embedding",
                normal(0.02),
                (self.action_count, self.action_embedding_dim),
            )
            prefix = detached.shape[:-1]
            feature_grid = jnp.broadcast_to(
                detached[..., None, :],
                prefix + (self.action_count, detached.shape[-1]),
            )
            action_grid = jnp.broadcast_to(
                action_embedding, prefix + action_embedding.shape
            )
            joined = jnp.concatenate((feature_grid, action_grid), axis=-1)

            response_logits_by_slot = []
            reward_mean_by_slot = []
            reward_log_std_by_slot = []
            hidden_by_slot = []
            for slot in range(self.slot_count):
                hidden = nn.relu(
                    nn.Dense(
                        self.hidden_dim,
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=zeros,
                        name=f"slot_{slot}_hidden",
                    )(joined)
                )
                hidden_by_slot.append(hidden)
                response_logits_by_slot.append(
                    nn.Dense(
                        self.response_count,
                        kernel_init=zeros,
                        bias_init=zeros,
                        name=f"slot_{slot}_response_logits",
                    )(hidden)
                )
                reward_mean_by_slot.append(
                    nn.Dense(
                        1,
                        kernel_init=zeros,
                        bias_init=zeros,
                        name=f"slot_{slot}_reward_mean",
                    )(hidden)[..., 0]
                )
                reward_log_std_by_slot.append(
                    nn.Dense(
                        1,
                        kernel_init=zeros,
                        bias_init=zeros,
                        name=f"slot_{slot}_reward_log_standard_deviation",
                    )(hidden)[..., 0]
                )

            response_logits = jnp.stack(response_logits_by_slot, axis=-3)
            response_probabilities = jax.nn.softmax(response_logits, axis=-1)
            reward_mean = jnp.stack(reward_mean_by_slot, axis=-2)
            reward_log_std = jnp.stack(reward_log_std_by_slot, axis=-2)

            physical_joint = belief[..., :, None, None] * response_probabilities
            marginal = jnp.sum(physical_joint, axis=-3)
            posterior = physical_joint / jnp.maximum(
                marginal[..., None, :, :], 1.0e-8
            )
            # [..., slot, action, response] -> [..., action, response, slot]
            use_belief = jnp.moveaxis(posterior, -3, -1)
            mask_belief = jnp.broadcast_to(
                belief[..., None, None, :], use_belief.shape
            )

            def belief_features(values: Any) -> Any:
                clipped = jnp.clip(values, 1.0e-8, 1.0)
                entropy_parts = -clipped * jnp.log(clipped)
                entropy = jnp.sum(entropy_parts, axis=-1, keepdims=True)
                return jnp.concatenate(
                    (clipped, jnp.square(clipped), entropy_parts, entropy),
                    axis=-1,
                )

            use_features = belief_features(use_belief)
            mask_features = belief_features(mask_belief)
            belief_width = 3 * self.slot_count + 1
            parameter_width = 2 * self.response_count * (belief_width + 1)

            def evaluate(parameters: Any, branch_features: Any) -> Any:
                bias = parameters[..., 0]
                coefficients = parameters[..., 1:]
                return bias + jnp.einsum(
                    "...aeyf,...ayf->...aey", coefficients, branch_features
                )

            use_mean_slots = []
            mask_mean_slots = []
            use_std_slots = []
            mask_std_slots = []
            for slot, hidden in enumerate(hidden_by_slot):
                mean_parameters = nn.Dense(
                    parameter_width,
                    kernel_init=zeros,
                    bias_init=zeros,
                    name=f"slot_{slot}_continuation_mean_parameters",
                )(hidden).reshape(
                    prefix
                    + (
                        self.action_count,
                        2,
                        self.response_count,
                        belief_width + 1,
                    )
                )
                std_parameters = nn.Dense(
                    parameter_width,
                    kernel_init=zeros,
                    bias_init=zeros,
                    name=f"slot_{slot}_continuation_log_std_parameters",
                )(hidden).reshape(mean_parameters.shape)
                use_mean_slots.append(evaluate(mean_parameters, use_features))
                mask_mean_slots.append(evaluate(mean_parameters, mask_features))
                use_std_slots.append(evaluate(std_parameters, use_features))
                mask_std_slots.append(evaluate(std_parameters, mask_features))

            def arrange(values: list[Any]) -> Any:
                # slot items [..., action, estimator, response] ->
                # [..., estimator, slot, action, response]
                return jnp.moveaxis(jnp.stack(values, axis=-2), -4, -2)

            use_mean = arrange(use_mean_slots)
            mask_mean = arrange(mask_mean_slots)
            use_log_std = arrange(use_std_slots)
            mask_log_std = arrange(mask_std_slots)
            terminal = self.response_count - 1
            use_mean = use_mean.at[..., terminal].set(0.0)
            mask_mean = mask_mean.at[..., terminal].set(0.0)
            lower = self.log_standard_deviation_minimum
            upper = self.log_standard_deviation_maximum
            return {
                "response_logits": response_logits,
                "response_probabilities": response_probabilities,
                "reward_mean": reward_mean,
                "reward_log_standard_deviation": jnp.clip(
                    reward_log_std, lower, upper
                ),
                "continuation_use_mean": use_mean,
                "continuation_use_log_standard_deviation": jnp.clip(
                    use_log_std, lower, upper
                ),
                "continuation_mask_mean": mask_mean,
                "continuation_mask_log_standard_deviation": jnp.clip(
                    mask_log_std, lower, upper
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
                raise ValueError("Response encoder observations must match.")
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

    return IndependentDuelingEstimator, TwinDuelingQ, OutcomeHead, ResponseEncoder


def _model_class() -> Any:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import flax.linen as nn
    import jax.numpy as jnp

    unused_estimator, TwinDuelingQ, OutcomeHead, ResponseEncoder = _head_classes()
    del unused_estimator

    class PathCHeads(nn.Module):
        hidden_dim: int
        slot_count: int
        action_count: int
        response_count: int
        prior_scale: float
        action_embedding_dim: int
        log_standard_deviation_minimum: float
        log_standard_deviation_maximum: float

        def setup(self) -> None:
            self.previous_action_embedding = nn.Embed(
                num_embeddings=self.action_count + 1,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
            )
            self.previous_action_to_hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )
            self.previous_reward_to_hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )
            self.q_heads = TwinDuelingQ(
                slot_count=self.slot_count,
                action_count=self.action_count,
                hidden_dim=self.hidden_dim,
                prior_scale=self.prior_scale,
            )
            self.outcome = OutcomeHead(
                slot_count=self.slot_count,
                action_count=self.action_count,
                response_count=self.response_count,
                hidden_dim=self.hidden_dim,
                action_embedding_dim=self.action_embedding_dim,
                log_standard_deviation_minimum=self.log_standard_deviation_minimum,
                log_standard_deviation_maximum=self.log_standard_deviation_maximum,
            )
            self.response_encoder = ResponseEncoder(
                action_count=self.action_count,
                hidden_dim=self.hidden_dim,
            )

        def history_features(
            self,
            official_features: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
        ) -> Any:
            features = jnp.asarray(official_features, dtype=jnp.float32)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            rewards = jnp.asarray(previous_team_rewards, dtype=jnp.float32)
            if actions.shape != features.shape[:-1] or rewards.shape != actions.shape:
                raise ValueError(
                    "Previous actions and rewards must share the feature prefix axes."
                )
            action_features = self.previous_action_to_hidden(
                self.previous_action_embedding(actions)
            )
            reward_features = self.previous_reward_to_hidden(rewards[..., None])
            return features + action_features + reward_features

        def from_features(
            self, features: Any, slot_log_belief: Any
        ) -> Mapping[str, Any]:
            q_values, learned, prior, centered = self.q_heads(
                features, slot_log_belief
            )
            return {
                "features": features,
                "q_values": q_values,
                "learned_q_values": learned,
                "prior_q_values": prior,
                "centered_advantages": centered,
                **self.outcome(features, slot_log_belief),
            }

        def __call__(
            self,
            official_features: Any,
            previous_actions: Any,
            previous_team_rewards: Any,
            slot_log_belief: Any,
        ) -> Mapping[str, Any]:
            features = self.history_features(
                official_features,
                previous_actions,
                previous_team_rewards,
            )
            return self.from_features(features, slot_log_belief)

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

    _MODEL_CLASS = PathCHeads
    return PathCHeads


def build_model(
    *,
    hidden_dim: int,
    slot_count: int,
    action_count: int,
    response_count: int,
    prior_scale: float,
    action_embedding_dim: int,
    log_standard_deviation_minimum: float,
    log_standard_deviation_maximum: float,
) -> Any:
    """Construct only the heads; the official adapter owns the recurrent net."""

    return _model_class()(
        hidden_dim=int(hidden_dim),
        slot_count=int(slot_count),
        action_count=int(action_count),
        response_count=int(response_count),
        prior_scale=float(prior_scale),
        action_embedding_dim=int(action_embedding_dim),
        log_standard_deviation_minimum=float(
            log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=float(
            log_standard_deviation_maximum
        ),
    )


def initialize_heads(
    model: Any,
    *,
    random_key: Any,
    example_features: Any,
    example_previous_actions: Any,
    example_previous_team_rewards: Any,
    example_slot_log_belief: Any,
    example_observations: Any,
) -> Mapping[str, Any]:
    """Initialize research heads without copying or remapping official leaves."""

    import jax
    import jax.numpy as jnp

    head_key, response_key = jax.random.split(random_key)
    head_variables = model.init(
        head_key,
        example_features,
        example_previous_actions,
        example_previous_team_rewards,
        example_slot_log_belief,
    )
    time_batch = example_observations.shape[:2]
    response_variables = model.init(
        response_key,
        example_observations,
        jnp.zeros(time_batch, dtype=jnp.int32),
        example_observations,
        jnp.zeros(time_batch, dtype=jnp.bool_),
        method=model.encode_response,
    )
    params = dict(head_variables["params"])
    params["response_encoder"] = response_variables["params"]["response_encoder"]
    return params


def response_logits_from_codebook(
    response_signature: Any, codebook_embeddings: Any
) -> Any:
    import jax.numpy as jnp

    signature = jnp.asarray(response_signature)
    codebook = jnp.asarray(codebook_embeddings)
    if signature.shape[-1] != codebook.shape[-1]:
        raise ValueError("Response signatures and codebook entries differ in width.")
    return -jnp.sum(jnp.square(signature[..., None, :] - codebook), axis=-1)


def encode_response_codes(
    *,
    model: Any,
    params: Mapping[str, Any],
    observations: Any,
    actions: Any,
    next_observations: Any,
    dones: Any,
    codebook_embeddings: Any,
    terminal_response: int,
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


def model_forward(
    *,
    raw_output: Mapping[str, Any],
    reference_logits: Any,
    slot_log_belief: Any,
    temperature: Any,
    generic_temperature: Any,
    deployment_mode: str,
    gamma: float,
) -> ModelOutput:
    import jax.numpy as jnp

    signatures, radii = value_signatures(raw_output["q_values"])
    class_ids = complete_link_value_class_ids(signatures, radii)
    slot_probabilities = jnp.exp(normalized_log_belief(slot_log_belief))
    control = bellman_control_values(
        slot_belief=slot_probabilities,
        response_probabilities=raw_output["response_probabilities"],
        reward_mean=raw_output["reward_mean"],
        continuation_use_mean=raw_output["continuation_use_mean"],
        continuation_mask_mean=raw_output["continuation_mask_mean"],
        gamma=gamma,
    )
    information = generic_response_information(
        slot_belief=slot_probabilities,
        response_probabilities=raw_output["response_probabilities"],
    )
    effects = policy_effect_decomposition(
        reference_logits=reference_logits,
        j_use=control.j_use,
        j_mask=control.j_mask,
        temperature=temperature,
    )
    policy = deployment_policy(
        mode=deployment_mode,
        reference_logits=reference_logits,
        control_values=control,
        information_gain=information,
        temperature=temperature,
        generic_temperature=generic_temperature,
    )
    return ModelOutput(
        features=raw_output["features"],
        q_values=raw_output["q_values"],
        learned_q_values=raw_output["learned_q_values"],
        prior_q_values=raw_output["prior_q_values"],
        centered_advantages=raw_output["centered_advantages"],
        response_logits=raw_output["response_logits"],
        response_probabilities=raw_output["response_probabilities"],
        reward_mean=raw_output["reward_mean"],
        reward_log_standard_deviation=raw_output[
            "reward_log_standard_deviation"
        ],
        continuation_use_mean=raw_output["continuation_use_mean"],
        continuation_use_log_standard_deviation=raw_output[
            "continuation_use_log_standard_deviation"
        ],
        continuation_mask_mean=raw_output["continuation_mask_mean"],
        continuation_mask_log_standard_deviation=raw_output[
            "continuation_mask_log_standard_deviation"
        ],
        value_class_ids=class_ids,
        supported_value_class_count=posterior_supported_value_class_count(
            class_ids=class_ids, slot_log_belief=slot_log_belief
        ),
        j_use=control.j_use,
        j_mask=control.j_mask,
        per_action_response_value=control.per_action_response_value,
        per_action_net_value=control.information_net_value,
        information_gain=information,
        execution_logits=policy.logits,
        mask_execution_logits=effects.mask_policy.logits,
        predicted_response_effect=effects.raw_response_effect,
        predicted_policy_cost=effects.raw_policy_cost,
        predicted_net_effect=effects.raw_net_effect,
        predicted_regularized_net_effect=effects.regularized_net_effect,
        predicted_policy_total_variation=effects.total_variation,
    )


__all__ = [
    "ModelOutput",
    "build_model",
    "encode_response_codes",
    "initialize_heads",
    "model_forward",
    "response_logits_from_codebook",
]
