"""Unified DELTA-ZSC model.

The implementation has two estimators with separate parameter trees:

1. ``BasePolicyModel`` learns task competence by PPO from task-only recurrence
   and a memoryless current-partner branch.
2. ``LatentCoordinationModel`` maximizes one joint response-decision likelihood.

``UnifiedAgent`` combines them only through an analytic KL-constrained mirror
policy.  No auxiliary actor, comparator, separation geometry, learned
capability encoder, context dropout, or loss-weighted control path remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from .behavior_statistics import (
    behavior_features,
    initial_behavior_posterior,
    update_behavior_posterior,
)
from .filtering import filter_step, initial_belief, transition_matrix
from .mirror_policy import expected_action_values, kl_constrained_policy
from .response_model import response_emission_class, response_log_probability
from .types import (
    AgentOutput,
    AgentState,
    BasePolicyOutput,
    LatentOutput,
)
from .voi import coarse_response_probability, myopic_value_of_information


BASE_VARIANT = "base"
RESPONSE_ONLY_VARIANT = "response_only"
JOINT_VARIANT = "joint"
FULL_VARIANT = "full"
METHOD_VARIANTS = (
    BASE_VARIANT,
    RESPONSE_ONLY_VARIANT,
    JOINT_VARIANT,
    FULL_VARIANT,
)

_BASE_CLASS: Any | None = None
_LATENT_CLASS: Any | None = None


def base_policy_model_class() -> Any:
    global _BASE_CLASS
    if _BASE_CLASS is not None:
        return _BASE_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    from src.path_c.task_encoder import (
        instant_partner_encoder_class,
        task_encoder_classes,
    )

    TaskCell, _ = task_encoder_classes()
    InstantPartnerEncoder = instant_partner_encoder_class()

    class BasePolicyModel(nn.Module):
        action_count: int
        task_hidden_dim: int
        instant_partner_dim: int

        def setup(self) -> None:
            self.task_cell = TaskCell(
                hidden_dim=self.task_hidden_dim,
                mask_partner_history=True,
                name="task_encoder",
            )
            self.instant_partner_encoder = InstantPartnerEncoder(
                output_dim=self.instant_partner_dim,
                name="instant_partner_encoder",
            )

        def _heads(self, task_features: Any, instant_partner: Any) -> tuple[Any, Any]:
            combined = jnp.concatenate((task_features, instant_partner), axis=-1)
            hidden = nn.relu(
                nn.Dense(
                    self.task_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="policy_hidden_0",
                )(combined)
            )
            hidden = nn.relu(
                nn.Dense(
                    self.task_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="policy_hidden_1",
                )(hidden)
            )
            logits = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="base_policy_logits",
            )(hidden)
            value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="base_state_value",
            )(hidden)[..., 0]
            return logits, value

        def step(
            self,
            task_carry: Any,
            observation: Any,
            episode_start: Any,
        ) -> tuple[Any, BasePolicyOutput]:
            next_carry, task_features = self.task_cell(
                task_carry, (observation, episode_start)
            )
            instant_partner = self.instant_partner_encoder(observation)
            logits, value = self._heads(task_features, instant_partner)
            return next_carry, BasePolicyOutput(
                task_features=task_features,
                instant_partner=instant_partner,
                base_logits=logits,
                value=value,
            )

        def sequence(
            self,
            initial_task_carry: Any,
            observations: Any,
            episode_starts: Any,
        ) -> tuple[Any, BasePolicyOutput]:
            def one(carry: Any, item: tuple[Any, Any]):
                return self.step(carry, item[0], item[1])

            return jax.lax.scan(
                one, initial_task_carry, (observations, episode_starts)
            )

        def initialize_all(
            self, task_carry: Any, observation: Any, episode_start: Any
        ) -> tuple[Any, BasePolicyOutput]:
            return self.step(task_carry, observation, episode_start)

    _BASE_CLASS = BasePolicyModel
    return BasePolicyModel


def latent_coordination_model_class() -> Any:
    global _LATENT_CLASS
    if _LATENT_CLASS is not None:
        return _LATENT_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    ResponseEmission = response_emission_class()

    class LatentCoordinationModel(nn.Module):
        action_count: int
        component_count: int
        latent_hidden_dim: int
        response_hidden_dim: int
        action_embedding_dim: int
        inventory_factor_count: int

        def setup(self) -> None:
            self.response_emission = ResponseEmission(
                action_count=self.action_count,
                component_count=self.component_count,
                component_dim=self.latent_hidden_dim,
                hidden_dim=self.response_hidden_dim,
                action_embedding_dim=self.action_embedding_dim,
                inventory_factor_count=self.inventory_factor_count,
                name="response_emission",
            )

        def transition(self) -> Any:
            def initializer(key: Any, shape: tuple[int, ...], dtype: Any) -> Any:
                del key
                # A weak identity-biased initialization, not a fixed transition
                # or a scanned method hyperparameter.  Maximum likelihood is
                # free to move every entry.
                return jnp.eye(shape[0], dtype=dtype) * 2.0

            logits = self.param(
                "transition_logits",
                initializer,
                (self.component_count, self.component_count),
            )
            return transition_matrix(logits)

        def response_logits(
            self,
            previous_frame: Any,
            behavior: Any,
            previous_action: Any,
            *,
            include_component_residual: bool = True,
        ) -> Any:
            return self.response_emission(
                jax.lax.stop_gradient(previous_frame),
                behavior,
                previous_action,
                include_component_residual=include_component_residual,
            )

        def response_all_actions(
            self,
            frame: Any,
            behavior: Any,
        ) -> Any:
            frame_array = jnp.asarray(frame)
            behavior_array = jnp.asarray(behavior)
            lead = behavior_array.shape[:-1]
            if frame_array.shape[:-3] != lead:
                raise ValueError("Response all-action frame/context axes differ.")
            actions = jnp.broadcast_to(
                jnp.arange(self.action_count, dtype=jnp.int32),
                lead + (self.action_count,),
            )
            frame_actions = jnp.broadcast_to(
                frame_array[..., None, :, :, :],
                lead + (self.action_count,) + frame_array.shape[-3:],
            )
            behavior_actions = jnp.broadcast_to(
                behavior_array[..., None, :],
                lead + (self.action_count, behavior_array.shape[-1]),
            )
            return self.response_emission(
                jax.lax.stop_gradient(frame_actions),
                behavior_actions,
                actions,
                include_component_residual=True,
            )

        def decision_emission(
            self,
            task_features: Any,
            instant_partner: Any,
            behavior: Any,
            *,
            include_component_residual: bool = True,
        ) -> tuple[Any, Any]:
            task = jnp.asarray(task_features, dtype=jnp.float32)
            instant = jnp.asarray(instant_partner, dtype=jnp.float32)
            statistics = jnp.asarray(behavior, dtype=jnp.float32)
            if not (
                task.shape[:-1] == instant.shape[:-1] == statistics.shape[:-1]
            ):
                raise ValueError("Decision-emission context axes differ.")
            lead = task.shape[:-1]
            common = jnp.concatenate((task, instant, statistics), axis=-1)
            hidden = nn.tanh(
                nn.Dense(
                    self.latent_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="decision_base_hidden_0",
                )(common)
            )
            hidden = nn.tanh(
                nn.Dense(
                    self.latent_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="decision_base_hidden_1",
                )(hidden)
            )
            base_mean = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="decision_base_mean",
            )(hidden)
            base_log_scale = nn.Dense(
                self.action_count,
                kernel_init=zeros,
                bias_init=nn.initializers.constant(1.0),
                name="decision_base_log_scale",
            )(hidden)

            components = self.param(
                "decision_component_embeddings",
                nn.initializers.normal(0.02),
                (self.component_count, self.latent_hidden_dim),
            )
            components = components / jnp.maximum(
                jnp.linalg.norm(components, axis=-1, keepdims=True), 1.0e-6
            )
            common_k = jnp.broadcast_to(
                common[..., None, :],
                lead + (self.component_count, common.shape[-1]),
            )
            components_k = jnp.broadcast_to(
                components.reshape((1,) * len(lead) + components.shape),
                lead + components.shape,
            )
            residual_hidden = nn.tanh(
                nn.Dense(
                    self.latent_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="decision_residual_hidden_0",
                )(jnp.concatenate((common_k, components_k), axis=-1))
            )
            residual_hidden = nn.tanh(
                nn.Dense(
                    self.latent_hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="decision_residual_hidden_1",
                )(residual_hidden)
            )
            residual_mean = nn.Dense(
                self.action_count,
                kernel_init=zeros,
                bias_init=zeros,
                name="decision_residual_mean",
            )(residual_hidden)
            residual_log_scale = nn.Dense(
                self.action_count,
                kernel_init=zeros,
                bias_init=zeros,
                name="decision_residual_log_scale",
            )(residual_hidden)
            if not include_component_residual:
                residual_mean = jnp.zeros_like(residual_mean)
                residual_log_scale = jnp.zeros_like(residual_log_scale)
            mean = base_mean[..., None, :] + residual_mean
            log_scale = base_log_scale[..., None, :] + residual_log_scale
            return mean, log_scale

        def initialize_all(
            self,
            previous_frame: Any,
            current_frame: Any,
            task_features: Any,
            instant_partner: Any,
            behavior: Any,
            previous_action: Any,
        ) -> tuple[Any, Any, Any, Any]:
            transition = self.transition()
            response = self.response_logits(
                previous_frame, behavior, previous_action
            )
            decision = self.decision_emission(
                task_features, instant_partner, behavior
            )
            all_actions = self.response_all_actions(current_frame, behavior)
            return transition, response, decision, all_actions

    _LATENT_CLASS = LatentCoordinationModel
    return LatentCoordinationModel


BasePolicyModel = base_policy_model_class()
LatentCoordinationModel = latent_coordination_model_class()


@dataclass(frozen=True, slots=True)
class UnifiedAgent:
    base_model: Any
    latent_model: Any
    action_count: int
    component_count: int
    adaptation_kl_budget: float
    gamma: float

    def step(
        self,
        *,
        base_params: Any,
        latent_params: Any,
        state: AgentState,
        observation: Any,
        variant: str = FULL_VARIANT,
    ) -> tuple[AgentState, AgentOutput]:
        """Advance legal history and return the analytic deployment policy."""

        import jax
        import jax.numpy as jnp

        normalized_variant = str(variant).lower()
        if normalized_variant not in METHOD_VARIANTS:
            raise ValueError(f"Unknown unified DELTA variant: {variant}")
        next_task_carry, base = self.base_model.apply(
            {"params": base_params},
            state.task_carry,
            observation,
            state.episode_start,
            method=self.base_model.step,
        )
        previous_statistics = behavior_features(state.behavior)
        next_behavior = update_behavior_posterior(
            state.behavior,
            previous_observation=state.previous_observation,
            current_observation=observation,
            episode_start=state.episode_start,
        )
        current_statistics = behavior_features(next_behavior)

        response_logits = self.latent_model.apply(
            {"params": latent_params},
            state.previous_observation,
            previous_statistics,
            state.previous_action,
            method=self.latent_model.response_logits,
        )
        from src.path_c.response_targets import (
            extract_partner_response_targets,
            official_partner_observation_planes,
        )

        planes = official_partner_observation_planes(
            jnp.asarray(observation).shape[-1]
        )
        targets = extract_partner_response_targets(
            state.previous_observation, observation, planes=planes
        )
        response_log_likelihood = response_log_probability(
            response_logits, targets
        )
        response_log_likelihood = jnp.where(
            jnp.asarray(state.episode_start, dtype=jnp.bool_)[..., None],
            jnp.zeros_like(response_log_likelihood),
            response_log_likelihood,
        )
        transition = self.latent_model.apply(
            {"params": latent_params}, method=self.latent_model.transition
        )
        filtered = filter_step(
            state.belief,
            transition,
            response_log_likelihood,
            episode_start=state.episode_start,
        )

        include_decision_residual = normalized_variant in {
            JOINT_VARIANT,
            FULL_VARIANT,
        }
        decision_mean, decision_log_scale = self.latent_model.apply(
            {"params": latent_params},
            jax.lax.stop_gradient(base.task_features),
            jax.lax.stop_gradient(base.instant_partner),
            current_statistics,
            include_component_residual=include_decision_residual,
            method=self.latent_model.decision_emission,
        )
        expected_values = expected_action_values(filtered.posterior, decision_mean)
        voi = jnp.zeros_like(expected_values)
        if normalized_variant == FULL_VARIANT:
            all_response = self.latent_model.apply(
                {"params": latent_params},
                observation,
                current_statistics,
                method=self.latent_model.response_all_actions,
            )
            outcome_probability = coarse_response_probability(
                all_response.visibility, all_response.inventory_change
            )
            voi = myopic_value_of_information(
                belief=filtered.posterior,
                transition=transition,
                component_action_values=decision_mean,
                response_outcome_probability=outcome_probability,
            ).value

        if normalized_variant in {BASE_VARIANT, RESPONSE_ONLY_VARIANT}:
            mirror_logits = base.base_logits
            base_probability = jax.nn.softmax(base.base_logits, axis=-1)
            adaptation_temperature = jnp.full(
                base.base_logits.shape[:-1], jnp.inf, dtype=jnp.float32
            )
            adaptation_kl = jnp.zeros(base.base_logits.shape[:-1], dtype=jnp.float32)
        else:
            bayes_values = expected_values + float(self.gamma) * voi
            mirror = kl_constrained_policy(
                base.base_logits,
                bayes_values,
                kl_budget=float(self.adaptation_kl_budget),
            )
            mirror_logits = mirror.logits
            base_probability = mirror.probabilities
            adaptation_temperature = mirror.temperature
            adaptation_kl = mirror.kl_to_base

        next_state = AgentState(
            task_carry=next_task_carry,
            behavior=next_behavior,
            belief=filtered.posterior,
            previous_observation=jnp.asarray(observation),
            previous_action=state.previous_action,
            episode_start=state.episode_start,
        )
        return next_state, AgentOutput(
            base=base,
            behavior_features=current_statistics,
            latent=LatentOutput(
                predictive_belief=filtered.predictive,
                belief=filtered.posterior,
                response_log_likelihood=response_log_likelihood,
                response_log_evidence=filtered.log_evidence,
                decision_mean=decision_mean,
                decision_log_scale=decision_log_scale,
                expected_action_values=expected_values,
                value_of_information=voi,
                adapted_logits=mirror_logits,
                adaptation_temperature=adaptation_temperature,
                adaptation_kl=adaptation_kl,
            ),
        )


def build_models(
    *,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    instant_partner_dim: int,
    latent_hidden_dim: int,
    response_hidden_dim: int,
    action_embedding_dim: int,
    component_count: int,
) -> tuple[Any, Any]:
    inventory_factor_count = (int(observation_shape[-1]) - 27) // 4 + 2
    if inventory_factor_count <= 0:
        raise ValueError("Official observation inventory contract differs.")
    return (
        BasePolicyModel(
            action_count=int(action_count),
            task_hidden_dim=int(task_hidden_dim),
            instant_partner_dim=int(instant_partner_dim),
        ),
        LatentCoordinationModel(
            action_count=int(action_count),
            component_count=int(component_count),
            latent_hidden_dim=int(latent_hidden_dim),
            response_hidden_dim=int(response_hidden_dim),
            action_embedding_dim=int(action_embedding_dim),
            inventory_factor_count=inventory_factor_count,
        ),
    )


def initial_agent_state(
    *,
    batch_size: int,
    observation_shape: tuple[int, ...],
    task_hidden_dim: int,
    component_count: int,
) -> AgentState:
    import jax.numpy as jnp

    from src.path_c.task_encoder import initial_task_carry

    return AgentState(
        task_carry=initial_task_carry(batch_size, task_hidden_dim),
        behavior=initial_behavior_posterior((int(batch_size),)),
        belief=initial_belief((int(batch_size),), component_count),
        previous_observation=jnp.zeros(
            (int(batch_size), *tuple(int(value) for value in observation_shape)),
            dtype=jnp.float32,
        ),
        previous_action=jnp.zeros((int(batch_size),), dtype=jnp.int32),
        episode_start=jnp.ones((int(batch_size),), dtype=jnp.bool_),
    )


def initialize_parameters(
    *,
    base_model: Any,
    latent_model: Any,
    base_key: Any,
    latent_key: Any,
    example_state: AgentState,
    example_observation: Any,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    base_variables = base_model.init(
        base_key,
        example_state.task_carry,
        example_observation,
        example_state.episode_start,
        method=base_model.initialize_all,
    )
    next_carry, base_output = base_model.apply(
        base_variables,
        example_state.task_carry,
        example_observation,
        example_state.episode_start,
        method=base_model.step,
    )
    del next_carry
    statistics = behavior_features(example_state.behavior)
    latent_variables = latent_model.init(
        latent_key,
        example_state.previous_observation,
        example_observation,
        base_output.task_features,
        base_output.instant_partner,
        statistics,
        example_state.previous_action,
        method=latent_model.initialize_all,
    )
    return base_variables["params"], latent_variables["params"]


__all__ = [
    "BASE_VARIANT",
    "BasePolicyModel",
    "FULL_VARIANT",
    "JOINT_VARIANT",
    "LatentCoordinationModel",
    "METHOD_VARIANTS",
    "RESPONSE_ONLY_VARIANT",
    "UnifiedAgent",
    "build_models",
    "initial_agent_state",
    "initialize_parameters",
]
