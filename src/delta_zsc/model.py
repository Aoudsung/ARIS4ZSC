"""Unified DELTA-ZSC model and deployment state transition.

The model contains exactly two parameter owners:

* ``base_params``: task competence, trained by on-policy PPO only;
* ``latent_params``: transition, response emission, and decision emission,
  trained by one shared-latent composite predictive score only.

Deployment combines the two estimates through an analytic KL-constrained
mirror policy.  DELTA-active adds the deterministic Bayesian VOI computed from
the same response/decision model; it is not a separately trained actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_policy import base_policy_sequence, base_policy_step, init_base_params
from .bayes_voi import myopic_value_of_information_details
from .behavior_statistics import (
    BEHAVIOR_FEATURE_DIM,
    behavior_features,
    initial_behavior_statistics,
)
from .belief_filter import uniform_belief
from .latent_model import init_latent_params, observe_response, predict_decision
from .mirror_policy import mirror_policy_logits
from .observation import partner_visibility
from .response_model import response_predict
from .transition import predict_belief
from .types import ModelOutput, PolicyState, ResponsePrediction


@dataclass(frozen=True, slots=True)
class DeltaModel:
    """Static executable model specification used by JAX transformations."""

    config: Any
    observation_shape: tuple[int, ...]
    action_count: int

    def init_parameters(self, key: Any) -> tuple[Any, Any]:
        import jax

        base_key, latent_key = jax.random.split(key)
        base = init_base_params(
            base_key,
            observation_shape=self.observation_shape,
            task_hidden_dim=self.config.model.task_hidden_dim,
            task_embedding_dim=self.config.model.task_embedding_dim,
            instant_partner_dim=self.config.model.instant_partner_dim,
            action_count=self.action_count,
        )
        latent = init_latent_params(
            latent_key,
            observation_shape=self.observation_shape,
            component_count=self.config.method.latent_components,
            component_embedding_dim=self.config.model.latent_embedding_dim,
            task_dim=self.config.model.task_hidden_dim,
            instant_dim=self.config.model.instant_partner_dim,
            behavior_dim=BEHAVIOR_FEATURE_DIM,
            action_count=self.action_count,
            action_embedding_dim=self.config.model.action_embedding_dim,
            hidden_dim=self.config.model.latent_hidden_dim,
        )
        return base, latent

    def initial_state(self, batch_size: int) -> PolicyState:
        import jax.numpy as jnp

        batch = (int(batch_size),)
        return PolicyState(
            task_carry=jnp.zeros(
                batch + (int(self.config.model.task_hidden_dim),), dtype=jnp.float32
            ),
            belief=uniform_belief(batch, self.config.method.latent_components),
            behavior=initial_behavior_statistics(batch),
            previous_observation=jnp.zeros(
                batch + tuple(self.observation_shape), dtype=jnp.float32
            ),
            previous_action=jnp.zeros(batch, dtype=jnp.int32),
            episode_start=jnp.ones(batch, dtype=jnp.bool_),
        )

    def step(
        self,
        base_params: Any,
        latent_params: Any,
        state: PolicyState,
        observation: Any,
        *,
        compute_latent: bool = True,
        compute_decision: bool = True,
        execute_adaptation: bool = True,
        precomputed_base: tuple[Any, Any, Any, Any, Any] | None = None,
    ) -> tuple[PolicyState, ModelOutput]:
        """Advance one legal observation and form the deployment policy."""

        import jax.numpy as jnp

        start = jnp.asarray(state.episode_start, dtype=jnp.bool_)
        if precomputed_base is None:
            next_task, task, instant, base_logits, value = base_policy_step(
                base_params,
                state.task_carry,
                observation,
                start,
                mask_partner_history=(self.config.method_variant != "history_rnn"),
            )
        else:
            next_task, task, instant, base_logits, value = precomputed_base
        if not bool(compute_latent):
            lead = tuple(base_logits.shape[:-1])
            components = int(self.config.method.latent_components)
            ingredients = (int(self.observation_shape[-1]) - 27) // 4
            factors = ingredients + 2
            zero_component = jnp.zeros(lead + (components,), dtype=jnp.float32)
            zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
            response = ResponsePrediction(
                visibility_logit=zero_component,
                relative_position_logits=jnp.zeros(
                    lead + (components, 25), dtype=jnp.float32
                ),
                direction_logits=jnp.zeros(
                    lead + (components, 4), dtype=jnp.float32
                ),
                inventory_logits=jnp.zeros(
                    lead + (components, factors, 2), dtype=jnp.float32
                ),
                inventory_change_logit=zero_component,
            )
            next_state = state._replace(
                task_carry=next_task,
                previous_observation=jnp.asarray(observation, dtype=jnp.float32),
            )
            return next_state, ModelOutput(
                task_features=task,
                instant_partner=instant,
                base_policy_logits=base_logits,
                policy_logits=base_logits,
                value=value,
                predictive_belief=state.belief,
                belief=state.belief,
                behavior_features=behavior_features(state.behavior),
                response_prediction=response,
                response_negative_log_likelihood=jnp.zeros(lead, dtype=jnp.float32),
                component_decision_means=jnp.zeros(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
                component_decision_variances=jnp.ones(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
                expected_decision_values=zero_action,
                active_voi=zero_action,
                active_voi_raw=zero_action,
                active_information_gain=zero_action,
                active_voi_quadrature_error=zero_action,
                adaptation_kl=jnp.zeros(lead, dtype=jnp.float32),
                adaptation_temperature=jnp.full(lead, jnp.inf, dtype=jnp.float32),
            )
        prior = jnp.where(
            start[..., None],
            uniform_belief(start.shape, self.config.method.latent_components),
            state.belief,
        )
        predictive = jnp.where(
            start[..., None],
            prior,
            predict_belief(prior, latent_params["transition_logits"]),
        )
        (
            next_belief,
            next_behavior,
            response_nll,
            response_prediction,
            unused_response_target,
        ) = observe_response(
            latent_params,
            state.belief,
            state.behavior,
            state.previous_observation,
            observation,
            state.previous_action,
            start,
        )
        del unused_response_target
        statistics = behavior_features(next_behavior)
        if bool(execute_adaptation) and not bool(compute_decision):
            raise ValueError("Deployment adaptation requires the decision emission.")
        if bool(compute_decision):
            decision = predict_decision(
                latent_params, task, instant, next_behavior
            )
            expected_values = jnp.sum(
                next_belief[..., :, None] * decision.means, axis=-2
            )
        else:
            from .types import DecisionPrediction

            lead = tuple(base_logits.shape[:-1])
            components = int(self.config.method.latent_components)
            decision = DecisionPrediction(
                means=jnp.zeros(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
                variances=jnp.ones(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
            )
            expected_values = jnp.zeros_like(base_logits, dtype=jnp.float32)

        zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
        active_voi = zero_action
        active_voi_raw = zero_action
        information_gain = zero_action
        quadrature_error = zero_action
        action_values = expected_values
        variant = str(self.config.method_variant)
        if bool(execute_adaptation) and variant == "delta_active":
            lead = tuple(base_logits.shape[:-1])
            actions = jnp.broadcast_to(
                jnp.arange(self.action_count, dtype=jnp.int32),
                lead + (self.action_count,),
            )
            frame = jnp.broadcast_to(
                jnp.asarray(observation, dtype=jnp.float32)[..., None, :, :, :],
                lead + (self.action_count,) + tuple(self.observation_shape),
            )
            stats = jnp.broadcast_to(
                statistics[..., None, :],
                lead + (self.action_count, statistics.shape[-1]),
            )
            response_by_action = response_predict(
                latent_params["response"],
                latent_params["component_embeddings"],
                frame,
                stats,
                actions,
            )
            voi = myopic_value_of_information_details(
                next_belief,
                latent_params["transition_logits"],
                response_by_action,
                decision.means,
                previous_visibility=partner_visibility(observation),
                sample_count=self.config.model.voi_quadrature_samples,
            )
            active_voi = voi.value
            active_voi_raw = voi.raw_value
            information_gain = voi.expected_information_gain
            quadrature_error = voi.quadrature_error_estimate
            action_values = expected_values + float(self.config.ppo.gamma) * active_voi

        if bool(execute_adaptation) and variant in {"delta_passive", "delta_active"}:
            policy_logits, adaptation_kl, adaptation_temperature = mirror_policy_logits(
                base_logits,
                action_values,
                kl_budget=self.config.method.adaptation_kl_budget,
            )
        else:
            policy_logits = base_logits
            adaptation_kl = jnp.zeros(base_logits.shape[:-1], dtype=jnp.float32)
            adaptation_temperature = jnp.full(
                base_logits.shape[:-1], jnp.inf, dtype=jnp.float32
            )

        next_state = PolicyState(
            task_carry=next_task,
            belief=next_belief,
            behavior=next_behavior,
            previous_observation=jnp.asarray(observation, dtype=jnp.float32),
            # The executed action is inserted by ``observe_after_transition``.
            previous_action=state.previous_action,
            episode_start=start,
        )
        return next_state, ModelOutput(
            task_features=task,
            instant_partner=instant,
            base_policy_logits=base_logits,
            policy_logits=policy_logits,
            value=value,
            predictive_belief=predictive,
            belief=next_belief,
            behavior_features=statistics,
            response_prediction=response_prediction,
            response_negative_log_likelihood=response_nll,
            component_decision_means=decision.means,
            component_decision_variances=decision.variances,
            expected_decision_values=expected_values,
            active_voi=active_voi,
            active_voi_raw=active_voi_raw,
            active_information_gain=information_gain,
            active_voi_quadrature_error=quadrature_error,
            adaptation_kl=adaptation_kl,
            adaptation_temperature=adaptation_temperature,
        )

    def sequence(
        self,
        base_params: Any,
        latent_params: Any,
        initial_state: PolicyState,
        observations: Any,
        previous_actions: Any,
        episode_starts: Any,
        *,
        compute_latent: bool = True,
        compute_decision: bool = True,
        execute_adaptation: bool = True,
    ) -> tuple[PolicyState, ModelOutput]:
        """Replay a time-major legal history under current parameters."""

        import jax
        import jax.numpy as jnp

        (
            final_task_carry,
            task_carries,
            task_features,
            instant,
            base_logits,
            value,
        ) = base_policy_sequence(
            base_params,
            initial_state.task_carry,
            observations,
            episode_starts,
            mask_partner_history=(self.config.method_variant != "history_rnn"),
        )
        if not bool(compute_latent):
            lead = tuple(base_logits.shape[:-1])
            components = int(self.config.method.latent_components)
            ingredients = (int(self.observation_shape[-1]) - 27) // 4
            factors = ingredients + 2
            zero_component = jnp.zeros(lead + (components,), dtype=jnp.float32)
            zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
            response = ResponsePrediction(
                visibility_logit=zero_component,
                relative_position_logits=jnp.zeros(
                    lead + (components, 25), dtype=jnp.float32
                ),
                direction_logits=jnp.zeros(
                    lead + (components, 4), dtype=jnp.float32
                ),
                inventory_logits=jnp.zeros(
                    lead + (components, factors, 2), dtype=jnp.float32
                ),
                inventory_change_logit=zero_component,
            )
            belief = jnp.broadcast_to(initial_state.belief, lead + (components,))
            statistics = behavior_features(initial_state.behavior)
            statistics = jnp.broadcast_to(
                statistics, lead + (statistics.shape[-1],)
            )
            final_state = initial_state._replace(
                task_carry=final_task_carry,
                previous_observation=jnp.asarray(observations[-1], dtype=jnp.float32),
                previous_action=jnp.asarray(previous_actions[-1], dtype=jnp.int32),
                episode_start=jnp.asarray(episode_starts[-1], dtype=jnp.bool_),
            )
            return final_state, ModelOutput(
                task_features=task_features,
                instant_partner=instant,
                base_policy_logits=base_logits,
                policy_logits=base_logits,
                value=value,
                predictive_belief=belief,
                belief=belief,
                behavior_features=statistics,
                response_prediction=response,
                response_negative_log_likelihood=jnp.zeros(lead, dtype=jnp.float32),
                component_decision_means=jnp.zeros(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
                component_decision_variances=jnp.ones(
                    lead + (components, self.action_count), dtype=jnp.float32
                ),
                expected_decision_values=zero_action,
                active_voi=zero_action,
                active_voi_raw=zero_action,
                active_information_gain=zero_action,
                active_voi_quadrature_error=zero_action,
                adaptation_kl=jnp.zeros(lead, dtype=jnp.float32),
                adaptation_temperature=jnp.full(lead, jnp.inf, dtype=jnp.float32),
            )

        def one(state: PolicyState, values: tuple[Any, ...]):
            (
                observation,
                previous_action,
                start,
                next_task,
                task,
                current_instant,
                current_logits,
                current_value,
            ) = values
            current = state._replace(
                previous_action=previous_action,
                episode_start=start,
            )
            return self.step(
                base_params,
                latent_params,
                current,
                observation,
                compute_latent=compute_latent,
                compute_decision=compute_decision,
                execute_adaptation=execute_adaptation,
                precomputed_base=(
                    next_task,
                    task,
                    current_instant,
                    current_logits,
                    current_value,
                ),
            )

        return jax.lax.scan(
            one,
            initial_state,
            (
                observations,
                previous_actions,
                episode_starts,
                task_carries,
                task_features,
                instant,
                base_logits,
                value,
            ),
        )


def observe_after_transition(
    state: PolicyState,
    *,
    action: Any,
    done: Any,
) -> PolicyState:
    """Insert the only post-action facts required by the next legal step."""

    import jax.numpy as jnp

    return state._replace(
        previous_action=jnp.asarray(action, dtype=jnp.int32),
        episode_start=jnp.asarray(done, dtype=jnp.bool_),
    )


__all__ = ["DeltaModel", "observe_after_transition"]
