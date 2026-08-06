"""Unified DELTA-ZSC v4 model and deployment transition.

``base_params`` own task competence and are trained only by on-policy PPO.
``latent_params`` own the episode-static semantic response, delayed probe
response, and current/post-response decision emissions.  Deployment uses an
analytic KL-constrained mirror policy; no additional actor or critic is created
per latent component.
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
from .decision_model import successor_decision_predict
from .latent_model import init_latent_params, observe_response, predict_decision
from .mirror_policy import mirror_policy_logits
from .observation import (
    INTERFACE_EVENT_CLASSES,
    PARTNER_DIRECTION_CLASSES,
    PARTNER_INVENTORY_FACTOR_CLASSES,
    PARTNER_POSITION_CLASSES,
)
from .response_model import probe_response_predict
from .types import (
    DecisionPrediction,
    DirectResponsePrediction,
    ModelOutput,
    PolicyState,
    ProbeResponsePrediction,
    ResponsePrediction,
)


def _zero_response(
    lead: tuple[int, ...], components: int, factors: int
) -> ResponsePrediction:
    import jax.numpy as jnp

    return ResponsePrediction(
        direct=DirectResponsePrediction(
            visibility_logit=jnp.zeros(lead, dtype=jnp.float32),
            relative_position_logits=jnp.zeros(
                lead + (components, PARTNER_POSITION_CLASSES), dtype=jnp.float32
            ),
            direction_logits=jnp.zeros(lead + (components, PARTNER_DIRECTION_CLASSES), dtype=jnp.float32),
            inventory_logits=jnp.zeros(
                lead + (components, factors, PARTNER_INVENTORY_FACTOR_CLASSES), dtype=jnp.float32
            ),
            inventory_change_logit=jnp.zeros(lead, dtype=jnp.float32),
        ),
        interface_availability_logit=jnp.zeros(lead, dtype=jnp.float32),
        interface_change_logit=jnp.zeros(lead, dtype=jnp.float32),
        interface_event_logits=jnp.zeros(
            lead + (components, INTERFACE_EVENT_CLASSES), dtype=jnp.float32
        ),
        recipe_change_logit=jnp.zeros(lead, dtype=jnp.float32),
    )


def _zero_probe_response(
    lead: tuple[int, ...], probes: int, components: int
) -> ProbeResponsePrediction:
    import jax.numpy as jnp

    shared = lead + (int(probes),)
    return ProbeResponsePrediction(
        visibility_logit=jnp.zeros(shared, dtype=jnp.float32),
        interface_availability_logit=jnp.zeros(shared, dtype=jnp.float32),
        interface_change_logit=jnp.zeros(shared, dtype=jnp.float32),
        interface_event_logits=jnp.zeros(
            shared + (int(components), INTERFACE_EVENT_CLASSES), dtype=jnp.float32
        ),
    )


def _zero_decision(
    lead: tuple[int, ...], components: int, actions: int
) -> DecisionPrediction:
    import jax.numpy as jnp

    return DecisionPrediction(
        means=jnp.zeros(lead + (components, actions), dtype=jnp.float32),
        variances=jnp.ones(lead + (components, actions), dtype=jnp.float32),
        shared_means=jnp.zeros(lead + (actions,), dtype=jnp.float32),
        component_residuals=jnp.zeros(
            lead + (components, actions), dtype=jnp.float32
        ),
    )


@dataclass(frozen=True, slots=True)
class DeltaModel:
    """Static executable model specification used by JAX transformations."""

    config: Any
    observation_shape: tuple[int, ...]
    action_count: int

    def init_parameters(
        self,
        key: Any,
        *,
        semantic_initializer: Any | None = None,
        semantic_event_bias: Any | None = None,
    ) -> tuple[Any, Any]:
        import jax

        if semantic_initializer is not None and semantic_event_bias is not None:
            raise ValueError(
                "Provide either semantic_initializer or semantic_event_bias, not both."
            )
        if semantic_initializer is not None:
            semantic_event_bias = semantic_initializer.event_component_bias

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
            semantic_event_bias=semantic_event_bias,
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
            probe_continuation_pending=jnp.zeros(batch, dtype=jnp.bool_),
        )

    def _empty_output(
        self,
        *,
        state: PolicyState,
        next_task: Any,
        task: Any,
        instant: Any,
        base_logits: Any,
        value: Any,
        observation: Any,
    ) -> tuple[PolicyState, ModelOutput]:
        import jax.numpy as jnp

        lead = tuple(base_logits.shape[:-1])
        components = int(self.config.method.latent_components)
        factors = (int(self.observation_shape[-1]) - 27) // 4 + 2
        zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
        response = _zero_response(lead, components, factors)
        probe = _zero_probe_response(lead, self.action_count, components)
        decision = _zero_decision(lead, components, self.action_count)
        successor = _zero_decision(
            lead + (self.action_count,), components, self.action_count
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
            probe_response_prediction=probe,
            response_negative_log_likelihood=jnp.zeros(lead, dtype=jnp.float32),
            component_decision_means=decision.means,
            component_decision_variances=decision.variances,
            component_successor_decision_means=successor.means,
            component_successor_decision_variances=successor.variances,
            expected_decision_values=zero_action,
            active_voi=zero_action,
            active_information_gain=zero_action,
            active_probe_eligible=jnp.zeros(lead, dtype=jnp.bool_),
            adaptation_kl=jnp.zeros(lead, dtype=jnp.float32),
            adaptation_temperature=jnp.full(lead, jnp.inf, dtype=jnp.float32),
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
        pending_continuation = jnp.where(
            start,
            jnp.zeros_like(start, dtype=jnp.bool_),
            jnp.asarray(state.probe_continuation_pending, dtype=jnp.bool_),
        )
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
            return self._empty_output(
                state=state,
                next_task=next_task,
                task=task,
                instant=instant,
                base_logits=base_logits,
                value=value,
                observation=observation,
            )

        (
            next_belief,
            next_behavior,
            response_nll,
            response_prediction,
            unused_target,
            observed_prior,
        ) = observe_response(
            latent_params,
            state.belief,
            state.behavior,
            state.previous_observation,
            observation,
            state.previous_action,
            start,
        )
        del unused_target
        statistics = behavior_features(next_behavior)
        if bool(execute_adaptation) and not bool(compute_decision):
            raise ValueError("Deployment adaptation requires the decision emission.")

        lead = tuple(base_logits.shape[:-1])
        components = int(self.config.method.latent_components)
        zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
        if bool(compute_decision):
            decision = predict_decision(latent_params, task, instant, next_behavior)
            expected_values = jnp.sum(
                next_belief[..., :, None] * decision.means, axis=-2
            )
        else:
            decision = _zero_decision(lead, components, self.action_count)
            expected_values = zero_action

        probe_prediction = _zero_probe_response(
            lead, self.action_count, components
        )
        successor = _zero_decision(
            lead + (self.action_count,), components, self.action_count
        )
        active_voi = zero_action
        information_gain = zero_action
        action_values = expected_values
        variant = str(self.config.method_variant)
        next_pending = jnp.zeros_like(pending_continuation, dtype=jnp.bool_)
        if bool(execute_adaptation) and variant == "delta_active":
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
            probe_prediction = probe_response_predict(
                latent_params["probe_response"],
                latent_params["component_embeddings"],
                frame,
                stats,
                actions,
            )
            successor = successor_decision_predict(
                latent_params["decision"],
                latent_params["component_embeddings"],
                task,
                instant,
                statistics,
                actions,
            )
            voi = myopic_value_of_information_details(
                next_belief, probe_prediction, successor.means
            )
            raw_voi = voi.value
            raw_information_gain = voi.expected_information_gain
            # The delayed response is first observable at t+2, after one
            # collection-time base continuation.  The information value is
            # therefore discounted by gamma squared.
            active_values = expected_values + (
                float(self.config.ppo.gamma) ** 2
            ) * raw_voi
            (
                candidate_logits,
                candidate_kl,
                candidate_temperature,
            ) = mirror_policy_logits(
                base_logits,
                active_values,
                kl_budget=self.config.method.adaptation_kl_budget,
            )
            active_lane = ~pending_continuation
            policy_logits = jnp.where(
                active_lane[..., None], candidate_logits, base_logits
            )
            active_voi = jnp.where(active_lane[..., None], raw_voi, zero_action)
            information_gain = jnp.where(
                active_lane[..., None], raw_information_gain, zero_action
            )
            adaptation_kl = jnp.where(
                active_lane, candidate_kl, jnp.zeros_like(candidate_kl)
            )
            adaptation_temperature = jnp.where(
                active_lane,
                candidate_temperature,
                jnp.full_like(candidate_temperature, jnp.inf),
            )
            # An active probe is followed by exactly one base-policy action.
            # On a pending lane this step consumes that continuation and does
            # not immediately launch a second overlapping probe.
            next_pending = active_lane
        elif bool(execute_adaptation) and variant == "delta_passive":
            policy_logits, adaptation_kl, adaptation_temperature = mirror_policy_logits(
                base_logits,
                action_values,
                kl_budget=self.config.method.adaptation_kl_budget,
            )
        else:
            policy_logits = base_logits
            adaptation_kl = jnp.zeros(lead, dtype=jnp.float32)
            adaptation_temperature = jnp.full(lead, jnp.inf, dtype=jnp.float32)

        next_state = PolicyState(
            task_carry=next_task,
            belief=next_belief,
            behavior=next_behavior,
            previous_observation=jnp.asarray(observation, dtype=jnp.float32),
            previous_action=state.previous_action,
            episode_start=start,
            probe_continuation_pending=next_pending,
        )
        return next_state, ModelOutput(
            task_features=task,
            instant_partner=instant,
            base_policy_logits=base_logits,
            policy_logits=policy_logits,
            value=value,
            predictive_belief=observed_prior,
            belief=next_belief,
            behavior_features=statistics,
            response_prediction=response_prediction,
            probe_response_prediction=probe_prediction,
            response_negative_log_likelihood=response_nll,
            component_decision_means=decision.means,
            component_decision_variances=decision.variances,
            component_successor_decision_means=successor.means,
            component_successor_decision_variances=successor.variances,
            expected_decision_values=expected_values,
            active_voi=active_voi,
            active_information_gain=information_gain,
            active_probe_eligible=(
                (~pending_continuation)
                if bool(execute_adaptation) and variant == "delta_active"
                else jnp.zeros(lead, dtype=jnp.bool_)
            ),
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
            factors = (int(self.observation_shape[-1]) - 27) // 4 + 2
            zero_action = jnp.zeros_like(base_logits, dtype=jnp.float32)
            response = _zero_response(lead, components, factors)
            probe = _zero_probe_response(lead, self.action_count, components)
            decision = _zero_decision(lead, components, self.action_count)
            successor = _zero_decision(
                lead + (self.action_count,), components, self.action_count
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
                probe_response_prediction=probe,
                response_negative_log_likelihood=jnp.zeros(lead, dtype=jnp.float32),
                component_decision_means=decision.means,
                component_decision_variances=decision.variances,
                component_successor_decision_means=successor.means,
                component_successor_decision_variances=successor.variances,
                expected_decision_values=zero_action,
                active_voi=zero_action,
                active_information_gain=zero_action,
                active_probe_eligible=jnp.zeros(lead, dtype=jnp.bool_),
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

    terminal = jnp.asarray(done, dtype=jnp.bool_)
    return state._replace(
        previous_action=jnp.asarray(action, dtype=jnp.int32),
        episode_start=terminal,
        probe_continuation_pending=jnp.where(
            terminal,
            jnp.zeros_like(terminal, dtype=jnp.bool_),
            jnp.asarray(state.probe_continuation_pending, dtype=jnp.bool_),
        ),
    )


__all__ = ["DeltaModel", "observe_after_transition"]
