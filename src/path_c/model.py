"""Single-path DELTA-ZSC V6 deployable model."""

from __future__ import annotations

from typing import Any, Mapping

from .types import ContextOutput, GaussianBelief, ModelOutput, PolicyState, ResponsePrediction

_MODEL_CLASS: Any | None = None


def _model_class() -> Any:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    from .belief_encoder import belief_encoder_classes
    from .belief_set_encoder import gaussian_summary, prior_gaussian_summary
    from .response_decoder import response_decoder_class
    from .task_encoder import task_encoder_classes
    from .universal_actor import universal_actor_class
    from .universal_critic import universal_critic_class

    TaskCell, _ = task_encoder_classes()
    BeliefCell, _ = belief_encoder_classes()
    Actor = universal_actor_class()
    Critic = universal_critic_class()
    Decoder = response_decoder_class()

    class DELTAZSCModel(nn.Module):
        observation_shape: tuple[int, ...]
        action_count: int
        task_hidden_dim: int
        belief_hidden_dim: int
        latent_dim: int
        actor_hidden_dim: int
        critic_hidden_dim: int
        response_hidden_dim: int
        modulation_rank: int
        action_embedding_dim: int
        log_standard_deviation_minimum: float
        log_standard_deviation_maximum: float

        def setup(self) -> None:
            self.task_cell = TaskCell(
                hidden_dim=self.task_hidden_dim,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                name="task_encoder",
            )
            self.belief_cell = BeliefCell(
                hidden_dim=self.belief_hidden_dim,
                latent_dim=self.latent_dim,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                log_standard_deviation_minimum=self.log_standard_deviation_minimum,
                log_standard_deviation_maximum=self.log_standard_deviation_maximum,
                name="belief_encoder",
            )
            self.actor = Actor(
                action_count=self.action_count,
                hidden_dim=self.actor_hidden_dim,
                modulation_rank=self.modulation_rank,
                name="universal_actor",
            )
            self.critic = Critic(
                action_count=self.action_count,
                hidden_dim=self.critic_hidden_dim,
                name="universal_critic",
            )
            self.decoder = Decoder(
                action_count=self.action_count,
                hidden_dim=self.response_hidden_dim,
                action_embedding_dim=self.action_embedding_dim,
                inventory_factor_count=(self.observation_shape[-1] - 27) // 4 + 2,
                name="response_decoder",
            )

        def _context_step(
            self, state: PolicyState, observation: Any
        ) -> tuple[PolicyState, ContextOutput]:
            next_task_carry, task_features = self.task_cell(
                state.task_carry,
                (observation, state.previous_action, state.episode_start),
            )
            next_belief_carry, belief_values = self.belief_cell(
                state.belief.recurrent_carry,
                (
                    state.previous_observation,
                    observation,
                    state.previous_action,
                    state.episode_start,
                ),
            )
            mean, log_std, uncertainty = belief_values
            context = ContextOutput(task_features, mean, log_std, uncertainty)
            next_state = PolicyState(
                task_carry=next_task_carry,
                belief=GaussianBelief(next_belief_carry, mean, log_std, uncertainty),
                previous_observation=jnp.asarray(
                    observation, dtype=state.previous_observation.dtype
                ),
                previous_action=state.previous_action,
                episode_start=state.episode_start,
            )
            return next_state, context

        def _outputs_from_context(
            self, *, context: ContextOutput, context_dropout_mask: Any
        ) -> ModelOutput:
            full_summary = gaussian_summary(
                context.belief_mean,
                context.belief_log_standard_deviation,
                context.normalized_uncertainty,
            )
            prior_summary = prior_gaussian_summary(full_summary, self.latent_dim)
            drop = jnp.asarray(context_dropout_mask, dtype=jnp.bool_)
            if drop.shape != context.task_features.shape[:-1]:
                drop = jnp.broadcast_to(drop, context.task_features.shape[:-1])
            behavior_summary = jnp.where(drop[..., None], prior_summary, full_summary)
            logits = self.actor(context.task_features, behavior_summary)
            state_value, raw_q1, raw_q2 = self.critic(
                context.task_features,
                context.belief_mean,
                behavior_summary,
            )
            return ModelOutput(
                task_features=context.task_features,
                belief_summary=full_summary,
                belief_mean=context.belief_mean,
                belief_log_standard_deviation=context.belief_log_standard_deviation,
                normalized_uncertainty=context.normalized_uncertainty,
                policy_logits=logits,
                state_value=state_value,
                raw_q1=raw_q1,
                raw_q2=raw_q2,
                action_values=jnp.minimum(raw_q1, raw_q2),
            )

        def step(
            self,
            state: PolicyState,
            observation: Any,
            context_dropout_mask: Any = False,
        ) -> tuple[PolicyState, ModelOutput]:
            next_state, context = self._context_step(state, observation)
            return next_state, self._outputs_from_context(
                context=context, context_dropout_mask=context_dropout_mask
            )

        def context_sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            episode_starts: Any,
        ) -> tuple[PolicyState, ContextOutput]:
            obs = jnp.asarray(observations)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
            if not (obs.shape[0] == actions.shape[0] == starts.shape[0]):
                raise ValueError("Context sequence time axes differ.")

            def one(current: PolicyState, values: tuple[Any, Any, Any]):
                observation, action, start = values
                current = current._replace(previous_action=action, episode_start=start)
                return self._context_step(current, observation)

            return jax.lax.scan(one, initial_state, (obs, actions, starts))

        def sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            episode_starts: Any,
            context_dropout_masks: Any,
        ) -> tuple[PolicyState, ModelOutput]:
            obs = jnp.asarray(observations)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
            drops = jnp.asarray(context_dropout_masks, dtype=jnp.bool_)
            if not (
                obs.shape[0] == actions.shape[0] == starts.shape[0] == drops.shape[0]
            ):
                raise ValueError("Sequence time axes differ.")

            def one(current: PolicyState, values: tuple[Any, Any, Any, Any]):
                observation, action, start, drop = values
                current = current._replace(previous_action=action, episode_start=start)
                return self.step(current, observation, drop)

            return jax.lax.scan(one, initial_state, (obs, actions, starts, drops))

        def response_from_context_and_action(
            self, task_features: Any, belief_summary: Any, actions: Any
        ) -> ResponsePrediction:
            prefix = task_features.shape[:-1]
            action = jnp.asarray(actions, dtype=jnp.int32)
            if action.shape != prefix:
                raise ValueError("Selected response actions do not match context axes.")
            values = self.decoder(
                jax.lax.stop_gradient(task_features), belief_summary, action
            )
            return ResponsePrediction(*values)

        def response_sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            episode_starts: Any,
            executed_actions: Any,
        ) -> tuple[PolicyState, ResponsePrediction]:
            final_state, context = self.context_sequence(
                initial_state, observations, previous_actions, episode_starts
            )
            summary = gaussian_summary(
                context.belief_mean,
                context.belief_log_standard_deviation,
                context.normalized_uncertainty,
            )
            prediction = self.response_from_context_and_action(
                context.task_features[:-1], summary[:-1], executed_actions
            )
            return final_state, prediction

        def policy_logits_from_features_and_summary(
            self, task_features: Any, belief_summary: Any
        ) -> Any:
            return self.actor(task_features, belief_summary)

        def twin_action_values_from_features_and_latent(
            self, task_features: Any, latent: Any
        ) -> tuple[Any, Any]:
            value = jnp.asarray(latent, dtype=jnp.float32)
            # Shaped value is not consumed on the particle path.  A correctly
            # shaped deterministic summary keeps parameter reuse explicit.
            shaped_summary = jnp.concatenate(
                (value, jnp.zeros_like(value), jnp.zeros(value.shape[:-1] + (1,))),
                axis=-1,
            )
            _, raw_q1, raw_q2 = self.critic(task_features, value, shaped_summary)
            return raw_q1, raw_q2

        def action_values_from_features_and_latent(
            self, task_features: Any, latent: Any
        ) -> Any:
            raw_q1, raw_q2 = self.twin_action_values_from_features_and_latent(
                task_features, latent
            )
            return jnp.minimum(raw_q1, raw_q2)

        def initialize_all(
            self, state: PolicyState, observation: Any, partner_code: Any
        ) -> tuple[Any, ...]:
            del partner_code
            next_state, output = self.step(
                state,
                observation,
                jnp.zeros(state.previous_action.shape, dtype=jnp.bool_),
            )
            response = self.response_from_context_and_action(
                output.task_features,
                output.belief_summary,
                jnp.zeros(state.previous_action.shape, dtype=jnp.int32),
            )
            return next_state, output, response

        def __call__(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            episode_starts: Any,
            context_dropout_masks: Any,
        ) -> tuple[PolicyState, ModelOutput]:
            return self.sequence(
                initial_state,
                observations,
                previous_actions,
                episode_starts,
                context_dropout_masks,
            )

    _MODEL_CLASS = DELTAZSCModel
    return DELTAZSCModel


def build_model(
    *,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    belief_hidden_dim: int,
    latent_dim: int,
    actor_hidden_dim: int,
    critic_hidden_dim: int,
    response_hidden_dim: int,
    modulation_rank: int,
    action_embedding_dim: int,
    log_standard_deviation_minimum: float,
    log_standard_deviation_maximum: float,
) -> Any:
    return _model_class()(
        observation_shape=tuple(int(value) for value in observation_shape),
        action_count=int(action_count),
        task_hidden_dim=int(task_hidden_dim),
        belief_hidden_dim=int(belief_hidden_dim),
        latent_dim=int(latent_dim),
        actor_hidden_dim=int(actor_hidden_dim),
        critic_hidden_dim=int(critic_hidden_dim),
        response_hidden_dim=int(response_hidden_dim),
        modulation_rank=int(modulation_rank),
        action_embedding_dim=int(action_embedding_dim),
        log_standard_deviation_minimum=float(log_standard_deviation_minimum),
        log_standard_deviation_maximum=float(log_standard_deviation_maximum),
    )


def initial_policy_state(
    *,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    belief_hidden_dim: int,
    latent_dim: int,
) -> PolicyState:
    import jax.numpy as jnp

    from .belief_encoder import initial_belief_carry
    from .task_encoder import initial_task_carry

    count = int(batch_size)
    return PolicyState(
        task_carry=initial_task_carry(count, task_hidden_dim),
        belief=GaussianBelief(
            recurrent_carry=initial_belief_carry(count, belief_hidden_dim),
            mean=jnp.zeros((count, latent_dim), dtype=jnp.float32),
            log_standard_deviation=jnp.zeros((count, latent_dim), dtype=jnp.float32),
            normalized_uncertainty=jnp.full(
                (count,), 5.0 / 7.0, dtype=jnp.float32
            ),
        ),
        previous_observation=jnp.zeros(
            (count,) + tuple(observation_shape), dtype=jnp.float32
        ),
        previous_action=jnp.full((count,), int(action_count), dtype=jnp.int32),
        episode_start=jnp.ones((count,), dtype=jnp.bool_),
    )


def initialize_model_parameters(
    model: Any,
    *,
    key: Any,
    example_state: PolicyState,
    example_observation: Any,
    partner_code_dim: int,
) -> Mapping[str, Any]:
    import jax.numpy as jnp

    if int(partner_code_dim) <= 0:
        raise ValueError("partner_code_dim must be positive.")
    partner_code = jnp.zeros(
        example_state.previous_action.shape + (int(partner_code_dim),),
        dtype=jnp.float32,
    )
    variables = model.init(
        key,
        example_state,
        example_observation,
        partner_code,
        method=model.initialize_all,
    )
    return variables["params"]


__all__ = [
    "ModelOutput",
    "build_model",
    "initial_policy_state",
    "initialize_model_parameters",
]
