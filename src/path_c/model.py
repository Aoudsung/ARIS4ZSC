"""Unified fixed-capacity DELTA-ZSC model.

There is exactly one task encoder, one continuous belief encoder, one actor, one
critic, and one response decoder.  Privileged teacher contexts pass through the
same belief-set encoder, actor, and critic.
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import (
    ContextOutput,
    GaussianMixtureBelief,
    ModelOutput,
    PolicyState,
    ResponsePrediction,
    TeacherOutput,
)

_MODEL_CLASS: Any | None = None


def _model_class() -> Any:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    from .belief_encoder import belief_encoder_classes
    from .belief_set_encoder import belief_set_encoder_class
    from .response_decoder import response_decoder_class
    from .task_encoder import task_encoder_classes
    from .teacher_context import code_teacher_class, degenerate_gaussian_mixture
    from .universal_actor import universal_actor_class
    from .universal_critic import universal_critic_class

    TaskCell, unused_task_scan = task_encoder_classes()
    BeliefCell, unused_belief_scan, FullTeacher = belief_encoder_classes()
    del unused_task_scan, unused_belief_scan
    BeliefSet = belief_set_encoder_class()
    Actor = universal_actor_class()
    Critic = universal_critic_class()
    Decoder = response_decoder_class()
    CodeTeacher = code_teacher_class()

    class DELTAZSCModel(nn.Module):
        observation_shape: tuple[int, ...]
        action_count: int
        task_hidden_dim: int
        belief_hidden_dim: int
        latent_dim: int
        mixture_components: int
        belief_embedding_dim: int
        actor_hidden_dim: int
        critic_hidden_dim: int
        response_hidden_dim: int
        modulation_rank: int
        action_embedding_dim: int
        log_variance_minimum: float
        log_variance_maximum: float
        response_log_std_minimum: float
        response_log_std_maximum: float

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
                mixture_components=self.mixture_components,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                log_variance_minimum=self.log_variance_minimum,
                log_variance_maximum=self.log_variance_maximum,
                name="belief_encoder",
            )
            self.belief_set = BeliefSet(
                latent_dim=self.latent_dim,
                hidden_dim=self.belief_hidden_dim,
                output_dim=self.belief_embedding_dim,
                name="belief_set_encoder",
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
                observation_size=int(__import__("math").prod(self.observation_shape)),
                action_count=self.action_count,
                hidden_dim=self.response_hidden_dim,
                action_embedding_dim=self.action_embedding_dim,
                log_std_minimum=self.response_log_std_minimum,
                log_std_maximum=self.response_log_std_maximum,
                name="response_decoder",
            )
            self.code_teacher = CodeTeacher(
                latent_dim=self.latent_dim,
                hidden_dim=self.belief_hidden_dim,
                name="code_teacher",
            )
            self.full_teacher = FullTeacher(
                hidden_dim=self.belief_hidden_dim,
                latent_dim=self.latent_dim,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                name="full_trajectory_teacher",
            )

        def _outputs_from_context(
            self,
            *,
            context: ContextOutput,
            gate: Any,
        ) -> ModelOutput:
            belief_embedding = self.belief_set(
                context.mixture_logits,
                context.mixture_means,
                context.mixture_log_variances,
                context.support_score,
            )
            # Actor gradients are deliberately blocked at the belief interface.
            actor_belief = jax.lax.stop_gradient(belief_embedding)
            base_logits, residual_logits, execution_logits = self.actor(
                context.task_features, actor_belief, gate
            )
            state_value, action_values = self.critic(
                context.task_features, belief_embedding
            )
            prefix = context.task_features.shape[:-1]
            return ModelOutput(
                task_features=context.task_features,
                belief_embedding=belief_embedding,
                mixture_logits=context.mixture_logits,
                mixture_means=context.mixture_means,
                mixture_log_variances=context.mixture_log_variances,
                support_score=context.support_score,
                base_logits=base_logits,
                residual_logits=residual_logits,
                gate=jnp.broadcast_to(jnp.asarray(gate), prefix),
                execution_logits=execution_logits,
                state_value=state_value,
                action_values=action_values,
            )

        def _context_step(
            self,
            state: PolicyState,
            observation: Any,
        ) -> tuple[PolicyState, ContextOutput]:
            next_task_carry, task_features = self.task_cell(
                state.task_carry,
                (
                    observation,
                    state.previous_action,
                    state.previous_reward,
                    state.episode_start,
                ),
            )
            next_belief_carry, belief_values = self.belief_cell(
                state.belief.recurrent_carry,
                (
                    state.previous_observation,
                    observation,
                    state.previous_action,
                    state.previous_reward,
                    state.episode_start,
                ),
            )
            mixture_logits, means, log_variances, support_score = belief_values
            context = ContextOutput(
                task_features=task_features,
                mixture_logits=mixture_logits,
                mixture_means=means,
                mixture_log_variances=log_variances,
                support_score=support_score,
            )
            next_state = PolicyState(
                task_carry=next_task_carry,
                belief=GaussianMixtureBelief(
                    recurrent_carry=next_belief_carry,
                    mixture_logits=mixture_logits,
                    means=means,
                    log_variances=log_variances,
                    support_score=support_score,
                ),
                # Official OvercookedV2 observations are integer tensors, while
                # the recurrent history carry is initialized as float32.  Keep
                # the carry dtype invariant across `lax.scan`; the belief and
                # task encoders both consume observations in float32.
                previous_observation=jnp.asarray(
                    observation, dtype=state.previous_observation.dtype
                ),
                previous_action=state.previous_action,
                previous_reward=state.previous_reward,
                episode_start=state.episode_start,
            )
            return next_state, context

        def step(
            self,
            state: PolicyState,
            observation: Any,
            gate_override: Any,
        ) -> tuple[PolicyState, ModelOutput]:
            next_state, context = self._context_step(state, observation)
            output = self._outputs_from_context(
                context=context,
                gate=jnp.asarray(gate_override, dtype=jnp.float32),
            )
            return next_state, output

        def context_sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            previous_rewards: Any,
            episode_starts: Any,
        ) -> tuple[PolicyState, ContextOutput]:
            """Replay legal history without evaluating actor, critic, or decoder."""

            obs = jnp.asarray(observations)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            rewards = jnp.asarray(previous_rewards, dtype=jnp.float32)
            starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
            if not (
                obs.shape[0]
                == actions.shape[0]
                == rewards.shape[0]
                == starts.shape[0]
            ):
                raise ValueError("Context sequence time axes differ.")

            def one(
                current: PolicyState,
                values: tuple[Any, Any, Any, Any],
            ) -> tuple[PolicyState, ContextOutput]:
                observation, action, reward, start = values
                current = current._replace(
                    previous_action=action,
                    previous_reward=reward,
                    episode_start=start,
                )
                return self._context_step(current, observation)

            return jax.lax.scan(
                one,
                initial_state,
                (obs, actions, rewards, starts),
            )

        def sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            previous_rewards: Any,
            episode_starts: Any,
            gate_overrides: Any,
        ) -> tuple[PolicyState, ModelOutput]:
            """Run a legal-history sequence with explicit recurrent inputs."""

            obs = jnp.asarray(observations)
            actions = jnp.asarray(previous_actions, dtype=jnp.int32)
            rewards = jnp.asarray(previous_rewards, dtype=jnp.float32)
            starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
            gates = jnp.asarray(gate_overrides, dtype=jnp.float32)
            if not (
                obs.shape[0]
                == actions.shape[0]
                == rewards.shape[0]
                == starts.shape[0]
                == gates.shape[0]
            ):
                raise ValueError("Sequence time axes differ.")

            def one(
                current: PolicyState,
                values: tuple[Any, Any, Any, Any, Any],
            ) -> tuple[PolicyState, ModelOutput]:
                observation, action, reward, start, gate = values
                current = current._replace(
                    previous_action=action,
                    previous_reward=reward,
                    episode_start=start,
                )
                next_state, output = self.step(current, observation, gate)
                return next_state, output

            return jax.lax.scan(
                one,
                initial_state,
                (obs, actions, rewards, starts, gates),
            )

        def response_from_context_and_action(
            self,
            task_features: Any,
            belief_embedding: Any,
            actions: Any,
        ) -> ResponsePrediction:
            """Predict only the observed action branch used by the response loss."""

            prefix = task_features.shape[:-1]
            action = jnp.asarray(actions, dtype=jnp.int32)
            if action.shape != prefix:
                raise ValueError("Selected response actions do not match context axes.")
            (
                delta_mean,
                delta_log_std,
                reward_mean,
                reward_log_std,
                done_logit,
            ) = self.decoder(
                jax.lax.stop_gradient(task_features),
                belief_embedding,
                action,
            )
            return ResponsePrediction(
                observation_delta_mean=delta_mean.reshape(
                    prefix + self.observation_shape
                ),
                observation_delta_log_std=delta_log_std.reshape(
                    prefix + self.observation_shape
                ),
                reward_mean=reward_mean,
                reward_log_std=reward_log_std,
                done_logit=done_logit,
            )

        def _belief_embedding_from_latent(
            self,
            task_features: Any,
            latent: Any,
        ) -> Any:
            logits, means, log_variances = degenerate_gaussian_mixture(
                latent, mixture_components=self.mixture_components
            )
            support = jnp.ones(task_features.shape[:-1], dtype=jnp.float32)
            return self.belief_set(logits, means, log_variances, support)

        def action_values_from_features_and_latent(
            self,
            task_features: Any,
            latent: Any,
        ) -> Any:
            """Evaluate the frozen full-information critic and nothing else."""

            belief_embedding = self._belief_embedding_from_latent(
                task_features, latent
            )
            unused_value, action_values = self.critic(
                task_features, belief_embedding
            )
            del unused_value
            return action_values

        def from_features_and_latent(
            self,
            task_features: Any,
            latent: Any,
            gate: Any = 1.0,
        ) -> TeacherOutput:
            belief_embedding = self._belief_embedding_from_latent(
                task_features, latent
            )
            actor_belief = jax.lax.stop_gradient(belief_embedding)
            unused_base, unused_residual, execution_logits = self.actor(
                task_features, actor_belief, gate
            )
            del unused_base, unused_residual
            unused_value, action_values = self.critic(
                task_features, belief_embedding
            )
            del unused_value
            return TeacherOutput(
                latent=latent,
                belief_embedding=belief_embedding,
                logits=execution_logits,
                action_values=action_values,
            )

        def latent_from_code(
            self,
            partner_code: Any,
            task_features: Any,
        ) -> Any:
            """Return the training-only continuous teacher context only."""

            return self.code_teacher(partner_code, task_features)

        def teacher_from_code(
            self,
            partner_code: Any,
            task_features: Any,
            gate: Any = 1.0,
        ) -> TeacherOutput:
            latent = self.latent_from_code(partner_code, task_features)
            return self.from_features_and_latent(task_features, latent, gate)

        def full_trajectory_latents(
            self,
            observations: Any,
            response_next_observations: Any,
            actions: Any,
            rewards: Any,
            dones: Any,
        ) -> Any:
            return self.full_teacher(
                observations,
                response_next_observations,
                actions,
                rewards,
                dones,
            )

        def initialize_all(
            self,
            state: PolicyState,
            observation: Any,
            partner_code: Any,
        ) -> tuple[Any, ...]:
            """Initialize every trainable subtree in one consistent Flax call."""

            batch_shape = state.previous_action.shape
            next_state, output = self.step(
                state,
                observation,
                jnp.ones(batch_shape, dtype=jnp.float32),
            )
            teacher = self.teacher_from_code(
                partner_code, output.task_features, 1.0
            )
            response = self.response_from_context_and_action(
                output.task_features,
                output.belief_embedding,
                jnp.zeros(batch_shape, dtype=jnp.int32),
            )
            observations = jnp.stack((observation, observation), axis=0)
            response_next = observation[None, ...]
            actions = jnp.zeros((1,) + batch_shape, dtype=jnp.int32)
            rewards = jnp.zeros((1,) + batch_shape, dtype=jnp.float32)
            dones = jnp.zeros((1,) + batch_shape, dtype=jnp.bool_)
            full_latent = self.full_trajectory_latents(
                observations, response_next, actions, rewards, dones
            )
            return next_state, output, response, teacher, full_latent

        def __call__(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            previous_rewards: Any,
            episode_starts: Any,
            gate_overrides: Any,
        ) -> tuple[PolicyState, ModelOutput]:
            return self.sequence(
                initial_state,
                observations,
                previous_actions,
                previous_rewards,
                episode_starts,
                gate_overrides,
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
    mixture_components: int,
    belief_embedding_dim: int,
    actor_hidden_dim: int,
    critic_hidden_dim: int,
    response_hidden_dim: int,
    modulation_rank: int,
    action_embedding_dim: int,
    log_variance_minimum: float,
    log_variance_maximum: float,
    response_log_std_minimum: float,
    response_log_std_maximum: float,
) -> Any:
    return _model_class()(
        observation_shape=tuple(int(value) for value in observation_shape),
        action_count=int(action_count),
        task_hidden_dim=int(task_hidden_dim),
        belief_hidden_dim=int(belief_hidden_dim),
        latent_dim=int(latent_dim),
        mixture_components=int(mixture_components),
        belief_embedding_dim=int(belief_embedding_dim),
        actor_hidden_dim=int(actor_hidden_dim),
        critic_hidden_dim=int(critic_hidden_dim),
        response_hidden_dim=int(response_hidden_dim),
        modulation_rank=int(modulation_rank),
        action_embedding_dim=int(action_embedding_dim),
        log_variance_minimum=float(log_variance_minimum),
        log_variance_maximum=float(log_variance_maximum),
        response_log_std_minimum=float(response_log_std_minimum),
        response_log_std_maximum=float(response_log_std_maximum),
    )


def initial_policy_state(
    *,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    belief_hidden_dim: int,
    latent_dim: int,
    mixture_components: int,
) -> PolicyState:
    import jax.numpy as jnp

    from .belief_encoder import initial_belief_carry
    from .task_encoder import initial_task_carry

    count = int(batch_size)
    mixture_logits = jnp.zeros((count, mixture_components), dtype=jnp.float32)
    means = jnp.zeros((count, mixture_components, latent_dim), dtype=jnp.float32)
    log_variances = jnp.zeros_like(means)
    return PolicyState(
        task_carry=initial_task_carry(count, task_hidden_dim),
        belief=GaussianMixtureBelief(
            recurrent_carry=initial_belief_carry(count, belief_hidden_dim),
            mixture_logits=mixture_logits,
            means=means,
            log_variances=log_variances,
            support_score=jnp.zeros((count,), dtype=jnp.float32),
        ),
        previous_observation=jnp.zeros(
            (count,) + tuple(observation_shape), dtype=jnp.float32
        ),
        previous_action=jnp.full((count,), int(action_count), dtype=jnp.int32),
        previous_reward=jnp.zeros((count,), dtype=jnp.float32),
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
