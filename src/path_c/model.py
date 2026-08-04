"""Executable B0--B2 model family for DEPI (METHOD_SPEC §1/§8).

One task encoder (x_t), one capability encoder (u), and one exact protocol
filter (pi_t over K=4 exchangeable components) share the actor/critic interface
``(task_features, context=concat(u, c_t))``.  The observation difference is
computed inside the capability/protocol pathway; the task pathway only ever
sees the current observation (structural input-layer isolation, §1.1/§1.2).

Context dropout (§1.2 new definition): with probability p the pair (u, c_t)
is replaced by the *prior context* (uniform-posterior mixture embedding c_0,
zero capability u=0).  The task pathway never participates in dropout.
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import ContextOutput, ModelOutput, PolicyState

_MODEL_CLASS: Any | None = None


def _model_class() -> Any:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    from .protocol_mixture import mixture_summary, prior_mixture_summary
    from .protocol_encoder import (
        capability_encoder_classes,
        exact_bayes_filter_step,
        posterior_entropy,
    )
    from .response_decoder import response_decoder_class
    from .response_targets import (
        ResponsePrediction,
        component_joint_log_probability,
        extract_partner_response_targets,
        official_partner_observation_planes,
    )
    from .task_encoder import task_encoder_classes
    from .universal_actor import universal_actor_class
    from .universal_critic import universal_critic_class

    TaskCell, _ = task_encoder_classes()
    CapabilityCell = capability_encoder_classes()
    Actor = universal_actor_class()
    Critic = universal_critic_class()
    Decoder = response_decoder_class()

    class DEPIModel(nn.Module):
        method_variant: str
        observation_shape: tuple[int, ...]
        action_count: int
        task_hidden_dim: int
        capability_hidden_dim: int
        capability_dim: int
        protocol_components: int
        component_embedding_dim: int
        actor_hidden_dim: int
        critic_hidden_dim: int
        response_hidden_dim: int
        modulation_rank: int
        action_embedding_dim: int

        def setup(self) -> None:
            self.task_cell = TaskCell(
                hidden_dim=self.task_hidden_dim,
                mask_partner_history=self.method_variant != "b0",
                name="task_encoder",
            )
            self.capability_cell = CapabilityCell(
                hidden_dim=self.capability_hidden_dim,
                output_dim=self.capability_dim,
                action_count=self.action_count,
                action_embedding_dim=self.action_embedding_dim,
                name="capability_encoder",
            )
            # Shared K x D component embeddings m_k (METHOD_SPEC §1.1); the
            # posterior-mean context c_t and the response mixture both read it.
            self.component_embeddings = nn.Embed(
                num_embeddings=self.protocol_components,
                features=self.component_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="protocol_component_embeddings",
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

        def _component_matrix(self) -> Any:
            return self.component_embeddings(
                jnp.arange(self.protocol_components, dtype=jnp.int32)
            )

        def _context_step(
            self, state: PolicyState, observation: Any
        ) -> tuple[PolicyState, ContextOutput]:
            observation = jnp.asarray(observation)
            start = jnp.asarray(state.episode_start, dtype=jnp.bool_)
            next_task_carry, task_features = self.task_cell(
                state.task_carry, (observation, start)
            )
            evidence = (
                state.previous_observation,
                observation,
                state.previous_action,
                start,
            )
            next_capability_carry, capability = self.capability_cell(
                state.capability_carry, evidence
            )
            # ``protocol_carry`` is exactly the previous categorical
            # posterior.  There is no recurrent recognition-network carry in
            # the deployable policy: the shared response likelihood supplies
            # only the newly observed emission evidence and the registered
            # sticky transition is applied exactly once.
            previous_probs = jnp.asarray(
                state.protocol_carry, dtype=jnp.float32
            )
            # Episode starts restart the filter from the uniform prior (§2.1).
            previous_probs = jnp.where(
                start[..., None],
                jnp.full_like(previous_probs, 1.0 / float(self.protocol_components)),
                previous_probs,
            )
            component_matrix = self._component_matrix()
            previous_capability = state.context_summary[..., : self.capability_dim]
            emission_values = self.decoder(
                jax.lax.stop_gradient(state.previous_observation),
                jax.lax.stop_gradient(state.task_carry),
                component_matrix,
                previous_capability,
                jnp.where(start, 0, state.previous_action),
            )
            emission_prediction = ResponsePrediction(
                posterior_log_probabilities=jnp.log(
                    jnp.maximum(previous_probs, 1.0e-12)
                ),
                visibility_logit=emission_values[0],
                relative_position_logits=emission_values[1],
                direction_logits=emission_values[2],
                inventory_logits=emission_values[3],
                interaction_change_logit=emission_values[4],
            )
            planes = official_partner_observation_planes(observation.shape[-1])
            observed_response = extract_partner_response_targets(
                state.previous_observation, observation, planes=planes
            )
            emission_log_likelihood = component_joint_log_probability(
                emission_prediction, observed_response
            )
            filtered = exact_bayes_filter_step(
                previous_probs, emission_log_likelihood
            )
            protocol_probabilities = jnp.where(
                start[..., None],
                jnp.full_like(filtered, 1.0 / float(self.protocol_components)),
                filtered,
            )
            protocol_embedding = mixture_summary(
                protocol_probabilities, component_matrix
            )
            context = ContextOutput(
                task_features, capability, protocol_probabilities, protocol_embedding
            )
            summary = jnp.concatenate((capability, protocol_embedding), axis=-1)
            next_state = PolicyState(
                task_carry=next_task_carry,
                capability_carry=next_capability_carry,
                protocol_carry=protocol_probabilities,
                context_summary=summary,
                previous_observation=observation.astype(
                    state.previous_observation.dtype
                ),
                previous_action=state.previous_action,
                episode_start=state.episode_start,
            )
            return next_state, context

        def _behavior_context(
            self, *, context: ContextOutput, context_dropout_mask: Any
        ) -> tuple[Any, Any]:
            """Apply the prior-context replacement of §1.2 context dropout."""

            component_matrix = self._component_matrix()
            prior_c = prior_mixture_summary(
                context.capability,
                component_matrix,
                component_count=self.protocol_components,
            )
            drop = jnp.asarray(context_dropout_mask, dtype=jnp.bool_)
            if drop.shape != context.task_features.shape[:-1]:
                drop = jnp.broadcast_to(drop, context.task_features.shape[:-1])
            behavior_capability = jnp.where(
                drop[..., None], jnp.zeros_like(context.capability), context.capability
            )
            behavior_embedding = jnp.where(
                drop[..., None], prior_c, context.protocol_embedding
            )
            return behavior_capability, behavior_embedding

        def _outputs_from_context(
            self, *, context: ContextOutput, context_dropout_mask: Any
        ) -> ModelOutput:
            behavior_capability, behavior_embedding = self._behavior_context(
                context=context, context_dropout_mask=context_dropout_mask
            )
            summary = jnp.concatenate((behavior_capability, behavior_embedding), axis=-1)
            # B0 is the capacity-matched full-observation recurrent PPO
            # baseline.  Its actor/critic cannot consume explicit protocol
            # state even though inert matching parameters remain present.
            if self.method_variant == "b0":
                summary = jnp.zeros_like(summary)
            logits = self.actor(context.task_features, summary)
            state_value, raw_q1, raw_q2 = self.critic(context.task_features, summary)
            return ModelOutput(
                task_features=context.task_features,
                capability=context.capability,
                protocol_probabilities=context.protocol_probabilities,
                protocol_embedding=context.protocol_embedding,
                context_summary=summary,
                posterior_entropy=posterior_entropy(context.protocol_probabilities),
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
                next_state, context = self._context_step(current, observation)
                output = self._outputs_from_context(
                    context=context, context_dropout_mask=drop
                )
                return next_state, output

            return jax.lax.scan(one, initial_state, (obs, actions, starts, drops))

        def response_from_context_and_action(
            self,
            context: ContextOutput,
            frame_features: Any,
            actions: Any,
        ) -> ResponsePrediction:
            """Per-component response logits for the mixture likelihood (§2.2).

            ``frame_features`` must be ``sg[frame_t]``; the caller owns the
            stop-gradient decision (kinematic components may read it, the
            event head never does).
            """

            prefix = context.task_features.shape[:-1]
            action = jnp.asarray(actions, dtype=jnp.int32)
            if action.shape != prefix:
                raise ValueError("Selected response actions do not match context axes.")
            component_matrix = self._component_matrix()
            values = self.decoder(
                jax.lax.stop_gradient(frame_features),
                jax.lax.stop_gradient(context.task_features),
                component_matrix,
                context.capability,
                action,
            )
            return ResponsePrediction(
                posterior_log_probabilities=jnp.log(
                    jnp.maximum(context.protocol_probabilities, 1.0e-12)
                ),
                visibility_logit=values[0],
                relative_position_logits=values[1],
                direction_logits=values[2],
                inventory_logits=values[3],
                interaction_change_logit=values[4],
            )

        def response_sequence(
            self,
            initial_state: PolicyState,
            observations: Any,
            previous_actions: Any,
            episode_starts: Any,
            executed_actions: Any,
        ) -> tuple[PolicyState, ResponsePrediction]:
            """Predict y_{t+1} from (pi_t, u_t, sg[frame_t], a_t^ego) (§2.3)."""

            final_state, context = self.context_sequence(
                initial_state, observations, previous_actions, episode_starts
            )
            sliced = ContextOutput(
                context.task_features[:-1],
                context.capability[:-1],
                context.protocol_probabilities[:-1],
                context.protocol_embedding[:-1],
            )
            prediction = self.response_from_context_and_action(
                sliced, jnp.asarray(observations)[:-1], executed_actions
            )
            return final_state, prediction

        def policy_logits_from_features_and_context(
            self, task_features: Any, context: Any
        ) -> Any:
            return self.actor(task_features, context)

        def twin_action_values_from_features_and_context(
            self, task_features: Any, context: Any
        ) -> tuple[Any, Any]:
            _, raw_q1, raw_q2 = self.critic(task_features, context)
            return raw_q1, raw_q2

        def action_values_from_features_and_context(
            self, task_features: Any, context: Any
        ) -> Any:
            raw_q1, raw_q2 = self.twin_action_values_from_features_and_context(
                task_features, context
            )
            return jnp.minimum(raw_q1, raw_q2)

        def initialize_all(
            self, state: PolicyState, observation: Any
        ) -> tuple[Any, ...]:
            next_state, context = self._context_step(state, observation)
            output = self._outputs_from_context(
                context=context,
                context_dropout_mask=jnp.zeros(
                    state.previous_action.shape, dtype=jnp.bool_
                ),
            )
            response = self.response_from_context_and_action(
                context,
                observation,
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

    _MODEL_CLASS = DEPIModel
    return DEPIModel


def build_model(
    *,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    capability_hidden_dim: int,
    capability_dim: int,
    protocol_components: int,
    component_embedding_dim: int,
    actor_hidden_dim: int,
    critic_hidden_dim: int,
    response_hidden_dim: int,
    modulation_rank: int,
    action_embedding_dim: int,
    method_variant: str = "b2",
) -> Any:
    kwargs = {
        "observation_shape": observation_shape,
        "action_count": action_count,
        "task_hidden_dim": task_hidden_dim,
        "capability_hidden_dim": capability_hidden_dim,
        "capability_dim": capability_dim,
        "protocol_components": protocol_components,
        "component_embedding_dim": component_embedding_dim,
        "actor_hidden_dim": actor_hidden_dim,
        "critic_hidden_dim": critic_hidden_dim,
        "response_hidden_dim": response_hidden_dim,
        "modulation_rank": modulation_rank,
        "action_embedding_dim": action_embedding_dim,
    }
    variant = str(method_variant).lower()
    if variant == "b0":
        return build_b0_full_history_ppo(**kwargs)
    if variant == "b1":
        return build_b1_protocol_architecture(**kwargs)
    if variant == "b2":
        return build_b2_decision_supervision(**kwargs)
    if variant == "b3":
        return build_b3_active_voi()
    raise ValueError("method_variant must be one of b0, b1, b2, b3.")


def _build_variant(*, method_variant: str, **kwargs: Any) -> Any:
    return _model_class()(
        method_variant=method_variant,
        observation_shape=tuple(int(value) for value in kwargs["observation_shape"]),
        action_count=int(kwargs["action_count"]),
        task_hidden_dim=int(kwargs["task_hidden_dim"]),
        capability_hidden_dim=int(kwargs["capability_hidden_dim"]),
        capability_dim=int(kwargs["capability_dim"]),
        protocol_components=int(kwargs["protocol_components"]),
        component_embedding_dim=int(kwargs["component_embedding_dim"]),
        actor_hidden_dim=int(kwargs["actor_hidden_dim"]),
        critic_hidden_dim=int(kwargs["critic_hidden_dim"]),
        response_hidden_dim=int(kwargs["response_hidden_dim"]),
        modulation_rank=int(kwargs["modulation_rank"]),
        action_embedding_dim=int(kwargs["action_embedding_dim"]),
    )


def build_b0_full_history_ppo(**kwargs: Any) -> Any:
    """Capacity-matched full-observation recurrent PPO baseline."""

    return _build_variant(method_variant="b0", **kwargs)


def build_b1_protocol_architecture(**kwargs: Any) -> Any:
    """Structurally isolated task/protocol architecture trained by PPO."""

    return _build_variant(method_variant="b1", **kwargs)


def build_b2_decision_supervision(**kwargs: Any) -> Any:
    """B1 architecture with legal all-action decision supervision."""

    return _build_variant(method_variant="b2", **kwargs)


def build_b3_active_voi(**unused_kwargs: Any) -> Any:
    """Fail closed until action-conditioned value of information exists."""

    del unused_kwargs
    raise NotImplementedError(
        "B3 action-conditioned VOI is not implemented and cannot produce a model."
    )


def initial_policy_state(
    *,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_count: int,
    task_hidden_dim: int,
    capability_hidden_dim: int,
    capability_dim: int,
    component_embedding_dim: int,
    protocol_components: int = 4,
) -> PolicyState:
    import jax.numpy as jnp

    from .protocol_encoder import initial_capability_carry
    from .task_encoder import initial_task_carry

    count = int(batch_size)
    # Prior context summary: u=0 and c_0 = mean_k m_k at zero-initialized
    # component embeddings (the true c_0 is computed inside the module).
    prior_summary = jnp.zeros((count, int(capability_dim) + int(component_embedding_dim)), dtype=jnp.float32)
    return PolicyState(
        task_carry=initial_task_carry(count, task_hidden_dim),
        capability_carry=initial_capability_carry(
            count, capability_hidden_dim, capability_dim
        ),
        protocol_carry=jnp.full(
            (count, int(protocol_components)),
            1.0 / float(protocol_components),
            dtype=jnp.float32,
        ),
        context_summary=prior_summary,
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
) -> Mapping[str, Any]:
    variables = model.init(
        key,
        example_state,
        example_observation,
        method=model.initialize_all,
    )
    return variables["params"]


__all__ = [
    "ModelOutput",
    "build_b0_full_history_ppo",
    "build_b1_protocol_architecture",
    "build_b2_decision_supervision",
    "build_b3_active_voi",
    "build_model",
    "initial_policy_state",
    "initialize_model_parameters",
]
