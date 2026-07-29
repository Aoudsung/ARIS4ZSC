"""Device-side environment loop for Path C V4.4."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .method import (
    PolicyState,
    deployment_belief_after_response,
    normalized_log_belief,
    regularized_policy,
    slot_bayes_update,
    uniform_slot_log_belief,
)
from .model import initial_control_carry, model_forward
from .training import TransitionBatch


class RunnerFunctions(NamedTuple):
    online_step: Callable[..., tuple[Any, Any, Any, Any]]
    reference_step: Callable[..., tuple[Any, Any, Any]]
    heads_apply: Callable[..., tuple[Any, Mapping[str, Any]]]
    encode_response: Callable[..., tuple[Any, Any, Any]]
    partner_step: Callable[..., Any]
    partner_observe: Callable[..., Any]


class DecisionRecord(NamedTuple):
    reference_logits: Any
    execution_logits: Any
    behavior_logits: Any
    exploration_logits: Any
    behavior_action_probability: Any
    behavior_slot: Any
    behavior_estimator: Any
    mask_execution_logits: Any
    j_use: Any
    j_mask: Any
    per_action_response_value: Any
    per_action_net_value: Any
    per_action_policy_mediated_gain: Any
    per_action_predicted_gain_lower_score: Any
    per_action_policy_gain_uncertainty: Any
    per_action_expected_next_policy_tv: Any
    information_gain: Any
    kl_divergence: Any
    action: Any
    reference_greedy_action: Any
    value_class_count: Any
    belief_entropy: Any
    response_code: Any
    predicted_response_effect: Any
    predicted_policy_cost: Any
    predicted_net_effect: Any
    predicted_regularized_net_effect: Any
    predicted_policy_total_variation: Any
    predicted_policy_mediated_effect: Any
    predicted_policy_gain_lower_score: Any
    predicted_policy_gain_uncertainty: Any
    predicted_next_policy_total_variation: Any
    executed_action_response_value: Any
    executed_action_net_value: Any
    executed_action_policy_mediated_gain: Any
    executed_action_predicted_gain_lower_score: Any
    executed_action_policy_gain_uncertainty: Any
    executed_action_expected_next_policy_tv: Any
    maximum_action_net_value: Any
    maximum_action_policy_mediated_gain: Any
    maximum_action_predicted_gain_lower_score: Any


class RunnerState(NamedTuple):
    environment_state: Any
    observations: Any
    ego_policy: PolicyState
    partner_state: Any
    partner_member: Any
    ego_seat: Any
    behavior_slot: Any
    behavior_estimator: Any
    episode_step: Any
    episode_id: Any
    episode_return: Any
    episode_correct_deliveries: Any
    episode_wrong_deliveries: Any
    episode_indicator_activations: Any
    completed_episodes: Any
    effective_environment_steps: Any
    random_key: Any


class PartnerState(NamedTuple):
    dynamic_policy: PolicyState
    static_carry: Any


class PartnerStepContext(NamedTuple):
    dynamic_lane: Any
    dynamic_output: Any
    dynamic_action: Any


def tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def choose(selected_leaf: Any, alternative_leaf: Any) -> Any:
        values = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = values.reshape(
            values.shape
            + (1,) * (jnp.ndim(selected_leaf) - values.ndim)
        )
        return jnp.where(expanded, selected_leaf, alternative_leaf)

    return jax.tree_util.tree_map(choose, selected, alternative)


def seat_observations(observations: Any, seats: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(observations)
    if values.ndim < 3 or values.shape[1] != 2:
        raise ValueError("Vector observations require [environment, seat, ...].")
    selector = jnp.asarray(seats, dtype=jnp.bool_).reshape(
        (values.shape[0],) + (1,) * (values.ndim - 2)
    )
    return (
        jnp.where(selector, values[:, 1], values[:, 0]),
        jnp.where(selector, values[:, 0], values[:, 1]),
    )


def initialize_policy_state(
    *,
    official_initial_carry: Callable[[int], Any],
    batch_size: int,
    slot_count: int,
    action_count: int,
    hidden_dim: int,
    initial_temperature: float,
) -> PolicyState:
    import jax.numpy as jnp

    return PolicyState(
        reference_carry=official_initial_carry(batch_size),
        trainable_carry=official_initial_carry(batch_size),
        control_carry=initial_control_carry(batch_size, hidden_dim),
        slot_log_belief=uniform_slot_log_belief((batch_size,), slot_count),
        previous_action=jnp.full((batch_size,), action_count, dtype=jnp.int32),
        previous_team_reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        episode_start=jnp.ones((batch_size,), dtype=jnp.bool_),
        log_temperature=jnp.full(
            (batch_size,), jnp.log(initial_temperature), dtype=jnp.float32
        ),
        generic_log_temperature=jnp.full(
            (batch_size,), jnp.log(initial_temperature), dtype=jnp.float32
        ),
    )


def initialize_runner(
    *,
    environment: Any,
    official_initial_carry: Callable[[int], Any],
    partner_initial_state: Callable[[int], Any],
    random_key: Any,
    slot_count: int,
    action_count: int,
    hidden_dim: int,
    partner_member_count: int,
    initial_temperature: float,
) -> RunnerState:
    import jax
    import jax.numpy as jnp

    if partner_member_count <= 0:
        raise ValueError("Training requires at least one frozen partner.")
    (
        next_key,
        environment_key,
        partner_key,
        seat_key,
        behavior_slot_key,
        behavior_estimator_key,
    ) = jax.random.split(random_key, 6)
    environment_state, observations = environment.reset(environment_key)
    count = int(environment.num_envs)
    return RunnerState(
        environment_state=environment_state,
        observations=observations,
        ego_policy=initialize_policy_state(
            official_initial_carry=official_initial_carry,
            batch_size=count,
            slot_count=slot_count,
            action_count=action_count,
            hidden_dim=hidden_dim,
            initial_temperature=initial_temperature,
        ),
        partner_state=partner_initial_state(count),
        partner_member=jax.random.randint(
            partner_key, (count,), 0, partner_member_count + 1
        ),
        ego_seat=jax.random.bernoulli(seat_key, shape=(count,)).astype(jnp.int32),
        behavior_slot=jax.random.randint(
            behavior_slot_key, (count,), 0, slot_count
        ),
        behavior_estimator=jax.random.randint(
            behavior_estimator_key, (count,), 0, 2
        ),
        episode_step=jnp.zeros((count,), dtype=jnp.int32),
        episode_id=jnp.arange(count, dtype=jnp.int64),
        episode_return=jnp.zeros((count,), dtype=jnp.float32),
        episode_correct_deliveries=jnp.zeros((count,), dtype=jnp.int32),
        episode_wrong_deliveries=jnp.zeros((count,), dtype=jnp.int32),
        episode_indicator_activations=jnp.zeros((count,), dtype=jnp.int32),
        completed_episodes=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        random_key=next_key,
    )


def policy_action(
    *,
    functions: RunnerFunctions,
    params: Mapping[str, Any],
    policy_state: PolicyState,
    observations: Any,
    key: Any,
    deployment_mode: str,
    gamma: float,
    uncertainty_penalty: float,
    behavior_slot: Any | None = None,
    behavior_estimator: Any | None = None,
    behavior_exploration_mix: float = 0.0,
    behavior_exploration_temperature: float = 1.0,
    behavior_uniform_floor: float = 0.0,
) -> tuple[PolicyState, Any, Any, DecisionRecord, Any]:
    import jax
    import jax.numpy as jnp

    belief = policy_state.slot_log_belief
    if deployment_mode == "prior_only":
        belief = uniform_slot_log_belief(
            belief.shape[:-1], int(belief.shape[-1])
        )
    belief = normalized_log_belief(belief)
    next_reference_carry, reference_logits, unused_reference_value = (
        functions.reference_step(
            policy_state.reference_carry,
            observations,
            policy_state.episode_start,
        )
    )
    del unused_reference_value
    (
        next_trainable_carry,
        features,
        unused_online_logits,
        unused_online_value,
    ) = functions.online_step(
        params["official"],
        policy_state.trainable_carry,
        observations,
        policy_state.episode_start,
    )
    del unused_online_logits, unused_online_value
    next_control_carry, raw = functions.heads_apply(
        params["heads"],
        policy_state.control_carry,
        features,
        policy_state.previous_action,
        policy_state.previous_team_reward,
        policy_state.episode_start,
        belief,
    )
    output = model_forward(
        raw_output=raw,
        reference_logits=reference_logits,
        slot_log_belief=belief,
        temperature=jnp.exp(policy_state.log_temperature),
        generic_temperature=jnp.exp(policy_state.generic_log_temperature),
        deployment_mode=deployment_mode,
        gamma=gamma,
        uncertainty_penalty=uncertainty_penalty,
    )
    generic = regularized_policy(
        reference_logits,
        output.information_gain,
        jnp.exp(policy_state.generic_log_temperature),
    )
    exploration_mix = float(behavior_exploration_mix)
    uniform_floor = float(behavior_uniform_floor)
    if not 0.0 <= exploration_mix < 1.0:
        raise ValueError("Behavior exploration mix must lie in [0, 1).")
    if not 0.0 <= uniform_floor < 1.0:
        raise ValueError("Behavior uniform floor must lie in [0, 1).")
    if exploration_mix + uniform_floor >= 1.0:
        raise ValueError("Behavior mixtures must leave execution-policy mass.")
    if behavior_exploration_temperature <= 0.0:
        raise ValueError("Behavior exploration temperature must be positive.")

    execution_log = jax.nn.log_softmax(output.execution_logits, axis=-1)
    execution_probability = jnp.exp(execution_log)
    action_count = int(output.execution_logits.shape[-1])
    batch_count = int(output.q_values.shape[0])
    if behavior_slot is None:
        selected_slot = jnp.zeros((batch_count,), dtype=jnp.int32)
    else:
        selected_slot = jnp.asarray(behavior_slot, dtype=jnp.int32)
    if behavior_estimator is None:
        selected_estimator = jnp.zeros((batch_count,), dtype=jnp.int32)
    else:
        selected_estimator = jnp.asarray(behavior_estimator, dtype=jnp.int32)
    if selected_slot.shape != (batch_count,) or selected_estimator.shape != (batch_count,):
        raise ValueError("Behavior expert indexes must align with the policy batch.")
    batch_indexes = jnp.arange(batch_count, dtype=jnp.int32)
    sampled_q = output.q_values[
        batch_indexes, selected_estimator, selected_slot, :
    ]
    exploration_policy = regularized_policy(
        reference_logits,
        sampled_q,
        jnp.asarray(behavior_exploration_temperature, dtype=jnp.float32),
    )
    exploration_probability = exploration_policy.probabilities
    behavior_probability = (
        (1.0 - exploration_mix - uniform_floor) * execution_probability
        + exploration_mix * exploration_probability
        + uniform_floor / action_count
    )
    behavior_logits = jnp.log(jnp.maximum(behavior_probability, 1.0e-12))
    keys = jnp.asarray(key)
    if keys.ndim == 2:
        action = jax.vmap(
            lambda lane_key, lane_logits: jax.random.categorical(
                lane_key, lane_logits
            )
        )(keys, behavior_logits)
    else:
        action = jax.random.categorical(keys, behavior_logits)

    reference_log = jax.nn.log_softmax(reference_logits, axis=-1)
    kl = jnp.sum(
        execution_probability * (execution_log - reference_log), axis=-1
    )
    probability = jnp.exp(belief)
    entropy = -jnp.sum(probability * belief, axis=-1)
    executed_response = jnp.take_along_axis(
        output.per_action_response_value, action[..., None], axis=-1
    )[..., 0]
    executed_net = jnp.take_along_axis(
        output.per_action_net_value, action[..., None], axis=-1
    )[..., 0]
    selected_behavior_probability = jnp.take_along_axis(
        behavior_probability, action[..., None], axis=-1
    )[..., 0]
    executed_policy_gain = jnp.take_along_axis(
        output.per_action_policy_mediated_gain,
        action[..., None],
        axis=-1,
    )[..., 0]
    executed_gain_lower_score = jnp.take_along_axis(
        output.per_action_predicted_gain_lower_score,
        action[..., None],
        axis=-1,
    )[..., 0]
    executed_policy_gain_uncertainty = jnp.take_along_axis(
        output.per_action_policy_gain_uncertainty,
        action[..., None],
        axis=-1,
    )[..., 0]
    executed_next_policy_tv = jnp.take_along_axis(
        output.per_action_expected_next_policy_tv,
        action[..., None],
        axis=-1,
    )[..., 0]
    next_state = policy_state._replace(
        reference_carry=next_reference_carry,
        trainable_carry=next_trainable_carry,
        control_carry=next_control_carry,
        slot_log_belief=belief,
    )
    record = DecisionRecord(
        reference_logits=reference_logits,
        execution_logits=output.execution_logits,
        behavior_logits=behavior_logits,
        exploration_logits=exploration_policy.logits,
        behavior_action_probability=selected_behavior_probability,
        behavior_slot=selected_slot,
        behavior_estimator=selected_estimator,
        mask_execution_logits=output.mask_execution_logits,
        j_use=output.j_use,
        j_mask=output.j_mask,
        per_action_response_value=output.per_action_response_value,
        per_action_net_value=output.per_action_net_value,
        per_action_policy_mediated_gain=(
            output.per_action_policy_mediated_gain
        ),
        per_action_predicted_gain_lower_score=(
            output.per_action_predicted_gain_lower_score
        ),
        per_action_policy_gain_uncertainty=(
            output.per_action_policy_gain_uncertainty
        ),
        per_action_expected_next_policy_tv=(
            output.per_action_expected_next_policy_tv
        ),
        information_gain=output.information_gain,
        kl_divergence=kl,
        action=action,
        reference_greedy_action=jnp.argmax(reference_logits, axis=-1),
        value_class_count=output.supported_value_class_count,
        belief_entropy=entropy,
        response_code=jnp.full(action.shape, -1, dtype=jnp.int32),
        predicted_response_effect=output.predicted_response_effect,
        predicted_policy_cost=output.predicted_policy_cost,
        predicted_net_effect=output.predicted_net_effect,
        predicted_regularized_net_effect=(
            output.predicted_regularized_net_effect
        ),
        predicted_policy_total_variation=(
            output.predicted_policy_total_variation
        ),
        predicted_policy_mediated_effect=(
            output.predicted_policy_mediated_effect
        ),
        predicted_policy_gain_lower_score=(
            output.predicted_policy_gain_lower_score
        ),
        predicted_policy_gain_uncertainty=(
            output.predicted_policy_gain_uncertainty
        ),
        predicted_next_policy_total_variation=(
            output.predicted_next_policy_total_variation
        ),
        executed_action_response_value=executed_response,
        executed_action_net_value=executed_net,
        executed_action_policy_mediated_gain=executed_policy_gain,
        executed_action_predicted_gain_lower_score=executed_gain_lower_score,
        executed_action_policy_gain_uncertainty=(
            executed_policy_gain_uncertainty
        ),
        executed_action_expected_next_policy_tv=executed_next_policy_tv,
        maximum_action_net_value=jnp.max(output.per_action_net_value, axis=-1),
        maximum_action_policy_mediated_gain=jnp.max(
            output.per_action_policy_mediated_gain, axis=-1
        ),
        maximum_action_predicted_gain_lower_score=jnp.max(
            output.per_action_predicted_gain_lower_score, axis=-1
        ),
    )
    return next_state, action, output, record, generic.logits


def collect_rollout(
    *,
    state: RunnerState,
    length: int,
    environment: Any,
    functions: RunnerFunctions,
    params: Mapping[str, Any],
    codebook_embeddings: Any,
    partner_member_count: int,
    deployment_mode: str,
    gamma: float,
    uncertainty_penalty: float,
    terminal_response: int,
    behavior_exploration_mix: float = 0.0,
    behavior_exploration_temperature: float = 1.0,
    behavior_uniform_floor: float = 0.0,
) -> tuple[RunnerState, TransitionBatch, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    if length != int(environment.episode_steps):
        raise ValueError("A training rollout must contain one complete episode.")
    count = int(environment.num_envs)
    initial_official_carry = state.ego_policy.trainable_carry
    initial_control_carry_value = state.ego_policy.control_carry

    def one_step(
        current: RunnerState, unused: Any
    ) -> tuple[RunnerState, tuple[Any, ...]]:
        del unused
        (
            next_root_key,
            action_root_key,
            environment_key,
            partner_key,
            next_partner_key,
            next_seat_key,
            next_behavior_slot_key,
            next_behavior_estimator_key,
        ) = jax.random.split(current.random_key, 8)
        action_keys = jax.random.split(action_root_key, count)
        ego_observation, partner_observation = seat_observations(
            current.observations, current.ego_seat
        )
        stepped_policy, ego_action, output, decision, generic_logits = (
            policy_action(
                functions=functions,
                params=params,
                policy_state=current.ego_policy,
                observations=ego_observation,
                key=action_keys,
                deployment_mode=deployment_mode,
                gamma=gamma,
                uncertainty_penalty=uncertainty_penalty,
                behavior_slot=current.behavior_slot,
                behavior_estimator=current.behavior_estimator,
                behavior_exploration_mix=behavior_exploration_mix,
                behavior_exploration_temperature=(
                    behavior_exploration_temperature
                ),
                behavior_uniform_floor=behavior_uniform_floor,
            )
        )
        partner_action, tentative_partner_state, partner_context = (
            functions.partner_step(
                params,
                current.partner_member,
                partner_observation,
                current.partner_state,
                current.ego_policy.episode_start,
                partner_key,
            )
        )
        joint_actions = jnp.stack(
            (
                jnp.where(current.ego_seat == 0, ego_action, partner_action),
                jnp.where(current.ego_seat == 1, ego_action, partner_action),
            ),
            axis=-1,
        )
        (
            next_environment_state,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step(
            current.environment_state, joint_actions, environment_key
        )
        dones = jnp.asarray(dones, dtype=jnp.bool_)
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        next_ego_observation, next_partner_observation = seat_observations(
            next_observations, current.ego_seat
        )
        terminal_ego, terminal_partner = seat_observations(
            info["terminal_observations"], current.ego_seat
        )
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_ego_observation.ndim - 1)
        )
        response_next_observation = jnp.where(
            observation_mask, terminal_ego, next_ego_observation
        )
        response_code, unused_response_logits, unused_signature = (
            functions.encode_response(
                params["heads"],
                ego_observation[None, ...],
                ego_action[None, ...],
                response_next_observation[None, ...],
                dones[None, ...],
                codebook_embeddings,
                terminal_response,
            )
        )
        del unused_response_logits, unused_signature
        response_code = response_code[0]
        decision = decision._replace(response_code=response_code)

        partner_response_next = jnp.where(
            dones.reshape(
                dones.shape + (1,) * (next_partner_observation.ndim - 1)
            ),
            terminal_partner,
            next_partner_observation,
        )
        next_partner_state = functions.partner_observe(
            params,
            current.partner_member,
            tentative_partner_state,
            partner_context,
            partner_observation,
            partner_response_next,
            partner_action,
            rewards,
            dones,
            codebook_embeddings,
        )

        updated_belief = slot_bayes_update(
            slot_log_belief=stepped_policy.slot_log_belief,
            slot_response_probabilities=output.response_probabilities,
            action=ego_action,
            response_code=response_code,
        )
        updated_belief = deployment_belief_after_response(
            mode=deployment_mode,
            current_log_belief=stepped_policy.slot_log_belief,
            updated_log_belief=updated_belief,
        )
        uniform = uniform_slot_log_belief((count,), int(updated_belief.shape[-1]))
        next_policy = stepped_policy._replace(
            slot_log_belief=jnp.where(dones[:, None], uniform, updated_belief),
            previous_action=jnp.where(
                dones,
                jnp.full((count,), output.q_values.shape[-1], dtype=jnp.int32),
                ego_action,
            ),
            previous_team_reward=jnp.where(dones, 0.0, rewards),
            episode_start=dones,
        )
        completed_return = current.episode_return + rewards
        completed_correct = (
            current.episode_correct_deliveries
            + jnp.asarray(info["correct_delivery"], dtype=jnp.int32)
        )
        completed_wrong = (
            current.episode_wrong_deliveries
            + jnp.asarray(info["wrong_delivery"], dtype=jnp.int32)
        )
        completed_indicator = (
            current.episode_indicator_activations
            + jnp.asarray(info["indicator_activation"], dtype=jnp.int32)
        )
        next_episode_id = current.episode_id + count
        sampled_partner = jax.random.randint(
            next_partner_key, (count,), 0, partner_member_count + 1
        )
        sampled_seat = jax.random.bernoulli(
            next_seat_key, shape=(count,)
        ).astype(jnp.int32)
        sampled_behavior_slot = jax.random.randint(
            next_behavior_slot_key, (count,), 0, int(updated_belief.shape[-1])
        )
        sampled_behavior_estimator = jax.random.randint(
            next_behavior_estimator_key, (count,), 0, 2
        )
        next_state = RunnerState(
            environment_state=next_environment_state,
            observations=next_observations,
            ego_policy=next_policy,
            partner_state=next_partner_state,
            partner_member=jnp.where(
                dones, sampled_partner, current.partner_member
            ),
            ego_seat=jnp.where(dones, sampled_seat, current.ego_seat),
            behavior_slot=jnp.where(
                dones, sampled_behavior_slot, current.behavior_slot
            ),
            behavior_estimator=jnp.where(
                dones, sampled_behavior_estimator, current.behavior_estimator
            ),
            episode_step=jnp.where(dones, 0, current.episode_step + 1),
            episode_id=jnp.where(dones, next_episode_id, current.episode_id),
            episode_return=jnp.where(dones, 0.0, completed_return),
            episode_correct_deliveries=jnp.where(
                dones, 0, completed_correct
            ),
            episode_wrong_deliveries=jnp.where(dones, 0, completed_wrong),
            episode_indicator_activations=jnp.where(
                dones, 0, completed_indicator
            ),
            completed_episodes=current.completed_episodes
            + jnp.sum(dones, dtype=jnp.int64),
            effective_environment_steps=current.effective_environment_steps
            + count,
            random_key=next_root_key,
        )
        return next_state, {
            "observations": ego_observation,
            "response_next_observations": response_next_observation,
            "episode_start": current.ego_policy.episode_start,
            "previous_actions": current.ego_policy.previous_action,
            "previous_team_rewards": current.ego_policy.previous_team_reward,
            "reference_logits": decision.reference_logits,
            "execution_logits": decision.execution_logits,
            "behavior_logits": decision.behavior_logits,
            "behavior_action_probabilities": (
                decision.behavior_action_probability
            ),
            "exploration_logits": decision.exploration_logits,
            "behavior_slots": decision.behavior_slot,
            "behavior_estimators": decision.behavior_estimator,
            "mask_execution_logits": decision.mask_execution_logits,
            "generic_execution_logits": generic_logits,
            "slot_log_beliefs": stepped_policy.slot_log_belief,
            "actions": ego_action,
            "rewards": rewards,
            "dones": dones,
            "response_codes": response_code,
            "episode_ids": current.episode_id,
            "episode_steps": current.episode_step,
            "partner_members": current.partner_member,
            "ego_seats": current.ego_seat,
            "completed_episode_returns": jnp.where(
                dones, completed_return, 0.0
            ),
            "completed_episode_mask": dones,
            "completed_correct_deliveries": jnp.where(
                dones, completed_correct, 0
            ),
            "completed_wrong_deliveries": jnp.where(
                dones, completed_wrong, 0
            ),
            "completed_indicator_activations": jnp.where(
                dones, completed_indicator, 0
            ),
            "value_class_counts": decision.value_class_count,
            "j_use": decision.j_use,
            "j_mask": decision.j_mask,
            "per_action_response_values": decision.per_action_response_value,
            "per_action_net_values": decision.per_action_net_value,
            "per_action_policy_mediated_gains": (
                decision.per_action_policy_mediated_gain
            ),
            "per_action_predicted_gain_lower_scores": (
                decision.per_action_predicted_gain_lower_score
            ),
            "per_action_policy_gain_uncertainties": (
                decision.per_action_policy_gain_uncertainty
            ),
            "per_action_expected_next_policy_tvs": (
                decision.per_action_expected_next_policy_tv
            ),
            "information_gains": decision.information_gain,
            "predicted_response_effects": decision.predicted_response_effect,
            "predicted_policy_costs": decision.predicted_policy_cost,
            "predicted_net_effects": decision.predicted_net_effect,
            "predicted_regularized_net_effects": (
                decision.predicted_regularized_net_effect
            ),
            "predicted_policy_total_variations": (
                decision.predicted_policy_total_variation
            ),
            "predicted_policy_mediated_effects": (
                decision.predicted_policy_mediated_effect
            ),
            "predicted_policy_gain_lower_scores": (
                decision.predicted_policy_gain_lower_score
            ),
            "predicted_policy_gain_uncertainties": (
                decision.predicted_policy_gain_uncertainty
            ),
            "predicted_next_policy_total_variations": (
                decision.predicted_next_policy_total_variation
            ),
            "kl_divergences": decision.kl_divergence,
            "reference_greedy_actions": decision.reference_greedy_action,
            "belief_entropies": decision.belief_entropy,
            "executed_action_response_values": (
                decision.executed_action_response_value
            ),
            "executed_action_net_values": decision.executed_action_net_value,
            "executed_action_policy_mediated_gains": (
                decision.executed_action_policy_mediated_gain
            ),
            "executed_action_predicted_gain_lower_scores": (
                decision.executed_action_predicted_gain_lower_score
            ),
            "executed_action_policy_gain_uncertainties": (
                decision.executed_action_policy_gain_uncertainty
            ),
            "executed_action_expected_next_policy_tvs": (
                decision.executed_action_expected_next_policy_tv
            ),
            "maximum_action_net_values": decision.maximum_action_net_value,
            "maximum_action_policy_mediated_gains": (
                decision.maximum_action_policy_mediated_gain
            ),
            "maximum_action_predicted_gain_lower_scores": (
                decision.maximum_action_predicted_gain_lower_score
            ),
        }

    final_state, recorded = jax.lax.scan(
        one_step, state, xs=None, length=length
    )
    final_ego_observation, unused_partner_observation = seat_observations(
        final_state.observations, final_state.ego_seat
    )
    del unused_partner_observation
    unused_carry, final_reference_logits, unused_value = functions.reference_step(
        final_state.ego_policy.reference_carry,
        final_ego_observation,
        final_state.ego_policy.episode_start,
    )
    del unused_carry, unused_value
    batch = TransitionBatch(
        observations=jnp.concatenate(
            (recorded["observations"], final_ego_observation[None, ...]), axis=0
        ),
        response_next_observations=recorded["response_next_observations"],
        episode_start=jnp.concatenate(
            (
                recorded["episode_start"],
                final_state.ego_policy.episode_start[None, ...],
            ),
            axis=0,
        ),
        previous_actions=jnp.concatenate(
            (
                recorded["previous_actions"],
                final_state.ego_policy.previous_action[None, ...],
            ),
            axis=0,
        ),
        previous_team_rewards=jnp.concatenate(
            (
                recorded["previous_team_rewards"],
                final_state.ego_policy.previous_team_reward[None, ...],
            ),
            axis=0,
        ),
        initial_official_carry=initial_official_carry,
        initial_control_carry=initial_control_carry_value,
        reference_logits=jnp.concatenate(
            (recorded["reference_logits"], final_reference_logits[None, ...]),
            axis=0,
        ),
        execution_logits=recorded["execution_logits"],
        generic_execution_logits=recorded["generic_execution_logits"],
        behavior_logits=recorded["behavior_logits"],
        posterior_scores=recorded["j_use"],
        generic_scores=recorded["information_gains"],
        slot_log_beliefs=jnp.concatenate(
            (
                recorded["slot_log_beliefs"],
                final_state.ego_policy.slot_log_belief[None, ...],
            ),
            axis=0,
        ),
        actions=recorded["actions"],
        rewards=recorded["rewards"],
        dones=recorded["dones"],
        response_codes=recorded["response_codes"],
    )
    records = {
        name: recorded[name]
        for name in (
            "mask_execution_logits",
            "episode_ids",
            "episode_steps",
            "partner_members",
            "ego_seats",
            "completed_episode_returns",
            "completed_episode_mask",
            "completed_correct_deliveries",
            "completed_wrong_deliveries",
            "completed_indicator_activations",
            "value_class_counts",
            "j_use",
            "j_mask",
            "per_action_response_values",
            "per_action_net_values",
            "per_action_policy_mediated_gains",
            "per_action_predicted_gain_lower_scores",
            "per_action_policy_gain_uncertainties",
            "per_action_expected_next_policy_tvs",
            "information_gains",
            "behavior_action_probabilities",
            "behavior_slots",
            "behavior_estimators",
            "exploration_logits",
            "predicted_response_effects",
            "predicted_policy_costs",
            "predicted_net_effects",
            "predicted_regularized_net_effects",
            "predicted_policy_total_variations",
            "predicted_policy_mediated_effects",
            "predicted_policy_gain_lower_scores",
            "predicted_policy_gain_uncertainties",
            "predicted_next_policy_total_variations",
            "kl_divergences",
            "reference_greedy_actions",
            "belief_entropies",
            "executed_action_response_values",
            "executed_action_net_values",
            "executed_action_policy_mediated_gains",
            "executed_action_predicted_gain_lower_scores",
            "executed_action_policy_gain_uncertainties",
            "executed_action_expected_next_policy_tvs",
            "maximum_action_net_values",
            "maximum_action_policy_mediated_gains",
            "maximum_action_predicted_gain_lower_scores",
        )
    }
    return final_state, batch, records


def partner_callbacks(
    *,
    base_functions: RunnerFunctions,
    official_initial_carry: Callable[[int], Any],
    static_partner_step: Callable[..., tuple[Any, Any]],
    slot_count: int,
    action_count: int,
    hidden_dim: int,
    initial_temperature: float,
    gamma: float,
    uncertainty_penalty: float,
    terminal_response: int,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Build frozen-current and frozen-official partner behavior.

    Member zero is a snapshot of the current policy at rollout start.  Higher
    member numbers select one frozen official policy.  Partner state is kept
    independently from the ego state for the full episode.
    """

    import jax
    import jax.numpy as jnp

    placeholder = RunnerFunctions(
        online_step=base_functions.online_step,
        reference_step=base_functions.reference_step,
        heads_apply=base_functions.heads_apply,
        encode_response=base_functions.encode_response,
        partner_step=lambda *unused: None,
        partner_observe=lambda *unused: None,
    )

    def initial(batch_size: int) -> PartnerState:
        return PartnerState(
            dynamic_policy=initialize_policy_state(
                official_initial_carry=official_initial_carry,
                batch_size=batch_size,
                slot_count=slot_count,
                action_count=action_count,
                hidden_dim=hidden_dim,
                initial_temperature=initial_temperature,
            ),
            static_carry=official_initial_carry(batch_size),
        )

    def step(
        params: Mapping[str, Any],
        member: Any,
        observations: Any,
        state: PartnerState,
        episode_start: Any,
        key: Any,
    ) -> tuple[Any, PartnerState, PartnerStepContext]:
        frozen_params = jax.lax.stop_gradient(params)
        dynamic_key, static_key = jax.random.split(key)
        dynamic_lane = jnp.asarray(member) == 0
        static_action, static_carry = static_partner_step(
            jnp.maximum(jnp.asarray(member) - 1, 0),
            observations,
            state.static_carry,
            episode_start,
            static_key,
        )
        (
            dynamic_policy,
            dynamic_action,
            dynamic_output,
            unused_record,
            unused_generic,
        ) = policy_action(
            functions=placeholder,
            params=frozen_params,
            policy_state=state.dynamic_policy,
            observations=observations,
            key=jax.random.split(dynamic_key, observations.shape[0]),
            deployment_mode="posterior_use",
            gamma=gamma,
            uncertainty_penalty=uncertainty_penalty,
            behavior_exploration_mix=0.0,
            behavior_uniform_floor=0.0,
        )
        del unused_record, unused_generic
        return (
            jnp.where(dynamic_lane, dynamic_action, static_action),
            PartnerState(
                dynamic_policy=tree_select(
                    dynamic_lane, dynamic_policy, state.dynamic_policy
                ),
                static_carry=tree_select(
                    dynamic_lane, state.static_carry, static_carry
                ),
            ),
            PartnerStepContext(
                dynamic_lane=dynamic_lane,
                dynamic_output=dynamic_output,
                dynamic_action=dynamic_action,
            ),
        )

    def observe(
        params: Mapping[str, Any],
        member: Any,
        state: PartnerState,
        context: PartnerStepContext,
        observations: Any,
        next_observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        codebook_embeddings: Any,
    ) -> PartnerState:
        del member, actions
        frozen_params = jax.lax.stop_gradient(params)
        response_codes, unused_logits, unused_signatures = (
            base_functions.encode_response(
                frozen_params["heads"],
                observations[None, ...],
                context.dynamic_action[None, ...],
                next_observations[None, ...],
                dones[None, ...],
                codebook_embeddings,
                terminal_response,
            )
        )
        del unused_logits, unused_signatures
        response_codes = response_codes[0]
        updated_belief = slot_bayes_update(
            slot_log_belief=state.dynamic_policy.slot_log_belief,
            slot_response_probabilities=(
                context.dynamic_output.response_probabilities
            ),
            action=context.dynamic_action,
            response_code=response_codes,
        )
        uniform = uniform_slot_log_belief(
            updated_belief.shape[:-1], slot_count
        )
        candidate = state.dynamic_policy._replace(
            slot_log_belief=jnp.where(
                dones[:, None], uniform, updated_belief
            ),
            previous_action=jnp.where(
                dones,
                jnp.full(dones.shape, action_count, dtype=jnp.int32),
                context.dynamic_action,
            ),
            previous_team_reward=jnp.where(dones, 0.0, rewards),
            episode_start=dones,
        )
        updated_dynamic = tree_select(
            context.dynamic_lane, candidate, state.dynamic_policy
        )
        return PartnerState(
            dynamic_policy=updated_dynamic,
            static_carry=state.static_carry,
        )

    return initial, step, observe


def compiled_collector(**static_arguments: Any) -> Callable[..., Any]:
    import functools
    import jax

    return jax.jit(functools.partial(collect_rollout, **static_arguments))


__all__ = [
    "DecisionRecord",
    "RunnerFunctions",
    "RunnerState",
    "collect_rollout",
    "compiled_collector",
    "initialize_policy_state",
    "initialize_runner",
    "partner_callbacks",
    "policy_action",
    "seat_observations",
]
