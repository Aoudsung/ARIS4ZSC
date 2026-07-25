"""Device-side recurrent rollout for all four Path C conditions.

The environment, partner, response, safety, and model operations are injected
callbacks.  Consequently this module has no OvercookedV2 dependency while the
whole time loop remains a JAX ``scan`` suitable for compilation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from ..belief.update import likelihood_for_token, log_bayes_update, uniform_log_belief
from ..probe.candidates import enumerate_candidates
from ..probe.controllers import select_action
from ..probe.scores import response_information_scores, sequential_scores


class RolloutState(NamedTuple):
    env_state: Any
    observations: Any
    model_carry: Any
    partner_carry: Any
    log_belief: Any
    partner_log_belief: Any
    partner_index: Any
    ego_seat: Any
    budget_remaining: Any
    partner_budget_remaining: Any
    episode_step: Any
    episode_id: Any
    episode_start: Any
    episode_return: Any
    episode_probe_count: Any
    correct_delivery_count: Any
    wrong_delivery_count: Any
    indicator_cost: Any
    completed_episodes: Any
    effective_environment_steps: Any
    random_key: Any


class RolloutBatch(NamedTuple):
    observations: Any
    next_observations: Any
    episode_start: Any
    model_carry: Any
    features: Any
    next_feature_targets: Any
    actor_logits: Any
    shared_values: Any
    next_shared_values: Any
    advantages: Any
    returns: Any
    prototype_values: Any
    next_prototype_values: Any
    response_logits: Any
    transition_response_logits: Any
    reward_estimates: Any
    next_feature_summaries: Any
    actions: Any
    base_actions: Any
    old_log_probabilities: Any
    actor_owned_action: Any
    rewards: Any
    dones: Any
    response_tokens: Any
    partner_indices: Any
    partner_member_indices: Any
    ego_seats: Any
    episode_ids: Any
    episode_steps: Any
    probed: Any
    candidate_actions: Any
    decision_scores: Any
    information_scores: Any
    has_safe_candidate: Any
    budget_remaining: Any
    episode_completed: Any
    completed_episode_returns: Any
    completed_episode_probe_counts: Any
    correct_deliveries: Any
    wrong_deliveries: Any
    indicator_costs: Any
    chosen_j_use: Any
    chosen_j_mask: Any
    value_base: Any
    value_mask: Any
    chosen_s_seq: Any
    chosen_response_information: Any


class RolloutCallbacks(NamedTuple):
    """Pure functions injected into the compiled rollout.

    ``model_apply`` takes ``(params, carry, observation[1,B,...], starts[1,B])``.
    ``partner_step`` takes the selected prototype indexes, observations, carry,
    starts, and one random key.  ``event_values`` extracts three per-environment
    arrays: correct deliveries, wrong deliveries, and indicator cost.
    """

    model_apply: Callable[..., Any]
    partner_step: Callable[..., Any]
    response_tokens: Callable[..., Any]
    safe_action_mask: Callable[..., Any]
    event_values: Callable[..., Any]


def _seat_observations(observations: Any, seats: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(observations)
    if values.ndim < 3 or values.shape[1] != 2:
        raise ValueError("Vector observations must have shape [environment, seat, ...].")
    selector = jnp.asarray(seats, dtype=jnp.bool_).reshape(
        (values.shape[0],) + (1,) * (values.ndim - 2)
    )
    ego = jnp.where(selector, values[:, 1], values[:, 0])
    partner = jnp.where(selector, values[:, 0], values[:, 1])
    return ego, partner


def _where_done(done: Any, reset_value: Any, continuing_value: Any) -> Any:
    import jax
    import jax.numpy as jnp

    mask = jnp.asarray(done, dtype=jnp.bool_)

    def choose(reset_leaf: Any, continuing_leaf: Any) -> Any:
        expanded = mask.reshape(mask.shape + (1,) * (jnp.ndim(reset_leaf) - mask.ndim))
        return jnp.where(expanded, reset_leaf, continuing_leaf)

    return jax.tree_util.tree_map(choose, reset_value, continuing_value)


def balanced_family_member_assignment(
    episode_ids: Any, family_member_counts: tuple[int, ...]
) -> Any:
    """Choose families equally, then rotate uniformly through their members."""

    import jax.numpy as jnp

    counts_tuple = tuple(int(value) for value in family_member_counts)
    if len(counts_tuple) <= 1 or any(value <= 0 for value in counts_tuple):
        raise ValueError("A balanced schedule requires at least two non-empty families.")
    episodes = jnp.asarray(episode_ids, dtype=jnp.int64)
    family_count = len(counts_tuple)
    family = jnp.mod(episodes, family_count)
    round_index = episodes // family_count
    counts = jnp.asarray(counts_tuple, dtype=jnp.int64)
    offsets = jnp.asarray(
        tuple(sum(counts_tuple[:index]) for index in range(family_count)),
        dtype=jnp.int64,
    )
    return offsets[family] + jnp.mod(round_index, counts[family])


def initialize_rollout(
    *,
    environment: Any,
    model_initial_carry: Callable[[int], Any],
    partner_initial_carry: Callable[[int], Any],
    random_key: Any,
    num_prototypes: int,
    budget_per_episode: int,
    family_member_counts: tuple[int, ...] | None = None,
) -> RolloutState:
    """Reset every state component and independently sample partner and seat."""

    import jax
    import jax.numpy as jnp

    if environment.num_envs <= 0 or num_prototypes <= 1 or budget_per_episode <= 0:
        raise ValueError("Rollout sizes, prototype count, and probe budget must be positive.")
    next_key, environment_key, partner_key, seat_key = jax.random.split(random_key, 4)
    env_state, observations = environment.reset(environment_key)
    count = int(environment.num_envs)
    episode_ids = jnp.arange(count, dtype=jnp.int64)
    if family_member_counts is None:
        partner_indices = jax.random.randint(
            partner_key, (count,), 0, num_prototypes
        )
    else:
        del partner_key
        if (
            len(family_member_counts) != num_prototypes
            or any(int(value) <= 0 for value in family_member_counts)
        ):
            raise ValueError("Every family must contain at least one scheduled member.")
        partner_indices = balanced_family_member_assignment(
            episode_ids, family_member_counts
        )
    return RolloutState(
        env_state=env_state,
        observations=observations,
        model_carry=model_initial_carry(count),
        partner_carry=partner_initial_carry(count),
        log_belief=uniform_log_belief(count, num_prototypes),
        partner_log_belief=uniform_log_belief(count, num_prototypes),
        partner_index=partner_indices,
        ego_seat=jax.random.bernoulli(seat_key, shape=(count,)).astype(jnp.int32),
        budget_remaining=jnp.full((count,), budget_per_episode, dtype=jnp.int32),
        partner_budget_remaining=jnp.full(
            (count,), budget_per_episode, dtype=jnp.int32
        ),
        episode_step=jnp.zeros((count,), dtype=jnp.int32),
        episode_id=episode_ids,
        episode_start=jnp.ones((count,), dtype=jnp.bool_),
        episode_return=jnp.zeros((count,), dtype=jnp.float32),
        episode_probe_count=jnp.zeros((count,), dtype=jnp.int32),
        correct_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
        wrong_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
        indicator_cost=jnp.zeros((count,), dtype=jnp.float32),
        completed_episodes=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        random_key=next_key,
    )


def collect_rollout(
    *,
    state: RolloutState,
    length: int,
    environment: Any,
    callbacks: RolloutCallbacks,
    params: Mapping[str, Any],
    controller: str,
    candidate_window: int,
    budget_per_episode: int,
    gamma: float,
    gae_lambda: float,
    probability_floor: float,
    decision_threshold: float,
    information_threshold: float,
    random_trigger_probability: float,
    num_prototypes: int,
    partner_prototype_lookup: tuple[int, ...] | None = None,
    family_member_counts: tuple[int, ...] | None = None,
    current_policy_partner: bool = False,
) -> tuple[RolloutState, RolloutBatch]:
    """Collect one full recurrent segment without host input or output."""

    import jax
    import jax.numpy as jnp

    if length <= 0:
        raise ValueError("Rollout length must be positive.")
    count = int(environment.num_envs)
    lookup_values = (
        tuple(range(num_prototypes))
        if partner_prototype_lookup is None
        else tuple(int(value) for value in partner_prototype_lookup)
    )
    prototype_lookup = (
        jnp.arange(num_prototypes, dtype=jnp.int32)
        if partner_prototype_lookup is None
        else jnp.asarray(lookup_values, dtype=jnp.int32)
    )
    if prototype_lookup.ndim != 1:
        raise ValueError("The partner-to-prototype lookup must be one-dimensional.")
    if family_member_counts is not None and (
        tuple(int(value) for value in family_member_counts)
        != tuple(
            lookup_values.count(index)
            for index in range(num_prototypes)
        )
    ):
        raise ValueError("Family member counts differ from the prototype lookup.")

    def one_step(current: RolloutState, unused: Any) -> tuple[RolloutState, tuple[Any, ...]]:
        del unused
        (
            next_root_key,
            model_key,
            environment_key,
            partner_key,
            partner_model_key,
            partner_controller_key,
            controller_key,
            next_partner_key,
            next_seat_key,
        ) = jax.random.split(current.random_key, 9)
        ego_observation, partner_observation = _seat_observations(
            current.observations, current.ego_seat
        )
        next_model_carry, model_output = callbacks.model_apply(
            params,
            current.model_carry,
            ego_observation[None, ...],
            current.episode_start[None, ...],
        )
        output = jax.tree_util.tree_map(lambda value: value[0], model_output)
        actor_logits = output["actor_logits"]
        base_action = jax.random.categorical(model_key, actor_logits, axis=-1)
        base_log_probability = jnp.take_along_axis(
            jax.nn.log_softmax(actor_logits, axis=-1), base_action[:, None], axis=-1
        )[:, 0]
        candidates = enumerate_candidates(
            safe_action_mask=callbacks.safe_action_mask(ego_observation),
            base_action=base_action,
            episode_step=current.episode_step,
            budget_remaining=current.budget_remaining,
            candidate_window=candidate_window,
        )
        sequential = sequential_scores(
            log_belief=current.log_belief,
            response_probabilities=output["transition_response_probabilities"],
            reward_estimates=output["reward_estimates"],
            next_values=output["next_values"],
            base_action=base_action,
            candidate_mask=candidates.mask,
            gamma=gamma,
            probability_floor=probability_floor,
        )
        information = response_information_scores(
            log_belief=current.log_belief,
            response_probabilities=output["transition_response_probabilities"],
            probability_floor=probability_floor,
        )
        decision = select_action(
            controller,
            key=controller_key,
            sequential=sequential,
            information_scores=information,
            candidate_mask=candidates.mask,
            base_action=base_action,
            budget_remaining=current.budget_remaining,
            decision_threshold=decision_threshold,
            information_threshold=information_threshold,
            random_trigger_probability=random_trigger_probability,
        )
        static_partner_index = (
            jnp.maximum(current.partner_index - 1, 0)
            if current_policy_partner
            else current.partner_index
        )
        frozen_partner_action, frozen_partner_carry = callbacks.partner_step(
            static_partner_index,
            partner_observation,
            current.partner_carry,
            current.episode_start,
            partner_key,
        )
        if current_policy_partner:
            frozen_params = jax.tree_util.tree_map(
                jax.lax.stop_gradient, params
            )
            dynamic_carry, dynamic_output_raw = callbacks.model_apply(
                frozen_params,
                current.partner_carry,
                partner_observation[None, ...],
                current.episode_start[None, ...],
            )
            dynamic_output = jax.tree_util.tree_map(
                lambda value: value[0], dynamic_output_raw
            )
            dynamic_base_action = jax.random.categorical(
                partner_model_key, dynamic_output["actor_logits"], axis=-1
            )
            dynamic_candidates = enumerate_candidates(
                safe_action_mask=callbacks.safe_action_mask(partner_observation),
                base_action=dynamic_base_action,
                episode_step=current.episode_step,
                budget_remaining=current.partner_budget_remaining,
                candidate_window=candidate_window,
            )
            dynamic_sequential = sequential_scores(
                log_belief=current.partner_log_belief,
                response_probabilities=dynamic_output[
                    "transition_response_probabilities"
                ],
                reward_estimates=dynamic_output["reward_estimates"],
                next_values=dynamic_output["next_values"],
                base_action=dynamic_base_action,
                candidate_mask=dynamic_candidates.mask,
                gamma=gamma,
                probability_floor=probability_floor,
            )
            dynamic_information = response_information_scores(
                log_belief=current.partner_log_belief,
                response_probabilities=dynamic_output[
                    "transition_response_probabilities"
                ],
                probability_floor=probability_floor,
            )
            dynamic_decision = select_action(
                controller,
                key=partner_controller_key,
                sequential=dynamic_sequential,
                information_scores=dynamic_information,
                candidate_mask=dynamic_candidates.mask,
                base_action=dynamic_base_action,
                budget_remaining=current.partner_budget_remaining,
                decision_threshold=decision_threshold,
                information_threshold=information_threshold,
                random_trigger_probability=random_trigger_probability,
            )
            dynamic_lane = current.partner_index == 0
            partner_action = jnp.where(
                dynamic_lane, dynamic_decision.chosen_action, frozen_partner_action
            )

            def select_partner_state(dynamic_leaf: Any, frozen_leaf: Any) -> Any:
                mask = dynamic_lane.reshape(
                    dynamic_lane.shape
                    + (1,) * (jnp.ndim(dynamic_leaf) - dynamic_lane.ndim)
                )
                return jnp.where(mask, dynamic_leaf, frozen_leaf)

            next_partner_carry = jax.tree_util.tree_map(
                select_partner_state, dynamic_carry, frozen_partner_carry
            )
        else:
            dynamic_lane = jnp.zeros((count,), dtype=jnp.bool_)
            dynamic_output = None
            dynamic_decision = None
            partner_action = frozen_partner_action
            next_partner_carry = frozen_partner_carry
        joint_actions = jnp.stack(
            (
                jnp.where(current.ego_seat == 0, decision.chosen_action, partner_action),
                jnp.where(current.ego_seat == 1, decision.chosen_action, partner_action),
            ),
            axis=-1,
        )
        next_env_state, next_observations, rewards, done, info = environment.step(
            current.env_state, joint_actions, environment_key
        )
        done = jnp.asarray(done, dtype=jnp.bool_)
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        next_ego_observation, next_partner_observation = _seat_observations(
            next_observations, current.ego_seat
        )
        observed_token = callbacks.response_tokens(
            ego_observation, next_ego_observation, done, info
        )
        action_index = decision.chosen_action[:, None, None, None]
        response_for_action = jnp.take_along_axis(
            output["response_probabilities"],
            jnp.broadcast_to(
                action_index,
                (
                    count,
                    num_prototypes,
                    1,
                    output["response_probabilities"].shape[-1],
                ),
            ),
            axis=2,
        )[:, :, 0, :]
        likelihood = likelihood_for_token(response_for_action, observed_token)
        updated_belief = log_bayes_update(
            current.log_belief,
            likelihood,
            probability_floor=probability_floor,
        )
        if current_policy_partner:
            partner_token = callbacks.response_tokens(
                partner_observation, next_partner_observation, done, info
            )
            partner_action_index = dynamic_decision.chosen_action[:, None, None, None]
            partner_response_for_action = jnp.take_along_axis(
                dynamic_output["response_probabilities"],
                jnp.broadcast_to(
                    partner_action_index,
                    (
                        count,
                        num_prototypes,
                        1,
                        dynamic_output["response_probabilities"].shape[-1],
                    ),
                ),
                axis=2,
            )[:, :, 0, :]
            partner_likelihood = likelihood_for_token(
                partner_response_for_action, partner_token
            )
            dynamic_partner_belief = log_bayes_update(
                current.partner_log_belief,
                partner_likelihood,
                probability_floor=probability_floor,
            )
            continued_partner_belief = jnp.where(
                dynamic_lane[:, None],
                dynamic_partner_belief,
                current.partner_log_belief,
            )
            continued_partner_budget = jnp.where(
                dynamic_lane,
                dynamic_decision.budget_remaining,
                current.partner_budget_remaining,
            )
        else:
            continued_partner_belief = current.partner_log_belief
            continued_partner_budget = current.partner_budget_remaining
        correct_step, wrong_step, indicator_step = callbacks.event_values(info)
        episode_return = current.episode_return + rewards
        episode_probe_count = current.episode_probe_count + decision.probed.astype(jnp.int32)
        correct_count = current.correct_delivery_count + jnp.asarray(correct_step, dtype=jnp.int32)
        wrong_count = current.wrong_delivery_count + jnp.asarray(wrong_step, dtype=jnp.int32)
        indicator_cost = current.indicator_cost + jnp.asarray(indicator_step, dtype=jnp.float32)
        next_episode_id = current.episode_id + count
        if family_member_counts is None:
            sampled_partner = jax.random.randint(
                next_partner_key, (count,), 0, prototype_lookup.shape[0]
            )
        else:
            del next_partner_key
            sampled_partner = balanced_family_member_assignment(
                next_episode_id, family_member_counts
            )
        sampled_seat = jax.random.bernoulli(next_seat_key, shape=(count,)).astype(jnp.int32)
        uniform = uniform_log_belief(count, num_prototypes)
        next_state = RolloutState(
            env_state=next_env_state,
            observations=next_observations,
            model_carry=next_model_carry,
            partner_carry=next_partner_carry,
            log_belief=jnp.where(done[:, None], uniform, updated_belief),
            partner_log_belief=jnp.where(
                done[:, None], uniform, continued_partner_belief
            ),
            partner_index=jnp.where(done, sampled_partner, current.partner_index),
            ego_seat=jnp.where(done, sampled_seat, current.ego_seat),
            budget_remaining=jnp.where(
                done,
                jnp.full((count,), budget_per_episode, dtype=jnp.int32),
                decision.budget_remaining,
            ),
            partner_budget_remaining=jnp.where(
                done,
                jnp.full((count,), budget_per_episode, dtype=jnp.int32),
                continued_partner_budget,
            ),
            episode_step=jnp.where(done, 0, current.episode_step + 1),
            episode_id=jnp.where(
                done, next_episode_id, current.episode_id
            ),
            episode_start=done,
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_probe_count=jnp.where(done, 0, episode_probe_count),
            correct_delivery_count=jnp.where(done, 0, correct_count),
            wrong_delivery_count=jnp.where(done, 0, wrong_count),
            indicator_cost=jnp.where(done, 0.0, indicator_cost),
            completed_episodes=current.completed_episodes + jnp.sum(done, dtype=jnp.int64),
            effective_environment_steps=current.effective_environment_steps + count,
            random_key=next_root_key,
        )
        recorded = (
            ego_observation,
            next_ego_observation,
            current.episode_start,
            current.model_carry,
            output["features"],
            actor_logits,
            output["shared_value"],
            output["prototype_values"],
            output["response_logits"],
            output["transition_response_logits"],
            output["reward_estimates"],
            output["next_feature_summaries"],
            decision.chosen_action,
            base_action,
            base_log_probability,
            decision.actor_owned_action,
            rewards,
            done,
            observed_token,
            prototype_lookup[current.partner_index],
            current.partner_index,
            current.ego_seat,
            current.episode_id,
            current.episode_step,
            decision.probed,
            decision.candidate,
            jnp.max(jnp.where(candidates.mask, sequential.s_seq, -jnp.inf), axis=-1),
            jnp.max(jnp.where(candidates.mask, information, -jnp.inf), axis=-1),
            candidates.any_candidate,
            decision.budget_remaining,
            done,
            jnp.where(done, episode_return, jnp.nan),
            jnp.where(done, episode_probe_count, 0),
            jnp.where(done, correct_count, 0),
            jnp.where(done, wrong_count, 0),
            jnp.where(done, indicator_cost, 0.0),
            decision.j_use,
            decision.j_mask,
            decision.v_base,
            decision.v_mask,
            decision.s_seq,
            decision.response_information,
        )
        return next_state, recorded

    final_state, recorded = jax.lax.scan(one_step, state, xs=None, length=length)
    final_ego_observation, _ = _seat_observations(
        final_state.observations, final_state.ego_seat
    )
    _, bootstrap_output = callbacks.model_apply(
        params,
        final_state.model_carry,
        final_ego_observation[None, ...],
        final_state.episode_start[None, ...],
    )
    bootstrap_output = jax.tree_util.tree_map(lambda value: value[0], bootstrap_output)
    next_features = jnp.concatenate(
        (recorded[4][1:], bootstrap_output["features"][None, ...]), axis=0
    )
    next_shared = jnp.concatenate(
        (recorded[6][1:], bootstrap_output["shared_value"][None, ...]), axis=0
    )
    next_prototype = jnp.concatenate(
        (recorded[7][1:], bootstrap_output["prototype_values"][None, ...]), axis=0
    )
    from .adaptation import generalized_advantage_estimate

    advantages, returns = generalized_advantage_estimate(
        rewards=recorded[16],
        values=recorded[6],
        next_values=next_shared,
        dones=recorded[17],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    batch = RolloutBatch(
        observations=recorded[0],
        next_observations=recorded[1],
        episode_start=recorded[2],
        model_carry=recorded[3],
        features=recorded[4],
        next_feature_targets=jax.lax.stop_gradient(next_features),
        actor_logits=recorded[5],
        shared_values=recorded[6],
        next_shared_values=jax.lax.stop_gradient(next_shared),
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        prototype_values=recorded[7],
        next_prototype_values=jax.lax.stop_gradient(next_prototype),
        response_logits=recorded[8],
        transition_response_logits=recorded[9],
        reward_estimates=recorded[10],
        next_feature_summaries=recorded[11],
        actions=recorded[12],
        base_actions=recorded[13],
        old_log_probabilities=recorded[14],
        actor_owned_action=recorded[15],
        rewards=recorded[16],
        dones=recorded[17],
        response_tokens=recorded[18],
        partner_indices=recorded[19],
        partner_member_indices=recorded[20],
        ego_seats=recorded[21],
        episode_ids=recorded[22],
        episode_steps=recorded[23],
        probed=recorded[24],
        candidate_actions=recorded[25],
        decision_scores=recorded[26],
        information_scores=recorded[27],
        has_safe_candidate=recorded[28],
        budget_remaining=recorded[29],
        episode_completed=recorded[30],
        completed_episode_returns=recorded[31],
        completed_episode_probe_counts=recorded[32],
        correct_deliveries=recorded[33],
        wrong_deliveries=recorded[34],
        indicator_costs=recorded[35],
        chosen_j_use=recorded[36],
        chosen_j_mask=recorded[37],
        value_base=recorded[38],
        value_mask=recorded[39],
        chosen_s_seq=recorded[40],
        chosen_response_information=recorded[41],
    )
    return final_state, batch


def make_compiled_collector(**static_arguments: Any) -> Callable[..., Any]:
    """Bind runtime callbacks and return one compiled segment collector."""

    import functools

    import jax

    return jax.jit(functools.partial(collect_rollout, **static_arguments))
