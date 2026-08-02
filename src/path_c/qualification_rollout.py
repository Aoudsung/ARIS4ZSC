"""Paired full-episode rollouts used only by r3 signal qualification.

All comparisons reuse the same episode, ego-action, partner-action and
environment keys.  The module exposes returns and block labels; statistical
decisions remain in :mod:`qualification` and :mod:`signal_audit`.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class PairedEpisodeReturns(NamedTuple):
    candidate: Any
    reference: Any
    partner_run_ids: Any


class ContextEpisodeReturns(NamedTuple):
    oracle_context: Any
    online_context: Any
    state_only: Any
    partner_run_ids: Any


def _generator_partner_branch(
    *, environment: Any, model: Any, ego_params: Any, model_config: Any,
    generator: Any, generator_params: Any, codes: Any, partner_pool: Any,
    partner_members: Any, episode_keys: Any, roles: Any, use_generator: bool,
) -> Any:
    import jax
    import jax.numpy as jnp

    from .model import initial_policy_state
    from .partner_generator import initial_generator_carry
    from .runner import observe_policy_after_transition

    count = int(partner_members.shape[0])
    state, observations = environment.reset_with_keys(_reset_keys(episode_keys))
    ego = initial_policy_state(
        batch_size=count, observation_shape=tuple(environment.observation_shape),
        action_count=6, task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
        mixture_components=model_config.mixture_components,
    )
    partner = (
        initial_generator_carry(count, int(generator.hidden_dim))
        if use_generator else partner_pool.initial_carry(count)
    )
    active = jnp.ones((count,), dtype=jnp.bool_)
    total = jnp.zeros((count,), dtype=jnp.float32)

    def one(carry: Any, time_index: Any) -> tuple[Any, Any]:
        env_state, joint_obs, ego_state, partner_state, mask, returns = carry
        lane = jnp.arange(count, dtype=jnp.int32)
        ego_obs = joint_obs[lane, roles]
        partner_obs = joint_obs[lane, 1 - roles]
        ego_keys, partner_keys, env_keys = _step_keys(episode_keys, time_index)
        stepped, output = model.apply(
            {"params": ego_params}, ego_state, ego_obs,
            jnp.zeros((count,), dtype=jnp.float32), method=model.step,
        )
        ego_action = jax.vmap(jax.random.categorical)(ego_keys, output.base_logits)
        if use_generator:
            next_partner, generator_output = generator.apply(
                {"params": generator_params}, partner_state, partner_obs, codes,
                ego_state.episode_start, partner_keys, method=generator.step,
            )
            partner_action = generator_output.action
        else:
            partner_action, next_partner = partner_pool.step_with_keys(
                partner_members, partner_obs, partner_state,
                ego_state.episode_start, partner_keys,
            )
        next_env, next_obs, unused, dones, info = environment.step_with_keys(
            env_state, _joint_actions(ego_action, partner_action, roles), env_keys
        )
        del unused
        reward = info["raw_rewards_by_agent"][lane, roles]
        ego_next = _terminal_ego_observation(
            next_obs, info["terminal_observations"], dones, roles
        )
        next_ego = observe_policy_after_transition(
            stepped_state=stepped, action=ego_action, reward=reward, done=dones,
            next_observation=ego_next, model_config=model_config,
        )
        next_active = mask & ~dones
        next_returns = returns + mask.astype(jnp.float32) * reward
        return (
            next_env, next_obs, next_ego, next_partner, next_active, next_returns
        ), next_returns

    final, unused_rows = jax.lax.scan(
        one, (state, observations, ego, partner, active, total),
        jnp.arange(int(environment.episode_steps), dtype=jnp.int32),
    )
    del unused_rows
    return final[-1]


def _episode_member_schedule(*, member_count: int, episodes_per_member: int) -> Any:
    import jax.numpy as jnp

    if member_count < 2 or episodes_per_member <= 0:
        raise ValueError("Qualification needs repeated episodes from independent partners.")
    return jnp.repeat(
        jnp.arange(int(member_count), dtype=jnp.int32), int(episodes_per_member)
    )


def _episode_keys(root_key: Any, count: int) -> Any:
    import jax

    return jax.random.split(root_key, int(count))


def _step_keys(episode_keys: Any, time_index: Any) -> tuple[Any, Any, Any]:
    import jax

    roots = jax.vmap(lambda key: jax.random.fold_in(key, time_index))(episode_keys)
    split = jax.vmap(lambda key: jax.random.split(key, 3))(roots)
    return split[:, 0], split[:, 1], split[:, 2]


def _reset_keys(episode_keys: Any) -> Any:
    import jax

    return jax.vmap(lambda key: jax.random.fold_in(key, 91_001))(episode_keys)


def _joint_actions(ego_actions: Any, partner_actions: Any, roles: Any) -> Any:
    import jax.numpy as jnp

    ego_first = jnp.stack((ego_actions, partner_actions), axis=-1)
    partner_first = jnp.stack((partner_actions, ego_actions), axis=-1)
    return jnp.where(roles[:, None] == 0, ego_first, partner_first)


def _terminal_ego_observation(
    observations: Any, terminal: Any, dones: Any, roles: Any
) -> Any:
    import jax.numpy as jnp

    lane = jnp.arange(roles.shape[0], dtype=jnp.int32)
    current = observations[lane, roles]
    terminal_current = terminal[lane, roles]
    mask = dones.reshape(dones.shape + (1,) * (current.ndim - 1))
    return jnp.where(mask, terminal_current, current)


def _official_branch(
    *, environment: Any, ego_pool: Any, partner_pool: Any, partner_members: Any,
    episode_keys: Any, roles: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    count = int(partner_members.shape[0])
    state, observations = environment.reset_with_keys(_reset_keys(episode_keys))
    ego_members = jnp.zeros((count,), dtype=jnp.int32)
    ego_carry = ego_pool.initial_carry(count)
    partner_carry = partner_pool.initial_carry(count)
    starts = jnp.ones((count,), dtype=jnp.bool_)
    active = jnp.ones((count,), dtype=jnp.bool_)
    returns = jnp.zeros((count,), dtype=jnp.float32)

    def one(carry: Any, time_index: Any) -> tuple[Any, Any]:
        env_state, joint_obs, ego_state, other_state, episode_start, mask, total = carry
        lane = jnp.arange(count, dtype=jnp.int32)
        ego_obs = joint_obs[lane, roles]
        other_obs = joint_obs[lane, 1 - roles]
        ego_keys, partner_keys, env_keys = _step_keys(episode_keys, time_index)
        ego_action, next_ego = ego_pool.step_with_keys(
            ego_members, ego_obs, ego_state, episode_start, ego_keys
        )
        partner_action, next_partner = partner_pool.step_with_keys(
            partner_members, other_obs, other_state, episode_start, partner_keys
        )
        next_env, next_obs, unused, dones, info = environment.step_with_keys(
            env_state, _joint_actions(ego_action, partner_action, roles), env_keys
        )
        del unused
        reward = info["raw_rewards_by_agent"][lane, roles]
        next_total = total + mask.astype(jnp.float32) * reward
        next_active = mask & ~dones
        return (
            next_env, next_obs, next_ego, next_partner, dones, next_active, next_total
        ), next_total

    final, unused_rows = jax.lax.scan(
        one,
        (state, observations, ego_carry, partner_carry, starts, active, returns),
        jnp.arange(int(environment.episode_steps), dtype=jnp.int32),
    )
    del unused_rows
    return final[-1]


def _delta_branch(
    *, environment: Any, model: Any, params: Any, model_config: Any,
    partner_pool: Any, partner_members: Any, episode_keys: Any, roles: Any,
    gate: float,
    fixed_latent: Any | None = None,
) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    from .belief_set_encoder import mixture_moments
    from .model import initial_policy_state
    from .counterfactual_anchor import tree_select
    from .runner import observe_policy_after_transition

    count = int(partner_members.shape[0])
    state, observations = environment.reset_with_keys(_reset_keys(episode_keys))
    policy = initial_policy_state(
        batch_size=count,
        observation_shape=tuple(environment.observation_shape),
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
        mixture_components=model_config.mixture_components,
    )
    partner_carry = partner_pool.initial_carry(count)
    active = jnp.ones((count,), dtype=jnp.bool_)
    returns = jnp.zeros((count,), dtype=jnp.float32)
    entropy_sum = jnp.zeros((count,), dtype=jnp.float32)
    kl_sum = jnp.zeros((count,), dtype=jnp.float32)
    latest_online_latent = jnp.zeros(
        (count, int(model_config.latent_dim)), dtype=jnp.float32
    )

    def one(carry: Any, time_index: Any) -> tuple[Any, Any]:
        (
            env_state,
            joint_obs,
            ego_state,
            other_state,
            mask,
            total,
            entropy_total,
            kl_total,
            latest_latent,
        ) = carry
        lane = jnp.arange(count, dtype=jnp.int32)
        ego_obs = joint_obs[lane, roles]
        other_obs = joint_obs[lane, 1 - roles]
        ego_keys, partner_keys, env_keys = _step_keys(episode_keys, time_index)
        stepped, output = model.apply(
            {"params": params}, ego_state, ego_obs,
            jnp.full((count,), float(gate), dtype=jnp.float32), method=model.step,
        )
        online_latent = mixture_moments(
            output.mixture_logits,
            output.mixture_means,
            output.mixture_log_variances,
        )[0]
        if fixed_latent is None:
            logits = output.base_logits + float(gate) * output.residual_logits
        else:
            diagnostic = model.apply(
                {"params": params},
                output.task_features,
                jnp.asarray(fixed_latent, dtype=jnp.float32),
                float(gate),
                method=model.from_features_and_latent,
            )
            logits = diagnostic.base_logits + float(gate) * diagnostic.residual_logits
        ego_action = jax.vmap(jax.random.categorical)(ego_keys, logits)
        partner_action, next_partner = partner_pool.step_with_keys(
            partner_members, other_obs, other_state, ego_state.episode_start, partner_keys
        )
        next_env, next_obs, unused, dones, info = environment.step_with_keys(
            env_state, _joint_actions(ego_action, partner_action, roles), env_keys
        )
        del unused
        reward = info["raw_rewards_by_agent"][lane, roles]
        ego_next = _terminal_ego_observation(
            next_obs, info["terminal_observations"], dones, roles
        )
        next_policy = observe_policy_after_transition(
            stepped_state=stepped,
            action=ego_action,
            reward=reward,
            done=dones,
            next_observation=ego_next,
            model_config=model_config,
        )
        base_log = jax.nn.log_softmax(output.base_logits, axis=-1)
        conditional_log = jax.nn.log_softmax(
            output.base_logits + output.residual_logits, axis=-1
        )
        conditional_prob = jnp.exp(conditional_log)
        entropy = -jnp.sum(conditional_prob * conditional_log, axis=-1)
        divergence = jnp.sum(conditional_prob * (conditional_log - base_log), axis=-1)
        next_active = mask & ~dones
        candidate = (
            next_env,
            next_obs,
            next_policy,
            next_partner,
            next_active,
            total + mask.astype(jnp.float32) * reward,
            entropy_total + mask.astype(jnp.float32) * entropy,
            kl_total + mask.astype(jnp.float32) * divergence,
            jnp.where(mask[:, None], online_latent, latest_latent),
        )
        frozen = (
            env_state,
            joint_obs,
            ego_state,
            other_state,
            mask,
            total,
            entropy_total,
            kl_total,
            latest_latent,
        )
        return tree_select(mask, candidate, frozen), online_latent

    final, posterior_means = jax.lax.scan(
        one,
        (
            state,
            observations,
            policy,
            partner_carry,
            active,
            returns,
            entropy_sum,
            kl_sum,
            latest_online_latent,
        ),
        jnp.arange(int(environment.episode_steps), dtype=jnp.int32),
    )
    del posterior_means
    return final[5], final[8], (final[6], final[7])


def paired_owner_base_returns(
    *, environment: Any, model: Any, params: Any, model_config: Any,
    owner_pool: Any, partner_pool: Any, episodes_per_partner: int, key: Any,
) -> PairedEpisodeReturns:
    """C0 candidate/owner return differences with exact paired random domains."""

    import jax.numpy as jnp

    members = _episode_member_schedule(
        member_count=int(partner_pool.member_count),
        episodes_per_member=int(episodes_per_partner),
    )
    total = int(members.shape[0])
    if total % int(environment.num_envs):
        raise ValueError("C0 episode blocks must divide the fixed vector environment.")
    keys = _episode_keys(key, total)
    candidate_rows, reference_rows, block_rows = [], [], []
    for start in range(0, total, int(environment.num_envs)):
        stop = start + int(environment.num_envs)
        block_members = members[start:stop]
        block_keys = keys[start:stop]
        roles = (jnp.arange(environment.num_envs) >= environment.num_envs // 2).astype(jnp.int32)
        candidate, unused_latent, unused_diagnostics = _delta_branch(
            environment=environment, model=model, params=params,
            model_config=model_config, partner_pool=partner_pool,
            partner_members=block_members, episode_keys=block_keys, roles=roles, gate=0.0,
        )
        del unused_latent, unused_diagnostics
        reference = _official_branch(
            environment=environment, ego_pool=owner_pool, partner_pool=partner_pool,
            partner_members=block_members, episode_keys=block_keys, roles=roles,
        )
        candidate_rows.append(candidate)
        reference_rows.append(reference)
        block_rows.append(block_members)
    return PairedEpisodeReturns(
        candidate=jnp.concatenate(candidate_rows),
        reference=jnp.concatenate(reference_rows),
        partner_run_ids=jnp.concatenate(block_rows),
    )


def paired_conditional_base_returns(
    *, environment: Any, model: Any, params: Any, model_config: Any,
    partner_pool: Any, episodes_per_partner: int, key: Any,
) -> tuple[PairedEpisodeReturns, Any]:
    """C4 full-episode conditional/base comparison and policy diagnostics."""

    import jax
    import jax.numpy as jnp

    members = _episode_member_schedule(
        member_count=int(partner_pool.member_count),
        episodes_per_member=int(episodes_per_partner),
    )
    total = int(members.shape[0])
    if total % int(environment.num_envs):
        raise ValueError("C4 episode blocks must divide the fixed vector environment.")
    keys = _episode_keys(key, total)
    conditional_rows, base_rows, block_rows, diagnostics = [], [], [], []
    for start in range(0, total, int(environment.num_envs)):
        stop = start + int(environment.num_envs)
        block_members = members[start:stop]
        block_keys = keys[start:stop]
        roles = (jnp.arange(environment.num_envs) >= environment.num_envs // 2).astype(jnp.int32)
        conditional, unused_latent, conditional_diagnostics = _delta_branch(
            environment=environment, model=model, params=params,
            model_config=model_config, partner_pool=partner_pool,
            partner_members=block_members, episode_keys=block_keys, roles=roles, gate=1.0,
        )
        base, unused_latent, unused_base_diagnostics = _delta_branch(
            environment=environment, model=model, params=params,
            model_config=model_config, partner_pool=partner_pool,
            partner_members=block_members, episode_keys=block_keys, roles=roles, gate=0.0,
        )
        del unused_latent, unused_base_diagnostics
        conditional_rows.append(conditional)
        base_rows.append(base)
        block_rows.append(block_members)
        diagnostics.append(conditional_diagnostics)
    return PairedEpisodeReturns(
        candidate=jnp.concatenate(conditional_rows),
        reference=jnp.concatenate(base_rows),
        partner_run_ids=jnp.concatenate(block_rows),
    ), jax.tree_util.tree_map(lambda *values: jnp.concatenate(values), *diagnostics)


def paired_generator_external_returns(
    *, environment: Any, model: Any, ego_params: Any, model_config: Any,
    generator: Any, generator_params: Any, partner_pool: Any, codes: Any,
    key: Any,
) -> PairedEpisodeReturns:
    """Generator admission competence against qualified external partners."""

    import jax
    import jax.numpy as jnp

    count = int(environment.num_envs)
    code_array = jnp.asarray(codes, dtype=jnp.float32)
    if code_array.shape[0] != count:
        raise ValueError("Generator admission needs one code per complete episode.")
    episode_keys = _episode_keys(key, count)
    members = jnp.arange(count, dtype=jnp.int32) % int(partner_pool.member_count)
    roles = (jnp.arange(count, dtype=jnp.int32) >= count // 2).astype(jnp.int32)
    candidate = _generator_partner_branch(
        environment=environment, model=model, ego_params=ego_params,
        model_config=model_config, generator=generator,
        generator_params=generator_params, codes=code_array,
        partner_pool=partner_pool, partner_members=members,
        episode_keys=episode_keys, roles=roles, use_generator=True,
    )
    reference = _generator_partner_branch(
        environment=environment, model=model, ego_params=ego_params,
        model_config=model_config, generator=generator,
        generator_params=generator_params, codes=code_array,
        partner_pool=partner_pool, partner_members=members,
        episode_keys=episode_keys, roles=roles, use_generator=False,
    )
    return PairedEpisodeReturns(candidate, reference, members)


def paired_context_returns(
    *, environment: Any, model: Any, params: Any, model_config: Any,
    partner_pool: Any, episodes_per_partner: int, key: Any,
    privileged_functional_contexts: Any,
) -> ContextEpisodeReturns:
    """C3 paired oracle/online/state-only return comparison.

    ``online_context`` uses only the legal recurrent history.  ``state_only``
    uses the prior latent at every step.  The oracle diagnostic uses a frozen
    SVD coordinate derived solely from independent real-return action
    signatures.  Partner identity only indexes that functional coordinate;
    neither it nor the coordinate enters a loss or deployment artifact.
    """

    import jax
    import jax.numpy as jnp

    members = _episode_member_schedule(
        member_count=int(partner_pool.member_count),
        episodes_per_member=int(episodes_per_partner),
    )
    functional = jnp.asarray(privileged_functional_contexts, dtype=jnp.float32)
    if functional.shape != (
        int(partner_pool.member_count), int(model_config.latent_dim)
    ):
        raise ValueError("C3 privileged functional contexts do not match partner runs.")
    total = int(members.shape[0])
    if total % int(environment.num_envs):
        raise ValueError("C3 episode blocks must divide the fixed vector environment.")
    keys = _episode_keys(key, total)
    oracle_rows, online_rows, state_rows, block_rows = [], [], [], []
    for start in range(0, total, int(environment.num_envs)):
        stop = start + int(environment.num_envs)
        block_members = members[start:stop]
        block_keys = keys[start:stop]
        roles = (
            jnp.arange(environment.num_envs) >= environment.num_envs // 2
        ).astype(jnp.int32)
        online, unused_online_latent, unused_diagnostics = _delta_branch(
            environment=environment,
            model=model,
            params=params,
            model_config=model_config,
            partner_pool=partner_pool,
            partner_members=block_members,
            episode_keys=block_keys,
            roles=roles,
            gate=1.0,
        )
        del unused_online_latent, unused_diagnostics
        oracle, unused_oracle_latent, unused_oracle_diagnostics = _delta_branch(
            environment=environment,
            model=model,
            params=params,
            model_config=model_config,
            partner_pool=partner_pool,
            partner_members=block_members,
            episode_keys=block_keys,
            roles=roles,
            gate=1.0,
            fixed_latent=jax.lax.stop_gradient(functional[block_members]),
        )
        state_only, unused_state_latent, unused_state_diagnostics = _delta_branch(
            environment=environment,
            model=model,
            params=params,
            model_config=model_config,
            partner_pool=partner_pool,
            partner_members=block_members,
            episode_keys=block_keys,
            roles=roles,
            gate=1.0,
            fixed_latent=jnp.zeros(
                (block_members.shape[0], int(model_config.latent_dim)),
                dtype=jnp.float32,
            ),
        )
        del (
            unused_oracle_latent,
            unused_oracle_diagnostics,
            unused_state_latent,
            unused_state_diagnostics,
        )
        oracle_rows.append(oracle)
        online_rows.append(online)
        state_rows.append(state_only)
        block_rows.append(block_members)
    return ContextEpisodeReturns(
        oracle_context=jnp.concatenate(oracle_rows),
        online_context=jnp.concatenate(online_rows),
        state_only=jnp.concatenate(state_rows),
        partner_run_ids=jnp.concatenate(block_rows),
    )


def paired_delta_params_returns(
    *, environment: Any, model: Any, candidate_params: Any, reference_params: Any,
    model_config: Any, partner_pool: Any, episodes_per_partner: int, key: Any,
    gate: float = 1.0,
) -> PairedEpisodeReturns:
    """Paired full-episode control audit for two frozen DELTA parameter trees."""

    import jax.numpy as jnp

    members = _episode_member_schedule(
        member_count=int(partner_pool.member_count),
        episodes_per_member=int(episodes_per_partner),
    )
    total = int(members.shape[0])
    if total % int(environment.num_envs):
        raise ValueError("Paired control blocks must divide the fixed vector environment.")
    keys = _episode_keys(key, total)
    candidate_rows, reference_rows, block_rows = [], [], []
    for start in range(0, total, int(environment.num_envs)):
        stop = start + int(environment.num_envs)
        block_members = members[start:stop]
        block_keys = keys[start:stop]
        roles = (
            jnp.arange(environment.num_envs) >= environment.num_envs // 2
        ).astype(jnp.int32)
        candidate, unused_latent, unused_diagnostics = _delta_branch(
            environment=environment,
            model=model,
            params=candidate_params,
            model_config=model_config,
            partner_pool=partner_pool,
            partner_members=block_members,
            episode_keys=block_keys,
            roles=roles,
            gate=float(gate),
        )
        reference, unused_reference_latent, unused_reference_diagnostics = _delta_branch(
            environment=environment,
            model=model,
            params=reference_params,
            model_config=model_config,
            partner_pool=partner_pool,
            partner_members=block_members,
            episode_keys=block_keys,
            roles=roles,
            gate=float(gate),
        )
        del (
            unused_latent,
            unused_diagnostics,
            unused_reference_latent,
            unused_reference_diagnostics,
        )
        candidate_rows.append(candidate)
        reference_rows.append(reference)
        block_rows.append(block_members)
    return PairedEpisodeReturns(
        candidate=jnp.concatenate(candidate_rows),
        reference=jnp.concatenate(reference_rows),
        partner_run_ids=jnp.concatenate(block_rows),
    )


__all__ = [
    "ContextEpisodeReturns",
    "PairedEpisodeReturns",
    "paired_conditional_base_returns",
    "paired_context_returns",
    "paired_delta_params_returns",
    "paired_generator_external_returns",
    "paired_owner_base_returns",
]
