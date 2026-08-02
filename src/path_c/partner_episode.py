"""Complete-episode partner-generator records and admission safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np


GENERATOR_EPISODES_PER_UPDATE = 32
GENERATOR_EPISODE_STEPS = 400
GENERATOR_TRUST_REGION_KL = 0.03


class PartnerEpisodeBatch(NamedTuple):
    observations: Any
    actions: Any
    raw_rewards: Any
    official_shaped_rewards: Any
    dones: Any
    codes: Any
    behavior_log_probabilities: Any
    initial_carries: Any
    parameter_version_ids: Any
    valid_mask: Any


def validate_complete_episode_batch(batch: PartnerEpisodeBatch) -> None:
    arrays = tuple(np.asarray(value) for value in batch[:7])
    time_batch = arrays[1].shape
    if len(time_batch) != 2:
        raise ValueError("Generator episode actions must be [time, episode].")
    if time_batch[0] != GENERATOR_EPISODE_STEPS:
        raise ValueError("Generator qualification requires complete 400-step episodes.")
    for value in arrays[2:5] + arrays[6:7]:
        if value.shape != time_batch:
            raise ValueError("Generator episode transition axes differ.")
    codes = np.asarray(batch.codes)
    if codes.shape[:2] != time_batch:
        raise ValueError("The actually sampled code must be stored at every step.")
    versions = np.asarray(batch.parameter_version_ids)
    if versions.shape != time_batch:
        raise ValueError("Generator parameter version must be stored at every step.")
    valid = np.asarray(batch.valid_mask, dtype=bool)
    if valid.shape != time_batch:
        raise ValueError("Generator valid mask axes differ.")

    for episode in range(time_batch[1]):
        active = np.flatnonzero(valid[:, episode])
        if active.size == 0:
            raise ValueError("Empty generator episode lane.")
        episode_codes = codes[active, episode]
        if not np.allclose(episode_codes, episode_codes[0], atol=0.0, rtol=0.0):
            raise ValueError("Generator code changed inside one episode.")
        episode_versions = versions[active, episode]
        if not np.all(episode_versions == episode_versions[0]):
            raise ValueError("Generator parameters changed inside one episode.")
        done_positions = np.flatnonzero(np.asarray(batch.dones)[active, episode])
        if done_positions.size and done_positions[0] != active.size - 1:
            raise ValueError("Generator replay contains post-done transitions.")


def complete_episode_raw_returns(batch: PartnerEpisodeBatch) -> np.ndarray:
    validate_complete_episode_batch(batch)
    reward = np.asarray(batch.raw_rewards, dtype=np.float64)
    valid = np.asarray(batch.valid_mask, dtype=np.float64)
    return np.sum(reward * valid, axis=0)


def lower_tail_cvar(values: Iterable[float], *, level: float = 0.2) -> float:
    sample = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if sample.size == 0 or not 0.0 < level <= 1.0:
        raise ValueError("Invalid CVaR sample or level.")
    count = max(int(np.ceil(level * sample.size)), 1)
    return float(np.mean(sample[:count]))


@dataclass(frozen=True, slots=True)
class GeneratorAdmission:
    competence_passed: bool
    signature_variance_passed: bool
    oracle_code_lift_passed: bool
    stable_action_regions: int
    trust_region_kl: float
    behavior_kl: float = 0.0
    interpolation_uniform_kl: float = 0.0
    interpolation_uniform_kl_minimum: float = 0.0
    delivery_noninferiority_margin: float = 20.0
    mean_noninferiority_lcb: float = float("-inf")
    cvar_noninferiority_lcb: float = float("-inf")

    @property
    def passed(self) -> bool:
        return bool(
            self.competence_passed
            and self.signature_variance_passed
            and self.oracle_code_lift_passed
            and self.stable_action_regions >= 2
            and self.trust_region_kl <= GENERATOR_TRUST_REGION_KL
            and self.behavior_kl <= 0.05
            and self.interpolation_uniform_kl
            >= self.interpolation_uniform_kl_minimum
            and self.mean_noninferiority_lcb
            >= -self.delivery_noninferiority_margin
            and self.cvar_noninferiority_lcb
            >= -self.delivery_noninferiority_margin
        )


def generator_ppo_objective(
    *,
    new_logits: Any,
    new_values: Any,
    actions: Any,
    behavior_log_probabilities: Any,
    official_shaped_rewards: Any,
    dones: Any,
    valid_mask: Any,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_weight: float,
    entropy_weight: float,
    diversity_episode_bonus: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Official-shaped recurrent PPO on complete generator episodes.

    Raw return is deliberately absent.  It is reserved for competence and
    admission.  A diversity bonus may be broadcast to the final valid step of
    each episode only after C1/C2 qualify decision signatures.
    """

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(new_logits, dtype=jnp.float32)
    values = jnp.asarray(new_values, dtype=jnp.float32)
    selected = jnp.asarray(actions, dtype=jnp.int32)
    old_logp = jnp.asarray(behavior_log_probabilities, dtype=jnp.float32)
    reward = jnp.asarray(official_shaped_rewards, dtype=jnp.float32)
    terminal = jnp.asarray(dones, dtype=jnp.bool_)
    mask = jnp.asarray(valid_mask, dtype=jnp.float32)
    if logits.shape[:-1] != selected.shape or values.shape != selected.shape:
        raise ValueError("Generator PPO axes differ.")
    if any(jnp.asarray(value).shape != selected.shape for value in (old_logp, reward, terminal, mask)):
        raise ValueError("Generator PPO transition records are misaligned.")

    if diversity_episode_bonus is not None:
        bonus = jnp.asarray(diversity_episode_bonus, dtype=jnp.float32)
        if bonus.shape != selected.shape[1:]:
            raise ValueError("One generator diversity bonus is required per episode.")
        valid_count = jnp.sum(mask, axis=0).astype(jnp.int32)
        time = jnp.arange(selected.shape[0])[:, None]
        last = time == jnp.maximum(valid_count - 1, 0)[None, :]
        reward = reward + last.astype(jnp.float32) * bonus[None, :]

    def backward(carry: Any, row: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
        reward_t, value_t, done_t, mask_t = row
        not_done = 1.0 - done_t.astype(jnp.float32)
        delta = reward_t + float(gamma) * not_done * carry[1] - value_t
        advantage = delta + float(gamma * gae_lambda) * not_done * carry[0]
        advantage = advantage * mask_t
        return (advantage, value_t), advantage

    zero = jnp.zeros_like(values[-1])
    _, reverse_advantage = jax.lax.scan(
        backward,
        (zero, zero),
        (reward[::-1], values[::-1], terminal[::-1], mask[::-1]),
    )
    advantages = jax.lax.stop_gradient(reverse_advantage[::-1])
    returns = jax.lax.stop_gradient(advantages + values)
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    new_logp = jnp.take_along_axis(
        log_probabilities, selected[..., None], axis=-1
    )[..., 0]
    ratio = jnp.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = jnp.clip(
        ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)
    ) * advantages
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    actor_loss = -jnp.sum(mask * jnp.minimum(unclipped, clipped)) / denominator
    value_loss = 0.5 * jnp.sum(mask * jnp.square(values - returns)) / denominator
    probabilities = jnp.exp(log_probabilities)
    entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
    mean_entropy = jnp.sum(mask * entropy) / denominator
    approx_kl = jnp.sum(mask * (old_logp - new_logp)) / denominator
    total = actor_loss + float(value_weight) * value_loss - float(entropy_weight) * mean_entropy
    return total, {
        "generator_ppo_total": total,
        "generator_actor_loss": actor_loss,
        "generator_value_loss": value_loss,
        "generator_entropy": mean_entropy,
        "generator_approx_kl": approx_kl,
        "generator_valid_transitions": jnp.sum(mask),
    }


def source_logit_distillation_loss(
    generator_logits: Any,
    source_logits: Any,
    valid_mask: Any,
) -> Any:
    """Forward KL from each capable Official source into the shared generator."""

    import jax.numpy as jnp
    import jax.nn as jnn

    candidate = jnp.asarray(generator_logits, dtype=jnp.float32)
    source = jnp.asarray(source_logits, dtype=jnp.float32)
    mask = jnp.asarray(valid_mask, dtype=jnp.float32)
    if candidate.shape != source.shape or candidate.shape[:-1] != mask.shape:
        raise ValueError("Generator distillation records are misaligned.")
    source_probability = jnn.softmax(source, axis=-1)
    pointwise = jnp.sum(
        source_probability * (jnn.log_softmax(source, axis=-1) - jnn.log_softmax(candidate, axis=-1)),
        axis=-1,
    )
    return jnp.sum(mask * pointwise) / jnp.maximum(jnp.sum(mask), 1.0)


def collect_complete_generator_episodes(
    *,
    environment: Any,
    model: Any,
    ego_params: Any,
    model_config: Any,
    generator: Any,
    generator_params: Any,
    codes: Any,
    key: Any,
    parameter_version_id: int,
    official_shaping_factor: Any = 1.0,
) -> PartnerEpisodeBatch:
    """Collect 32 one-code, one-parameter-version complete Official episodes."""

    import jax
    import jax.numpy as jnp

    from .counterfactual_anchor import tree_select
    from .model import initial_policy_state
    from .partner_generator import initial_generator_carry
    from .runner import observe_policy_after_transition

    episode_count = int(jnp.asarray(codes).shape[0])
    if episode_count != int(environment.num_envs):
        raise ValueError("Generator episode environment must have one lane per code.")
    if int(environment.episode_steps) != GENERATOR_EPISODE_STEPS:
        raise ValueError("Generator collection must use complete 400-step episodes.")
    reset_key, scan_key = jax.random.split(key)
    environment_state, joint_observation = environment.reset(reset_key)
    ego_state = initial_policy_state(
        batch_size=episode_count,
        observation_shape=tuple(environment.observation_shape),
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
        mixture_components=model_config.mixture_components,
    )
    generator_carry = initial_generator_carry(
        episode_count, int(generator.hidden_dim)
    )
    episode_start = jnp.ones((episode_count,), dtype=jnp.bool_)
    active = jnp.ones((episode_count,), dtype=jnp.bool_)
    roles = (jnp.arange(episode_count) >= episode_count // 2).astype(jnp.int32)
    initial_carry = generator_carry

    def one(carry: Any, time_index: Any) -> tuple[Any, Any]:
        env_state, observations, policy_state, partner_carry, starts, is_active = carry
        lane = jnp.arange(episode_count, dtype=jnp.int32)
        ego_observation = observations[lane, roles]
        partner_observation = observations[lane, 1 - roles]
        root = jax.random.fold_in(scan_key, time_index)
        ego_root, partner_root, environment_root = jax.random.split(root, 3)
        ego_keys = jax.random.split(ego_root, episode_count)
        partner_keys = jax.random.split(partner_root, episode_count)
        stepped_ego, ego_output = model.apply(
            {"params": ego_params},
            policy_state,
            ego_observation,
            jnp.zeros((episode_count,), dtype=jnp.float32),
            method=model.step,
        )
        ego_action = jax.vmap(jax.random.categorical)(
            ego_keys, ego_output.base_logits
        )
        next_partner_carry, generator_output = generator.apply(
            {"params": generator_params},
            partner_carry,
            partner_observation,
            codes,
            starts,
            partner_keys,
            method=generator.step,
        )
        ego_first = jnp.stack((ego_action, generator_output.action), axis=-1)
        partner_first = jnp.stack((generator_output.action, ego_action), axis=-1)
        joint_actions = jnp.where(roles[:, None] == 0, ego_first, partner_first)
        next_env, next_observations, unused_agent0_reward, dones, info = environment.step(
            env_state, joint_actions, environment_root
        )
        del unused_agent0_reward
        raw_by_agent = info["raw_rewards_by_agent"]
        shaped_by_agent = info["official_shaped_rewards_by_agent"]
        ego_reward = raw_by_agent[lane, roles]
        partner_reward = raw_by_agent[lane, 1 - roles]
        partner_shaped = (
            partner_reward
            + jnp.asarray(official_shaping_factor, dtype=jnp.float32)
            * shaped_by_agent[lane, 1 - roles]
        )
        terminal = info["terminal_observations"]
        mask = dones.reshape(dones.shape + (1,) * (ego_observation.ndim - 1))
        ego_next = jnp.where(
            mask,
            terminal[lane, roles],
            next_observations[lane, roles],
        )
        next_ego = observe_policy_after_transition(
            stepped_state=stepped_ego,
            action=ego_action,
            reward=ego_reward,
            done=dones,
            next_observation=ego_next,
            model_config=model_config,
        )
        next_active = is_active & ~dones
        candidate = (
            next_env,
            next_observations,
            next_ego,
            next_partner_carry,
            dones,
            next_active,
        )
        # After termination, a lane remains frozen and every later record is
        # masked.  Fixed 400-step execution is only an XLA shape contract.
        frozen = (env_state, observations, policy_state, partner_carry, starts, is_active)
        next_carry = tree_select(is_active, candidate, frozen)
        row = (
            partner_observation,
            generator_output.action,
            partner_reward,
            partner_shaped,
            dones,
            jnp.broadcast_to(codes, (episode_count, codes.shape[-1])),
            generator_output.log_probability,
            jnp.full((episode_count,), int(parameter_version_id), dtype=jnp.int32),
            is_active,
        )
        return next_carry, row

    unused_final, rows = jax.lax.scan(
        one,
        (
            environment_state,
            joint_observation,
            ego_state,
            generator_carry,
            episode_start,
            active,
        ),
        jnp.arange(GENERATOR_EPISODE_STEPS, dtype=jnp.int32),
    )
    del unused_final
    (
        observations,
        actions,
        raw_rewards,
        shaped_rewards,
        dones,
        recorded_codes,
        behavior_log_probabilities,
        versions,
        valid_mask,
    ) = rows
    return PartnerEpisodeBatch(
        observations=observations,
        actions=actions,
        raw_rewards=raw_rewards,
        official_shaped_rewards=shaped_rewards,
        dones=dones,
        codes=recorded_codes,
        behavior_log_probabilities=behavior_log_probabilities,
        initial_carries=initial_carry,
        parameter_version_ids=versions,
        valid_mask=valid_mask,
    )


def admitted_source_probabilities(consecutive_passes: int) -> tuple[float, float, float]:
    count = int(consecutive_passes)
    if count <= 0:
        return 0.0, 0.0, 1.0
    if count == 1:
        return 0.125, 0.125, 0.75
    if count < 4:
        return 0.25, 0.25, 0.50
    return 0.50, 0.25, 0.25


def rollback_unqualified_generator(
    candidate_params: Any,
    last_qualified_params: Any,
    admission: GeneratorAdmission,
) -> Any:
    del candidate_params
    if admission.passed:
        raise ValueError("Passed candidates must be committed, not rolled back.")
    return last_qualified_params


__all__ = [
    "GENERATOR_EPISODES_PER_UPDATE",
    "GENERATOR_EPISODE_STEPS",
    "GENERATOR_TRUST_REGION_KL",
    "GeneratorAdmission",
    "PartnerEpisodeBatch",
    "admitted_source_probabilities",
    "complete_episode_raw_returns",
    "collect_complete_generator_episodes",
    "generator_ppo_objective",
    "lower_tail_cvar",
    "rollback_unqualified_generator",
    "source_logit_distillation_loss",
    "validate_complete_episode_batch",
]
