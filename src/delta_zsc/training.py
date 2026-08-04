"""Training for unified DELTA-ZSC.

Task competence and partner inference are two separate estimators:

* base parameters optimize the standard PPO objective of the executed analytic
  policy while latent outputs are stop-gradient;
* latent parameters optimize one joint response-decision marginal likelihood
  while base features are stop-gradient.

There is no weighted sum between these estimators, no shared Adam moments, and
no comparator/separation/anti-collapse/auxiliary actor transaction.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .behavior_statistics import behavior_features, update_behavior_posterior
from .filtering import (
    decision_emission_log_probability,
    sequence_log_likelihood,
)
from .model import (
    BASE_VARIANT,
    FULL_VARIANT,
    JOINT_VARIANT,
    RESPONSE_ONLY_VARIANT,
    METHOD_VARIANTS,
)
from .response_model import response_log_probability
from .types import BaseLoss, DecisionAnchorBatch, LatentLoss, RolloutBatch


class OptimizerUpdate(NamedTuple):
    params: Any
    optimizer_state: Any
    metrics: Mapping[str, Any]
    applied: Any


def action_log_probability(logits: Any, actions: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    log_probability = jnn.log_softmax(logits, axis=-1)
    index = jnp.asarray(actions, dtype=jnp.int32)[..., None]
    return jnp.take_along_axis(log_probability, index, axis=-1)[..., 0]


def categorical_entropy(logits: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    log_probability = jnn.log_softmax(logits, axis=-1)
    probability = jnp.exp(log_probability)
    return -jnp.sum(probability * log_probability, axis=-1)


def generalized_advantage_estimation(
    *,
    rewards: Any,
    dones: Any,
    values: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    value = jnp.asarray(values, dtype=jnp.float32)
    if value.shape[0] != reward.shape[0] + 1:
        raise ValueError("GAE values need one bootstrap step.")

    def one(carry: Any, items: tuple[Any, Any, Any, Any]):
        reward_t, done_t, value_t, next_value = items
        not_done = 1.0 - done_t.astype(jnp.float32)
        delta = reward_t + float(gamma) * not_done * next_value - value_t
        advantage = delta + float(gamma) * float(gae_lambda) * not_done * carry
        return advantage, advantage

    _, reverse = jax.lax.scan(
        one,
        jnp.zeros_like(value[0]),
        (
            reward[::-1],
            done[::-1],
            value[:-1][::-1],
            value[1:][::-1],
        ),
    )
    advantage = reverse[::-1]
    return advantage, advantage + value[:-1]


def replay_agent_sequence(
    *,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    initial_state: Any,
    observations: Any,
    previous_actions: Any,
    episode_starts: Any,
    variant: str,
) -> tuple[Any, Any]:
    import jax

    normalized = str(variant).lower()
    if normalized not in METHOD_VARIANTS:
        raise ValueError(f"Unknown unified variant: {variant}")

    def one(state: Any, item: tuple[Any, Any, Any]):
        observation, previous_action, start = item
        current = state._replace(
            previous_action=previous_action,
            episode_start=start,
        )
        return agent.step(
            base_params=base_params,
            latent_params=latent_params,
            state=current,
            observation=observation,
            variant=normalized,
        )

    return jax.lax.scan(
        one,
        initial_state,
        (observations, previous_actions, episode_starts),
    )


def base_ppo_loss(
    *,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    batch: RolloutBatch,
    variant: str,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_weight: float,
    entropy_weight: float,
    normalize_advantages: bool,
) -> BaseLoss:
    import jax
    import jax.numpy as jnp

    frozen_latent = jax.tree_util.tree_map(jax.lax.stop_gradient, latent_params)
    _, output = replay_agent_sequence(
        agent=agent,
        base_params=base_params,
        latent_params=frozen_latent,
        initial_state=batch.initial_state,
        observations=batch.observations,
        previous_actions=batch.previous_actions,
        episode_starts=batch.episode_starts,
        variant=variant,
    )
    logits = output.latent.adapted_logits[:-1]
    values = output.base.value
    advantages, returns = generalized_advantage_estimation(
        rewards=batch.shaped_rewards,
        dones=batch.dones,
        values=values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    if normalize_advantages:
        mean = jnp.sum(mask * advantages) / jnp.maximum(jnp.sum(mask), 1.0)
        variance = jnp.sum(mask * jnp.square(advantages - mean)) / jnp.maximum(
            jnp.sum(mask), 1.0
        )
        advantages = (advantages - mean) / jnp.sqrt(variance + 1.0e-8)

    new_log_probability = action_log_probability(logits, batch.actions)
    ratio = jnp.exp(new_log_probability - batch.old_log_probabilities)
    unclipped = ratio * advantages
    clipped = jnp.clip(
        ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)
    ) * advantages
    actor = -jnp.sum(mask * jnp.minimum(unclipped, clipped)) / jnp.maximum(
        jnp.sum(mask), 1.0
    )
    value_error = jnp.square(values[:-1] - returns)
    value = 0.5 * jnp.sum(mask * value_error) / jnp.maximum(jnp.sum(mask), 1.0)
    entropy = jnp.sum(mask * categorical_entropy(logits)) / jnp.maximum(
        jnp.sum(mask), 1.0
    )
    total = actor + float(value_weight) * value - float(entropy_weight) * entropy

    new_log_all = jax.nn.log_softmax(logits, axis=-1)
    old_probability = jnp.asarray(batch.behavior_probabilities, dtype=jnp.float32)
    old_log_all = jnp.log(jnp.maximum(old_probability, 1.0e-12))
    exact_kl = jnp.sum(
        mask
        * jnp.sum(
            old_probability * (old_log_all - new_log_all),
            axis=-1,
        )
    ) / jnp.maximum(jnp.sum(mask), 1.0)
    return BaseLoss(total, actor, value, entropy, exact_kl)


def replay_behavior_features(batch: RolloutBatch) -> Any:
    """Recompute analytic behaviour sufficient statistics at every action state."""

    import jax

    initial = (
        batch.initial_state.behavior,
        batch.initial_state.previous_observation,
    )

    def one(carry: tuple[Any, Any], item: tuple[Any, Any]):
        behavior, previous_observation = carry
        observation, start = item
        updated = update_behavior_posterior(
            behavior,
            previous_observation=previous_observation,
            current_observation=observation,
            episode_start=start,
        )
        return (updated, observation), behavior_features(updated)

    _, features = jax.lax.scan(
        one,
        initial,
        (batch.observations[:-1], batch.episode_starts[:-1]),
    )
    return features


def latent_joint_loss(
    *,
    base_model: Any,
    latent_model: Any,
    base_params: Any,
    latent_params: Any,
    batch: RolloutBatch,
    anchors: DecisionAnchorBatch | None,
    variant: str,
) -> LatentLoss:
    """One marginal-likelihood objective for response and decision evidence."""

    import jax
    import jax.numpy as jnp

    normalized = str(variant).lower()
    if normalized == BASE_VARIANT:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return LatentLoss(zero, zero, zero, zero, zero)
    if normalized not in {
        RESPONSE_ONLY_VARIANT,
        JOINT_VARIANT,
        FULL_VARIANT,
    }:
        raise ValueError(f"Unknown latent-training variant: {variant}")

    frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
    _, base_output = base_model.apply(
        {"params": frozen_base},
        batch.initial_state.task_carry,
        batch.observations[:-1],
        batch.episode_starts[:-1],
        method=base_model.sequence,
    )
    task_features = jax.lax.stop_gradient(base_output.task_features)
    instant_partner = jax.lax.stop_gradient(base_output.instant_partner)
    statistics = replay_behavior_features(batch)

    response_logits = latent_model.apply(
        {"params": latent_params},
        batch.observations[:-1],
        statistics,
        batch.actions,
        method=latent_model.response_logits,
    )
    from src.path_c.response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1],
        batch.response_next_observations,
        planes=planes,
    )
    response_logp = response_log_probability(response_logits, targets)
    transition = latent_model.apply(
        {"params": latent_params}, method=latent_model.transition
    )

    decision_logp = jnp.zeros_like(response_logp)
    decision_mask = jnp.zeros(response_logp.shape[:-1], dtype=jnp.float32)
    if anchors is not None and normalized in {JOINT_VARIANT, FULL_VARIANT}:
        decision_mean, decision_log_scale = latent_model.apply(
            {"params": latent_params},
            task_features,
            instant_partner,
            statistics,
            include_component_residual=True,
            method=latent_model.decision_emission,
        )
        time = jnp.asarray(anchors.time_indexes, dtype=jnp.int32)
        lane = jnp.asarray(anchors.lane_indexes, dtype=jnp.int32)
        mean_anchor = decision_mean[time, lane]
        scale_anchor = decision_log_scale[time, lane]
        anchor_logp = decision_emission_log_probability(
            mean_anchor,
            scale_anchor,
            anchors.centered_returns,
            anchors.standard_errors,
            anchors.action_mask,
        )
        valid_anchor = jnp.any(
            jnp.asarray(anchors.action_mask, dtype=jnp.bool_), axis=-1
        ).astype(jnp.float32)
        decision_logp = decision_logp.at[time, lane].set(anchor_logp)
        decision_mask = decision_mask.at[time, lane].set(valid_anchor)

    likelihood = sequence_log_likelihood(
        transition=transition,
        response_log_likelihood=response_logp,
        episode_starts=batch.episode_starts[:-1],
        valid_mask=batch.ppo_mask,
        initial_belief=batch.initial_state.belief,
        decision_log_likelihood=decision_logp,
        decision_mask=decision_mask,
    )
    return LatentLoss(
        total=likelihood.total_nll,
        response_nll=likelihood.response_nll,
        decision_nll=likelihood.decision_nll,
        response_observation_count=likelihood.response_count,
        decision_observation_count=likelihood.decision_count,
    )


def make_optimizer(
    params: Any,
    *,
    learning_rate: float,
    gradient_clip_norm: float,
    adam_epsilon: float,
) -> tuple[Any, Any]:
    import optax

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(gradient_clip_norm)),
        optax.adam(float(learning_rate), eps=float(adam_epsilon)),
    )
    return optimizer, optimizer.init(params)


def apply_base_update(
    *,
    agent: Any,
    params: Any,
    latent_params: Any,
    optimizer: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    variant: str,
    ppo_config: Any,
) -> OptimizerUpdate:
    import jax
    import jax.numpy as jnp
    import optax

    def objective(candidate: Any):
        loss = base_ppo_loss(
            agent=agent,
            base_params=candidate,
            latent_params=latent_params,
            batch=batch,
            variant=variant,
            gamma=float(ppo_config.gamma),
            gae_lambda=float(ppo_config.gae_lambda),
            clip_epsilon=float(ppo_config.clip_epsilon),
            value_weight=float(ppo_config.value_weight),
            entropy_weight=float(ppo_config.entropy_weight),
            normalize_advantages=bool(ppo_config.normalize_advantages),
        )
        return loss.total, loss

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    updates, next_optimizer_state = optimizer.update(
        gradients, optimizer_state, params
    )
    candidate = optax.apply_updates(params, updates)
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree_util.tree_leaves(
                    (candidate, next_optimizer_state, metrics)
                )
            ]
        )
    )
    next_params = jax.lax.cond(finite, lambda _: candidate, lambda _: params, None)
    next_state = jax.lax.cond(
        finite, lambda _: next_optimizer_state, lambda _: optimizer_state, None
    )
    return OptimizerUpdate(
        params=next_params,
        optimizer_state=next_state,
        metrics={
            "total": metrics.total,
            "actor": metrics.actor,
            "value": metrics.value,
            "entropy": metrics.entropy,
            "exact_kl": metrics.exact_kl,
        },
        applied=finite,
    )


def apply_latent_update(
    *,
    base_model: Any,
    latent_model: Any,
    base_params: Any,
    params: Any,
    optimizer: Any,
    optimizer_state: Any,
    batch: RolloutBatch,
    anchors: DecisionAnchorBatch | None,
    variant: str,
) -> OptimizerUpdate:
    import jax
    import jax.numpy as jnp
    import optax

    def objective(candidate: Any):
        loss = latent_joint_loss(
            base_model=base_model,
            latent_model=latent_model,
            base_params=base_params,
            latent_params=candidate,
            batch=batch,
            anchors=anchors,
            variant=variant,
        )
        return loss.total, loss

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    updates, next_optimizer_state = optimizer.update(
        gradients, optimizer_state, params
    )
    candidate = optax.apply_updates(params, updates)
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree_util.tree_leaves(
                    (candidate, next_optimizer_state, metrics)
                )
            ]
        )
    )
    next_params = jax.lax.cond(finite, lambda _: candidate, lambda _: params, None)
    next_state = jax.lax.cond(
        finite, lambda _: next_optimizer_state, lambda _: optimizer_state, None
    )
    return OptimizerUpdate(
        params=next_params,
        optimizer_state=next_state,
        metrics={
            "total": metrics.total,
            "response_nll": metrics.response_nll,
            "decision_nll": metrics.decision_nll,
            "response_observation_count": metrics.response_observation_count,
            "decision_observation_count": metrics.decision_observation_count,
        },
        applied=finite,
    )


__all__ = [
    "OptimizerUpdate",
    "action_log_probability",
    "apply_base_update",
    "apply_latent_update",
    "base_ppo_loss",
    "categorical_entropy",
    "generalized_advantage_estimation",
    "latent_joint_loss",
    "make_optimizer",
    "replay_agent_sequence",
    "replay_behavior_features",
]
