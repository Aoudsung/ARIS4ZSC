"""Proper objectives for the two DELTA-ZSC parameter owners."""

from __future__ import annotations

from typing import Any

from .decision_model import decision_component_log_probability, decision_predict
from .response_model import response_joint_log_probability, response_predict
from .transition import predict_belief
from .types import AnchorBatch, LossResult, RolloutBatch


def categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    index = jnp.asarray(actions, dtype=jnp.int32)[..., None]
    return jnp.take_along_axis(logp, index, axis=-1)[..., 0]


def categorical_entropy(logits: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


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
        raise ValueError("GAE values must contain T+1 states.")
    delta = reward + float(gamma) * (~done).astype(jnp.float32) * value[1:] - value[:-1]

    def one(carry: Any, item: tuple[Any, Any]):
        delta_t, done_t = item
        current = delta_t + float(gamma) * float(gae_lambda) * (
            ~done_t
        ).astype(jnp.float32) * carry
        return current, current

    _, reverse = jax.lax.scan(
        one, jnp.zeros_like(delta[-1]), (delta[::-1], done[::-1])
    )
    advantage = reverse[::-1]
    returns = advantage + value[:-1]
    return jax.lax.stop_gradient(advantage), jax.lax.stop_gradient(returns)


def _masked_mean(value: Any, mask: Any) -> Any:
    import jax.numpy as jnp

    weight = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(weight * jnp.asarray(value, dtype=jnp.float32)) / jnp.maximum(
        jnp.sum(weight), 1.0
    )


def ppo_loss(
    model: Any,
    base_params: Any,
    latent_params: Any,
    batch: RolloutBatch,
) -> LossResult:
    """Clipped recurrent PPO for the task-competence base policy only."""

    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    _, output = model.sequence(
        base_params,
        jax.lax.stop_gradient(latent_params),
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        compute_latent=False,
        execute_adaptation=False,
    )
    logits = output.base_policy_logits[:-1]
    value = output.value
    reward = jnp.asarray(batch.rewards, dtype=jnp.float32) + jnp.asarray(
        batch.shaped_rewards, dtype=jnp.float32
    )
    advantages, returns = generalized_advantage_estimation(
        rewards=reward,
        dones=batch.dones,
        values=jax.lax.stop_gradient(value),
        gamma=model.config.ppo.gamma,
        gae_lambda=model.config.ppo.gae_lambda,
    )
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    if bool(model.config.ppo.normalize_advantages):
        mean = _masked_mean(advantages, mask)
        variance = _masked_mean(jnp.square(advantages - mean), mask)
        advantages = (advantages - mean) / jnp.sqrt(variance + 1.0e-8)

    new_logp = categorical_log_probability(logits, batch.actions)
    ratio = jnp.exp(new_logp - jnp.asarray(batch.old_log_probabilities))
    clipped_ratio = jnp.clip(
        ratio,
        1.0 - float(model.config.ppo.clip_epsilon),
        1.0 + float(model.config.ppo.clip_epsilon),
    )
    actor = -_masked_mean(
        jnp.minimum(ratio * advantages, clipped_ratio * advantages), mask
    )
    old_value = jnp.asarray(batch.old_values, dtype=jnp.float32)
    current_value = value[:-1]
    value_clipped = old_value + jnp.clip(
        current_value - old_value,
        -float(model.config.ppo.value_clip_epsilon),
        float(model.config.ppo.value_clip_epsilon),
    )
    value_error = jnp.maximum(
        jnp.square(current_value - returns), jnp.square(value_clipped - returns)
    )
    value_loss = 0.5 * _masked_mean(value_error, mask)
    entropy = _masked_mean(categorical_entropy(logits), mask)
    old_logp_all = jnn.log_softmax(
        # The old scalar log probability cannot reconstruct the full old
        # distribution.  This diagnostic therefore reports the exact sampled
        # action log-ratio, while clipping remains the actual trust region.
        logits,
        axis=-1,
    )
    del old_logp_all
    approximate_kl = 0.5 * _masked_mean(jnp.square(new_logp - batch.old_log_probabilities), mask)
    total = (
        actor
        + float(model.config.ppo.value_weight) * value_loss
        - float(model.config.ppo.entropy_weight) * entropy
    )
    return LossResult(
        total=total,
        metrics={
            "ppo_total": total,
            "ppo_actor": actor,
            "ppo_value": value_loss,
            "ppo_entropy": entropy,
            "ppo_sampled_action_kl": approximate_kl,
            "ppo_ratio_mean": _masked_mean(ratio, mask),
            "ppo_return_mean": _masked_mean(returns, mask),
        },
    )


def latent_composite_loss(
    model: Any,
    latent_params: Any,
    base_params: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
) -> LossResult:
    """Shared-latent proper predictive score with no auxiliary loss weights.

    Response observations and sparse CRN decision observations are independent
    measurement channels conditional on the same latent component.  Their
    negative log probabilities are summed and normalized by the number of
    actual observations.  Decision evidence is never inserted into the online
    filter state, preserving the legal deployment recursion.
    """

    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    stopped_base = jax.lax.stop_gradient(base_params)
    _, output = model.sequence(
        stopped_base,
        latent_params,
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        compute_latent=True,
        execute_adaptation=False,
    )
    from .observation import extract_response_target

    response_target = extract_response_target(
        batch.observations[:-1], batch.response_next_observations
    )
    response_prediction = response_predict(
        latent_params["response"],
        latent_params["component_embeddings"],
        batch.observations[:-1],
        output.behavior_features[:-1],
        batch.actions,
    )
    component_response_logp = response_joint_log_probability(
        response_prediction, response_target
    )
    predictive = predict_belief(
        output.belief[:-1], latent_params["transition_logits"]
    )
    response_logp = jsp.special.logsumexp(
        jnp.log(jnp.maximum(predictive, 1.0e-30)) + component_response_logp,
        axis=-1,
    )
    response_mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    response_sum = -jnp.sum(response_mask * response_logp)
    response_count = jnp.sum(response_mask)
    response_nll = response_sum / jnp.maximum(response_count, 1.0)

    decision_sum = jnp.asarray(0.0, dtype=jnp.float32)
    decision_count = jnp.asarray(0.0, dtype=jnp.float32)
    decision_nll = jnp.asarray(0.0, dtype=jnp.float32)
    decision_top_action_agreement = jnp.asarray(0.0, dtype=jnp.float32)
    decision_empirical_regret = jnp.asarray(0.0, dtype=jnp.float32)
    uses_decision_channel = (
        anchors is not None
        and model.config.method_variant in {"delta_passive", "delta_active"}
    )
    if uses_decision_channel:
        time = anchors.time_indexes
        lane = anchors.lane_indexes
        prediction = decision_predict(
            latent_params["decision"],
            latent_params["component_embeddings"],
            jax.lax.stop_gradient(output.task_features[time, lane]),
            jax.lax.stop_gradient(output.instant_partner[time, lane]),
            output.behavior_features[time, lane],
        )
        component_decision_logp = decision_component_log_probability(
            prediction,
            anchors.fit_returns_by_action,
            anchors.measurement_covariances,
            anchors.action_mask,
        )
        response_only_belief = output.belief[time, lane]
        decision_logp = jsp.special.logsumexp(
            jnp.log(jnp.maximum(response_only_belief, 1.0e-30))
            + component_decision_logp,
            axis=-1,
        )
        valid = jnp.all(anchors.action_mask, axis=-1).astype(jnp.float32)
        decision_sum = -jnp.sum(valid * decision_logp)
        decision_count = jnp.sum(valid)
        decision_nll = decision_sum / jnp.maximum(decision_count, 1.0)
        expected = jnp.sum(
            response_only_belief[..., :, None] * prediction.means, axis=-2
        )
        selected = jnp.argmax(expected, axis=-1)
        oracle = jnp.argmax(anchors.evaluation_returns_by_action, axis=-1)
        rows = jnp.arange(selected.shape[0])
        decision_top_action_agreement = jnp.sum(valid * (selected == oracle)) / jnp.maximum(
            decision_count, 1.0
        )
        decision_empirical_regret = jnp.sum(
            valid
            * (
                anchors.evaluation_returns_by_action[rows, oracle]
                - anchors.evaluation_returns_by_action[rows, selected]
            )
        ) / jnp.maximum(decision_count, 1.0)

    total_count = response_count + decision_count
    total = (response_sum + decision_sum) / jnp.maximum(total_count, 1.0)
    posterior_entropy = -jnp.sum(
        output.belief
        * jnp.log(jnp.maximum(output.belief, 1.0e-30)),
        axis=-1,
    )
    return LossResult(
        total=total,
        metrics={
            "latent_composite_nll": total,
            "latent_response_nll": response_nll,
            "latent_decision_nll": decision_nll,
            "latent_response_observations": response_count,
            "latent_decision_observations": decision_count,
            "latent_mean_posterior_entropy": jnp.mean(posterior_entropy),
            "latent_decision_top_action_agreement": decision_top_action_agreement,
            "latent_decision_empirical_regret": decision_empirical_regret,
        },
    )


__all__ = [
    "categorical_entropy",
    "categorical_log_probability",
    "generalized_advantage_estimation",
    "latent_composite_loss",
    "ppo_loss",
]
