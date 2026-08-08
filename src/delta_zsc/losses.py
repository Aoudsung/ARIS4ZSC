"""Proper objectives for DELTA v4's two parameter owners."""

from __future__ import annotations

from typing import Any

from .decision_model import (
    decision_component_log_probability,
    decision_predict,
    successor_decision_predict,
)
from .observation import extract_probe_response_target, extract_response_target
from .response_model import (
    probe_response_predict,
    probe_response_semantic_component_log_probability,
    probe_response_shared_factor_log_probabilities,
    probe_response_shared_log_probability,
    response_predict,
    response_semantic_component_log_probability,
    response_semantic_factor_log_probabilities,
    response_shared_factor_log_probabilities,
    response_shared_log_probability,
)
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


def _masked_nll(log_probability: Any, mask: Any) -> tuple[Any, Any, Any]:
    import jax.numpy as jnp

    weight = jnp.asarray(mask, dtype=jnp.float32)
    count = jnp.sum(weight)
    total = -jnp.sum(weight * jnp.asarray(log_probability, dtype=jnp.float32))
    return total / jnp.maximum(count, 1.0), total, count


def _mixture_log_probability(belief: Any, component_logp: Any) -> Any:
    import jax.numpy as jnp
    import jax.scipy as jsp

    probability = jnp.asarray(belief, dtype=jnp.float32)
    probability = probability / jnp.maximum(
        jnp.sum(probability, axis=-1, keepdims=True), 1.0e-30
    )
    return jsp.special.logsumexp(
        jnp.log(jnp.maximum(probability, 1.0e-30))
        + jnp.asarray(component_logp, dtype=jnp.float32),
        axis=-1,
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
        # Replay the posteriors the behaviour policy was conditioned on.  The
        # latent update commits before this one, so recomputing them here would
        # evaluate the ratio against a policy that never generated the data.
        beliefs=batch.beliefs,
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
    approximate_kl = 0.5 * _masked_mean(
        jnp.square(new_logp - batch.old_log_probabilities), mask
    )
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
    """Channel-normalized proper score for the unified v4 latent model.

    Each independently sampled measurement channel contributes its own mean
    negative log probability with a fixed coefficient of one.  Consequently
    the method is invariant to rollout length and anchor instrumentation rate;
    no tunable auxiliary-loss weight is introduced.
    """

    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    stopped_base = jax.lax.stop_gradient(base_params)
    _, output = model.sequence(
        stopped_base,
        latent_params,
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        compute_latent=True,
        compute_decision=False,
        execute_adaptation=False,
    )
    response_mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    target = extract_response_target(
        batch.observations[:-1],
        batch.response_next_observations,
        batch.actions,
        jnp.zeros_like(batch.dones, dtype=jnp.bool_),
    )
    prediction = response_predict(
        latent_params["response"],
        latent_params["component_embeddings"],
        batch.observations[:-1],
        output.behavior_features[:-1],
        batch.actions,
    )
    shared_logp = response_shared_log_probability(prediction, target)
    semantic_component_logp = response_semantic_component_log_probability(
        prediction, target
    )
    semantic_logp = _mixture_log_probability(
        output.belief[:-1], semantic_component_logp
    )
    shared_nll, shared_sum, shared_count = _masked_nll(shared_logp, response_mask)
    semantic_mask = response_mask * jnp.maximum(
        jnp.asarray(target.direct.visible_mask, dtype=jnp.float32),
        jnp.asarray(target.interface_available, dtype=jnp.float32)
        * jnp.asarray(target.interface_changed, dtype=jnp.float32),
    )
    semantic_nll, semantic_sum, semantic_count = _masked_nll(semantic_logp, semantic_mask)

    # Delayed probe-response channel: t's probe is supervised by the legal
    # transition o[t+1] -> o[t+2] under the second ego action.
    probe_shared_nll = jnp.asarray(0.0, dtype=jnp.float32)
    probe_semantic_nll = jnp.asarray(0.0, dtype=jnp.float32)
    probe_count = jnp.asarray(0.0, dtype=jnp.float32)
    probe_shared_sum = jnp.asarray(0.0, dtype=jnp.float32)
    probe_semantic_sum = jnp.asarray(0.0, dtype=jnp.float32)
    probe_semantic_count = jnp.asarray(0.0, dtype=jnp.float32)
    probe_event_js = jnp.asarray(0.0, dtype=jnp.float32)
    uses_probe_channel = (
        model.config.method_variant == "delta_active" and batch.actions.shape[0] >= 2
    )
    if uses_probe_channel:
        invalid = jnp.asarray(batch.dones[:-1], dtype=jnp.bool_) | jnp.asarray(
            batch.dones[1:], dtype=jnp.bool_
        )
        probe_target = extract_probe_response_target(
            batch.response_next_observations[:-1],
            batch.response_next_observations[1:],
            batch.actions[1:],
            invalid,
        )
        probe_prediction = probe_response_predict(
            latent_params["probe_response"],
            latent_params["component_embeddings"],
            batch.observations[:-2],
            output.behavior_features[:-2],
            batch.actions[:-1],
        )
        probe_valid = (
            response_mask[:-1]
            * response_mask[1:]
            * jnp.asarray(probe_target.valid_mask, dtype=jnp.float32)
        )
        probe_shared_logp = probe_response_shared_log_probability(
            probe_prediction, probe_target
        )
        probe_shared_nll, probe_shared_sum, probe_count = _masked_nll(
            probe_shared_logp, probe_valid
        )
        probe_semantic_component = (
            probe_response_semantic_component_log_probability(
                probe_prediction, probe_target
            )
        )
        probe_semantic_logp = _mixture_log_probability(
            output.belief[:-2], probe_semantic_component
        )
        probe_semantic_mask = (
            probe_valid
            * jnp.asarray(probe_target.interface_available, dtype=jnp.float32)
            * jnp.asarray(probe_target.interface_changed, dtype=jnp.float32)
        )
        probe_semantic_nll, probe_semantic_sum, probe_semantic_count = _masked_nll(
            probe_semantic_logp, probe_semantic_mask
        )
        probability = jnn.softmax(probe_prediction.interface_event_logits, axis=-1)
        mean_probability = jnp.mean(probability, axis=-2, keepdims=True)
        probe_event_js = jnp.mean(
            jnp.sum(
                probability
                * (
                    jnp.log(jnp.maximum(probability, 1.0e-30))
                    - jnp.log(jnp.maximum(mean_probability, 1.0e-30))
                ),
                axis=-1,
            )
        )

    decision_nll = jnp.asarray(0.0, dtype=jnp.float32)
    decision_sum = jnp.asarray(0.0, dtype=jnp.float32)
    decision_count = jnp.asarray(0.0, dtype=jnp.float32)
    decision_top_action_agreement = jnp.asarray(0.0, dtype=jnp.float32)
    decision_empirical_regret = jnp.asarray(0.0, dtype=jnp.float32)
    decision_component_disagreement = jnp.asarray(0.0, dtype=jnp.float32)
    successor_decision_nll = jnp.asarray(0.0, dtype=jnp.float32)
    successor_decision_sum = jnp.asarray(0.0, dtype=jnp.float32)
    successor_decision_count = jnp.asarray(0.0, dtype=jnp.float32)
    successor_top_action_agreement = jnp.asarray(0.0, dtype=jnp.float32)
    successor_empirical_regret = jnp.asarray(0.0, dtype=jnp.float32)
    uses_decision_channel = (
        anchors is not None
        and model.config.method_variant in {"delta_passive", "delta_active"}
    )
    if uses_decision_channel:
        time = anchors.time_indexes
        lane = anchors.lane_indexes
        current_prediction = decision_predict(
            latent_params["decision"],
            latent_params["component_embeddings"],
            jax.lax.stop_gradient(output.task_features[time, lane]),
            jax.lax.stop_gradient(output.instant_partner[time, lane]),
            output.behavior_features[time, lane],
        )
        current_component_logp = decision_component_log_probability(
            current_prediction,
            anchors.fit_returns_by_action,
            anchors.measurement_covariances,
            anchors.action_mask,
        )
        anchor_belief = output.belief[time, lane]
        current_logp = _mixture_log_probability(anchor_belief, current_component_logp)
        current_valid = jnp.all(anchors.action_mask, axis=-1).astype(jnp.float32)
        decision_nll, decision_sum, decision_count = _masked_nll(
            current_logp, current_valid
        )
        expected = jnp.sum(
            anchor_belief[..., :, None] * current_prediction.means, axis=-2
        )
        selected = jnp.argmax(expected, axis=-1)
        oracle = jnp.argmax(anchors.evaluation_returns_by_action, axis=-1)
        rows = jnp.arange(selected.shape[0])
        decision_top_action_agreement = _masked_mean(selected == oracle, current_valid)
        decision_empirical_regret = _masked_mean(
            anchors.evaluation_returns_by_action[rows, oracle]
            - anchors.evaluation_returns_by_action[rows, selected],
            current_valid,
        )
        component_top = jnp.argmax(current_prediction.means, axis=-1)
        decision_component_disagreement = jnp.mean(
            (component_top[..., :, None] != component_top[..., None, :]).astype(
                jnp.float32
            )
        )

        if model.config.method_variant == "delta_active":
            action_count = int(anchors.probe_fit_returns_by_action.shape[-2])
            probe_actions = jnp.broadcast_to(
                jnp.arange(action_count, dtype=jnp.int32),
                (time.shape[0], action_count),
            )
            successor_prediction = successor_decision_predict(
                latent_params["decision"],
                latent_params["component_embeddings"],
                jax.lax.stop_gradient(output.task_features[time, lane]),
                jax.lax.stop_gradient(output.instant_partner[time, lane]),
                output.behavior_features[time, lane],
                probe_actions,
            )
            successor_component_logp = decision_component_log_probability(
                successor_prediction,
                anchors.probe_fit_returns_by_action,
                anchors.probe_measurement_covariances,
                anchors.probe_action_mask,
            )
            successor_belief = jnp.broadcast_to(
                anchor_belief[:, None, :], successor_component_logp.shape
            )
            successor_logp = _mixture_log_probability(
                successor_belief, successor_component_logp
            )
            successor_valid = jnp.all(
                anchors.probe_action_mask, axis=-1
            ).astype(jnp.float32)
            (
                successor_decision_nll,
                successor_decision_sum,
                successor_decision_count,
            ) = _masked_nll(successor_logp, successor_valid)
            successor_expected = jnp.sum(
                anchor_belief[:, None, :, None] * successor_prediction.means,
                axis=-2,
            )
            successor_selected = jnp.argmax(successor_expected, axis=-1)
            successor_oracle = jnp.argmax(
                anchors.probe_evaluation_returns_by_action, axis=-1
            )
            probe_rows = jnp.arange(successor_selected.shape[0])[:, None]
            probe_indexes = jnp.arange(action_count)[None, :]
            successor_top_action_agreement = _masked_mean(
                successor_selected == successor_oracle, successor_valid
            )
            successor_empirical_regret = _masked_mean(
                anchors.probe_evaluation_returns_by_action[
                    probe_rows, probe_indexes, successor_oracle
                ]
                - anchors.probe_evaluation_returns_by_action[
                    probe_rows, probe_indexes, successor_selected
                ],
                successor_valid,
            )

    # Three channel-normalized proper scores. Immediate and delayed response
    # measurements share their channel denominator; current and successor CRN
    # observations share the decision denominator. Instrumentation frequency
    # therefore cannot become an implicit method weight.
    shared_total_sum = shared_sum + probe_shared_sum
    shared_total_count = shared_count + probe_count
    shared_channel_nll = shared_total_sum / jnp.maximum(shared_total_count, 1.0)
    semantic_total_sum = semantic_sum + probe_semantic_sum
    semantic_total_count = semantic_count + probe_semantic_count
    semantic_channel_nll = semantic_total_sum / jnp.maximum(
        semantic_total_count, 1.0
    )
    decision_total_sum = decision_sum + successor_decision_sum
    decision_total_count = decision_count + successor_decision_count
    decision_channel_nll = decision_total_sum / jnp.maximum(
        decision_total_count, 1.0
    )
    total = shared_channel_nll + semantic_channel_nll
    if uses_decision_channel:
        total = total + decision_channel_nll

    posterior = jnp.asarray(output.belief, dtype=jnp.float32)
    prior = jnp.asarray(output.predictive_belief, dtype=jnp.float32)
    posterior_entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    filter_kl = jnp.sum(
        posterior
        * (
            jnp.log(jnp.maximum(posterior, 1.0e-30))
            - jnp.log(jnp.maximum(prior, 1.0e-30))
        ),
        axis=-1,
    )
    event_probability = jnn.softmax(prediction.interface_event_logits, axis=-1)
    mean_event_probability = jnp.mean(event_probability, axis=-2, keepdims=True)
    event_component_js = jnp.mean(
        jnp.sum(
            event_probability
            * (
                jnp.log(jnp.maximum(event_probability, 1.0e-30))
                - jnp.log(jnp.maximum(mean_event_probability, 1.0e-30))
            ),
            axis=-1,
        )
    )

    factor_metrics: dict[str, Any] = {}
    shared_masks = {
        "visibility": jnp.ones_like(response_mask),
        "inventory_change": target.direct.event_mask,
        "interface_availability": jnp.ones_like(response_mask),
        "interface_change": target.interface_available,
        "recipe_change": target.recipe_mask,
    }
    for name, logp in response_shared_factor_log_probabilities(
        prediction, target
    ).items():
        mask = response_mask * jnp.asarray(shared_masks[name], dtype=jnp.float32)
        nll, _, count = _masked_nll(logp, mask)
        factor_metrics[f"latent_shared_{name}_nll"] = nll
        factor_metrics[f"latent_shared_{name}_count"] = count
    semantic_masks = {
        "position": target.direct.visible_mask,
        "direction": target.direct.visible_mask,
        "inventory": target.direct.visible_mask,
        "interface_event": target.interface_available * target.interface_changed,
    }
    for name, component_logp in response_semantic_factor_log_probabilities(
        prediction, target
    ).items():
        mask = response_mask * jnp.asarray(semantic_masks[name], dtype=jnp.float32)
        nll, _, count = _masked_nll(
            _mixture_log_probability(output.belief[:-1], component_logp), mask
        )
        factor_metrics[f"latent_semantic_{name}_nll"] = nll
        factor_metrics[f"latent_semantic_{name}_count"] = count

    return LossResult(
        total=total,
        metrics={
            "latent_composite_nll": total,
            "latent_shared_response_nll": shared_nll,
            "latent_semantic_response_nll": semantic_nll,
            "latent_probe_shared_nll": probe_shared_nll,
            "latent_probe_semantic_nll": probe_semantic_nll,
            "latent_shared_total_nll": shared_channel_nll,
            "latent_semantic_total_nll": semantic_channel_nll,
            "latent_response_nll": shared_nll + semantic_nll,
            "latent_decision_nll": decision_nll,
            "latent_successor_decision_nll": successor_decision_nll,
            "latent_decision_total_nll": decision_channel_nll,
            "latent_shared_response_observations": shared_count,
            "latent_semantic_response_observations": semantic_count,
            "latent_probe_observations": probe_count,
            "latent_probe_semantic_observations": probe_semantic_count,
            "latent_decision_observations": decision_count,
            "latent_successor_decision_observations": successor_decision_count,
            "latent_shared_total_observations": shared_total_count,
            "latent_semantic_total_observations": semantic_total_count,
            "latent_decision_total_observations": decision_total_count,
            "latent_mean_posterior_entropy": jnp.mean(posterior_entropy),
            "latent_mean_filter_kl": jnp.mean(filter_kl),
            "latent_component_event_js": event_component_js,
            "latent_probe_component_event_js": probe_event_js,
            "latent_decision_top_action_agreement": decision_top_action_agreement,
            "latent_decision_empirical_regret": decision_empirical_regret,
            "latent_decision_component_disagreement": decision_component_disagreement,
            "latent_successor_top_action_agreement": successor_top_action_agreement,
            "latent_successor_empirical_regret": successor_empirical_regret,
            "latent_interface_coverage_rate": _masked_mean(
                target.interface_available, response_mask
            ),
            "latent_interface_change_rate": _masked_mean(
                target.interface_changed,
                response_mask * target.interface_available,
            ),
            "latent_interface_other_multi_rate": _masked_mean(
                (target.interface_event == 30).astype(jnp.float32),
                response_mask
                * target.interface_available
                * target.interface_changed,
            ),
            "latent_recipe_coverage_rate": _masked_mean(
                target.recipe_mask, response_mask
            ),
            "latent_visibility_positive_rate": _masked_mean(
                target.direct.visibility, response_mask
            ),
            "latent_inventory_change_positive_rate": _masked_mean(
                target.direct.inventory_change,
                response_mask * target.direct.event_mask,
            ),
            "latent_recipe_change_positive_rate": _masked_mean(
                target.recipe_changed, response_mask * target.recipe_mask
            ),
            **factor_metrics,
        },
    )


__all__ = [
    "categorical_entropy",
    "categorical_log_probability",
    "generalized_advantage_estimation",
    "latent_composite_loss",
    "ppo_loss",
]
