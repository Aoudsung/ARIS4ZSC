"""Response and decision objectives for DELTA v5's two parameter owners."""

from __future__ import annotations

from typing import Any, Mapping

from .belief_value import belief_value_predict, huber, td_lambda_targets
from .successor_feature import outcome_encoding, successor_feature_predict
from .contrast import (
    PairwiseContrast,
    contrast_regression_loss,
    exact_tie_fraction,
    kendall_tau_b,
    pairwise_contrasts_from_replicas,
    pairwise_sign_agreement,
    resolvable_pair_statistics,
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



RAW_VALUE_METRIC_NAMES = (
    "raw_value_total",
    "raw_value_state",
    "raw_value_action",
    "raw_value_mean",
    "raw_value_target_mean",
    "raw_advantage_dispersion",
)

CONTRAST_METRIC_NAMES = (
    "contrast_total",
    "contrast_weight",
    "contrast_sign_agreement",
    "contrast_pairs",
    "contrast_resolvable_fraction",
    "contrast_mean_absolute",
    "contrast_mean_standard_error",
)

ANCHOR_ORACLE_METRIC_NAMES = (
    "mirror_same_world_improvement",
    "mirror_same_world_improvement_standard_error",
    "mirror_same_world_win_rate",
    "mirror_realised_kl",
    "anchor_oracle_repeatability",
    "anchor_top_action_agreement",
    "anchor_best_action_regret",
    "anchor_best_action_regret_standard_error",
    "anchor_evaluation_sign_agreement",
    "anchor_evaluation_kendall_tau_b",
    "anchor_evaluation_exact_tie_fraction",
    "anchor_evaluation_pairs",
    "anchor_evaluation_resolvable_fraction",
    "anchor_evaluation_mean_absolute",
    "anchor_evaluation_mean_standard_error",
)

SUCCESSOR_METRIC_NAMES = (
    "successor_feature_total",
    "successor_feature_task",
    "successor_feature_instant",
    "successor_feature_identity_baseline",
    "successor_feature_dispersion",
    "successor_value_total",
    "successor_value_mean",
    "successor_value_realised_gap",
)

DECISION_METRIC_NAMES = (
    RAW_VALUE_METRIC_NAMES
    + CONTRAST_METRIC_NAMES
    + ANCHOR_ORACLE_METRIC_NAMES
    + SUCCESSOR_METRIC_NAMES
)
"""Every decision-side metric key.

The composite must emit all of them on every update, zero-filled when the
channel did not run: an anchor batch arrives on a minority of updates, and a
metrics dict whose keys depend on that would either break accumulation or
silently drop the updates that carry the measurement.
"""


def zero_decision_metrics() -> dict[str, Any]:
    import jax.numpy as jnp

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {name: zero for name in DECISION_METRIC_NAMES}


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
    # Fixed at collection time.  Recomputing the advantage here from the
    # candidate critic -- as this function used to -- means every minibatch and
    # every epoch optimises a different objective, and the value target becomes
    # a regression onto the very network being regressed.
    advantages = jnp.asarray(batch.advantages, dtype=jnp.float32)
    returns = jnp.asarray(batch.returns, dtype=jnp.float32)
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    if bool(model.config.ppo.normalize_advantages):
        # std + eps, matching the Official normalisation exactly.  The
        # previous sqrt(variance + eps) differs whenever the spread is
        # small, which is precisely the regime this task sits in.
        mean = _masked_mean(advantages, mask)
        variance = _masked_mean(jnp.square(advantages - mean), mask)
        advantages = (advantages - mean) / (jnp.sqrt(variance) + 1.0e-8)

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
    old_value = jnp.asarray(batch.old_values, dtype=jnp.float32)[:-1]
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
            # The mean GAE *target*, not an episode return.  Naming it
            # "return" invited exactly the misreading it got.
            "gae_target_mean": _masked_mean(returns, mask),
            "gae_advantage_mean": _masked_mean(advantages, mask),
        },
    )



def raw_task_value_loss(
    model: Any,
    latent_params: Any,
    output: Any,
    batch: RolloutBatch,
    *,
    target_params: Any = None,
    lambda_: float = 0.95,
) -> tuple[Any, Mapping[str, Any]]:
    """TD(lambda) on raw task reward for the belief-conditioned critic.

    This runs on every outer update.  The component decision head it stands in
    for received a non-zero gradient on 28 of 3656 updates, because its channel
    only entered the objective when an anchor batch existed; every rollout step
    carries ``(x_t, b_t, a_t, r_t, x_{t+1}, b_{t+1})`` and can train this one.

    Raw reward only.  The PPO critic keeps the annealed shaping term because it
    is training task competence; this target must match the anchor contrasts,
    which accumulate raw reward, or the two decision-side estimands describe
    different quantities.
    """

    import jax
    import jax.numpy as jnp

    params = latent_params["belief_value"]
    policy = jax.nn.softmax(
        jax.lax.stop_gradient(output.base_policy_logits), axis=-1
    )
    prediction = belief_value_predict(
        params,
        jax.lax.stop_gradient(output.task_features),
        jax.lax.stop_gradient(output.instant_partner),
        output.behavior_features,
        jax.lax.stop_gradient(output.belief),
        policy,
    )
    values = prediction.value_mean
    # Bootstrap from the slow target copy.  The online critic is also the thing
    # the mirror step reads, so letting it chase its own bootstrap couples the
    # deployed policy to its own estimation noise -- and the anchor contrasts
    # arrive too rarely to correct it.
    bootstrap = values
    if target_params is not None:
        bootstrap = belief_value_predict(
            target_params["belief_value"],
            jax.lax.stop_gradient(output.task_features),
            jax.lax.stop_gradient(output.instant_partner),
            jax.lax.stop_gradient(output.behavior_features),
            jax.lax.stop_gradient(output.belief),
            policy,
        ).value_mean
    targets = jax.lax.stop_gradient(
        td_lambda_targets(
            jnp.asarray(batch.rewards, dtype=jnp.float32),
            bootstrap,
            jnp.asarray(batch.dones, dtype=jnp.float32),
            gamma=float(model.config.ppo.gamma),
            lambda_=float(lambda_),
        )
    )
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    state_loss = _masked_mean(huber(values[:-1] - targets), mask)

    # The advantage head is anchored to the realised action's residual return so
    # it does not drift freely between anchor batches.
    actions = jnp.asarray(batch.actions, dtype=jnp.int32)
    taken = jnp.take_along_axis(
        prediction.advantage_mean[:-1], actions[..., None], axis=-1
    )[..., 0]
    residual = jax.lax.stop_gradient(targets - values[:-1])
    action_loss = _masked_mean(huber(taken - residual), mask)
    total = state_loss + action_loss
    return total, {
        "raw_value_total": total,
        "raw_value_state": state_loss,
        "raw_value_action": action_loss,
        "raw_value_mean": _masked_mean(values[:-1], mask),
        "raw_value_target_mean": _masked_mean(targets, mask),
        "raw_advantage_dispersion": _masked_mean(
            jnp.mean(prediction.advantage_dispersion()[:-1], axis=-1), mask
        ),
    }


def pairwise_crn_contrast_loss(
    latent_params: Any,
    output: Any,
    anchors: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Calibrate the critic's action differences against the CRN contrasts.

    Only differences enter, weighted by the inverse measurement variance the
    anchor reports.  A pair the simulator could not resolve therefore trains
    "these two actions are close" with a small weight, instead of being handed
    an ``argmax`` winner decided by index order -- which is what happened on
    every anchor row of the previous target.
    """

    import jax
    import jax.numpy as jnp

    time = anchors.time_indexes
    lane = anchors.lane_indexes
    policy = jax.nn.softmax(
        jax.lax.stop_gradient(output.base_policy_logits[time, lane]), axis=-1
    )
    prediction = belief_value_predict(
        latent_params["belief_value"],
        jax.lax.stop_gradient(output.task_features[time, lane]),
        jax.lax.stop_gradient(output.instant_partner[time, lane]),
        output.behavior_features[time, lane],
        jax.lax.stop_gradient(output.belief[time, lane]),
        policy,
    )
    contrast = PairwiseContrast(
        mean=anchors.contrast_mean,
        standard_error=anchors.contrast_standard_error,
        valid=anchors.contrast_valid,
    )
    loss, weight = contrast_regression_loss(prediction.advantage_mean, contrast)
    statistics = resolvable_pair_statistics(contrast)
    return loss, {
        "contrast_total": loss,
        "contrast_weight": weight,
        "contrast_sign_agreement": pairwise_sign_agreement(
            prediction.advantage_mean, contrast
        ),
        **{f"contrast_{name}": value for name, value in statistics.items()},
    }




def successor_feature_loss(
    latent_params: Any,
    output: Any,
    batch: RolloutBatch,
    probe_target: Any,
    valid: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit the two-step landing state the probe actually produces.

    Supervision is free: every consecutive pair of rollout rows already carries
    the probe action at ``t``, the delayed response observed over
    ``t+1 -> t+2``, and the encoder features at ``t+2``.  The targets are
    stop-gradiented -- the encoder belongs to PPO, and a successor model able to
    reshape it would be making the representation predictable rather than
    useful.
    """

    import jax
    import jax.numpy as jnp

    prediction = successor_feature_predict(
        latent_params["successor_feature"],
        jax.lax.stop_gradient(output.task_features[:-2]),
        jax.lax.stop_gradient(output.instant_partner[:-2]),
        output.behavior_features[:-2],
        jax.lax.stop_gradient(output.belief[:-2]),
        _probe_action_index(batch),
        outcome_encoding(
            probe_target.visibility,
            probe_target.interface_available,
            probe_target.interface_changed,
            probe_target.interface_event,
        ),
    )
    task_target = jax.lax.stop_gradient(output.task_features[2:])
    instant_target = jax.lax.stop_gradient(output.instant_partner[2:])
    task_error = _masked_mean(
        jnp.mean(huber(prediction.task_mean - task_target), axis=-1), valid
    )
    instant_error = _masked_mean(
        jnp.mean(huber(prediction.instant_mean - instant_target), axis=-1), valid
    )
    total = task_error + instant_error
    # The null model is "nothing moves in two steps".  A successor model that
    # cannot beat it is not contributing anything VOI could use, so the same
    # error under the identity prediction is reported alongside.
    identity = _masked_mean(
        jnp.mean(
            huber(jax.lax.stop_gradient(output.task_features[:-2]) - task_target),
            axis=-1,
        ),
        valid,
    ) + _masked_mean(
        jnp.mean(
            huber(
                jax.lax.stop_gradient(output.instant_partner[:-2]) - instant_target
            ),
            axis=-1,
        ),
        valid,
    )
    return total, {
        "successor_feature_total": total,
        "successor_feature_task": task_error,
        "successor_feature_instant": instant_error,
        "successor_feature_identity_baseline": identity,
        "successor_feature_dispersion": _masked_mean(
            prediction.dispersion(), valid
        ),
    }


def successor_belief_value_loss(
    model: Any,
    latent_params: Any,
    output: Any,
    batch: RolloutBatch,
    probe_target: Any,
    valid: Any,
    *,
    lambda_: float = 0.95,
) -> tuple[Any, Mapping[str, Any]]:
    """Value the predicted landing state against what actually happened there.

    Active VOI reads ``Q`` at the successor, so the successor's *value* is the
    quantity that has to be right -- a feature error that changes no action
    value costs nothing, and one that flips an ordering costs everything.  This
    regresses the critic at the predicted landing state onto the same TD(lambda)
    target the critic is fitted to at the real one.

    The critic enters stop-gradiented.  Otherwise the cheapest way to satisfy
    this loss would be to flatten ``Q`` until every successor looks alike, which
    is exactly the degeneracy the whole redesign is trying to leave behind.
    """

    import jax
    import jax.numpy as jnp

    critic_params = jax.lax.stop_gradient(latent_params["belief_value"])
    policy = jax.nn.softmax(
        jax.lax.stop_gradient(output.base_policy_logits), axis=-1
    )
    realised = belief_value_predict(
        critic_params,
        jax.lax.stop_gradient(output.task_features),
        jax.lax.stop_gradient(output.instant_partner),
        output.behavior_features,
        jax.lax.stop_gradient(output.belief),
        policy,
    )
    # A successor at ``t`` lands on state ``t+2``, and TD targets exist for
    # states ``0..T-1`` only -- the final row is the bootstrap state, not an
    # estimate.  The usable window is therefore ``t <= T-3``.
    target = jax.lax.stop_gradient(
        td_lambda_targets(
            jnp.asarray(batch.rewards, dtype=jnp.float32),
            realised.value_mean,
            jnp.asarray(batch.dones, dtype=jnp.float32),
            gamma=float(model.config.ppo.gamma),
            lambda_=float(lambda_),
        )
    )[2:]
    steps = int(target.shape[0])

    successor = successor_feature_predict(
        latent_params["successor_feature"],
        jax.lax.stop_gradient(output.task_features[:-2]),
        jax.lax.stop_gradient(output.instant_partner[:-2]),
        output.behavior_features[:-2],
        jax.lax.stop_gradient(output.belief[:-2]),
        _probe_action_index(batch),
        outcome_encoding(
            probe_target.visibility,
            probe_target.interface_available,
            probe_target.interface_changed,
            probe_target.interface_event,
        ),
    )
    predicted = belief_value_predict(
        critic_params,
        successor.task_mean[:steps],
        successor.instant_mean[:steps],
        output.behavior_features[2 : 2 + steps],
        jax.lax.stop_gradient(output.belief[2 : 2 + steps]),
        policy[2 : 2 + steps],
    )
    window = valid[:steps]
    loss = _masked_mean(huber(predicted.value_mean - target), window)
    return loss, {
        "successor_value_total": loss,
        "successor_value_mean": _masked_mean(predicted.value_mean, window),
        "successor_value_realised_gap": _masked_mean(
            jnp.abs(predicted.value_mean - realised.value_mean[2 : 2 + steps]),
            window,
        ),
    }


def _probe_action_index(batch: RolloutBatch) -> Any:
    """The action played at ``t``, whose delayed response arrives at ``t+2``.

    This must be the action the probe-response target is conditioned on, which
    is ``actions[:-1]`` -- the same slice ``probe_response_predict`` receives.
    Deriving it from ``output.active_voi`` instead would have trained the whole
    successor model on action zero: the latent objective evaluates the sequence
    with ``compute_decision`` off, so that field is identically zero there and
    its argmax is constant.
    """

    import jax.numpy as jnp

    return jnp.asarray(batch.actions[:-1], dtype=jnp.int32)


def _anchor_oracle_diagnostics(
    latent_params: Any, output: Any, anchors: AnchorBatch
) -> Mapping[str, Any]:
    """Audit the critic against the held-out anchor replicas.

    Every quantity here is measured on ``evaluation_*``, which no loss reads.
    ``oracle_repeatability`` is the one that exposed the old target: when the
    fit and evaluation halves of the same anchor disagree about the best
    action, no head can score well on it, and agreement measured against the
    fit half is reporting noise rather than skill.
    """

    import jax
    import jax.numpy as jnp

    time = anchors.time_indexes
    lane = anchors.lane_indexes
    policy = jax.nn.softmax(output.base_policy_logits[time, lane], axis=-1)
    prediction = belief_value_predict(
        latent_params["belief_value"],
        output.task_features[time, lane],
        output.instant_partner[time, lane],
        output.behavior_features[time, lane],
        output.belief[time, lane],
        policy,
    )
    advantage = jax.lax.stop_gradient(prediction.advantage_mean)
    legal = jnp.asarray(anchors.action_mask, dtype=jnp.bool_)
    evaluation = pairwise_contrasts_from_replicas(
        anchors.evaluation_replica_returns_by_action, legal
    )
    evaluation_returns = jnp.asarray(
        anchors.evaluation_returns_by_action, dtype=jnp.float32
    )
    fit_best = jnp.argmax(
        jnp.where(legal, anchors.fit_returns_by_action, -jnp.inf), axis=-1
    )
    oracle = jnp.argmax(jnp.where(legal, evaluation_returns, -jnp.inf), axis=-1)
    selected = jnp.argmax(jnp.where(legal, advantage, -jnp.inf), axis=-1)
    rows = jnp.arange(oracle.shape[0])
    regret = evaluation_returns[rows, oracle] - evaluation_returns[rows, selected]
    # Same-world mirror improvement.  The anchor measured every action's
    # continuation return under one set of random numbers, so the base policy
    # and the mirror policy can be scored against the *same* worlds: the
    # difference is what the adaptation step actually bought, not a comparison
    # of two separate runs.  Everything here uses the evaluation replicas the
    # loss never sees.
    from .mirror_policy import MIRROR_UNCERTAINTY_PENALTY, robust_mirror_policy_logits

    base_logits = jax.lax.stop_gradient(output.base_policy_logits[time, lane])
    mirror_logits, mirror_kl, _ = robust_mirror_policy_logits(
        base_logits,
        advantage,
        jax.lax.stop_gradient(prediction.advantage_dispersion()),
        kl_budget=0.04,
        uncertainty_penalty=MIRROR_UNCERTAINTY_PENALTY,
    )
    legal_float = legal.astype(jnp.float32)

    def masked_policy(logits: Any) -> Any:
        weights = jax.nn.softmax(logits, axis=-1) * legal_float
        return weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-30)

    base_return = jnp.sum(masked_policy(base_logits) * evaluation_returns, axis=-1)
    mirror_return = jnp.sum(masked_policy(mirror_logits) * evaluation_returns, axis=-1)
    improvement = mirror_return - base_return
    rows_float = jnp.maximum(float(improvement.shape[0]), 1.0)
    return {
        "mirror_same_world_improvement": jnp.mean(improvement),
        "mirror_same_world_improvement_standard_error": jnp.std(improvement)
        / jnp.sqrt(rows_float),
        "mirror_same_world_win_rate": jnp.mean(
            (improvement > 0.0).astype(jnp.float32)
        ),
        "mirror_realised_kl": jnp.mean(mirror_kl),
        "anchor_oracle_repeatability": jnp.mean(
            (fit_best == oracle).astype(jnp.float32)
        ),
        "anchor_top_action_agreement": jnp.mean(
            (selected == oracle).astype(jnp.float32)
        ),
        "anchor_best_action_regret": jnp.mean(regret),
        "anchor_best_action_regret_standard_error": jnp.std(regret)
        / jnp.sqrt(jnp.maximum(float(regret.shape[0]), 1.0)),
        "anchor_evaluation_sign_agreement": pairwise_sign_agreement(
            advantage, evaluation
        ),
        "anchor_evaluation_kendall_tau_b": jnp.mean(
            kendall_tau_b(advantage, evaluation_returns, legal)
        ),
        "anchor_evaluation_exact_tie_fraction": exact_tie_fraction(
            evaluation_returns, legal
        ),
        **{
            f"anchor_evaluation_{name}": value
            for name, value in resolvable_pair_statistics(evaluation).items()
        },
    }



def latent_contrast_loss(
    model: Any,
    latent_params: Any,
    base_params: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch,
) -> LossResult:
    """The anchor calibration alone, evaluated after the TD step has committed.

    Sequencing matters here because both objectives write the same critic: the
    contrast is what pins the *ordering* of actions, and it should correct a
    critic that has already absorbed this rollout's returns rather than one
    that has not.
    """

    import jax
    import jax.numpy as jnp

    # The sequence is evaluated at detached latent parameters.  This is exact,
    # not an approximation: with ``compute_decision`` off, the recursion reads
    # the response emissions and the belief filter and never touches
    # ``belief_value``, which is the only subtree this objective writes.  The
    # contrast additionally consumes every feature stop-gradiented.  Passing
    # live parameters here would therefore build a backward pass through a
    # 256-step scan whose gradient is identically zero -- and that backward
    # pass is most of the graph.
    _, output = model.sequence(
        jax.lax.stop_gradient(base_params),
        jax.lax.stop_gradient(latent_params),
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        compute_latent=True,
        compute_decision=False,
        execute_adaptation=False,
    )
    loss, metrics = pairwise_crn_contrast_loss(latent_params, output, anchors)
    # The oracle diagnostics stay out of this objective deliberately.  They
    # contain a forty-eight-iteration mirror solve and a held-out re-scoring of
    # every anchor, none of which carries gradient -- but placing them inside
    # ``value_and_grad`` still put them in the differentiated graph, and the
    # resulting XLA module did not finish backend compilation in forty minutes
    # (against 173 seconds for the same kernel without an anchor batch).  They
    # are computed once, gradient-free, in the final audit.
    return LossResult(total=loss, metrics=metrics)


def latent_composite_loss(
    model: Any,
    latent_params: Any,
    base_params: Any,
    batch: RolloutBatch,
    anchors: AnchorBatch | None,
    *,
    target_latent_params: Any = None,
    include_contrast: bool = True,
) -> LossResult:
    """Response proper scores and direct decision losses for DELTA v5.

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
    successor_loss = jnp.asarray(0.0, dtype=jnp.float32)
    successor_metrics: dict[str, Any] = {}
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
        # Successor channel.  The same two-step window that supervises the
        # delayed response also says where the pair ended up and what it was
        # worth there, so active VOI reads a landing state the data measured
        # rather than the current state wearing its name.
        feature_loss, feature_metrics = successor_feature_loss(
            latent_params, output, batch, probe_target, probe_valid
        )
        value_consistency, value_consistency_metrics = successor_belief_value_loss(
            model, latent_params, output, batch, probe_target, probe_valid
        )
        successor_loss = feature_loss + value_consistency
        successor_metrics = {**feature_metrics, **value_consistency_metrics}

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

    # Decision channel.  Two terms replace the component return mixture: a
    # TD(lambda) fit of the belief-conditioned raw value on every rollout step,
    # and -- when anchors exist -- a precision-weighted fit of the critic's
    # action differences to the measured CRN contrasts.
    #
    # Both gradients reach only ``latent_params["belief_value"]``.  The task
    # features, the instant partner encoding, the posterior and the acting
    # policy all enter stop-gradiented, and the head does not read the component
    # embeddings.  The response channels above and this one are therefore
    # disjoint optimisation problems that share an Adam state, which is why the
    # differing units -- nats against raw return -- need no relative weight and
    # must not be given one.
    value_loss = jnp.asarray(0.0, dtype=jnp.float32)
    contrast_loss = jnp.asarray(0.0, dtype=jnp.float32)
    decision_metrics = zero_decision_metrics()
    trains_decision = model.config.method_variant in {"delta_passive", "delta_active"}
    uses_decision_channel = trains_decision and anchors is not None
    decision_metrics = {**decision_metrics, **successor_metrics}
    if trains_decision:
        value_loss, value_reports = raw_task_value_loss(
            model, latent_params, output, batch, target_params=target_latent_params
        )
        decision_metrics = {**decision_metrics, **value_reports}
    if uses_decision_channel and include_contrast:
        contrast_loss, contrast_reports = pairwise_crn_contrast_loss(
            latent_params, output, anchors
        )
        decision_metrics = {
            **decision_metrics,
            **contrast_reports,
            **_anchor_oracle_diagnostics(latent_params, output, anchors),
        }

    # Two channel-normalized proper scores for the response measurements.
    # Immediate and delayed observations share their channel denominator, so
    # instrumentation frequency cannot become an implicit method weight.
    shared_total_sum = shared_sum + probe_shared_sum
    shared_total_count = shared_count + probe_count
    shared_channel_nll = shared_total_sum / jnp.maximum(shared_total_count, 1.0)
    semantic_total_sum = semantic_sum + probe_semantic_sum
    semantic_total_count = semantic_count + probe_semantic_count
    semantic_channel_nll = semantic_total_sum / jnp.maximum(
        semantic_total_count, 1.0
    )
    total = (
        shared_channel_nll
        + semantic_channel_nll
        + value_loss
        + contrast_loss
        + successor_loss
    )

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
            "latent_shared_response_observations": shared_count,
            "latent_semantic_response_observations": semantic_count,
            "latent_probe_observations": probe_count,
            "latent_probe_semantic_observations": probe_semantic_count,
            "latent_shared_total_observations": shared_total_count,
            "latent_semantic_total_observations": semantic_total_count,
            "latent_mean_posterior_entropy": jnp.mean(posterior_entropy),
            "latent_mean_filter_kl": jnp.mean(filter_kl),
            "latent_component_event_js": event_component_js,
            "latent_probe_component_event_js": probe_event_js,
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
            **decision_metrics,
        },
    )


__all__ = [
    "categorical_entropy",
    "categorical_log_probability",
    "generalized_advantage_estimation",
    "latent_composite_loss",
    "ppo_loss",
]
