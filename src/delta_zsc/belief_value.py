"""Belief-conditioned raw-return action value.

This replaces the component-wise Gaussian return mixture as the quantity the
deployment policy improves against.  Four measurements motivated the change,
all taken on `postfix_matched/seed0` with frozen parameters:

* Holding the component residual branches at zero changed the fitted training
  NLL by 0.082 out of 7.55 (1.1%).  The four component-conditional means were
  therefore not identified by the data: a single shared function explained
  essentially the whole likelihood.
* Each anchor observes one real partner's return vector, never K of them, so
  there is no cross-component paired counterfactual from which to recover the
  decomposition.
* Fitting the mixture NLL to saturation drove training NLL from 0.52 to -7.55
  while training top-action agreement peaked at 0.43 and holdout regret rose.
  The density objective and the action ordering came apart.
* Pairing one decision update to every response update for 2000 steps left the
  ordering metrics without a trend, so the update-frequency imbalance was not
  what held the decomposition back.

What the trajectories *do* identify is the belief-conditioned marginal value:
every rollout step yields ``(x_t, b_t, a_t, r_t, x_{t+1}, b_{t+1})``.  That is
what this module models.

The dueling form keeps the advantage centred under the acting policy, so the
value head absorbs the state's level and the advantage carries only the action
contrast the mirror step consumes.  The ensemble exists to report disagreement:
mirror control needs a spread, not a learned variance that the objective can
shrink for free.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .nn import init_linear, init_mlp, linear, mlp


class BeliefConditionedValue(NamedTuple):
    """Dueling raw-return value under the current posterior.

    ``advantage_mean`` is already centred against the acting policy, so
    ``value_mean[..., None] + advantage_mean`` is the action value and
    ``advantage_mean`` alone is what the mirror improvement ranks.
    ``ensemble_advantage`` carries one row per ensemble member for the
    disagreement term; it is never collapsed into a trainable variance.
    """

    value_mean: Any
    advantage_mean: Any
    ensemble_advantage: Any

    @property
    def action_value(self) -> Any:
        return self.value_mean[..., None] + self.advantage_mean

    def advantage_dispersion(self) -> Any:
        """Per-action ensemble standard deviation of the advantage."""

        import jax.numpy as jnp

        if self.ensemble_advantage.shape[0] < 2:
            return jnp.zeros_like(self.advantage_mean)
        return jnp.std(self.ensemble_advantage, axis=0)


def init_belief_value_params(
    key: Any,
    *,
    task_dim: int,
    instant_dim: int,
    behavior_dim: int,
    component_count: int,
    hidden_dim: int,
    action_count: int,
    ensemble_size: int,
) -> dict[str, Any]:
    import jax

    if int(ensemble_size) < 1:
        raise ValueError("The belief-conditioned critic needs at least one member.")
    inputs = int(task_dim) + int(instant_dim) + int(behavior_dim) + int(component_count)
    keys = jax.random.split(key, 2 + 2 * int(ensemble_size))
    params: dict[str, Any] = {
        "trunk": init_mlp(keys[0], (inputs, int(hidden_dim), int(hidden_dim))),
        "value": init_linear(keys[1], int(hidden_dim), 1, scale=1.0),
    }
    # Independent advantage heads on a shared trunk.  Sharing the trunk keeps
    # the parameter count near the head it replaces; the heads differ only in
    # initialisation, which is what makes their spread informative.
    for member in range(int(ensemble_size)):
        params[f"advantage_{member}"] = init_linear(
            keys[2 + member], int(hidden_dim), int(action_count), scale=0.01
        )
    return params


def belief_value_predict(
    params: dict[str, Any],
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
    belief: Any,
    policy_probabilities: Any | None = None,
) -> BeliefConditionedValue:
    """Evaluate the critic under the posterior the actor is conditioned on.

    ``policy_probabilities`` centres the advantage against the acting policy.
    When omitted the mean over actions is removed instead, which is the uniform
    special case and keeps the head usable before a policy is available.
    """

    import jax.numpy as jnp

    features = jnp.concatenate(
        (
            jnp.asarray(task_features, dtype=jnp.float32),
            jnp.asarray(instant_partner, dtype=jnp.float32),
            jnp.asarray(behavior_features, dtype=jnp.float32),
            jnp.asarray(belief, dtype=jnp.float32),
        ),
        axis=-1,
    )
    hidden = mlp(params["trunk"], features, final_activation=True)
    value = linear(params["value"], hidden)[..., 0]

    members = sorted(name for name in params if name.startswith("advantage_"))
    raw = jnp.stack([linear(params[name], hidden) for name in members], axis=0)
    if policy_probabilities is None:
        baseline = jnp.mean(raw, axis=-1, keepdims=True)
    else:
        weights = jnp.asarray(policy_probabilities, dtype=jnp.float32)
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1.0e-30
        )
        baseline = jnp.sum(raw * weights[None, ...], axis=-1, keepdims=True)
    centered = raw - baseline
    return BeliefConditionedValue(
        value_mean=value,
        advantage_mean=jnp.mean(centered, axis=0),
        ensemble_advantage=centered,
    )


def huber(residual: Any, delta: float = 1.0) -> Any:
    """Bounded loss on value residuals.

    Raw Overcooked returns are dominated by rare +/-20 delivery events; a
    squared residual would let a handful of them set the gradient for every
    state.
    """

    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(residual, dtype=jnp.float32))
    scale = jnp.asarray(delta, dtype=jnp.float32)
    return jnp.where(
        absolute <= scale, 0.5 * absolute ** 2, scale * (absolute - 0.5 * scale)
    )


def td_lambda_targets(
    rewards: Any,
    values: Any,
    dones: Any,
    *,
    gamma: float,
    lambda_: float,
) -> Any:
    """Backward TD(lambda) returns on raw task reward only.

    The PPO critic may keep the annealed shaping term -- it is training task
    competence.  This target must not: the anchor contrasts it is calibrated
    against accumulate raw reward, and mixing the two would leave the two
    decision-side estimands describing different quantities.
    """

    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    value = jnp.asarray(values, dtype=jnp.float32)
    continues = 1.0 - jnp.asarray(dones, dtype=jnp.float32)
    discount = jnp.asarray(gamma, dtype=jnp.float32)
    trace = jnp.asarray(lambda_, dtype=jnp.float32)

    def step(carry: Any, values_at_t: tuple[Any, Any, Any, Any]):
        current_reward, current_value, next_value, keep = values_at_t
        target = current_reward + discount * keep * (
            (1.0 - trace) * next_value + trace * carry
        )
        return target, target

    _, targets = jax.lax.scan(
        step,
        value[-1],
        (reward, value[:-1], value[1:], continues),
        reverse=True,
    )
    return targets


__all__ = [
    "BeliefConditionedValue",
    "belief_value_predict",
    "huber",
    "init_belief_value_params",
    "td_lambda_targets",
]
