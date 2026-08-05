"""Structured, partner-visible response labels and losses.

The extractor consumes explicit semantic planes supplied by the fixed Official
environment adapter.  It never guesses channel numbers from a tensor shape.
This makes a changed upstream observation contract fail closed instead of
silently turning the response objective back into world-model reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


PARTNER_POSITION_CLASSES = 26  # 5x5 local view plus not-visible.
PARTNER_DIRECTION_CLASSES = 4
PARTNER_INVENTORY_FACTOR_CLASSES = 4


@dataclass(frozen=True, slots=True)
class PartnerObservationPlanes:
    visibility_channel: int
    direction_channels: tuple[int, int, int, int]
    inventory_channels: tuple[int, ...]
    interaction_channels: tuple[int, ...]

    def validate(self, channel_count: int) -> None:
        indexes = (
            (self.visibility_channel,)
            + self.direction_channels
            + self.inventory_channels
            + self.interaction_channels
        )
        if not self.inventory_channels or not self.interaction_channels:
            raise ValueError("Partner inventory and interaction planes are required.")
        structural = (
            (self.visibility_channel,) + self.direction_channels + self.inventory_channels
        )
        if len(set(structural)) != len(structural):
            raise ValueError("Official partner position/direction/inventory planes overlap.")
        if not set(self.interaction_channels).issubset(set(self.inventory_channels)):
            raise ValueError(
                "Visible interaction change must be derived only from the partner inventory planes."
            )
        if min(indexes) < 0 or max(indexes) >= int(channel_count):
            raise ValueError("Partner response plane lies outside Official observation.")


class PartnerResponseTargets(NamedTuple):
    visibility: Any
    relative_position: Any
    direction: Any
    inventory: Any
    interaction_change: Any
    visible_mask: Any
    event_mask: Any


def official_partner_observation_planes(channel_count: int) -> PartnerObservationPlanes:
    """Return the exact DEFAULT-observation planes from Official commit 5ce1707.

    The pinned encoder concatenates ``self agent`` then ``other agent``.  Each
    agent block is ``position[1], direction[4], inventory[num_ingredients+2]``
    and the complete channel count is ``27 + 4*num_ingredients`` when the
    required successful-delivery plane is enabled.  This function validates
    that contract rather than guessing arbitrary channels from activity.
    """

    channels = int(channel_count)
    remainder = channels - 27
    if remainder < 0 or remainder % 4:
        raise ValueError(
            "Observation channels do not match Official DEFAULT + delivery-indicator layout."
        )
    ingredient_count = remainder // 4
    if ingredient_count <= 0:
        raise ValueError("Official response contract requires at least one ingredient.")
    agent_block = ingredient_count + 7
    other_start = agent_block
    inventory = tuple(range(other_start + 5, other_start + agent_block))
    planes = PartnerObservationPlanes(
        visibility_channel=other_start,
        direction_channels=tuple(range(other_start + 1, other_start + 5)),
        inventory_channels=inventory,
        # The official observation does not expose another agent's action.
        # The only legal visible interaction outcome is its inventory change.
        interaction_channels=inventory,
    )
    planes.validate(channels)
    return planes


class PartnerResponseLosses(NamedTuple):
    total: Any
    visibility: Any
    relative_position: Any
    direction: Any
    inventory: Any
    interaction_change: Any


class ResponsePrediction(NamedTuple):
    """Per-step mixture-likelihood prediction bundle (METHOD_SPEC §2.2/§2.3).

    Every logit tensor carries a trailing ``K`` protocol-component axis; the
    mixture is formed with ``posterior_log_probabilities`` = log pi_t.
    """

    posterior_log_probabilities: Any
    visibility_logit: Any
    relative_position_logits: Any
    direction_logits: Any
    inventory_logits: Any
    interaction_change_logit: Any


def extract_partner_response_targets(
    previous_observation: Any,
    next_observation: Any,
    *,
    planes: PartnerObservationPlanes,
) -> PartnerResponseTargets:
    import jax.numpy as jnp

    previous = jnp.asarray(previous_observation)
    current = jnp.asarray(next_observation)
    if previous.shape != current.shape or current.ndim < 3:
        raise ValueError("Partner response observations have incompatible shapes.")
    height, width, channels = current.shape[-3:]
    if (height, width) != (5, 5):
        raise ValueError("DEPI response contract is registered for a 5x5 Official view.")
    planes.validate(channels)

    previous_visible_plane = previous[..., planes.visibility_channel] > 0
    visible_plane = current[..., planes.visibility_channel] > 0
    previous_flat_visible = previous_visible_plane.reshape(
        previous_visible_plane.shape[:-2] + (height * width,)
    )
    flat_visible = visible_plane.reshape(visible_plane.shape[:-2] + (height * width,))
    previous_visible = jnp.any(previous_flat_visible, axis=-1)
    visible = jnp.any(flat_visible, axis=-1)
    position = jnp.argmax(flat_visible.astype(jnp.int32), axis=-1)
    position = jnp.where(visible, position, height * width)

    direction_planes = current[..., list(planes.direction_channels)]
    direction_scores = jnp.sum(direction_planes * visible_plane[..., None], axis=(-3, -2))
    direction = jnp.argmax(direction_scores, axis=-1)

    inventory_planes = current[..., list(planes.inventory_channels)]
    inventory_scores = jnp.sum(
        inventory_planes * visible_plane[..., None], axis=(-3, -2)
    )
    inventory = jnp.clip(
        jnp.rint(inventory_scores), 0, PARTNER_INVENTORY_FACTOR_CLASSES - 1
    )

    before_interaction = jnp.sum(
        previous[..., list(planes.interaction_channels)]
        * previous_visible_plane[..., None],
        axis=(-3, -2),
    )
    after_interaction = jnp.sum(
        current[..., list(planes.interaction_channels)]
        * visible_plane[..., None],
        axis=(-3, -2),
    )
    visible_inventory_change = jnp.any(
        jnp.rint(before_interaction) != jnp.rint(after_interaction), axis=-1
    )
    visible_inventory_change = previous_visible & visible & visible_inventory_change
    # Visibility transitions belong exclusively to the visibility head.  The
    # event head is evaluated only when inventory is observable at both ends.
    interaction_change = visible_inventory_change
    return PartnerResponseTargets(
        visibility=visible.astype(jnp.float32),
        relative_position=position.astype(jnp.int32),
        direction=direction.astype(jnp.int32),
        inventory=inventory.astype(jnp.int32),
        interaction_change=interaction_change.astype(jnp.float32),
        visible_mask=visible.astype(jnp.float32),
        event_mask=(previous_visible & visible).astype(jnp.float32),
    )


def _categorical_cross_entropy(logits: Any, labels: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    label = jnp.asarray(labels, dtype=jnp.int32)
    return -jnp.take_along_axis(jnn.log_softmax(prediction), label[..., None], axis=-1)[..., 0]


def _binary_cross_entropy(logits: Any, labels: Any) -> Any:
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    target = jnp.asarray(labels, dtype=jnp.float32)
    return jnp.maximum(prediction, 0.0) - prediction * target + jnp.log1p(
        jnp.exp(-jnp.abs(prediction))
    )


def _bernoulli_log_probability(logits: Any, labels: Any) -> Any:
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    target = jnp.asarray(labels, dtype=jnp.float32)
    return -(
        jnp.maximum(prediction, 0.0)
        - prediction * target
        + jnp.log1p(jnp.exp(-jnp.abs(prediction)))
    )


def _component_categorical_log_probability(
    logits: Any, labels: Any, *, factorized: bool = False
) -> Any:
    """Return target log probability while preserving the component axis K."""

    import jax.nn as jnn
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    label = jnp.asarray(labels, dtype=jnp.int32)
    if factorized:
        # logits [..., K, F, C], labels [..., F]
        index = jnp.broadcast_to(
            label[..., None, :, None], prediction.shape[:-1] + (1,)
        )
    else:
        # logits [..., K, C], labels [...]
        index = jnp.broadcast_to(
            label[..., None, None], prediction.shape[:-1] + (1,)
        )
    return jnp.take_along_axis(
        jnn.log_softmax(prediction, axis=-1), index, axis=-1
    )[..., 0]


def _mixture_log_probability(
    posterior_log_probabilities: Any, component_log_probability: Any
) -> Any:
    """log sum_k pi_{t,k} p_k(y) via logsumexp (proper mixture NLL term)."""

    from jax.scipy.special import logsumexp
    import jax.numpy as jnp

    log_pi = jnp.asarray(posterior_log_probabilities, dtype=jnp.float32)
    component = jnp.asarray(component_log_probability, dtype=jnp.float32)
    return logsumexp(log_pi + component, axis=-1)


def _masked_mean(values: Any, mask: Any | None) -> Any:
    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    if mask is None:
        return jnp.mean(array)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(weight * array) / jnp.maximum(jnp.sum(weight), 1.0)


def component_joint_log_probability(
    prediction: ResponsePrediction,
    targets: PartnerResponseTargets,
) -> Any:
    """Log p(y | z=k, H, a) for the one shared latent component.

    Every response head is combined *before* the single mixture
    marginalization.  Consequently one component must jointly explain the
    complete visible response, rather than allowing a different component for
    each head.  Position is conditional on visibility, so the not-visible
    position class is never double-counted.
    """

    import jax.numpy as jnp

    visible = jnp.asarray(targets.visible_mask, dtype=jnp.float32)
    visibility_target = jnp.asarray(targets.visibility, dtype=jnp.float32)[..., None]
    event_target = jnp.asarray(targets.interaction_change, dtype=jnp.float32)[..., None]
    visibility_lp = _bernoulli_log_probability(
        prediction.visibility_logit, visibility_target
    )
    position_lp = _component_categorical_log_probability(
        prediction.relative_position_logits, targets.relative_position
    )
    direction_lp = _component_categorical_log_probability(
        prediction.direction_logits, targets.direction
    )
    inventory_lp = jnp.sum(
        _component_categorical_log_probability(
            prediction.inventory_logits, targets.inventory, factorized=True
        ),
        axis=-1,
    )
    event_lp = _bernoulli_log_probability(
        prediction.interaction_change_logit, event_target
    )
    visible_component = visible[..., None]
    event_valid = jnp.asarray(targets.event_mask, dtype=jnp.float32)[..., None]
    return (
        visibility_lp
        + visible_component * (position_lp + direction_lp + inventory_lp)
        + event_valid * event_lp
    )


def mixture_response_loss(
    prediction: ResponsePrediction,
    targets: PartnerResponseTargets,
    *,
    mask: Any | None = None,
) -> PartnerResponseLosses:
    """Proper mixture negative log-likelihood L_response (METHOD_SPEC §2.3).

    A single component jointly explains all heads and the component is then
    marginalized exactly once.  This is the registered shared-latent mixture
    likelihood, not a sum of independently marginalized head losses.
    """

    import jax.numpy as jnp

    log_pi = prediction.posterior_log_probabilities
    visible = jnp.asarray(targets.visible_mask, dtype=jnp.float32)
    component_logp = component_joint_log_probability(prediction, targets)
    joint_lp = _mixture_log_probability(log_pi, component_logp)
    per_step_nll = -joint_lp
    total = _masked_mean(per_step_nll, mask)

    # Head metrics are descriptive posterior-predictive marginals.  They do
    # not contribute separately to ``total``.
    visibility_lp = _mixture_log_probability(
        log_pi,
        _bernoulli_log_probability(
            prediction.visibility_logit,
            jnp.asarray(targets.visibility, dtype=jnp.float32)[..., None],
        ),
    )
    position_lp = _mixture_log_probability(
        log_pi,
        _component_categorical_log_probability(
            prediction.relative_position_logits, targets.relative_position
        ),
    )
    direction_lp = _mixture_log_probability(
        log_pi,
        _component_categorical_log_probability(
            prediction.direction_logits, targets.direction
        ),
    )
    inventory_lp = _mixture_log_probability(
        log_pi,
        jnp.sum(
            _component_categorical_log_probability(
                prediction.inventory_logits, targets.inventory, factorized=True
            ),
            axis=-1,
        ),
    )
    event_lp = _mixture_log_probability(
        log_pi,
        _bernoulli_log_probability(
            prediction.interaction_change_logit,
            jnp.asarray(targets.interaction_change, dtype=jnp.float32)[..., None],
        ),
    )
    return PartnerResponseLosses(
        total,
        _masked_mean(-visibility_lp, mask),
        _masked_mean(-visible * position_lp, mask),
        _masked_mean(-visible * direction_lp, mask),
        _masked_mean(-visible * inventory_lp, mask),
        _masked_mean(-jnp.asarray(targets.event_mask) * event_lp, mask),
    )


def pairwise_component_response_divergence(
    prediction: ResponsePrediction,
    *,
    mask: Any | None = None,
) -> Any:
    """Permutation-invariant mean pairwise symmetric KL across regimes.

    This is a diagnostic of whether the response decoder actually uses the
    exchangeable component axis.  It does not assign names to components and
    it never enters posterior filtering or deployment decisions.
    """

    import jax
    import jax.numpy as jnp

    def categorical_symmetric_kl(logits: Any) -> Any:
        log_probability = jax.nn.log_softmax(
            jnp.asarray(logits, dtype=jnp.float32), axis=-1
        )
        probability = jnp.exp(log_probability)
        forward = jnp.sum(
            probability[..., :, None, :]
            * (
                log_probability[..., :, None, :]
                - log_probability[..., None, :, :]
            ),
            axis=-1,
        )
        return 0.5 * (forward + jnp.swapaxes(forward, -1, -2))

    def bernoulli_logits(logits: Any) -> Any:
        value = jnp.asarray(logits, dtype=jnp.float32)
        return jnp.stack((jnp.zeros_like(value), value), axis=-1)

    divergences = (
        categorical_symmetric_kl(bernoulli_logits(prediction.visibility_logit)),
        categorical_symmetric_kl(prediction.relative_position_logits),
        categorical_symmetric_kl(prediction.direction_logits),
        jnp.mean(
            categorical_symmetric_kl(
                jnp.swapaxes(prediction.inventory_logits, -3, -2)
            ),
            axis=-3,
        ),
        categorical_symmetric_kl(
            bernoulli_logits(prediction.interaction_change_logit)
        ),
    )
    combined = sum(divergences) / float(len(divergences))
    component_count = combined.shape[-1]
    off_diagonal = 1.0 - jnp.eye(component_count, dtype=jnp.float32)
    lead_weight = jnp.ones(combined.shape[:-2], dtype=jnp.float32)
    if mask is not None:
        lead_weight = jnp.asarray(mask, dtype=jnp.float32)
    weight = lead_weight[..., None, None] * off_diagonal
    return jnp.sum(weight * combined) / jnp.maximum(jnp.sum(weight), 1.0)


__all__ = [
    "PARTNER_DIRECTION_CLASSES",
    "PARTNER_INVENTORY_FACTOR_CLASSES",
    "PARTNER_POSITION_CLASSES",
    "PartnerObservationPlanes",
    "PartnerResponseLosses",
    "PartnerResponseTargets",
    "ResponsePrediction",
    "component_joint_log_probability",
    "extract_partner_response_targets",
    "mixture_response_loss",
    "official_partner_observation_planes",
    "pairwise_component_response_divergence",
]
