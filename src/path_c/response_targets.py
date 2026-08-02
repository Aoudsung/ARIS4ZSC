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
        raise ValueError("r3 response contract is registered for a 5x5 Official view.")
    planes.validate(channels)

    visible_plane = current[..., planes.visibility_channel] > 0
    flat_visible = visible_plane.reshape(visible_plane.shape[:-2] + (height * width,))
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

    before_interaction = previous[..., list(planes.interaction_channels)]
    after_interaction = current[..., list(planes.interaction_channels)]
    interaction_change = jnp.any(before_interaction != after_interaction, axis=(-3, -2, -1))
    interaction_change = jnp.logical_and(interaction_change, visible)
    return PartnerResponseTargets(
        visibility=visible.astype(jnp.float32),
        relative_position=position.astype(jnp.int32),
        direction=direction.astype(jnp.int32),
        inventory=inventory.astype(jnp.int32),
        interaction_change=interaction_change.astype(jnp.float32),
        visible_mask=visible.astype(jnp.float32),
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


def structured_partner_response_loss(
    prediction: Any,
    targets: PartnerResponseTargets,
    *,
    interaction_positive_weight: float = 4.0,
) -> PartnerResponseLosses:
    import jax.numpy as jnp

    visibility = jnp.mean(_binary_cross_entropy(prediction.visibility_logit, targets.visibility))
    position = jnp.mean(
        _categorical_cross_entropy(prediction.relative_position_logits, targets.relative_position)
    )
    visible = jnp.asarray(targets.visible_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(visible), 1.0)
    direction = jnp.sum(
        visible * _categorical_cross_entropy(prediction.direction_logits, targets.direction)
    ) / denominator
    inventory_items = _categorical_cross_entropy(
        prediction.inventory_logits, targets.inventory
    )
    inventory = jnp.sum(visible[..., None] * inventory_items) / jnp.maximum(
        denominator * inventory_items.shape[-1], 1.0
    )
    interaction_weight = 1.0 + (
        float(interaction_positive_weight) - 1.0
    ) * jnp.asarray(targets.interaction_change, dtype=jnp.float32)
    interaction = jnp.sum(
        visible
        * interaction_weight
        * _binary_cross_entropy(prediction.interaction_change_logit, targets.interaction_change)
    ) / denominator
    total = visibility + position + direction + inventory + interaction
    return PartnerResponseLosses(total, visibility, position, direction, inventory, interaction)


def response_auxiliary_objective(
    *,
    model: Any,
    params: Any,
    batch: Any,
) -> tuple[Any, dict[str, Any]]:
    """Selected-action response loss with no task/actor/value/raw-Q gradient."""

    import jax
    import jax.numpy as jnp

    _, prediction = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.actions,
        method=model.response_sequence,
    )
    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1], batch.response_next_observations, planes=planes
    )
    losses = structured_partner_response_loss(prediction, targets)
    _, diagnostic_prediction = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        batch.actions,
        method=model.diagnostic_response_sequence,
    )
    diagnostic_error = (
        diagnostic_prediction.diagnostic_reward_mean - batch.rewards
    )
    diagnostic_absolute = jnp.abs(diagnostic_error)
    diagnostic_reward = jnp.mean(jnp.where(
        diagnostic_absolute <= 1.0,
        0.5 * jnp.square(diagnostic_error),
        diagnostic_absolute - 0.5,
    ))
    diagnostic_done = jnp.mean(_binary_cross_entropy(
        diagnostic_prediction.done_logit, batch.dones.astype(jnp.float32)
    ))
    diagnostic_total = diagnostic_reward + diagnostic_done
    return losses.total + diagnostic_total, {
        "response_total_loss": losses.total,
        "response_visibility_loss": losses.visibility,
        "response_position_loss": losses.relative_position,
        "response_direction_loss": losses.direction,
        "response_inventory_loss": losses.inventory,
        "response_interaction_loss": losses.interaction_change,
        "diagnostic_reward_mae": diagnostic_reward,
        "diagnostic_done_bce": diagnostic_done,
        "diagnostic_total_loss": diagnostic_total,
    }


__all__ = [
    "PARTNER_DIRECTION_CLASSES",
    "PARTNER_INVENTORY_FACTOR_CLASSES",
    "PARTNER_POSITION_CLASSES",
    "PartnerObservationPlanes",
    "PartnerResponseLosses",
    "PartnerResponseTargets",
    "extract_partner_response_targets",
    "official_partner_observation_planes",
    "response_auxiliary_objective",
    "structured_partner_response_loss",
]
