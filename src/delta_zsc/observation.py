"""Pinned OvercookedV2 observation semantics used by unified DELTA-ZSC."""

from __future__ import annotations

from typing import Any, NamedTuple


PARTNER_POSITION_CLASSES = 25
PARTNER_DIRECTION_CLASSES = 4
PARTNER_INVENTORY_FACTOR_CLASSES = 2


class ResponseTarget(NamedTuple):
    visibility: Any
    relative_position: Any
    direction: Any
    inventory: Any
    inventory_change: Any
    visible_mask: Any
    event_mask: Any
    movement: Any
    carrying: Any


def ingredient_count(channel_count: int) -> int:
    channels = int(channel_count)
    remainder = channels - 27
    if remainder < 0 or remainder % 4:
        raise ValueError("Observation channels do not match pinned OvercookedV2.")
    count = remainder // 4
    if count <= 0:
        raise ValueError("Pinned OvercookedV2 requires at least one ingredient.")
    return count


def partner_channel_indexes(channel_count: int) -> tuple[int, ...]:
    ingredients = ingredient_count(channel_count)
    block = ingredients + 7
    return tuple(range(block, 2 * block))


def task_only_observation(observation: Any) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(observation, dtype=jnp.float32)
    channels = jnp.asarray(partner_channel_indexes(value.shape[-1]), dtype=jnp.int32)
    return value.at[..., channels].set(0.0)


def instantaneous_partner_observation(observation: Any) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(observation, dtype=jnp.float32)
    return value[..., list(partner_channel_indexes(value.shape[-1]))]


def _partner_semantics(observation: Any) -> tuple[Any, Any, Any, Any]:
    import jax.numpy as jnp

    value = jnp.asarray(observation, dtype=jnp.float32)
    if tuple(value.shape[-3:-1]) != (5, 5):
        raise ValueError("Unified DELTA is registered for the 5x5 local frame.")
    ingredients = ingredient_count(value.shape[-1])
    block = ingredients + 7
    start = block
    position_plane = value[..., start]
    direction_planes = value[..., start + 1 : start + 5]
    inventory_planes = value[..., start + 5 : start + block]
    position_mass = jnp.sum(position_plane, axis=(-2, -1))
    visible = position_mass > 0.5
    flat_position = position_plane.reshape(position_plane.shape[:-2] + (-1,))
    position = jnp.argmax(flat_position, axis=-1).astype(jnp.int32)
    direction_scores = jnp.sum(direction_planes, axis=(-3, -2))
    direction = jnp.argmax(direction_scores, axis=-1).astype(jnp.int32)
    inventory_scores = jnp.sum(inventory_planes, axis=(-3, -2))
    inventory = (inventory_scores > 0.5).astype(jnp.int32)
    return visible, position, direction, inventory


def partner_visibility(observation: Any) -> Any:
    return _partner_semantics(observation)[0]


def extract_response_target(previous_observation: Any, observation: Any) -> ResponseTarget:
    import jax.numpy as jnp

    previous_visible, previous_position, _, previous_inventory = _partner_semantics(
        previous_observation
    )
    visible, position, direction, inventory = _partner_semantics(observation)
    event_mask = previous_visible & visible
    inventory_change = jnp.any(inventory != previous_inventory, axis=-1)
    movement = event_mask & (position != previous_position)
    carrying = visible & jnp.any(inventory > 0, axis=-1)
    return ResponseTarget(
        visibility=visible.astype(jnp.float32),
        relative_position=position,
        direction=direction,
        inventory=inventory,
        inventory_change=inventory_change.astype(jnp.float32),
        visible_mask=visible.astype(jnp.float32),
        event_mask=event_mask.astype(jnp.float32),
        movement=movement.astype(jnp.float32),
        carrying=carrying.astype(jnp.float32),
    )


__all__ = [
    "PARTNER_DIRECTION_CLASSES",
    "PARTNER_INVENTORY_FACTOR_CLASSES",
    "PARTNER_POSITION_CLASSES",
    "ResponseTarget",
    "extract_response_target",
    "ingredient_count",
    "instantaneous_partner_observation",
    "partner_channel_indexes",
    "partner_visibility",
    "task_only_observation",
]
