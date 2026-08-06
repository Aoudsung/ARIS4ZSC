"""Pinned OvercookedV2 observation semantics used by unified DELTA-ZSC."""

from __future__ import annotations

from typing import Any, NamedTuple

from .types import DirectResponseTarget, ResponseTarget


PARTNER_POSITION_CLASSES = 25
PARTNER_DIRECTION_CLASSES = 4
PARTNER_INVENTORY_FACTOR_CLASSES = 2
INTERFACE_EVENT_CLASSES = 31
OTHER_MULTI_EVENT = 30

STATIC_CHANNEL_START = 20
STATIC_CHANNEL_END = 29
DYNAMIC_CHANNEL_START = 29
DYNAMIC_CHANNEL_END = 34
RECIPE_CHANNEL_START = 34
RECIPE_CHANNEL_END = 39

ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_STAY = 4
ACTION_INTERACT = 5


class LocalTaskState(NamedTuple):
    static: Any
    dynamic: Any
    recipe: Any


class FrameAlignment(NamedTuple):
    available: Any
    previous_y: Any
    previous_x: Any
    overlap: Any
    excluded: Any


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


def decode_local_task_state(observation: Any) -> LocalTaskState:
    """Decode only pinned legal task planes used by the interface extractor."""

    import jax.numpy as jnp

    value = jnp.asarray(observation, dtype=jnp.float32)
    if value.shape[-3:] != (5, 5, 39):
        raise ValueError("Interface responses require the registered 5x5x39 frame.")
    return LocalTaskState(
        static=value[..., STATIC_CHANNEL_START:STATIC_CHANNEL_END],
        dynamic=value[..., DYNAMIC_CHANNEL_START:DYNAMIC_CHANNEL_END],
        recipe=value[..., RECIPE_CHANNEL_START:RECIPE_CHANNEL_END],
    )


def align_egocentric_frames(
    previous: LocalTaskState, current: LocalTaskState, previous_action: Any
) -> FrameAlignment:
    """Choose a successful-move or stationary alignment from static overlap."""

    import jax.numpy as jnp

    action = jnp.asarray(previous_action, dtype=jnp.int32)
    # Current local (y,x) maps to previous local (y+dy,x+dx).
    movement = jnp.asarray(
        ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0), (0, 0)),
        dtype=jnp.int32,
    )[action]
    candidate_offsets = jnp.stack((movement, jnp.zeros_like(movement)), axis=-2)
    moving = action < ACTION_STAY
    candidate_enabled = jnp.stack((moving, jnp.ones_like(moving)), axis=-1)
    y, x = jnp.meshgrid(
        jnp.arange(5, dtype=jnp.int32),
        jnp.arange(5, dtype=jnp.int32),
        indexing="ij",
    )
    py = y + candidate_offsets[..., :, 0, None, None]
    px = x + candidate_offsets[..., :, 1, None, None]
    overlap = (py >= 0) & (py < 5) & (px >= 0) & (px < 5)
    clipped_y = jnp.clip(py, 0, 4)
    clipped_x = jnp.clip(px, 0, 4)
    lead = action.shape
    batch_index = tuple(
        jnp.arange(size).reshape(
            (1,) * axis + (size,) + (1,) * (len(lead) - axis - 1) + (1, 1, 1)
        )
        for axis, size in enumerate(lead)
    )
    previous_static = previous.static[batch_index + (clipped_y, clipped_x)]
    current_static = current.static[..., None, :, :, :]
    equal = jnp.all(previous_static == current_static, axis=-1)
    matches = candidate_enabled & jnp.all((~overlap) | equal, axis=(-2, -1))
    unique = jnp.sum(matches.astype(jnp.int32), axis=-1) == 1
    selected = jnp.argmax(matches.astype(jnp.int32), axis=-1)
    select = selected[..., None, None, None]
    selected_py = jnp.take_along_axis(py, select, axis=-3)[..., 0, :, :]
    selected_px = jnp.take_along_axis(px, select, axis=-3)[..., 0, :, :]
    selected_overlap = jnp.take_along_axis(overlap, select, axis=-3)[..., 0, :, :]

    excluded = jnp.zeros_like(selected_overlap)
    return FrameAlignment(unique, selected_py, selected_px, selected_overlap, excluded)


def _with_interact_exclusion(
    alignment: FrameAlignment, previous_observation: Any, previous_action: Any
) -> FrameAlignment:
    import jax.numpy as jnp

    action = jnp.asarray(previous_action, dtype=jnp.int32)
    previous_value = jnp.asarray(previous_observation, dtype=jnp.float32)
    direction = jnp.argmax(previous_value[..., 2, 2, 1:5], axis=-1)
    # Official direction order is up, down, right, left.
    delta = jnp.asarray(((-1, 0), (1, 0), (0, 1), (0, -1)), dtype=jnp.int32)[direction]
    front_y = 2 + delta[..., 0]
    front_x = 2 + delta[..., 1]
    excluded = (alignment.previous_y == front_y[..., None, None]) & (
        alignment.previous_x == front_x[..., None, None]
    ) & (action == ACTION_INTERACT)[..., None, None]
    return alignment._replace(excluded=excluded)


def _object_code(vector: Any) -> tuple[Any, Any]:
    """Return object index plate/dish/ingredient0..2 and uniqueness."""

    import jax.numpy as jnp

    value = jnp.asarray(vector)
    plate = value[..., 0] > 0.5
    cooked = value[..., 1] > 0.5
    ingredients = value[..., 2:5]
    ingredient_present = ingredients > 0.5
    single_ingredient = jnp.sum(ingredient_present.astype(jnp.int32), axis=-1) == 1
    plate_only = plate & (~cooked) & (~jnp.any(ingredient_present, axis=-1))
    dish = cooked
    ingredient_code = 2 + jnp.argmax(ingredients, axis=-1)
    code = jnp.where(plate_only, 0, jnp.where(dish, 1, ingredient_code))
    valid = plate_only | dish | ((~plate) & (~cooked) & single_ingredient)
    return code.astype(jnp.int32), valid


def extract_interface_target(
    previous_observation: Any,
    observation: Any,
    previous_action: Any,
    cross_episode: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Extract aligned world-interface and independently covered recipe events."""

    import jax.numpy as jnp

    previous = decode_local_task_state(previous_observation)
    current = decode_local_task_state(observation)
    alignment = _with_interact_exclusion(
        align_egocentric_frames(previous, current, previous_action),
        previous_observation,
        previous_action,
    )
    py = jnp.clip(alignment.previous_y, 0, 4)
    px = jnp.clip(alignment.previous_x, 0, 4)
    lead = jnp.asarray(previous_action).shape
    batch_index = tuple(
        jnp.arange(size).reshape(
            (1,) * axis + (size,) + (1,) * (len(lead) - axis - 1) + (1, 1)
        )
        for axis, size in enumerate(lead)
    )
    previous_static = previous.static[batch_index + (py, px)]
    previous_dynamic = previous.dynamic[batch_index + (py, px)]
    previous_recipe = previous.recipe[batch_index + (py, px)]
    overlap = alignment.overlap & (~alignment.excluded)
    counter = previous_static[..., 0] > 0.5
    goal = previous_static[..., 1] > 0.5
    pot = previous_static[..., 2] > 0.5
    facility = jnp.where(counter, 0, jnp.where(pot, 1, 2)).astype(jnp.int32)
    target_cell = overlap & (counter | pot | goal)
    available = alignment.available & jnp.any(target_cell, axis=(-2, -1))
    different = jnp.any(previous_dynamic != current.dynamic, axis=-1) & target_cell
    changed_cells = jnp.sum(different.astype(jnp.int32), axis=(-2, -1))
    changed = changed_cells > 0

    old_empty = ~jnp.any(previous_dynamic > 0.5, axis=-1)
    new_empty = ~jnp.any(current.dynamic > 0.5, axis=-1)
    new_object, new_valid = _object_code(current.dynamic)
    old_object, old_valid = _object_code(previous_dynamic)
    appeared = old_empty & (~new_empty) & new_valid
    disappeared = (~old_empty) & new_empty & old_valid

    delta = current.dynamic - previous_dynamic
    ingredient_delta = delta[..., 2:5]
    one_ingredient_factor = (
        jnp.sum((ingredient_delta != 0).astype(jnp.int32), axis=-1) == 1
    ) & jnp.all(delta[..., :2] == 0, axis=-1) & (
        jnp.max(jnp.abs(ingredient_delta), axis=-1) == 1
    )
    pot_ingredient_object = 2 + jnp.argmax(jnp.abs(ingredient_delta), axis=-1)
    pot_appeared = one_ingredient_factor & (jnp.sum(ingredient_delta, axis=-1) == 1)
    pot_disappeared = one_ingredient_factor & (jnp.sum(ingredient_delta, axis=-1) == -1)
    cooked_appeared = (delta[..., 1] == 1) & jnp.all(
        delta[..., [0, 2, 3, 4]] == 0, axis=-1
    )
    cooked_cleared = (previous_dynamic[..., 1] > 0.5) & new_empty
    pot_valid = pot_appeared | pot_disappeared | cooked_appeared | cooked_cleared
    pot_direction = pot_disappeared | cooked_cleared
    pot_object = jnp.where(cooked_appeared | cooked_cleared, 1, pot_ingredient_object)

    ordinary_valid = appeared | disappeared
    ordinary_direction = disappeared
    ordinary_object = jnp.where(appeared, new_object, old_object)
    cell_valid = jnp.where(pot, pot_valid, ordinary_valid)
    direction = jnp.where(pot, pot_direction, ordinary_direction).astype(jnp.int32)
    object_kind = jnp.where(pot, pot_object, ordinary_object).astype(jnp.int32)
    event_by_cell = facility * 10 + direction * 5 + object_kind
    selected_event = jnp.max(jnp.where(different & cell_valid, event_by_cell, 0), axis=(-2, -1))
    structured = (changed_cells == 1) & jnp.all((~different) | cell_valid, axis=(-2, -1))
    event = jnp.where(structured, selected_event, OTHER_MULTI_EVENT).astype(jnp.int32)

    indicator = (previous_static[..., 3] > 0.5) | (previous_static[..., 4] > 0.5)
    recipe_cells = alignment.overlap & (~alignment.excluded) & indicator
    recipe_mask = alignment.available & jnp.any(recipe_cells, axis=(-2, -1))
    recipe_changed = jnp.any(
        recipe_cells[..., None] & (previous_recipe != current.recipe), axis=(-3, -2, -1)
    )
    cross = jnp.asarray(cross_episode, dtype=jnp.bool_)
    available = available & (~cross)
    recipe_mask = recipe_mask & (~cross)
    return (
        available.astype(jnp.float32),
        (changed & available).astype(jnp.float32),
        event,
        recipe_mask.astype(jnp.float32),
        (recipe_changed & recipe_mask).astype(jnp.float32),
    )


def extract_response_target(
    previous_observation: Any,
    observation: Any,
    previous_action: Any,
    cross_episode: Any,
) -> ResponseTarget:
    import jax.numpy as jnp

    previous_visible, previous_position, _, previous_inventory = _partner_semantics(
        previous_observation
    )
    visible, position, direction, inventory = _partner_semantics(observation)
    event_mask = previous_visible & visible
    inventory_change = jnp.any(inventory != previous_inventory, axis=-1)
    movement = event_mask & (position != previous_position)
    carrying = visible & jnp.any(inventory > 0, axis=-1)
    direct = DirectResponseTarget(
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
    interface = extract_interface_target(
        previous_observation, observation, previous_action, cross_episode
    )
    return ResponseTarget(direct, *interface)


__all__ = [
    "FrameAlignment",
    "INTERFACE_EVENT_CLASSES",
    "LocalTaskState",
    "OTHER_MULTI_EVENT",
    "PARTNER_DIRECTION_CLASSES",
    "PARTNER_INVENTORY_FACTOR_CLASSES",
    "PARTNER_POSITION_CLASSES",
    "align_egocentric_frames",
    "decode_local_task_state",
    "extract_interface_target",
    "extract_response_target",
    "ingredient_count",
    "instantaneous_partner_observation",
    "partner_channel_indexes",
    "partner_visibility",
    "task_only_observation",
]
