"""Heuristic partner family for the §7.3 static wide partner pool.

METHOD_SPEC §7.3 registers two scripted partners alongside the 60 trained
checkpoints: a greedy courier and a stationary helper.  The family is held
out of training pools and reserved for testing
(``HEURISTIC_FAMILY_TEST_ONLY``), so evaluation can report zero-shot
coordination against behaviour never seen during training.

Both partners implement the standard ``PartnerFunctions`` interface with
``source == 3`` and run ids ``20_000 + member`` (kept disjoint from the
static-pool range ``10_000 + member`` and the generator ranges ``-1``/``-2``).
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .runner import PartnerFunctions

HEURISTIC_PARTNER_COUNT = 2
HEURISTIC_FAMILY_TEST_ONLY = True

HEURISTIC_COURIER = 0
HEURISTIC_STATIONARY_HELPER = 1

# Official action layout: 0 noop, 1 north, 2 south, 3 west, 4 east, 5 interact.
ACTION_NOOP = 0
ACTION_NORTH = 1
ACTION_SOUTH = 2
ACTION_WEST = 3
ACTION_EAST = 4
ACTION_INTERACT = 5

# Pinned encoder layout (response_targets.official_partner_observation_planes):
# each agent block starts with a one-hot position plane; the *self* agent of a
# partner observation occupies channel 0.
EGO_POSITION_CHANNEL = 0

HEURISTIC_RUN_ID_BASE = 20_000


class HeuristicPartnerState(NamedTuple):
    member: Any


class HeuristicPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


def _courier_action(observation: Any) -> Any:
    """Move towards the ego agent's grid position; interact when adjacent."""

    import jax.numpy as jnp

    grid = jnp.asarray(observation, dtype=jnp.float32)[..., EGO_POSITION_CHANNEL]
    flat = grid.reshape((-1,))
    target = jnp.argmax(flat)
    rows, cols = grid.shape[-2], grid.shape[-1]
    target_row = target // cols
    target_col = target % cols
    # The partner's own cell is the grid centre of its local 5x5 view; the
    # courier closes the offset one axis per step (axis-priority greedy).
    centre_row = jnp.asarray(rows // 2, dtype=jnp.int32)
    centre_col = jnp.asarray(cols // 2, dtype=jnp.int32)
    delta_row = target_row - centre_row
    delta_col = target_col - centre_col
    adjacent = (jnp.abs(delta_row) + jnp.abs(delta_col)) <= 1
    return jnp.where(
        adjacent,
        ACTION_INTERACT,
        jnp.where(
            delta_row < 0,
            ACTION_NORTH,
            jnp.where(
                delta_row > 0,
                ACTION_SOUTH,
                jnp.where(delta_col < 0, ACTION_WEST, ACTION_EAST),
            ),
        ),
    )


def make_heuristic_partner_functions(*, action_count: int = 6) -> PartnerFunctions:
    """Scripted courier/helper partners over the ``PartnerFunctions`` surface."""

    import jax
    import jax.numpy as jnp

    if int(action_count) != 6:
        raise ValueError("Heuristic partners are registered for the 6-action layout.")

    def initial_state(batch_size: int, key: Any) -> HeuristicPartnerState:
        member = jax.random.randint(
            key, (int(batch_size),), 0, HEURISTIC_PARTNER_COUNT
        )
        return HeuristicPartnerState(member=member)

    def step(
        parameters: Any,
        state: HeuristicPartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ):
        del parameters, episode_start, keys
        observation_array = jnp.asarray(observations, dtype=jnp.float32)
        courier = jax.vmap(_courier_action)(observation_array)
        stationary = jnp.full(
            (observation_array.shape[0],), ACTION_NOOP, dtype=jnp.int32
        )
        action = jnp.where(state.member == HEURISTIC_COURIER, courier, stationary)
        context = HeuristicPartnerContext(
            member=state.member,
            reset_keys=jax.vmap(lambda key: jax.random.fold_in(key, 3))(keys),
        )
        return (
            action,
            state,
            context,
            jnp.zeros((observation_array.shape[0],), dtype=jnp.float32),
        )

    def observe(
        parameters: Any,
        state: HeuristicPartnerState,
        context: HeuristicPartnerContext,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> HeuristicPartnerState:
        del parameters, observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        fresh_members = jax.vmap(
            lambda key: jax.random.randint(key, (), 0, HEURISTIC_PARTNER_COUNT)
        )(context.reset_keys)
        member = jnp.where(done, fresh_members, state.member)
        return HeuristicPartnerState(member=member)

    def run_id(
        parameters: Any,
        state: HeuristicPartnerState,
        context: Any,
    ) -> Any:
        del parameters, context
        return (HEURISTIC_RUN_ID_BASE + state.member).astype(jnp.int32)

    def diagnostics(
        parameters: Any,
        state: HeuristicPartnerState,
        context: HeuristicPartnerContext,
    ) -> Mapping[str, Any]:
        del parameters, state
        count = context.member.shape[0]
        return {
            "source": jnp.full((count,), 3, dtype=jnp.int32),
            "member": context.member,
            "code": jnp.asarray(context.member, dtype=jnp.float32)[:, None],
            "generator_logits": jnp.zeros((count, 6), dtype=jnp.float32),
            "generator_value": jnp.zeros((count,), dtype=jnp.float32),
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


__all__ = [
    "ACTION_EAST",
    "ACTION_INTERACT",
    "ACTION_NOOP",
    "ACTION_NORTH",
    "ACTION_SOUTH",
    "ACTION_WEST",
    "EGO_POSITION_CHANNEL",
    "HEURISTIC_COURIER",
    "HEURISTIC_FAMILY_TEST_ONLY",
    "HEURISTIC_PARTNER_COUNT",
    "HEURISTIC_RUN_ID_BASE",
    "HEURISTIC_STATIONARY_HELPER",
    "HeuristicPartnerContext",
    "HeuristicPartnerState",
    "make_heuristic_partner_functions",
]
