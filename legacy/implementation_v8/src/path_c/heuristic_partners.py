"""Heuristic partner family for the §7.3 static wide partner pool.

METHOD_SPEC §7.3 registers two scripted partners as a checkpoint-free
family-disjoint test panel: a greedy courier and a stationary helper.  The
family is held out of training pools and reserved for testing
(``HEURISTIC_FAMILY_TEST_ONLY``), so evaluation can report zero-shot
coordination against behaviour never seen during training.

Both partners implement the standard ``PartnerFunctions`` interface with
``source == 3`` and run ids ``20_000 + member`` (kept disjoint from the
trained static-pool range ``10_000 + member``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

from .runner import PartnerFunctions

HEURISTIC_PARTNER_COUNT = 2
HEURISTIC_FAMILY_TEST_ONLY = True

HEURISTIC_COURIER = 0
HEURISTIC_STATIONARY_HELPER = 1

# Pinned Official action layout (``official_adapter.ACTION_ORDER``):
# right, down, left, up, stay, interact.
ACTION_EAST = 0
ACTION_SOUTH = 1
ACTION_WEST = 2
ACTION_NORTH = 3
ACTION_NOOP = 4
ACTION_INTERACT = 5

# Retained public constant for the observing agent's self plane.  Courier
# targeting uses the pinned *other-agent* plane computed from the full channel
# contract below.
EGO_POSITION_CHANNEL = 0

HEURISTIC_RUN_ID_BASE = 20_000
HEURISTIC_RUN_NAMES = ("heuristic-greedy-courier", "heuristic-stationary-helper")


@dataclass(frozen=True, slots=True)
class HeuristicPanelMember:
    run_id: str
    generation_mechanism: str = "heuristic"


@dataclass(frozen=True, slots=True)
class OfficialHeuristicPolicy:
    """Official-evaluator compatible deterministic heuristic policy."""

    member: int

    def __post_init__(self) -> None:
        if int(self.member) not in {HEURISTIC_COURIER, HEURISTIC_STATIONARY_HELPER}:
            raise ValueError("Unknown heuristic panel member.")

    def init_hstate(self, batch_size: int) -> Any:
        import jax.numpy as jnp

        return jnp.zeros((int(batch_size), 1), dtype=jnp.int32)

    def compute_action(
        self, observation: Any, done: Any, hstate: Any, key: Any
    ) -> tuple[Any, Any]:
        import jax.numpy as jnp

        del done, key
        action = (
            _courier_action(observation)
            if int(self.member) == HEURISTIC_COURIER
            else jnp.asarray(ACTION_NOOP, dtype=jnp.int32)
        )
        return jnp.asarray(action, dtype=jnp.int32), hstate


def official_heuristic_panel() -> tuple[tuple[HeuristicPanelMember, OfficialHeuristicPolicy], ...]:
    """Return the two virtual, checkpoint-free family-disjoint partners."""

    return tuple(
        (
            HeuristicPanelMember(run_id=HEURISTIC_RUN_NAMES[member]),
            OfficialHeuristicPolicy(member=member),
        )
        for member in range(HEURISTIC_PARTNER_COUNT)
    )


class HeuristicPartnerState(NamedTuple):
    member: Any


class HeuristicPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


def _courier_action(observation: Any) -> Any:
    """Move towards the ego agent's grid position; interact when adjacent."""

    import jax.numpy as jnp

    from .task_encoder import official_partner_channel_indexes

    observation_array = jnp.asarray(observation, dtype=jnp.float32)
    other_position_channel = official_partner_channel_indexes(
        observation_array.shape[-1]
    )[0]
    grid = observation_array[..., other_position_channel]
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
    distance = jnp.abs(delta_row) + jnp.abs(delta_col)
    visible = jnp.any(grid > 0.0)
    action = jnp.where(
        distance == 1,
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
    return jnp.where(
        visible,
        action,
        ACTION_NOOP,
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
        del parameters, episode_start
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
            "family_id": jnp.zeros((count,), dtype=jnp.int32),
            "checkpoint_stage": jnp.ones((count,), dtype=jnp.float32),
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
    "HEURISTIC_RUN_NAMES",
    "HEURISTIC_RUN_ID_BASE",
    "HEURISTIC_STATIONARY_HELPER",
    "HeuristicPartnerContext",
    "HeuristicPanelMember",
    "HeuristicPartnerState",
    "OfficialHeuristicPolicy",
    "make_heuristic_partner_functions",
    "official_heuristic_panel",
]
