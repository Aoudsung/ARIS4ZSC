"""Immutable PyTree-compatible records for CETR-ZSC.

The deployable state is exactly one GRU carry and the episode-boundary flag.
Training records hold only completed whole episodes: the training estimand is
the raw undiscounted episodic return, so no cross-rollout continuation state
exists.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class AdamState(NamedTuple):
    count: Any
    first_moment: Any
    second_moment: Any


class PolicyState(NamedTuple):
    """Complete deployable recurrent state.

    The actor reads only its legal local observation and this carry.  There is
    no partner estimate, no privileged history, and no training-only field.
    """

    carry: Any
    episode_start: Any


class EpisodeBatch(NamedTuple):
    """One update's completed whole episodes, time-major ``[T, lanes]``.

    Every lane runs exactly one 400-step episode from a fresh reset, so the
    per-lane sum of ``rewards`` is the complete undiscounted episodic return
    the method is defined on.  Lanes ``[0, lanes//2)`` are self-play: both
    agents execute the current policy with independent recurrent carries, and
    the ``sp_other_*`` arrays hold the second agent's stream.  Lanes
    ``[lanes//2, lanes)`` are external: the ego executes the current policy
    against one frozen parent checkpoint.

    ``lane_parent`` is the development-support parent index for external lanes
    and -1 for self-play lanes.  ``lane_fold`` is the deterministic A/B split
    used by cross-fitting.  ``lane_weight`` is filled by
    ``training.compute_tail_weights`` after collection and is 1 for self-play
    lanes.

    ``old_values`` and ``sp_other_old_values`` are the collection-time value
    baseline outputs.  ``value_targets`` and ``sp_other_value_targets`` are
    filled by ``runner.collect_episodes`` from complete return-to-go values.
    The two advantage arrays are filled there before any lane slicing; training
    then normalizes both arrays once on the complete batch and holds them fixed
    across every PPO epoch and minibatch.
    """

    observations: Any
    actions: Any
    old_log_probabilities: Any
    old_values: Any
    rewards: Any
    dones: Any
    value_targets: Any
    sp_other_value_targets: Any
    advantages: Any
    sp_other_advantages: Any
    episode_starts: Any
    sp_other_observations: Any
    sp_other_actions: Any
    sp_other_old_log_probabilities: Any
    sp_other_old_values: Any
    lane_stream: Any
    lane_parent: Any
    lane_fold: Any
    lane_weight: Any
    episode_return: Any
    ego_roles: Any
    member_index: Any


class TrainState(NamedTuple):
    """The complete resumable training transaction.

    ``dual_lambda`` is the adaptive SP-constraint multiplier.  It is part of
    the state, not a derived quantity: resuming must continue the same
    primal-dual process rather than restarting the dual variable at zero.
    """

    params: Any
    optimizer_state: AdamState
    dual_lambda: Any
    random_key: Any
    update_count: Any
    effective_environment_steps: Any
    resource_ledger: Mapping[str, Any]


class LossResult(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


__all__ = [
    "AdamState",
    "EpisodeBatch",
    "LossResult",
    "PolicyState",
    "TrainState",
]
