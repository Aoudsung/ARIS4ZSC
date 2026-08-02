"""Target-policy epochs bind every raw-return value label to one policy."""

from __future__ import annotations

from typing import Any, NamedTuple


TARGET_POLICY_EPOCH_UPDATES = 16


class TargetPolicyEpoch(NamedTuple):
    epoch_id: int
    actor_fingerprint: str
    belief_fingerprint: str
    raw_q_fingerprint: str
    target_params: Any
    started_update: int


def epoch_id_for_update(
    update_number: int, *, updates_per_epoch: int = TARGET_POLICY_EPOCH_UPDATES
) -> int:
    if update_number <= 0:
        raise ValueError("Outer updates are one-indexed.")
    if updates_per_epoch <= 0:
        raise ValueError("Target-policy epoch length must be positive.")
    return (int(update_number) - 1) // int(updates_per_epoch)


def validate_epoch_alignment(
    epoch: TargetPolicyEpoch,
    *,
    update_number: int,
    updates_per_epoch: int = TARGET_POLICY_EPOCH_UPDATES,
) -> None:
    expected = epoch_id_for_update(
        update_number, updates_per_epoch=updates_per_epoch
    )
    if int(epoch.epoch_id) != expected:
        raise ValueError(
            f"Target policy epoch mismatch: state={epoch.epoch_id}, expected={expected}."
        )
    if int(epoch.started_update) != expected * int(updates_per_epoch) + 1:
        raise ValueError("Target policy epoch start update is inconsistent.")
    for value in (
        epoch.actor_fingerprint,
        epoch.belief_fingerprint,
        epoch.raw_q_fingerprint,
    ):
        if not value:
            raise ValueError("Every target-policy component needs a fingerprint.")


def validate_label_policy(
    *,
    label_epoch_id: int,
    label_policy_fingerprint: str,
    current_epoch: TargetPolicyEpoch,
) -> None:
    if int(label_epoch_id) != int(current_epoch.epoch_id):
        raise ValueError("Old target-policy epoch labels cannot train current raw-Q.")
    expected = ":".join(
        (
            current_epoch.actor_fingerprint,
            current_epoch.belief_fingerprint,
            current_epoch.raw_q_fingerprint,
        )
    )
    if label_policy_fingerprint != expected:
        raise ValueError("Anchor/Retrace continuation policy fingerprint mismatch.")


def combined_policy_fingerprint(epoch: TargetPolicyEpoch) -> str:
    return ":".join(
        (epoch.actor_fingerprint, epoch.belief_fingerprint, epoch.raw_q_fingerprint)
    )


def clone_epoch_target_for_live_control(epoch: TargetPolicyEpoch) -> Any:
    """Clone the frozen epoch target before passing it to donated PPO kernels.

    Retrace, anchors, and decision regret keep reading the immutable epoch
    snapshot while PPO evolves a separate Polyak target.  Sharing their device
    buffers would let donation invalidate the frozen value object.
    """

    import jax

    return jax.tree_util.tree_map(lambda value: value.copy(), epoch.target_params)


def start_target_policy_epoch(
    *,
    params: Any,
    update_number: int,
    updates_per_epoch: int,
    fingerprint_function: Any,
) -> TargetPolicyEpoch:
    """Freeze one actor/belief/raw-Q snapshot and its exact component identity."""

    import jax

    components = {
        "actor": params["universal_actor"],
        "belief": {
            "belief_encoder": params["belief_encoder"],
            "belief_set_encoder": params["belief_set_encoder"],
        },
        "raw_q": {
            key: value
            for key, value in params["universal_critic"].items()
            if str(key).startswith("raw_q")
        },
    }
    if not components["raw_q"]:
        raise ValueError("Target-policy epoch cannot start without twin raw-Q heads.")
    frozen = jax.tree_util.tree_map(lambda value: value.copy(), params)
    epoch_id = epoch_id_for_update(
        update_number, updates_per_epoch=updates_per_epoch
    )
    return TargetPolicyEpoch(
        epoch_id=epoch_id,
        actor_fingerprint=fingerprint_function(components["actor"]),
        belief_fingerprint=fingerprint_function(components["belief"]),
        raw_q_fingerprint=fingerprint_function(components["raw_q"]),
        target_params=frozen,
        started_update=epoch_id * int(updates_per_epoch) + 1,
    )


__all__ = [
    "TARGET_POLICY_EPOCH_UPDATES",
    "TargetPolicyEpoch",
    "clone_epoch_target_for_live_control",
    "combined_policy_fingerprint",
    "epoch_id_for_update",
    "start_target_policy_epoch",
    "validate_epoch_alignment",
    "validate_label_policy",
]
