from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("orbax.checkpoint")
optax = pytest.importorskip("optax")

from src.path_c.storage import (  # noqa: E402
    orbax_manager,
    restore_latest_checkpoint,
    save_checkpoint,
)
from src.path_c.anchor_replay import (  # noqa: E402
    AnchorAuditManifest,
    AnchorAuditRecord,
    AnchorReplayState,
    audit_manifest_from_mapping,
    audit_manifest_to_mapping,
)
from src.path_c.types import (  # noqa: E402
    CounterfactualAnchorBatch,
    GaussianMixtureBelief,
    PolicyState,
    TrainState,
)
from experiments.overcooked_v2.training_app import (  # noqa: E402
    _checkpoint_epoch_validation_update,
    _rehydrate_r3_checkpoint,
)
from src.path_c.policy_epoch import TargetPolicyEpoch  # noqa: E402


def test_resume_validates_the_last_completed_target_policy_epoch() -> None:
    assert _checkpoint_epoch_validation_update(jnp.asarray(0)) == 1
    assert _checkpoint_epoch_validation_update(jnp.asarray(16)) == 16
    with pytest.raises(ValueError, match="cannot be negative"):
        _checkpoint_epoch_validation_update(jnp.asarray(-1))


def _next_update(state):
    draw_key, next_key = jax.random.split(state["random_key"])
    noise = jax.random.normal(draw_key, state["params"].shape)
    return {
        "params": state["params"] + noise + state["generator_params"],
        "optimizer_state": state["optimizer_state"] + 1,
        "random_key": next_key,
        "runner_state": state["runner_state"] + 3,
        "generator_params": state["generator_params"] * 0.9,
        "target_policy_epoch": state["target_policy_epoch"],
        "anchor_replay": state["anchor_replay"],
        "qualification": state["qualification"],
        "optimizer_steps": state["optimizer_steps"] + jnp.asarray(
            [1, 4, 2, 32], dtype=jnp.int32
        ),
        "random_domains": state["random_domains"] + 1,
        "last_qualified_generator_params": state[
            "last_qualified_generator_params"
        ],
    }


def test_checkpoint_resume_preserves_rng_runner_generator_and_next_update(
    tmp_path: Path,
) -> None:
    state = {
        "params": jnp.asarray([1.0, 2.0]),
        "optimizer_state": jnp.asarray(4, dtype=jnp.int32),
        "random_key": jax.random.PRNGKey(17),
        "runner_state": jnp.asarray([5, 6], dtype=jnp.int32),
        "generator_params": jnp.asarray([0.25, -0.5]),
        "target_policy_epoch": {
            "epoch_id": jnp.asarray(2, dtype=jnp.int32),
            "fingerprint": jnp.asarray([11, 12], dtype=jnp.uint32),
        },
        "anchor_replay": {
            "epoch_id": jnp.asarray(2, dtype=jnp.int32),
            "returns": jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
        },
        "qualification": jnp.asarray([1, 1, 0, 0, 0, 0], dtype=jnp.int32),
        "optimizer_steps": jnp.asarray([9, 36, 18, 288], dtype=jnp.int32),
        "random_domains": jnp.asarray([2, 3, 5, 7], dtype=jnp.int32),
        "last_qualified_generator_params": jnp.asarray(
            [0.25, -0.5], dtype=jnp.float32
        ),
    }
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        save_checkpoint(manager, step=9, state=state)
        restored = restore_latest_checkpoint(manager, item=state)
        assert restored is not None and restored[0] == 9
        direct = _next_update(state)
        resumed = _next_update(restored[1])
        for left, right in zip(
            jax.tree_util.tree_leaves(direct),
            jax.tree_util.tree_leaves(resumed),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        manager.close()


def test_checkpoint_roundtrip_supports_r3_anchor_audit_manifest(
    tmp_path: Path,
) -> None:
    """The host audit ledger crosses the Orbax boundary as plain metadata.

    This exercises the exact type that is embedded in the r3 ``TrainState``.
    Passing the slotted dataclass through unchanged makes Orbax reach the final
    checkpoint and fail with an unknown-TypeHandler error after all updates.
    """

    record = AnchorAuditRecord(
        milestone="C0",
        artifact_path="qualification/audit.json",
        artifact_fingerprint="a" * 64,
        random_domain="audit_anchor/fixture",
        fit_replicas=32,
        evaluation_replicas=64,
        continuation_horizon=400,
    )
    manifest = AnchorAuditManifest().add(record)
    saved_state = {
        "params": jnp.asarray([1.0], dtype=jnp.float32),
        "anchor_audit_manifest": audit_manifest_to_mapping(manifest),
    }
    restore_template = {
        "params": jnp.asarray([0.0], dtype=jnp.float32),
        "anchor_audit_manifest": audit_manifest_to_mapping(
            AnchorAuditManifest()
        ),
    }
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        save_checkpoint(manager, step=1, state=saved_state)
        restored = restore_latest_checkpoint(manager, item=restore_template)
        assert restored is not None and restored[0] == 1
        observed = audit_manifest_from_mapping(
            restored[1]["anchor_audit_manifest"]
        )
        assert observed == manifest
        assert observed.records == (record,)
    finally:
        manager.close()


def test_full_r3_checkpoint_restore_rehydrates_dynamic_replay_and_namedtuples(
    tmp_path: Path,
) -> None:
    """Resume uses the stored structure, not the empty initialization shape."""

    belief = GaussianMixtureBelief(
        recurrent_carry=jnp.zeros((1, 2)),
        mixture_logits=jnp.zeros((1, 1)),
        means=jnp.zeros((1, 1, 2)),
        log_variances=jnp.zeros((1, 1, 2)),
        support_score=jnp.zeros((1,)),
    )
    policy = PolicyState(
        task_carry=jnp.zeros((1, 2)),
        belief=belief,
        previous_observation=jnp.zeros((1, 2)),
        previous_action=jnp.zeros((1,), dtype=jnp.int32),
        previous_reward=jnp.zeros((1,)),
        episode_start=jnp.ones((1,), dtype=jnp.bool_),
    )
    batch = CounterfactualAnchorBatch(
        anchor_ids=jnp.asarray([7], dtype=jnp.int32),
        rollout_flat_indexes=jnp.asarray([3], dtype=jnp.int32),
        policy_states=policy,
        observations=jnp.zeros((1, 2)),
        partner_codes=jnp.zeros((1, 2)),
        partner_sources=jnp.asarray([1], dtype=jnp.int32),
        fit_returns_by_action=jnp.zeros((1, 6)),
        evaluation_returns_by_action=jnp.zeros((1, 6)),
        partner_run_ids=jnp.asarray([4], dtype=jnp.int32),
        action_mask=jnp.ones((1, 6), dtype=jnp.bool_),
    )
    replay = AnchorReplayState(
        target_policy_epoch_id=jnp.asarray(0, dtype=jnp.int32),
        target_policy_fingerprint="epoch-zero",
        batch=batch,
        item_count=jnp.asarray(1, dtype=jnp.int32),
        capacity=512,
    )
    optimizer_state = optax.chain(
        optax.clip_by_global_norm(0.25), optax.adam(2.5e-4)
    ).init({"w": jnp.asarray([1.0])})
    fields = {name: jnp.asarray(0, dtype=jnp.int32) for name in TrainState._fields}
    fields.update(
        params={"w": jnp.asarray([1.0])},
        target_params={"w": jnp.asarray([1.0])},
        ppo_optimizer_state=optimizer_state,
        raw_q_optimizer_state=optimizer_state,
        response_optimizer_state=optimizer_state,
        generator_optimizer_state=optimizer_state,
        generator_params={"w": jnp.asarray([2.0])},
        generator_target_params={"w": jnp.asarray([2.0])},
        target_policy_epoch=TargetPolicyEpoch(
            jnp.asarray(0), "a", "b", "q", {"w": jnp.asarray([1.0])}, jnp.asarray(1)
        ),
        qualified_base_params=None,
        owner_source_artifact={"sha256": "owner"},
        last_qualified_generator_params={"w": jnp.asarray([2.0])},
        last_qualified_generator_optimizer_state=None,
        generator_signature_readout=None,
        generator_snapshot_archive=tuple(),
        qualification={"decisions": {}, "highest_passed": None, "fingerprint": "x"},
        anchor_training_replay=None,
        anchor_audit_manifest=audit_manifest_to_mapping(AnchorAuditManifest()),
        runner_state=policy,
        random_domains={"rollout": jnp.asarray(0)},
        resource_ledger={"ego_policy_steps": 0},
        calibration={"alpha": jnp.asarray(0.05)},
    )
    template = TrainState(**fields)
    saved = template._replace(
        anchor_training_replay=replay,
        last_qualified_generator_optimizer_state=optimizer_state,
        generator_snapshot_archive=({"w": jnp.asarray([2.0])},),
        qualification={
            "decisions": {"C0": {"passed": False}},
            "highest_passed": None,
            "fingerprint": "y",
        },
    )
    manager = orbax_manager(tmp_path / "full_checkpoints")
    try:
        save_checkpoint(manager, step=2, state=saved)
        generic = restore_latest_checkpoint(manager)
        assert generic is not None
        restored = _rehydrate_r3_checkpoint(generic[1], template=template)
        assert isinstance(restored, TrainState)
        assert isinstance(restored.runner_state, PolicyState)
        assert isinstance(restored.anchor_training_replay, AnchorReplayState)
        assert isinstance(restored.anchor_training_replay.batch, CounterfactualAnchorBatch)
        assert isinstance(restored.anchor_training_replay.batch.policy_states, PolicyState)
        assert len(restored.generator_snapshot_archive) == 1
        assert type(restored.ppo_optimizer_state[0]) is type(optimizer_state[0])
        assert type(restored.ppo_optimizer_state[1]) is type(optimizer_state[1])
        np.testing.assert_array_equal(
            np.asarray(restored.anchor_training_replay.batch.anchor_ids), [7]
        )
    finally:
        manager.close()
