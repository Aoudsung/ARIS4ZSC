"""Checkpoint resume tests.

Specification entries covered (docs/METHOD_SPEC.md):
- §1.4 PolicyState (task/capability/protocol carries) serializes through the
  anchor replay rows stored inside the V6 checkpoint.
- §3.3 TrainState keeps the frozen legacy optimizer slots for checkpoint
  structure compatibility; only the PPO core is actively updated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("orbax.checkpoint")

from experiments.overcooked_v2.deployment import _validate_training_identity  # noqa: E402
from experiments.overcooked_v2.training_app import (  # noqa: E402
    _load_source_batch,
    _restore_v6_checkpoint,
    _save_source_batch,
    _save_v6_checkpoint,
)
from src.path_c.anchor_replay import AnchorReplayState  # noqa: E402
from src.path_c.base_distillation import OwnerBehaviorBatch  # noqa: E402
from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import initial_policy_state  # noqa: E402
from src.path_c.storage import orbax_manager  # noqa: E402
from src.path_c.training import polyak_update  # noqa: E402
from src.path_c.types import CounterfactualAnchorBatch, TrainState  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _replay() -> AnchorReplayState:
    count = 4
    policy = initial_policy_state(
        batch_size=count,
        observation_shape=(1,),
        action_count=6,
        task_hidden_dim=2,
        capability_hidden_dim=2,
        protocol_hidden_dim=2,
        capability_dim=2,
        component_embedding_dim=2,
        protocol_components=4,
    )
    values = jnp.arange(count * 6, dtype=jnp.float32).reshape((count, 6))
    batch = CounterfactualAnchorBatch(
        anchor_ids=jnp.arange(count),
        rollout_flat_indexes=jnp.arange(count),
        policy_states=policy,
        observations=jnp.zeros((count, 1)),
        partner_codes=jnp.zeros((count, 2)),
        partner_sources=jnp.zeros((count,), dtype=jnp.int32),
        partner_run_ids=jnp.arange(count),
        fit_returns_by_action=values,
        return_sum_by_action=values * 4,
        return_squared_sum_by_action=values * values * 4,
        replica_count=jnp.full((count, 6), 4, dtype=jnp.int32),
        collection_policy_logits=jnp.zeros((count, 6)),
        collection_update=jnp.zeros((count,), dtype=jnp.int32),
        collection_target_fingerprint=jnp.zeros((count, 2), dtype=jnp.uint32),
        matched_pair_ids=jnp.asarray([-1, -1, 0, 0]),
        action_mask=jnp.ones((count, 6), dtype=jnp.bool_),
    )
    return AnchorReplayState(batch, jnp.asarray(count), count, jnp.zeros((count,), jnp.int32))


def _state() -> TrainState:
    optimizer = {"count": jnp.asarray(3), "moment": jnp.asarray([0.1, 0.2])}
    return TrainState(
        params={"w": jnp.asarray([1.0, 2.0])},
        target_params={"w": jnp.asarray([0.5, 1.5])},
        ppo_optimizer_state=optimizer,
        raw_q_optimizer_state=optimizer,
        response_optimizer_state=optimizer,
        belief_optimizer_state=optimizer,
        generator_optimizer_state=optimizer,
        generator_params={"g": jnp.asarray([0.25, -0.5])},
        generator_target_params={"g": jnp.asarray([0.20, -0.4])},
        generator_competence_multiplier=jnp.asarray(0.7),
        generator_cvar_ema=jnp.asarray(-3.0),
        external_reference_cvar_ema=jnp.asarray(5.0),
        anchor_replay=_replay(),
        anchor_sampling_counter=jnp.asarray(2),
        belief_gradient_norm_ema={"ppo": jnp.asarray(0.3)},
        action_range_ema=jnp.asarray(1.4),
        anchor_advantage_scale_ema=jnp.asarray(2.5),
        ppo_optimizer_step=jnp.asarray(9),
        raw_q_optimizer_step=jnp.asarray(36),
        response_optimizer_step=jnp.asarray(18),
        belief_optimizer_step=jnp.asarray(9),
        generator_optimizer_step=jnp.asarray(288),
        runner_state={"carry": jnp.asarray([5, 6]), "random_key": jax.random.PRNGKey(17)},
        random_domains={"rollout": [1, 2], "anchor_replay": [3, 4]},
        update_count=jnp.asarray(2),
        effective_environment_steps=jnp.asarray(1024),
        resource_ledger={"ego_policy_steps": 1024, "total_training_simulator_steps": 2048},
    )


def _next_update(state: TrainState) -> TrainState:
    draw, next_key = jax.random.split(state.runner_state["random_key"])
    params = {"w": state.params["w"] + jax.random.normal(draw, (2,))}
    target = polyak_update(state.target_params, params, 0.005)
    replay = state.anchor_replay._replace(use_counts=state.anchor_replay.use_counts.at[0].add(1))
    incremented_optimizer = {
        "count": state.ppo_optimizer_state["count"] + 1,
        "moment": state.ppo_optimizer_state["moment"] + 0.01,
    }
    return state._replace(
        params=params,
        target_params=target,
        ppo_optimizer_state=incremented_optimizer,
        raw_q_optimizer_state=incremented_optimizer,
        response_optimizer_state=incremented_optimizer,
        belief_optimizer_state=incremented_optimizer,
        generator_optimizer_state=incremented_optimizer,
        anchor_replay=replay,
        runner_state={**state.runner_state, "random_key": next_key},
        ppo_optimizer_step=state.ppo_optimizer_step + 1,
        raw_q_optimizer_step=state.raw_q_optimizer_step + 4,
        response_optimizer_step=state.response_optimizer_step + 2,
        belief_optimizer_step=state.belief_optimizer_step + 1,
        generator_optimizer_step=state.generator_optimizer_step + 32,
        update_count=state.update_count + 1,
        effective_environment_steps=state.effective_environment_steps + 512,
    )


def test_v6_checkpoint_resume_preserves_rng_replay_ema_and_all_optimizers(
    tmp_path: Path,
) -> None:
    state = _state()
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        _save_v6_checkpoint(manager, tmp_path, state)
        restored = _restore_v6_checkpoint(manager, tmp_path, state)
        assert restored is not None
        direct = _next_update(state)
        resumed = _next_update(restored)
        for left, right in zip(
            jax.tree_util.tree_leaves(direct),
            jax.tree_util.tree_leaves(resumed),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        manager.close()


def test_v5_training_identity_is_hard_rejected() -> None:
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/delta_zsc_simple_formal.yaml",
        run_kind="formal",
    )
    identity = {
        "method": "delta_zsc_v5_decision_equivalent_bayes_r3_signal_contract",
        "stage": "train",
        "config": {"version": 7},
    }
    with pytest.raises(ValueError, match="V6 training run"):
        _validate_training_identity(identity, config)


def test_generator_initialization_replay_is_bound_to_run_identity(
    tmp_path: Path,
) -> None:
    batch = OwnerBehaviorBatch(
        observations=jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3)),
        previous_actions=jnp.zeros((2, 2), dtype=jnp.int32),
        episode_starts=jnp.zeros((2, 2), dtype=jnp.bool_),
        owner_logits=jnp.ones((2, 2, 6), dtype=jnp.float32),
        valid_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        source_members=jnp.asarray([0, 1], dtype=jnp.int32),
    )
    identity = {"method": "v6", "seed": 3, "nested": {"manifest": "abc"}}
    path = tmp_path / "generator_source_batch"
    _save_source_batch(path, batch, run_identity=identity)

    restored = _load_source_batch(path, run_identity=identity)
    for left, right in zip(batch, restored, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    with pytest.raises(RuntimeError, match="another run identity"):
        _load_source_batch(path, run_identity={**identity, "seed": 4})

    sidecar = path.parent / f"{path.name}.identity.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["batch_fingerprint"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        _load_source_batch(path, run_identity=identity)
