"""Atomic scientific-state checkpoint and deterministic resume tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("orbax.checkpoint")
optax = pytest.importorskip("optax")

from experiments.overcooked_v2.deployment import _validate_training_identity  # noqa: E402
from experiments.overcooked_v2.training_app import (  # noqa: E402
    _restore_depi_checkpoint,
    _save_depi_checkpoint,
)
from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.storage import orbax_manager  # noqa: E402
from src.path_c.training import (  # noqa: E402
    apply_training_core_update,
    polyak_update,
    training_core_state,
)
from src.path_c.types import RolloutBatch, TrainState  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _state() -> TrainState:
    return TrainState(
        params={"w": jnp.asarray([1.0, 2.0])},
        target_params={"w": jnp.asarray([0.5, 1.5])},
        ppo_optimizer_state={"count": jnp.asarray(3), "moment": jnp.asarray([0.1])},
        supervision_anchor_batch={"returns": jnp.asarray([[1.0, 2.0]])},
        current_separation_terms={"weights": jnp.asarray([0.5])},
        current_pair_comparator={"weights": jnp.asarray([0.25, -0.25])},
        supervision_readings={"pair_class_fractions": jnp.asarray([0.2, 0.3, 0.5])},
        anchor_sampling_counter=jnp.asarray(2, dtype=jnp.int32),
        anchor_microbatch_size=jnp.asarray(288, dtype=jnp.int32),
        effective_update_epochs=jnp.asarray(3, dtype=jnp.int32),
        bootstrap_encoder_params=({"w": jnp.asarray([1.0])},) * 3,
        bootstrap_optimizer_states=({"count": jnp.asarray(4)},) * 3,
        bootstrap_sampling_counters=jnp.asarray([4, 5, 6], dtype=jnp.int32),
        m1_summary_state={
            "evaluations": jnp.asarray(1, dtype=jnp.int32),
            "latest_passed": jnp.asarray(True),
        },
        m1_history={
            "updates": jnp.asarray([1, -1], dtype=jnp.int32),
            "passed": jnp.asarray([True, False]),
            "path_passing_fractions": jnp.asarray(
                [[1.0, 1.0, 1.0, 1.0], [jnp.nan] * 4]
            ),
        },
        ppo_optimizer_step=jnp.asarray(9, dtype=jnp.int32),
        runner_state={
            "carry": jnp.asarray([5, 6]),
            "random_key": jax.random.PRNGKey(17),
        },
        random_domains={"ego": jnp.asarray([1, 2], dtype=jnp.uint32)},
        update_count=jnp.asarray(2, dtype=jnp.int32),
        effective_environment_steps=jnp.asarray(1024, dtype=jnp.int32),
        resource_ledger={
            "ego_policy_steps": 1024,
            "counterfactual_continuation_steps": 128,
            "matched_pair_probe_steps": 16,
        },
    )


def _next_update(state: TrainState) -> TrainState:
    draw, next_key = jax.random.split(state.runner_state["random_key"])
    params = {"w": state.params["w"] + jax.random.normal(draw, (2,))}
    history = dict(state.m1_history)
    history["updates"] = history["updates"].at[1].set(3)
    history["passed"] = history["passed"].at[1].set(False)
    return state._replace(
        params=params,
        target_params=polyak_update(state.target_params, params, 0.005),
        ppo_optimizer_state={
            "count": state.ppo_optimizer_state["count"] + 1,
            "moment": state.ppo_optimizer_state["moment"] + 0.01,
        },
        anchor_sampling_counter=state.anchor_sampling_counter + 1,
        effective_update_epochs=state.effective_update_epochs - 1,
        bootstrap_sampling_counters=state.bootstrap_sampling_counters + 32,
        m1_summary_state={
            "evaluations": state.m1_summary_state["evaluations"] + 1,
            "latest_passed": jnp.asarray(False),
        },
        m1_history=history,
        runner_state={**state.runner_state, "random_key": next_key},
        ppo_optimizer_step=state.ppo_optimizer_step + 1,
        update_count=state.update_count + 1,
        effective_environment_steps=state.effective_environment_steps + 512,
    )


def test_train_state_contains_all_active_and_no_retired_scientific_fields() -> None:
    fields = set(TrainState._fields)
    assert {
        "supervision_anchor_batch",
        "current_separation_terms",
        "current_pair_comparator",
        "supervision_readings",
        "anchor_microbatch_size",
        "effective_update_epochs",
        "bootstrap_encoder_params",
        "bootstrap_optimizer_states",
        "bootstrap_sampling_counters",
        "m1_summary_state",
        "m1_history",
        "random_domains",
    } <= fields
    assert not fields & {
        "raw_q_optimizer_state",
        "response_optimizer_state",
        "belief_optimizer_state",
        "anchor_replay",
        "action_range_ema",
    }


def test_save_restore_then_update_equals_uninterrupted_update(tmp_path: Path) -> None:
    state = _state()
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        _save_depi_checkpoint(manager, tmp_path, state)
        restored = _restore_depi_checkpoint(manager, tmp_path, state)
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


def test_actual_model_optimizer_update_is_resume_equivalent(tmp_path: Path) -> None:
    observation_shape = (5, 5, 39)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        protocol_components=4,
        component_embedding_dim=4,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        response_hidden_dim=8,
        modulation_rank=2,
        action_embedding_dim=4,
        method_variant="b1",
    )
    policy_state = initial_policy_state(
        batch_size=2,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )
    observations = jax.random.normal(
        jax.random.PRNGKey(30), (3, 2) + observation_shape
    )
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(31),
        example_state=policy_state,
        example_observation=observations[0],
    )
    optimizer = optax.adam(1.0e-3)
    scientific = _state()._replace(
        params=params,
        target_params=params,
        ppo_optimizer_state=optimizer.init(params),
    )
    batch = RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros((3, 2), dtype=jnp.int32),
        episode_starts=jnp.zeros((3, 2), dtype=jnp.bool_).at[0].set(True),
        action_keys=jnp.zeros((3, 2, 2), dtype=jnp.uint32),
        context_dropout_masks=jnp.zeros((3, 2), dtype=jnp.bool_),
        actions=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        rewards=jnp.ones((2, 2)),
        official_shaped_rewards=jnp.zeros((2, 2)),
        official_shaping_factors=jnp.zeros((2, 2)),
        shaped_rewards=jnp.ones((2, 2)),
        dones=jnp.zeros((2, 2), dtype=jnp.bool_),
        old_log_probabilities=jnp.full((2, 2), -np.log(6.0)),
        old_values=jnp.zeros((3, 2)),
        behavior_probabilities=jnp.full((2, 2), 1.0 / 6.0),
        ppo_mask=jnp.ones((2, 2)),
        partner_sources=jnp.zeros((2, 2), dtype=jnp.int32),
        partner_members=jnp.zeros((2, 2), dtype=jnp.int32),
        partner_family_ids=jnp.zeros((2, 2), dtype=jnp.int32),
        partner_checkpoint_stages=jnp.ones((2, 2)),
        partner_run_ids=jnp.zeros((2, 2), dtype=jnp.int32),
        initial_policy_state=policy_state,
        initial_target_policy_state=policy_state,
    )
    config = SimpleNamespace(
        method_variant="b1",
        ppo=SimpleNamespace(
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            value_weight=0.5,
            entropy_weight=0.01,
            normalize_advantages=True,
        ),
        loss_v2=SimpleNamespace(
            signature_weight=1.0,
            response_weight=1.0,
            separation_weight=0.1,
            rank_hinge_margin=0.1,
            rank_hinge_advantage_gap=2.0,
            decision_policy_weight=0.25,
            decision_policy_temperature=1.0,
            capability_consistency_weight=0.01,
            combined_policy_kl_threshold=10.0,
        ),
    )
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        _save_depi_checkpoint(manager, tmp_path, scientific)
        restored = _restore_depi_checkpoint(manager, tmp_path, scientific)
        assert restored is not None
        direct, direct_metrics = apply_training_core_update(
            model=model,
            core=training_core_state(scientific),
            optimizer=optimizer,
            batch=batch,
            config=config,
        )
        resumed, resumed_metrics = apply_training_core_update(
            model=model,
            core=training_core_state(restored),
            optimizer=optimizer,
            batch=batch,
            config=config,
        )
        for left, right in zip(
            jax.tree_util.tree_leaves((direct, direct_metrics)),
            jax.tree_util.tree_leaves((resumed, resumed_metrics)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        manager.close()


def test_superseded_training_identity_is_hard_rejected() -> None:
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/depi_simple_formal.yaml",
        run_kind="formal",
    )
    with pytest.raises(ValueError, match="DEPI training run"):
        _validate_training_identity(
            {"method": "superseded", "stage": "train", "config": {"version": 1}},
            config,
        )
