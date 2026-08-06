from __future__ import annotations

from pathlib import Path

import numpy as np


def test_deterministic_and_spectral_initializers_are_centered_and_round_trip(
    tmp_path: Path,
) -> None:
    from src.delta_zsc.semantic_initializer import (
        deterministic_simplex_initializer,
        fit_spectral_simplex_initializer,
        load_semantic_initializer,
        save_semantic_initializer,
    )

    fallback = deterministic_simplex_initializer(4, 31)
    assert fallback.event_component_bias.shape == (4, 31)
    np.testing.assert_allclose(
        np.mean(fallback.event_component_bias, axis=0), 0.0, atol=1.0e-6
    )

    # The first residual direction is an explicit event-12/event-13 convention
    # contrast; no mechanism or partner label is supplied to the fitter.
    residuals = np.zeros((20, 31), dtype=np.float32)
    residuals[:10, 12] = 1.0
    residuals[:10, 13] = -1.0
    residuals[10:, 12] = -1.0
    residuals[10:, 13] = 1.0
    fitted = fit_spectral_simplex_initializer(
        residuals,
        component_count=4,
        source={"uses_partner_labels": False, "panel": "unit-test"},
    )
    np.testing.assert_allclose(
        np.mean(fitted.event_component_bias, axis=0), 0.0, atol=1.0e-6
    )
    assert fitted.singular_values[0] > 1.0
    assert np.std(
        fitted.event_component_bias[:, 12] - fitted.event_component_bias[:, 13]
    ) > 0.1
    npz_path, json_path = save_semantic_initializer(tmp_path, fitted)
    assert npz_path.is_file() and json_path.is_file()
    restored = load_semantic_initializer(
        tmp_path, component_count=4, event_count=31
    )
    np.testing.assert_allclose(
        restored.event_component_bias, fitted.event_component_bias, atol=0.0
    )
    assert restored.source["uses_partner_labels"] is False

    import json
    import pytest

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["uses_partner_labels"] = True
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="partner labels"):
        load_semantic_initializer(tmp_path, component_count=4, event_count=31)


def test_fitted_initializer_provenance_is_layout_and_lineage_bound() -> None:
    import pytest

    from src.delta_zsc.config import METHOD_VERSION
    from src.delta_zsc.semantic_initializer import (
        deterministic_simplex_initializer,
        fit_spectral_simplex_initializer,
        validate_semantic_initializer_provenance,
    )

    source = {
        "method": METHOD_VERSION,
        "layout": "test_time_simple",
        "partner_role": "calibration",
        "component_count": 4,
        "official_protocol_version": "overcooked_v2_iclr2025_5ce1707_v1",
        "official_source_commit": "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
        "partner_run_ids": ["cal-run-a", "cal-run-b"],
        "parent_training_run_ids": ["cal-parent-a", "cal-parent-b"],
        "event_count": 40,
        "episode_ids": [0, 1, 2, 3],
        "uses_partner_labels": False,
    }
    fitted = fit_spectral_simplex_initializer(
        np.eye(31, dtype=np.float32), component_count=4, source=source
    )
    validate_semantic_initializer_provenance(
        fitted,
        method=METHOD_VERSION,
        layout="test_time_simple",
        component_count=4,
        event_classes=31,
        protocol_version="overcooked_v2_iclr2025_5ce1707_v1",
        official_source_commit="5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
        training_parent_ids={"train-parent-a"},
        expected_calibration_run_ids={"cal-run-a", "cal-run-b"},
        require_fitted=True,
    )
    with pytest.raises(ValueError, match="exact calibration panel"):
        validate_semantic_initializer_provenance(
            fitted,
            method=METHOD_VERSION,
            layout="test_time_simple",
            component_count=4,
            event_classes=31,
            protocol_version="overcooked_v2_iclr2025_5ce1707_v1",
            official_source_commit="5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
            training_parent_ids={"train-parent-a"},
            expected_calibration_run_ids={"cal-run-a", "different-run"},
            require_fitted=True,
        )
    with pytest.raises(ValueError, match="overlap"):
        validate_semantic_initializer_provenance(
            fitted,
            method=METHOD_VERSION,
            layout="test_time_simple",
            component_count=4,
            event_classes=31,
            protocol_version="overcooked_v2_iclr2025_5ce1707_v1",
            official_source_commit="5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
            training_parent_ids={"cal-parent-a"},
            expected_calibration_run_ids={"cal-run-a", "cal-run-b"},
            require_fitted=True,
        )
    with pytest.raises(ValueError, match="provenance field 'layout'"):
        validate_semantic_initializer_provenance(
            fitted,
            method=METHOD_VERSION,
            layout="test_time_wide",
            component_count=4,
            event_classes=31,
            protocol_version="overcooked_v2_iclr2025_5ce1707_v1",
            official_source_commit="5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
            training_parent_ids={"train-parent-a"},
            expected_calibration_run_ids={"cal-run-a", "cal-run-b"},
            require_fitted=True,
        )
    with pytest.raises(ValueError, match="require a fitted"):
        validate_semantic_initializer_provenance(
            deterministic_simplex_initializer(4, 31),
            method=METHOD_VERSION,
            layout="test_time_simple",
            component_count=4,
            event_classes=31,
            protocol_version="overcooked_v2_iclr2025_5ce1707_v1",
            official_source_commit="5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
            training_parent_ids={"train-parent-a"},
            expected_calibration_run_ids={"cal-run-a", "cal-run-b"},
            require_fitted=True,
        )


def test_delayed_probe_target_uses_second_action_window_and_masks_boundaries() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.observation import extract_probe_response_target

    intermediate = jnp.zeros((2, 5, 5, 39), dtype=jnp.float32)
    delayed = intermediate
    target = extract_probe_response_target(
        intermediate,
        delayed,
        jnp.asarray([4, 5], dtype=jnp.int32),
        jnp.asarray([False, True]),
    )
    np.testing.assert_allclose(np.asarray(target.valid_mask), [1.0, 0.0])
    assert float(target.interface_available[1]) == 0.0
    assert float(target.interface_changed[1]) == 0.0
    assert float(target.visibility[1]) == 0.0


def test_three_channel_objective_is_exact_sum_of_normalized_channels() -> None:
    """The instrumentation rate cannot silently become a loss coefficient."""

    from pathlib import Path
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.config import load_config
    from src.delta_zsc.losses import latent_composite_loss
    from src.delta_zsc.model import DeltaModel
    from src.delta_zsc.semantic_initializer import deterministic_simplex_initializer
    from src.delta_zsc.types import RolloutBatch

    config = load_config(
        Path("experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml"),
        run_kind="mechanical",
    )
    config = replace(
        config,
        method_variant="response_only",
        model=replace(
            config.model,
            task_hidden_dim=8,
            task_embedding_dim=8,
            instant_partner_dim=4,
            latent_hidden_dim=8,
            latent_embedding_dim=4,
            action_embedding_dim=3,
        ),
    )
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(
        jax.random.PRNGKey(7),
        semantic_initializer=deterministic_simplex_initializer(4, 31),
    )
    time, lanes = 3, 1
    batch = RolloutBatch(
        observations=jnp.zeros((time + 1, lanes, 5, 5, 39)),
        response_next_observations=jnp.zeros((time, lanes, 5, 5, 39)),
        previous_actions=jnp.zeros((time + 1, lanes), dtype=jnp.int32),
        episode_starts=jnp.zeros((time + 1, lanes), dtype=jnp.bool_).at[0].set(True),
        actions=jnp.zeros((time, lanes), dtype=jnp.int32),
        rewards=jnp.zeros((time, lanes)),
        shaped_rewards=jnp.zeros((time, lanes)),
        dones=jnp.zeros((time, lanes), dtype=jnp.bool_),
        old_log_probabilities=jnp.zeros((time, lanes)),
        old_values=jnp.zeros((time, lanes)),
        ppo_mask=jnp.ones((time, lanes)),
        initial_policy_state=model.initial_state(lanes),
    )
    result = latent_composite_loss(model, latent, base, batch, None)
    expected = (
        result.metrics["latent_shared_total_nll"]
        + result.metrics["latent_semantic_total_nll"]
    )
    np.testing.assert_allclose(float(result.total), float(expected), atol=1.0e-6)


def test_active_probe_commits_exactly_one_base_continuation() -> None:
    """Delayed VOI and execution use the same intervening base policy."""

    from dataclasses import replace
    from pathlib import Path

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.config import load_config
    from src.delta_zsc.model import DeltaModel, observe_after_transition
    from src.delta_zsc.semantic_initializer import deterministic_simplex_initializer

    config = load_config(
        Path("experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml"),
        run_kind="mechanical",
    )
    config = replace(
        config,
        model=replace(
            config.model,
            task_hidden_dim=8,
            task_embedding_dim=8,
            instant_partner_dim=4,
            latent_hidden_dim=8,
            latent_embedding_dim=4,
            action_embedding_dim=3,
        ),
    )
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(
        jax.random.PRNGKey(23),
        semantic_initializer=deterministic_simplex_initializer(4, 31),
    )
    observation = jnp.zeros((1, 5, 5, 39), dtype=jnp.float32)
    state = model.initial_state(1)

    stepped_probe, first = model.step(base, latent, state, observation)
    assert bool(stepped_probe.probe_continuation_pending[0])
    after_probe = observe_after_transition(
        stepped_probe, action=jnp.asarray([4]), done=jnp.asarray([False])
    )

    stepped_bridge, bridge = model.step(base, latent, after_probe, observation)
    np.testing.assert_allclose(
        np.asarray(bridge.policy_logits),
        np.asarray(bridge.base_policy_logits),
        atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(bridge.active_voi), 0.0, atol=0.0)
    assert not bool(stepped_bridge.probe_continuation_pending[0])

    after_bridge = observe_after_transition(
        stepped_bridge, action=jnp.asarray([4]), done=jnp.asarray([False])
    )
    stepped_next_probe, _ = model.step(base, latent, after_bridge, observation)
    assert bool(stepped_next_probe.probe_continuation_pending[0])

    terminal = observe_after_transition(
        stepped_next_probe, action=jnp.asarray([4]), done=jnp.asarray([True])
    )
    assert not bool(terminal.probe_continuation_pending[0])


def test_probe_anchor_forces_decision_only_after_base_bridge() -> None:
    """Post-response utility starts at t+2 and excludes probe/bridge rewards."""

    from typing import NamedTuple

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.anchors import (
        AnchorFunctions,
        AnchorWorld,
        collect_probe_successor_continuations,
    )

    class Carry(NamedTuple):
        value: object

    def ego_step(base, latent, state, observation, keys):
        del base, latent, observation, keys
        # The intervening unforced action is the registered base continuation.
        action = jnp.full_like(state.value, 2, dtype=jnp.int32)
        return Carry(state.value + 1), action

    def ego_observe(state, action, done):
        del action, done
        return state

    def partner_step(state, observation, episode_start, keys):
        del observation, episode_start, keys
        action = jnp.zeros_like(state.value, dtype=jnp.int32)
        return action, Carry(state.value + 1), state

    def partner_observe(
        state, context, observation, action, reward, done, next_observation
    ):
        del context, observation, action, reward, done, next_observation
        return state

    def environment_step(state, joint_actions, keys):
        del keys
        # State 0 is the probe transition, state 1 the base bridge, and state 2
        # the first post-response decision. Only the final reward may appear in
        # the returned Q matrix.
        reward = state.astype(jnp.float32) * 100.0 + joint_actions[:, 0]
        next_state = state + 1
        observations = jnp.zeros((state.shape[0], 2, 1), dtype=jnp.float32)
        done = jnp.zeros_like(state, dtype=jnp.bool_)
        by_agent = jnp.stack((reward, reward), axis=-1)
        return next_state, observations, reward, done, {
            "raw_rewards_by_agent": by_agent
        }

    functions = AnchorFunctions(
        ego_step=ego_step,
        ego_observe=ego_observe,
        partner_step=partner_step,
        partner_observe=partner_observe,
        environment_step=environment_step,
    )
    world = AnchorWorld(
        environment_state=jnp.asarray([0], dtype=jnp.int32),
        observations=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        ego_state=Carry(jnp.asarray([0], dtype=jnp.int32)),
        partner_state=Carry(jnp.asarray([0], dtype=jnp.int32)),
        partner_episode_start=jnp.asarray([False]),
        ego_roles=jnp.asarray([0], dtype=jnp.int32),
        done=jnp.asarray([False]),
    )
    fit, evaluation, _, _, mask = collect_probe_successor_continuations(
        world=world,
        root_keys=jnp.asarray([jax.random.PRNGKey(1)]),
        functions=functions,
        base_params=None,
        latent_params=None,
        action_count=3,
        fit_replicas=1,
        evaluation_replicas=1,
        horizon=1,
        gamma=0.9,
    )
    expected = np.broadcast_to(
        np.asarray([200.0, 201.0, 202.0], dtype=np.float32), (1, 3, 3)
    )
    np.testing.assert_allclose(np.asarray(fit), expected, atol=0.0)
    np.testing.assert_allclose(np.asarray(evaluation), expected, atol=0.0)
    assert bool(jnp.all(mask))


def test_successor_anchor_waits_for_delayed_response_and_excludes_bridge_rewards() -> None:
    """The forced decision action is at t+2, after one base-policy bridge."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.anchors import (
        AnchorFunctions,
        AnchorWorld,
        collect_probe_successor_continuations,
    )

    def ego_step(base, latent, state, observation, keys):
        del base, latent, observation, keys
        return state, jnp.zeros(state.shape, dtype=jnp.int32)

    def ego_observe(state, action, done):
        del action, done
        return state

    def partner_step(state, observation, episode_start, keys):
        del observation, episode_start, keys
        return jnp.zeros(state.shape, dtype=jnp.int32), state, state

    def partner_observe(state, context, observation, action, reward, done, next_observation):
        del context, observation, action, reward, done, next_observation
        return state

    def environment_step(state, joint_actions, keys):
        del joint_actions, keys
        next_state = state + 1
        batch = state.shape[0]
        observations = jnp.zeros((batch, 2, 1), dtype=jnp.float32)
        reward = next_state.astype(jnp.float32)
        done = jnp.zeros((batch,), dtype=jnp.bool_)
        by_agent = jnp.stack((reward, reward), axis=-1)
        return next_state, observations, reward, done, {
            "raw_rewards_by_agent": by_agent
        }

    world = AnchorWorld(
        environment_state=jnp.zeros((1,), dtype=jnp.int32),
        observations=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        ego_state=jnp.zeros((1,), dtype=jnp.int32),
        partner_state=jnp.zeros((1,), dtype=jnp.int32),
        partner_episode_start=jnp.ones((1,), dtype=jnp.bool_),
        ego_roles=jnp.zeros((1,), dtype=jnp.int32),
        done=jnp.zeros((1,), dtype=jnp.bool_),
    )
    functions = AnchorFunctions(
        ego_step=ego_step,
        ego_observe=ego_observe,
        partner_step=partner_step,
        partner_observe=partner_observe,
        environment_step=environment_step,
    )
    fit, evaluation, _, _, mask = collect_probe_successor_continuations(
        world=world,
        root_keys=jax.random.split(jax.random.PRNGKey(91), 1),
        functions=functions,
        base_params=None,
        latent_params=None,
        action_count=2,
        fit_replicas=1,
        evaluation_replicas=1,
        horizon=1,
        gamma=0.99,
    )
    # Probe reward is 1, bridge reward is 2, forced t+2 decision reward is 3.
    # Only the last quantity belongs to the successor decision target.
    np.testing.assert_allclose(np.asarray(fit), 3.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(evaluation), 3.0, atol=0.0)
    assert bool(jnp.all(mask))


def test_passive_anchor_batch_has_no_successor_observation() -> None:
    """False probe masks distinguish absent targets from zero return labels."""

    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.tests.test_delta_unified_runner_storage import (
        MockEnvironment,
        _model,
        _partner_functions,
    )
    from experiments.overcooked_v2.training_app import _anchor_functions
    from src.delta_zsc.anchors import collect_anchor_batch
    from src.delta_zsc.runner import collect_rollout, initialize_runner

    _, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner,
        random_key=jax.random.PRNGKey(92),
    )
    _, _, records = collect_rollout(
        state=runner,
        length=2,
        environment=environment,
        model=model,
        base_params=base,
        latent_params=latent,
        partner_functions=partner,
        partner_parameters=None,
        official_shaping_factor=0.0,
        record_anchors=True,
        use_deployment_policy=False,
    )
    anchors = collect_anchor_batch(
        key=jax.random.PRNGKey(93),
        records=records,
        functions=_anchor_functions(
            model=model,
            partner_functions=partner,
            partner_parameters=None,
            environment=environment,
        ),
        base_params=base,
        latent_params=latent,
        states_per_trigger=1,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=1,
        horizon=1,
        gamma=0.99,
        collect_successor=False,
    )
    assert not bool(jnp.any(anchors.probe_action_mask))
    np.testing.assert_allclose(
        np.asarray(anchors.probe_fit_returns_by_action), 0.0, atol=0.0
    )


def test_development_budget_distinguishes_passive_and_active_anchor_cost() -> None:
    import yaml

    from experiments.overcooked_v2.development_matrix_app import _anchor_budget

    payload = yaml.safe_load(
        Path(
            "experiments/overcooked_v2/configs/delta_unified_simple_development.yaml"
        ).read_text(encoding="utf-8")
    )
    passive = _anchor_budget(payload, method_variant="delta_passive")
    active = _anchor_budget(payload, method_variant="delta_active")
    assert passive > 0
    assert active > passive
    anchors = payload["anchors"]
    triggers = 1_228_800 // int(anchors["interval_environment_steps"])
    replicas = int(anchors["fit_replicas"]) + int(anchors["evaluation_replicas"])
    expected_extra = (
        triggers
        * int(anchors["states_per_trigger"])
        * 6
        * replicas
        * (2 + 6 * int(payload["method"]["continuation_horizon"]))
    )
    assert active - passive == expected_extra


def test_development_matrix_resolves_k_specific_initializer_root(tmp_path: Path) -> None:
    from experiments.overcooked_v2.development_matrix_app import (
        _initializer_for_component_count,
    )
    from src.delta_zsc.semantic_initializer import (
        deterministic_simplex_initializer,
        save_semantic_initializer,
    )

    for count in (2, 4, 8):
        save_semantic_initializer(
            tmp_path / f"k-{count}", deterministic_simplex_initializer(count, 31)
        )
    for count in (2, 4, 8):
        assert _initializer_for_component_count(tmp_path, count) == (
            tmp_path / f"k-{count}"
        ).resolve()

def test_final_audit_component_event_js_reduces_probe_axis_before_anchor_mask():
    import jax.numpy as jnp

    from experiments.overcooked_v2.training_app import _component_event_js_by_anchor

    # Deliberately use anchor_count != probe_count: this is the shape that
    # exposed the final-audit broadcasting defect.
    logits = jnp.zeros((5, 6, 4, 31), dtype=jnp.float32)
    uninformative = _component_event_js_by_anchor(logits)
    assert uninformative.shape == (5,)
    assert jnp.allclose(uninformative, 0.0, atol=1.0e-7)

    revealing = logits.at[:, :, 0, 0].set(8.0)
    revealing = revealing.at[:, :, 1, 1].set(8.0)
    revealing = revealing.at[:, :, 2, 2].set(8.0)
    revealing = revealing.at[:, :, 3, 3].set(8.0)
    value = _component_event_js_by_anchor(revealing)
    assert value.shape == (5,)
    assert jnp.all(value > 0.1)

    eligibility = jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.0])
    aggregate = jnp.sum(eligibility * value) / jnp.sum(eligibility)
    assert jnp.isfinite(aggregate)

