from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class _PartnerState(NamedTuple):
    value: Any


class MockEnvironment:
    num_envs = 4
    observation_shape = (5, 5, 39)

    def _observations(self, state):
        import jax.numpy as jnp

        count = int(state.shape[0])
        value = jnp.zeros((count, 2, 5, 5, 39), dtype=jnp.float32)
        # Each local view sees the teammate in the registered partner block.
        value = value.at[:, :, 2, 2, 10].set(1.0)
        value = value.at[:, :, 2, 2, 11].set(1.0)
        return value

    def reset(self, key):
        import jax.numpy as jnp

        del key
        state = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        return state, self._observations(state)

    def step_with_keys(self, state, joint_actions, keys):
        import jax.numpy as jnp

        del keys
        terminal_state = state + 1
        terminal = self._observations(terminal_state)
        done = terminal_state % 3 == 0
        reset_state = jnp.zeros_like(state)
        next_state = jnp.where(done, reset_state, terminal_state)
        reset = self._observations(reset_state)
        mask = done[:, None, None, None, None]
        observation = jnp.where(mask, reset, terminal)
        reward = (joint_actions[:, 0] == joint_actions[:, 1]).astype(jnp.float32)
        by_agent = jnp.stack((reward, reward), axis=-1)
        return next_state, observation, reward, done, {
            "terminal_observations": terminal,
            "raw_rewards_by_agent": by_agent,
            "official_shaped_rewards_by_agent": jnp.zeros_like(by_agent),
        }


def _partner_functions():
    import jax.numpy as jnp

    from src.delta_zsc.runner import PartnerFunctions

    def initial(batch_size, key):
        del key
        return _PartnerState(jnp.zeros((int(batch_size),), dtype=jnp.int32))

    def step(parameters, state, observation, episode_start, keys):
        del parameters, observation, episode_start, keys
        action = state.value % 6
        return action, _PartnerState(state.value + 1), state, jnp.zeros_like(action)

    def observe(parameters, state, context, observation, action, reward, done, next_observation):
        del parameters, context, observation, action, reward, next_observation
        return _PartnerState(jnp.where(done, 0, state.value))

    def run_id(parameters, state, context):
        del parameters, context
        return jnp.zeros_like(state.value)

    def diagnostics(parameters, state, context):
        del parameters, context
        return {"member": jnp.zeros_like(state.value)}

    return PartnerFunctions(initial, step, observe, run_id, diagnostics)


def _model():
    import jax

    from src.delta_zsc.config import load_config
    from src.delta_zsc.model import DeltaModel

    config = load_config(
        "experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        model=replace(
            config.model,
            task_hidden_dim=16,
            task_embedding_dim=16,
            instant_partner_dim=8,
            latent_hidden_dim=16,
            latent_embedding_dim=8,
            action_embedding_dim=4,
        ),
    )
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(jax.random.PRNGKey(0))
    return config, model, base, latent


def test_mock_end_to_end_rollout_and_anchor_use_base_policy() -> None:
    import jax
    import jax.numpy as jnp

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
        random_key=jax.random.PRNGKey(10),
    )
    next_runner, batch, records = collect_rollout(
        state=runner,
        length=4,
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
    assert batch.observations.shape == (5, 4, 5, 5, 39)
    assert batch.response_next_observations.shape == (4, 4, 5, 5, 39)
    # Training is exactly base-policy collection: no active value is computed
    # and no analytic deployment adjustment enters the behavior distribution.
    np.testing.assert_allclose(np.asarray(records["active_voi"]), 0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(records["deployment_policy_logits"]),
        np.asarray(records["base_policy_logits"]),
        atol=0.0,
    )
    assert int(next_runner.completed_episodes) > 0
    # Training behavior log-probabilities are computed from the base logits.
    from src.delta_zsc.losses import categorical_log_probability

    expected = categorical_log_probability(records["base_policy_logits"], batch.actions)
    np.testing.assert_allclose(
        np.asarray(batch.old_log_probabilities), np.asarray(expected), atol=1e-6
    )
    functions = _anchor_functions(
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        environment=environment,
    )
    anchors = collect_anchor_batch(
        key=jax.random.PRNGKey(11),
        records=records,
        functions=functions,
        base_params=base,
        latent_params=latent,
        states_per_trigger=2,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=2,
        horizon=2,
        gamma=0.99,
        xp_lanes_only=True,
    )
    assert anchors.fit_returns_by_action.shape == (2, 6)
    assert anchors.probe_fit_returns_by_action.shape == (2, 6, 6)
    assert anchors.probe_action_mask.shape == (2, 6, 6)


def test_generic_rollout_keeps_frozen_partners_and_training_self_state_is_independent() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.runner import collect_rollout, initialize_runner

    _, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner,
        random_key=jax.random.PRNGKey(17),
    )
    frozen, frozen_batch, _ = collect_rollout(
        state=runner,
        length=2,
        environment=environment,
        model=model,
        base_params=base,
        latent_params=latent,
        partner_functions=partner,
        partner_parameters=None,
        official_shaping_factor=0.0,
        record_anchors=False,
    )
    np.testing.assert_allclose(np.asarray(frozen_batch.policy_group), 1.0, atol=0.0)
    assert _same_tree(frozen.self_policy_state, runner.self_policy_state)

    mixed, mixed_batch, _ = collect_rollout(
        state=runner,
        length=2,
        environment=environment,
        model=model,
        base_params=base,
        latent_params=latent,
        partner_functions=partner,
        partner_parameters=None,
        official_shaping_factor=0.0,
        record_anchors=False,
        mixed_training_partners=True,
    )
    expected = jnp.broadcast_to(jnp.asarray([0, 0, 1, 1]), (2, 4))
    np.testing.assert_array_equal(np.asarray(mixed_batch.policy_group), np.asarray(expected))
    assert not _same_tree(mixed.self_policy_state, runner.self_policy_state)


def test_sparse_anchor_rollout_matches_full_recording() -> None:
    """P0 sparse capture preserves rollout data and the CRN anchor sample."""

    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.training_app import _anchor_functions
    from src.delta_zsc.anchors import collect_anchor_batch, make_anchor_batch_kernel
    from src.delta_zsc.runner import (
        collect_rollout,
        initialize_runner,
        make_anchor_snapshot_rollout_kernel,
        make_compact_training_rollout_kernel,
    )

    config, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner,
        random_key=jax.random.PRNGKey(40),
    )
    legacy_runner, legacy_batch, records = collect_rollout(
        state=runner,
        length=4,
        environment=environment,
        model=model,
        base_params=base,
        latent_params=latent,
        partner_functions=partner,
        partner_parameters=None,
        official_shaping_factor=0.25,
        record_anchors=True,
        mixed_training_partners=True,
    )
    compact = make_compact_training_rollout_kernel(
        environment=environment,
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        length=4,
    )
    compact_runner, compact_batch, no_snapshots = compact(
        runner, base, latent, jnp.asarray(0.25), jax.random.PRNGKey(0)
    )
    assert no_snapshots is None
    assert _same_tree(legacy_runner, compact_runner)
    assert _same_tree(legacy_batch, compact_batch)

    anchor_key = jax.random.PRNGKey(41)
    index_key, root_key = jax.random.split(anchor_key)
    sparse = make_anchor_snapshot_rollout_kernel(
        environment=environment,
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        length=4,
        states_per_trigger=2,  # pilot candidates, narrowed to 1 below
    )
    sparse_runner, sparse_batch, snapshots = sparse(
        runner, base, latent, jnp.asarray(0.25), index_key
    )
    assert _same_tree(legacy_runner, sparse_runner)
    assert _same_tree(legacy_batch, sparse_batch)
    functions = _anchor_functions(
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        environment=environment,
    )
    # Both paths run the production two-tier configuration: a wide pilot,
    # then the registered measurement on the narrowed set.  Parity at
    # pilot_replicas=0 alone would leave the shipped path untested.
    legacy_anchors = collect_anchor_batch(
        key=anchor_key,
        records=records,
        functions=functions,
        base_params=base,
        latent_params=latent,
        states_per_trigger=1,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=1,
        horizon=1,
        gamma=0.99,
        pilot_states=2,
        pilot_replicas=2,
        xp_lanes_only=True,
    )
    sparse_anchor_kernel = make_anchor_batch_kernel(
        functions=functions,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=1,
        horizon=1,
        collect_successor=True,
        states_per_trigger=1,
        pilot_replicas=2,
    )
    sparse_anchors = sparse_anchor_kernel(
        root_key, snapshots, base, latent, jnp.asarray(0.99)
    )
    assert _same_tree(legacy_anchors, sparse_anchors)


def _same_tree(left, right) -> bool:
    """Compare two trees leaf by leaf.

    Integer and boolean leaves must match exactly: those carry indexes, masks
    and identities where any difference is a real disagreement.  Floating-point
    leaves are compared to a tight tolerance instead of bit-for-bit, because the
    sparse and full-recording paths evaluate the same arithmetic inside two
    differently shaped jit graphs.  Since the anchor continuation began
    bootstrapping its truncation with the value network, XLA fuses those graphs
    differently and the float32 results agree to ~1e-7 rather than exactly.
    """

    import jax

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    for one, other in zip(left_leaves, right_leaves):
        a = np.asarray(jax.device_get(one))
        b = np.asarray(jax.device_get(other))
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.floating):
            if not np.allclose(a, b, rtol=1.0e-5, atol=1.0e-6):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def test_checkpoint_and_deployment_round_trip(tmp_path: Path) -> None:
    import jax

    from experiments.overcooked_v2.deployment import (
        export_deployment_bundle,
        load_deployment,
    )
    from src.delta_zsc.optimizer import init_adam
    from src.delta_zsc.base_policy import trainable_base_params
    from src.delta_zsc.semantic_initializer import deterministic_simplex_initializer
    from src.delta_zsc.storage import load_latest_checkpoint, save_checkpoint
    from src.delta_zsc.types import TrainState
    from src.delta_zsc.runner import initialize_runner

    config, model, base, latent = _model()
    runner = initialize_runner(
        environment=MockEnvironment(),
        model=model,
        partner_functions=_partner_functions(),
        random_key=jax.random.PRNGKey(3),
    )
    from src.delta_zsc.training import init_latent_optimizer

    state = TrainState(
        base,
        latent,
        latent,
        init_adam(trainable_base_params(base)),
        init_latent_optimizer(latent),
        None,
        runner,
        1,
        1024,
        __import__("src.delta_zsc.resources", fromlist=["ResourceLedger"]).ResourceLedger().to_mapping(),
    )
    identity = {"method": "test", "seed": 0}
    save_checkpoint(tmp_path / "checkpoints", step=1024, state=state, identity=identity)
    step, restored = load_latest_checkpoint(
        tmp_path / "checkpoints", expected_identity=identity
    )
    assert step == 1024
    assert _same_tree(restored.base_params, base)

    bundle = export_deployment_bundle(
        tmp_path / "deployment",
        ego_run_id="ego-0",
        config=config,
        observation_shape=(5, 5, 39),
        base_params=base,
        latent_params=latent,
        source_training_run=tmp_path,
        semantic_initializer=deterministic_simplex_initializer(4, 31).to_mapping(
            include_bias=False
        ),
    )
    loaded = load_deployment(bundle)
    assert loaded.ego_run_id == "ego-0"
    assert _same_tree(loaded.latent_params, latent)
    assert loaded.semantic_initializer["uses_partner_labels"] is False
    assert loaded.semantic_initializer["event_component_bias_shape"] == [4, 31]
    assert loaded.execution_mode == "active"

    import json

    descriptor = bundle / "deployment_bundle.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["version"] = 4
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="method/schema identity"):
        load_deployment(bundle)


def test_all_deployment_modes_preserve_reference_kl_and_passive_probe_state() -> None:
    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.deployment import (
        Deployment,
        EXECUTION_MODES,
        deployment_action,
    )
    from src.delta_zsc.losses import categorical_log_probability
    from src.delta_zsc.mirror_policy import (
        MIRROR_UNCERTAINTY_PENALTY,
        categorical_kl_from_logits,
        project_policy_logits,
        robust_mirror_policy_logits,
    )

    config, model, base, latent = _model()
    # Exercise the projection rather than relying on the exact-zero residual at
    # initialization.  The immutable reference subtree itself is unchanged.
    trainable = base["trainable"]
    residual = trainable["residual_actor"]
    base = {
        **base,
        "trainable": {
            **trainable,
            "residual_actor": {
                **residual,
                "bias": jnp.asarray(
                    [12.0, -12.0, 8.0, -8.0, 4.0, -4.0], dtype=jnp.float32
                ),
            },
        },
    }
    deployment = Deployment(
        ego_run_id="fixture",
        config=config,
        model=model,
        base_params=base,
        latent_params=latent,
        execution_mode="active",
        semantic_initializer={},
    )
    observation = jnp.zeros((1, 5, 5, 39), dtype=jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(91), 1)
    budget = float(config.method.adaptation_kl_budget)

    for mode in EXECUTION_MODES:
        stepped, action, output, logp = deployment_action(
            deployment=deployment,
            state=model.initial_state(1),
            observation=observation,
            keys=keys,
            execution_mode=mode,
        )
        if mode == "reference_only":
            selected = output.reference_policy_logits
        elif mode == "residual":
            selected = output.base_policy_logits
        elif mode == "passive":
            mirror, _, _ = robust_mirror_policy_logits(
                output.base_policy_logits,
                output.expected_decision_values,
                output.expected_decision_variances**0.5,
                kl_budget=budget,
                uncertainty_penalty=MIRROR_UNCERTAINTY_PENALTY,
            )
            selected, _, _ = project_policy_logits(
                output.reference_policy_logits,
                mirror,
                kl_budget=budget,
            )
        else:
            selected = output.policy_logits

        expected_logp = categorical_log_probability(selected, action)
        np.testing.assert_allclose(
            np.asarray(logp), np.asarray(expected_logp), atol=1.0e-6
        )
        final_kl = categorical_kl_from_logits(
            selected, output.reference_policy_logits
        )
        assert float(jnp.max(final_kl)) <= budget + 1.0e-5
        if mode == "passive":
            assert not bool(stepped.probe_continuation_pending[0])

    assert bool(
        deployment_action(
            deployment=deployment,
            state=model.initial_state(1),
            observation=observation,
            keys=keys,
            execution_mode="active",
        )[0].probe_continuation_pending[0]
    )


def test_decision_chain_replays_modes_and_posterior_paths_on_matched_keys() -> None:
    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.intervention_app import (
        _decision_chain,
        _episode_shuffled_belief,
        _latent_history,
        _sample_actions,
    )
    from src.delta_zsc.model import EXECUTION_MODES
    from src.delta_zsc.runner import collect_rollout, initialize_runner

    config, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    reference_environment_keys = None
    for mode_index, mode in enumerate(EXECUTION_MODES):
        runner = initialize_runner(
            environment=environment,
            model=model,
            partner_functions=partner,
            random_key=jax.random.PRNGKey(92),
        )
        _, batch, records = collect_rollout(
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
            use_deployment_policy=True,
            execution_mode=mode,
        )
        history = _latent_history(
            model=model,
            base_params=base,
            latent_params=latent,
            batch=batch,
        )
        chain = _decision_chain(
            model=model,
            base_params=base,
            latent_params=latent,
            initial_task_carry=batch.initial_policy_state.task_carry,
            observations=batch.observations[:-1],
            episode_starts=batch.episode_starts[:-1],
            behavior=history.behavior_features,
            belief=history.belief,
        )
        replay = _sample_actions(
            records["ego_action_keys"], chain["final_policy_logits"][mode_index]
        )
        np.testing.assert_array_equal(np.asarray(replay), np.asarray(batch.actions))
        np.testing.assert_allclose(
            np.asarray(chain["final_policy_logits"][mode_index]),
            np.asarray(records["deployment_policy_logits"]),
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        assert chain["final_policy_logits"].shape == (4, 2, 4, 6)
        assert float(jnp.max(chain["final_reference_kl"])) <= (
            config.method.adaptation_kl_budget + 1.0e-5
        )
        current_keys = np.asarray(records["environment_keys"])
        if reference_environment_keys is None:
            reference_environment_keys = current_keys
        else:
            np.testing.assert_array_equal(current_keys, reference_environment_keys)

    posterior = np.asarray(history.belief)
    shuffled = _episode_shuffled_belief(
        posterior,
        np.asarray(batch.episode_starts[:-1]),
        seed=0,
    )
    assert shuffled.shape == posterior.shape
    np.testing.assert_allclose(
        np.sort(shuffled, axis=0), np.sort(posterior, axis=0), atol=0.0
    )


def test_final_v4_audit_reduces_probe_axis_before_anchor_mask() -> None:
    """Final report accepts [anchor, probe, K, event] logits without broadcasting."""

    import jax
    import numpy as np

    from experiments.overcooked_v2.training_app import (
        _anchor_functions,
        _final_decision_audit,
    )
    from src.delta_zsc.anchors import collect_anchor_batch
    from src.delta_zsc.runner import collect_rollout, initialize_runner

    _, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner,
        random_key=jax.random.PRNGKey(70),
    )
    _, batch, records = collect_rollout(
        state=runner,
        length=4,
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
    functions = _anchor_functions(
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        environment=environment,
    )
    anchors = collect_anchor_batch(
        key=jax.random.PRNGKey(71),
        records=records,
        functions=functions,
        base_params=base,
        latent_params=latent,
        states_per_trigger=2,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=2,
        horizon=1,
        gamma=0.99,
    )
    audit = _final_decision_audit(
        model=model,
        base_params=base,
        latent_params=latent,
        batch=batch,
        anchors=anchors,
    )
    assert audit["schema_version"] == 4
    assert audit["anchor_count"] == 2
    assert audit["valid_probe_count"] >= 0
    for name in (
        "probe_component_event_js",
        "mean_voi_action_spread",
        "mean_information_gain_action_spread",
        "mean_active_passive_policy_tv",
        "mean_filter_kl",
    ):
        assert np.isfinite(audit[name]), name


def test_pilot_narrows_the_measured_anchor_set() -> None:
    """End to end: the batch that reaches the loss is the narrowed one.

    The rollout emits the wide candidate set and the anchor kernel keeps only
    the states the cheap pass found separable, so the expensive replicas are
    never spent on a state where every action leads to the same trajectory.
    """

    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.training_app import _anchor_functions
    from src.delta_zsc.anchors import make_anchor_batch_kernel
    from src.delta_zsc.runner import (
        initialize_runner,
        make_anchor_snapshot_rollout_kernel,
    )

    config, model, base, latent = _model()
    environment = MockEnvironment()
    partner = _partner_functions()
    runner = initialize_runner(
        environment=environment,
        model=model,
        partner_functions=partner,
        random_key=jax.random.PRNGKey(41),
    )
    rollout = make_anchor_snapshot_rollout_kernel(
        environment=environment,
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        length=4,
        states_per_trigger=4,
    )
    _, _, snapshots = rollout(
        runner, base, latent, jnp.asarray(0.25), jax.random.PRNGKey(1)
    )
    assert snapshots.time_indexes.shape == (4,)

    functions = _anchor_functions(
        model=model,
        partner_functions=partner,
        partner_parameters=None,
        environment=environment,
    )
    kernel = make_anchor_batch_kernel(
        functions=functions,
        action_count=6,
        fit_replicas=2,
        evaluation_replicas=1,
        horizon=1,
        collect_successor=False,
        states_per_trigger=1,
        pilot_replicas=2,
    )
    batch = kernel(jax.random.PRNGKey(2), snapshots, base, latent, jnp.asarray(0.99))
    assert batch.time_indexes.shape == (1,)
    assert batch.contrast_mean.shape == (1, 6, 6)
    kept = (int(batch.time_indexes[0]), int(batch.lane_indexes[0]))
    candidates = {
        (int(t), int(l))
        for t, l in zip(snapshots.time_indexes, snapshots.lane_indexes)
    }
    assert kept in candidates


def test_formal_anchor_candidate_sampler_is_exactly_without_replacement() -> None:
    import jax
    import numpy as np

    from src.delta_zsc.anchors import select_anchor_indexes

    time, lane = jax.jit(
        lambda key: select_anchor_indexes(
            key,
            time_count=256,
            environment_count=256,
            requested=256,
        )
    )(jax.random.PRNGKey(7))
    flat = np.asarray(time) * 256 + np.asarray(lane)
    assert flat.shape == (256,)
    assert len(np.unique(flat)) == 256
    assert np.all((0 <= flat) & (flat < 256 * 256))
