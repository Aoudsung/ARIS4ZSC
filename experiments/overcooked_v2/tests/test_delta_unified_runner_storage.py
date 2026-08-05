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
            voi_quadrature_samples=4,
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
    assert records["active_voi_raw"].shape == (4, 4, 6)
    assert records["active_voi_quadrature_error"].shape == (4, 4, 6)
    # Training is exactly base-policy collection: no active value is computed
    # and no analytic deployment adjustment enters the behavior distribution.
    np.testing.assert_allclose(np.asarray(records["active_voi_raw"]), 0.0, atol=0.0)
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
    )
    assert anchors.fit_returns_by_action.shape == (2, 6)
    assert anchors.measurement_covariances.shape == (2, 5, 5)
    assert bool(jnp.all(jnp.isfinite(anchors.measurement_covariances)))


def test_checkpoint_and_deployment_round_trip(tmp_path: Path) -> None:
    import jax

    from experiments.overcooked_v2.deployment import (
        export_deployment_bundle,
        load_deployment,
    )
    from src.delta_zsc.optimizer import init_adam
    from src.delta_zsc.storage import (
        load_latest_checkpoint,
        pytree_fingerprint,
        save_checkpoint,
    )
    from src.delta_zsc.types import TrainState
    from src.delta_zsc.runner import initialize_runner

    config, model, base, latent = _model()
    runner = initialize_runner(
        environment=MockEnvironment(),
        model=model,
        partner_functions=_partner_functions(),
        random_key=jax.random.PRNGKey(3),
    )
    state = TrainState(
        base,
        latent,
        init_adam(base),
        init_adam(latent),
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
    assert pytree_fingerprint(restored.base_params) == pytree_fingerprint(base)

    bundle = export_deployment_bundle(
        tmp_path / "deployment",
        ego_run_id="ego-0",
        config=config,
        observation_shape=(5, 5, 39),
        base_params=base,
        latent_params=latent,
        source_training_run=tmp_path,
    )
    loaded = load_deployment(bundle)
    assert loaded.ego_run_id == "ego-0"
    assert pytree_fingerprint(loaded.latent_params) == pytree_fingerprint(latent)
