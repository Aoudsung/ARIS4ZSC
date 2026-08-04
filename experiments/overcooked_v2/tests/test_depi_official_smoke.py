from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxmarl")

from experiments.overcooked_v2.official_adapter import (  # noqa: E402
    ACTION_ORDER,
    VectorEnvironment,
    _official_symbol,
    official_pairing_rollouts,
)
from experiments.overcooked_v2.official_br_prox_app import (  # noqa: E402
    empirical_all_action_continuations_from_snapshots,
    record_official_history_anchor_snapshots,
)
from src.path_c.anchor_sampling import (  # noqa: E402
    fit_pair_comparator,
    pair_legal_history_features,
    pair_feature_rows,
    predict_pair_comparator,
)
from src.path_c.decision_geometry import centered_action_values  # noqa: E402
from src.path_c.experiment import load_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def test_official_environment_reset_and_step_smoke() -> None:
    config = load_config(
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "configs"
        / "depi_simple_development.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        environment=replace(config.environment, num_envs=2, episode_steps=8),
    )
    environment = VectorEnvironment.create(config)
    state, observations = environment.reset(jax.random.PRNGKey(1))
    assert observations.shape[:2] == (2, 2)
    assert environment.observation_shape == tuple(observations.shape[2:])
    actions = jnp.full((2, 2), 4, dtype=jnp.int32)
    next_state, next_observations, rewards, dones, info = environment.step(
        state, actions, jax.random.PRNGKey(2)
    )
    del next_state
    assert next_observations.shape == observations.shape
    assert rewards.shape == dones.shape == (2,)
    assert info["terminal_observations"].shape == observations.shape
    assert np.all(np.isfinite(np.asarray(rewards)))
    assert ACTION_ORDER == ("right", "down", "left", "up", "stay", "interact")


def test_vectorized_pairing_wrapper_is_trajectory_identical_to_official_get_rollout() -> None:
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType, OvercookedV2

    class RandomPolicy:
        def init_hstate(self, batch_size, key=None):
            del batch_size, key
            return None

        def compute_action(self, obs, done, hstate, key):
            del obs, done
            return jax.random.randint(key, (), 0, 6), hstate

    environment = OvercookedV2(
        layout="test_time_simple",
        observation_type=ObservationType.DEFAULT,
        agent_view_size=2,
        negative_rewards=True,
        random_agent_positions=True,
        sample_recipe_on_delivery=True,
        indicate_successful_delivery=True,
        max_steps=400,
    )
    root = jax.random.PRNGKey(0)
    left = RandomPolicy()
    right = RandomPolicy()
    observed, keys = official_pairing_rollouts(
        left_policy=left,
        right_policy=right,
        environment=environment,
        root_key=root,
        episodes=2,
    )
    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    get_rollout = _official_symbol(
        "overcooked_v2_experiments.eval.rollout", "get_rollout"
    )
    expected = jax.vmap(
        lambda key: get_rollout(PolicyPairing(left, right), environment, key)
    )(jax.random.split(root, 2))
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(jax.random.split(root, 2)))
    np.testing.assert_array_equal(
        np.asarray(observed.actions_seq["agent_0"]),
        np.asarray(expected.actions_seq["agent_0"]),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.actions_seq["agent_1"]),
        np.asarray(expected.actions_seq["agent_1"]),
    )
    np.testing.assert_allclose(
        np.asarray(observed.total_reward), np.asarray(expected.total_reward)
    )


def test_real_comparator_history_collector_preserves_past_and_future_windows() -> None:
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType, OvercookedV2

    class RandomPolicy:
        def init_hstate(self, batch_size, key=None):
            del batch_size, key
            return None

        def compute_action(self, obs, done, hstate, key):
            del obs, done
            return jax.random.randint(key, (), 0, 6), hstate

    environment = OvercookedV2(
        layout="test_time_simple",
        observation_type=ObservationType.DEFAULT,
        agent_view_size=2,
        negative_rewards=True,
        random_agent_positions=True,
        sample_recipe_on_delivery=True,
        indicate_successful_delivery=True,
        max_steps=40,
    )
    policy = RandomPolicy()
    recorded = record_official_history_anchor_snapshots(
        left=policy,
        right=policy,
        ego_role=0,
        environment=environment,
        root_key=jax.random.PRNGKey(7),
        anchors=2,
        history_steps=4,
        continuation_horizon=3,
        episodes=3,
    )
    assert recorded["history"].ego_actions.shape == (2, 4)
    assert recorded["history"].ego_observations.shape[:2] == (2, 4)
    assert np.all(np.asarray(recorded["time_indexes"]) >= 4)
    assert np.all(np.asarray(recorded["time_indexes"]) <= 37)
    features = pair_legal_history_features(recorded["history"])
    assert features.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(features)))


def test_real_official_continuations_fit_and_validate_pair_comparator() -> None:
    """Exercise collector -> all-action continuation -> frozen fit end to end."""

    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType, OvercookedV2

    class RandomPolicy:
        def init_hstate(self, batch_size, key=None):
            del batch_size, key
            return None

        def compute_action(self, obs, done, hstate, key):
            del obs, done
            return jax.random.randint(key, (), 0, 6), hstate

    environment = OvercookedV2(
        layout="test_time_simple",
        observation_type=ObservationType.DEFAULT,
        agent_view_size=2,
        negative_rewards=True,
        random_agent_positions=True,
        sample_recipe_on_delivery=True,
        indicate_successful_delivery=True,
        max_steps=48,
    )
    policy = RandomPolicy()

    def collect(root):
        recorded = record_official_history_anchor_snapshots(
            left=policy,
            right=policy,
            ego_role=0,
            environment=environment,
            root_key=jax.random.PRNGKey(root),
            anchors=8,
            history_steps=4,
            continuation_horizon=8,
            episodes=8,
        )
        continuation = empirical_all_action_continuations_from_snapshots(
            left=policy,
            right=policy,
            ego_role=0,
            environment=environment,
            selected=recorded["selected"],
            anchor_roots=recorded["anchor_roots"],
            fit_replicas=2,
            evaluation_replicas=2,
            continuation_horizon=8,
        )
        return (
            np.asarray(pair_legal_history_features(recorded["history"])),
            np.asarray(
                centered_action_values(continuation["fit_returns_by_action"])
            ),
            np.asarray(
                centered_action_values(
                    continuation["evaluation_returns_by_action"]
                )
            ),
        )

    fit_features, fit_signatures, _ = collect(17)
    validation_features, _, validation_signatures = collect(23)

    assert fit_signatures.shape == validation_signatures.shape == (8, 6)
    assert np.all(np.isfinite(fit_signatures))
    assert np.all(np.isfinite(validation_signatures))

    # Random policies may produce no raw delivery in a short CPU smoke horizon,
    # so label semantics are protected by the dedicated K-independent unit test.
    # Here real simulator histories and continuations exercise the wiring while
    # a balanced deterministic fixture makes the optimizer/validation path live.
    def pair_fixture(features, prefix):
        rows = []
        labels = []
        blocks = []
        pair_index = 0
        for left in range(features.shape[0]):
            for right in range(left + 1, features.shape[0]):
                rows.append(pair_feature_rows(features[left], features[right]))
                labels.append(float(pair_index % 2))
                blocks.append(f"{prefix}-{left}|{prefix}-{right}")
                pair_index += 1
        return np.stack(rows), np.asarray(labels), np.asarray(blocks)

    fit_rows, fit_labels, fit_blocks = pair_fixture(fit_features, "fit")
    validation_rows, validation_labels, validation_blocks = pair_fixture(
        validation_features, "validation"
    )
    comparator = fit_pair_comparator(
        fit_rows,
        fit_labels,
        validation_pair_features=validation_rows,
        validation_pair_labels=validation_labels,
        history_feature_dim=fit_features.shape[1],
        training_block_ids=fit_blocks,
        validation_block_ids=validation_blocks,
        bootstrap_replicates=25,
        logistic_iterations=25,
    )
    probabilities = predict_pair_comparator(comparator, validation_rows)
    assert probabilities.shape == validation_labels.shape
    assert np.all(np.isfinite(probabilities))
    assert comparator.validation_row_count == validation_rows.shape[0]
