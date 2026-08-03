"""End-to-end mechanism tests for the DEPI core.

Specification entries covered (docs/METHOD_SPEC.md):
- §2 replay bookkeeping stays valid under the new three-object architecture
  (anchor rows now carry capability/protocol contexts via PolicyState).
- Partner generator mechanics (mixture admission, episode validation,
  fixed-behavior PPO) stay importable; the learned generator is rejected by
  the §7.3 boundary (static wide partner pool is the default), so these
  modules are exercised here but disconnected from the default path.

Deprecated mechanics removed from this file per §3.3 (belief gradient
routing / PCGrad / normalized gradient caps are abolished).
"""

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from src.path_c.anchor_replay import (  # noqa: E402
    append_replay_batch,
    policy_drift_age_weights,
    sample_replay_batch,
)
from src.path_c.partner_episode import (  # noqa: E402
    PartnerEpisodeBatch,
    generator_ppo_objective,
    generator_weight_schedule,
    validate_complete_episode_batch,
)
from src.path_c.partner_sources import soft_generator_mixture  # noqa: E402
from src.path_c.model import initial_policy_state  # noqa: E402
from src.path_c.types import CounterfactualAnchorBatch  # noqa: E402


def _policy_state(rows: int):
    return initial_policy_state(
        batch_size=rows,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        protocol_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )


def _anchor_rows() -> CounterfactualAnchorBatch:
    count = 64
    pair_ids = np.full(count, -1, dtype=np.int32)
    for pair in range(16):
        pair_ids[32 + 2 * pair : 34 + 2 * pair] = pair
    returns = np.arange(count * 6, dtype=np.float32).reshape((count, 6)) / 17.0
    return CounterfactualAnchorBatch(
        anchor_ids=jnp.arange(count),
        rollout_flat_indexes=jnp.arange(count),
        policy_states=_policy_state(count),
        observations=jnp.zeros((count, 5, 5, 39)),
        partner_codes=jnp.zeros((count, 8)),
        partner_sources=jnp.asarray(np.arange(count) % 3, dtype=jnp.int32),
        partner_run_ids=jnp.asarray(np.arange(count) % 8, dtype=jnp.int32),
        fit_returns_by_action=jnp.asarray(returns),
        return_sum_by_action=jnp.asarray(returns * 4.0),
        return_squared_sum_by_action=jnp.asarray(returns * returns * 4.0),
        replica_count=jnp.full((count, 6), 4, dtype=jnp.int32),
        collection_policy_logits=jnp.zeros((count, 6)),
        collection_update=jnp.asarray(np.repeat([0, 16], 32), dtype=jnp.int32),
        collection_target_fingerprint=jnp.zeros((count, 2), dtype=jnp.uint32),
        matched_pair_ids=jnp.asarray(pair_ids),
        action_mask=jnp.ones((count, 6), dtype=jnp.bool_),
    )


def test_replay_uses_soft_policy_drift_and_age_without_epoch_clear() -> None:
    state = append_replay_batch(None, batch=_anchor_rows(), capacity=512)
    state, sample = sample_replay_batch(
        state, key=jax.random.PRNGKey(4), batch_size=64
    )
    assert int(state.item_count) == 64
    assert int(jnp.sum(state.use_counts)) == 64
    matched = np.asarray(sample.matched_pair_ids)
    for index in range(32, 64, 2):
        assert matched[index] >= 0 and matched[index] == matched[index + 1]

    current = jnp.asarray([[5.0, -5.0], [5.0, -5.0], [5.0, -5.0]])
    collected = jnp.asarray([[5.0, -5.0], [0.0, 0.0], [5.0, -5.0]])
    weights = policy_drift_age_weights(
        current_policy_logits=current,
        collection_policy_logits=collected,
        current_update=jnp.asarray([20, 20, 20]),
        collection_update=jnp.asarray([20, 20, 0]),
    )
    assert float(weights[0]) > float(weights[1])
    assert float(weights[0]) > float(weights[2])
    assert np.all(np.asarray(weights) >= 1.0e-3 / np.mean(np.asarray(weights)))


def test_generator_mixture_is_continuous_and_has_no_admission_jump() -> None:
    start = soft_generator_mixture(
        progress=0.0, generator_cvar_ema=-100.0, reference_cvar_ema=20.0
    )
    low = soft_generator_mixture(
        progress=0.3, generator_cvar_ema=-100.0, reference_cvar_ema=20.0
    )
    equal = soft_generator_mixture(
        progress=0.3, generator_cvar_ema=20.0, reference_cvar_ema=20.0
    )
    high = soft_generator_mixture(
        progress=0.3, generator_cvar_ema=100.0, reference_cvar_ema=20.0
    )
    np.testing.assert_allclose(start, [0.0, 0.0, 1.0], atol=1e-7)
    assert float(low[0]) < float(equal[0]) < float(high[0])
    assert float(equal[0]) == pytest.approx(0.1875, abs=1e-6)
    for probabilities in (low, equal, high):
        assert float(jnp.sum(probabilities)) == pytest.approx(1.0, abs=1e-6)
    rho, imitation, diversity = generator_weight_schedule(jnp.asarray(0.15))
    assert float(rho) == pytest.approx(0.375)
    assert float(imitation) == pytest.approx(0.5)
    assert float(diversity) == pytest.approx(0.5)


def _generator_episode_batch() -> PartnerEpisodeBatch:
    time, count, code_dim = 400, 2, 3
    valid = np.zeros((time, count), dtype=bool)
    valid[:11] = True
    done = np.zeros((time, count), dtype=bool)
    done[10] = True
    codes = np.broadcast_to(
        np.asarray([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])[None],
        (time, count, code_dim),
    ).copy()
    return PartnerEpisodeBatch(
        observations=np.zeros((time, count, 5, 5, 39), dtype=np.float32),
        actions=np.zeros((time, count), dtype=np.int32),
        raw_rewards=np.zeros((time, count), dtype=np.float32),
        official_shaped_rewards=np.zeros((time, count), dtype=np.float32),
        dones=done,
        codes=codes,
        behavior_log_probabilities=np.zeros((time, count), dtype=np.float32),
        behavior_values=np.zeros((time, count), dtype=np.float32),
        initial_carries=np.zeros((count, 4), dtype=np.float32),
        parameter_version_ids=np.full((time, count), 17, dtype=np.int32),
        valid_mask=valid,
        ego_action_signatures=np.zeros((time, count, 6), dtype=np.float32),
    )


def test_generator_complete_episode_records_do_not_cross_done_code_or_version() -> None:
    batch = _generator_episode_batch()
    validate_complete_episode_batch(batch)
    codes = np.asarray(batch.codes).copy()
    codes[5, 0, 0] = 99.0
    with pytest.raises(ValueError, match="code changed"):
        validate_complete_episode_batch(batch._replace(codes=codes))
    versions = np.asarray(batch.parameter_version_ids).copy()
    versions[5, 0] = 18
    with pytest.raises(ValueError, match="parameters changed"):
        validate_complete_episode_batch(batch._replace(parameter_version_ids=versions))
    valid = np.asarray(batch.valid_mask).copy()
    valid[11, 0] = True
    with pytest.raises(ValueError, match="post-done"):
        validate_complete_episode_batch(batch._replace(valid_mask=valid))


def test_generator_ppo_uses_fixed_behavior_gae_and_official_value_clipping() -> None:
    logits = jnp.zeros((2, 1, 2), dtype=jnp.float32)
    loss, metrics = generator_ppo_objective(
        new_logits=logits,
        new_values=jnp.asarray([[0.7], [0.5]], dtype=jnp.float32),
        actions=jnp.asarray([[0], [1]], dtype=jnp.int32),
        behavior_log_probabilities=jnp.full((2, 1), -np.log(2.0)),
        behavior_values=jnp.asarray([[0.5], [1.0]], dtype=jnp.float32),
        official_shaped_rewards=jnp.asarray([[1.0], [2.0]], dtype=jnp.float32),
        dones=jnp.asarray([[False], [True]]),
        valid_mask=jnp.ones((2, 1), dtype=jnp.float32),
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        value_clip_epsilon=0.2,
        value_weight=0.5,
        entropy_weight=0.01,
        normalize_advantages=True,
    )
    expected_value_loss = 0.5 * ((2.62 - 0.7) ** 2 + (2.0 - 0.5) ** 2) / 2.0
    assert float(metrics["generator_actor_loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["generator_value_loss"]) == pytest.approx(
        expected_value_loss, abs=1e-6
    )
    assert float(loss) == pytest.approx(
        0.5 * expected_value_loss - 0.01 * np.log(2.0), abs=1e-6
    )
