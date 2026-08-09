"""Belief-conditioned critic, pairwise CRN contrasts, robust mirror control."""

from __future__ import annotations

import numpy as np


def test_advantage_is_centered_under_the_acting_policy() -> None:
    """The value head carries the level; the advantage carries only contrast."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.belief_value import belief_value_predict, init_belief_value_params

    params = init_belief_value_params(
        jax.random.PRNGKey(0),
        task_dim=8,
        instant_dim=4,
        behavior_dim=6,
        component_count=4,
        hidden_dim=16,
        action_count=6,
        ensemble_size=3,
    )
    lanes = 5
    task = jax.random.normal(jax.random.PRNGKey(1), (lanes, 8))
    instant = jax.random.normal(jax.random.PRNGKey(2), (lanes, 4))
    behavior = jax.random.normal(jax.random.PRNGKey(3), (lanes, 6))
    belief = jnp.full((lanes, 4), 0.25)
    policy = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(4), (lanes, 6)), axis=-1)

    prediction = belief_value_predict(params, task, instant, behavior, belief, policy)
    weighted = jnp.sum(prediction.advantage_mean * policy, axis=-1)
    np.testing.assert_allclose(np.asarray(weighted), 0.0, atol=1.0e-5)
    assert prediction.ensemble_advantage.shape == (3, lanes, 6)
    np.testing.assert_allclose(
        np.asarray(prediction.action_value),
        np.asarray(prediction.value_mean)[:, None] + np.asarray(prediction.advantage_mean),
        atol=1.0e-6,
    )


def test_belief_changes_the_predicted_contrast() -> None:
    """Swapping the posterior must move the action contrast.

    A critic that ignores b_t would make the whole belief-conditioned design
    vacuous, which is exactly the failure the previous decision head had.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.belief_value import belief_value_predict, init_belief_value_params

    params = init_belief_value_params(
        jax.random.PRNGKey(7),
        task_dim=8,
        instant_dim=4,
        behavior_dim=6,
        component_count=4,
        hidden_dim=16,
        action_count=6,
        ensemble_size=2,
    )
    task = jax.random.normal(jax.random.PRNGKey(1), (1, 8))
    instant = jax.random.normal(jax.random.PRNGKey(2), (1, 4))
    behavior = jax.random.normal(jax.random.PRNGKey(3), (1, 6))

    left = belief_value_predict(
        params, task, instant, behavior, jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    )
    right = belief_value_predict(
        params, task, instant, behavior, jnp.asarray([[0.0, 0.0, 0.0, 1.0]])
    )
    difference = np.abs(
        np.asarray(left.advantage_mean) - np.asarray(right.advantage_mean)
    ).max()
    assert difference > 1.0e-6


def test_same_replica_differencing_cancels_shared_noise() -> None:
    """A common per-replica offset must not survive the contrast."""

    import jax.numpy as jnp

    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    truth = jnp.asarray([[0.0, 1.0, 2.0]])
    shared = jnp.asarray([[[10.0, -4.0, 7.0, 0.5]]])  # one offset per replica
    replicas = truth[..., None] + shared
    contrast = pairwise_contrasts_from_replicas(
        replicas, jnp.ones((1, 3), dtype=bool)
    )
    expected = np.asarray(truth)[..., :, None] - np.asarray(truth)[..., None, :]
    np.testing.assert_allclose(np.asarray(contrast.mean), expected, atol=1.0e-5)
    # Shared noise cancels exactly, so every pair is resolvable.
    valid = np.asarray(contrast.valid)
    np.testing.assert_allclose(
        np.asarray(contrast.standard_error)[valid], 0.0, atol=1.0e-5
    )


def test_unresolvable_pairs_receive_negligible_weight() -> None:
    """Noise-dominated pairs must not be handed an arbitrary winner."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.contrast import (
        pairwise_contrasts_from_replicas,
        resolvable_pair_statistics,
    )

    # Two actions differ by far less than the replica spread.
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16)) * 5.0
    replicas = jnp.asarray([[[0.0], [1.0e-4]]]) + noise
    contrast = pairwise_contrasts_from_replicas(
        replicas, jnp.ones((1, 2), dtype=bool)
    )
    stats = resolvable_pair_statistics(contrast)
    assert float(stats["resolvable_fraction"]) == 0.0

    clean = jnp.asarray([[[0.0] * 16, [50.0] * 16]])
    resolvable = pairwise_contrasts_from_replicas(clean, jnp.ones((1, 2), dtype=bool))
    assert float(resolvable_pair_statistics(resolvable)["resolvable_fraction"]) == 1.0


def test_contrast_loss_ignores_state_level_offsets() -> None:
    """Only differences are fitted, matching what the mirror step consumes."""

    import jax.numpy as jnp

    from src.delta_zsc.contrast import (
        contrast_regression_loss,
        pairwise_contrasts_from_replicas,
    )

    replicas = jnp.asarray([[[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]])
    contrast = pairwise_contrasts_from_replicas(
        replicas, jnp.ones((1, 3), dtype=bool)
    )
    advantage = jnp.asarray([[0.0, 2.0, 5.0]])
    exact, _ = contrast_regression_loss(advantage, contrast)
    shifted, _ = contrast_regression_loss(advantage + 100.0, contrast)
    np.testing.assert_allclose(float(exact), float(shifted), atol=1.0e-6)
    assert float(exact) < 1.0e-6


def test_robust_mirror_shrinks_toward_base_when_the_ensemble_disagrees() -> None:
    """Uncertainty must cost KL, not merely be reported.

    The unmodified solver is invariant to positive rescaling of the advantage,
    so it spends the whole budget on a contrast of any magnitude.  With a
    dispersion that swamps the mean the robust objective must stay closer to the
    base policy than the plain one.
    """

    import jax.numpy as jnp

    from src.delta_zsc.mirror_policy import (
        mirror_policy_logits,
        robust_mirror_policy_logits,
    )

    base = jnp.zeros((1, 6))
    advantage = jnp.asarray([[0.0, 0.02, -0.01, 0.0, 0.0, 0.0]])
    confident = jnp.zeros((1, 6))
    uncertain = jnp.full((1, 6), 1.0)

    _, plain_kl, _ = mirror_policy_logits(base, advantage, kl_budget=0.04)
    _, sure_kl, _ = robust_mirror_policy_logits(
        base, advantage, confident, kl_budget=0.04, uncertainty_penalty=1.0
    )
    _, unsure_kl, _ = robust_mirror_policy_logits(
        base, advantage, uncertain, kl_budget=0.04, uncertainty_penalty=1.0
    )
    assert float(sure_kl[0]) > float(unsure_kl[0])
    np.testing.assert_allclose(float(sure_kl[0]), float(plain_kl[0]), atol=1.0e-6)


def test_td_lambda_targets_match_a_direct_discounted_sum() -> None:
    """lambda=1 with no termination is the plain discounted return."""

    import jax.numpy as jnp

    from src.delta_zsc.belief_value import td_lambda_targets

    rewards = jnp.asarray([[1.0], [2.0], [3.0]])
    values = jnp.zeros((4, 1))
    dones = jnp.zeros((3, 1))
    targets = td_lambda_targets(rewards, values, dones, gamma=0.5, lambda_=1.0)
    expected = [1.0 + 0.5 * (2.0 + 0.5 * 3.0), 2.0 + 0.5 * 3.0, 3.0]
    np.testing.assert_allclose(np.asarray(targets)[:, 0], expected, atol=1.0e-6)


def test_td_lambda_stops_at_termination() -> None:
    """A done transition must not bootstrap across the episode boundary."""

    import jax.numpy as jnp

    from src.delta_zsc.belief_value import td_lambda_targets

    rewards = jnp.asarray([[1.0], [100.0]])
    values = jnp.asarray([[0.0], [0.0], [999.0]])
    dones = jnp.asarray([[1.0], [0.0]])
    targets = td_lambda_targets(rewards, values, dones, gamma=0.9, lambda_=1.0)
    np.testing.assert_allclose(float(targets[0, 0]), 1.0, atol=1.0e-6)


def test_pilot_scores_rank_separable_states_above_flat_ones() -> None:
    """The pilot must answer "is there anything to measure here?".

    Diagnostic 1 found no anchor pair separated by two standard errors and an
    exact tie in every row: the replica budget was buying a precise estimate of
    zero.  A state whose actions genuinely differ must outrank one whose
    actions are interchangeable, or the two-tier pass is just a slower uniform
    draw.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.anchors import select_by_pilot_score
    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    # Three states: flat, noisy-but-flat, genuinely separated.
    noise = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 8)) * 0.01
    means = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.001],
            [0.0, 5.0, -5.0, 1.0],
        ]
    )
    returns = means[..., None] + noise
    contrast = pairwise_contrasts_from_replicas(
        returns, jnp.ones((3, 4), dtype=bool)
    )
    ratio = jnp.abs(contrast.mean) / jnp.maximum(contrast.standard_error, 1.0e-6)
    scores = jnp.max(jnp.where(contrast.valid, ratio, 0.0), axis=(-2, -1))
    assert int(jnp.argmax(scores)) == 2
    assert int(select_by_pilot_score(scores, 1)[0]) == 2
    # And the ordering is by resolvability, not by raw contrast magnitude
    # alone: the flat state carries no signal at any noise level.
    assert float(scores[2]) > float(scores[1]) > 0.0



def test_exact_ties_do_not_dominate_the_contrast_loss() -> None:
    """A re-merged pair must not outweigh a measured one by orders of magnitude.

    Under CRN two actions whose continuations converge give bit-identical
    returns in every replica, so the sample variance is exactly zero.  Plain
    inverse-variance weighting turns that into near-unbounded confidence that
    the two actions are equal -- on the first measured batch, 1e6 against 1e3
    for a genuinely resolvable pair.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    # Action 0 vs 1 re-merge exactly; action 2 is separated and noisy.
    replicas = jnp.stack(
        [
            jnp.zeros((8,)),
            jnp.zeros((8,)),
            5.0 + jax.random.normal(jax.random.PRNGKey(0), (8,)) * 0.5,
        ]
    )[None]
    contrast = pairwise_contrasts_from_replicas(
        replicas, jnp.ones((1, 3), dtype=bool)
    )
    weights = np.asarray(contrast.precision_weights())[0]
    tied = weights[0, 1]
    measured = weights[0, 2]
    assert np.isfinite(tied)
    assert tied / measured < 100.0
    # Precision still orders the pairs the right way round.
    assert tied > measured


def test_precision_weight_recovers_inverse_variance_when_all_pairs_are_measured() -> None:
    """Shrinkage must not distort a batch with no degenerate pair.

    Where every pair carries comparable noise, the pooled term is a common
    factor and the relative weights are the inverse-variance ones.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    key = jax.random.PRNGKey(3)
    scales = jnp.asarray([1.0, 2.0, 4.0])
    replicas = (
        jnp.asarray([0.0, 10.0, 20.0])[:, None]
        + jax.random.normal(key, (3, 64)) * scales[:, None]
    )[None]
    contrast = pairwise_contrasts_from_replicas(
        replicas, jnp.ones((1, 3), dtype=bool)
    )
    weights = np.asarray(contrast.precision_weights())[0]
    variance = np.asarray(contrast.standard_error)[0] ** 2
    # The noisiest pair (1,2) must weigh less than the quietest (0,1).
    assert variance[1, 2] > variance[0, 1]
    assert weights[1, 2] < weights[0, 1]


def _official_tree(key, *, channels: int = 39, hidden: int = 128, actions: int = 6):
    """A synthetic Official parameter tree with the pinned shapes."""

    import jax
    import jax.numpy as jnp

    keys = iter(jax.random.split(key, 64))

    def block(shape):
        return jax.random.normal(next(keys), shape) * 0.1

    conv_specs = [
        ((1, 1), channels, 128),
        ((1, 1), 128, 128),
        ((1, 1), 128, 8),
        ((3, 3), 8, 16),
        ((3, 3), 16, 32),
        ((3, 3), 32, 32),
    ]
    cnn = {
        f"Conv_{index}": {
            "kernel": block(size + (inp, out)),
            "bias": block((out,)),
        }
        for index, (size, inp, out) in enumerate(conv_specs)
    }
    cnn["Dense_0"] = {"kernel": block((800, hidden)), "bias": block((hidden,))}
    gru = {
        name: {"kernel": block((hidden, hidden)), "bias": block((hidden,))}
        for name in ("ir", "iz", "in", "hn")
    }
    gru["hr"] = {"kernel": block((hidden, hidden))}
    gru["hz"] = {"kernel": block((hidden, hidden))}
    return {
        "CNN_0": cnn,
        "LayerNorm_0": {"scale": block((hidden,)), "bias": block((hidden,))},
        "ScannedRNN_0": {"GRUCell_1": gru},
        "Dense_0": {"kernel": block((hidden, hidden)), "bias": block((hidden,))},
        "Dense_1": {"kernel": block((hidden, actions)), "bias": block((actions,))},
        "Dense_2": {"kernel": block((hidden, hidden)), "bias": block((hidden,))},
        "Dense_3": {"kernel": block((hidden, 1)), "bias": block((1,))},
    }


def test_sp_transplant_copies_every_official_parameter() -> None:
    """No Official weight may be silently dropped or reshaped."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.base_policy import init_base_params
    from src.delta_zsc.official_initializer import transplant_official_base_params

    base = init_base_params(
        jax.random.PRNGKey(0),
        observation_shape=(5, 5, 39),
        task_hidden_dim=128,
        task_embedding_dim=128,
        instant_partner_dim=64,
        action_count=6,
        component_count=4,
    )
    official = _official_tree(jax.random.PRNGKey(1))
    moved = transplant_official_base_params(base, official)

    for index in range(6):
        np.testing.assert_allclose(
            np.asarray(moved["task_conv"][index]["kernel"]),
            np.asarray(official["CNN_0"][f"Conv_{index}"]["kernel"]),
        )
    np.testing.assert_allclose(
        np.asarray(moved["task_dense"]["kernel"]),
        np.asarray(official["CNN_0"]["Dense_0"]["kernel"]),
    )
    np.testing.assert_allclose(
        np.asarray(moved["actor"]["kernel"]), np.asarray(official["Dense_1"]["kernel"])
    )
    np.testing.assert_allclose(
        np.asarray(moved["value"]["kernel"]), np.asarray(official["Dense_3"]["kernel"])
    )
    # The task rows carry the Official actor; the instant and posterior rows
    # start at zero, so the transplanted policy *is* the Official policy.
    kernel = np.asarray(moved["actor_trunk"]["kernel"])
    np.testing.assert_allclose(
        kernel[:128], np.asarray(official["Dense_0"]["kernel"])
    )
    np.testing.assert_allclose(kernel[128:], 0.0)
    assert kernel.shape == (128 + 64 + 4, 128)


def test_sp_transplant_rejects_a_shape_mismatch() -> None:
    """A partial transplant would be neither Official nor freshly initialised."""

    import jax
    import pytest

    from src.delta_zsc.base_policy import init_base_params
    from src.delta_zsc.official_initializer import transplant_official_base_params

    base = init_base_params(
        jax.random.PRNGKey(0),
        observation_shape=(5, 5, 39),
        task_hidden_dim=128,
        task_embedding_dim=128,
        instant_partner_dim=64,
        action_count=6,
        component_count=4,
    )
    official = _official_tree(jax.random.PRNGKey(1), actions=5)
    with pytest.raises(ValueError, match="Dense_1/kernel"):
        transplant_official_base_params(base, official)


def test_delta_gru_matches_the_official_recurrent_cell() -> None:
    """The transplanted cell must compute what it was trained to compute."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.nn import gru_step, init_gru

    hidden_dim = 6
    params = init_gru(jax.random.PRNGKey(0), hidden_dim, hidden_dim)
    carry = jax.random.normal(jax.random.PRNGKey(1), (2, hidden_dim))
    value = jax.random.normal(jax.random.PRNGKey(2), (2, hidden_dim))

    # Flax nn.GRUCell, written out.
    def official(carry, x):
        r = jax.nn.sigmoid(
            x @ params["input_reset"]["kernel"]
            + params["input_reset"]["bias"]
            + carry @ params["hidden_reset"]["kernel"]
        )
        z = jax.nn.sigmoid(
            x @ params["input_update"]["kernel"]
            + params["input_update"]["bias"]
            + carry @ params["hidden_update"]["kernel"]
        )
        n = jnp.tanh(
            x @ params["input_candidate"]["kernel"]
            + params["input_candidate"]["bias"]
            + r
            * (
                carry @ params["hidden_candidate"]["kernel"]
                + params["hidden_candidate"]["bias"]
            )
        )
        return (1.0 - z) * n + z * carry

    np.testing.assert_allclose(
        np.asarray(gru_step(params, carry, value)),
        np.asarray(official(carry, value)),
        atol=1.0e-6,
    )


def _snapshot_stub(states: int, marker: float):
    import jax.numpy as jnp

    from src.delta_zsc.types import AnchorSnapshots

    return AnchorSnapshots(
        time_indexes=jnp.full((states,), int(marker), dtype=jnp.int32),
        lane_indexes=jnp.zeros((states,), dtype=jnp.int32),
        environment_state=jnp.full((states, 2), marker),
        observations=jnp.full((states, 2, 3), marker),
        ego_state=jnp.full((states, 4), marker),
        partner_state=jnp.full((states, 4), marker),
        partner_episode_start=jnp.zeros((states,), dtype=bool),
        ego_roles=jnp.zeros((states,), dtype=jnp.int32),
        task_phase=jnp.zeros((states,), dtype=jnp.int32),
    )


def test_anchor_buffer_only_samples_slots_it_has_filled() -> None:
    """An empty slot is zeros, and zeros are a state the simulator never saw."""

    import jax

    from src.delta_zsc.anchor_buffer import (
        init_anchor_buffer,
        push_anchor_buffer,
        sample_anchor_worlds,
    )

    template = _snapshot_stub(3, 0.0)
    buffer = init_anchor_buffer(template, 4)
    buffer = push_anchor_buffer(buffer, _snapshot_stub(3, 7.0), 11)
    assert int(buffer.filled) == 1

    drawn, versions = sample_anchor_worlds(buffer, jax.random.PRNGKey(0), 16)
    # Only the one filled slot exists, so every draw must come from it.
    np.testing.assert_allclose(np.asarray(drawn.environment_state), 7.0)
    np.testing.assert_array_equal(np.asarray(versions), 11)


def test_anchor_buffer_overwrites_the_oldest_slot() -> None:
    """Capacity bounds how stale a replayed world can be."""

    import jax

    from src.delta_zsc.anchor_buffer import (
        init_anchor_buffer,
        push_anchor_buffer,
        sample_anchor_worlds,
    )

    buffer = init_anchor_buffer(_snapshot_stub(2, 0.0), 2)
    for step, marker in enumerate((1.0, 2.0, 3.0), start=1):
        buffer = push_anchor_buffer(buffer, _snapshot_stub(2, marker), step)
    assert int(buffer.filled) == 2

    drawn, _ = sample_anchor_worlds(buffer, jax.random.PRNGKey(1), 32)
    present = set(np.unique(np.asarray(drawn.environment_state)).tolist())
    # The first push has been evicted; only the two most recent remain.
    assert present <= {2.0, 3.0}
    assert 1.0 not in present


def test_replayed_worlds_are_measured_not_recalled() -> None:
    """The buffer stores worlds; it must not carry any measured return.

    Replaying a stored *return* would fit the decision head to a continuation
    generated by a policy that no longer exists, which is the exact ordering
    guarantee the alternating update was built to preserve.
    """

    from src.delta_zsc.anchor_buffer import AnchorBuffer
    from src.delta_zsc.types import AnchorSnapshots

    stored = set(AnchorSnapshots._fields) | set(AnchorBuffer._fields)
    forbidden = {
        name
        for name in stored
        if "return" in name or "contrast" in name or "replica" in name
    }
    assert not forbidden


def test_outcome_encoding_order_matches_the_voi_enumeration() -> None:
    """Row i of the encoding table must be outcome i of the VOI integral.

    These two orderings are written in different modules and neither derives
    from the other.  If they drift, every probe is valued at the successor
    state of a different outcome -- silently, with no shape error and no NaN.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import compact_active_outcome_log_probabilities
    from src.delta_zsc.observation import INTERFACE_EVENT_CLASSES
    from src.delta_zsc.successor_feature import enumerated_outcome_encodings
    from src.delta_zsc.types import ProbeResponsePrediction

    classes = INTERFACE_EVENT_CLASSES
    key = jax.random.PRNGKey(0)
    prediction = ProbeResponsePrediction(
        visibility_logit=jax.random.normal(key, (1, 1)),
        interface_availability_logit=jax.random.normal(
            jax.random.fold_in(key, 1), (1, 1)
        ),
        interface_change_logit=jax.random.normal(jax.random.fold_in(key, 2), (1, 1)),
        interface_event_logits=jax.random.normal(
            jax.random.fold_in(key, 3), (1, 1, 1, classes)
        ),
    )
    logp = np.asarray(compact_active_outcome_log_probabilities(prediction))[0, 0, :, 0]
    table = np.asarray(enumerated_outcome_encodings())
    assert table.shape[0] == logp.shape[0] == 4 + 2 * classes

    # Recompute each outcome's log probability straight from the encoding and
    # check it against the integral's own ordering.
    visibility = float(prediction.visibility_logit[0, 0])
    availability = float(prediction.interface_availability_logit[0, 0])
    change = float(prediction.interface_change_logit[0, 0])
    events = np.asarray(jax.nn.log_softmax(prediction.interface_event_logits))[0, 0, 0]

    def log_sigmoid(x: float) -> float:
        return float(-np.logaddexp(0.0, -x))

    for index in range(table.shape[0]):
        seen, available, changed = table[index][:3]
        expected = log_sigmoid(visibility if seen else -visibility)
        expected += log_sigmoid(availability if available else -availability)
        if available:
            expected += log_sigmoid(change if changed else -change)
        if changed:
            expected += float(events[int(np.argmax(table[index][3:]))])
        np.testing.assert_allclose(logp[index], expected, atol=1.0e-5)


def test_outcome_table_is_a_concrete_constant_not_a_traced_array() -> None:
    """It must not belong to whichever jit trace built it first.

    Caching a ``jnp`` array here made the table a tracer owned by one trace,
    which then escaped into every later one.  The failure is call-order
    dependent -- touching the function outside jit first hides it -- so this
    test builds it *inside* one jit and then uses it inside another.
    """

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.successor_feature import enumerated_outcome_encodings

    @jax.jit
    def first(x):
        return jnp.sum(enumerated_outcome_encodings()) + x

    @jax.jit
    def second(x):
        return jnp.sum(enumerated_outcome_encodings()) * x

    a = float(first(jnp.asarray(1.0)))
    b = float(second(jnp.asarray(2.0)))
    assert np.isfinite(a) and np.isfinite(b)
    # And it is a plain host array, so nothing about it is trace-scoped.
    assert isinstance(enumerated_outcome_encodings(), np.ndarray)
