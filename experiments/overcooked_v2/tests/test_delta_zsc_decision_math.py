from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.decision_geometry import (  # noqa: E402
    brdiv_logdet,
    centered_action_values,
    decision_distance,
    quotient_geometry_loss,
)
from src.path_c.regret_potential import (  # noqa: E402
    common_optimal_action_regret_zero,
    decision_regret_from_action_values,
    potential_shaping,
)


def test_decision_signatures_are_translation_invariant() -> None:
    values = jnp.asarray([[1.0, 2.0, 4.0], [11.0, 12.0, 14.0]])
    centered = centered_action_values(values)
    np.testing.assert_allclose(
        np.asarray(centered[0]), np.asarray(centered[1]), atol=1.0e-6
    )
    np.testing.assert_allclose(np.asarray(jnp.mean(centered, axis=-1)), 0.0, atol=1e-6)
    assert float(decision_distance(centered[0], centered[1])) < 2e-6


def test_quotient_loss_collapses_equivalent_and_separates_distinct_contexts() -> None:
    equivalent = quotient_geometry_loss(
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([0.0]),
        equivalence_epsilon=0.05,
        separation_epsilon=0.25,
        margin=1.0,
    )
    separated = quotient_geometry_loss(
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([1.0]),
        equivalence_epsilon=0.05,
        separation_epsilon=0.25,
        margin=1.0,
    )
    assert float(equivalent) < 1e-8
    assert float(separated) < 1e-8


def test_brdiv_rewards_decision_distinct_signatures() -> None:
    duplicate = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    distinct = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert float(brdiv_logdet(distinct, 1.0)) > float(brdiv_logdet(duplicate, 1.0))


def test_decision_regret_is_zero_only_when_an_optimal_action_is_shared() -> None:
    shared = jnp.asarray([[2.0, 1.0], [3.0, 0.0]])
    opposed = jnp.asarray([[2.0, 0.0], [0.0, 2.0]])
    common, shared_regret = common_optimal_action_regret_zero(shared)
    assert bool(common)
    assert float(shared_regret) == pytest.approx(0.0)
    assert float(decision_regret_from_action_values(opposed)) == pytest.approx(1.0)


def test_discounted_potential_shaping_telescopes() -> None:
    regret = jnp.asarray([3.0, 2.0, 1.0])
    following = jnp.asarray([2.0, 1.0, 0.0])
    dones = jnp.asarray([False, False, True])
    gamma = 0.9
    shaping = potential_shaping(
        regret, following, dones, gamma=gamma, weight=1.0
    )
    discounted = sum((gamma**index) * float(value) for index, value in enumerate(shaping))
    assert discounted == pytest.approx(3.0, abs=1e-6)


def test_potential_shaping_accepts_runtime_weight_in_jitted_kernel() -> None:
    dynamic = jax.jit(
        lambda value: potential_shaping(
            jnp.asarray([2.0], dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([False]),
            gamma=0.99,
            weight=value,
        )
    )
    np.testing.assert_allclose(
        np.asarray(dynamic(jnp.asarray(0.1, dtype=jnp.float32))),
        np.asarray([0.101], dtype=np.float32),
        atol=1.0e-6,
    )
