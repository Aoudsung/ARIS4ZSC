from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.decision_geometry import (  # noqa: E402
    brdiv_logdet,
    centered_action_values,
    decision_distance,
)
from src.path_c.raw_q import decision_equivalence_metric_loss  # noqa: E402
from src.path_c.regret_potential import (  # noqa: E402
    common_optimal_action_regret_zero,
    decision_regret_from_action_values,
    potential_shaping,
)
from src.path_c.runner import decision_regret_weight  # noqa: E402
from src.path_c.response_targets import (  # noqa: E402
    extract_partner_response_targets,
    official_partner_observation_planes,
)


def test_decision_signatures_are_translation_invariant() -> None:
    values = jnp.asarray([[1.0, 2.0, 4.0], [11.0, 12.0, 14.0]])
    centered = centered_action_values(values)
    np.testing.assert_allclose(centered[0], centered[1], atol=1.0e-6)
    np.testing.assert_allclose(jnp.mean(centered, axis=-1), 0.0, atol=1e-6)
    assert float(decision_distance(centered[0], centered[1])) < 2e-6


def test_partner_movement_is_not_mislabeled_as_inventory_interaction() -> None:
    planes = official_partner_observation_planes(39)
    previous = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    current = jnp.zeros_like(previous)
    previous = previous.at[1, 1, planes.visibility_channel].set(1.0)
    current = current.at[1, 2, planes.visibility_channel].set(1.0)
    previous = previous.at[1, 1, planes.inventory_channels[0]].set(1.0)
    current = current.at[1, 2, planes.inventory_channels[0]].set(1.0)
    targets = extract_partner_response_targets(previous, current, planes=planes)
    assert float(targets.interaction_change) == 0.0
    changed = current.at[1, 2, planes.inventory_channels[0]].set(2.0)
    targets = extract_partner_response_targets(previous, changed, planes=planes)
    assert float(targets.interaction_change) == 1.0


def test_v6_continuous_decision_equivalence_matches_scaled_action_distance() -> None:
    advantage_a = jnp.asarray([[1.0, 0.0, -1.0]])
    advantage_b = jnp.asarray([[-1.0, 0.0, 1.0]])
    target = float(jnp.linalg.norm(advantage_a - advantage_b) / 2.0)
    latent_a = jnp.asarray([[0.0, 0.0]])
    latent_b = jnp.asarray([[target, 0.0]])
    loss, latent_distance, decision_target = decision_equivalence_metric_loss(
        mean_a=latent_a,
        mean_b=latent_b,
        advantage_a=advantage_a,
        advantage_b=advantage_b,
        advantage_scale=jnp.asarray(2.0),
        weights=jnp.asarray([1.0]),
    )
    assert float(loss) == pytest.approx(0.0, abs=1.0e-7)
    np.testing.assert_allclose(latent_distance, decision_target, atol=1.0e-7)
    assert float(jax.grad(
        lambda values: decision_equivalence_metric_loss(
            mean_a=latent_a,
            mean_b=latent_b,
            advantage_a=values,
            advantage_b=advantage_b,
            advantage_scale=jnp.asarray(2.0),
            weights=jnp.asarray([1.0]),
        )[0]
    )(advantage_a).sum()) == 0.0


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


def test_decision_regret_schedule_and_terminal_detached_potential() -> None:
    early = decision_regret_weight(
        effective_steps=0,
        total_steps=100,
        maximum=0.1,
        midpoint=0.2,
        temperature=0.05,
    )
    midpoint = decision_regret_weight(
        effective_steps=20,
        total_steps=100,
        maximum=0.1,
        midpoint=0.2,
        temperature=0.05,
    )
    late = decision_regret_weight(
        effective_steps=100,
        total_steps=100,
        maximum=0.1,
        midpoint=0.2,
        temperature=0.05,
    )
    assert float(early) < 0.002
    assert float(midpoint) == pytest.approx(0.05, abs=1.0e-7)
    assert float(late) > 0.099

    current = jnp.asarray([2.0, 3.0])
    following = jnp.asarray([100.0, 5.0])
    terminal = jnp.asarray([True, False])
    shaped = potential_shaping(
        current, following, terminal, gamma=0.99, weight=0.1
    )
    np.testing.assert_allclose(
        shaped, [0.2, 0.1 * (3.0 - 0.99 * 5.0)], atol=1.0e-7
    )
    gradients = jax.grad(
        lambda left, right: jnp.sum(
            potential_shaping(left, right, terminal, gamma=0.99, weight=0.1)
        ),
        argnums=(0, 1),
    )(current, following)
    np.testing.assert_array_equal(gradients[0], 0.0)
    np.testing.assert_array_equal(gradients[1], 0.0)
