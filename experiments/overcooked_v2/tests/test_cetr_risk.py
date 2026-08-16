from __future__ import annotations

import numpy as np


def _greedy_reference(
    parent_returns: np.ndarray,
    nominal_weights: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    restricted = nominal_weights * observed_mask.astype(np.float64)
    restricted = restricted / restricted.sum()
    result = np.zeros_like(restricted)
    remaining = 1.0
    from src.cetr_zsc.config import TAIL_DENSITY_RATIO_CAP

    for index in np.argsort(parent_returns, kind="stable"):
        allocation = min(
            float(TAIL_DENSITY_RATIO_CAP * restricted[index]),
            max(remaining, 0.0),
        )
        result[index] = allocation
        remaining -= allocation
    return result


def test_lower_half_cvar_small_case() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.risk import lower_half_cvar_weights, tail_objective

    returns = jnp.asarray([3.0, 1.0, 2.0, 4.0])
    nominal = jnp.full((4,), 0.25)
    weights = lower_half_cvar_weights(returns, nominal)
    np.testing.assert_allclose(np.asarray(weights), [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(np.asarray(tail_objective(returns, nominal)), 1.5)


def test_lower_half_cvar_matches_numpy_greedy_reference() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.risk import lower_half_cvar_weights

    generator = np.random.default_rng(17)
    for _ in range(16):
        count = int(generator.integers(2, 10))
        returns = generator.normal(size=count).astype(np.float32)
        nominal = generator.uniform(0.05, 2.0, size=count).astype(np.float32)
        observed = generator.random(count) > 0.25
        if not np.any(observed):
            observed[0] = True
        expected = _greedy_reference(returns, nominal, observed)
        actual = lower_half_cvar_weights(
            jnp.asarray(returns),
            jnp.asarray(nominal),
            jnp.asarray(observed),
        )
        np.testing.assert_allclose(np.asarray(actual), expected, atol=2.0e-6)


def test_tail_weights_respect_cap_and_mass() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.config import TAIL_DENSITY_RATIO_CAP
    from src.cetr_zsc.risk import lower_half_cvar_weights, renormalize_weights

    returns = jnp.asarray([4.0, -2.0, 1.0, 3.0, 0.0])
    nominal = jnp.asarray([1.0, 2.0, 1.0, 3.0, 2.0])
    observed = jnp.asarray([True, False, True, True, False])
    normalized = renormalize_weights(nominal, observed)
    weights = lower_half_cvar_weights(returns, nominal, observed)
    assert bool(jnp.all(weights <= TAIL_DENSITY_RATIO_CAP * normalized + 1.0e-6))
    np.testing.assert_allclose(np.asarray(jnp.sum(weights)), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(weights)[~np.asarray(observed)], 0.0)
    np.testing.assert_allclose(np.asarray(jnp.sum(normalized)), 1.0, atol=1.0e-6)


def test_equal_returns_use_stable_index_order() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.risk import lower_half_cvar_weights

    weights = lower_half_cvar_weights(
        jnp.ones((6,)),
        jnp.full((6,), 1.0 / 6.0),
    )
    np.testing.assert_allclose(np.asarray(weights), [1.0 / 3.0] * 3 + [0.0] * 3)


def test_sp_dual_update_has_three_projected_directions() -> None:
    from src.cetr_zsc.risk import update_sp_dual

    np.testing.assert_allclose(float(update_sp_dual(0.5, 3.0, 2.0, 0.1)), 0.6)
    np.testing.assert_allclose(float(update_sp_dual(0.5, 2.0, 3.0, 0.1)), 0.4)
    np.testing.assert_allclose(float(update_sp_dual(0.1, 0.0, 3.0, 0.1)), 0.0)


def test_risk_functions_run_under_jit() -> None:
    import jax
    import jax.numpy as jnp

    from src.cetr_zsc.risk import (
        lower_half_cvar_weights,
        renormalize_weights,
        tail_objective,
        update_sp_dual,
    )

    returns = jnp.asarray([3.0, 1.0, 2.0, 4.0])
    nominal = jnp.full((4,), 0.25)
    observed = jnp.asarray([True, True, True, True])
    normalized = jax.jit(renormalize_weights)(nominal, observed)
    weights = jax.jit(lower_half_cvar_weights)(returns, nominal, observed)
    objective = jax.jit(tail_objective)(returns, nominal, observed)
    updated = jax.jit(update_sp_dual)(0.0, 1.0, 0.5, 0.1)
    assert bool(jnp.all(jnp.isfinite(normalized)))
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert bool(jnp.isfinite(objective))
    assert bool(jnp.isfinite(updated))
