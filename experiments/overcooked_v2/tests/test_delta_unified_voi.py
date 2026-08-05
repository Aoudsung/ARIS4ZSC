from __future__ import annotations

import numpy as np


def _prediction(
    visibility_logits,
    *,
    probes: int,
    components: int,
    factors: int = 5,
    inventory_change_logits=None,
):
    import jax.numpy as jnp

    from src.delta_zsc.types import ResponsePrediction

    lead = tuple(visibility_logits.shape[:-2])
    return ResponsePrediction(
        visibility_logit=visibility_logits,
        relative_position_logits=jnp.zeros(
            lead + (probes, components, 25), dtype=jnp.float32
        ),
        direction_logits=jnp.zeros(
            lead + (probes, components, 4), dtype=jnp.float32
        ),
        inventory_logits=jnp.zeros(
            lead + (probes, components, factors, 2), dtype=jnp.float32
        ),
        inventory_change_logit=(
            jnp.zeros(lead + (probes, components), dtype=jnp.float32)
            if inventory_change_logits is None
            else jnp.asarray(inventory_change_logits, dtype=jnp.float32)
        ),
    )


def test_halton_design_is_deterministic_bounded_and_multidimensional() -> None:
    from src.delta_zsc.bayes_voi import halton_points

    left = np.asarray(halton_points(16, 9))
    right = np.asarray(halton_points(16, 9))
    np.testing.assert_array_equal(left, right)
    assert left.shape == (16, 9)
    assert np.all((left > 0.0) & (left < 1.0))
    assert np.unique(left, axis=0).shape[0] == 16
    # Distinct factor dimensions must not be shifted copies of one shared grid.
    assert not np.allclose(np.sort(left[:, 0]), np.sort(left[:, 1]))


def test_uninformative_response_has_zero_voi_and_information_gain() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    batch, probes, components, actions = 2, 6, 3, 6
    prediction = _prediction(
        jnp.zeros((batch, probes, components), dtype=jnp.float32),
        probes=probes,
        components=components,
    )
    belief = jnp.full((batch, components), 1.0 / components)
    means = jnp.arange(batch * components * actions, dtype=jnp.float32).reshape(
        batch, components, actions
    )
    result = myopic_value_of_information_details(
        belief,
        jnp.eye(components),
        prediction,
        means,
        previous_visibility=jnp.ones((batch,)),
        sample_count=32,
    )
    np.testing.assert_allclose(np.asarray(result.value), 0.0, atol=3e-5)
    np.testing.assert_allclose(
        np.asarray(result.expected_information_gain), 0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        np.asarray(result.quadrature_error_estimate), 0.0, atol=3e-5
    )


def test_revealing_response_has_positive_decision_voi() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    visibility = jnp.asarray([[[-8.0, 8.0], [0.0, 0.0]]], dtype=jnp.float32)
    prediction = _prediction(visibility, probes=2, components=2, factors=2)
    means = jnp.asarray([[[2.0, 0.0], [0.0, 2.0]]], dtype=jnp.float32)
    result = myopic_value_of_information_details(
        jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
        jnp.eye(2),
        prediction,
        means,
        previous_visibility=jnp.asarray([0.0]),
        sample_count=32,
    )
    assert float(result.value[0, 0]) > 0.95
    assert float(result.expected_information_gain[0, 0]) > 0.65
    assert abs(float(result.value[0, 1])) < 1e-5


def test_information_without_decision_relevance_has_zero_voi() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(
        jnp.asarray([[[-8.0, 8.0]]], dtype=jnp.float32),
        probes=1,
        components=2,
        factors=2,
    )
    # Components are identifiable but imply the same action ordering/value.
    means = jnp.asarray([[[2.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32)
    result = myopic_value_of_information_details(
        jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
        jnp.eye(2),
        prediction,
        means,
        previous_visibility=jnp.asarray([0.0]),
        sample_count=32,
    )
    assert float(result.expected_information_gain[0, 0]) > 0.65
    assert abs(float(result.value[0, 0])) < 1e-5


def test_probe_conditioned_future_utility_is_supported_and_jittable() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(
        jnp.asarray([[[-8.0, 8.0], [8.0, -8.0]]], dtype=jnp.float32),
        probes=2,
        components=2,
        factors=2,
    )
    means = jnp.asarray(
        [[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]]]],
        dtype=jnp.float32,
    )

    def evaluate(belief):
        return myopic_value_of_information_details(
            belief,
            jnp.eye(2),
            prediction,
            means,
            previous_visibility=jnp.asarray([0.0]),
            sample_count=16,
        ).value

    value = jax.jit(evaluate)(jnp.asarray([[0.5, 0.5]], dtype=jnp.float32))
    assert value.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(value)))



def test_binary_visibility_and_inventory_change_are_summed_exactly() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    belief = jnp.asarray([[0.5, 0.5]], dtype=jnp.float32)
    means = jnp.asarray([[[2.0, 0.0], [0.0, 2.0]]], dtype=jnp.float32)

    visibility_prediction = _prediction(
        jnp.asarray([[[-8.0, 8.0]]], dtype=jnp.float32),
        probes=1,
        components=2,
        factors=2,
    )
    visibility_small = myopic_value_of_information_details(
        belief,
        jnp.eye(2),
        visibility_prediction,
        means,
        previous_visibility=jnp.asarray([0.0]),
        sample_count=2,
    )
    visibility_large = myopic_value_of_information_details(
        belief,
        jnp.eye(2),
        visibility_prediction,
        means,
        previous_visibility=jnp.asarray([0.0]),
        sample_count=32,
    )
    np.testing.assert_allclose(
        np.asarray(visibility_small.raw_value),
        np.asarray(visibility_large.raw_value),
        atol=1e-6,
    )

    event_prediction = _prediction(
        jnp.asarray([[[8.0, 8.0]]], dtype=jnp.float32),
        probes=1,
        components=2,
        factors=2,
        inventory_change_logits=jnp.asarray([[[-8.0, 8.0]]], dtype=jnp.float32),
    )
    event_small = myopic_value_of_information_details(
        belief,
        jnp.eye(2),
        event_prediction,
        means,
        previous_visibility=jnp.asarray([1.0]),
        sample_count=2,
    )
    event_large = myopic_value_of_information_details(
        belief,
        jnp.eye(2),
        event_prediction,
        means,
        previous_visibility=jnp.asarray([1.0]),
        sample_count=32,
    )
    np.testing.assert_allclose(
        np.asarray(event_small.raw_value),
        np.asarray(event_large.raw_value),
        atol=1e-6,
    )

def test_nested_halton_prefix_reports_numerical_error_without_changing_value() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(
        jnp.asarray([[[-2.0, 2.0]]], dtype=jnp.float32),
        probes=1,
        components=2,
        factors=2,
    )
    result = myopic_value_of_information_details(
        jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
        jnp.eye(2),
        prediction,
        jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32),
        previous_visibility=jnp.asarray([0.0]),
        sample_count=16,
    )
    assert result.quadrature_error_estimate.shape == result.value.shape
    assert float(result.quadrature_error_estimate[0, 0]) >= 0.0
    assert np.isfinite(np.asarray(result.quadrature_error_estimate)).all()
