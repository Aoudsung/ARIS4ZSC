from __future__ import annotations

import numpy as np


def _prediction(visibility_logits, *, probes: int, components: int):
    import jax.numpy as jnp

    from src.delta_zsc.types import DirectResponsePrediction, ResponsePrediction

    lead = tuple(visibility_logits.shape[:-2])
    direct = DirectResponsePrediction(
        visibility_logit=visibility_logits,
        relative_position_logits=jnp.zeros(lead + (probes, components, 25)),
        direction_logits=jnp.zeros(lead + (probes, components, 4)),
        inventory_logits=jnp.zeros(lead + (probes, components, 5, 2)),
        inventory_change_logit=jnp.zeros(lead + (probes, components)),
    )
    return ResponsePrediction(
        direct=direct,
        interface_availability_logit=jnp.zeros(lead + (probes,)),
        interface_change_logit=jnp.zeros(lead + (probes, components)),
        interface_event_logits=jnp.zeros(lead + (probes, components, 31)),
        recipe_change_logit=jnp.zeros(lead + (probes, components)),
    )


def test_compact_active_distribution_has_exactly_66_normalized_outcomes() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import compact_active_outcome_log_probabilities

    prediction = _prediction(jnp.zeros((2, 6, 3)), probes=6, components=3)
    logp = compact_active_outcome_log_probabilities(prediction)
    assert logp.shape == (2, 6, 66, 3)
    np.testing.assert_allclose(np.asarray(jnp.sum(jnp.exp(logp), axis=-2)), 1.0, atol=1e-6)


def test_uninformative_response_has_zero_exact_voi_and_information_gain() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(jnp.zeros((2, 6, 3)), probes=6, components=3)
    result = myopic_value_of_information_details(
        jnp.full((2, 3), 1.0 / 3.0),
        jnp.eye(3),
        prediction,
        jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 6),
    )
    np.testing.assert_allclose(np.asarray(result.value), 0.0, atol=3e-5)
    np.testing.assert_allclose(np.asarray(result.expected_information_gain), 0.0, atol=3e-5)


def test_information_without_decision_relevance_has_zero_voi() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(jnp.asarray([[[-8.0, 8.0]]]), probes=1, components=2)
    means = jnp.asarray([[[2.0, 0.0], [2.0, 0.0]]])
    result = myopic_value_of_information_details(
        jnp.asarray([[0.5, 0.5]]), jnp.eye(2), prediction, means
    )
    assert float(result.expected_information_gain[0, 0]) > 0.65
    assert abs(float(result.value[0, 0])) < 1e-5


def test_probe_conditioned_future_utility_is_supported_and_jittable() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(
        jnp.asarray([[[-8.0, 8.0], [8.0, -8.0]]]), probes=2, components=2
    )
    means = jnp.asarray(
        [[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]]]]
    )
    value = jax.jit(
        lambda belief: myopic_value_of_information_details(
            belief, jnp.eye(2), prediction, means
        ).value
    )(jnp.asarray([[0.5, 0.5]]))
    assert value.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(value)))
