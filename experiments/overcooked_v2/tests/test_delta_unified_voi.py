from __future__ import annotations

import numpy as np


def _prediction(event_logits, *, availability: float = 0.0, change: float = 0.0):
    import jax.numpy as jnp

    from src.delta_zsc.types import ProbeResponsePrediction

    lead_probe = tuple(event_logits.shape[:-2])
    shared = jnp.zeros(lead_probe, dtype=jnp.float32)
    return ProbeResponsePrediction(
        visibility_logit=shared,
        interface_availability_logit=jnp.full(lead_probe, availability),
        interface_change_logit=jnp.full(lead_probe, change),
        interface_event_logits=event_logits,
    )


def test_compact_delayed_distribution_has_exactly_66_normalized_outcomes() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import compact_active_outcome_log_probabilities

    prediction = _prediction(jnp.zeros((2, 6, 3, 31)))
    logp = compact_active_outcome_log_probabilities(prediction)
    assert logp.shape == (2, 6, 66, 3)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(jnp.exp(logp), axis=-2)), 1.0, atol=1e-6
    )


def test_uninformative_delayed_response_has_zero_exact_voi_and_information_gain() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    prediction = _prediction(jnp.zeros((2, 6, 3, 31)))
    means = jnp.broadcast_to(
        jnp.arange(18, dtype=jnp.float32).reshape(1, 1, 3, 6),
        (2, 6, 3, 6),
    )
    result = myopic_value_of_information_details(
        jnp.full((2, 3), 1.0 / 3.0), prediction, means
    )
    np.testing.assert_allclose(np.asarray(result.value), 0.0, atol=3e-5)
    np.testing.assert_allclose(
        np.asarray(result.expected_information_gain), 0.0, atol=3e-5
    )


def test_information_without_successor_decision_relevance_has_zero_voi() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    event = jnp.full((1, 1, 2, 31), -8.0)
    event = event.at[0, 0, 0, 0].set(8.0)
    event = event.at[0, 0, 1, 1].set(8.0)
    prediction = _prediction(event, availability=10.0, change=10.0)
    means = jnp.asarray([[[[2.0, 0.0], [2.0, 0.0]]]])
    result = myopic_value_of_information_details(
        jnp.asarray([[0.5, 0.5]]), prediction, means
    )
    assert float(result.expected_information_gain[0, 0]) > 0.65
    assert abs(float(result.value[0, 0])) < 1e-5


def test_probe_conditioned_successor_utility_is_required_and_jittable() -> None:
    import jax
    import jax.numpy as jnp
    import pytest

    from src.delta_zsc.bayes_voi import myopic_value_of_information_details

    event = jnp.full((1, 2, 2, 31), -8.0)
    event = event.at[0, :, 0, 0].set(8.0)
    event = event.at[0, :, 1, 1].set(8.0)
    prediction = _prediction(event, availability=10.0, change=10.0)
    means = jnp.asarray(
        [[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]]]]
    )
    value = jax.jit(
        lambda belief: myopic_value_of_information_details(
            belief, prediction, means
        ).value
    )(jnp.asarray([[0.5, 0.5]]))
    assert value.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(value)))
    assert float(jnp.max(value) - jnp.min(value)) > 1.0e-3
    with pytest.raises(ValueError, match="must be probe-conditioned"):
        myopic_value_of_information_details(
            jnp.asarray([[0.5, 0.5]]),
            prediction,
            jnp.asarray([[[2.0, 0.0], [0.0, 2.0]]]),
        )
