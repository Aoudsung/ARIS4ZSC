"""Posterior-predictive calibration invariants for DEPI."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.overcooked_v2.calibration_app import _score_calibration_block  # noqa: E402
from src.path_c.calibration import (  # noqa: E402
    calibration_pass_decision,
    event_brier_score,
    highest_probability_set_coverage,
    hungarian_component_alignment,
    posterior_predictive_probabilities,
)
from src.path_c.types import ContextOutput  # noqa: E402
from src.path_c.response_targets import ResponsePrediction  # noqa: E402


def test_highest_probability_set_uses_the_true_label_rank() -> None:
    probabilities = jnp.asarray(
        [[0.60, 0.25, 0.10, 0.05], [0.60, 0.25, 0.10, 0.05]],
        dtype=jnp.float32,
    )
    # At 80%, classes ranked first and second are included.  This detects the
    # former ``sum(order != label)`` bug, which assigned every label rank K-1.
    included = highest_probability_set_coverage(
        probabilities, jnp.asarray([0, 1]), credibility=0.80
    )
    excluded = highest_probability_set_coverage(
        probabilities, jnp.asarray([0, 3]), credibility=0.80
    )
    assert float(included) == pytest.approx(1.0)
    assert float(excluded) == pytest.approx(0.5)


def test_coverage_consumes_posterior_predictive_probabilities() -> None:
    posterior = jnp.asarray([[0.75, 0.25]], dtype=jnp.float32)
    component_logits = jnp.asarray(
        [[[6.0, -6.0], [-6.0, 6.0]]], dtype=jnp.float32
    )
    predictive = posterior_predictive_probabilities(posterior, component_logits)
    np.testing.assert_allclose(np.asarray(predictive), [[0.75, 0.25]], atol=2e-5)
    assert float(
        highest_probability_set_coverage(
            predictive, jnp.asarray([0]), credibility=0.70
        )
    ) == pytest.approx(1.0)


def test_calibration_gate_requires_both_coverages_and_both_baselines() -> None:
    kwargs = dict(
        model_log_score=0.80,
        uniform_baseline_log_score=0.83,
        no_history_baseline_log_score=0.84,
        position_coverage=0.90,
        direction_coverage=0.90,
        event_brier=0.08,
        prior_baseline_brier=0.10,
    )
    assert calibration_pass_decision(**kwargs)
    assert not calibration_pass_decision(**{**kwargs, "direction_coverage": 0.80})
    assert not calibration_pass_decision(
        **{**kwargs, "no_history_baseline_log_score": 0.81}
    )


def test_brier_and_component_alignment_are_well_defined() -> None:
    assert float(event_brier_score(jnp.asarray([0.8, 0.1]), jnp.asarray([1, 0]))) == (
        pytest.approx(0.025)
    )
    components = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    order, aligned = hungarian_component_alignment(components, components[::-1])
    np.testing.assert_array_equal(order, [1, 0])
    np.testing.assert_array_equal(aligned, components[::-1])


def test_no_history_counterfactual_resets_every_partner_history_carrier() -> None:
    observed_contexts = []

    class FakeModel:
        def response_sequence(self):
            raise AssertionError

        def context_sequence(self):
            raise AssertionError

        def response_from_context_and_action(self):
            raise AssertionError

        def _prediction(self, log_pi):
            prefix = log_pi.shape[:-1]
            return ResponsePrediction(
                posterior_log_probabilities=log_pi,
                visibility_logit=jnp.zeros(prefix + (2,)),
                relative_position_logits=jnp.zeros(prefix + (2, 26)),
                direction_logits=jnp.zeros(prefix + (2, 4)),
                inventory_logits=jnp.zeros(prefix + (2, 5, 4)),
                interaction_change_logit=jnp.zeros(prefix + (2,)),
            )

        def apply(self, variables, *args, method):
            del variables
            if method.__name__ == "response_sequence":
                return None, self._prediction(
                    jnp.log(jnp.asarray([[[0.9, 0.1]]], dtype=jnp.float32))
                )
            if method.__name__ == "context_sequence":
                return None, ContextOutput(
                    task_features=jnp.ones((2, 1, 3)),
                    instant_partner=jnp.ones((2, 1, 2)),
                    capability=jnp.ones((2, 1, 2)) * 7.0,
                    protocol_probabilities=jnp.asarray(
                        [[[0.9, 0.1]], [[0.8, 0.2]]], dtype=jnp.float32
                    ),
                    protocol_embedding=jnp.ones((2, 1, 2)) * 5.0,
                )
            if method.__name__ == "response_from_context_and_action":
                context = args[0]
                observed_contexts.append(context)
                return self._prediction(jnp.log(context.protocol_probabilities))
            raise AssertionError(method)

    model = FakeModel()
    deployment = SimpleNamespace(
        model=model,
        params={
            "protocol_component_embeddings": {
                "embedding": jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
            }
        },
    )
    observations = jnp.zeros((2, 1, 5, 5, 39), dtype=jnp.float32)
    batch = SimpleNamespace(
        initial_policy_state=None,
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros((2, 1), dtype=jnp.int32),
        episode_starts=jnp.asarray([[True], [False]]),
        actions=jnp.zeros((1, 1), dtype=jnp.int32),
    )
    payload = _score_calibration_block(deployment=deployment, batch=batch)
    assert payload["no_history_log_probabilities"].shape == (1, 1)
    assert len(observed_contexts) == 1
    context = observed_contexts[0]
    np.testing.assert_array_equal(np.asarray(context.capability), 0.0)
    np.testing.assert_allclose(np.asarray(context.protocol_probabilities), 0.5)
    np.testing.assert_allclose(np.asarray(context.protocol_embedding), [[[0.5, 0.5]]])
