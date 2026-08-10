#!/usr/bin/env python3
"""Generate deterministic DELTA v5 exact-VOI acceptance diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np

from src.delta_zsc.bayes_voi import (
    compact_active_outcome_log_probabilities,
    myopic_value_of_information_details,
)
from src.delta_zsc.config import METHOD_VERSION
from src.delta_zsc.types import ProbeResponsePrediction


def prediction(*, revealing: bool, probes: int = 1) -> ProbeResponsePrediction:
    components = 2
    event = jnp.zeros((1, probes, components, 31), dtype=jnp.float32)
    if revealing:
        event = event.at[..., 0, 0].set(8.0)
        event = event.at[..., 0, 1].set(-8.0)
        event = event.at[..., 1, 0].set(-8.0)
        event = event.at[..., 1, 1].set(8.0)
    # Make a valid changed event overwhelmingly likely so the diagnostic tests
    # the semantic Bayes path rather than shared occurrence uncertainty.
    shared = (1, probes)
    return ProbeResponsePrediction(
        visibility_logit=jnp.zeros(shared, dtype=jnp.float32),
        interface_availability_logit=jnp.full(shared, 8.0, dtype=jnp.float32),
        interface_change_logit=jnp.full(shared, 8.0, dtype=jnp.float32),
        interface_event_logits=event,
    )


def host(value):
    array = np.asarray(value, dtype=np.float64)
    return array.tolist() if array.ndim else float(array)


def main() -> None:
    belief = jnp.asarray([[0.5, 0.5]], dtype=jnp.float32)
    decision_relevant = jnp.asarray(
        [[[[2.0, 0.0], [0.0, 2.0]]]], dtype=jnp.float32
    )
    decision_irrelevant = jnp.asarray(
        [[[[2.0, 0.0], [2.0, 0.0]]]], dtype=jnp.float32
    )

    uninformative_prediction = prediction(revealing=False)
    revealing_prediction = prediction(revealing=True)
    uninformative = myopic_value_of_information_details(
        belief, uninformative_prediction, decision_relevant
    )
    revealing = myopic_value_of_information_details(
        belief, revealing_prediction, decision_relevant
    )
    identity_only = myopic_value_of_information_details(
        belief, revealing_prediction, decision_irrelevant
    )
    outcome_logp = compact_active_outcome_log_probabilities(revealing_prediction)
    outcome_mass = np.sum(np.exp(np.asarray(outcome_logp)), axis=-2)

    outcome_count = int(outcome_logp.shape[-2])
    mass_error = float(np.max(np.abs(outcome_mass - 1.0)))
    uninformative_voi = float(np.max(np.abs(np.asarray(uninformative.value))))
    revealing_voi = float(np.min(np.asarray(revealing.value)))
    identity_voi = float(np.max(np.abs(np.asarray(identity_only.value))))
    identity_information = float(
        np.min(np.asarray(identity_only.expected_information_gain))
    )
    assert outcome_count == 66
    assert mass_error <= 1.0e-6
    assert uninformative_voi <= 1.0e-5
    assert revealing_voi > 0.1
    assert identity_voi <= 1.0e-5
    assert identity_information > 0.1

    payload = {
        "version": 5,
        "artifact_type": "delta_v5_delayed_exact_voi_synthetic_acceptance",
        "method": METHOD_VERSION,
        "outcome_count": outcome_count,
        "maximum_component_outcome_mass_error": mass_error,
        "uninformative_response": {
            "voi": host(uninformative.value),
            "information_gain": host(uninformative.expected_information_gain),
        },
        "decision_revealing_response": {
            "voi": host(revealing.value),
            "information_gain": host(revealing.expected_information_gain),
        },
        "identity_information_without_decision_relevance": {
            "voi": host(identity_only.value),
            "information_gain": host(identity_only.expected_information_gain),
        },
        "validation_status": "generated_locally",
    }
    Path("validation/voi_synthetic_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
