#!/usr/bin/env python3
"""Generate deterministic synthetic VOI acceptance diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np

from src.delta_zsc.bayes_voi import myopic_value_of_information_details
from src.delta_zsc.types import DirectResponsePrediction, ResponsePrediction


def prediction(visibility_logits, *, probes: int, components: int, factors: int = 2):
    lead = tuple(visibility_logits.shape[:-2])
    return ResponsePrediction(
        direct=DirectResponsePrediction(
            visibility_logit=jnp.asarray(visibility_logits, dtype=jnp.float32),
            relative_position_logits=jnp.zeros(lead + (probes, components, 25)),
            direction_logits=jnp.zeros(lead + (probes, components, 4)),
            inventory_logits=jnp.zeros(lead + (probes, components, factors, 2)),
            inventory_change_logit=jnp.zeros(lead + (probes, components)),
        ),
        interface_availability_logit=jnp.zeros(lead + (probes,)),
        interface_change_logit=jnp.zeros(lead + (probes, components)),
        interface_event_logits=jnp.zeros(lead + (probes, components, 31)),
        recipe_change_logit=jnp.zeros(lead + (probes, components)),
    )


def host(value):
    array = np.asarray(value, dtype=np.float64)
    return array.tolist() if array.ndim else float(array)


def main() -> None:
    belief = jnp.asarray([[0.5, 0.5]], dtype=jnp.float32)
    transition = jnp.eye(2, dtype=jnp.float32)
    decision_relevant = jnp.asarray([[[2.0, 0.0], [0.0, 2.0]]], dtype=jnp.float32)

    uninformative = myopic_value_of_information_details(
        belief,
        transition,
        prediction(jnp.zeros((1, 1, 2)), probes=1, components=2),
        decision_relevant,
    )
    revealing = myopic_value_of_information_details(
        belief,
        transition,
        prediction(jnp.asarray([[[-8.0, 8.0]]]), probes=1, components=2),
        decision_relevant,
    )
    identity_only = myopic_value_of_information_details(
        belief,
        transition,
        prediction(jnp.asarray([[[-8.0, 8.0]]]), probes=1, components=2),
        jnp.asarray([[[2.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32),
    )

    payload = {
        "version": 3,
        "artifact_type": "delta_exact_voi_v3_synthetic_acceptance",
        "method": "delta_joint_geometry_interface_decision_exact_voi_v3",
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
    }
    Path("validation/voi_synthetic_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
