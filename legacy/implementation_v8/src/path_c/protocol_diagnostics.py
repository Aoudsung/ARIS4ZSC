"""Registered occlusion and re-identification diagnostics for protocol filtering."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_STAY_SENSITIVITY = (0.90, 0.97, 0.99)


def analytic_posterior_memory_curves(
    *, component_count: int = 4, maximum_occlusion_steps: int = 64
) -> Mapping[str, Any]:
    """Contrast the registered evidence gate with an always-transition filter."""

    count = int(component_count)
    length = int(maximum_occlusion_steps)
    if count < 2 or length < 1:
        raise ValueError("Protocol memory curves need K>=2 and a positive horizon.")
    steps = np.arange(length + 1, dtype=np.int64)
    curves = {}
    for stay in PROTOCOL_STAY_SENSITIVITY:
        nontrivial_eigenvalue = stay - (1.0 - stay) / (count - 1)
        curves[f"{stay:.2f}"] = {
            "ungated_deviation_retention": (
                nontrivial_eigenvalue**steps
            ).tolist(),
            "evidence_gated_deviation_retention": np.ones_like(
                steps, dtype=np.float64
            ).tolist(),
            "ungated_half_life_steps": float(
                np.log(0.5) / np.log(nontrivial_eigenvalue)
            ),
        }
    return {
        "component_count": count,
        "occlusion_steps": steps.tolist(),
        "curves": curves,
    }


def empirical_protocol_memory_metrics(
    sequences: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Summarize posterior memory, false switches, and re-identification time.

    Each sequence supplies ``posterior`` with shape ``[time,K]`` and
    ``evidence_valid`` with shape ``[time]``.  Re-identification is the first
    valid-evidence step after an occlusion whose argmax returns to the
    pre-occlusion component.  The metric is diagnostic and does not treat a
    latent component index as globally named.
    """

    occlusion_lengths: list[int] = []
    endpoint_retention: list[float] = []
    false_switches = 0
    invalid_transitions = 0
    reidentification_steps: list[int] = []
    unresolved = 0
    for raw in sequences:
        posterior = np.asarray(raw["posterior"], dtype=np.float64)
        valid = np.asarray(raw["evidence_valid"], dtype=bool)
        if (
            posterior.ndim != 2
            or posterior.shape[0] != valid.shape[0]
            or posterior.shape[1] < 2
            or not np.all(np.isfinite(posterior))
            or not np.allclose(np.sum(posterior, axis=-1), 1.0, atol=1.0e-5)
        ):
            raise ValueError("Protocol sensitivity sequence shape differs.")
        labels = np.argmax(posterior, axis=-1)
        time = 0
        while time < valid.size:
            if valid[time]:
                time += 1
                continue
            start = time
            while time < valid.size and not valid[time]:
                time += 1
            stop = time
            if start == 0:
                continue
            anchor_component = int(labels[start - 1])
            anchor_probability = float(posterior[start - 1, anchor_component])
            occlusion_lengths.append(stop - start)
            endpoint_retention.append(
                float(posterior[stop - 1, anchor_component])
                / max(anchor_probability, 1.0e-12)
            )
            segment_labels = labels[start:stop]
            false_switches += int(
                np.sum(segment_labels != np.concatenate(
                    ([anchor_component], segment_labels[:-1])
                ))
            )
            invalid_transitions += int(stop - start)
            recovered = None
            for index in range(stop, valid.size):
                if valid[index] and labels[index] == anchor_component:
                    recovered = index - stop + 1
                    break
            if recovered is None:
                unresolved += 1
            else:
                reidentification_steps.append(int(recovered))
    if not occlusion_lengths:
        raise ValueError("Protocol sensitivity data contains no measurable occlusion.")
    return {
        "occlusion_count": len(occlusion_lengths),
        "occlusion_length_mean": float(np.mean(occlusion_lengths)),
        "occlusion_length_q90": float(np.quantile(occlusion_lengths, 0.90)),
        "posterior_memory_endpoint_retention_mean": float(
            np.mean(endpoint_retention)
        ),
        "false_switch_rate_during_occlusion": float(
            false_switches / max(invalid_transitions, 1)
        ),
        "reidentification_time_mean": (
            None
            if not reidentification_steps
            else float(np.mean(reidentification_steps))
        ),
        "reidentification_unresolved_fraction": float(
            unresolved / len(occlusion_lengths)
        ),
    }


__all__ = [
    "PROTOCOL_STAY_SENSITIVITY",
    "analytic_posterior_memory_curves",
    "empirical_protocol_memory_metrics",
]
