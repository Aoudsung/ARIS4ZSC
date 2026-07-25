"""Registered ten-token response vocabulary from official local observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResponseVocabulary:
    tokens: tuple[str, ...]
    sha256: str
    token_ids: Mapping[str, int]

    @property
    def size(self) -> int:
        return len(self.tokens)


def _specification() -> Any:
    from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1

    return ResponseSummarySpecV1(
        response_classes=("visible", "unseen", "local_non_agent_change"),
        latency_bin_upper_bounds=(1,),
    )


def registered_response_vocabulary() -> ResponseVocabulary:
    specification = _specification()
    if specification.q != 10:
        raise ValueError("The registered local response vocabulary must contain ten tokens.")
    return ResponseVocabulary(
        tokens=tuple(specification.vocabulary),
        sha256=str(specification.sha256),
        token_ids=dict(specification.token_ids),
    )


def response_tokens(
    previous_observations: Any,
    next_observations: Any,
    done: Any,
    info: Mapping[str, Any],
) -> Any:
    """Encode one-step passive responses without reading environment state."""

    del info
    import jax.numpy as jnp

    from .env_dock import infer_default_observation_layout

    previous = jnp.asarray(previous_observations)
    following = jnp.asarray(next_observations)
    if previous.shape != following.shape or previous.ndim != 4:
        raise ValueError("Response encoding requires matching [batch, height, width, channel] observations.")
    layout = infer_default_observation_layout(previous)
    other_channel = layout["other_agent_position_channel"]
    first_non_agent = layout["first_non_agent_channel"]
    partner_visible = jnp.any(following[..., other_channel] > 0, axis=(1, 2))
    center_y = previous.shape[1] // 2
    center_x = previous.shape[2] // 2
    local_change = jnp.any(
        previous[
            :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2, first_non_agent:
        ]
        != following[
            :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2, first_non_agent:
        ],
        axis=(1, 2, 3),
    )
    specification = _specification()
    visible = int(specification.encode(response_class="visible", latency_steps=1))
    unseen = int(specification.encode(response_class="unseen", latency_steps=1))
    changed = int(
        specification.encode(response_class="local_non_agent_change", latency_steps=1)
    )
    terminal = int(specification.encode(terminal=True))
    regular = jnp.where(local_change, changed, jnp.where(partner_visible, visible, unseen))
    return jnp.where(jnp.asarray(done, dtype=jnp.bool_), terminal, regular).astype(jnp.int32)
