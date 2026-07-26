"""Matched response-use/mask branch contracts for behavior-consistent VQBC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


RESPONSE_CONTRAST_BRANCHES = ("A1", "A2-mask", "A2-use")
RESPONSE_CONTRAST_SCHEMA_VERSION = "path_c_vqbc_response_contrast_rows_v4"
RESPONSE_CONTRAST_SUMMARY_VERSION = "path_c_vqbc_response_contrast_summary_v4"
INFORMATION_TRIGGER_FLOAT32_ULPS = 256.0


def policy_effect_trigger_tolerance(j_use: Any, j_mask: Any) -> Any:
    """Return a scale-aware float32 floor for a policy-level raw effect."""

    import jax.numpy as jnp

    use = jnp.asarray(j_use, dtype=jnp.float32)
    mask = jnp.asarray(j_mask, dtype=jnp.float32)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(use), axis=-1),
            jnp.max(jnp.abs(mask), axis=-1),
        ),
    )
    return (
        jnp.asarray(INFORMATION_TRIGGER_FLOAT32_ULPS, dtype=jnp.float32)
        * jnp.finfo(jnp.float32).eps
        * scale
    )


# Historical import name retained for callers; semantics are policy-level.
information_trigger_tolerance = policy_effect_trigger_tolerance


@dataclass(frozen=True, slots=True)
class VQBCResponseContrastRow:
    schema_version: str
    pairing_id: str
    episode_index: int
    episode_seed: int
    triggered: bool
    trigger_step: int | None
    trigger_tolerance: float | None
    predicted_response_effect: float | None
    predicted_policy_cost: float | None
    predicted_net_effect: float | None
    predicted_regularized_net_effect: float | None
    predicted_policy_total_variation: float | None
    maximum_action_net_value: float | None
    executed_action_net_value: float | None
    executed_action_response_value: float | None
    executed_action: int | None
    maximum_net_action: int | None
    post_response_belief_l1: float | None
    first_left_action_difference_step: int | None
    first_observation_difference_step: int | None
    first_response_code_difference_step: int | None
    first_reward_difference_step: int | None
    left_action_difference_count: int | None
    observation_difference_count: int | None
    response_code_difference_count: int | None
    reward_difference_count: int | None
    a1_raw_return: float
    a2_mask_raw_return: float
    a2_use_raw_return: float

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_CONTRAST_SCHEMA_VERSION:
            raise ValueError("Response-contrast row schema changed.")
        float_trigger_fields = (
            "trigger_tolerance",
            "predicted_response_effect",
            "predicted_policy_cost",
            "predicted_net_effect",
            "predicted_regularized_net_effect",
            "predicted_policy_total_variation",
            "maximum_action_net_value",
            "executed_action_net_value",
            "executed_action_response_value",
            "post_response_belief_l1",
        )
        action_fields = ("executed_action", "maximum_net_action")
        optional_step_fields = (
            "first_left_action_difference_step",
            "first_observation_difference_step",
            "first_response_code_difference_step",
            "first_reward_difference_step",
        )
        count_fields = (
            "left_action_difference_count",
            "observation_difference_count",
            "response_code_difference_count",
            "reward_difference_count",
        )
        if self.triggered:
            if (
                isinstance(self.trigger_step, bool)
                or not isinstance(self.trigger_step, Integral)
                or not 0 <= int(self.trigger_step) < 400
            ):
                raise ValueError("A triggered row requires a valid first trigger step.")
            for name in float_trigger_fields:
                value = getattr(self, name)
                if (
                    value is None
                    or isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"Triggered field {name} must be finite.")
            if float(self.trigger_tolerance) < 0.0:
                raise ValueError("Trigger tolerance cannot be negative.")
            if float(self.predicted_net_effect) <= float(self.trigger_tolerance):
                raise ValueError(
                    "A trigger requires policy-level raw net value above tolerance."
                )
            tv = float(self.predicted_policy_total_variation)
            if not 0.0 <= tv <= 1.0 + 1.0e-6:
                raise ValueError("Policy total variation must lie in [0, 1].")
            belief_l1 = float(self.post_response_belief_l1)
            if not 0.0 <= belief_l1 <= 2.0 + 1.0e-6:
                raise ValueError("Post-response belief L1 must lie in [0, 2].")
            for name in action_fields:
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"Triggered field {name} must be an action index.")
                if not 0 <= int(value) < 6:
                    raise ValueError(f"Triggered field {name} is outside six actions.")
            for name in optional_step_fields:
                value = getattr(self, name)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or not int(self.trigger_step) <= int(value) < 400
                ):
                    raise ValueError(f"{name} must be null or a post-trigger step.")
            for name in count_fields:
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or int(value) < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer.")
            paired = (
                ("first_left_action_difference_step", "left_action_difference_count"),
                ("first_observation_difference_step", "observation_difference_count"),
                ("first_response_code_difference_step", "response_code_difference_count"),
                ("first_reward_difference_step", "reward_difference_count"),
            )
            for step_name, count_name in paired:
                first = getattr(self, step_name)
                count = int(getattr(self, count_name))
                if (first is None) != (count == 0):
                    raise ValueError(
                        f"{step_name} and {count_name} disagree about divergence."
                    )
        else:
            trigger_fields = (
                *float_trigger_fields,
                *action_fields,
                *optional_step_fields,
                *count_fields,
            )
            if self.trigger_step is not None or any(
                getattr(self, name) is not None for name in trigger_fields
            ):
                raise ValueError("An untriggered row cannot contain trigger diagnostics.")
        for name in ("a1_raw_return", "a2_mask_raw_return", "a2_use_raw_return"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite.")
        if not self.triggered and len(
            {
                float(self.a1_raw_return),
                float(self.a2_mask_raw_return),
                float(self.a2_use_raw_return),
            }
        ) != 1:
            raise ValueError("Untriggered branches must be identical in return.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def first_positive_trigger(
    predicted_net_effects: Any, tolerances: Any
) -> tuple[int | None, float | None]:
    """Find the first policy-level raw net effect above its numerical floor."""

    import numpy as np

    values = np.asarray(predicted_net_effects, dtype=np.float64)
    floors = np.asarray(tolerances, dtype=np.float64)
    if values.ndim != 1 or floors.shape != values.shape:
        raise ValueError("Policy effect and tolerance sequences must be aligned vectors.")
    positive = np.argwhere(values > floors)
    if positive.size == 0:
        return None, None
    step = int(positive[0, 0])
    return step, float(values[step])


def effect_components(row: VQBCResponseContrastRow) -> Mapping[str, float]:
    return {
        "A2-use_minus_A2-mask": row.a2_use_raw_return - row.a2_mask_raw_return,
        "A1_minus_A2-mask": row.a1_raw_return - row.a2_mask_raw_return,
        "A2-use_minus_A1": row.a2_use_raw_return - row.a1_raw_return,
    }


def equal_frequency_bins(
    rows: Sequence[VQBCResponseContrastRow], *, bin_count: int = 10
) -> tuple[Mapping[str, Any], ...]:
    triggered = sorted(
        (row for row in rows if row.triggered),
        key=lambda row: (
            float(row.predicted_net_effect),
            row.pairing_id,
            row.episode_index,
        ),
    )
    if len(triggered) < bin_count:
        raise ValueError("Ten equal-frequency bins require at least ten triggers.")
    bins = []
    for index in range(bin_count):
        start = index * len(triggered) // bin_count
        end = (index + 1) * len(triggered) // bin_count
        members = triggered[start:end]
        effects = [effect_components(row) for row in members]
        bins.append(
            {
                "bin_index": index,
                "count": len(members),
                "minimum_predicted_net_effect": float(
                    members[0].predicted_net_effect
                ),
                "maximum_predicted_net_effect": float(
                    members[-1].predicted_net_effect
                ),
                "mean_predicted_response_effect": sum(
                    float(row.predicted_response_effect) for row in members
                )
                / len(members),
                "mean_predicted_policy_cost": sum(
                    float(row.predicted_policy_cost) for row in members
                )
                / len(members),
                "mean_predicted_policy_total_variation": sum(
                    float(row.predicted_policy_total_variation) for row in members
                )
                / len(members),
                "mean_post_response_belief_l1": sum(
                    float(row.post_response_belief_l1) for row in members
                )
                / len(members),
                "action_divergence_rate": sum(
                    int(row.left_action_difference_count) > 0 for row in members
                )
                / len(members),
                "reward_divergence_rate": sum(
                    int(row.reward_difference_count) > 0 for row in members
                )
                / len(members),
                "effects": {
                    name: sum(item[name] for item in effects) / len(effects)
                    for name in effects[0]
                },
            }
        )
    return tuple(bins)


def summarize_response_contrast(
    rows: Sequence[VQBCResponseContrastRow],
) -> Mapping[str, Any]:
    values = tuple(rows)
    trigger_count = sum(row.triggered for row in values)
    base: dict[str, Any] = {
        "schema_version": RESPONSE_CONTRAST_SUMMARY_VERSION,
        "scientific_readout_allowed": False,
        "row_count": len(values),
        "trigger_count": trigger_count,
    }
    if trigger_count:
        triggered = [row for row in values if row.triggered]
        base.update(
            {
                "mean_predicted_response_effect": sum(
                    float(row.predicted_response_effect) for row in triggered
                )
                / trigger_count,
                "mean_predicted_policy_cost": sum(
                    float(row.predicted_policy_cost) for row in triggered
                )
                / trigger_count,
                "mean_predicted_net_effect": sum(
                    float(row.predicted_net_effect) for row in triggered
                )
                / trigger_count,
                "mean_predicted_policy_total_variation": sum(
                    float(row.predicted_policy_total_variation) for row in triggered
                )
                / trigger_count,
                "mean_post_response_belief_l1": sum(
                    float(row.post_response_belief_l1) for row in triggered
                )
                / trigger_count,
                "left_action_divergence_episode_count": sum(
                    int(row.left_action_difference_count) > 0 for row in triggered
                ),
                "observation_divergence_episode_count": sum(
                    int(row.observation_difference_count) > 0 for row in triggered
                ),
                "response_code_divergence_episode_count": sum(
                    int(row.response_code_difference_count) > 0 for row in triggered
                ),
                "reward_divergence_episode_count": sum(
                    int(row.reward_difference_count) > 0 for row in triggered
                ),
            }
        )
    if trigger_count < 10:
        return {
            **base,
            "equal_frequency_bins": [],
            "primary_effect_non_decreasing_across_bins": None,
            "primary_effect_linear_slope_per_bin": None,
            "highest_bin_primary_effect": None,
            "binning_status": "fewer_than_ten_triggered_episodes",
        }
    bins = equal_frequency_bins(values)
    primary = [item["effects"]["A2-use_minus_A2-mask"] for item in bins]
    center = (len(primary) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(primary)))
    slope = sum(
        (index - center) * value for index, value in enumerate(primary)
    ) / denominator
    return {
        **base,
        "equal_frequency_bins": list(bins),
        "primary_effect_non_decreasing_across_bins": all(
            right >= left for left, right in zip(primary, primary[1:])
        ),
        "primary_effect_linear_slope_per_bin": slope,
        "highest_bin_primary_effect": primary[-1],
        "binning_status": "complete",
    }


__all__ = [
    "RESPONSE_CONTRAST_BRANCHES",
    "RESPONSE_CONTRAST_SCHEMA_VERSION",
    "RESPONSE_CONTRAST_SUMMARY_VERSION",
    "INFORMATION_TRIGGER_FLOAT32_ULPS",
    "VQBCResponseContrastRow",
    "effect_components",
    "equal_frequency_bins",
    "first_positive_trigger",
    "information_trigger_tolerance",
    "policy_effect_trigger_tolerance",
    "summarize_response_contrast",
]
