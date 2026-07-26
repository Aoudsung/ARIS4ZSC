"""Response-use versus response-mask matched-branch contracts and summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


RESPONSE_CONTRAST_BRANCHES = ("A1", "A2-mask", "A2-use")
RESPONSE_CONTRAST_SCHEMA_VERSION = "path_c_vqbc_response_contrast_rows_v1"


@dataclass(frozen=True, slots=True)
class VQBCResponseContrastRow:
    schema_version: str
    pairing_id: str
    episode_index: int
    episode_seed: int
    triggered: bool
    trigger_step: int | None
    predicted_information_net_value: float | None
    a1_raw_return: float
    a2_mask_raw_return: float
    a2_use_raw_return: float

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_CONTRAST_SCHEMA_VERSION:
            raise ValueError("Response-contrast row schema changed.")
        if self.triggered:
            if (
                isinstance(self.trigger_step, bool)
                or not isinstance(self.trigger_step, Integral)
                or not 0 <= int(self.trigger_step) < 400
                or self.predicted_information_net_value is None
                or float(self.predicted_information_net_value) <= 0.0
            ):
                raise ValueError("A triggered row requires the first positive S(a) step.")
        elif self.trigger_step is not None or self.predicted_information_net_value is not None:
            raise ValueError("An untriggered row cannot contain trigger data.")
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
            raise ValueError("Untriggered branches must be byte-equivalent in return.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def first_positive_trigger(information_net_values: Any) -> tuple[int | None, float | None]:
    import numpy as np

    values = np.asarray(information_net_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 6:
        raise ValueError("Trigger values must have [step, action] shape.")
    positive = np.argwhere(np.max(values, axis=-1) > 0.0)
    if positive.size == 0:
        return None, None
    step = int(positive[0, 0])
    return step, float(np.max(values[step]))


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
            float(row.predicted_information_net_value),
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
                "minimum_predicted_information_net_value": float(
                    members[0].predicted_information_net_value
                ),
                "maximum_predicted_information_net_value": float(
                    members[-1].predicted_information_net_value
                ),
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
    if trigger_count < 10:
        return {
            "schema_version": "path_c_vqbc_response_contrast_summary_v1",
            "scientific_readout_allowed": False,
            "row_count": len(values),
            "trigger_count": trigger_count,
            "equal_frequency_bins": [],
            "primary_effect_non_decreasing_across_bins": None,
            "primary_effect_linear_slope_per_bin": None,
            "highest_bin_primary_effect": None,
            "binning_status": "fewer_than_ten_triggered_episodes",
        }
    bins = equal_frequency_bins(values)
    primary = [
        item["effects"]["A2-use_minus_A2-mask"] for item in bins
    ]
    center = (len(primary) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(primary)))
    slope = sum(
        (index - center) * value for index, value in enumerate(primary)
    ) / denominator
    return {
        "schema_version": "path_c_vqbc_response_contrast_summary_v1",
        "scientific_readout_allowed": False,
        "row_count": len(values),
        "trigger_count": trigger_count,
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
    "VQBCResponseContrastRow",
    "effect_components",
    "equal_frequency_bins",
    "first_positive_trigger",
    "summarize_response_contrast",
]
