"""Standard zero-shot coordination evaluation and response-use contrast."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .method import DEPLOYMENT_MODES, policy_effect_trigger_tolerance

OUTER_UNIT_COUNT = 10
EPISODES_PER_PAIRING = 500
RESPONSE_CONTRAST_BRANCHES = ("A1", "A2-mask", "A2-use")


@dataclass(frozen=True, slots=True)
class PopulationEntry:
    outer_unit_id: int
    checkpoint_path: Path
    reference_checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class Population:
    name: str
    layout: str
    evaluation_kind: str
    entries: tuple[PopulationEntry, ...]


@dataclass(frozen=True, slots=True)
class Pairing:
    deployment_mode: str
    split: str
    left_outer_unit_id: int
    right_outer_unit_id: int

    @property
    def pairing_id(self) -> str:
        return (
            f"{self.left_outer_unit_id:02d}_to_"
            f"{self.right_outer_unit_id:02d}"
        )


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    population: str
    layout: str
    deployment_mode: str
    split: str
    pairing_id: str
    left_outer_unit_id: int
    right_outer_unit_id: int
    episode_index: int
    episode_seed: int
    environment_steps: int
    raw_return: float
    correct_delivery_count: int
    wrong_delivery_count: int
    indicator_activation_count: int
    positive_predicted_response_effect_count: int
    cumulative_kl: float
    reference_action_deviation_count: int
    mean_value_class_count: float
    mean_belief_entropy: float
    response_code_counts: tuple[int, ...]

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["response_code_counts"] = list(self.response_code_counts)
        return payload


@dataclass(frozen=True, slots=True)
class ResponseContrastRow:
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
    environment_steps: int
    a1_raw_return: float
    a1_correct_delivery_count: int
    a1_wrong_delivery_count: int
    a1_indicator_activation_count: int
    a2_mask_raw_return: float
    a2_mask_correct_delivery_count: int
    a2_mask_wrong_delivery_count: int
    a2_mask_indicator_activation_count: int
    a2_use_raw_return: float
    a2_use_correct_delivery_count: int
    a2_use_wrong_delivery_count: int
    a2_use_indicator_activation_count: int

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_population(path: str | Path) -> Population:
    """Load paths and unit numbers; Orbax validates checkpoint readability."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {"name", "layout", "evaluation_kind", "policies"}
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError(
            "Population manifest fields must be name, layout, "
            "evaluation_kind, and policies."
        )
    if payload["layout"] not in {"test_time_simple", "test_time_wide"}:
        raise ValueError("Population manifest has an unknown layout.")
    if not isinstance(payload["policies"], list):
        raise ValueError("Population policies must be a list.")
    policy_fields = {
        "outer_unit_id",
        "checkpoint_path",
        "reference_checkpoint_path",
    }
    if any(
        not isinstance(item, Mapping) or set(item) != policy_fields
        for item in payload["policies"]
    ):
        raise ValueError("Every population policy must use the three public fields.")
    entries = tuple(
        PopulationEntry(
            outer_unit_id=int(item["outer_unit_id"]),
            checkpoint_path=_relative_path(
                manifest_path.parent, item["checkpoint_path"]
            ),
            reference_checkpoint_path=_relative_path(
                manifest_path.parent, item["reference_checkpoint_path"]
            ),
        )
        for item in payload["policies"]
    )
    if len(entries) != OUTER_UNIT_COUNT:
        raise ValueError("A standard population contains ten policies.")
    if tuple(entry.outer_unit_id for entry in entries) != tuple(
        range(OUTER_UNIT_COUNT)
    ):
        raise ValueError("Population entries must be ordered by training unit.")
    if len({entry.checkpoint_path for entry in entries}) != OUTER_UNIT_COUNT:
        raise ValueError("Each training unit must load a different final policy.")
    if len({entry.reference_checkpoint_path for entry in entries}) != OUTER_UNIT_COUNT:
        raise ValueError("Each training unit must load a different reference policy.")
    evaluation_kind = str(payload["evaluation_kind"])
    if evaluation_kind not in {"standard_matrix", "response_contrast"}:
        raise ValueError("Unknown evaluation kind in population manifest.")
    return Population(
        name=str(payload["name"]),
        layout=str(payload["layout"]),
        evaluation_kind=evaluation_kind,
        entries=entries,
    )


def write_population(path: str | Path, population: Population) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": population.name,
        "layout": population.layout,
        "evaluation_kind": population.evaluation_kind,
        "policies": [
            {
                "outer_unit_id": entry.outer_unit_id,
                "checkpoint_path": str(entry.checkpoint_path),
                "reference_checkpoint_path": str(
                    entry.reference_checkpoint_path
                ),
            }
            for entry in population.entries
        ],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _relative_path(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else base / path).resolve()


def standard_pairings(
    deployment_modes: Sequence[str] = DEPLOYMENT_MODES,
) -> tuple[Pairing, ...]:
    pairings = []
    for mode in deployment_modes:
        if mode not in DEPLOYMENT_MODES:
            raise ValueError(f"Unknown deployment mode: {mode}")
        for left in range(OUTER_UNIT_COUNT):
            for right in range(OUTER_UNIT_COUNT):
                pairings.append(
                    Pairing(
                        deployment_mode=mode,
                        split="sp" if left == right else "xp",
                        left_outer_unit_id=left,
                        right_outer_unit_id=right,
                    )
                )
    return tuple(pairings)


def standard_episode_seed(
    *,
    evaluation_seed: int,
    layout: str,
    left_outer_unit_id: int,
    right_outer_unit_id: int,
    episode_index: int,
) -> int:
    """Derive matched environment seeds without including deployment mode."""

    payload = (
        f"{int(evaluation_seed)}\0{layout}\0{left_outer_unit_id}\0"
        f"{right_outer_unit_id}\0{episode_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def validate_standard_rows(
    rows: Sequence[EpisodeRow],
    *,
    deployment_modes: Sequence[str] = DEPLOYMENT_MODES,
    episodes_per_pairing: int = EPISODES_PER_PAIRING,
) -> None:
    expected = len(tuple(deployment_modes)) * 100 * int(episodes_per_pairing)
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} evaluation rows, received {len(rows)}."
        )
    keys = {
        (
            row.deployment_mode,
            row.left_outer_unit_id,
            row.right_outer_unit_id,
            row.episode_index,
        )
        for row in rows
    }
    if len(keys) != expected:
        raise ValueError("Evaluation rows contain duplicate matrix cells.")
    seeds: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    allowed_modes = set(deployment_modes)
    for row in rows:
        if row.deployment_mode not in allowed_modes:
            raise ValueError("An evaluation row has an unknown deployment mode.")
        if not (
            0 <= row.left_outer_unit_id < OUTER_UNIT_COUNT
            and 0 <= row.right_outer_unit_id < OUTER_UNIT_COUNT
            and 0 <= row.episode_index < int(episodes_per_pairing)
        ):
            raise ValueError("An evaluation row is outside the standard matrix.")
        expected_pairing = (
            f"{row.left_outer_unit_id:02d}_to_"
            f"{row.right_outer_unit_id:02d}"
        )
        if row.pairing_id != expected_pairing:
            raise ValueError("An evaluation row has the wrong pairing identifier.")
        expected_split = (
            "sp"
            if row.left_outer_unit_id == row.right_outer_unit_id
            else "xp"
        )
        if row.split != expected_split:
            raise ValueError("A row has the wrong self-play/cross-play label.")
        if row.environment_steps <= 0:
            raise ValueError("An evaluation row has no environment steps.")
        if (
            row.correct_delivery_count < 0
            or row.wrong_delivery_count < 0
            or row.indicator_activation_count < 0
        ):
            raise ValueError("Delivery counts cannot be negative.")
        if not all(
            math.isfinite(value)
            for value in (
                row.raw_return,
                row.cumulative_kl,
                row.mean_value_class_count,
                row.mean_belief_entropy,
            )
        ):
            raise ValueError("Evaluation statistics must be finite.")
        seeds[
            (
                row.left_outer_unit_id,
                row.right_outer_unit_id,
                row.episode_index,
            )
        ].add(row.episode_seed)
    if any(len(values) != 1 for values in seeds.values()):
        raise ValueError("Matched deployment modes must share environment seeds.")


def summarize_standard_rows(
    rows: Sequence[EpisodeRow],
) -> Mapping[str, Any]:
    """Use pairing means as the official statistical units."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.deployment_mode, row.split, row.pairing_id)
        ].append(float(row.raw_return))
    modes: dict[str, Any] = {}
    for mode in sorted({row.deployment_mode for row in rows}):
        split_summary: dict[str, Any] = {}
        for split, expected in (("sp", 10), ("xp", 90)):
            pairing_means = [
                mean(values)
                for (group_mode, group_split, unused_pairing), values
                in grouped.items()
                if group_mode == mode and group_split == split
            ]
            if len(pairing_means) != expected:
                raise ValueError(
                    f"{mode} has {len(pairing_means)} {split} pairings."
                )
            split_summary[split] = {
                "pairing_count": len(pairing_means),
                "mean_raw_return": mean(pairing_means),
                "pairing_standard_deviation": stdev(pairing_means),
                "episode_count": sum(
                    1
                    for row in rows
                    if row.deployment_mode == mode and row.split == split
                ),
            }
        split_summary["sp_minus_xp"] = (
            split_summary["sp"]["mean_raw_return"]
            - split_summary["xp"]["mean_raw_return"]
        )
        modes[mode] = split_summary
    return {"deployment_modes": modes}


def effect_components(row: ResponseContrastRow) -> Mapping[str, float]:
    response = row.a2_use_raw_return - row.a2_mask_raw_return
    cost = row.a1_raw_return - row.a2_mask_raw_return
    net = row.a2_use_raw_return - row.a1_raw_return
    if not math.isclose(net, response - cost, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Response contrast effect identity is violated.")
    return {
        "delta_response": response,
        "delta_cost": cost,
        "delta_net": net,
    }


def validate_response_contrast_rows(
    rows: Sequence[ResponseContrastRow],
    *,
    evaluation_seed: int,
    layout: str,
    episodes_per_pairing: int = EPISODES_PER_PAIRING,
) -> None:
    """Validate the complete 90-cell directed response-use contrast."""

    expected_count = 90 * int(episodes_per_pairing)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} response-contrast rows, received {len(rows)}."
        )
    keys: set[tuple[str, int]] = set()
    for row in rows:
        parts = row.pairing_id.split("_to_")
        if len(parts) != 2:
            raise ValueError("Response contrast has an invalid pairing identifier.")
        left, right = (int(value) for value in parts)
        if not (
            0 <= left < OUTER_UNIT_COUNT
            and 0 <= right < OUTER_UNIT_COUNT
            and left != right
            and 0 <= row.episode_index < int(episodes_per_pairing)
        ):
            raise ValueError("Response contrast row is outside the directed XP matrix.")
        key = (row.pairing_id, row.episode_index)
        if key in keys:
            raise ValueError("Response contrast contains a duplicate episode.")
        keys.add(key)
        expected_seed = standard_episode_seed(
            evaluation_seed=evaluation_seed,
            layout=layout,
            left_outer_unit_id=left,
            right_outer_unit_id=right,
            episode_index=row.episode_index,
        )
        if row.episode_seed != expected_seed:
            raise ValueError("Response contrast row has the wrong environment seed.")
        branch_returns = (
            row.a1_raw_return,
            row.a2_mask_raw_return,
            row.a2_use_raw_return,
        )
        if not all(math.isfinite(value) for value in branch_returns):
            raise ValueError("Response contrast returns must be finite.")
        if not row.triggered:
            if not (
                row.a1_raw_return
                == row.a2_mask_raw_return
                == row.a2_use_raw_return
            ):
                raise ValueError("Untriggered response branches must be identical.")
            if any(
                value is not None
                for value in (
                    row.trigger_step,
                    row.trigger_tolerance,
                    row.predicted_response_effect,
                    row.predicted_policy_cost,
                    row.predicted_net_effect,
                )
            ):
                raise ValueError("Untriggered response rows cannot report a trigger.")
        else:
            required_trigger_fields = (
                row.trigger_step,
                row.trigger_tolerance,
                row.predicted_response_effect,
                row.predicted_policy_cost,
                row.predicted_net_effect,
                row.predicted_regularized_net_effect,
                row.predicted_policy_total_variation,
                row.maximum_action_net_value,
                row.executed_action_net_value,
                row.executed_action_response_value,
                row.executed_action,
                row.maximum_net_action,
                row.post_response_belief_l1,
                row.left_action_difference_count,
                row.observation_difference_count,
                row.response_code_difference_count,
                row.reward_difference_count,
            )
            if any(value is None for value in required_trigger_fields):
                raise ValueError("Triggered response rows must retain every result.")
            trigger_values = (
                row.trigger_tolerance,
                row.predicted_response_effect,
                row.predicted_policy_cost,
                row.predicted_net_effect,
                row.predicted_regularized_net_effect,
                row.predicted_policy_total_variation,
                row.maximum_action_net_value,
                row.executed_action_net_value,
                row.executed_action_response_value,
                row.post_response_belief_l1,
            )
            if not all(math.isfinite(float(value)) for value in trigger_values):
                raise ValueError("Triggered response values must be finite.")
        if row.environment_steps <= 0:
            raise ValueError("A response contrast row has no environment steps.")
        branch_counts = (
            row.a1_correct_delivery_count,
            row.a1_wrong_delivery_count,
            row.a1_indicator_activation_count,
            row.a2_mask_correct_delivery_count,
            row.a2_mask_wrong_delivery_count,
            row.a2_mask_indicator_activation_count,
            row.a2_use_correct_delivery_count,
            row.a2_use_wrong_delivery_count,
            row.a2_use_indicator_activation_count,
        )
        if any(value < 0 for value in branch_counts):
            raise ValueError("Response contrast counts cannot be negative.")
        effect_components(row)


def equal_frequency_bins(
    rows: Sequence[ResponseContrastRow], *, bin_count: int = 10
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
        return ()
    bins = []
    for index in range(bin_count):
        start = index * len(triggered) // bin_count
        end = (index + 1) * len(triggered) // bin_count
        members = triggered[start:end]
        effects = [effect_components(row) for row in members]
        bins.append(
            {
                "bin_index": index,
                "source_start": start,
                "source_end": end,
                "count": len(members),
                "minimum_predicted_net_effect": float(
                    members[0].predicted_net_effect
                ),
                "maximum_predicted_net_effect": float(
                    members[-1].predicted_net_effect
                ),
                "effects": {
                    name: mean(item[name] for item in effects)
                    for name in effects[0]
                },
            }
        )
    return tuple(bins)


def summarize_response_contrast(
    rows: Sequence[ResponseContrastRow],
) -> Mapping[str, Any]:
    values = tuple(rows)
    if not values:
        raise ValueError("Response contrast summary requires episode rows.")
    effects = [effect_components(row) for row in values]
    triggered = [row for row in values if row.triggered]
    return {
        "row_count": len(values),
        "trigger_count": len(triggered),
        "trigger_rate": len(triggered) / len(values),
        "mean_effects": {
            name: mean(item[name] for item in effects)
            for name in effects[0]
        },
        "equal_frequency_bins": list(equal_frequency_bins(values)),
    }


__all__ = [
    "DEPLOYMENT_MODES",
    "EPISODES_PER_PAIRING",
    "EpisodeRow",
    "OUTER_UNIT_COUNT",
    "Pairing",
    "Population",
    "PopulationEntry",
    "RESPONSE_CONTRAST_BRANCHES",
    "ResponseContrastRow",
    "effect_components",
    "equal_frequency_bins",
    "load_population",
    "policy_effect_trigger_tolerance",
    "standard_episode_seed",
    "standard_pairings",
    "summarize_response_contrast",
    "summarize_standard_rows",
    "validate_standard_rows",
    "validate_response_contrast_rows",
    "write_population",
]
