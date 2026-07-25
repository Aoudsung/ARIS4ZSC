"""Configuration-driven self-play and directed cross-play execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PAIRING_ROWS_SCHEMA_VERSION = "path_c_pairing_rows_v2"


@dataclass(frozen=True, slots=True)
class PairingSpec:
    pairing_id: str
    policy_0_id: str
    policy_1_id: str
    policy_0_source: str
    policy_1_source: str
    randomization_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.pairing_id,
                self.policy_0_id,
                self.policy_1_id,
                self.policy_0_source,
                self.policy_1_source,
            )
        ):
            raise ValueError("Pairing identifiers and policy sources must be non-empty strings.")
        if self.randomization_id is None:
            object.__setattr__(self, "randomization_id", self.pairing_id)
        elif not isinstance(self.randomization_id, str) or not self.randomization_id:
            raise ValueError("randomization_id must be a non-empty string when supplied.")
        allowed_sources = {
            "adaptation_checkpoint",
            "frozen_official_partner",
            "frozen_official_backbone",
        }
        if self.policy_0_source not in allowed_sources or self.policy_1_source not in allowed_sources:
            raise ValueError("A policy source is not registered.")


def development_pairings(
    focal_policy_id: str,
    partner_ids: Sequence[str],
    *,
    focal_policy_source: str = "adaptation_checkpoint",
) -> tuple[PairingSpec, ...]:
    """Return one self pairing and eight directed pairings for four partners."""

    partners = tuple(str(value) for value in partner_ids)
    if len(partners) != 4 or len(set(partners)) != 4:
        raise ValueError("Development evaluation requires four distinct frozen partners.")
    pairings = [
        PairingSpec(
            pairing_id=f"{focal_policy_id}__seat0__{focal_policy_id}__seat1",
            policy_0_id=focal_policy_id,
            policy_1_id=focal_policy_id,
            policy_0_source=focal_policy_source,
            policy_1_source=focal_policy_source,
            randomization_id="development_focal_self_pairing",
        )
    ]
    for partner in partners:
        pairings.extend(
            (
                PairingSpec(
                    pairing_id=f"{focal_policy_id}__seat0__{partner}__seat1",
                    policy_0_id=focal_policy_id,
                    policy_1_id=partner,
                    policy_0_source=focal_policy_source,
                    policy_1_source="frozen_official_partner",
                    randomization_id=f"development_focal_seat0__{partner}__seat1",
                ),
                PairingSpec(
                    pairing_id=f"{partner}__seat0__{focal_policy_id}__seat1",
                    policy_0_id=partner,
                    policy_1_id=focal_policy_id,
                    policy_0_source="frozen_official_partner",
                    policy_1_source=focal_policy_source,
                    randomization_id=f"development_{partner}__seat0__focal_seat1",
                ),
            )
        )
    return tuple(pairings)


def validate_formal_pairings(pairings: Sequence[PairingSpec]) -> tuple[PairingSpec, ...]:
    """Validate a caller-supplied ten-policy directed matrix without generating it."""

    values = tuple(pairings)
    if len(values) != 100 or len({item.pairing_id for item in values}) != 100:
        raise ValueError("Formal evaluation requires 100 unique directed pairings.")
    policy_ids = {item.policy_0_id for item in values} | {item.policy_1_id for item in values}
    if len(policy_ids) != 10:
        raise ValueError("Formal evaluation requires exactly ten policy identifiers.")
    self_pairings = [item for item in values if item.policy_0_id == item.policy_1_id]
    cross_pairings = [item for item in values if item.policy_0_id != item.policy_1_id]
    if len(self_pairings) != 10 or len(cross_pairings) != 90:
        raise ValueError("Formal evaluation requires 10 self and 90 directed cross pairings.")
    expected = {(left, right) for left in policy_ids for right in policy_ids}
    observed = {(item.policy_0_id, item.policy_1_id) for item in values}
    if observed != expected:
        raise ValueError("Formal pairing matrix is incomplete or repeats an ordered pair.")
    return values


def canonical_episode_seed(evaluation_seed: int, pairing_id: str, episode_index: int) -> int:
    if (
        isinstance(evaluation_seed, bool)
        or not isinstance(evaluation_seed, Integral)
        or int(evaluation_seed) < 0
    ):
        raise ValueError("evaluation_seed must be a non-negative integer.")
    if not isinstance(pairing_id, str) or not pairing_id:
        raise ValueError("pairing_id must be a non-empty string.")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, Integral)
        or int(episode_index) < 0
    ):
        raise ValueError("episode_index must be a non-negative integer.")
    payload = f"path_c_evaluation_seed_v1\0{int(evaluation_seed)}\0{pairing_id}\0{int(episode_index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _normalize_result(
    pairing: PairingSpec, episode_index: int, seed: int, result: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = {
        "raw_return",
        "probe_count",
        "safe_candidate_opportunity_count",
        "maximum_probe_budget",
        "correct_delivery_count",
        "wrong_delivery_count",
        "indicator_cost",
    }
    unknown = sorted(set(result) - allowed)
    missing = sorted(
        {
            "raw_return",
            "probe_count",
            "safe_candidate_opportunity_count",
            "maximum_probe_budget",
        }
        - set(result)
    )
    if unknown or missing:
        raise ValueError(f"Pairing result fields are invalid; unknown={unknown}, missing={missing}.")
    raw_return = result["raw_return"]
    indicator_cost = result.get("indicator_cost", 0.0)
    for name, value in (("raw_return", raw_return), ("indicator_cost", indicator_cost)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(
            float(value)
        ):
            raise ValueError(f"{name} must be finite and numeric.")
    counts = {
        "probe_count": result["probe_count"],
        "safe_candidate_opportunity_count": result[
            "safe_candidate_opportunity_count"
        ],
        "maximum_probe_budget": result["maximum_probe_budget"],
        "correct_delivery_count": result.get("correct_delivery_count", 0),
        "wrong_delivery_count": result.get("wrong_delivery_count", 0),
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    probe_count = int(counts["probe_count"])
    safe_opportunities = int(counts["safe_candidate_opportunity_count"])
    maximum_budget = int(counts["maximum_probe_budget"])
    if probe_count > safe_opportunities:
        raise ValueError("probe_count cannot exceed safe candidate opportunities.")
    if probe_count > maximum_budget:
        raise ValueError("probe_count cannot exceed the maximum probe budget.")
    if maximum_budget == 0:
        if probe_count or safe_opportunities:
            raise ValueError("A pure frozen pairing cannot report probe opportunities or probes.")
        trigger_rate = None
        budget_usage_rate = None
    else:
        trigger_rate = probe_count / safe_opportunities if safe_opportunities else 0.0
        budget_usage_rate = probe_count / maximum_budget
    return {
        "schema_version": PAIRING_ROWS_SCHEMA_VERSION,
        "pairing_id": pairing.pairing_id,
        "randomization_id": pairing.randomization_id,
        "policy_0_id": pairing.policy_0_id,
        "policy_1_id": pairing.policy_1_id,
        "seat_assignment": {"seat_0": pairing.policy_0_id, "seat_1": pairing.policy_1_id},
        "episode_index": int(episode_index),
        "episode_seed": int(seed),
        "raw_return": float(raw_return),
        "probe_count": probe_count,
        "safe_candidate_opportunity_count": safe_opportunities,
        "probe_trigger_rate": trigger_rate,
        "maximum_probe_budget": maximum_budget,
        "probe_budget_usage_rate": budget_usage_rate,
        "correct_delivery_count": int(counts["correct_delivery_count"]),
        "wrong_delivery_count": int(counts["wrong_delivery_count"]),
        "indicator_cost": float(indicator_cost),
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def execute_pairings(
    pairings: Sequence[PairingSpec],
    *,
    episodes_per_pairing: int,
    evaluation_seed: int,
    evaluate_pairing: Callable[[PairingSpec, Sequence[int]], Sequence[Mapping[str, Any]]],
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute every pairing and write only normalized per-episode raw rows."""

    if (
        isinstance(episodes_per_pairing, bool)
        or not isinstance(episodes_per_pairing, Integral)
        or int(episodes_per_pairing) <= 0
    ):
        raise ValueError("episodes_per_pairing must be positive.")
    episodes_per_pairing = int(episodes_per_pairing)
    values = tuple(pairings)
    if not values or len({item.pairing_id for item in values}) != len(values):
        raise ValueError("Pairing identifiers must be non-empty and unique.")
    rows: list[dict[str, Any]] = []
    for pairing in values:
        seeds = [
            canonical_episode_seed(
                evaluation_seed, str(pairing.randomization_id), index
            )
            for index in range(episodes_per_pairing)
        ]
        results = tuple(evaluate_pairing(pairing, seeds))
        if len(results) != episodes_per_pairing:
            raise RuntimeError("A pairing evaluator returned the wrong episode count.")
        rows.extend(
            _normalize_result(pairing, index, seeds[index], result)
            for index, result in enumerate(results)
        )
    if output_path is not None:
        _atomic_text(
            Path(output_path).resolve(),
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        )
    return rows


def execute_formal_pairings(
    pairings: Sequence[PairingSpec],
    *,
    evaluation_seed: int,
    evaluate_pairing: Callable[[PairingSpec, Sequence[int]], Sequence[Mapping[str, Any]]],
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute the caller-supplied 10-by-10 matrix at 500 episodes per pair."""

    return execute_pairings(
        validate_formal_pairings(pairings),
        episodes_per_pairing=500,
        evaluation_seed=evaluation_seed,
        evaluate_pairing=evaluate_pairing,
        output_path=output_path,
    )
