"""Official-policy rollout adapter for the unchanged R015 support evidence schema."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.overcooked_v2.path_c_pool_admission import (
    R015_EPISODE_EVIDENCE_SCHEMA_VERSION,
    R015PartnerSupportSpec,
    select_r015_support_members,
    validate_r015_candidate_artifact_provenance,
    validate_r015_support_report_evidence,
)
from experiments.overcooked_v2.path_c_standard_training import derive_standard_seed


@runtime_checkable
class OfficialR015RolloutBackend(Protocol):
    """JAX-native evaluator supplied by the official experiments-package adapter."""

    def evaluate_pairing(
        self,
        policy_0_candidate_id: str,
        policy_1_candidate_id: str,
        *,
        canonical_episode_seeds: Sequence[int],
    ) -> Sequence[Mapping[str, Any]]:
        """Return one metrics mapping for each seed, in the same order."""


_ROLLOUT_FIELDS = frozenset(
    {
        "raw_episode_return",
        "correct_delivery_count",
        "wrong_delivery_count",
        "indicator_activation_count",
        "ambiguous_reward_step_count",
        "policy_0_action_counts",
        "policy_1_action_counts",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _action_distribution_jsd(
    left_counts: Sequence[int],
    right_counts: Sequence[int],
) -> float:
    left = np.asarray(left_counts, dtype=np.float64)
    right = np.asarray(right_counts, dtype=np.float64)
    if left.shape != (6,) or right.shape != (6,) or left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("Official evidence requires positive six-action counts.")
    left /= left.sum()
    right /= right.sum()
    mixture = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(
            np.sum(values[positive] * np.log(values[positive] / mixture[positive]))
        )

    return 0.5 * divergence(left) + 0.5 * divergence(right)


def _normalize_rollout_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _ROLLOUT_FIELDS:
        raise ValueError("Official rollout metrics have the wrong evidence fields.")
    raw_return = float(raw["raw_episode_return"])
    if not math.isfinite(raw_return):
        raise ValueError("Official rollout return must be finite.")
    normalized: dict[str, Any] = {"raw_episode_return": raw_return}
    for field in (
        "correct_delivery_count",
        "wrong_delivery_count",
        "indicator_activation_count",
        "ambiguous_reward_step_count",
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Official rollout {field} must be non-negative.")
        normalized[field] = int(value)
    for field in ("policy_0_action_counts", "policy_1_action_counts"):
        counts = list(raw[field])
        if len(counts) != 6 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ) or sum(counts) != 400:
            raise ValueError("Official rollout action counts must cover 400 decisions.")
        normalized[field] = counts
    return normalized


def generate_official_r015_support_report(
    spec: R015PartnerSupportSpec,
    rollout_backend: OfficialR015RolloutBackend,
    *,
    provenance_validator: Callable[..., Mapping[str, Any]] = (
        validate_r015_candidate_artifact_provenance
    ),
) -> dict[str, Any]:
    """Generate 4 by 4 by 100 rows, then pass the existing static validator."""

    if not isinstance(rollout_backend, OfficialR015RolloutBackend):
        raise TypeError("Official R015 evidence requires the registered rollout backend.")
    candidate_ids = [candidate.candidate_id for candidate in spec.candidates]
    pairings = [
        (left_id, right_id)
        for left_id in candidate_ids
        for right_id in candidate_ids
    ]
    rows: list[dict[str, Any]] = []
    for left_id, right_id in pairings:
        canonical_seeds = [
            derive_standard_seed(
                spec.evaluation_seed,
                "r015_partner_support",
                "test_time_simple",
                left_id,
                right_id,
                episode_index,
            )
            for episode_index in range(spec.episodes_per_pairing)
        ]
        raw_rows = rollout_backend.evaluate_pairing(
            left_id,
            right_id,
            canonical_episode_seeds=tuple(canonical_seeds),
        )
        if len(raw_rows) != spec.episodes_per_pairing:
            raise ValueError("Official rollout pairing did not return exactly 100 episodes.")
        left = next(item for item in spec.candidates if item.candidate_id == left_id)
        right = next(item for item in spec.candidates if item.candidate_id == right_id)
        for episode_index, (canonical_seed, raw) in enumerate(
            zip(canonical_seeds, raw_rows, strict=True)
        ):
            metrics = _normalize_rollout_metrics(raw)
            rows.append(
                {
                    "schema_version": "path_c_r015_partner_support_episode_v1",
                    "scientific_readout_allowed": False,
                    "layout": "test_time_simple",
                    "policy_0_candidate_id": left_id,
                    "policy_1_candidate_id": right_id,
                    "policy_0_family_id": left.family_id,
                    "policy_1_family_id": right.family_id,
                    "episode_index": episode_index,
                    "canonical_episode_seed": canonical_seed,
                    "environment_steps": 400,
                    **metrics,
                }
            )
    if len(rows) != 1_600:
        raise ValueError("Official R015 evidence must contain exactly 1,600 rows.")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = spec.output_dir / "episode_returns.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in spec.candidates:
        provenance = dict(
            provenance_validator(spec, candidate, reported=None)
        )
        self_rows = [
            row
            for row in rows
            if row["policy_0_candidate_id"] == candidate.candidate_id
            and row["policy_1_candidate_id"] == candidate.candidate_id
        ]
        correct_rate = float(
            np.mean([row["correct_delivery_count"] for row in self_rows])
        )
        candidates[candidate.candidate_id] = {
            **provenance,
            "action_rule": spec.family(candidate.family_id).action_rule,
            "correct_delivery_rate": correct_rate,
            "mean_raw_episode_return": float(
                np.mean([row["raw_episode_return"] for row in self_rows])
            ),
            "admitted": correct_rate >= spec.admission_floor,
        }
    selection = select_r015_support_members(spec, candidates)
    grouped = {
        pairing: [
            row
            for row in rows
            if (row["policy_0_candidate_id"], row["policy_1_candidate_id"])
            == pairing
        ]
        for pairing in pairings
    }
    report = {
        "schema_version": "path_c_r015_partner_support_report_v2",
        "registration_status": "configured",
        "support_status": selection["support_status"],
        "artifact_verification_status": "verified",
        "support_complete": selection["support_complete"],
        "scientific_readout_allowed": False,
        "support_registration_sha256": spec.support_registration_sha256,
        "admission_implementation_sha256": spec.admission_implementation_sha256,
        "admission_implementation_dependencies": dict(
            spec.admission_implementation_dependencies
        ),
        "evaluation_seed": spec.evaluation_seed,
        "environment_config_sha256": spec.environment_config_sha256,
        "layout": "test_time_simple",
        "episodes_per_pairing": spec.episodes_per_pairing,
        "admission_floor": spec.admission_floor,
        "episode_evidence": {
            "schema_version": R015_EPISODE_EVIDENCE_SCHEMA_VERSION,
            "path": str(evidence_path),
            "sha256": _file_sha256(evidence_path),
            "row_count": len(rows),
        },
        "families": {
            family.family_id: {
                **family.definition(),
                "family_spec_sha256": family.family_spec_sha256,
            }
            for family in spec.families
        },
        "candidates": candidates,
        "qualified_candidate_ids": selection["qualified_candidate_ids"],
        "members_by_family": selection["members_by_family"],
        "members": selection["members"],
        "pairing_matrix": [
            {
                "policy_0_candidate_id": left_id,
                "policy_1_candidate_id": right_id,
                "mean_raw_episode_return": float(
                    np.mean(
                        [row["raw_episode_return"] for row in grouped[(left_id, right_id)]]
                    )
                ),
                "correct_delivery_rate": float(
                    np.mean(
                        [
                            row["correct_delivery_count"]
                            for row in grouped[(left_id, right_id)]
                        ]
                    )
                ),
                "action_distribution_jsd": _action_distribution_jsd(
                    np.sum(
                        [
                            row["policy_0_action_counts"]
                            for row in grouped[(left_id, right_id)]
                        ],
                        axis=0,
                    ),
                    np.sum(
                        [
                            row["policy_1_action_counts"]
                            for row in grouped[(left_id, right_id)]
                        ],
                        axis=0,
                    ),
                ),
            }
            for left_id, right_id in pairings
        ],
    }
    validate_r015_support_report_evidence(
        spec,
        report,
        candidate_provenance_validator=provenance_validator,
    )
    (spec.output_dir / "partner_support_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
