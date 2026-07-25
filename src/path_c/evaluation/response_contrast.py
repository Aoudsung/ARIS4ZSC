"""Matched deployment contrast for the registered response channel.

The contrast is deliberately separate from the standard self-play and
cross-play summary.  It contains only directed cross-play cells from the
decision-focused ten-policy population and records three coupled branches for
each episode block.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .standard import (
    EPISODE_STEPS,
    EPISODES_PER_PAIRING,
    PopulationManifest,
    StandardPairing,
    file_sha256,
    standard_episode_seed,
    standard_pairings,
)


RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION = "path_c_response_contrast_manifest_v1"
RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION = "path_c_response_contrast_rows_v1"
RESPONSE_CONTRAST_SUMMARY_SCHEMA_VERSION = "path_c_response_contrast_summary_v1"
RESPONSE_CONTRAST_BRANCHES = ("A1", "A2-mask", "A2-use")
RESPONSE_CONTRAST_PAIRING_COUNT = 90
RESPONSE_CONTRAST_BLOCK_COUNT = (
    RESPONSE_CONTRAST_PAIRING_COUNT * EPISODES_PER_PAIRING
)
RESPONSE_CONTRAST_ROW_COUNT = (
    RESPONSE_CONTRAST_BLOCK_COUNT * len(RESPONSE_CONTRAST_BRANCHES)
)


@dataclass(frozen=True, slots=True)
class ResponseContrastManifest:
    """Bind one response contrast to a verified decision-focused population."""

    population: PopulationManifest
    output_root: Path
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "ResponseContrastManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "population_manifest_path",
            "population_manifest_sha256",
            "output_root",
            "run_kind",
            "scientific_readout_allowed",
        }:
            raise ValueError("The response-contrast manifest fields changed.")
        if payload["schema_version"] != RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION}."
            )
        if (
            payload["run_kind"] != "formal"
            or payload["scientific_readout_allowed"] is not False
        ):
            raise ValueError(
                "The response contrast must remain a non-claim formal diagnostic."
            )
        population_path = Path(str(payload["population_manifest_path"]))
        if not population_path.is_absolute():
            population_path = manifest_path.parent / population_path
        population_path = population_path.resolve()
        if (
            not population_path.is_file()
            or file_sha256(population_path)
            != str(payload["population_manifest_sha256"])
        ):
            raise ValueError("The response-contrast population manifest changed.")
        population = PopulationManifest.load(population_path)
        response_contrast_pairings(population)
        if any(
            entry.resolved_config is None
            or not entry.resolved_config.is_family_pool
            or entry.calibration_summary is None
            or entry.calibration_summary.get("parameter_artifact")
            != "adaptation.manifest"
            for entry in population.policies
        ):
            raise ValueError(
                "The response contrast requires ten v3 deployment-calibrated policies."
            )
        output_root = Path(str(payload["output_root"]))
        if not output_root.is_absolute():
            output_root = manifest_path.parent / output_root
        return cls(
            population=population,
            output_root=output_root.resolve(),
            path=manifest_path,
            sha256=file_sha256(manifest_path),
        )


def response_contrast_pairings(
    manifest: PopulationManifest,
) -> tuple[StandardPairing, ...]:
    """Return the 90 directed off-diagonal cells for decision-focused policies."""

    if (
        manifest.source_type != "adaptation_checkpoint"
        or manifest.population_id != "decision_focused"
        or manifest.condition_id != "decision_focused"
        or manifest.controller != "registered_response_sequential_branch_v1"
    ):
        raise ValueError(
            "The response contrast requires one decision-focused adapted population."
        )
    pairings = tuple(
        pairing for pairing in standard_pairings(manifest) if pairing.split == "xp"
    )
    if len(pairings) != RESPONSE_CONTRAST_PAIRING_COUNT:
        raise RuntimeError("The response contrast did not construct 90 directed XP cells.")
    return pairings


def response_contrast_episode_seed(
    *,
    manifest: PopulationManifest,
    pairing: StandardPairing,
    episode_index: int,
) -> int:
    """Reuse the standard matrix seed for the corresponding matched XP block."""

    return standard_episode_seed(
        manifest=manifest,
        outer_unit_0=pairing.outer_unit_0,
        outer_unit_1=pairing.outer_unit_1,
        episode_index=episode_index,
    )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _normalize_branch(
    payload: Mapping[str, Any], *, name: str
) -> dict[str, int | float]:
    required = {
        "raw_return",
        "correct_delivery_count",
        "wrong_delivery_count",
        "probe_count",
        "safe_candidate_opportunity_count",
        "maximum_probe_budget",
        "indicator_cost",
    }
    if set(payload) != required:
        raise ValueError(f"{name} fields changed.")
    counts = {
        field: _count(payload[field], f"{name}.{field}")
        for field in (
            "correct_delivery_count",
            "wrong_delivery_count",
            "probe_count",
            "safe_candidate_opportunity_count",
            "maximum_probe_budget",
        )
    }
    if counts["probe_count"] > counts["safe_candidate_opportunity_count"]:
        raise ValueError(f"{name} probes exceed safe candidate opportunities.")
    if counts["probe_count"] > counts["maximum_probe_budget"]:
        raise ValueError(f"{name} probes exceed the maximum budget.")
    indicator_cost = _finite(payload["indicator_cost"], f"{name}.indicator_cost")
    if indicator_cost < 0.0:
        raise ValueError(f"{name}.indicator_cost must be non-negative.")
    return {
        "raw_return": _finite(payload["raw_return"], f"{name}.raw_return"),
        **counts,
        "indicator_cost": indicator_cost,
    }


def execute_response_contrast_pairing(
    manifest: PopulationManifest,
    pairing: StandardPairing,
    *,
    evaluate_pairing: Callable[
        [StandardPairing, Sequence[int]], Sequence[Mapping[str, Any]]
    ],
) -> list[dict[str, Any]]:
    """Normalize one 500-block paired response-channel experiment."""

    if pairing.split != "xp":
        raise ValueError("The response contrast is defined only for cross-play.")
    seeds = tuple(
        response_contrast_episode_seed(
            manifest=manifest,
            pairing=pairing,
            episode_index=index,
        )
        for index in range(EPISODES_PER_PAIRING)
    )
    if len(set(seeds)) != EPISODES_PER_PAIRING:
        raise RuntimeError("A response-contrast pairing contains duplicate seeds.")
    results = tuple(evaluate_pairing(pairing, seeds))
    if len(results) != EPISODES_PER_PAIRING:
        raise RuntimeError("A response-contrast evaluator must return 500 blocks.")
    rows: list[dict[str, Any]] = []
    for episode_index, (episode_seed, result) in enumerate(zip(seeds, results)):
        if not isinstance(result, Mapping):
            raise TypeError("A response-contrast block result must be a mapping.")
        if set(result) != {
            "triggered",
            "first_probe_step",
            "executed_environment_steps",
            "unique_environment_steps",
            "branches",
        }:
            raise ValueError("A response-contrast block result changed fields.")
        triggered = result["triggered"]
        if not isinstance(triggered, bool):
            raise ValueError("triggered must be boolean.")
        first_probe_step = result["first_probe_step"]
        if triggered:
            if (
                isinstance(first_probe_step, bool)
                or not isinstance(first_probe_step, Integral)
                or not 0 <= int(first_probe_step) < EPISODE_STEPS
            ):
                raise ValueError("A triggered block requires a valid first probe step.")
            normalized_probe_step: int | None = int(first_probe_step)
        elif first_probe_step is not None:
            raise ValueError("An untriggered block cannot have a first probe step.")
        else:
            normalized_probe_step = None
        executed_steps = _count(
            result["executed_environment_steps"], "executed_environment_steps"
        )
        unique_steps = _count(
            result["unique_environment_steps"], "unique_environment_steps"
        )
        if executed_steps != 3 * EPISODE_STEPS:
            raise ValueError("The registered runtime must account for three branch steps.")
        if not EPISODE_STEPS <= unique_steps <= 3 * EPISODE_STEPS:
            raise ValueError("unique_environment_steps is outside the matched-block bounds.")
        branches = result["branches"]
        if not isinstance(branches, Mapping) or set(branches) != set(
            RESPONSE_CONTRAST_BRANCHES
        ):
            raise ValueError("A response-contrast block must contain exactly three branches.")
        normalized = {
            branch: _normalize_branch(
                branches[branch], name=f"branches[{branch!r}]"
            )
            for branch in RESPONSE_CONTRAST_BRANCHES
        }
        if not triggered and len(
            {
                tuple(normalized[branch].items())
                for branch in RESPONSE_CONTRAST_BRANCHES
            }
        ) != 1:
            raise ValueError("Untriggered blocks must reuse one identical complete trajectory.")
        block_id = f"{pairing.pairing_id}__episode_{episode_index:03d}"
        for branch in RESPONSE_CONTRAST_BRANCHES:
            rows.append(
                {
                    "schema_version": RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION,
                    "population_id": manifest.population_id,
                    "population_manifest_sha256": manifest.sha256,
                    "outer_units_manifest_sha256": manifest.outer_units.sha256,
                    "layout": manifest.layout,
                    "split": "xp",
                    "pairing_id": pairing.pairing_id,
                    "matched_block_id": block_id,
                    "outer_unit_0": pairing.outer_unit_0,
                    "outer_unit_1": pairing.outer_unit_1,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "branch": branch,
                    "triggered": triggered,
                    "first_probe_step": normalized_probe_step,
                    "executed_environment_steps": executed_steps,
                    "unique_environment_steps": unique_steps,
                    **normalized[branch],
                }
            )
    return rows


def summarize_response_contrast_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Recompute paired response, cost, and net effects from raw branch rows."""

    if not rows:
        raise ValueError("Response-contrast rows cannot be empty.")
    by_block: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("schema_version") != RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION:
            raise ValueError("A response-contrast row has the wrong schema.")
        if row.get("split") != "xp":
            raise ValueError("The response contrast cannot contain self-play rows.")
        block_id = str(row.get("matched_block_id"))
        branch = str(row.get("branch"))
        if branch not in RESPONSE_CONTRAST_BRANCHES or branch in by_block[block_id]:
            raise ValueError("A response-contrast block has a duplicate or unknown branch.")
        by_block[block_id][branch] = row
    if any(set(branches) != set(RESPONSE_CONTRAST_BRANCHES) for branches in by_block.values()):
        raise ValueError("Every response-contrast block must contain all three branches.")

    response: list[float] = []
    cost: list[float] = []
    net: list[float] = []
    triggered_blocks = 0
    executed_steps = 0
    unique_steps = 0
    pairings: set[str] = set()
    for branches in by_block.values():
        triggered_values = {
            bool(branches[branch].get("triggered"))
            for branch in RESPONSE_CONTRAST_BRANCHES
        }
        if len(triggered_values) != 1:
            raise ValueError("A matched block disagrees about whether probing triggered.")
        if triggered_values == {False}:
            comparable_fields = (
                "raw_return",
                "correct_delivery_count",
                "wrong_delivery_count",
                "probe_count",
                "safe_candidate_opportunity_count",
                "indicator_cost",
            )
            for field in comparable_fields:
                present = [
                    branches[branch].get(field)
                    for branch in RESPONSE_CONTRAST_BRANCHES
                    if field in branches[branch]
                ]
                if present and len(set(present)) != 1:
                    raise ValueError(
                        "An untriggered block contains unequal complete trajectories."
                    )
        a1 = _finite(branches["A1"]["raw_return"], "A1.raw_return")
        masked = _finite(branches["A2-mask"]["raw_return"], "A2-mask.raw_return")
        used = _finite(branches["A2-use"]["raw_return"], "A2-use.raw_return")
        response.append(used - masked)
        cost.append(a1 - masked)
        net.append(used - a1)
        metadata = branches["A2-use"]
        triggered_blocks += int(next(iter(triggered_values)))
        executed_steps += _count(
            metadata["executed_environment_steps"], "executed_environment_steps"
        )
        unique_steps += _count(
            metadata["unique_environment_steps"], "unique_environment_steps"
        )
        pairings.add(str(metadata["pairing_id"]))

    delta_response = float(np.mean(np.asarray(response, dtype=np.float64)))
    delta_cost = float(np.mean(np.asarray(cost, dtype=np.float64)))
    delta_net = float(np.mean(np.asarray(net, dtype=np.float64)))
    residual = delta_net - (delta_response - delta_cost)
    if not math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("The paired response-effect identity failed.")
    return {
        "schema_version": RESPONSE_CONTRAST_SUMMARY_SCHEMA_VERSION,
        "not_an_official_sp_xp_summary": True,
        "pairing_count": len(pairings),
        "matched_block_count": len(by_block),
        "raw_row_count": len(rows),
        "triggered_block_count": triggered_blocks,
        "triggered_block_fraction": triggered_blocks / len(by_block),
        "executed_environment_steps": executed_steps,
        "unique_environment_steps": unique_steps,
        "delta_response": delta_response,
        "delta_cost": delta_cost,
        "delta_net": delta_net,
        "identity_residual": residual,
    }


__all__ = [
    "RESPONSE_CONTRAST_BLOCK_COUNT",
    "RESPONSE_CONTRAST_BRANCHES",
    "RESPONSE_CONTRAST_PAIRING_COUNT",
    "RESPONSE_CONTRAST_MANIFEST_SCHEMA_VERSION",
    "RESPONSE_CONTRAST_ROW_COUNT",
    "RESPONSE_CONTRAST_ROWS_SCHEMA_VERSION",
    "RESPONSE_CONTRAST_SUMMARY_SCHEMA_VERSION",
    "ResponseContrastManifest",
    "execute_response_contrast_pairing",
    "response_contrast_episode_seed",
    "response_contrast_pairings",
    "summarize_response_contrast_rows",
]
