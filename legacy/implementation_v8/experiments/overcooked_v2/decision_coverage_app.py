"""Development-only decision-signature support coverage artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.path_c.decision_geometry import centered_action_values
from src.path_c.storage import ensure_run_identity, runtime_provenance, sha256_path, write_json


DECISION_COVERAGE_SOURCE_VERSION = 1
DECISION_COVERAGE_REPORT_VERSION = 1
_SOURCE_ROLES = {"training_support", "development_coverage"}


def _history_rows(payload: Mapping[str, Any], partition: str) -> Sequence[Mapping[str, Any]]:
    key = {"fit": "fit_histories", "validation": "validation_histories"}[partition]
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Decision-coverage source partition is empty.")
    required = {
        "fit_returns_by_action",
        "partner_run_id",
        "partner_mechanism",
        "task_features",
        "episode_time",
        "recipe_order_state",
        "ego_role",
        "task_state_hash",
    }
    if any(not isinstance(row, Mapping) or not required <= set(row) for row in rows):
        raise ValueError("Decision-coverage history rows lack registered fields.")
    return rows


def build_decision_coverage_source(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "depi_pair_comparator_source":
        raise ValueError("Coverage sources must originate in the real continuation collector.")
    partition = str(args.partition)
    role = str(args.role)
    if role not in _SOURCE_ROLES:
        raise ValueError("Decision-coverage source role is not registered.")
    rows = _history_rows(payload, partition)
    parent_hash = str(payload.get("partner_manifest", {}).get("sha256", ""))
    if len(parent_hash) != 64:
        raise ValueError("Decision-coverage source has no partner-manifest lineage hash.")
    normalized = [
        {
            "fit_returns_by_action": list(row["fit_returns_by_action"]),
            "partner_run_id": str(row["partner_run_id"]),
            "partner_family": str(row["partner_mechanism"]),
            "task_features": list(row["task_features"]),
            "episode_time": int(row["episode_time"]),
            "recipe_order_state": str(row["recipe_order_state"]),
            "ego_role": int(row["ego_role"]),
            "task_state_hash": str(row["task_state_hash"]),
        }
        for row in rows
    ]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "version": DECISION_COVERAGE_SOURCE_VERSION,
        "artifact_type": "depi_decision_coverage_source",
        "development_only": True,
        "confirmatory_partner_use_forbidden": True,
        "role": role,
        "layout": str(payload["layout"]),
        "gamma": float(payload["continuation_contract"]["gamma"]),
        "continuation_horizon": int(payload["continuation_contract"]["horizon"]),
        "partner_manifest_sha256": parent_hash,
        "source": {"path": str(source), "sha256": sha256_path(source)},
        "rows": normalized,
    }
    write_json(output, result)


def _coverage_source(path: str | Path, *, expected_role: str) -> Mapping[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "version",
        "artifact_type",
        "development_only",
        "confirmatory_partner_use_forbidden",
        "role",
        "layout",
        "gamma",
        "continuation_horizon",
        "partner_manifest_sha256",
        "source",
        "rows",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["version"] != DECISION_COVERAGE_SOURCE_VERSION
        or payload["artifact_type"] != "depi_decision_coverage_source"
        or payload["development_only"] is not True
        or payload["confirmatory_partner_use_forbidden"] is not True
        or payload["role"] != expected_role
    ):
        raise ValueError("Decision-coverage source contract differs.")
    descriptor = payload["source"]
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256"}
        or sha256_path(descriptor["path"]) != descriptor["sha256"]
    ):
        raise ValueError("Decision-coverage source provenance differs.")
    return payload


def _row_arrays(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    signatures = centered_action_values(
        np.asarray([row["fit_returns_by_action"] for row in rows], dtype=np.float64)
    )
    tasks = np.asarray([row["task_features"] for row in rows], dtype=np.float64)
    if signatures.ndim != 2 or signatures.shape[1] != 6 or tasks.ndim != 2:
        raise ValueError("Decision-coverage source arrays differ.")
    if tasks.shape[0] != signatures.shape[0] or not np.all(
        np.isfinite(signatures)
    ) or not np.all(np.isfinite(tasks)):
        raise ValueError("Decision-coverage source arrays are invalid.")
    return {
        "signatures": signatures,
        "tasks": tasks,
        "runs": np.asarray([str(row["partner_run_id"]) for row in rows]),
        "families": np.asarray([str(row["partner_family"]) for row in rows]),
        "times": np.asarray([int(row["episode_time"]) for row in rows]),
        "regimes": np.asarray([str(row["recipe_order_state"]) for row in rows]),
        "roles": np.asarray([int(row["ego_role"]) for row in rows]),
    }


def _coverage_metrics(
    training: Mapping[str, Any],
    coverage: Mapping[str, Any],
    *,
    task_epsilon: float,
    signature_threshold: float,
) -> Mapping[str, Any]:
    train_signature = training["signatures"]
    cover_signature = coverage["signatures"]
    signature_distance = np.sqrt(
        np.mean(
            np.square(cover_signature[:, None, :] - train_signature[None, :, :]),
            axis=-1,
        )
    )
    task_distance = np.sqrt(
        np.mean(
            np.square(coverage["tasks"][:, None, :] - training["tasks"][None, :, :]),
            axis=-1,
        )
    )
    controlled = (
        (task_distance <= float(task_epsilon))
        & (coverage["roles"][:, None] == training["roles"][None, :])
        & (coverage["regimes"][:, None] == training["regimes"][None, :])
        & (coverage["times"][:, None] == training["times"][None, :])
    )
    nearest = np.min(signature_distance, axis=1)
    controlled_distance = np.where(controlled, signature_distance, np.inf)
    controlled_nearest = np.min(controlled_distance, axis=1)
    controlled_matched = np.isfinite(controlled_nearest)
    nearest_index = np.argmin(signature_distance, axis=1)
    top_action_match = (
        np.argmax(cover_signature, axis=-1)
        == np.argmax(train_signature[nearest_index], axis=-1)
    )

    def summarize(mask: np.ndarray) -> Mapping[str, Any]:
        values = nearest[mask]
        conditional = controlled_nearest[mask]
        conditional_valid = np.isfinite(conditional)
        return {
            "history_count": int(np.sum(mask)),
            "signature_coverage": float(
                np.mean(values <= float(signature_threshold))
            ),
            "nearest_support_distance_mean": float(np.mean(values)),
            "nearest_support_distance_q90": float(np.quantile(values, 0.90)),
            "top_action_coverage": float(np.mean(top_action_match[mask])),
            "state_conditional_match_fraction": float(np.mean(conditional_valid)),
            "state_conditional_coverage": (
                None
                if not np.any(conditional_valid)
                else float(
                    np.mean(
                        conditional[conditional_valid]
                        <= float(signature_threshold)
                    )
                )
            ),
        }

    overall_mask = np.ones((cover_signature.shape[0],), dtype=bool)
    families = {
        family: summarize(coverage["families"] == family)
        for family in sorted(np.unique(coverage["families"]).tolist())
    }
    return {
        "overall": summarize(overall_mask),
        "by_partner_family": families,
        "maximum_family_coverage_gap": float(
            max(value["signature_coverage"] for value in families.values())
            - min(value["signature_coverage"] for value in families.values())
        ),
        "unmatched_state_conditional_fraction": float(
            1.0 - np.mean(controlled_matched)
        ),
    }


def summarize_decision_coverage(args: argparse.Namespace) -> None:
    training_path = Path(args.training_source).resolve()
    coverage_path = Path(args.coverage_source).resolve()
    training_payload = _coverage_source(
        training_path, expected_role="training_support"
    )
    coverage_payload = _coverage_source(
        coverage_path, expected_role="development_coverage"
    )
    if (
        training_payload["layout"] != coverage_payload["layout"]
        or training_payload["gamma"] != coverage_payload["gamma"]
        or training_payload["continuation_horizon"]
        != coverage_payload["continuation_horizon"]
        or training_payload["partner_manifest_sha256"]
        == coverage_payload["partner_manifest_sha256"]
    ):
        raise ValueError("Decision-coverage contracts or development lineages differ.")
    training = _row_arrays(training_payload["rows"])
    coverage = _row_arrays(coverage_payload["rows"])
    if set(training["runs"].tolist()) & set(coverage["runs"].tolist()):
        raise ValueError("Training support and development coverage run IDs overlap.")
    task_epsilon = float(args.task_epsilon)
    signature_threshold = float(args.signature_threshold)
    if task_epsilon <= 0.0 or signature_threshold <= 0.0:
        raise ValueError("Decision-coverage thresholds must be positive.")
    metrics = _coverage_metrics(
        training,
        coverage,
        task_epsilon=task_epsilon,
        signature_threshold=signature_threshold,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "training_support": {
            "path": str(training_path),
            "sha256": sha256_path(training_path),
        },
        "development_coverage": {
            "path": str(coverage_path),
            "sha256": sha256_path(coverage_path),
        },
    }
    ensure_run_identity(
        output,
        {
            "stage": "summarize-decision-coverage",
            "repository_runtime": runtime_provenance(),
            "sources": sources,
            "development_only": True,
        },
    )
    write_json(
        output / "decision_coverage_report.json",
        {
            "version": DECISION_COVERAGE_REPORT_VERSION,
            "artifact_type": "depi_decision_signature_coverage",
            "development_only": True,
            "confirmatory_partner_use_forbidden": True,
            "layout": training_payload["layout"],
            "gamma": training_payload["gamma"],
            "continuation_horizon": training_payload["continuation_horizon"],
            "task_match_epsilon": task_epsilon,
            "signature_coverage_threshold": signature_threshold,
            "sources": sources,
            **metrics,
        },
    )


__all__ = [
    "build_decision_coverage_source",
    "summarize_decision_coverage",
]
