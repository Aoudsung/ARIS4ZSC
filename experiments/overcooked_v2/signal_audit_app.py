"""Post-training, read-only DEPI mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.path_c.experiment import METHOD_VERSION
from src.path_c.storage import read_run_identity, write_json


def _nested(rows: list[Mapping[str, Any]], *path: str) -> np.ndarray:
    values = []
    for row in rows:
        current: Any = row
        for name in path:
            if not isinstance(current, Mapping) or name not in current:
                current = None
                break
            current = current[name]
        if isinstance(current, (int, float)) and np.isfinite(current):
            values.append(float(current))
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray) -> Mapping[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "final": None, "minimum": None, "maximum": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "final": float(values[-1]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def run_signal_audit(args: argparse.Namespace) -> None:
    source = Path(args.training_run).resolve()
    identity = read_run_identity(source)
    if identity.get("method") != METHOD_VERSION or identity.get("stage") != "train":
        raise ValueError("Signal audit accepts only active DEPI training runs.")
    records = []
    for path in sorted((source / "records" / "metrics").glob("update_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError("Training run has no per-update DEPI metrics.")
    report = {
        "method": METHOD_VERSION,
        "training_run": str(source),
        "read_only": True,
        "affects_training": False,
        "affects_checkpoint_selection": False,
        "affects_deployment": False,
        "mechanism_readouts": {
            "task_and_policy": {
                "raw_reward": _summary(_nested(records, "ppo", "mean_raw_reward")),
                "policy_entropy": _summary(_nested(records, "ppo", "entropy")),
                "post_update_policy_kl": _summary(
                    _nested(records, "ppo", "combined_policy_kl")
                ),
            },
            "exact_protocol_filter": {
                "posterior_entropy": _summary(
                    _nested(records, "ppo", "mean_posterior_entropy")
                ),
                "response_joint_nll": _summary(
                    _nested(records, "ppo", "response_total_loss")
                ),
                "capability_consistency": _summary(
                    _nested(records, "ppo", "capability_consistency_loss")
                ),
            },
            "decision_supervision": {
                "signature_loss": _summary(
                    _nested(records, "ppo", "signature_loss")
                ),
                "actor_decision_kl": _summary(
                    _nested(records, "ppo", "decision_policy_loss")
                ),
                "anchor_effective_sample_size": _summary(
                    _nested(records, "ppo", "anchor_effective_sample_size")
                ),
                "auxiliary_gradient_norm": _summary(
                    _nested(records, "ppo", "auxiliary_gradient_norm")
                ),
            },
            "matched_pair_separation": {
                "loss": _summary(_nested(records, "ppo", "separation_loss")),
                "ambiguity_fraction": _summary(
                    _nested(
                        records,
                        "anchor_supervision",
                        "readings",
                        "irreducible_ambiguity_fraction",
                    )
                ),
            },
            "m1_independent_value_diagnostic": {
                "evaluations": int(
                    sum(bool(row.get("m1_gate", {}).get("evaluated", True)) for row in records)
                ),
                "latest": records[-1].get("m1_gate", {}),
            },
        },
        "note": (
            "These training diagnostics do not establish mechanism attribution. "
            "Use evaluate-identifiability and evaluate-recoverable-value for the "
            "registered held-out causal controls."
        ),
    }
    output = Path(args.output).resolve()
    write_json(output / "signal_audit.json", report)
    print(f"Complete read-only DEPI signal audit: {output}")


__all__ = ["run_signal_audit"]
