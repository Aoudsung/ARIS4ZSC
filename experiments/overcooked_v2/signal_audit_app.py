"""Post-training, read-only V6 mechanism report."""

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
        raise ValueError("Signal audit accepts only V6 training runs.")
    records = []
    for path in sorted((source / "records" / "metrics").glob("update_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError("Training run has no per-update V6 metrics.")
    report = {
        "method": METHOD_VERSION,
        "training_run": str(source),
        "read_only": True,
        "affects_training": False,
        "affects_checkpoint_selection": False,
        "affects_deployment": False,
        "mechanism_readouts": {
            "C0_task_competence_context": {
                "raw_reward": _summary(_nested(records, "ppo", "mean_raw_reward")),
                "policy_entropy": _summary(_nested(records, "ppo", "entropy")),
            },
            "C1_partner_decision_heterogeneity": {
                "anchor_action_range": _summary(
                    _nested(records, "counterfactual", "anchor_empirical_action_range")
                ),
            },
            "C2_raw_q_quality": {
                "retrace_loss": _summary(_nested(records, "raw_q", "raw_q_retrace_loss")),
                "action_range": _summary(_nested(records, "raw_q", "raw_q_action_range_mean")),
                "head_disagreement": _summary(
                    _nested(records, "raw_q", "raw_q_head_disagreement")
                ),
            },
            "C3_online_belief_recovery": {
                "posterior_uncertainty": _summary(
                    _nested(records, "ppo", "mean_belief_uncertainty")
                ),
                "information_bottleneck": _summary(
                    _nested(records, "belief_objective_losses", "information_bottleneck")
                ),
            },
            "C4_belief_conditioned_control": {
                "q_policy_weight": _summary(
                    _nested(records, "ppo", "q_policy_weight_mean")
                ),
                "context_robustness_kl": _summary(
                    _nested(records, "ppo", "robust_generalist_kl")
                ),
            },
            "C5_information_value": {
                "decision_regret": _summary(
                    _nested(records, "regret", "mean_decision_regret")
                ),
                "regret_shaping": _summary(
                    _nested(records, "regret", "mean_decision_regret_shaping")
                ),
            },
        },
        "note": (
            "C0-C5 labels are retrospective mechanism readouts only. They did not "
            "enable, disable, select, replace, or export any policy."
        ),
    }
    write_json(Path(args.output).resolve() / "signal_audit.json", report)
    print(f"Complete read-only V6 signal audit: {Path(args.output).resolve()}")


__all__ = ["run_signal_audit"]
