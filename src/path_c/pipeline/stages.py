"""Reusable static checks and host artifact helpers for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .run import file_sha256


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_pool_check(
    context: Any,
    *,
    validate_checkpoint: Callable[[Any], Mapping[str, Any]],
) -> Mapping[str, Path]:
    """Verify the backbone and four partners through the injected official dock."""

    config = context.config
    checks = [
        {"role": "backbone", **dict(validate_checkpoint(config.backbone_init))}
    ]
    checks.extend(
        {
            "role": "partner",
            "family_id": member.family_id,
            **dict(validate_checkpoint(member)),
        }
        for member in config.partner_pool
    )
    report = {
        "schema_version": "path_c_partner_pool_check_v1",
        "run_kind": config.run_kind,
        "scientific_readout_allowed": config.scientific_readout_allowed,
        "partner_count": len(config.partner_pool),
        "checks": checks,
    }
    target = context.stage_directory / "pool_check.json"
    _atomic_json(target, report)
    return {"pool_check": target}


def write_stage_summary(
    context: Any,
    *,
    stage: str,
    effective_environment_steps: int,
    completed_episodes: int,
    extra: Mapping[str, Any],
) -> Path:
    """Write effective counts read back by a completed stage implementation."""

    if stage != context.stage:
        raise ValueError("Stage summary name does not match its pipeline context.")
    payload = {
        "schema_version": "path_c_stage_summary_v1",
        "stage": stage,
        "run_kind": context.config.run_kind,
        "scientific_readout_allowed": context.config.scientific_readout_allowed,
        "effective_environment_steps": int(effective_environment_steps),
        "completed_episodes": int(completed_episodes),
        **dict(extra),
    }
    target = context.stage_directory / "summary.json"
    _atomic_json(target, payload)
    return target
