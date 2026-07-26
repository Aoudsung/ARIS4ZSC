"""Two-stage pool-check and training pipeline for Path C model version four."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import VQBCConfig
from .integrity import canonical_sha256, file_sha256


VQBC_PIPELINE_SCHEMA_VERSION = "path_c_vqbc_pipeline_v1"
VQBC_PIPELINE_STAGES = ("pool_check", "training")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_vqbc_pool_check(
    config: VQBCConfig,
    *,
    validate_checkpoint: Callable[[Any, Path], Mapping[str, Any]],
    reference_call_check: Callable[[], Mapping[str, Any]],
) -> Path:
    """Verify the unit-owned reference and every audit-only training source."""

    reference = dict(
        validate_checkpoint(
            config.backbone_init, config.backbone_init.launch_config_path
        )
    )
    if (
        reference.get("training_run_id") != config.backbone_init.training_run_id
        or reference.get("flax_weights_sha256")
        not in {None, config.backbone_init.flax_weights_sha256}
    ):
        raise ValueError("Pool check loaded a reference from another official run.")
    checks = [
        {
            "role": "unit_owned_reference",
            "outer_unit_id": (
                None
                if config.outer_unit is None
                else config.outer_unit.outer_unit_id
            ),
            **reference,
        }
    ]
    for member in config.partner_sampling.members:
        checked = dict(
            validate_checkpoint(
                member.checkpoint,
                member.checkpoint.launch_config_path,
            )
        )
        if (
            checked.get("training_run_id")
            != member.checkpoint.training_run_id
            or checked.get("flax_weights_sha256")
            not in {None, member.checkpoint.flax_weights_sha256}
        ):
            raise ValueError("Pool check loaded a different historical partner.")
        checks.append(
            {
                "role": "training_partner",
                "member_id": member.member_id,
                "family_id": member.family_id,
                "training_seed": member.training_seed,
                "checkpoint_index": member.checkpoint_index,
                "update_step": member.update_step,
                "effective_environment_steps": (
                    member.effective_environment_steps
                ),
                **checked,
            }
        )
    reference_call = dict(reference_call_check())
    reference_output_sha256 = reference_call.get("reference_output_sha256")
    if (
        reference_call.get("reference_training_run_id")
        != config.backbone_init.training_run_id
        or reference_call.get("reference_weights_sha256")
        != config.backbone_init.flax_weights_sha256
        or reference_call.get("carry_exact") is not True
        or not isinstance(reference_output_sha256, str)
        or len(reference_output_sha256) != 64
        or not set(reference_output_sha256).issubset(
            frozenset("0123456789abcdef")
        )
    ):
        raise ValueError("The live reference call did not use this unit's checkpoint.")
    payload = {
        "schema_version": "path_c_vqbc_pool_check_v1",
        "config_sha256": config.config_sha256,
        "run_kind": config.run_kind,
        "scientific_readout_allowed": False,
        "reference_training_run_id": config.backbone_init.training_run_id,
        "partner_schedule": config.partner_sampling.schedule,
        "include_frozen_current_policy": (
            config.partner_sampling.include_frozen_current_policy
        ),
        "historical_partner_count": len(config.partner_sampling.members),
        "reference_call": reference_call,
        "checks": checks,
    }
    target = config.output_root / "pool_check" / "report.json"
    _atomic_json(target, payload)
    return target


def run_vqbc_pipeline(
    config: VQBCConfig,
    *,
    pool_check: Callable[[VQBCConfig], Path],
    training: Callable[[VQBCConfig], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Execute exactly pool_check then training and bind both artifacts."""

    state_path = config.output_root / "pipeline_state.json"
    pool_path = Path(pool_check(config)).resolve()
    if not pool_path.is_file():
        raise FileNotFoundError("The fourth-model pool check produced no report.")
    pool_record = {
        "path": str(pool_path),
        "sha256": file_sha256(pool_path),
    }
    training_artifacts = dict(training(config))
    normalized_training = {}
    for name, raw_path in training_artifacts.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training artifact is missing: {path}")
        normalized_training[str(name)] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    if not normalized_training:
        raise ValueError("The fourth-model training stage exposed no artifact.")
    payload = {
        "schema_version": VQBC_PIPELINE_SCHEMA_VERSION,
        "config_sha256": config.config_sha256,
        "run_kind": config.run_kind,
        "scientific_readout_allowed": False,
        "stages": {
            "pool_check": pool_record,
            "training": normalized_training,
        },
    }
    payload["pipeline_content_sha256"] = canonical_sha256(payload)
    _atomic_json(state_path, payload)
    return payload


__all__ = [
    "VQBC_PIPELINE_SCHEMA_VERSION",
    "VQBC_PIPELINE_STAGES",
    "run_vqbc_pipeline",
    "run_vqbc_pool_check",
]
