"""Fail-closed public training entry for unified DELTA-ZSC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.overcooked_v2.unified_training_app import run_training as _run_core
from src.delta_zsc.config import load_config
from src.delta_zsc.manifest import validate_unified_manifest
from src.delta_zsc.model import (
    BASE_VARIANT,
    JOINT_VARIANT,
    RESPONSE_ONLY_VARIANT,
)
from src.path_c.experiment import ENGINEERING_SEED_INDEX, load_partner_manifest


TRAINABLE_VARIANTS = (
    BASE_VARIANT,
    RESPONSE_ONLY_VARIANT,
    JOINT_VARIANT,
)


def _validate_run_contract(args: Any) -> None:
    config = load_config(args.config)
    variant = str(args.variant).lower()
    run_kind = str(args.run_kind)
    seed = int(args.seed_index)
    if variant not in TRAINABLE_VARIANTS:
        raise ValueError(
            "Only base, response_only, and joint are trainable. Full is the "
            "joint deployment plus the analytic VOI operator."
        )
    if run_kind not in {"mechanical", "development", "formal"}:
        raise ValueError("Unknown unified DELTA run kind.")
    if run_kind == "mechanical":
        if seed != ENGINEERING_SEED_INDEX:
            raise ValueError("Mechanical unified training uses seed index -1.")
    elif seed not in range(10):
        raise ValueError("Development/formal unified seeds lie in 0..9.")
    if run_kind == "formal":
        if bool(args.skip_manifest_hash_check):
            raise ValueError("Formal training cannot skip partner checkpoint hashes.")
        expected = {
            "latent_components": 4,
            "continuation_horizon": 128,
            "adaptation_kl_budget": 0.04,
            "num_envs": 256,
            "rollout_length": 256,
            "total_environment_steps": 29_949_952,
            "ppo_minibatches": 64,
            "ppo_update_epochs": 4,
            "evaluation_episodes": 500,
        }
        observed = {
            "latent_components": config.method.latent_components,
            "continuation_horizon": config.method.continuation_horizon,
            "adaptation_kl_budget": config.method.adaptation_kl_budget,
            "num_envs": config.data.num_envs,
            "rollout_length": config.data.rollout_length,
            "total_environment_steps": config.data.total_environment_steps,
            "ppo_minibatches": config.ppo.minibatches,
            "ppo_update_epochs": config.ppo.update_epochs,
            "evaluation_episodes": config.evaluation.episodes_per_pairing,
        }
        if observed != expected:
            raise ValueError(
                f"Formal unified DELTA config differs: observed={observed}, "
                f"expected={expected}."
            )
    manifest = load_partner_manifest(
        Path(args.partner_manifest).resolve(),
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    validate_unified_manifest(
        manifest,
        formal=(run_kind == "formal"),
        require_support=True,
    )


def run_training(args: Any) -> None:
    _validate_run_contract(args)
    _run_core(args)


__all__ = ["TRAINABLE_VARIANTS", "run_training"]
