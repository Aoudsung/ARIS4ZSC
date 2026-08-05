"""Executable paired matrices for unified DELTA-ZSC.

The matrix is intentionally small and follows the natural model nesting:
base, response-only, joint, and full-as-joint-plus-VOI.  It mechanically checks
that base parameters are identical across trainable variants for each seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

from experiments.overcooked_v2.unified_evaluation_app import run_evaluation
from experiments.overcooked_v2.unified_training_app import run_training
from src.delta_zsc.config import CONFIG_VERSION, METHOD_VERSION, load_config
from src.delta_zsc.model import (
    BASE_VARIANT,
    FULL_VARIANT,
    JOINT_VARIANT,
    RESPONSE_ONLY_VARIANT,
)
from src.path_c.storage import runtime_provenance, sha256_path


TRAINING_VARIANTS = (
    BASE_VARIANT,
    RESPONSE_ONLY_VARIANT,
    JOINT_VARIANT,
)
EVALUATION_VARIANTS = (
    BASE_VARIANT,
    RESPONSE_ONLY_VARIANT,
    JOINT_VARIANT,
    FULL_VARIANT,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _bundle(run_directory: Path) -> Mapping[str, Any]:
    path = run_directory / "final_deployment" / "deployment_bundle.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != METHOD_VERSION:
        raise ValueError("Matrix deployment is not unified DELTA-ZSC.")
    return payload


def _validate_seeds(seeds: tuple[int, ...], *, run_kind: str) -> None:
    if run_kind == "mechanical":
        if seeds != (-1,):
            raise ValueError("Mechanical matrix uses only engineering seed -1.")
        return
    if len(seeds) != len(set(seeds)) or any(seed not in range(10) for seed in seeds):
        raise ValueError("Matrix seed indexes must be unique values in 0..9.")
    if run_kind == "formal" and tuple(sorted(seeds)) != tuple(range(10)):
        raise ValueError("Formal matrix requires seed indexes 0..9 exactly once.")
    if run_kind == "development" and len(seeds) not in {5, 10}:
        raise ValueError("Development matrix requires five or ten paired seeds.")


def _training_args(
    *,
    config: Path,
    manifest: Path,
    variant: str,
    seed: int,
    run_kind: str,
    output: Path,
    resume: bool,
    skip_hash: bool,
    require_cuda: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(config),
        partner_manifest=str(manifest),
        variant=variant,
        seed_index=seed,
        run_kind=run_kind,
        output=str(output),
        resume=resume,
        require_cuda=require_cuda,
        skip_manifest_hash_check=skip_hash,
    )


def _evaluation_args(
    *,
    deployment: Path,
    manifest: Path,
    variant_override: str | None,
    partner_role: str,
    episodes: int | None,
    evaluation_seed: int,
    output: Path,
    skip_hash: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        deployment=str(deployment),
        partner_manifest=str(manifest),
        partner_role=partner_role,
        variant_override=variant_override,
        episodes=episodes,
        evaluation_seed=evaluation_seed,
        output=str(output),
        skip_manifest_hash_check=skip_hash,
    )


def run_matrix(args: Any) -> None:
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.partner_manifest).resolve()
    config = load_config(config_path)
    run_kind = str(args.run_kind)
    seeds = tuple(int(value) for value in args.seed_index)
    _validate_seeds(seeds, run_kind=run_kind)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    partner_role = str(getattr(args, "partner_role", "confirmatory"))
    episodes = getattr(args, "episodes", None)
    entries = []
    bundles: dict[tuple[str, int], Mapping[str, Any]] = {}

    for variant in TRAINING_VARIANTS:
        for seed in seeds:
            run_directory = output / "training" / variant / f"seed-{seed}"
            run_training(
                _training_args(
                    config=config_path,
                    manifest=manifest_path,
                    variant=variant,
                    seed=seed,
                    run_kind=run_kind,
                    output=run_directory,
                    resume=bool(args.resume),
                    skip_hash=bool(args.skip_manifest_hash_check),
                    require_cuda=bool(getattr(args, "require_cuda", False)),
                )
            )
            bundle = _bundle(run_directory)
            bundles[(variant, seed)] = bundle
            entries.append(
                {
                    "stage": "training",
                    "variant": variant,
                    "seed_index": seed,
                    "path": str(run_directory),
                    "sha256": sha256_path(run_directory),
                    "base_parameter_fingerprint": bundle[
                        "base_parameter_fingerprint"
                    ],
                    "latent_parameter_fingerprint": bundle[
                        "latent_parameter_fingerprint"
                    ],
                    "deployment_parameter_fingerprint": bundle[
                        "parameter_fingerprint"
                    ],
                }
            )

    for seed in seeds:
        fingerprints = {
            bundles[(variant, seed)]["base_parameter_fingerprint"]
            for variant in TRAINING_VARIANTS
        }
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"Base parameter identity differs across variants for seed {seed}."
            )

    for variant in EVALUATION_VARIANTS:
        for seed in seeds:
            training_variant = JOINT_VARIANT if variant == FULL_VARIANT else variant
            deployment = (
                output
                / "training"
                / training_variant
                / f"seed-{seed}"
                / "final_deployment"
            )
            evaluation = output / "evaluation" / variant / f"seed-{seed}"
            run_evaluation(
                _evaluation_args(
                    deployment=deployment,
                    manifest=manifest_path,
                    variant_override=(
                        FULL_VARIANT if variant == FULL_VARIANT else None
                    ),
                    partner_role=partner_role,
                    episodes=(None if episodes is None else int(episodes)),
                    evaluation_seed=int(getattr(args, "evaluation_seed", 0)),
                    output=evaluation,
                    skip_hash=bool(args.skip_manifest_hash_check),
                )
            )
            entries.append(
                {
                    "stage": "evaluation",
                    "variant": variant,
                    "seed_index": seed,
                    "path": str(evaluation),
                    "sha256": sha256_path(evaluation),
                    "source_training_variant": training_variant,
                    "deployment_parameter_fingerprint": bundles[
                        (training_variant, seed)
                    ]["parameter_fingerprint"],
                }
            )

    _write_json(
        output / "matrix.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_execution_matrix",
            "method": METHOD_VERSION,
            "run_kind": run_kind,
            "layout": config.environment.layout,
            "config": {
                "path": str(config_path),
                "sha256": sha256_path(config_path),
                "fingerprint": config.fingerprint,
            },
            "partner_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_path(manifest_path),
            },
            "paired_seed_indexes": list(seeds),
            "training_variants": list(TRAINING_VARIANTS),
            "evaluation_variants": list(EVALUATION_VARIANTS),
            "full_reuses_joint_parameters": True,
            "base_parameter_fingerprint_match": True,
            "entries": entries,
            "repository_runtime": runtime_provenance(),
        },
    )


def _variant_config(source: Path, target: Path, component_count: int) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("K-sensitivity source config must be a mapping.")
    payload = dict(payload)
    payload["version"] = CONFIG_VERSION
    payload["method"] = dict(payload["method"])
    payload["method"]["latent_components"] = int(component_count)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    load_config(target)
    return target


def run_k_sensitivity(args: Any) -> None:
    seeds = tuple(int(value) for value in args.seed_index)
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(seed not in range(10) for seed in seeds):
        raise ValueError("K sensitivity requires five distinct development seeds.")
    source = Path(args.config).resolve()
    manifest = Path(args.partner_manifest).resolve()
    output = Path(args.output).resolve()
    descriptors = []
    for component_count in (2, 4, 8):
        config_path = _variant_config(
            source,
            output / "configs" / f"k-{component_count}.yaml",
            component_count,
        )
        config = load_config(config_path)
        root = output / f"k-{component_count}"
        entries = []
        for seed in seeds:
            run = root / "training" / JOINT_VARIANT / f"seed-{seed}"
            run_training(
                _training_args(
                    config=config_path,
                    manifest=manifest,
                    variant=JOINT_VARIANT,
                    seed=seed,
                    run_kind="development",
                    output=run,
                    resume=bool(args.resume),
                    skip_hash=bool(args.skip_manifest_hash_check),
                    require_cuda=bool(getattr(args, "require_cuda", False)),
                )
            )
            bundle = _bundle(run)
            entries.append(
                {
                    "stage": "training",
                    "seed_index": seed,
                    "path": str(run),
                    "sha256": sha256_path(run),
                    "base_parameter_fingerprint": bundle[
                        "base_parameter_fingerprint"
                    ],
                    "latent_parameter_fingerprint": bundle[
                        "latent_parameter_fingerprint"
                    ],
                }
            )
            for evaluation_variant in (JOINT_VARIANT, FULL_VARIANT):
                evaluation = (
                    root
                    / "evaluation"
                    / evaluation_variant
                    / f"seed-{seed}"
                )
                run_evaluation(
                    _evaluation_args(
                        deployment=run / "final_deployment",
                        manifest=manifest,
                        variant_override=(
                            FULL_VARIANT
                            if evaluation_variant == FULL_VARIANT
                            else None
                        ),
                        partner_role=str(
                            getattr(args, "partner_role", "confirmatory")
                        ),
                        episodes=getattr(args, "episodes", None),
                        evaluation_seed=int(
                            getattr(args, "evaluation_seed", 0)
                        ),
                        output=evaluation,
                        skip_hash=bool(args.skip_manifest_hash_check),
                    )
                )
                entries.append(
                    {
                        "stage": "evaluation",
                        "variant": evaluation_variant,
                        "seed_index": seed,
                        "path": str(evaluation),
                        "sha256": sha256_path(evaluation),
                        "deployment_parameter_fingerprint": bundle[
                            "parameter_fingerprint"
                        ],
                    }
                )
        matrix_path = root / "matrix.json"
        _write_json(
            matrix_path,
            {
                "version": 1,
                "artifact_type": "unified_delta_k_cell",
                "development_only": True,
                "method": METHOD_VERSION,
                "layout": config.environment.layout,
                "latent_components": component_count,
                "paired_seed_indexes": list(seeds),
                "training_variants": [JOINT_VARIANT],
                "evaluation_variants": [JOINT_VARIANT, FULL_VARIANT],
                "entries": entries,
            },
        )
        descriptors.append(
            {
                "latent_components": component_count,
                "path": str(matrix_path),
                "sha256": sha256_path(matrix_path),
            }
        )
    _write_json(
        output / "k_sensitivity.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_k_sensitivity",
            "development_only": True,
            "method": METHOD_VERSION,
            "paired_seed_indexes": list(seeds),
            "component_counts": [2, 4, 8],
            "matrices": descriptors,
            "repository_runtime": runtime_provenance(),
        },
    )


__all__ = [
    "EVALUATION_VARIANTS",
    "TRAINING_VARIANTS",
    "run_k_sensitivity",
    "run_matrix",
]
