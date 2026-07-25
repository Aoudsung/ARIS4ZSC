#!/usr/bin/env python3
"""Run the complete Path C model pipeline on the authorized remote runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.path_c.contracts.config import (
    FAMILY_POOL_SCHEMA_VERSION,
    FORMAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    PathCFormalTemplateV2,
    PathCFormalTemplateV3,
    PathCModelConfig,
)
from src.path_c.contracts.outer_units import load_outer_units_manifest
from src.path_c.pipeline.run import STAGES, run_pipeline

from experiments.overcooked_v2.model_dock.stage_runtime import PathCStageRuntime


def _load_config(
    path: Path,
    *,
    outer_units_manifest: Path | None = None,
    outer_unit: int | None = None,
) -> PathCModelConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Path C model configuration must be a YAML mapping.")
    schema = payload.get("schema_version")
    if schema == SCHEMA_VERSION:
        if outer_units_manifest is not None or outer_unit is not None:
            raise ValueError("Historical path_c_model_v1 configs cannot select an outer unit.")
        config = PathCModelConfig.from_mapping(payload, base_dir=path.parent)
        if config.run_kind != "development":
            raise ValueError(
                "Historical path_c_model_v1 formal configs are audit-only and cannot launch."
            )
        return config
    if schema not in {FORMAL_SCHEMA_VERSION, FAMILY_POOL_SCHEMA_VERSION}:
        raise ValueError("The Path C model schema is not registered.")
    if outer_units_manifest is None or outer_unit is None:
        raise ValueError(
            "Formal Path C configs require --outer-units-manifest and --outer-unit."
        )
    template = (
        PathCFormalTemplateV3.from_mapping(payload, base_dir=path.parent)
        if schema == FAMILY_POOL_SCHEMA_VERSION
        else PathCFormalTemplateV2.from_mapping(payload, base_dir=path.parent)
    )
    manifest = load_outer_units_manifest(outer_units_manifest)
    return template.resolve(manifest, outer_unit_id=outer_unit)


def _seed_from_checkpoint(path: Path) -> int:
    for component in path.parts:
        if component.startswith("seed_") and component[5:].isdigit():
            return int(component[5:])
    raise ValueError(f"Official checkpoint path has no seed directory: {path}")


def _launch_config_for_member(repository_root: Path, member: Any) -> Path:
    seed = _seed_from_checkpoint(member.checkpoint_path)
    prefix = (
        "path_c_official_sp_simple_seed"
        if member.family_id == "official_rnn_sp_ippo_v1"
        else "path_c_official_op_simple_seed"
    )
    return repository_root / "experiments" / "overcooked_v2" / "configs" / f"{prefix}{seed}.yaml"


def _pipeline_dependency_roots(
    repository_root: Path,
    *,
    backbone_launch: Path,
    partner_launches: list[Path],
    layout: str,
    outer_units_manifest: Path | None = None,
) -> tuple[Path, ...]:
    """Return every repository file whose drift can change a pipeline stage."""

    overcooked_root = repository_root / "experiments" / "overcooked_v2"
    roots = (
        Path(__file__).resolve(),
        repository_root / "src" / "path_c",
        overcooked_root / "model_dock",
        overcooked_root / "official" / "overcooked_v2_experiments_adapter.py",
        overcooked_root / "official" / "path_c_family_pool_training.py",
        *(
            (
                overcooked_root
                / "official"
                / "overcooked_v2_experiments_wide_adapter.py",
                overcooked_root
                / "scripts"
                / "run_path_c_official_wide_training.py",
            )
            if layout == "test_time_wide"
            else ()
        ),
        overcooked_root / "path_c_official_artifact.py",
        overcooked_root / "path_c_flax_policy.py",
        overcooked_root / "path_c_official_evidence.py",
        overcooked_root / "path_c_pool_admission.py",
        overcooked_root / "path_c_response_summary.py",
        overcooked_root / "path_c_seed.py",
        overcooked_root / "path_c_standard_training.py",
        backbone_launch,
        *partner_launches,
    )
    return roots if outer_units_manifest is None else (*roots, outer_units_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--outer-units-manifest", type=Path)
    parser.add_argument("--outer-unit", type=int, choices=range(10))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--through", choices=STAGES, default="adaptation")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    manifest_path = (
        None
        if arguments.outer_units_manifest is None
        else arguments.outer_units_manifest.resolve()
    )
    config = _load_config(
        config_path,
        outer_units_manifest=manifest_path,
        outer_unit=arguments.outer_unit,
    )
    repository_root = Path(__file__).resolve().parents[3]
    if config.formal_outer_unit is None:
        backbone_launch = (
            repository_root
            / "experiments"
            / "overcooked_v2"
            / "configs"
            / "path_c_official_sp_simple_seed100.yaml"
        )
        partner_launches = [
            _launch_config_for_member(repository_root, member)
            for member in config.partner_pool
        ]
    else:
        backbone_launch = config.formal_outer_unit.backbone_launch_config
        partner_launches = list(config.formal_outer_unit.partner_launch_configs)
    missing = [path for path in (backbone_launch, *partner_launches) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official launch configuration is missing: "
            + ", ".join(str(path) for path in missing)
        )
    runtime = PathCStageRuntime(
        config,
        backbone_launch_config=backbone_launch,
        partner_launch_configs=partner_launches,
    )
    state = run_pipeline(
        config,
        source_roots=_pipeline_dependency_roots(
            repository_root,
            backbone_launch=backbone_launch,
            partner_launches=partner_launches,
            layout=config.environment.layout,
            outer_units_manifest=manifest_path,
        ),
        stage_runners=runtime.stage_runners(),
        resume=arguments.resume,
        through=arguments.through,
    )
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
