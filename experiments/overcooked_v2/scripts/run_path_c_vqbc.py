#!/usr/bin/env python3
"""Run the explicitly authorized fourth-model pool check and training stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from experiments.overcooked_v2.model_dock.official_dock import (
    validate_checkpoint_reference,
)
from experiments.overcooked_v2.model_dock.vqbc_runtime import (
    build_vqbc_wiring,
    run_vqbc_training,
)
from src.path_c.contracts.outer_units import load_outer_units_manifest
from src.path_c.vqbc.config import (
    VQBCConfig,
    VQBCFormalTemplate,
)
from src.path_c.vqbc.pipeline import (
    run_vqbc_pipeline,
    run_vqbc_pool_check,
)


def _load_config(
    path: Path, *, outer_units: Path | None, outer_unit_id: int | None
) -> VQBCConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The fourth-model YAML root must be a mapping.")
    if "backbone_init" in payload:
        if outer_units is not None or outer_unit_id is not None:
            raise ValueError("Resolved development configs do not accept outer-unit flags.")
        return VQBCConfig.from_mapping(payload, base_dir=path.parent)
    if outer_units is None or outer_unit_id is None:
        raise ValueError("Formal templates require --outer-units and --outer-unit.")
    template = VQBCFormalTemplate.from_mapping(payload, base_dir=path.parent)
    manifest = load_outer_units_manifest(outer_units)
    return template.resolve(manifest, outer_unit_id=outer_unit_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-units", type=Path)
    parser.add_argument("--outer-unit", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    config = _load_config(
        args.config.resolve(),
        outer_units=(
            None if args.outer_units is None else args.outer_units.resolve()
        ),
        outer_unit_id=args.outer_unit,
    )

    def pool_check(selected: VQBCConfig) -> Path:
        wiring = build_vqbc_wiring(selected)
        return run_vqbc_pool_check(
            selected,
            validate_checkpoint=lambda reference, launch: (
                validate_checkpoint_reference(
                    reference, launch_config_path=launch
                )
            ),
            reference_call_check=lambda: wiring.reference_parity,
        )

    run_vqbc_pipeline(
        config,
        pool_check=pool_check,
        training=lambda selected: run_vqbc_training(
            selected, resume_checkpoint=args.resume_checkpoint
        ),
    )


if __name__ == "__main__":
    main()
