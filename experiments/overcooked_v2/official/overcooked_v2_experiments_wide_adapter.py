"""Test Time Wide binding for the frozen official OvercookedV2 trainer.

The Test Time Simple adapter is kept byte-for-byte stable because completed
official checkpoints bind its source hash.  This module reuses that audited
implementation while changing only the registered layout and its Hydra
environment source.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from experiments.overcooked_v2 import path_c_official_artifact as artifact
from experiments.overcooked_v2.official import (
    overcooked_v2_experiments_adapter as base,
)


WIDE_LAYOUT = "test_time_wide"
_BASE_DEPENDENCIES = base.official_source_dependency_records
_BASE_REFERENCE_COMPARISON = base.reference_configuration_comparison
_BASE_VALIDATE_ACCEPTANCE = base.validate_official_reference_acceptance_report


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def validate_official_wide_launch_config(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply the frozen official launch contract to Test Time Wide."""

    if payload.get("layout") != WIDE_LAYOUT:
        raise ValueError("The Wide official adapter requires test_time_wide.")
    environment = payload.get("environment_kwargs")
    if not isinstance(environment, Mapping) or environment.get("layout") != WIDE_LAYOUT:
        raise ValueError("The Wide launch environment kwargs name another layout.")
    if Path(str(payload.get("environment_config", ""))).name != (
        "ocv2_test_time_wide_standard.yaml"
    ):
        raise ValueError("The Wide launch names the wrong environment config.")
    wiring = payload.get("wiring_verification")
    overrides = () if not isinstance(wiring, Mapping) else wiring.get(
        "hydra_overrides", ()
    )
    if "+env=test_time_wide" not in overrides:
        raise ValueError("The Wide launch did not register the Wide Hydra environment.")

    translated = copy.deepcopy(dict(payload))
    translated["layout"] = "test_time_simple"
    translated["environment_kwargs"]["layout"] = "test_time_simple"
    artifact.validate_official_launch_config(translated)
    return payload


def load_official_wide_launch_config(path: str | Path) -> Mapping[str, Any]:
    return validate_official_wide_launch_config(
        _load_mapping(Path(path).resolve())
    )


def compose_official_wide_training_config(
    launch_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the unchanged official network on the Wide environment."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    launch = validate_official_wide_launch_config(launch_config)
    variant = str(launch["experiment_variant"])
    if variant not in {"rnn-sp", "rnn-op"}:
        raise ValueError("Unsupported official experiment variant.")
    experiments, unused_jaxmarl = base._prepare_official_imports()
    del unused_jaxmarl
    overrides = [
        f"+experiment={variant}",
        "+env=test_time_wide",
        f"SEED={int(launch['seed'])}",
        "NUM_SEEDS=1",
        "NUM_CHECKPOINTS=3",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
        "model.TOTAL_TIMESTEPS=3e7",
        "model.REW_SHAPING_HORIZON=1.5e7",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(experiments / "ppo" / "config"),
    ):
        composed = compose(config_name="base", overrides=overrides)
    resolved = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("Official Hydra composition did not produce a mapping.")
    config = dict(resolved)
    kwargs = config.get("env", {}).get("ENV_KWARGS", {})
    if not isinstance(kwargs, Mapping) or kwargs.get("layout") != WIDE_LAYOUT:
        raise ValueError("Official Hydra composition did not select Test Time Wide.")

    translated_config = copy.deepcopy(config)
    translated_config["env"]["ENV_KWARGS"]["layout"] = "test_time_simple"
    translated_launch = copy.deepcopy(dict(launch))
    translated_launch["layout"] = "test_time_simple"
    translated_launch["environment_kwargs"]["layout"] = "test_time_simple"
    base._validate_resolved_config(translated_config, translated_launch)
    return config


def wide_reference_configuration_comparison(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the same accepted network while retaining Wide layout facts."""

    model = resolved_config.get("model")
    environment = resolved_config.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Wide resolved configuration lacks model or environment.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping):
        raise ValueError("Wide resolved configuration lacks environment kwargs.")
    variant_is_other_play = list(
        kwargs.get("op_ingredient_permutations", ())
    ) == [0, 1]
    registered_variant = (
        model.get("NUM_ENVS") == (64 if variant_is_other_play else 256)
    )
    registered_wide_layout = kwargs.get("layout") == WIDE_LAYOUT

    translated = copy.deepcopy(dict(resolved_config))
    translated["env"]["ENV_KWARGS"]["layout"] = "test_time_simple"
    if variant_is_other_play:
        translated["model"]["NUM_ENVS"] = 256
    comparison = _BASE_REFERENCE_COMPARISON(translated)
    comparison["checks"]["registered_wide_layout"] = registered_wide_layout
    comparison["checks"]["registered_training_variant"] = registered_variant
    comparison["matches_accepted_reference"] = all(
        comparison["checks"].values()
    )
    for field, passed in (
        ("registered_wide_layout", registered_wide_layout),
        ("registered_training_variant", registered_variant),
    ):
        if not passed:
            comparison["material_differences"].append(
                {
                    "field": field,
                    "accepted_reference": (
                        "test_time_wide"
                        if field == "registered_wide_layout"
                        else "256 SP environments or 64 Other-Play environments"
                    ),
                    "candidate": "different",
                }
            )
    return comparison


def official_wide_source_dependency_records(
    experiment_variant: str,
) -> list[dict[str, str]]:
    """Bind the Wide Hydra environment and this compatibility layer."""

    records = list(_BASE_DEPENDENCIES(experiment_variant))
    experiments, unused_jaxmarl = base._official_roots()
    del unused_jaxmarl
    simple_environment = (
        experiments / "ppo" / "config" / "env" / "test_time_simple.yaml"
    ).resolve()
    records = [
        item
        for item in records
        if Path(str(item["path"])).resolve() != simple_environment
    ]
    overcooked_root = Path(__file__).resolve().parents[1]
    additions = (
        experiments / "ppo" / "config" / "env" / "test_time_wide.yaml",
        Path(__file__).resolve(),
        overcooked_root / "scripts" / "run_path_c_official_wide_training.py",
    )
    records.extend(
        {"path": str(path.resolve()), "sha256": base._file_sha256(path.resolve())}
        for path in additions
    )
    by_path = {str(item["path"]): item for item in records}
    return [by_path[path] for path in sorted(by_path)]


@contextmanager
def _wide_artifact_loader() -> Iterator[None]:
    original = artifact.load_official_launch_config
    artifact.load_official_launch_config = load_official_wide_launch_config
    try:
        yield
    finally:
        artifact.load_official_launch_config = original


def build_official_wide_artifact_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _wide_artifact_loader():
        return artifact.build_official_artifact_manifest(*args, **kwargs)


def validate_official_wide_artifact_manifest(
    *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    with _wide_artifact_loader():
        return artifact.validate_official_artifact_manifest(*args, **kwargs)


def run_official_wide_production_training(
    launch_config_path: str | Path,
) -> dict[str, Any]:
    """Execute one Wide run without changing the frozen Simple adapter."""

    originals = {
        "load": base.load_official_launch_config,
        "compose": base.compose_official_training_config,
        "comparison": base.reference_configuration_comparison,
        "dependencies": base.official_source_dependency_records,
        "acceptance": base.validate_official_reference_acceptance_report,
        "build": base.build_official_artifact_manifest,
        "validate": base.validate_official_artifact_manifest,
    }

    def validate_simple_acceptance(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        current_dependencies = base.official_source_dependency_records
        current_loader = base.load_official_launch_config
        base.official_source_dependency_records = _BASE_DEPENDENCIES
        base.load_official_launch_config = originals["load"]
        try:
            return _BASE_VALIDATE_ACCEPTANCE(*args, **kwargs)
        finally:
            base.official_source_dependency_records = current_dependencies
            base.load_official_launch_config = current_loader

    base.load_official_launch_config = load_official_wide_launch_config
    base.compose_official_training_config = compose_official_wide_training_config
    base.reference_configuration_comparison = wide_reference_configuration_comparison
    base.official_source_dependency_records = (
        official_wide_source_dependency_records
    )
    base.validate_official_reference_acceptance_report = (
        validate_simple_acceptance
    )
    base.build_official_artifact_manifest = build_official_wide_artifact_manifest
    base.validate_official_artifact_manifest = (
        validate_official_wide_artifact_manifest
    )
    try:
        return base.run_official_production_training(
            Path(launch_config_path).resolve()
        )
    finally:
        base.load_official_launch_config = originals["load"]
        base.compose_official_training_config = originals["compose"]
        base.reference_configuration_comparison = originals["comparison"]
        base.official_source_dependency_records = originals["dependencies"]
        base.validate_official_reference_acceptance_report = originals[
            "acceptance"
        ]
        base.build_official_artifact_manifest = originals["build"]
        base.validate_official_artifact_manifest = originals["validate"]


__all__ = [
    "WIDE_LAYOUT",
    "build_official_wide_artifact_manifest",
    "compose_official_wide_training_config",
    "load_official_wide_launch_config",
    "official_wide_source_dependency_records",
    "run_official_wide_production_training",
    "validate_official_wide_artifact_manifest",
    "validate_official_wide_launch_config",
    "wide_reference_configuration_comparison",
]
