from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _partner(mechanism: str, parent: str, group: str | None = None):
    return SimpleNamespace(
        generation_mechanism=mechanism,
        parent_training_run_id=parent,
        co_training_group_id=group,
    )


def _real_policy_manifest(
    tmp_path: Path,
    *,
    parent: str,
    group: str | None = None,
) -> dict[str, object]:
    import jax

    from experiments.overcooked_v2.deployment import export_deployment_bundle
    from src.cetr_zsc.config import load_config
    from src.cetr_zsc.model import CetrModel
    from src.cetr_zsc.storage import write_json

    config = load_config(
        "experiments/overcooked_v2/configs/cetr_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    source = tmp_path / "training"
    source.mkdir()
    write_json(
        source / "partner_manifest.json",
        {
            "version": 2,
            "layout": config.environment.layout,
            "runs": [
                {
                    "parent_training_run_id": parent,
                    "co_training_group_id": group,
                }
            ],
        },
    )
    checkpoint = tmp_path / "initializer"
    checkpoint.write_bytes(b"checkpoint")
    model = CetrModel(config, (5, 5, 26), 6)
    bundle = tmp_path / "deployment"
    export_deployment_bundle(
        bundle,
        ego_run_id="ego-0",
        config=config,
        observation_shape=(5, 5, 26),
        params=model.init_parameters(jax.random.PRNGKey(0)),
        source_training_run=source,
        reference_sp_artifact={
            "artifact_type": "cetr_reference_sp",
            "version": 2,
            "layout": config.environment.layout,
            "seed_index": 0,
            "tau_sp": 1.0,
            "episodes_per_pairing": config.evaluation.episodes_per_pairing,
            "evaluation_root_seed": config.evaluation.evaluation_seed,
            "source_checkpoint": str(checkpoint.resolve()),
        },
    )
    return {"runs": [{"policy": str(bundle)}]}


def test_cetr_training_parent_overlap_is_rejected(tmp_path: Path):
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    manifest = _real_policy_manifest(tmp_path, parent="train-parent")
    with pytest.raises(ValueError, match="CETR train/confirmatory leakage"):
        evaluation_app._validate_cetr_confirmatory_lineage(
            manifest, [_partner("sp", "train-parent")]
        )


def test_cetr_co_training_group_overlap_is_rejected(tmp_path: Path):
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    manifest = _real_policy_manifest(tmp_path, parent="train-parent", group="shared-group")
    with pytest.raises(ValueError, match="CETR train/confirmatory leakage"):
        evaluation_app._validate_cetr_confirmatory_lineage(
            manifest, [_partner("sp", "confirm-parent", "shared-group")]
        )


def test_formal_panel_requires_four_distinct_parents_per_mechanism():
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    partners = [
        _partner(mechanism, f"{mechanism}-parent-{index}")
        for mechanism in ("sp", "sa", "op", "fcp")
        for index in range(4)
    ]
    partners[1] = _partner("sp", "sp-parent-0")
    with pytest.raises(ValueError, match="four independent"):
        evaluation_app._validate_formal_confirmatory_panel(partners)

    too_many = [
        _partner(mechanism, f"{mechanism}-parent-{index}")
        for mechanism in ("sp", "sa", "op", "fcp")
        for index in range(4)
    ]
    too_many.append(_partner("sp", "sp-parent-4"))
    with pytest.raises(ValueError, match="four independent"):
        evaluation_app._validate_formal_confirmatory_panel(too_many)


def test_formal_panel_accepts_exactly_sixteen_distinct_parents():
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    partners = [
        _partner(mechanism, f"{mechanism}-parent-{index}")
        for mechanism in ("sp", "sa", "op", "fcp")
        for index in range(4)
    ]
    assert len(partners) == 16
    evaluation_app._validate_formal_confirmatory_panel(partners)
