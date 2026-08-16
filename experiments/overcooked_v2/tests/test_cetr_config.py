from __future__ import annotations

from pathlib import Path

import pytest


_FORMAL_CONFIG = Path("experiments/overcooked_v2/configs/cetr_simple_formal.yaml")


def test_formal_config_loads_with_registered_identity() -> None:
    from src.cetr_zsc.config import (
        CONFIG_VERSION,
        OFFICIAL_BOOTSTRAP_REPLICATES,
        OFFICIAL_FORMAL_PARENTS_PER_MECHANISM,
        OFFICIAL_TRAINING_RUN_COUNT,
        load_config,
    )

    config = load_config(_FORMAL_CONFIG, run_kind="formal")
    assert config.version == CONFIG_VERSION == 6
    assert config.run_kind == "formal"
    assert config.evaluation.bootstrap_replicates == OFFICIAL_BOOTSTRAP_REPLICATES
    assert config.evaluation.minimum_ego_runs == OFFICIAL_TRAINING_RUN_COUNT
    assert (
        config.evaluation.minimum_partner_runs_per_mechanism
        == OFFICIAL_FORMAL_PARENTS_PER_MECHANISM
    )


def test_version_five_yaml_is_rejected(tmp_path: Path) -> None:
    from src.cetr_zsc.config import load_config

    payload = _FORMAL_CONFIG.read_text(encoding="utf-8").replace(
        "version: 6", "version: 5", 1
    )
    path = tmp_path / "version-5.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Active CETR config version is 6"):
        load_config(path, run_kind="formal")


def test_formal_registration_field_drift_is_rejected(tmp_path: Path) -> None:
    from src.cetr_zsc.config import load_config

    payload = _FORMAL_CONFIG.read_text(encoding="utf-8").replace(
        "bootstrap_replicates: 9999", "bootstrap_replicates: 9998", 1
    )
    path = tmp_path / "drifted-formal.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Formal evaluation registration differs"):
        load_config(path, run_kind="formal")
