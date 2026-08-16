from __future__ import annotations

from types import SimpleNamespace

import pytest


def _partner(mechanism: str, parent: str, group: str | None = None):
    return SimpleNamespace(
        generation_mechanism=mechanism,
        parent_training_run_id=parent,
        co_training_group_id=group,
    )


def test_cetr_training_parent_overlap_is_rejected(monkeypatch):
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    monkeypatch.setattr(
        evaluation_app,
        "_cetr_deployment_lineage",
        lambda manifest: ({"train-parent"}, set()),
    )
    with pytest.raises(ValueError, match="CETR train/confirmatory leakage"):
        evaluation_app._validate_cetr_confirmatory_lineage(
            {"runs": []}, [_partner("sp", "train-parent")]
        )


def test_cetr_co_training_group_overlap_is_rejected(monkeypatch):
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    monkeypatch.setattr(
        evaluation_app,
        "_cetr_deployment_lineage",
        lambda manifest: (set(), {"shared-group"}),
    )
    with pytest.raises(ValueError, match="CETR train/confirmatory leakage"):
        evaluation_app._validate_cetr_confirmatory_lineage(
            {"runs": []}, [_partner("sp", "confirm-parent", "shared-group")]
        )


def test_formal_panel_requires_four_distinct_parents_per_mechanism():
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    partners = [
        _partner("sp", "sp-parent-0"),
        _partner("sp", "sp-parent-0"),
        _partner("sp", "sp-parent-0"),
        _partner("sp", "sp-parent-1"),
    ]
    for mechanism in ("sa", "op", "fcp"):
        partners.extend(
            _partner(mechanism, f"{mechanism}-parent-{index}")
            for index in range(4)
        )
    with pytest.raises(ValueError, match="four independent"):
        evaluation_app._validate_formal_confirmatory_panel(partners)


def test_formal_panel_accepts_four_distinct_parents_per_mechanism():
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    partners = [
        _partner(mechanism, f"{mechanism}-parent-{index}")
        for mechanism in ("sp", "sa", "op", "fcp")
        for index in range(4)
    ]
    evaluation_app._validate_formal_confirmatory_panel(partners)
