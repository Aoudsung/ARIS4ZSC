from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _config(*, run_kind: str = "mechanical", minimum: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        run_kind=run_kind,
        partner_pool=SimpleNamespace(checkpoint_stages=(0.0, 0.5, 1.0)),
        evaluation=SimpleNamespace(minimum_partner_runs_per_mechanism=minimum),
    )


def _row(
    *,
    run_id: str,
    parent: str,
    mechanism: str,
    stage: float,
    owner_seed_index: int | None = None,
):
    from src.cetr_zsc.manifest import PartnerRun

    return PartnerRun(
        run_id=run_id,
        role="development_support",
        checkpoint=Path(f"/tmp/{run_id}"),
        parent_training_run_id=parent,
        generation_mechanism=mechanism,
        checkpoint_stage=stage,
        hyperparameter_family="test",
        seed=1,
        seed_index=0,
        jax_prng_key=(0, 1),
        owner_seed_index=owner_seed_index,
    )


def _manifest(
    *,
    mechanisms=("sp", "op", "sa", "fcp"),
    parents_per_mechanism: int = 1,
    stages=(0.0, 0.5, 1.0),
):
    from src.cetr_zsc.manifest import PartnerManifest

    rows = []
    for mechanism in mechanisms:
        for parent_index in range(parents_per_mechanism):
            parent = f"{mechanism}-parent-{parent_index}"
            for stage in stages:
                rows.append(
                    _row(
                        run_id=f"{parent}-{stage}",
                        parent=parent,
                        mechanism=mechanism,
                        stage=stage,
                    )
                )
    return PartnerManifest(layout="test_time_simple", runs=tuple(rows))


def test_training_pool_orders_members_and_aligns_parent_slots() -> None:
    from src.cetr_zsc.partners import build_training_partner_pool

    pool = build_training_partner_pool(_config(), _manifest())
    assert tuple(member.mechanism for member in pool.members) == (
        "fcp",
        "fcp",
        "fcp",
        "op",
        "op",
        "op",
        "sa",
        "sa",
        "sa",
        "sp",
        "sp",
        "sp",
    )
    assert pool.parent_mechanisms == ("fcp", "op", "sa", "sp")
    assert pool.parent_nominal_weights == (0.25,) * 4
    assert all(member.probability == pytest.approx(1.0 / 12.0) for member in pool.members)
    assert pool.parent_members == tuple(
        (3 * index, 3 * index + 1, 3 * index + 2) for index in range(4)
    )
    for parent_index, member_indexes in enumerate(pool.parent_members):
        assert all(pool.members[index].parent_index == parent_index for index in member_indexes)
        assert tuple(pool.members[index].stage_slot for index in member_indexes) == (0, 1, 2)


def test_formal_pool_requires_all_mechanisms_and_exact_parent_count() -> None:
    from src.cetr_zsc.partners import build_training_partner_pool

    with pytest.raises(ValueError, match="all four mechanisms"):
        build_training_partner_pool(_config(run_kind="formal"), _manifest(mechanisms=("sp",)))
    with pytest.raises(ValueError, match="exactly 4 parents"):
        build_training_partner_pool(
            _config(run_kind="formal", minimum=2),
            _manifest(parents_per_mechanism=1),
        )


def test_formal_pool_has_registered_parent_member_and_probability_counts() -> None:
    from src.cetr_zsc.partners import build_training_partner_pool

    pool = build_training_partner_pool(
        _config(run_kind="formal"),
        _manifest(parents_per_mechanism=4),
    )
    assert len(pool.parent_ids) == 16
    assert len(pool.members) == 48
    assert pool.parent_nominal_weights == (1.0 / 16.0,) * 16
    assert all(member.probability == pytest.approx(1.0 / 48.0) for member in pool.members)
    assert pool.parent_ids == tuple(
        f"{mechanism}-parent-{index}"
        for mechanism in ("fcp", "op", "sa", "sp")
        for index in range(4)
    )


def test_training_pool_rejects_parent_id_reused_across_mechanisms() -> None:
    from dataclasses import replace

    from src.cetr_zsc.manifest import PartnerManifest
    from src.cetr_zsc.partners import build_training_partner_pool

    rows = list(_manifest(parents_per_mechanism=4).runs)
    source = rows[12]  # first OP row after the four SP parents
    rows[12] = replace(source, parent_training_run_id="sp-parent-0")
    with pytest.raises(ValueError, match="globally unique"):
        build_training_partner_pool(
            _config(run_kind="formal"),
            PartnerManifest("test_time_simple", tuple(rows)),
        )


def test_pool_rejects_unknown_mechanism_stage_owner_duplicate_and_missing_stage() -> None:
    from src.cetr_zsc.manifest import PartnerManifest
    from src.cetr_zsc.partners import build_training_partner_pool

    valid = list(_manifest().runs)
    unknown = valid.copy()
    unknown[0] = _row(
        run_id=unknown[0].run_id,
        parent=unknown[0].parent_training_run_id,
        mechanism="unknown",
        stage=unknown[0].checkpoint_stage,
    )
    with pytest.raises(ValueError, match="mechanism"):
        build_training_partner_pool(
            _config(), PartnerManifest("test_time_simple", tuple(unknown))
        )

    invalid_stage = valid.copy()
    invalid_stage[0] = _row(
        run_id=invalid_stage[0].run_id,
        parent=invalid_stage[0].parent_training_run_id,
        mechanism=invalid_stage[0].generation_mechanism,
        stage=0.25,
    )
    with pytest.raises(ValueError, match="stage"):
        build_training_partner_pool(
            _config(), PartnerManifest("test_time_simple", tuple(invalid_stage))
        )

    owner = valid.copy()
    owner[0] = _row(
        run_id=owner[0].run_id,
        parent=owner[0].parent_training_run_id,
        mechanism=owner[0].generation_mechanism,
        stage=owner[0].checkpoint_stage,
        owner_seed_index=4,
    )
    with pytest.raises(ValueError, match="owner-free"):
        build_training_partner_pool(
            _config(), PartnerManifest("test_time_simple", tuple(owner))
        )

    duplicate = valid.copy()
    duplicate.append(duplicate[0])
    with pytest.raises(ValueError, match="repeats stage"):
        build_training_partner_pool(
            _config(), PartnerManifest("test_time_simple", tuple(duplicate))
        )

    missing = [
        row
        for row in valid
        if not (
            row.parent_training_run_id == "sp-parent-0"
            and row.checkpoint_stage == 1.0
        )
    ]
    with pytest.raises(ValueError, match="lacks stages"):
        build_training_partner_pool(
            _config(), PartnerManifest("test_time_simple", tuple(missing))
        )
