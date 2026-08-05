from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_joint_filter_places_decision_before_transition() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.filtering import sequence_log_likelihood

    # Decision evidence says current z_0=0.  The transition deterministically
    # moves it to z_1=1, where the next response is observed.  Attaching both
    # emissions to the same side of the transition would make them conflict.
    transition = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    decision = jnp.asarray([[[0.0, -12.0]]], dtype=jnp.float32)
    response = jnp.asarray([[[-12.0, 0.0]]], dtype=jnp.float32)
    result = sequence_log_likelihood(
        transition=transition,
        response_log_likelihood=response,
        decision_log_likelihood=decision,
        decision_mask=jnp.asarray([[1.0]], dtype=jnp.float32),
        episode_starts=jnp.asarray([[True]]),
        valid_mask=jnp.asarray([[1.0]], dtype=jnp.float32),
    )
    assert float(result.final_posterior[0, 1]) > 0.999
    assert float(result.decision_count) == 1.0
    assert float(result.response_count) == 1.0


def _run(
    *,
    run_id: str,
    role: str,
    parent: str,
    mechanism: str,
    stage: float,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        role=role,
        checkpoint=f"/{run_id}",
        checkpoint_sha256=(run_id.encode().hex() + "0" * 64)[:64],
        parent_training_run_id=parent,
        generation_mechanism=mechanism,
        checkpoint_stage=stage,
        hyperparameter_family="default",
        seed=seed,
        seed_index=None,
        jax_prng_key=(seed + 1, seed + 2),
        owner_seed_index=None,
        co_training_group_id=None,
        partner_type_id=None,
    )


class _Manifest:
    layout = "test_time_simple"

    def __init__(self, rows):
        self.runs = tuple(rows)

    def by_role(self, role):
        return tuple(row for row in self.runs if row.role == role)


def _support_rows() -> list[SimpleNamespace]:
    rows = []
    counter = 0
    for mechanism in ("sp", "op"):
        for parent_index in range(10):
            parent = f"{mechanism}-parent-{parent_index}"
            for stage in (0.0, 0.5, 1.0):
                rows.append(
                    _run(
                        run_id=f"{parent}-stage-{stage}",
                        role="development_support",
                        parent=parent,
                        mechanism=mechanism,
                        stage=stage,
                        seed=counter,
                    )
                )
                counter += 1
    return rows


def _panel_rows(role: str, per_mechanism: int, offset: int) -> list[SimpleNamespace]:
    rows = []
    for mechanism_index, mechanism in enumerate(("sp", "op", "sa", "fcp")):
        for index in range(per_mechanism):
            rows.append(
                _run(
                    run_id=f"{role}-{mechanism}-{index}",
                    role=role,
                    parent=f"{role}-{mechanism}-parent-{index}",
                    mechanism=mechanism,
                    stage=1.0,
                    seed=offset + mechanism_index * per_mechanism + index,
                )
            )
    return rows


def test_formal_unified_manifest_has_wide_disjoint_support() -> None:
    from src.delta_zsc.manifest import validate_unified_manifest

    manifest = _Manifest(
        _support_rows()
        + _panel_rows("calibration", 5, 100)
        + _panel_rows("confirmatory", 4, 200)
    )
    validate_unified_manifest(
        manifest,
        formal=True,
        require_support=True,
        require_calibration=True,
        require_confirmatory=True,
    )


def test_unified_manifest_rejects_cross_split_parent_overlap() -> None:
    from src.delta_zsc.manifest import validate_unified_manifest

    rows = _support_rows()
    calibration = _panel_rows("calibration", 5, 100)
    confirmatory = _panel_rows("confirmatory", 4, 200)
    confirmatory[0].parent_training_run_id = calibration[0].parent_training_run_id
    with pytest.raises(ValueError, match="parents overlap"):
        validate_unified_manifest(
            _Manifest(rows + calibration + confirmatory),
            formal=True,
            require_support=True,
            require_calibration=True,
            require_confirmatory=True,
        )


def test_matrix_seed_contract_supports_mechanical_and_blocks_partial_formal() -> None:
    from experiments.overcooked_v2.unified_matrix_app import _validate_seeds

    _validate_seeds((-1,), run_kind="mechanical")
    _validate_seeds((0, 1, 2, 3, 4), run_kind="development")
    with pytest.raises(ValueError, match="0..9"):
        _validate_seeds((0, 1, 2), run_kind="formal")


def test_internal_summary_requires_identical_base_parameters() -> None:
    from experiments.overcooked_v2.unified_summary_app import (
        EvaluationNodes,
        _validate_internal_identity,
    )
    from src.delta_zsc.model import (
        BASE_VARIANT,
        FULL_VARIANT,
        JOINT_VARIANT,
        RESPONSE_ONLY_VARIANT,
    )

    def node(base):
        return EvaluationNodes(
            means={0: 1.0, 1: 2.0},
            layout="test_time_simple",
            source={"path": "/tmp/source", "sha256": "0" * 64},
            base_fingerprints={0: base[0], 1: base[1]},
            latent_fingerprints={0: "l0", 1: "l1"},
            deployment_fingerprints={0: "d0", 1: "d1"},
            partner_manifest_sha256="p" * 64,
            evaluation_seed=0,
            episodes_per_pairing=500,
        )

    nodes = {
        BASE_VARIANT: node(("b0", "b1")),
        RESPONSE_ONLY_VARIANT: node(("b0", "b1")),
        JOINT_VARIANT: node(("b0", "b1")),
        FULL_VARIANT: node(("b0", "b1")),
    }
    assert _validate_internal_identity(nodes)[
        "base_parameter_fingerprint_match"
    ]
    nodes[RESPONSE_ONLY_VARIANT] = node(("different", "b1"))
    with pytest.raises(RuntimeError, match="Base policy parameters differ"):
        _validate_internal_identity(nodes)
