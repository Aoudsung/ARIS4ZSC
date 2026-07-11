from __future__ import annotations

import copy
import hashlib

import pytest

from experiments.overcooked_v2.path_c_protocol import (
    ActingEpisodeLog,
    ActingEpisodeSpec,
    ActingStepLog,
    PolicyAction,
    ProbeBudgetState,
)
from experiments.overcooked_v2.path_c_return_artifacts import (
    BoundReturnProbeBudgetPointV1,
    ReturnPointLedgerV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    *,
    role: str,
    policy: str,
    identity: str,
    budget: int,
    raw_return: float,
    episode_suffix: str = "",
) -> BoundReturnProbeBudgetPointV1:
    episode_uid = f"{role}:{policy}:{identity}:{budget}{episode_suffix}"
    budget_before = ProbeBudgetState(
        budget=budget,
        probes_used=0,
        cost_per_probe=0.25,
    )
    is_probe = budget > 0
    action = PolicyAction(
        action_id=1 if is_probe else 0,
        is_probe=is_probe,
        propensity=1.0,
        candidate_action_ids=(0, 1),
        candidate_scores=(0.0, 1.0) if is_probe else (1.0, 0.0),
        estimated_probe_cost=0.25 if is_probe else 0.0,
        policy_kind=policy,
    )
    budget_after = budget_before.consume(is_probe=is_probe)
    source_log = ActingEpisodeLog(
        policy_name=policy,
        spec=ActingEpisodeSpec(
            split_role=role,
            episode_uid=episode_uid,
            split_group_id=f"group-{identity}",
            mechanism="mechanism-a",
            identity_group=identity,
            style_group=f"style-{identity}",
            layout_group="layout-control",
            seed_group=f"seed-{identity}",
            seed=int(
                hashlib.sha256(f"{role}:{identity}".encode("utf-8")).hexdigest()[:8],
                16,
            ),
            probe_budget=budget,
            training_environment_steps=1000,
            gradient_updates=20,
            evaluation_environment_step_limit=100,
            evaluation_schedule_id="schedule-v1",
        ),
        probe_cost_per_use=0.25,
        steps=(
            ActingStepLog(
                decision_index=0,
                action=action,
                budget_before=budget_before,
                budget_after=budget_after,
                raw_reward=raw_return,
                realized_probe_cost=0.25 if is_probe else 0.0,
                environment_steps=100,
                terminated=True,
                truncated=False,
            ),
        ),
        raw_return=raw_return,
        realized_probe_cost=0.25 if is_probe else 0.0,
        environment_steps=100,
        final_budget_state=budget_after,
        policy_metrics={
            "probe_count": int(is_probe),
            "realized_probe_cost": 0.25 if is_probe else 0.0,
            "environment_steps": 100,
        },
    )
    return BoundReturnProbeBudgetPointV1(
        split_role=role,
        episode_uid=episode_uid,
        source_episode_log_sha256=source_log.sha256,
        source_episode_log=source_log,
        point=source_log.return_probe_budget_point(),
    )


def _ledger(role: str) -> ReturnPointLedgerV1:
    points = []
    for policy in ("probing_ego", "strong"):
        for identity in ("identity-0", "identity-1"):
            for budget in (0, 1):
                points.append(
                    _record(
                        role=role,
                        policy=policy,
                        identity=identity,
                        budget=budget,
                        raw_return=(2.0 if policy == "probing_ego" else 1.0),
                    )
                )
    return ReturnPointLedgerV1(
        points=tuple(points),
        split_role=role,
        probe_budget_grid=(0, 1),
        normalization_lower=0.0,
        normalization_upper=4.0,
        probe_cost_per_use=0.25,
        split_manifest_sha256="f" * 64,
        numeric_seed_schedule_sha256="e" * 64,
        factory_registry_sha256="d" * 64,
        environment_manifest_sha256="c" * 64,
        policy_artifact_sha256_by_name={
            "probing_ego": "a" * 64,
            "strong": "b" * 64,
        },
    )


def test_return_point_ledgers_round_trip_without_cross_role_access():
    design_ledger = _ledger("design")
    locked_ledger = _ledger("locked_audit")
    restored_design = ReturnPointLedgerV1.from_mapping(design_ledger.to_mapping())
    restored_locked = ReturnPointLedgerV1.from_mapping(locked_ledger.to_mapping())
    assert restored_design.sha256 == design_ledger.sha256
    assert restored_locked.sha256 == locked_ledger.sha256
    design = restored_design.curves(
        split_role="design",
        policy_names=("probing_ego", "strong"),
    )
    locked = restored_locked.curves(
        split_role="locked_audit",
        policy_names=("probing_ego", "strong"),
    )
    assert len(design) == 4
    assert len(locked) == 4
    assert {curve.policy_name for curve in design} == {"probing_ego", "strong"}


def test_return_point_ledger_rejects_hash_cost_and_schema_tampering():
    payload = _ledger("design").to_mapping()
    bad_hash = copy.deepcopy(payload)
    bad_hash["ledger_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        ReturnPointLedgerV1.from_mapping(bad_hash)

    bad_cost = copy.deepcopy(payload)
    bad_cost["points"][0]["source_episode_log"]["probe_cost_per_use"] = 0.5
    bad_cost["ledger_sha256"] = _digest("not-authoritative")
    with pytest.raises(ValueError, match="final budget state changed"):
        ReturnPointLedgerV1.from_mapping(bad_cost)

    unknown = copy.deepcopy(payload)
    unknown["unregistered"] = True
    with pytest.raises(ValueError, match="key mismatch"):
        ReturnPointLedgerV1.from_mapping(unknown)


def test_return_point_ledger_rejects_cross_role_contamination():
    design_points = list(_ledger("design").points)
    design_points.append(_ledger("locked_audit").points[0])
    with pytest.raises(ValueError, match="ledger split_role"):
        ReturnPointLedgerV1(
            points=tuple(design_points),
            split_role="design",
            probe_budget_grid=(0, 1),
            normalization_lower=0.0,
            normalization_upper=4.0,
            probe_cost_per_use=0.25,
            split_manifest_sha256="f" * 64,
            numeric_seed_schedule_sha256="e" * 64,
            factory_registry_sha256="d" * 64,
            environment_manifest_sha256="c" * 64,
            policy_artifact_sha256_by_name={
                "probing_ego": "a" * 64,
                "strong": "b" * 64,
            },
        )


def test_return_point_ledger_rejects_duplicate_numeric_seed_cells():
    points = list(_ledger("design").points)
    points.append(
        _record(
            role="design",
            policy="probing_ego",
            identity="identity-0",
            budget=0,
            raw_return=3.0,
            episode_suffix=":duplicate-episode",
        )
    )
    with pytest.raises(ValueError, match="numeric-seed/budget cell"):
        ReturnPointLedgerV1(
            points=tuple(points),
            split_role="design",
            probe_budget_grid=(0, 1),
            normalization_lower=0.0,
            normalization_upper=4.0,
            probe_cost_per_use=0.25,
            split_manifest_sha256="f" * 64,
            numeric_seed_schedule_sha256="e" * 64,
            factory_registry_sha256="d" * 64,
            environment_manifest_sha256="c" * 64,
            policy_artifact_sha256_by_name={
                "probing_ego": "a" * 64,
                "strong": "b" * 64,
            },
        )
