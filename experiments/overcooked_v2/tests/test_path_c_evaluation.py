from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.overcooked_v2.path_c_audit_battery import (
    FrozenAuditBatteryV1,
    ProbeScriptV1,
)
from experiments.overcooked_v2.path_c_belief_audit import (
    CategoricalPosteriorFullStateSampler,
    CategoricalRolloutResultV1,
    ForwardBranchV1,
    ForwardHypothesisV1,
    FrozenInstrumentProbeV1,
    InstrumentKernelCellV1,
    RNGKeyScheduleV1,
    SnapshotV1,
    build_instrument_cell_registry_v2,
    build_instrument_evidence_from_kernel_records_v2,
    collect_paired_battery_kernel_records,
    recompute_instrument_measurement_from_evidence_v2,
    sparse_state_merging_forward_recursion,
)
from experiments.overcooked_v2.path_c_evaluation import (
    PATH_C_DECISION_SCHEMA,
    PATH_C_MEASUREMENT_SCHEMA,
    canonical_sha256,
    assign_group_disjoint_folds,
    build_path_c_readout_features,
    build_secondary_profile_v1,
    assemble_path_c_measurements,
    empirical_kernel_distance_audit,
    evaluate_path_c_decision,
    fingerprint_admission_measurement,
    load_and_validate_path_c_inputs,
    load_frozen_preregistration,
    pass_af_claim_rule,
    phase_b_go_no_go_rule,
    probe_support_measurement,
    require_formal_benchmark_artifact,
    validate_cross_identity_folds,
    validate_runtime_path_c_config,
    _EXPECTED_VARIANT_INPUT_CONTRACTS,
    _runtime_contract,
    _validate_v3_dataset_shards,
    _validate_v3_policy_artifacts,
)
from experiments.overcooked_v2.path_c_protocol import (
    ActingEpisodeLog,
    ActingEpisodeSpec,
    ActingStepLog,
    PolicyAction,
    ProbeBudgetState,
    baseline_selection_payload,
    estimate_audit_cost,
    estimate_locked_primary_endpoint,
    matched_policy_metrics,
    select_strongest_baseline_on_design,
)
from experiments.overcooked_v2.path_c_return_artifacts import (
    BoundReturnProbeBudgetPointV1,
    ReturnPointLedgerV1,
)
from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1
from experiments.overcooked_v2.path_c_seed import derive_ocv2_execution_seed
from experiments.overcooked_v2.path_c_sequence import EgoEvidenceSpecV1
from experiments.overcooked_v2.path_c_split import SplitGroupV1, SplitManifestV1


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "path_c_preregistration.yaml"
)

_SOFTWARE_CHECKS = (
    "snapshot_replay_determinism",
    "fork_rng_stream_isolation",
    "option_distribution_generation_inference_shared",
    "exact_mode_no_positive_mass_pruning",
    "response_vocabulary_frozen",
    "sequence_target_alignment",
    "episode_bootstrap_scope",
    "acting_policy_protocol_complete",
    "four_role_split_feasible",
    "artifact_semantic_hashes_bound",
    "static_cost_estimate_within_budget",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_file_reference(path: Path, payload: bytes) -> dict[str, str]:
    path.write_bytes(payload)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_parquet_reference(
    path: Path,
    columns: dict[str, list],
) -> dict[str, str]:
    pytest.importorskip("pyarrow")
    from experiments.overcooked_v2.scripts.diag_d1_dataset import (
        _write_content_addressed_parquet_shard,
    )

    arrays = {
        name: np.asarray(
            values,
            dtype=(
                np.uint64
                if name in {"seed", "episode_seed"}
                else np.uint32 if name == "execution_seed" else None
            ),
        )
        for name, values in columns.items()
    }
    content_path, digest = _write_content_addressed_parquet_shard(path, arrays)
    return {
        "path": content_path.name,
        "sha256": digest,
    }


def _response_spec(payload: dict) -> ResponseSummarySpecV1:
    section = payload["response_summary_spec"]
    return ResponseSummarySpecV1.from_mapping(
        {
            "schema_version": section["schema_version"],
            "response_classes": section["response_classes"],
            "latency_bin_upper_bounds": section["latency_bin_upper_bounds"],
            "structured_multilabel_role": section["structured_multilabel_role"],
        }
    )


def _strict_instrument_evidence(preregistration) -> dict:
    """Build version-2 evidence from complete deterministic outer records."""

    belief = preregistration.payload["belief_kernel_audit"]
    posterior_mode = "exact" if belief["exact_mode"] is True else "approximate"
    posterior = sparse_state_merging_forward_recursion(
        (
            ForwardHypothesisV1("theta-a", "state-a", b"state-a", 0.7),
            ForwardHypothesisV1("theta-b", "state-b", b"state-b", 0.3),
        ),
        ("observation",),
        lambda state, _observation, _step: (
            ForwardBranchV1(
                state_key=state.state_key,
                state_bytes=state.state_bytes,
                probability=1.0,
            ),
        ),
        mode=posterior_mode,
        prune_below=0.0,
    )
    battery = preregistration.audit_battery
    scripts = battery.scripts_for_role("locked_audit")
    sampler = CategoricalPosteriorFullStateSampler(posterior)
    units = tuple(map(str, belief["audit_information_states"]))
    if len(units) != 3:
        raise ValueError("The test instrument fixture requires three registered states.")

    class DeterministicCellRunner:
        def run(self, *, snapshot, hidden_state, probe_script, rng_keys):
            del hidden_state
            token = 1 if snapshot.unit_id.endswith("-b") else 0
            payload = (
                f"{snapshot.sha256}:{probe_script.script_id}:"
                f"{rng_keys.coordinate}:{token}"
            ).encode("utf-8")
            return CategoricalRolloutResultV1(
                response_token_id=token,
                support_violation=False,
                trajectory_sha256=hashlib.sha256(payload).hexdigest(),
            )

    cells = []
    for cell_id, value_class_id, unit_id in (
        ("a-left", "value-a", units[0]),
        ("a-right", "value-a", units[1]),
        ("b", "value-b", units[2]),
    ):
        schedule = RNGKeyScheduleV1.from_original_seeds(
            manifest_seed=7,
            jax=11,
            numpy=22,
            python=33,
            torch=44,
        )
        snapshot = SnapshotV1(
            unit_id=unit_id,
            episode_uid=f"episode:{unit_id}",
            decision_index=0,
            env_state_bytes=f"env:{unit_id}".encode("utf-8"),
            raw_observation_bytes=f"observation:{unit_id}".encode("utf-8"),
            partner_state_bytes=f"partner:{unit_id}".encode("utf-8"),
            ego_state_bytes=f"ego:{unit_id}".encode("utf-8"),
            rng_key_schedule=schedule,
        )
        base_seed = 103
        records = collect_paired_battery_kernel_records(
            snapshot,
            sampler.sample(
                unit_id,
                int(belief["outer_replicas_M"]),
                seed=base_seed,
            ),
            battery,
            DeterministicCellRunner(),
            role="locked_audit",
        )
        cells.append(
            InstrumentKernelCellV1.from_posterior(
                cell_id=cell_id,
                value_class_id=value_class_id,
                audit_unit_id=unit_id,
                snapshot_sha256=snapshot.sha256,
                probe_id=scripts[0].script_id,
                sampler_base_seed=base_seed,
                outer_id_start=0,
                posterior=posterior,
                kernel_records=records,
            )
        )
    frozen_probes = tuple(
        FrozenInstrumentProbeV1(
            probe_id=script.script_id,
            sampling_probability=script.sampling_probability,
        )
        for script in scripts
    )
    registry = build_instrument_cell_registry_v2(
        cells,
        response_vocabulary_sha256=preregistration.response_vocabulary_sha256,
        battery_sha256=battery.sha256,
        frozen_probes=frozen_probes,
        summary_id="response-summary-v1",
        inner_forks_L_inner=battery.L_inner,
        support_violation_token_id=preregistration.response_summary_spec.encode(
            support_violation=True
        ),
    )
    assert registry["sha256"] == belief["instrument_cell_registry_sha256"]
    return build_instrument_evidence_from_kernel_records_v2(
        cells,
        response_vocabulary_sha256=preregistration.response_vocabulary_sha256,
        vocabulary_size_q=preregistration.response_summary_spec.q,
        confidence_delta=float(belief["confidence_delta"]),
        exact_or_approximate=(
            "exact" if belief["exact_mode"] is True else "approximate_bounded"
        ),
        rho_prune=float(posterior.rho_prune),
        posterior_bias_bound=float(belief["posterior_bias_bound"]),
        reset_bias_bound=float(belief["reset_bias_bound"]),
        battery_sha256=battery.sha256,
        frozen_probes=frozen_probes,
        summary_id="response-summary-v1",
        inner_forks_L_inner=battery.L_inner,
        support_violation_token_id=preregistration.response_summary_spec.encode(
            support_violation=True
        ),
        frozen_cell_registry_sha256=belief["instrument_cell_registry_sha256"],
    )


def _return_ledgers_and_selection(
    payload: dict,
    split_manifest: SplitManifestV1,
    *,
    locked_delta: float | None = None,
) -> tuple[ReturnPointLedgerV1, ReturnPointLedgerV1, object]:
    primary = payload["primary_endpoint"]
    budget_contract = payload["budget"]
    budgets = tuple(map(int, primary["probe_budget_grid"]))
    lower = float(primary["normalization_lower"])
    upper = float(primary["normalization_upper"])
    cost = float(primary["probe_cost_per_use"])
    candidates = tuple(map(
        str,
        payload["baselines"]["strongest_baseline_candidates"],
    ))
    locked_effect = (
        float(primary["preregistered_margin"]) + 0.04
        if locked_delta is None
        else float(locked_delta)
    )
    records: dict[str, list[BoundReturnProbeBudgetPointV1]] = {
        "design": [],
        "locked_audit": [],
    }

    def add_curve(
        *,
        split_role: str,
        policy_name: str,
        split_group: SplitGroupV1,
        seed: int,
        normalized_net_return: float,
    ) -> None:
        net_return = lower + normalized_net_return * (upper - lower)
        for probe_budget in budgets:
            episode_uid = (
                f"{split_role}:{policy_name}:{split_group.identity_group}:"
                f"{split_group.seed_group}:seed-{seed}:budget-{probe_budget}"
            )
            budget_state = ProbeBudgetState(
                budget=probe_budget,
                probes_used=0,
                cost_per_probe=cost,
            )
            action = PolicyAction(
                action_id=0,
                is_probe=False,
                propensity=1.0,
                candidate_action_ids=(0,),
                candidate_scores=(0.0,),
                estimated_probe_cost=0.0,
                policy_kind="test_non_probe_action",
            )
            step = ActingStepLog(
                decision_index=0,
                action=action,
                budget_before=budget_state,
                budget_after=budget_state,
                raw_reward=net_return,
                realized_probe_cost=0.0,
                environment_steps=1,
                terminated=True,
                truncated=False,
            )
            source_log = ActingEpisodeLog(
                policy_name=policy_name,
                spec=ActingEpisodeSpec(
                    split_role=split_role,
                    episode_uid=episode_uid,
                    split_group_id=split_group.group_id,
                    mechanism=split_group.mechanism,
                    style_group=split_group.style_group,
                    identity_group=split_group.identity_group,
                    layout_group=split_group.layout_group,
                    seed_group=split_group.seed_group,
                    seed=seed,
                    probe_budget=probe_budget,
                    training_environment_steps=int(
                        budget_contract["training_environment_steps"]
                    ),
                    gradient_updates=int(
                        budget_contract["training_gradient_updates"]
                    ),
                    evaluation_environment_step_limit=int(
                        budget_contract["evaluation_environment_step_limit"]
                    ),
                    evaluation_schedule_id=str(
                        budget_contract["evaluation_schedule_id"]
                    ),
                ),
                probe_cost_per_use=cost,
                steps=(step,),
                raw_return=net_return,
                realized_probe_cost=0.0,
                environment_steps=1,
                final_budget_state=budget_state,
                policy_metrics={
                    "probe_count": 0,
                    "realized_probe_cost": 0.0,
                    "environment_steps": 1,
                },
            )
            records[split_role].append(BoundReturnProbeBudgetPointV1(
                split_role=split_role,
                episode_uid=episode_uid,
                source_episode_log_sha256=source_log.sha256,
                source_episode_log=source_log,
                point=source_log.return_probe_budget_point(),
            ))

    design_groups = split_manifest.groups_for_role("design")
    locked_groups = split_manifest.groups_for_role("locked_audit")
    if len(design_groups) < 4 or len(locked_groups) < 4:
        raise ValueError("Test split manifest needs four design and locked groups.")
    design_policies = (
        "probing_ego",
        *candidates,
        "random_probe",
        "no_probe",
        "direct_information",
    )
    for candidate_index, policy_name in enumerate(design_policies):
        for identity_index, split_group in enumerate(design_groups):
            for seed in split_manifest.numeric_seeds_for_group(split_group.group_id):
                add_curve(
                    split_role="design",
                    policy_name=policy_name,
                    split_group=split_group,
                    seed=seed,
                    normalized_net_return=0.8 - 0.05 * candidate_index,
                )

    selected_baseline = candidates[0]
    locked_baselines = tuple(map(
        str,
        payload["baselines"]["deployable_required"],
    ))
    for identity_index, split_group in enumerate(locked_groups):
        for seed in split_manifest.numeric_seeds_for_group(split_group.group_id):
            common = {
                "split_role": "locked_audit",
                "split_group": split_group,
                "seed": seed,
            }
            for baseline_name in locked_baselines:
                add_curve(
                    **common,
                    policy_name=baseline_name,
                    normalized_net_return=(
                        0.4 if baseline_name == selected_baseline else 0.3
                    ),
                )
            add_curve(
                **common,
                policy_name="probing_ego",
                normalized_net_return=0.4 + locked_effect,
            )

    design_ledger = ReturnPointLedgerV1(
        points=tuple(records["design"]),
        split_role="design",
        split_manifest_sha256=str(payload["split"]["manifest_sha256"]),
        numeric_seed_schedule_sha256=split_manifest.numeric_seed_schedule_sha256,
        factory_registry_sha256=_digest("factory-registry"),
        environment_manifest_sha256=_digest("environment-manifest"),
        policy_artifact_sha256_by_name={
            name: _digest(f"policy:{name}")
            for name in design_policies
        },
        probe_budget_grid=budgets,
        normalization_rule=str(primary["normalization_rule"]),
        normalization_lower=lower,
        normalization_upper=upper,
        probe_cost_per_use=cost,
    )
    locked_ledger = ReturnPointLedgerV1(
        points=tuple(records["locked_audit"]),
        split_role="locked_audit",
        split_manifest_sha256=str(payload["split"]["manifest_sha256"]),
        numeric_seed_schedule_sha256=split_manifest.numeric_seed_schedule_sha256,
        factory_registry_sha256=_digest("factory-registry"),
        environment_manifest_sha256=_digest("environment-manifest"),
        policy_artifact_sha256_by_name={
            name: _digest(f"policy:{name}")
            for name in ("probing_ego", *locked_baselines)
        },
        probe_budget_grid=budgets,
        normalization_rule=str(primary["normalization_rule"]),
        normalization_lower=lower,
        normalization_upper=upper,
        probe_cost_per_use=cost,
    )
    primary_layout_by_mechanism = dict(split_manifest.primary_layout_strata)
    primary_design_group_ids = tuple(sorted(
        group.group_id
        for group in split_manifest.groups
        if split_manifest.role_for(group.group_id) == "design"
        and group.layout_stratum == primary_layout_by_mechanism[group.mechanism]
    ))
    primary_design_group_set = set(primary_design_group_ids)
    primary_design_curves = tuple(
        curve
        for curve in design_ledger.curves(
            split_role="design",
            policy_names=candidates,
        )
        if curve.split_group_id in primary_design_group_set
    )
    selection = select_strongest_baseline_on_design(
        primary_design_curves,
        candidate_order=candidates,
        split_role="design",
        return_point_ledger_sha256=design_ledger.sha256,
        design_split_group_ids=primary_design_group_ids,
    )
    return design_ledger, locked_ledger, selection


def _change_first_ledger_return(
    ledger_payload: dict,
    *,
    delta: float,
) -> None:
    record = ledger_payload["points"][0]
    source_log_payload = record["source_episode_log"]
    source_log_payload["steps"][0]["raw_reward"] += float(delta)
    source_log_payload["raw_return"] += float(delta)
    record["raw_return"] += float(delta)
    source_log = ActingEpisodeLog.from_mapping(source_log_payload)
    record["source_episode_log_sha256"] = source_log.sha256
    ledger_core = {
        key: value
        for key, value in ledger_payload.items()
        if key != "ledger_sha256"
    }
    ledger_payload["ledger_sha256"] = canonical_sha256(ledger_core)


def _frozen_payload(
    tmp_path: Path,
    *,
    locked_delta: float | None = None,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    payload["version"] = "path_c_preregistration_v3_test_fixture"
    payload["status"] = "frozen"
    payload["freeze_timestamp"] = "2026-07-10T00:00:00Z"
    payload["response_summary_spec"]["response_classes"] = ["response"]
    payload["response_summary_spec"]["latency_bin_upper_bounds"] = [0]

    code_commit = "a" * 40
    split_manifest = SplitManifestV1.build(
        tuple(
            SplitGroupV1(
                group_id=f"group-{index}",
                mechanism="test_mechanism",
                identity_group=f"identity-{index}",
                style_group=f"style-{index}",
                seed_group=f"seed-{index}",
                layout_group=f"layout-{index}",
                layout_stratum="layout-control",
            )
            for index in range(20)
        ),
        manifest_seed=7,
        cross_fit_folds=2,
    )
    split_manifest_bytes = json.dumps(
        split_manifest.to_mapping(), sort_keys=True
    ).encode("utf-8")
    audit_battery = FrozenAuditBatteryV1(
        battery_id="path-c-test-battery",
        battery_version="v1",
        T_probe=5,
        L_inner=3,
        core_scripts=(
            ProbeScriptV1("core-a", (0, 1, 0, 1, 0), 0.5),
            ProbeScriptV1("core-b", (1, 0, 1, 0, 1), 0.5),
        ),
    )
    audit_battery_bytes = json.dumps(
        audit_battery.to_manifest(), sort_keys=True
    ).encode("utf-8")
    semantic_source_refs = {
        "module_registry": _write_file_reference(
            tmp_path / "module-registry.json",
            json.dumps(
                {
                    "schema_version": "path_c_module_registry_v1",
                    "source_git_sha": code_commit,
                    "test_execution_status": "passed",
                    "modules": [
                        {
                            "id": "test-module",
                            "status": "tested",
                            "test_ids": ["tests/test_static.py::test_contract"],
                        }
                    ],
                },
                sort_keys=True,
            ).encode("utf-8"),
        ),
        "partner_registry": _write_file_reference(
            tmp_path / "partner-registry.json",
            b'{"schema_version":"path_c_partner_registry_v1"}',
        ),
        "option_policy": _write_file_reference(
            tmp_path / "option-policy.py",
            b"def option_distribution(): pass\n",
        ),
        "split_manifest": _write_file_reference(
            tmp_path / "split-manifest.json",
            split_manifest_bytes,
        ),
        "audit_battery": _write_file_reference(
            tmp_path / "audit-battery.json",
            audit_battery_bytes,
        ),
    }
    payload["semantic_sources"] = copy.deepcopy(semantic_source_refs)

    payload["primary_endpoint"].update(
        {
            "normalization_lower": -10.0,
            "normalization_upper": 20.0,
            "probe_cost_per_use": 0.25,
            "preregistered_margin": 0.01,
            "bootstrap_iterations": 200,
        }
    )
    payload["budget"].update(
        {
            "training_environment_steps": 1000,
            "training_gradient_updates": 200,
            "evaluation_environment_step_limit": 1000,
            "evaluation_schedule_id": "schedule-v1",
            "probe_budget_grid_sha256": canonical_sha256(
                list(payload["primary_endpoint"]["probe_budget_grid"])
            ),
            "action_support_sha256": _digest("action-support"),
            "maximum_audit_primitive_steps": 10_000_000,
        }
    )
    payload["split"]["manifest_sha256"] = split_manifest.sha256
    evidence_spec = EgoEvidenceSpecV1(
        observation_dim=4,
        num_primitive_actions=2,
        num_options=3,
        max_primitive_steps_per_decision=64,
        progress_event_dim=2,
        observation_schema="test_public_observation_v1",
        primitive_action_names=("stay", "interact"),
        option_names=("noop", "prepare", "serve"),
        progress_event_names=("progress", "delivery"),
    )
    payload["evidence_spec"].update({
        "observation_dim": evidence_spec.observation_dim,
        "num_primitive_actions": evidence_spec.num_primitive_actions,
        "num_options": evidence_spec.num_options,
        "progress_event_dim": evidence_spec.progress_event_dim,
        "observation_schema": evidence_spec.observation_schema,
        "primitive_action_names": list(evidence_spec.primitive_action_names),
        "option_names": list(evidence_spec.option_names),
        "progress_event_names": list(evidence_spec.progress_event_names),
    })
    _design_ledger, _locked_ledger, selection = _return_ledgers_and_selection(
        payload,
        split_manifest,
        locked_delta=locked_delta,
    )
    payload["baselines"]["locked_selection_artifact_sha256"] = selection.sha256
    payload["belief_kernel_audit"].update(
        {
            "outer_replicas_M": 256,
            "simultaneous_cell_count": 3,
            "confidence_delta": 0.05,
            "inner_forks_L_inner": 3,
            "probe_horizon_T_probe": 5,
            "audit_information_states": [
                "registered-state-a-left",
                "registered-state-a-right",
                "registered-state-b",
            ],
        }
    )
    payload["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ] = {
        "schema_version": "path_c_theta_prior_v1",
        "partner_registry_sha256": semantic_source_refs["partner_registry"][
            "sha256"
        ],
        "support": [
            {"theta_id": "theta-a", "probability": 0.5},
            {"theta_id": "theta-b", "probability": 0.5},
        ],
    }
    payload["audit_battery"]["battery_sha256"] = audit_battery.sha256
    payload["audit_battery"]["core_probe_count"] = len(audit_battery.core_scripts)
    payload["power_analysis"].update(
        {
            "clusters_per_trial": 4,
            "simulation_repetitions": 100,
            "bootstrap_iterations": 200,
            "seed": 19,
            "equivalence_margin": 0.05,
            "minimum_positive_power": 0.8,
            "minimum_null_equivalence_power": 0.8,
            "minimum_joint_power": 0.75,
        }
    )
    bindings = payload["semantic_bindings"]
    for key in bindings:
        if key == "rng_key_schedule_version":
            bindings[key] = "path_c_rng_key_schedule_v1"
        elif key == "code_commit":
            bindings[key] = code_commit
        else:
            bindings[key] = _digest(key)
    for source_name, reference in semantic_source_refs.items():
        bindings[f"{source_name}_sha256"] = reference["sha256"]
    bindings["evidence_spec_sha256"] = evidence_spec.sha256()
    bindings["response_vocabulary_sha256"] = _response_spec(payload).sha256
    bindings["audit_battery_sha256"] = payload["audit_battery"]["battery_sha256"]
    bindings["split_manifest_sha256"] = payload["split"]["manifest_sha256"]
    registry_cells = []
    for cell_id, value_class_id, unit_id in (
        ("a-left", "value-a", "registered-state-a-left"),
        ("a-right", "value-a", "registered-state-a-right"),
        ("b", "value-b", "registered-state-b"),
    ):
        schedule = RNGKeyScheduleV1.from_original_seeds(
            manifest_seed=7,
            jax=11,
            numpy=22,
            python=33,
            torch=44,
        )
        snapshot = SnapshotV1(
            unit_id=unit_id,
            episode_uid=f"episode:{unit_id}",
            decision_index=0,
            env_state_bytes=f"env:{unit_id}".encode("utf-8"),
            raw_observation_bytes=f"observation:{unit_id}".encode("utf-8"),
            partner_state_bytes=f"partner:{unit_id}".encode("utf-8"),
            ego_state_bytes=f"ego:{unit_id}".encode("utf-8"),
            rng_key_schedule=schedule,
        )
        registry_cells.append({
            "cell_id": cell_id,
            "value_class_id": value_class_id,
            "audit_unit_id": unit_id,
            "snapshot_sha256": snapshot.sha256,
            "probe_id": audit_battery.scripts_for_role("locked_audit")[0].script_id,
            "sampler_base_seed": 103,
            "outer_id_start": 0,
            "outer_replicas_M": 256,
        })
    registry_payload = build_instrument_cell_registry_v2(
        registry_cells,
        response_vocabulary_sha256=bindings["response_vocabulary_sha256"],
        battery_sha256=audit_battery.sha256,
        frozen_probes=tuple(
            FrozenInstrumentProbeV1(
                probe_id=script.script_id,
                sampling_probability=script.sampling_probability,
            )
            for script in audit_battery.scripts_for_role("locked_audit")
        ),
        summary_id="response-summary-v1",
        inner_forks_L_inner=audit_battery.L_inner,
        support_violation_token_id=_response_spec(payload).encode(
            support_violation=True
        ),
    )
    payload["belief_kernel_audit"]["instrument_cell_registry_sha256"] = (
        registry_payload["sha256"]
    )
    bindings["resolved_config_sha256"] = canonical_sha256(
        _runtime_contract(payload)
    )
    return payload


def _write_frozen_preregistration(tmp_path: Path, payload: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    body = copy.deepcopy(payload if payload is not None else _frozen_payload(tmp_path))
    path = tmp_path / "path_c_preregistration_v3.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path, load_frozen_preregistration(path)


def _valid_measurements(
    preregistration,
    *,
    locked_delta: float | None = None,
) -> dict:
    registered_validity_checks = preregistration.payload["belief_kernel_audit"][
        "validity_checks"
    ]
    design_ledger, locked_ledger, selection = _return_ledgers_and_selection(
        preregistration.payload,
        preregistration.split_manifest,
        locked_delta=locked_delta,
    )
    if selection.sha256 != preregistration.payload["baselines"][
        "locked_selection_artifact_sha256"
    ]:
        raise ValueError("Test fixture selection does not match the frozen selection.")
    primary_layout_by_mechanism = dict(
        preregistration.split_manifest.primary_layout_strata
    )
    primary_locked_group_ids = tuple(sorted(
        group.group_id
        for group in preregistration.split_manifest.groups
        if preregistration.split_manifest.role_for(group.group_id) == "locked_audit"
        and group.layout_stratum == primary_layout_by_mechanism[group.mechanism]
    ))
    locked_curves = tuple(
        curve
        for curve in locked_ledger.curves(
            split_role="locked_audit",
            policy_names=("probing_ego", selection.selected_baseline),
        )
        if curve.split_group_id in set(primary_locked_group_ids)
    )
    primary_result = estimate_locked_primary_endpoint(
        [curve for curve in locked_curves if curve.policy_name == "probing_ego"],
        [
            curve
            for curve in locked_curves
            if curve.policy_name == selection.selected_baseline
        ],
        baseline_selection=selection,
        locked_return_point_ledger_sha256=locked_ledger.sha256,
        split_role="locked_audit",
        primary_split_group_ids=primary_locked_group_ids,
        preregistered_margin=float(
            preregistration.primary_endpoint["preregistered_margin"]
        ),
        confidence_level=float(
            preregistration.primary_endpoint["confidence_level"]
        ),
        bootstrap_iterations=int(
            preregistration.primary_endpoint["bootstrap_iterations"]
        ),
        seed=int(preregistration.payload["power_analysis"]["seed"]),
    )
    primary_payload = {
        **asdict(primary_result),
        "bootstrap_iterations": int(
            preregistration.primary_endpoint["bootstrap_iterations"]
        ),
        "bootstrap_seed": int(preregistration.payload["power_analysis"]["seed"]),
    }
    required_policies = tuple(map(
        str,
        preregistration.payload["budget"]["required_policy_keys"],
    ))
    policy_rows = {
        name: {
            "training_environment_steps": int(
                preregistration.payload["budget"]["training_environment_steps"]
            ),
            "gradient_updates": int(
                preregistration.payload["budget"]["training_gradient_updates"]
            ),
            "evaluation_environment_step_limit": int(
                preregistration.payload["budget"][
                    "evaluation_environment_step_limit"
                ]
            ),
            "evaluation_schedule_id": str(
                preregistration.payload["budget"]["evaluation_schedule_id"]
            ),
            "probe_budget_grid_sha256": str(
                preregistration.payload["budget"]["probe_budget_grid_sha256"]
            ),
            "probe_cost_per_use": float(
                preregistration.primary_endpoint["probe_cost_per_use"]
            ),
            "action_support_sha256": str(
                preregistration.payload["budget"]["action_support_sha256"]
            ),
            "trainable_parameters": 10,
            "training_flops": 100.0,
            "wall_clock_seconds": 1.0,
            "inference_latency_ms": 0.1,
        }
        for name in required_policies
    }
    fairness_report = matched_policy_metrics(
        policy_rows,
        required_policies=required_policies,
    )
    belief = preregistration.payload["belief_kernel_audit"]
    audit_cost_estimate = estimate_audit_cost(
        audit_units=len(belief["audit_information_states"]),
        probes=int(preregistration.payload["audit_battery"]["core_probe_count"]),
        outer_replicas_M=int(belief["outer_replicas_M"]),
        inner_forks_L_inner=int(belief["inner_forks_L_inner"]),
        horizon_T_probe=int(belief["probe_horizon_T_probe"]),
        max_primitive_steps=int(
            preregistration.payload["evidence_spec"][
                "max_primitive_actions_per_decision"
            ]
        ),
        maximum_primitive_step_budget=int(
            preregistration.payload["budget"]["maximum_audit_primitive_steps"]
        ),
        design_audit_units=(
            len(belief["audit_information_states"])
            if preregistration.audit_battery.design_only_scripts
            else 0
        ),
        design_only_probes=len(preregistration.audit_battery.design_only_scripts),
    )
    instrument_evidence = _strict_instrument_evidence(preregistration)
    instrument_measurement = recompute_instrument_measurement_from_evidence_v2(
        instrument_evidence
    )
    return {
        "software_conformance": {
            "schema_version": "path_c_software_conformance_v1",
            "checks": {name: True for name in _SOFTWARE_CHECKS},
        },
        "software_test_report": {
            "schema_version": "path_c_software_test_report_v1",
            "module_registry_sha256": preregistration.semantic_bindings[
                "module_registry_sha256"
            ],
            "source_git_sha": preregistration.semantic_bindings["code_commit"],
            "test_execution_status": "passed",
            "tests": [
                {
                    "test_id": test_id,
                    "outcome": "passed",
                    "report_sha256": _digest(f"report:{test_id}"),
                    "covers": [
                        *_SOFTWARE_CHECKS,
                        *registered_validity_checks,
                    ],
                }
                for test_id in preregistration.module_registry_test_ids
            ],
        },
        "artifact_contract": {
            "schema_version": "path_c_artifacts_v3",
            "valid": True,
            "semantic_bindings_valid": True,
            "active_stage": "secondary_mechanisms",
        },
        "audit_cost_estimate": audit_cost_estimate,
        "policy_metrics": {
            "schema_version": "path_c_policy_metrics_v1",
            "required_policies": list(required_policies),
            "policies": policy_rows,
            **fairness_report,
        },
        "instrument_evidence": instrument_evidence,
        "instrument_measurement": instrument_measurement,
        "design_return_point_ledger": design_ledger.to_mapping(),
        "locked_return_point_ledger": locked_ledger.to_mapping(),
        "baseline_selection": {
            **baseline_selection_payload(selection),
            "artifact_sha256": _digest("baseline-selection-file"),
        },
        "primary_endpoint": primary_payload,
        "secondary_profile": {
            "schema_version": "path_c_secondary_profile_v1",
            "available": False,
            "decision_eligible": False,
            "measurements": {},
        },
    }


def _valid_dataset_shards(tmp_path: Path, preregistration) -> dict:
    datasets = {}
    snapshot_ref = _write_file_reference(
        tmp_path / "locked-audit-snapshots.json",
        b'{"schema_version":"path_c_snapshot_manifest_v1"}',
    )
    for role in ("train", "design", "calibration", "locked_audit"):
        claim_scale = role in {"train", "locked_audit"}
        groups = preregistration.split_manifest.groups_for_role(role)
        group_seed_cells = [
            (group, int(seed))
            for group in groups
            for seed in preregistration.split_manifest.numeric_seeds_for_group(
                group.group_id
            )
        ]
        episodes = 2000 if claim_scale else len(group_seed_cells)
        transitions = 40000 if claim_scale else len(group_seed_cells)
        episode_indexes = [index % episodes for index in range(transitions)]
        row_cells = [
            group_seed_cells[index % len(group_seed_cells)]
            for index in episode_indexes
        ]
        row_groups = [cell[0] for cell in row_cells]
        episode_seed_by_index = {
            episode_index: int(
                hashlib.sha256(
                    (
                        f"{group_seed_cells[episode_index % len(group_seed_cells)][1]}:"
                        f"{episode_index}"
                    ).encode("ascii")
                ).hexdigest()[:16],
                16,
            )
            for episode_index in set(episode_indexes)
        }
        chunk_ref = _write_parquet_reference(
            tmp_path / f"{role}.parquet",
            {
                "episode_uid": [
                    f"{role}-episode-{index}" for index in episode_indexes
                ],
                "option_transition": [1] * transitions,
                "split_group_id": [group.group_id for group in row_groups],
                "mechanism": [group.mechanism for group in row_groups],
                "style_group": [group.style_group for group in row_groups],
                "collection_role": [role] * transitions,
                "surface_identity_key": [
                    group.identity_group for group in row_groups
                ],
                "seed_group": [group.seed_group for group in row_groups],
                "seed": [
                    cell[1] for cell in row_cells
                ],
                "episode_seed": [
                    episode_seed_by_index[index] for index in episode_indexes
                ],
                "execution_seed": [
                    derive_ocv2_execution_seed(episode_seed_by_index[index])
                    for index in episode_indexes
                ],
                "layout_style": [group.layout_group for group in row_groups],
                "probe_selected": [0] * transitions,
                "probe_cost_per_use": [
                    float(preregistration.primary_endpoint["probe_cost_per_use"])
                ] * transitions,
                "probe_realized_cost": [0.0] * transitions,
            },
        )
        datasets[role] = {
            "schema_version": "path_c_dataset_shards_v1",
            "collection_role": role,
            "storage": "content_addressed_append_only_shards",
            "format": "parquet",
            "split_manifest_sha256": preregistration.semantic_bindings[
                "split_manifest_sha256"
            ],
            "split_group_ids": [group.group_id for group in groups],
            "mechanisms": sorted({group.mechanism for group in groups}),
            "style_groups": sorted({group.style_group for group in groups}),
            "response_vocabulary_sha256": (
                preregistration.response_vocabulary_sha256
            ),
            "evidence_spec_sha256": preregistration.evidence_spec_sha256,
            "effective_episodes": episodes,
            "effective_transitions": transitions,
            "audit_snapshot_manifest": (
                copy.deepcopy(snapshot_ref) if role == "locked_audit" else None
            ),
            "semantic_bindings": copy.deepcopy(
                preregistration.semantic_bindings
            ),
            "chunks": [{**chunk_ref, "rows": transitions}],
        }
    return datasets


def _refresh_instrument_measurement(measurements: dict) -> None:
    measurements["instrument_evidence"]["kernel_records_sha256"] = canonical_sha256([
        record
        for cell in measurements["instrument_evidence"]["cells"]
        for record in cell["kernel_records"]
    ])
    measurements["instrument_measurement"] = (
        recompute_instrument_measurement_from_evidence_v2(
            measurements["instrument_evidence"]
        )
    )


def _set_selected_probe_token(cell: dict, token_id: int) -> None:
    probe_id = str(cell["probe_id"])
    for record in cell["kernel_records"]:
        selected = [
            probe
            for probe in record["probe_responses"]
            if str(probe["probe_id"]) == probe_id
        ]
        if len(selected) != 1:
            raise ValueError("Test cell lacks one selected frozen probe.")
        selected[0]["responses"][0]["response_token_id"] = int(token_id)


def _valid_policy_artifacts(tmp_path: Path, preregistration) -> dict:
    required = {
        "probing_ego",
        *preregistration.required_baselines,
        "random_probe",
        "no_probe",
        "direct_information",
    }
    policies = {}
    for name in sorted(required):
        artifact_ref = _write_file_reference(
            tmp_path / f"{name}.bin",
            f"immutable-policy-{name}".encode("utf-8"),
        )
        policies[name] = {
            "schema_version": "path_c_policy_artifact_v1",
            "policy_name": name,
            "artifact": artifact_ref,
            "oracle_baseline": name == "direct_information",
            "input_contract": _EXPECTED_VARIANT_INPUT_CONTRACTS[name],
            "acting_protocol_version": "path_c_acting_policy_v1",
            "training_environment_steps": 1000,
            "gradient_updates": 200,
            "evaluation_environment_step_limit": 1000,
            "evaluation_schedule_id": "schedule-v1",
            "probe_budget_grid_sha256": preregistration.payload["budget"][
                "probe_budget_grid_sha256"
            ],
            "action_support_sha256": preregistration.payload["budget"][
                "action_support_sha256"
            ],
            "probe_cost_per_use": preregistration.primary_endpoint[
                "probe_cost_per_use"
            ],
            "trainable_parameters": 10,
            "training_flops": 100.0,
            "wall_clock_seconds": 1.0,
            "inference_latency_ms": 0.1,
            "semantic_bindings": copy.deepcopy(
                preregistration.semantic_bindings
            ),
        }
    return policies


def _write_bound_measurement_reference(
    tmp_path: Path,
    name: str,
    measurement: dict,
    preregistration,
) -> dict[str, str]:
    payload = copy.deepcopy(measurement)
    payload.update(
        {
            "measurement_schema_version": PATH_C_MEASUREMENT_SCHEMA,
            "preregistration_sha256": preregistration.sha256,
            "resolved_path_c_sha256": preregistration.runtime_contract_sha256,
            "semantic_bindings": copy.deepcopy(
                preregistration.semantic_bindings
            ),
        }
    )
    path = tmp_path / f"measurement-{name}.json"
    return _write_file_reference(
        path,
        json.dumps(payload, sort_keys=True).encode("utf-8"),
    )


def _write_stage_manifest_fixture(tmp_path: Path):
    prereg_path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _valid_measurements(preregistration)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    policies = _valid_policy_artifacts(tmp_path, preregistration)
    measurement_refs = {
        name: _write_bound_measurement_reference(
            tmp_path,
            name,
            measurement,
            preregistration,
        )
        for name, measurement in measurements.items()
        if name != "artifact_contract"
    }

    stage_names = {
        "instrument_validity": {
            "software_conformance",
            "software_test_report",
            "policy_metrics",
            "audit_cost_estimate",
            "instrument_evidence",
            "instrument_measurement",
        },
        "design_and_calibration_freeze": {
            "software_conformance",
            "software_test_report",
            "policy_metrics",
            "audit_cost_estimate",
            "instrument_evidence",
            "instrument_measurement",
            "design_return_point_ledger",
            "baseline_selection",
        },
        "locked_primary_efficacy": {
            "software_conformance",
            "software_test_report",
            "policy_metrics",
            "audit_cost_estimate",
            "instrument_evidence",
            "instrument_measurement",
            "design_return_point_ledger",
            "baseline_selection",
            "primary_endpoint",
        },
    }
    role_by_stage = {
        "instrument_validity": "calibration",
        "design_and_calibration_freeze": "design",
        "locked_primary_efficacy": "locked_audit",
    }
    paths = {}
    for stage, names in stage_names.items():
        refs = {name: copy.deepcopy(measurement_refs[name]) for name in names}
        if stage == "locked_primary_efficacy":
            primary_not_run = {
                "schema_version": "path_c_primary_endpoint_v1",
                "available": False,
            }
            refs["primary_endpoint"] = _write_bound_measurement_reference(
                tmp_path,
                "primary-endpoint-not-run",
                primary_not_run,
                preregistration,
            )
        role = role_by_stage[stage]
        payload = {
            "schema_version": "path_c_artifacts_v3",
            "active_stage": stage,
            "preregistration": {
                "sha256": preregistration.sha256,
                "version": preregistration.payload["version"],
                "schema_version": preregistration.payload["schema_version"],
            },
            "resolved_path_c_sha256": preregistration.runtime_contract_sha256,
            "semantic_bindings": copy.deepcopy(
                preregistration.semantic_bindings
            ),
            "dataset_shards": {role: copy.deepcopy(datasets[role])},
            "policy_artifacts": copy.deepcopy(policies),
            "measurement_artifacts": refs,
        }
        path = tmp_path / f"manifest-{stage}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths[stage] = path
    return prereg_path, preregistration, paths


def test_frozen_v3_preregistration_binds_all_semantic_hashes(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    assert preregistration.payload["schema_version"] == (
        "path_c_preregistration_schema_v3"
    )
    assert preregistration.payload["measurement_schema_version"] == (
        PATH_C_MEASUREMENT_SCHEMA
    )
    assert preregistration.evidence_spec_sha256 == (
        preregistration.evidence_spec.sha256()
    )
    assert preregistration.response_vocabulary_sha256 == (
        preregistration.response_summary_spec.sha256
    )
    for key, value in preregistration.semantic_bindings.items():
        if key not in {"code_commit", "rng_key_schedule_version"}:
            assert len(value) == 64
            int(value, 16)


def test_preregistration_rejects_evidence_and_response_hash_drift(tmp_path):
    evidence_drift = _frozen_payload(tmp_path)
    evidence_drift["evidence_spec"]["observation_dim"] += 1
    path = tmp_path / "evidence-drift.yaml"
    path.write_text(yaml.safe_dump(evidence_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_spec SHA-256"):
        load_frozen_preregistration(path)

    response_drift = _frozen_payload(tmp_path)
    response_drift["semantic_bindings"]["response_vocabulary_sha256"] = _digest(
        "wrong-response-vocabulary"
    )
    path = tmp_path / "response-drift.yaml"
    path.write_text(yaml.safe_dump(response_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="response vocabulary SHA-256"):
        load_frozen_preregistration(path)


def test_preregistration_rejects_unbound_runtime_split_and_battery_drift(tmp_path):
    runtime_drift = _frozen_payload(tmp_path)
    runtime_drift["probe"]["return_floor"] = 0.5
    path = tmp_path / "runtime-drift.yaml"
    path.write_text(yaml.safe_dump(runtime_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved_config_sha256"):
        load_frozen_preregistration(path)

    split_drift = _frozen_payload(tmp_path)
    split_drift["split"]["manifest_sha256"] = _digest("different-split")
    path = tmp_path / "split-drift.yaml"
    path.write_text(yaml.safe_dump(split_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="split manifest SHA-256"):
        load_frozen_preregistration(path)

    battery_drift = _frozen_payload(tmp_path)
    battery_drift["audit_battery"]["battery_sha256"] = _digest(
        "different-battery"
    )
    path = tmp_path / "battery-drift.yaml"
    path.write_text(yaml.safe_dump(battery_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="audit battery SHA-256"):
        load_frozen_preregistration(path)


def test_preregistration_rejects_noncanonical_evidence_and_exact_pruning(tmp_path):
    evidence_drift = _frozen_payload(tmp_path)
    evidence_drift["evidence_spec"]["public_observation"] = "partner_identity"
    path = tmp_path / "evidence-contract-drift.yaml"
    path.write_text(yaml.safe_dump(evidence_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="current observable state"):
        load_frozen_preregistration(path)

    exact_pruning = _frozen_payload(tmp_path)
    exact_pruning["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "rho_prune"
    ] = 0.01
    path = tmp_path / "exact-pruning.yaml"
    path.write_text(yaml.safe_dump(exact_pruning, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact Tier-1 reference recursion"):
        load_frozen_preregistration(path)

    exact_reset_bias = _frozen_payload(tmp_path)
    exact_reset_bias["belief_kernel_audit"]["reset_bias_bound"] = 0.01
    path = tmp_path / "exact-reset-bias.yaml"
    path.write_text(yaml.safe_dump(exact_reset_bias, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="zero posterior and reset bias"):
        load_frozen_preregistration(path)

    unbounded_approximation = _frozen_payload(tmp_path)
    unbounded_belief = unbounded_approximation["belief_kernel_audit"]
    unbounded_belief["exact_mode"] = False
    unbounded_belief["tier1_hypothesis_prune"] = 0.01
    unbounded_belief["tier1_approximate_mode"]["enable"] = True
    unbounded_approximation["semantic_bindings"][
        "resolved_config_sha256"
    ] = canonical_sha256(_runtime_contract(unbounded_approximation))
    path = tmp_path / "unbounded-approximation.yaml"
    path.write_text(
        yaml.safe_dump(unbounded_approximation, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive posterior_bias_bound"):
        load_frozen_preregistration(path)


def test_preregistration_schema_and_semantic_binding_keys_are_exact(tmp_path):
    missing_binding = _frozen_payload(tmp_path)
    del missing_binding["semantic_bindings"]["module_registry_sha256"]
    path = tmp_path / "missing-binding.yaml"
    path.write_text(yaml.safe_dump(missing_binding, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_bindings.*key mismatch"):
        load_frozen_preregistration(path)

    unknown_top_level = _frozen_payload(tmp_path)
    unknown_top_level["unregistered_section"] = {}
    path = tmp_path / "unknown-section.yaml"
    path.write_text(yaml.safe_dump(unknown_top_level, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistration key mismatch"):
        load_frozen_preregistration(path)


def test_every_preregistration_section_rejects_missing_and_unknown_keys(tmp_path):
    base = _frozen_payload(tmp_path)
    section_names = (
        "semantic_bindings",
        "semantic_sources",
        "evidence_spec",
        "ensemble",
        "probe",
        "response_summary_spec",
        "fingerprint",
        "synthetic_factorial",
        "split",
        "budget",
        "baselines",
        "primary_endpoint",
        "secondary_endpoints",
        "power_analysis",
        "belief_kernel_audit",
        "audit_battery",
        "artifact_contract",
        "decision",
    )
    for index, section_name in enumerate(section_names):
        missing = copy.deepcopy(base)
        del missing[section_name][next(iter(missing[section_name]))]
        path = tmp_path / f"missing-section-key-{index}.yaml"
        path.write_text(yaml.safe_dump(missing, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{section_name} key mismatch"):
            load_frozen_preregistration(path)

        unknown = copy.deepcopy(base)
        unknown[section_name]["unregistered_key"] = True
        path = tmp_path / f"unknown-section-key-{index}.yaml"
        path.write_text(yaml.safe_dump(unknown, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{section_name} key mismatch"):
            load_frozen_preregistration(path)


def test_belief_nested_mappings_and_theta_prior_items_have_exact_keys(tmp_path):
    base = _frozen_payload(tmp_path)
    nested_names = (
        "posterior_full_state_sampler",
        "tier1_exact_enumeration_replay",
        "tier1_approximate_mode",
        "tier2_hash_and_match",
        "rng",
        "validity_measurement_schema",
    )
    for index, nested_name in enumerate(nested_names):
        missing = copy.deepcopy(base)
        nested = missing["belief_kernel_audit"][nested_name]
        del nested[next(iter(nested))]
        path = tmp_path / f"missing-belief-key-{index}.yaml"
        path.write_text(yaml.safe_dump(missing, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{nested_name} key mismatch"):
            load_frozen_preregistration(path)

        payload = copy.deepcopy(base)
        payload["belief_kernel_audit"][nested_name]["unregistered_key"] = True
        path = tmp_path / f"unknown-belief-key-{index}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{nested_name} key mismatch"):
            load_frozen_preregistration(path)

    prior_unknown = copy.deepcopy(base)
    prior = prior_unknown["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ]
    prior["unregistered_key"] = True
    path = tmp_path / "unknown-prior-key.yaml"
    path.write_text(yaml.safe_dump(prior_unknown, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="prior_over_theta key mismatch"):
        load_frozen_preregistration(path)

    prior_missing = copy.deepcopy(base)
    del prior_missing["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ]["schema_version"]
    path = tmp_path / "missing-prior-key.yaml"
    path.write_text(yaml.safe_dump(prior_missing, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="prior_over_theta key mismatch"):
        load_frozen_preregistration(path)

    item_unknown = copy.deepcopy(base)
    item_unknown["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ]["support"][0]["unregistered_key"] = True
    path = tmp_path / "unknown-prior-item-key.yaml"
    path.write_text(yaml.safe_dump(item_unknown, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"support\[0\] key mismatch"):
        load_frozen_preregistration(path)

    item_missing = copy.deepcopy(base)
    del item_missing["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ]["support"][0]["probability"]
    path = tmp_path / "missing-prior-item-key.yaml"
    path.write_text(yaml.safe_dump(item_missing, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"support\[0\] key mismatch"):
        load_frozen_preregistration(path)


def test_semantic_sources_are_rehashed_and_bound_to_code_commit(tmp_path):
    tampered = _frozen_payload(tmp_path)
    partner_path = tmp_path / tampered["semantic_sources"]["partner_registry"]["path"]
    partner_path.write_bytes(b"tampered-partner-registry")
    path = tmp_path / "tampered-semantic-source.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="partner_registry SHA-256 mismatch"):
        load_frozen_preregistration(path)

    binding_drift = _frozen_payload(tmp_path)
    binding_drift["semantic_bindings"]["option_policy_sha256"] = _digest(
        "different-option-policy"
    )
    path = tmp_path / "semantic-source-binding-drift.yaml"
    path.write_text(yaml.safe_dump(binding_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from semantic_bindings"):
        load_frozen_preregistration(path)

    module_commit_drift = _frozen_payload(tmp_path)
    module_ref = _write_file_reference(
        tmp_path / "module-registry.json",
        json.dumps(
            {
                "schema_version": "path_c_module_registry_v1",
                "source_git_sha": "b" * 40,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    module_commit_drift["semantic_sources"]["module_registry"] = module_ref
    module_commit_drift["semantic_bindings"]["module_registry_sha256"] = module_ref[
        "sha256"
    ]
    path = tmp_path / "module-source-commit-drift.yaml"
    path.write_text(yaml.safe_dump(module_commit_drift, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source_git_sha must equal"):
        load_frozen_preregistration(path)


@pytest.mark.parametrize("bad_commit", ["a" * 39, "A" * 40, "not-a-commit"])
def test_code_commit_requires_a_lower_case_forty_character_sha(tmp_path, bad_commit):
    payload = _frozen_payload(tmp_path)
    payload["semantic_bindings"]["code_commit"] = bad_commit
    path = tmp_path / "bad-code-commit.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="forty-character lower-case"):
        load_frozen_preregistration(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty_support", "support must be non-empty"),
        ("duplicate_theta", "support ids must be unique"),
        ("zero_probability", "probability must be positive"),
        ("wrong_sum", "probabilities must sum to one"),
        ("wrong_registry", "does not bind the partner registry"),
        ("wrong_version", "wrong schema version"),
    ],
)
def test_theta_prior_is_versioned_positive_unique_and_normalized(
    tmp_path,
    mutation,
    message,
):
    payload = _frozen_payload(tmp_path)
    prior = payload["belief_kernel_audit"]["tier1_exact_enumeration_replay"][
        "prior_over_theta"
    ]
    if mutation == "empty_support":
        prior["support"] = []
    elif mutation == "duplicate_theta":
        prior["support"][1]["theta_id"] = prior["support"][0]["theta_id"]
    elif mutation == "zero_probability":
        prior["support"][0]["probability"] = 0.0
        prior["support"][1]["probability"] = 1.0
    elif mutation == "wrong_sum":
        prior["support"][0]["probability"] = 0.4
        prior["support"][1]["probability"] = 0.4
    elif mutation == "wrong_registry":
        prior["partner_registry_sha256"] = _digest("wrong-partner-registry")
    elif mutation == "wrong_version":
        prior["schema_version"] = "path_c_theta_prior_v0"
    path = tmp_path / f"invalid-prior-{mutation}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_frozen_preregistration(path)


def test_required_policy_keys_exactly_match_main_deployable_and_direct_policies(tmp_path):
    payload = _frozen_payload(tmp_path)
    payload["budget"]["required_policy_keys"].remove("particle_belief_filter")
    path = tmp_path / "missing-required-particle-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*particle_belief_filter"):
        load_frozen_preregistration(path)

    payload = _frozen_payload(tmp_path)
    payload["budget"]["required_policy_keys"].append("identity_embedding")
    path = tmp_path / "extra-required-oracle-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*identity_embedding"):
        load_frozen_preregistration(path)


def test_preregistration_rejects_alpha_without_a_valid_tost_interval(tmp_path):
    payload = _frozen_payload(tmp_path)
    payload["power_analysis"]["primary_alpha"] = 0.5
    path = tmp_path / "invalid-tost-alpha.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"primary_alpha.*\(0, 0\.5\)"):
        load_frozen_preregistration(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("probe_cost_per_use", -0.01, "probe_cost_per_use must be non-negative"),
        ("preregistered_margin", -0.01, "preregistered_margin must be non-negative"),
    ],
)
def test_preregistration_rejects_negative_primary_cost_or_margin(
    tmp_path,
    field,
    value,
    message,
):
    payload = _frozen_payload(tmp_path)
    payload["primary_endpoint"][field] = value
    path = tmp_path / f"negative-{field}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_frozen_preregistration(path)


def test_legacy_margin_and_gate_fields_require_explicit_migration(tmp_path):
    legacy = _frozen_payload(tmp_path)
    legacy["decision"]["epsilon_F"] = 0.01
    path = tmp_path / "legacy-margin.yaml"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="removed version-2 margin field"):
        load_frozen_preregistration(path)

    legacy_schema = _frozen_payload(tmp_path)
    legacy_schema["schema_version"] = "path_c_preregistration_schema_v2"
    path = tmp_path / "legacy-schema.yaml"
    path.write_text(yaml.safe_dump(legacy_schema, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit version-3 migration"):
        load_frozen_preregistration(path)


def test_runtime_contract_must_exactly_match_frozen_values(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    runtime = copy.deepcopy(preregistration.runtime_contract)
    runtime["ensemble"]["tie_atol"] = 0.1
    with pytest.raises(ValueError, match="ensemble.tie_atol"):
        validate_runtime_path_c_config(runtime, preregistration)


def test_locked_audit_dataset_binds_a_snapshot_manifest_file(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    _validate_v3_dataset_shards(datasets, preregistration, tmp_path)

    missing_reference = copy.deepcopy(datasets)
    missing_reference["locked_audit"]["audit_snapshot_manifest"] = None
    with pytest.raises(ValueError, match="audit_snapshot_manifest must be a mapping"):
        _validate_v3_dataset_shards(
            missing_reference, preregistration, tmp_path
        )

    wrong_digest = copy.deepcopy(datasets)
    wrong_digest["locked_audit"]["audit_snapshot_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit_snapshot_manifest SHA-256 mismatch"):
        _validate_v3_dataset_shards(wrong_digest, preregistration, tmp_path)


def test_non_audit_dataset_roles_require_an_explicit_null_snapshot_manifest(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    datasets["train"]["audit_snapshot_manifest"] = copy.deepcopy(
        datasets["locked_audit"]["audit_snapshot_manifest"]
    )
    with pytest.raises(ValueError, match="must be null outside locked_audit"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)

    legacy_boolean = copy.deepcopy(datasets)
    legacy_boolean["train"]["audit_snapshot_manifest"] = None
    locked = legacy_boolean["locked_audit"]
    del locked["audit_snapshot_manifest"]
    locked["audit_snapshot_references"] = True
    with pytest.raises(ValueError, match="key mismatch"):
        _validate_v3_dataset_shards(legacy_boolean, preregistration, tmp_path)


def test_dataset_counts_are_recomputed_from_parquet_rows(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    datasets["design"]["effective_episodes"] = 999999
    with pytest.raises(ValueError, match="effective episodes do not equal"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_dataset_validator_rejects_roles_exposed_before_active_stage(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    with pytest.raises(ValueError, match="before its active stage"):
        _validate_v3_dataset_shards(
            datasets,
            preregistration,
            tmp_path,
            required_roles={"design"},
            allowed_roles={"design"},
        )


def test_dataset_transition_count_is_recomputed_from_parquet_rows(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    datasets["design"]["effective_transitions"] += 1
    with pytest.raises(ValueError, match="effective transitions do not equal"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_dataset_row_split_metadata_must_match_frozen_group(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    group = preregistration.split_manifest.groups_for_role("design")[0]
    chunk_ref = _write_parquet_reference(
        tmp_path / "design.parquet",
        {
            "episode_uid": ["design-episode-0"],
            "option_transition": [1],
            "split_group_id": [group.group_id],
            "mechanism": ["forged-mechanism"],
            "style_group": [group.style_group],
            "collection_role": ["design"],
            "surface_identity_key": [group.identity_group],
            "seed_group": [group.seed_group],
            "seed": [
                preregistration.split_manifest.numeric_seeds_for_group(
                    group.group_id
                )[0]
            ],
            "episode_seed": [17],
            "execution_seed": [derive_ocv2_execution_seed(17)],
            "layout_style": [group.layout_group],
            "probe_selected": [0],
            "probe_cost_per_use": [
                float(preregistration.primary_endpoint["probe_cost_per_use"])
            ],
            "probe_realized_cost": [0.0],
        },
    )
    datasets["design"]["chunks"] = [{**chunk_ref, "rows": 1}]
    with pytest.raises(ValueError, match="mechanism differs from the frozen split group"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_dataset_declared_rows_must_match_parquet_rows(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    datasets["design"]["chunks"][0]["rows"] = 999999
    with pytest.raises(ValueError, match="declared rows do not match"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_dataset_probe_cost_is_recomputed_from_parquet_rows(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    group = preregistration.split_manifest.groups_for_role("design")[0]
    wrong_cost = float(preregistration.primary_endpoint["probe_cost_per_use"]) + 1.0
    chunk_ref = _write_parquet_reference(
        tmp_path / "design.parquet",
        {
            "episode_uid": ["design-episode-0"],
            "option_transition": [1],
            "split_group_id": [group.group_id],
            "mechanism": [group.mechanism],
            "style_group": [group.style_group],
            "collection_role": ["design"],
            "surface_identity_key": [group.identity_group],
            "seed_group": [group.seed_group],
            "seed": [
                preregistration.split_manifest.numeric_seeds_for_group(
                    group.group_id
                )[0]
            ],
            "episode_seed": [19],
            "execution_seed": [derive_ocv2_execution_seed(19)],
            "layout_style": [group.layout_group],
            "probe_selected": [0],
            "probe_cost_per_use": [wrong_cost],
            "probe_realized_cost": [0.0],
        },
    )
    datasets["design"]["chunks"] = [{**chunk_ref, "rows": 1}]
    with pytest.raises(ValueError, match="differs from the frozen cost"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_dataset_chunk_content_hash_is_verified_before_use(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    datasets["design"]["chunks"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"chunks\[0\] SHA-256 mismatch"):
        _validate_v3_dataset_shards(datasets, preregistration, tmp_path)


def test_policy_artifacts_bind_the_frozen_probe_cost(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    policies = _valid_policy_artifacts(tmp_path, preregistration)
    _validate_v3_policy_artifacts(policies, preregistration, tmp_path)

    changed_cost = copy.deepcopy(policies)
    changed_cost["exact_belief_filter"]["probe_cost_per_use"] = 0.5
    with pytest.raises(ValueError, match="differs from the frozen primary cost"):
        _validate_v3_policy_artifacts(changed_cost, preregistration, tmp_path)

    missing_cost = copy.deepcopy(policies)
    del missing_cost["probing_ego"]["probe_cost_per_use"]
    with pytest.raises(ValueError, match="key mismatch"):
        _validate_v3_policy_artifacts(missing_cost, preregistration, tmp_path)


def test_policy_fairness_is_recomputed_from_raw_policy_metrics(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _valid_measurements(preregistration)
    policy_name = preregistration.payload["budget"]["required_policy_keys"][-1]
    measurements["policy_metrics"]["policies"][policy_name][
        "training_environment_steps"
    ] += 1
    forged = evaluate_path_c_decision(measurements, preregistration)
    assert forged.status == "SOFTWARE_NONCONFORMANT"

    measurements = _valid_measurements(preregistration)
    measurements["policy_metrics"]["fairness_conformant"] = False
    reported_failure = evaluate_path_c_decision(measurements, preregistration)
    assert reported_failure.status == "SOFTWARE_NONCONFORMANT"


def test_software_conformance_requires_archived_registry_test_rows(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    failed = _valid_measurements(preregistration)
    failed["software_test_report"]["tests"][0]["outcome"] = "failed"
    failed["software_test_report"]["test_execution_status"] = "failed"
    decision = evaluate_path_c_decision(failed, preregistration)
    assert decision.status == "SOFTWARE_NONCONFORMANT"

    missing = _valid_measurements(preregistration)
    missing["software_test_report"]["tests"] = []
    with pytest.raises(ValueError, match="archived test rows"):
        evaluate_path_c_decision(missing, preregistration)

    forged = _valid_measurements(preregistration)
    forged["software_test_report"]["tests"][0]["report_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="archived report SHA-256"):
        evaluate_path_c_decision(forged, preregistration)


def test_design_and_locked_results_are_recomputed_from_separate_raw_ledgers(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)

    changed_design = _valid_measurements(preregistration)
    _change_first_ledger_return(
        changed_design["design_return_point_ledger"],
        delta=1.0,
    )
    with pytest.raises(ValueError, match="does not match the raw design"):
        evaluate_path_c_decision(changed_design, preregistration)

    changed_locked = _valid_measurements(preregistration)
    _change_first_ledger_return(
        changed_locked["locked_return_point_ledger"],
        delta=1.0,
    )
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(changed_locked, preregistration)


def test_formal_return_ledgers_use_nine_design_and_eight_locked_policies(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    design, locked, _selection = _return_ledgers_and_selection(
        preregistration.payload,
        preregistration.split_manifest,
    )
    assert {
        record.point.policy_name for record in design.points
    } == set(preregistration.payload["budget"]["required_policy_keys"])
    assert {
        record.point.policy_name for record in locked.points
    } == (
        set(preregistration.payload["budget"]["required_policy_keys"])
        - {"direct_information"}
    )


def test_decision_precedence_is_software_then_instrument_then_primary(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)

    software_failure = _valid_measurements(preregistration)
    software_failure["software_conformance"]["checks"][
        "sequence_target_alignment"
    ] = False
    for downstream in (
        "instrument_evidence",
        "instrument_measurement",
        "design_return_point_ledger",
        "locked_return_point_ledger",
        "baseline_selection",
        "primary_endpoint",
        "secondary_profile",
    ):
        software_failure.pop(downstream)
    decision = evaluate_path_c_decision(software_failure, preregistration)
    assert decision.status == "SOFTWARE_NONCONFORMANT"
    assert decision.software_conformant is False

    cost_failure = _valid_measurements(preregistration)
    cost_failure["audit_cost_estimate"]["primitive_step_upper_bound"] = (
        preregistration.payload["budget"]["maximum_audit_primitive_steps"] + 1
    )
    decision = evaluate_path_c_decision(cost_failure, preregistration)
    assert decision.status == "SOFTWARE_NONCONFORMANT"
    assert decision.instrument_valid is False

    forged_cost = _valid_measurements(preregistration)
    forged_cost["audit_cost_estimate"]["audit_units"] = 1
    forged_cost["audit_cost_estimate"]["primitive_step_upper_bound"] = 1
    forged_cost["audit_cost_estimate"]["within_budget"] = True
    decision = evaluate_path_c_decision(forged_cost, preregistration)
    assert decision.status == "SOFTWARE_NONCONFORMANT"
    assert decision.instrument_valid is False

    instrument_failure = _valid_measurements(preregistration)
    _set_selected_probe_token(
        instrument_failure["instrument_evidence"]["cells"][2],
        0,
    )
    _refresh_instrument_measurement(instrument_failure)
    for downstream in (
        "baseline_selection",
        "primary_endpoint",
        "secondary_profile",
        "design_return_point_ledger",
        "locked_return_point_ledger",
    ):
        instrument_failure.pop(downstream)
    decision = evaluate_path_c_decision(instrument_failure, preregistration)
    assert decision.status == "INSTRUMENT_INVALID"
    assert decision.instrument_valid is False

    primary_missing = _valid_measurements(preregistration)
    primary_missing["primary_endpoint"] = {
        "schema_version": "path_c_primary_endpoint_v1",
        "available": False,
    }
    primary_missing.pop("secondary_profile")
    primary_missing.pop("locked_return_point_ledger")
    decision = evaluate_path_c_decision(primary_missing, preregistration)
    assert decision.status == "PRIMARY_NOT_RUN"
    assert decision.allowed_claim == "instrument_valid_only"

    primary_effective = _valid_measurements(preregistration)
    decision = evaluate_path_c_decision(primary_effective, preregistration)
    assert decision.schema_version == PATH_C_DECISION_SCHEMA
    assert decision.status == "PRIMARY_EFFECTIVE_PENDING_TYPE_B_REVIEW"
    assert decision.primary_effective is True
    assert decision.requires_type_b_review is True


def test_real_stage_manifests_reach_the_official_decision_path(tmp_path):
    prereg_path, _preregistration, manifest_paths = _write_stage_manifest_fixture(
        tmp_path
    )
    expected_status = {
        "instrument_validity": "INSTRUMENT_VALID",
        "design_and_calibration_freeze": "DESIGN_FROZEN_PRIMARY_NOT_RUN",
        "locked_primary_efficacy": "PRIMARY_NOT_RUN",
    }
    for stage, manifest_path in manifest_paths.items():
        inputs = load_and_validate_path_c_inputs(prereg_path, manifest_path)
        assembled = assemble_path_c_measurements(inputs)
        decision = evaluate_path_c_decision(
            assembled,
            inputs.preregistration,
        )
        assert decision.status == expected_status[stage]
        assert decision.primary_effective is None


def test_instrument_summary_is_recomputed_and_validity_uses_raw_cells(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    forged = _valid_measurements(preregistration)
    forged["instrument_measurement"]["alpha_upper"] = 0.0
    with pytest.raises(ValueError, match="not recomputed from instrument_evidence"):
        evaluate_path_c_decision(forged, preregistration)

    invalid = _valid_measurements(preregistration)
    _set_selected_probe_token(invalid["instrument_evidence"]["cells"][2], 0)
    _refresh_instrument_measurement(invalid)
    invalid_decision = evaluate_path_c_decision(invalid, preregistration)
    assert invalid_decision.instrument_valid is False

    valid = evaluate_path_c_decision(
        _valid_measurements(preregistration), preregistration
    )
    assert valid.instrument_valid is True


def test_instrument_mode_rho_and_bias_must_match_the_frozen_estimator(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)

    exact_with_pruning = _valid_measurements(preregistration)
    exact_with_pruning["instrument_evidence"]["rho_prune"] = 0.01
    with pytest.raises(ValueError, match="cannot report approximation bias"):
        evaluate_path_c_decision(exact_with_pruning, preregistration)

    out_of_range = _valid_measurements(preregistration)
    out_of_range["instrument_evidence"]["rho_prune"] = 1.01
    out_of_range["instrument_evidence"]["posterior_bias_bound"] = 1.01
    out_of_range["instrument_evidence"]["exact_or_approximate"] = (
        "approximate_bounded"
    )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        evaluate_path_c_decision(out_of_range, preregistration)

    wrong_mode = _valid_measurements(preregistration)
    wrong_mode["instrument_evidence"]["exact_or_approximate"] = (
        "approximate_bounded"
    )
    with pytest.raises(ValueError, match="changed the frozen posterior mode"):
        evaluate_path_c_decision(wrong_mode, preregistration)


def test_approximate_instrument_requires_posterior_bound_to_cover_rho(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _valid_measurements(preregistration)
    measurements["instrument_evidence"].update(
        {
            "exact_or_approximate": "approximate_bounded",
            "rho_prune": 0.2,
            "posterior_bias_bound": 0.1,
        }
    )
    with pytest.raises(ValueError, match="must cover rho_prune"):
        evaluate_path_c_decision(measurements, preregistration)


def test_registered_approximate_mode_accepts_bounded_rho_prune(tmp_path):
    payload = _frozen_payload(tmp_path)
    belief = payload["belief_kernel_audit"]
    belief.update(
        {
            "exact_mode": False,
            "tier1_hypothesis_prune": 0.01,
            "posterior_bias_bound": 0.10,
            "reset_bias_bound": 0.02,
        }
    )
    belief["tier1_approximate_mode"]["enable"] = True
    payload["semantic_bindings"]["resolved_config_sha256"] = canonical_sha256(
        _runtime_contract(payload)
    )
    _path, preregistration = _write_frozen_preregistration(tmp_path, payload)

    measurements = _valid_measurements(preregistration)
    decision = evaluate_path_c_decision(measurements, preregistration)
    assert decision.instrument_valid is True
    assert decision.status == "PRIMARY_EFFECTIVE_PENDING_TYPE_B_REVIEW"


def test_instrument_cannot_count_inner_forks_as_outer_replicas(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    repeated_outer = _valid_measurements(preregistration)
    repeated_outer["instrument_evidence"]["cells"][0]["kernel_records"][1] = (
        copy.deepcopy(
            repeated_outer["instrument_evidence"]["cells"][0]["kernel_records"][0]
        )
    )
    with pytest.raises(ValueError, match="ordered contiguous outer replicas"):
        evaluate_path_c_decision(repeated_outer, preregistration)

    too_few_outer = _valid_measurements(preregistration)
    too_few_outer["instrument_evidence"]["cells"][0]["kernel_records"].pop()
    with pytest.raises(ValueError, match="frozen number of complete outer-replica"):
        evaluate_path_c_decision(too_few_outer, preregistration)

    forged_summary = _valid_measurements(preregistration)
    forged_summary["instrument_measurement"]["inner_forks_are_outer_samples"] = True
    with pytest.raises(ValueError, match="not recomputed from instrument_evidence"):
        evaluate_path_c_decision(forged_summary, preregistration)


def test_primary_effect_requires_lower_confidence_bound_above_margin(tmp_path):
    boundary_dir = tmp_path / "boundary"
    boundary_payload = _frozen_payload(boundary_dir)
    boundary_payload["primary_endpoint"]["preregistered_margin"] = 0.0
    _path, boundary_preregistration = _write_frozen_preregistration(
        boundary_dir,
        boundary_payload,
    )
    boundary_measurements = _valid_measurements(
        boundary_preregistration,
        locked_delta=0.0,
    )
    boundary_measurements.pop("secondary_profile")
    boundary = evaluate_path_c_decision(
        boundary_measurements,
        boundary_preregistration,
    )
    assert boundary.primary_effective is False
    assert boundary.status == "PRIMARY_NOT_EFFECTIVE"

    _path, preregistration = _write_frozen_preregistration(tmp_path / "above")
    measurements = _valid_measurements(preregistration)
    measurements.pop("secondary_profile")
    with pytest.raises(ValueError, match="secondary_profile"):
        evaluate_path_c_decision(measurements, preregistration)

    measurements["secondary_profile"] = {
        "schema_version": "path_c_secondary_profile_v1",
        "available": False,
        "decision_eligible": False,
        "measurements": {},
    }
    above = evaluate_path_c_decision(measurements, preregistration)
    assert above.primary_effective is True

    forged = _valid_measurements(preregistration)
    forged["primary_endpoint"]["confidence_interval"] = [-1.0, 1.0]
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(forged, preregistration)


def test_locked_primary_requires_design_selection_role_and_frozen_hash(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    wrong_role = _valid_measurements(preregistration)
    wrong_role["baseline_selection"]["selection_role"] = "locked_audit"
    with pytest.raises(ValueError, match="does not match the raw design"):
        evaluate_path_c_decision(wrong_role, preregistration)

    wrong_hash = _valid_measurements(preregistration)
    wrong_hash["baseline_selection"]["sha256"] = _digest(
        "post-audit-baseline-selection"
    )
    with pytest.raises(ValueError, match="does not match the raw design"):
        evaluate_path_c_decision(wrong_hash, preregistration)

    changed_selection = _valid_measurements(preregistration)
    selection = changed_selection["baseline_selection"]
    weakest = selection["candidate_order"][-1]
    selection["design_auc_by_baseline"][weakest] = 0.95
    selection_content = {
        key: value
        for key, value in selection.items()
        if key not in {"sha256", "artifact_sha256"}
    }
    selection["sha256"] = canonical_sha256(selection_content)
    with pytest.raises(ValueError, match="does not match the raw design"):
        evaluate_path_c_decision(changed_selection, preregistration)

    changed_primary_hash = _valid_measurements(preregistration)
    changed_primary_hash["primary_endpoint"]["baseline_selection_sha256"] = _digest(
        "changed-after-lock"
    )
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(changed_primary_hash, preregistration)


def test_locked_primary_cannot_change_the_frozen_estimand_or_inference_unit(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    changed_estimand = _valid_measurements(preregistration)
    changed_estimand["primary_endpoint"]["endpoint_name"] = "response_accuracy"
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(changed_estimand, preregistration)

    changed_cluster = _valid_measurements(preregistration)
    changed_cluster["primary_endpoint"]["cluster_unit"] = "transition"
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(changed_cluster, preregistration)

    too_few_clusters = _valid_measurements(preregistration)
    too_few_clusters["primary_endpoint"]["clusters"] = 3
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(too_few_clusters, preregistration)

    changed_margin = _valid_measurements(preregistration)
    changed_margin["primary_endpoint"]["preregistered_margin"] = -1.0
    with pytest.raises(ValueError, match="does not match the raw locked-audit"):
        evaluate_path_c_decision(changed_margin, preregistration)


def test_secondary_profile_cannot_replace_or_veto_primary_status(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _valid_measurements(preregistration)
    measurements["secondary_profile"] = {
        "schema_version": "path_c_secondary_profile_v1",
        "available": True,
        "decision_eligible": False,
        "measurements": {"unbound_claim": {"value": 1.0}},
    }
    with pytest.raises(ValueError, match="key mismatch"):
        evaluate_path_c_decision(measurements, preregistration)

    profile = build_secondary_profile_v1({
        "value_readout": {
            "estimand": "cross_fitted_value_accuracy",
            "role": "locked_audit",
            "cluster_unit": "episode_uid",
            "estimate": 0.6,
            "confidence_interval": [0.5, 0.7],
            "multiplicity_adjustment": "holm",
            "source_artifact_sha256": "a" * 64,
        }
    })
    measurements["secondary_profile"] = profile
    decision = evaluate_path_c_decision(measurements, preregistration)
    assert decision.primary_effective is True
    assert decision.secondary == profile["measurements"]


def test_historical_go_and_pass_entry_points_cannot_restore_v2_semantics(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    measurements = _valid_measurements(preregistration)
    compatibility = phase_b_go_no_go_rule(measurements, preregistration)
    assert compatibility["status"] == "PRIMARY_EFFECTIVE_PENDING_TYPE_B_REVIEW"
    assert compatibility["status"] != "GO"
    with pytest.raises(ValueError, match="removed in Path C version 3"):
        pass_af_claim_rule(measurements, preregistration)


def test_production_parquet_writer_preserves_unsigned_seed_identity(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from experiments.overcooked_v2.scripts.diag_d1_dataset import (
        _write_content_addressed_parquet_shard,
    )

    canonical_seed = (1 << 63) + 17
    path, _sha256 = _write_content_addressed_parquet_shard(
        tmp_path / "seed-round-trip.parquet",
        {
            "seed": np.asarray([canonical_seed], dtype=np.uint64),
            "episode_seed": np.asarray([canonical_seed + 1], dtype=np.uint64),
        },
    )
    table = pq.read_table(path)
    assert table.schema.field("seed").type == pa.uint64()
    assert table.schema.field("episode_seed").type == pa.uint64()
    assert table.column("seed").to_pylist() == [canonical_seed]
    assert table.column("episode_seed").to_pylist() == [canonical_seed + 1]


def test_training_loader_keeps_unsigned_seed_shards_exact(tmp_path):
    from experiments.overcooked_v2.scripts.diag_d1_dataset import (
        _write_content_addressed_parquet_shard,
    )
    from experiments.overcooked_v2.scripts.diag_d1_train import (
        _load_chunk_columns,
    )

    seeds = ((1 << 63) + 5, (1 << 63) + 9)
    loaded = []
    for index, seed in enumerate(seeds):
        path, _digest_value = _write_content_addressed_parquet_shard(
            tmp_path / f"seed-shard-{index}.parquet",
            {
                "seed": np.asarray([seed], dtype=np.uint64),
                "episode_seed": np.asarray([seed + 1], dtype=np.uint64),
                "execution_seed": np.asarray(
                    [derive_ocv2_execution_seed(seed + 1)],
                    dtype=np.uint32,
                ),
            },
        )
        loaded.append(_load_chunk_columns(path))
    combined = np.concatenate([item["seed"] for item in loaded])
    assert combined.dtype == np.uint64
    assert combined.tolist() == list(seeds)


def test_high_seed_bits_change_the_ocv2_execution_identity():
    from experiments.overcooked_v2.path_c_seed import (
        derive_ocv2_execution_seed,
    )

    low_seed = 7
    high_seed = (1 << 32) + low_seed
    assert (low_seed & 0xFFFFFFFF) == (high_seed & 0xFFFFFFFF)
    assert derive_ocv2_execution_seed(low_seed) != derive_ocv2_execution_seed(
        high_seed
    )


def test_return_ledger_rejects_missing_registered_group_seed_cells(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    design, _locked, _selection = _return_ledgers_and_selection(
        preregistration.payload,
        preregistration.split_manifest,
    )
    target_group = preregistration.split_manifest.groups_for_role("design")[0]
    missing_seed = preregistration.split_manifest.numeric_seeds_for_group(
        target_group.group_id
    )[-1]
    incomplete = replace(
        design,
        points=tuple(
            record
            for record in design.points
            if not (
                record.point.split_group_id == target_group.group_id
                and int(record.point.seed) == int(missing_seed)
            )
        ),
    )
    with pytest.raises(ValueError, match="group.*seed.*missing"):
        incomplete.validate_split_manifest(preregistration.split_manifest)


def test_dataset_rejects_missing_group_seed_cell_and_episode_seed_drift(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    datasets = _valid_dataset_shards(tmp_path, preregistration)
    design = datasets["design"]
    source_path = tmp_path / design["chunks"][0]["path"]
    columns = pq.read_table(source_path).to_pydict()

    target_group = str(columns["split_group_id"][0])
    target_seed = int(columns["seed"][0])
    keep = [
        not (
            str(group_id) == target_group and int(seed) == target_seed
        )
        for group_id, seed in zip(
            columns["split_group_id"],
            columns["seed"],
            strict=True,
        )
    ]
    missing_columns = {
        name: [value for value, retain in zip(values, keep, strict=True) if retain]
        for name, values in columns.items()
    }
    missing_ref = _write_parquet_reference(
        tmp_path / "design-missing-seed.parquet",
        missing_columns,
    )
    missing = copy.deepcopy(datasets)
    missing["design"]["chunks"] = [
        {**missing_ref, "rows": len(missing_columns["episode_uid"])}
    ]
    missing["design"]["effective_episodes"] = len(
        set(missing_columns["episode_uid"])
    )
    missing["design"]["effective_transitions"] = len(
        missing_columns["episode_uid"]
    )
    with pytest.raises(ValueError, match="group and seed schedule is incomplete"):
        _validate_v3_dataset_shards(missing, preregistration, tmp_path)

    drift_columns = copy.deepcopy(columns)
    same_group_indexes = [
        index
        for index, group_id in enumerate(drift_columns["split_group_id"])
        if str(group_id) == target_group
    ]
    first, second = same_group_indexes[:2]
    assert int(drift_columns["seed"][first]) != int(
        drift_columns["seed"][second]
    )
    drift_columns["episode_uid"][second] = drift_columns["episode_uid"][first]
    drift_ref = _write_parquet_reference(
        tmp_path / "design-seed-drift.parquet",
        drift_columns,
    )
    drift = copy.deepcopy(datasets)
    drift["design"]["chunks"] = [
        {**drift_ref, "rows": len(drift_columns["episode_uid"])}
    ]
    with pytest.raises(ValueError, match="episode_uid cannot change its numeric"):
        _validate_v3_dataset_shards(drift, preregistration, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("disagreement_threshold", -1.0, "disagreement_threshold.*non-negative"),
        ("min_selected_probes", -10, "min_selected_probes.*positive integer"),
        ("min_probe_opportunities", 0, "min_probe_opportunities.*positive integer"),
        ("min_context_coverage", -0.5, "min_context_coverage.*\[0, 1\]"),
        ("min_action_coverage", 1.5, "min_action_coverage.*\[0, 1\]"),
    ],
)
def test_frozen_probe_numeric_domains_fail_closed(
    tmp_path,
    field,
    value,
    message,
):
    payload = _frozen_payload(tmp_path)
    payload["probe"][field] = value
    payload["semantic_bindings"]["resolved_config_sha256"] = canonical_sha256(
        _runtime_contract(payload)
    )
    path = tmp_path / f"invalid-probe-{field}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_frozen_preregistration(path)


def test_live_readout_split_fingerprint_probe_and_benchmark_guards(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    context = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    features = build_path_c_readout_features(
        context,
        {"probe": np.ones((2, 1), dtype=np.float32)},
    )
    assert np.array_equal(features["base_only"], context)
    assert features["probe"].shape == (2, 3)

    surface = np.asarray(["a", "a", "b", "b", "c", "c"])
    seed_group = np.asarray(["s0", "s0", "s1", "s1", "s2", "s2"])
    layout = np.asarray(["l0", "l0", "l1", "l1", "l2", "l2"])
    mechanism = np.asarray(["m0", "m1", "m0", "m1", "m0", "m1"])
    folds = assign_group_disjoint_folds(
        surface,
        seed_group,
        layout,
        n_folds=3,
    )
    assert validate_cross_identity_folds(
        folds,
        surface,
        seed_group,
        layout,
        mechanism,
    )["valid"] is True

    fingerprint = fingerprint_admission_measurement(
        control_kind="metadata_only",
        mechanism=["m0", "m0", "m1", "m1", "m2", "m2"],
        fingerprint_id=[0, 1, 0, 1, 0, 1],
        visibility_ci_lo=0.9,
        chance_accuracy=0.5,
        value_null_ci=[-0.001, 0.001],
        joint_rv_null_ci=[-0.001, 0.001],
        normalized_advantage_null_ci=[-0.001, 0.001],
        preregistration=preregistration,
    )
    assert fingerprint["available_for_synthetic_power"] is False

    support = probe_support_measurement(
        [False, False],
        [True, True],
        ["c0", "c1"],
        [0, 1],
    )
    assert support["selected"] == 0
    assert support["context_coverage"] == 0.0
    assert support["action_coverage"] == 0.0

    with pytest.raises(ValueError, match="Data-collection"):
        require_formal_benchmark_artifact(
            {
                "evaluation_kind": "data_collection_only",
                "collection_role": "data_collection_only",
                "active_probe_collection": True,
                "benchmark_return_eligible": False,
            }
        )


def test_ecological_kernel_fallback_cannot_gate_instrument_validity(tmp_path):
    _path, preregistration = _write_frozen_preregistration(tmp_path)
    audit = empirical_kernel_distance_audit(
        [[0], [1]],
        ["m0", "m1"],
        ["i0", "i1"],
        ["c0", "c1"],
        ["e0", "e1"],
        preregistration,
    )
    assert audit["available"] is True
    assert audit["theorem_eligible"] is False
    assert audit["decision_eligible"] is False


def test_response_readout_retains_ordered_joint_likelihood(monkeypatch):
    from experiments.overcooked_v2.scripts import diag_d1_train as d1

    feature_dims = []

    def fake_fit(features, labels, fold_keys, **_kwargs):
        del fold_keys
        feature_dims.append(features.shape[1])
        n_classes = int(np.max(labels)) + 1
        probabilities = np.full((labels.size, n_classes), 0.1)
        probabilities[np.arange(labels.size), labels] = 0.9
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return {"available": True, "probs": probabilities}

    monkeypatch.setattr(d1, "_fit_predict_classifier", fake_fit)
    score = d1._readout_score_for_columns(
        np.ones((4, 2), dtype=np.float32),
        {
            "rv_first": np.asarray([0, 1, 0, 1]),
            "rv_second": np.asarray([1, 1, 0, 0]),
        },
        np.arange(4),
        ["rv_first", "rv_second"],
        np.asarray(["e0", "e1", "e2", "e3"]),
        device="cpu",
        seed=0,
    )
    assert score["likelihood"] == "autoregressive_joint"
    assert score["factor_order"] == ["rv_first", "rv_second"]
    assert feature_dims == [2, 4]
