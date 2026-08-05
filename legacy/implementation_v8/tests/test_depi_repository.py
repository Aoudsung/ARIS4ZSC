from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

import src.path_c.storage as storage


ROOT = Path(__file__).resolve().parents[3]


def test_every_internal_import_resolves() -> None:
    modules: set[str] = set()
    sources = list((ROOT / "src" / "path_c").glob("*.py"))
    sources += list((ROOT / "experiments" / "overcooked_v2").glob("*.py"))
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("src.path_c", "experiments.overcooked_v2")):
                    modules.add(node.module)
    missing = sorted(module for module in modules if importlib.util.find_spec(module) is None)
    assert missing == []


def test_cli_help_imports_without_initializing_optional_runtime() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.overcooked_v2.path_c", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in (
        "build-partner-manifest",
        "mechanical-e2e",
        "scientific-dry-run",
        "validate-manifest",
        "upstream",
        "train-official-baseline",
        "train",
        "collect-pair-comparator-source",
        "fit-pair-comparator",
        "run-development-matrix",
        "evaluate-development-matrix",
        "summarize-development-matrix",
        "cuda-preflight",
        "calibrate-posterior",
        "audit-signals",
        "evaluate-identifiability",
        "evaluate-recoverable-value",
        "build-depi-policy-manifest",
        "evaluate-official",
        "summarize-official",
        "summarize-capacity-control",
        "evaluate-common",
        "evaluate-common-br-prox",
        "summarize-resources",
        "build-formal-claim-report",
    ):
        assert command in result.stdout


def test_ci_targets_real_tests_and_active_branch() -> None:
    workflow = (ROOT / ".github" / "workflows" / "depi-ci.yml").read_text(
        encoding="utf-8"
    )
    assert 'branches: ["agent/delta-zsc-v5"]' in workflow
    assert "test_depi_*.py" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu]" in workflow
    assert "JAX_PLATFORMS: cuda" in workflow
    assert "cuda-preflight" in workflow
    assert "scientific-dry-run --help" in workflow
    assert list(
        (ROOT / "experiments" / "overcooked_v2" / "tests").glob(
            "test_depi_*.py"
        )
    )
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.11"' in project
    assert project.count("5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e") == 2


def test_scientific_dry_run_is_small_complete_and_non_evidentiary() -> None:
    from experiments.overcooked_v2.scientific_dry_run_app import (
        DRY_RUN_EGO_SEEDS,
        DRY_RUN_EPISODES,
        DRY_RUN_FRESH_PARTNERS,
        DRY_RUN_STAGE_ORDER,
    )

    assert DRY_RUN_EGO_SEEDS == (0, 1)
    assert DRY_RUN_EPISODES == 50
    assert DRY_RUN_FRESH_PARTNERS == 2
    assert DRY_RUN_STAGE_ORDER == (
        "two_ego_training",
        "fresh_partner_posterior_calibration",
        "raw_development_evaluation",
        "real_crn_mechanism_collection",
        "identifiability",
        "recoverable_value",
        "dry_run_claim_report",
    )
    source = (
        ROOT / "experiments" / "overcooked_v2" / "scientific_dry_run_app.py"
    ).read_text(encoding="utf-8")
    calibration_source = (
        ROOT / "experiments" / "overcooked_v2" / "calibration_app.py"
    ).read_text(encoding="utf-8")
    assert '"scientific_readout_allowed": False' in source
    assert (
        '"scientific_readout_allowed": config.run_kind == "formal"'
        in calibration_source
    )
    assert "official_pairing_rollouts" in source
    assert '"formal_claim_report_rehearsal": True' in source
    assert '"performance_claim"' in source


def test_active_source_contains_no_retired_v4_or_v5_control_semantics() -> None:
    retired = (
        "slot_log_belief",
        "TwinDuelingQ",
        "CodebookState",
        "next_q_use_targets",
        "next_q_mask_targets",
        "episode_responsibility_evidence",
        "bellman_control_values",
        "target_response_signatures",
        "CurriculumPhase",
        "qualified_base_params",
        "deployment_tier",
        "base_logits",
        "residual_logits",
        "conditional_enabled",
        "regret_enabled",
        "GeneratorAdmission",
        "TargetPolicyEpoch",
        "GaussianMixtureBelief",
        "previous_reward",
        "hard_adaptation_gate",
        "owner_sp_source_fallback",
    )
    matches: list[str] = []
    for path in (ROOT / "src" / "path_c").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in retired:
            if token in text:
                matches.append(f"{path.name}:{token}")
    assert matches == []


def test_counterfactual_ground_truth_is_only_the_truncated_simulator_return() -> None:
    source = (ROOT / "src" / "path_c" / "counterfactual_anchor.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "action_values",
        "target_params",
        "ego_endpoint_value",
        "bootstrapped_return",
        "next_q",
    ):
        assert forbidden not in source
    assert "final.raw_return" in source


def test_retired_v5_control_modules_are_absent() -> None:
    for name in (
        "curriculum.py",
        "fallback.py",
        "policy_epoch.py",
        "qualification.py",
        "qualification_rollout.py",
        "snapshot_archive.py",
        "anchor_replay.py",
        "generator_training.py",
        "partner_generator.py",
        "raw_q.py",
        "regret_potential.py",
    ):
        assert not (ROOT / "src" / "path_c" / name).exists()


def test_sensitive_operational_files_are_untracked_and_ignored() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert not any(Path(path).name in {"SSH_Document.md", ".DS_Store"} for path in tracked)
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "SSH_Document.md" in ignore
    assert ".DS_Store" in ignore


def test_formal_execution_rejects_uncommitted_source(monkeypatch) -> None:
    monkeypatch.setattr(storage, "_repository_state", lambda: ("a" * 40, True))
    with pytest.raises(RuntimeError, match="clean committed"):
        storage.validate_formal_repository_state()
    monkeypatch.setattr(storage, "_repository_state", lambda: ("a" * 40, False))
    storage.validate_formal_repository_state()
