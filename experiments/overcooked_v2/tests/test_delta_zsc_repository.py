from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

import src.path_c.storage as storage


ROOT = Path(__file__).resolve().parents[3]
REGISTERED_DESIGN_SHA256 = "375586afdc18cc3d285d7562f66a2c4f0369e16e054ef23bc3793727af8433b5"


def test_registered_design_document_is_byte_exact() -> None:
    path = ROOT / "docs" / "theory" / "DELTA_ZSC_COMPLETE_THEORY_AND_DESIGN.md"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTERED_DESIGN_SHA256


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
        "validate-manifest",
        "upstream",
        "train-official-baseline",
        "train",
        "calibrate",
        "evaluate",
        "build-delta-policy-manifest",
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
    workflow = (ROOT / ".github" / "workflows" / "delta-zsc-ci.yml").read_text(
        encoding="utf-8"
    )
    assert 'branches: ["agent/delta-zsc-v5"]' in workflow
    assert "test_delta_zsc_*.py" in workflow
    assert list(
        (ROOT / "experiments" / "overcooked_v2" / "tests").glob(
            "test_delta_zsc_*.py"
        )
    )
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.11"' in project
    assert project.count("5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e") == 2


def test_active_source_contains_no_retired_v44_semantics() -> None:
    retired = (
        "slot_log_belief",
        "TwinDuelingQ",
        "CodebookState",
        "next_q_use_targets",
        "next_q_mask_targets",
        "episode_responsibility_evidence",
        "bellman_control_values",
        "target_response_signatures",
    )
    matches: list[str] = []
    for path in (ROOT / "src" / "path_c").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in retired:
            if token in text:
                matches.append(f"{path.name}:{token}")
    assert matches == []


def test_counterfactual_ground_truth_has_no_critic_or_q_source() -> None:
    source = (ROOT / "src" / "path_c" / "counterfactual_anchor.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "action_values",
        "target_params",
        "critic",
        "bellman",
        "next_q",
    ):
        assert forbidden not in source
    assert "final.raw_return" in source


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
