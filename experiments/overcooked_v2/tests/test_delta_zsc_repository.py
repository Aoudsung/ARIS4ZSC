from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys


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
        "validate-manifest",
        "upstream",
        "train",
        "calibrate",
        "evaluate",
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
