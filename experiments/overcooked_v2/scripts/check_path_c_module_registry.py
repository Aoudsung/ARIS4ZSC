#!/usr/bin/env python3
"""Pure static consistency check for the Path C module registry.

This script reads files only. It does not import project modules, execute tests,
or infer that an implementation is scientifically valid.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "experiments/overcooked_v2/configs/module_registry.yaml"
DASHBOARD = ROOT / "PROJECT_DASHBOARD.md"
OPTIONAL_MODULE_DESIGN = ROOT / "idea-stage/refine-logs/PATH_C_MODULE_DESIGN.md"
PREREGISTRATION = ROOT / "experiments/overcooked_v2/configs/path_c_preregistration.yaml"
BEGIN = "<!-- PATH_C_MODULE_TRACEABILITY:BEGIN -->"
END = "<!-- PATH_C_MODULE_TRACEABILITY:END -->"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_REVIEW_MODULE_IDS = {
    "A1_CONFIG_BINDING",
    "A2_PRIME_RECURRENT_SEQUENCE",
    "A3_SEQUENCE_TEMPORAL_DIFFERENCE",
    "A4_NORMALIZED_ADVANTAGE_PROBE",
    "A5_BASE_RESIDUAL_SECONDARY",
    "B1_SYNTHETIC_FACTORIAL",
    "B2_RESPONSE_AND_STORAGE",
    "B3_ECOLOGICAL_VALUE_CLASSES",
    "C1_RESPONSE_READOUT_SECONDARY",
    "C2_PRIMARY_RETURN_BUDGET_AUC",
    "C3_RETAINED_DECISION_CODE",
    "C4_POWER_AND_SECONDARY_REPORT",
    "C5_ACTING_BASELINE_BENCHMARK",
    "C6_TWO_STAGE_DECISION",
    "I1_IMMUTABLE_OCV2_SNAPSHOT",
    "I2_EXACT_SHARED_POSTERIOR",
    "I3_FULL_STATE_OUTER_SAMPLING",
    "I4_EXACT_HISTORY_MATCHING",
    "I5_FROZEN_AUDIT_BATTERY",
    "I6_SIMULTANEOUS_KERNEL_BOUNDS",
    "I7_VALIDITY_CONTROLS",
    "I8_SPLIT_AND_CROSS_FITTING",
    "D1_ARTIFACT_CONTRACT",
    "D2_CONFORMANCE_TEST_DEFINITIONS",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _traceability_rows(text: str, *, source_name: str) -> dict[str, str]:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        _fail(f"{source_name} must contain exactly one traceability block")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    rows: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("|---") or "Module ID" in line:
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) != 2:
            _fail(f"malformed traceability row: {raw_line}")
        module_id, status = columns
        if module_id in rows:
            _fail(f"duplicate module id in {source_name}: {module_id}")
        rows[module_id] = status
    if not rows:
        _fail("traceability table is empty")
    return rows


def main() -> int:
    payload = _mapping(json.loads(REGISTRY.read_text(encoding="utf-8")), "registry")
    expected_top = {
        "schema_version",
        "registry_version",
        "source_git_sha",
        "worktree_binding",
        "allowed_statuses",
        "test_execution_status",
        "status_meaning",
        "modules",
    }
    if set(payload) != expected_top:
        _fail(f"registry top-level keys differ: {sorted(set(payload) ^ expected_top)}")
    if payload["schema_version"] != "path_c_module_registry_v1":
        _fail("unsupported registry schema_version")
    allowed = payload["allowed_statuses"]
    if allowed != ["planned", "implemented", "tested", "frozen"]:
        _fail("allowed_statuses must be planned/implemented/tested/frozen in that order")
    source_git_sha = str(payload["source_git_sha"])
    if not GIT_SHA_RE.fullmatch(source_git_sha):
        _fail("source_git_sha must be a forty-character git SHA")
    test_execution_status = str(payload["test_execution_status"])
    if test_execution_status not in {"not_run", "passed", "failed"}:
        _fail("test_execution_status must be not_run, passed, or failed")

    modules = payload["modules"]
    if not isinstance(modules, list) or not modules:
        _fail("modules must be a non-empty list")
    registry_rows: dict[str, str] = {}
    for index, raw_module in enumerate(modules):
        module = _mapping(raw_module, f"modules[{index}]")
        expected_module = {"id", "status", "git_sha", "implementation_files", "test_ids"}
        if set(module) != expected_module:
            _fail(f"{module.get('id', index)} keys differ: {sorted(set(module) ^ expected_module)}")
        module_id = str(module["id"])
        status = str(module["status"])
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", module_id):
            _fail(f"malformed module id: {module_id}")
        if module_id in registry_rows:
            _fail(f"duplicate module id: {module_id}")
        if status not in allowed:
            _fail(f"invalid status for {module_id}: {status}")
        if status in {"tested", "frozen"} and test_execution_status != "passed":
            _fail(
                f"{module_id} cannot be {status} unless the bound test execution passed"
            )
        if status == "frozen" and "status: frozen" not in PREREGISTRATION.read_text(
            encoding="utf-8"
        ):
            _fail(
                f"{module_id} cannot be frozen while the canonical preregistration "
                "is not frozen"
            )
        if not GIT_SHA_RE.fullmatch(str(module["git_sha"])):
            _fail(f"{module_id} lacks a valid git SHA")
        if str(module["git_sha"]) != source_git_sha:
            _fail(f"{module_id} git SHA differs from registry source_git_sha")
        files = module["implementation_files"]
        if not isinstance(files, list):
            _fail(f"{module_id} implementation_files must be a list")
        if status != "planned" and not files:
            _fail(f"{module_id} must bind at least one implementation file")
        if len({str(relative) for relative in files}) != len(files):
            _fail(f"{module_id} contains duplicate implementation files")
        for relative in files:
            path = ROOT / str(relative)
            if not path.is_file():
                _fail(f"{module_id} implementation file does not exist: {relative}")
        test_ids = module["test_ids"]
        if not isinstance(test_ids, list):
            _fail(f"{module_id} test_ids must be a list")
        if status != "planned" and not test_ids:
            _fail(f"{module_id} must bind at least one test identifier")
        if len({str(test_id) for test_id in test_ids}) != len(test_ids):
            _fail(f"{module_id} contains duplicate test identifiers")
        for raw_test_id in test_ids:
            test_id = str(raw_test_id)
            if test_id.count("::") != 1:
                _fail(f"{module_id} has a malformed test identifier: {test_id}")
            relative, function_name = test_id.split("::", 1)
            test_path = ROOT / relative
            if not test_path.is_file():
                _fail(f"{module_id} test file does not exist: {relative}")
            if not re.fullmatch(r"test_[A-Za-z0-9_]+", function_name):
                _fail(f"{module_id} has a malformed test function: {function_name}")
            definition = re.compile(rf"^def\s+{re.escape(function_name)}\s*\(", re.MULTILINE)
            if not definition.search(test_path.read_text(encoding="utf-8")):
                _fail(f"{module_id} bound test function does not exist: {test_id}")
        registry_rows[module_id] = status

    if set(registry_rows) != REQUIRED_REVIEW_MODULE_IDS:
        _fail(
            "registry must contain one explicit row for every reviewed module; "
            f"missing={sorted(REQUIRED_REVIEW_MODULE_IDS - set(registry_rows))}, "
            f"extra={sorted(set(registry_rows) - REQUIRED_REVIEW_MODULE_IDS)}"
        )

    design_rows = _traceability_rows(
        DASHBOARD.read_text(encoding="utf-8"),
        source_name="dashboard",
    )
    if design_rows != registry_rows:
        missing = sorted(set(registry_rows) - set(design_rows))
        extra = sorted(set(design_rows) - set(registry_rows))
        mismatched = sorted(
            key for key in set(design_rows) & set(registry_rows)
            if design_rows[key] != registry_rows[key]
        )
        _fail(
            "dashboard traceability and registry differ: "
            f"missing={missing}, extra={extra}, status_mismatch={mismatched}"
        )
    if OPTIONAL_MODULE_DESIGN.is_file():
        module_design_rows = _traceability_rows(
            OPTIONAL_MODULE_DESIGN.read_text(encoding="utf-8"),
            source_name="module design",
        )
        if module_design_rows != registry_rows:
            _fail("optional module-design traceability and registry differ")

    preregistration_text = PREREGISTRATION.read_text(encoding="utf-8")
    for required in (
        "module_registry_sha256:",
        "evidence_spec_sha256:",
        "response_vocabulary_sha256:",
        "audit_battery_sha256:",
        "split_manifest_sha256:",
        "rng_key_schedule_version:",
    ):
        if required not in preregistration_text:
            _fail(f"preregistration lacks semantic binding {required}")

    print(
        f"Path C registry is statically consistent for {len(registry_rows)} modules; "
        f"test execution status is {test_execution_status}."
    )
    if test_execution_status == "not_run":
        print("No tests were executed and no tested/frozen status is inferred.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Path C registry check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
