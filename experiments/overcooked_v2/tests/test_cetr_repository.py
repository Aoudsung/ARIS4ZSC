from __future__ import annotations

import ast
from pathlib import Path
import re


ACTIVE_APPS = {
    "__init__.py",
    "baseline_app.py",
    "cetr_zsc.py",
    "claim_app.py",
    "deployment.py",
    "evaluation_app.py",
    "official_adapter.py",
    "official_policy.py",
    "official_training.py",
    "partner_manifest_app.py",
    "reference_sp_app.py",
    "resource_report_app.py",
    "training_app.py",
    "upstream_app.py",
    "upstream_pipeline_app.py",
}

FORBIDDEN_CETR_TOKENS = {
    "pair_comparator",
    "separation_margin",
    "context_dropout",
    "capability_consistency_weight",
    "posterior_decision_weight",
    "decision_policy_weight",
    "gradient_routing",
    "decision_regret",
    "response_cross_log_likelihood",
    "latent",
    "posterior",
    "belief",
    "mirror",
    "voi",
    "anchor",
    "successor",
    "residual",
    "probe",
    "semantic_initializer",
    "execution_mode",
    "response_model",
    "delta",
}



def test_cetr_source_contains_no_retired_method_terms() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/cetr_zsc").glob("*.py")
    )
    assert not any(token in text for token in FORBIDDEN_CETR_TOKENS)


def test_retired_namespaces_and_directories_are_absent() -> None:
    for path in (
        Path("src/path_c"),
        Path("analysis"),
        Path("legacy"),
        Path("docs/legacy"),
        Path("src/delta_zsc"),
    ):
        assert not path.exists()


def test_active_application_and_status_file_sets_are_exact() -> None:
    assert {path.name for path in Path("experiments/overcooked_v2").glob("*.py")} == ACTIVE_APPS
    assert {path.name for path in Path("docs/status").glob("*.md")} == {"DASHBOARD.md"}


def test_active_apps_contain_no_retired_import_namespaces() -> None:
    for name in ACTIVE_APPS - {"__init__.py"}:
        text = Path("experiments/overcooked_v2", name).read_text(encoding="utf-8")
        assert "src.path_c" not in text
        assert "src.delta_zsc" not in text


def test_pyproject_packages_only_active_namespaces() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "src.cetr_zsc*" in text
    assert "src.delta_zsc" not in text


def test_requirements_match_pyproject_runtime_dependencies_exactly() -> None:
    source = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^dependencies\s*=\s*(\[.*?^\])", source)
    assert match is not None
    project_dependencies = set(ast.literal_eval(match.group(1)))
    requirements = {
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert requirements == project_dependencies


def test_only_cetr_workflow_is_active_and_method_identity_is_cetr() -> None:
    workflows = {path.name for path in Path(".github/workflows").glob("*.yml")}
    assert workflows == {"cetr-ci.yml"}
    from src.cetr_zsc.config import METHOD_VERSION

    assert METHOD_VERSION == "constrained_episodic_tail_robust_zsc_v1"


def test_authoritative_docs_describe_the_cetr_protocol() -> None:
    active = (
        Path("README.md"),
        Path("docs/SCIENTIFIC_SPEC.md"),
        Path("docs/METHOD_SPEC.md"),
        Path("docs/THEORY.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/RESEARCH_PLAN.md"),
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in active).lower()
    for marker in (
        "lower-half",
        "cross-fitting",
        "dual",
        "self-play",
        "lineage-disjoint",
        "constrained_episodic_tail_robust_zsc_v1",
        "version: 6",
    ):
        assert marker in text
    assert "expected cross log likelihood" not in text
    assert "voi_component" not in text
    assert "halton" not in text
