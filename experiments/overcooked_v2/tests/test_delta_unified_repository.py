from __future__ import annotations

from pathlib import Path
import ast
import re


ACTIVE_APPS = {
    "__init__.py",
    "baseline_app.py",
    "calibration_app.py",
    "delta_manifest_app.py",
    "delta_zsc.py",
    "deployment.py",
    "development_matrix_app.py",
    "evaluation_app.py",
    "formal_claim_app.py",
    "intervention_app.py",
    "official_adapter.py",
    "official_policy.py",
    "official_training.py",
    "resource_report_app.py",
    "training_app.py",
    "upstream_app.py",
    "upstream_pipeline_app.py",
}


def test_active_namespace_contains_no_retired_patch_chain() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/delta_zsc").glob("*.py")
    )
    retired = {
        "pair_comparator",
        "separation_margin",
        "context_dropout",
        "capability_consistency_weight",
        "posterior_decision_weight",
        "decision_policy_weight",
        "gradient_routing",
        "decision_regret",
        "response_cross_log_likelihood",
    }
    assert not any(token in text for token in retired)


def test_active_tree_contains_only_unified_method_and_registered_apps() -> None:
    assert not Path("src/path_c").exists()
    assert not Path("analysis").exists()
    assert {path.name for path in Path("experiments/overcooked_v2").glob("*.py")} == ACTIVE_APPS
    assert Path("legacy/implementation_v8/src/path_c/model.py").is_file()
    assert Path("legacy/implementation_v8/experiments/overcooked_v2/path_c.py").is_file()
    assert Path(
        "legacy/implementation_v8/experiments/overcooked_v2/configs/depi_simple_formal.yaml"
    ).is_file()
    assert Path(
        "legacy/implementation_v8/analysis/l0_dimension_localization/l0_nn_partition_discriminant.py"
    ).is_file()
    assert Path("docs/legacy/v8/status/DECISION_LOG.md").is_file()
    assert Path("docs/legacy/v8/status/EVIDENCE_LEDGER.md").is_file()
    assert {path.name for path in Path("docs/status").glob("*.md")} == {"DASHBOARD.md"}


def test_active_experiment_apps_import_only_unified_method_namespace() -> None:
    for name in ACTIVE_APPS - {"__init__.py"}:
        text = Path("experiments/overcooked_v2", name).read_text(encoding="utf-8")
        assert "src.path_c" not in text


def test_pyproject_packages_only_active_namespace() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "src.delta_zsc*" in text
    assert "src.path_c*" not in text



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

def test_only_unified_workflow_is_active_and_method_identity_is_v4() -> None:
    workflows = {path.name for path in Path(".github/workflows").glob("*.yml")}
    assert workflows == {"delta-unified-ci.yml"}
    assert Path("legacy/implementation_v8/.github/workflows/depi-ci.yml").is_file()
    from src.delta_zsc.config import METHOD_VERSION

    assert METHOD_VERSION == "delta_belief_conditioned_raw_return_pairwise_crn_v5"


def test_authoritative_docs_describe_only_exact_bayes_voi() -> None:
    active = (
        Path("README.md"),
        Path("docs/SCIENTIFIC_SPEC.md"),
        Path("docs/METHOD_SPEC.md"),
        Path("docs/THEORY.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/EVALUATION_SPEC.md"),
        Path("docs/FORMAL_EXPERIMENT_PROTOCOL.md"),
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in active).lower()
    assert "expected cross log likelihood" not in text
    assert "voi_component" not in text
    assert "halton" not in text
    assert "66" in text
    assert "exact" in text
    assert "episode-static" in text
    assert "centered residual" in text
    assert "delayed" in text
