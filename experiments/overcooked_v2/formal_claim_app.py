"""Closed-hierarchy synthesis of the three primary DELTA hypotheses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from src.delta_zsc.config import LAYOUTS, METHOD_VERSION
from src.delta_zsc.storage import read_json, write_json


def _layout_sources(values: list[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        try:
            layout, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{label} inputs use LAYOUT=/path/to/artifact.json") from error
        if layout not in LAYOUTS or layout in result:
            raise ValueError(f"{label} must cover each registered layout exactly once.")
        result[layout] = Path(raw_path).resolve()
    if set(result) != set(LAYOUTS):
        raise ValueError(f"{label} must cover {LAYOUTS}.")
    return result


def _source(path: Path) -> Mapping[str, str]:
    return {"path": str(path)}


def build_formal_claim_report(args: argparse.Namespace) -> None:
    official_paths = _layout_sources(args.official_summary, label="Official summary")
    development_paths = _layout_sources(
        args.development_summary, label="Development summary"
    )
    intervention_paths = _layout_sources(
        args.belief_intervention, label="Belief intervention"
    )
    diagnostic_paths = _layout_sources(
        args.posterior_diagnostics, label="Posterior diagnostics"
    )
    resource_path = Path(args.resource_report).resolve()

    official: dict[str, Mapping[str, Any]] = {}
    development: dict[str, Mapping[str, Any]] = {}
    intervention: dict[str, Mapping[str, Any]] = {}
    diagnostics: dict[str, Mapping[str, Any]] = {}
    for layout in LAYOUTS:
        official[layout] = read_json(official_paths[layout])
        development[layout] = read_json(development_paths[layout])
        intervention[layout] = read_json(intervention_paths[layout])
        diagnostics[layout] = read_json(diagnostic_paths[layout])
        if (
            official[layout].get("artifact_type") != "delta_official_summary"
            or official[layout].get("version") != 3
            or official[layout].get("method") != METHOD_VERSION
            or official[layout].get("execution_mode") != "active"
            or official[layout].get("layout") != layout
        ):
            raise ValueError(f"Official summary identity differs on {layout}.")
        if (
            development[layout].get("artifact_type") != "delta_development_summary"
            or development[layout].get("version") != 2
            or development[layout].get("method") != METHOD_VERSION
            or development[layout].get("layout") != layout
        ):
            raise ValueError(f"Development summary identity differs on {layout}.")
        if (
            intervention[layout].get("artifact_type")
            != "delta_belief_value_intervention"
            or intervention[layout].get("version") != 2
            or intervention[layout].get("method") != METHOD_VERSION
            or intervention[layout].get("execution_mode") != "active"
            or intervention[layout].get("layout") != layout
        ):
            raise ValueError(f"Belief intervention identity differs on {layout}.")
        if (
            diagnostics[layout].get("artifact_type")
            != "delta_v6_posterior_predictive_diagnostics"
            or diagnostics[layout].get("version") != 6
            or diagnostics[layout].get("method") != METHOD_VERSION
            or diagnostics[layout].get("layout") != layout
            or diagnostics[layout].get("claim_role") != "diagnostic_only"
        ):
            raise ValueError(f"Posterior diagnostic identity differs on {layout}.")

    resource_report = read_json(resource_path)
    if (
        resource_report.get("artifact_type") != "delta_resource_report"
        or resource_report.get("version") != 1
        or not isinstance(resource_report.get("methods"), list)
        or not resource_report.get("methods")
    ):
        raise ValueError("Resource-report identity differs.")

    performance_evidence = {
        layout: {
            "strongest_baseline": official[layout]["strongest_baseline"],
            "strongest_contrast": official[layout]["delta_active_vs_strongest"],
            "all_contrasts": official[layout]["delta_active_vs_each_baseline"],
        }
        for layout in LAYOUTS
    }
    h1_by_layout = {
        layout: bool(
            official[layout]["all_baselines_material_superiority_gate"] is True
        )
        for layout in LAYOUTS
    }
    h1_pass = all(h1_by_layout.values())

    decision_evidence = {
        layout: development[layout]["primary_contrasts"]["decision_emission"]
        for layout in LAYOUTS
    }
    h2_by_layout = {
        layout: bool(float(decision_evidence[layout]["interval_95"][0]) > 0.0)
        for layout in LAYOUTS
    }
    h2_raw_pass = all(h2_by_layout.values())

    intervention_evidence = {
        layout: intervention[layout] for layout in LAYOUTS
    }
    h3_by_layout = {
        layout: bool(float(intervention_evidence[layout]["one_sided_lcb"]) > 0.0)
        for layout in LAYOUTS
    }
    h3_raw_pass = all(h3_by_layout.values())

    active_voi = {
        layout: development[layout]["primary_contrasts"]["active_voi"]
        for layout in LAYOUTS
    }
    claims = {
        "H1_performance": {
            "statement": (
                "Frozen DELTA-active exceeds every registered same-protocol "
                "baseline by at least one delivery on both Official layouts."
            ),
            "eligible": True,
            "layout_passes": h1_by_layout,
            "passed": h1_pass,
            "evidence": performance_evidence,
        },
        "H2_decision_emission": {
            "statement": (
                "Adding the sparse counterfactual decision emission improves XP "
                "over response-only latent inference on both layouts."
            ),
            "eligible": h1_pass,
            "layout_raw_test_passes": h2_by_layout,
            "raw_test_passed": h2_raw_pass,
            "passed": bool(h1_pass and h2_raw_pass),
            "evidence": decision_evidence,
        },
        "H3_causal_belief_value": {
            "statement": (
                "The correct legal-history belief selects higher-value actions than "
                "a shuffled belief in the same source world on both layouts."
            ),
            "eligible": bool(h1_pass and h2_raw_pass),
            "layout_raw_test_passes": h3_by_layout,
            "raw_test_passed": h3_raw_pass,
            "passed": bool(h1_pass and h2_raw_pass and h3_raw_pass),
            "evidence": intervention_evidence,
        },
    }
    write_json(
        args.output,
        {
            "version": 3,
            "artifact_type": "delta_formal_claim_report",
            "method": METHOD_VERSION,
            "layouts": list(LAYOUTS),
            "claims": claims,
            "secondary_active_voi_increment": active_voi,
            "posterior_predictive_diagnostics": diagnostics,
            "resource_report": resource_report,
            "testing_rule": "closed_hierarchy_H1_then_H2_then_H3_across_both_layouts",
            "sources": {
                "official": {
                    layout: _source(official_paths[layout]) for layout in LAYOUTS
                },
                "development": {
                    layout: _source(development_paths[layout]) for layout in LAYOUTS
                },
                "belief_intervention": {
                    layout: _source(intervention_paths[layout]) for layout in LAYOUTS
                },
                "posterior_diagnostics": {
                    layout: _source(diagnostic_paths[layout]) for layout in LAYOUTS
                },
                "resource_report": _source(resource_path),
            },
        },
    )


__all__ = ["build_formal_claim_report"]
