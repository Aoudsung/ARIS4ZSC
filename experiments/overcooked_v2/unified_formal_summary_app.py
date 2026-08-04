"""Two-layout confirmatory synthesis for unified DELTA-ZSC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.path_c.storage import runtime_provenance, sha256_path


LAYOUTS = ("test_time_simple", "test_time_wide")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _load(path: Path, *, expected_layout: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("version") != 2
        or payload.get("artifact_type")
        != "unified_delta_three_hypothesis_summary"
        or payload.get("layout") != expected_layout
    ):
        raise ValueError(
            f"Layout summary identity differs for {expected_layout}: {path}"
        )
    hierarchy = payload.get("hypothesis_hierarchy")
    if not isinstance(hierarchy, Mapping) or set(hierarchy) != {
        "H1_performance",
        "H2_decision_emission",
        "H3_causal_belief_value",
    }:
        raise ValueError("Layout summary hypothesis hierarchy is malformed.")
    return payload


def run_formal_summary(args: Any) -> None:
    supplied = {}
    for value in args.layout_summary:
        try:
            layout, raw = value.split("=", 1)
        except ValueError as error:
            raise ValueError(
                "Layout summary inputs use LAYOUT=/path/unified_delta_summary.json."
            ) from error
        if layout in supplied or layout not in LAYOUTS:
            raise ValueError("Layout summaries must cover each registered layout once.")
        supplied[layout] = Path(raw).resolve()
    if set(supplied) != set(LAYOUTS):
        raise ValueError("Formal synthesis requires simple and wide summaries.")
    summaries = {
        layout: _load(path, expected_layout=layout)
        for layout, path in supplied.items()
    }
    h1_by_layout = {
        layout: bool(
            payload["hypothesis_hierarchy"]["H1_performance"]["passed"]
        )
        for layout, payload in summaries.items()
    }
    h1_passed = all(h1_by_layout.values())
    h2_local = {
        layout: bool(
            payload["hypothesis_hierarchy"]["H2_decision_emission"]["passed"]
        )
        for layout, payload in summaries.items()
    }
    h2_passed = h1_passed and all(h2_local.values())
    h3_local = {
        layout: bool(
            payload["hypothesis_hierarchy"]["H3_causal_belief_value"]["passed"]
        )
        for layout, payload in summaries.items()
    }
    h3_passed = h2_passed and all(h3_local.values())
    hierarchy = {
        "H1_performance": {
            "by_layout": h1_by_layout,
            "passed": h1_passed,
        },
        "H2_decision_emission": {
            "by_layout": h2_local,
            "confirmatory_unlocked": h1_passed,
            "confirmatory_passed": h2_passed,
        },
        "H3_causal_belief_value": {
            "by_layout": h3_local,
            "confirmatory_unlocked": h2_passed,
            "confirmatory_passed": h3_passed,
        },
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "unified_delta_formal_summary.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_two_layout_formal_summary",
            "layouts": {
                layout: {
                    "method_means": payload["method_means"],
                    "comparisons": payload["comparisons"],
                    "hypothesis_hierarchy": payload["hypothesis_hierarchy"],
                    "internal_identity": payload["internal_identity"],
                }
                for layout, payload in summaries.items()
            },
            "formal_hypothesis_hierarchy": hierarchy,
            "all_three_confirmatory_hypotheses_passed": h3_passed,
            "sources": {
                layout: {"path": str(path), "sha256": sha256_path(path)}
                for layout, path in supplied.items()
            },
            "repository_runtime": runtime_provenance(),
        },
    )


__all__ = ["LAYOUTS", "run_formal_summary"]
