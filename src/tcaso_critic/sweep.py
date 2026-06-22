from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from .canonical import canonical_hash, dump_json
from .diagnostics import diagnose_exact_matching
from .graph_builder import CertifiedGraphBuilder, GraphBuildError
from .jaxmarl_wrapper import JaxmarlOvercookedV2Backend, SourceBackendError
from .kappa import KappaComputer
from .matcher import ExactKappaMatcher
from .models import TaskSpec
from .public_edge_checker import CertifiedEdgeChecker
from .quotient import ContinuationQuotientComputer
from .reporting import make_summary, write_run_artifacts
from .state_codec import PublicStateCodec
from .tau_family import TauFamilyGenerator
from .validators import InvariantViolation


class SweepRunError(RuntimeError):
    pass


def run_one(cfg: dict[str, Any], out_dir: str) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    backend_name = cfg.get("backend", "jaxmarl_overcooked_v2")
    if backend_name != "jaxmarl_overcooked_v2":
        raise ValueError("Only backend='jaxmarl_overcooked_v2' is accepted for certification runs")
    layout_id = str(cfg["layout_id"])
    max_steps = int(cfg.get("max_steps", 8))
    max_depth = int(cfg.get("max_depth", 2))
    max_nodes = int(cfg.get("max_nodes", 5000))
    seed = int(cfg.get("seed", 0))
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    audited_agent_i = int(cfg.get("audited_agent_i", 0))
    step_mode = str(cfg.get("step_mode", "vectorized"))
    batch_size = int(cfg.get("batch_size", 512))
    backend = JaxmarlOvercookedV2Backend(
        layout_id=layout_id,
        max_steps=max_steps,
        env_kwargs=env_kwargs,
        jit_batch_step=bool(cfg.get("jit_batch_step", True)),
    )
    codec = PublicStateCodec(layout_id=layout_id, config_snapshot=backend.config_snapshot, reset_domain_tag=str(cfg.get("reset_domain_tag", "RESET_DEFAULT")))
    checker = CertifiedEdgeChecker(codec=codec, backend=backend)
    graph = CertifiedGraphBuilder(
        backend=backend,
        codec=codec,
        checker=checker,
        max_depth=max_depth,
        max_nodes=max_nodes,
        seed=seed,
        step_mode=step_mode,
        batch_size=batch_size,
    ).build_from_reset()
    initial_hash = next((h for h, d in graph.depths.items() if d == 0), None)
    if initial_hash is None:
        raise GraphBuildError("reset state not present in graph")
    task_specs = _task_specs_from_config(cfg, graph.nodes[initial_hash], audited_agent_i)
    if len(task_specs) != 1:
        raise ValueError("run_one expects exactly one tau; use run_sweep for tau families")
    task_spec = task_specs[0]
    return _analyze_graph_for_tau(
        cfg=cfg,
        out_dir=out_dir,
        backend_name=backend_name,
        backend=backend,
        graph=graph,
        task_spec=task_spec,
        audited_agent_i=audited_agent_i,
        max_depth=max_depth,
    )


def run_sweep(cfg: dict[str, Any], out_root: str) -> dict[str, Any]:
    os.makedirs(out_root, exist_ok=True)
    backend_name = cfg.get("backend", "jaxmarl_overcooked_v2")
    if backend_name != "jaxmarl_overcooked_v2":
        raise ValueError("Only backend='jaxmarl_overcooked_v2' is accepted for certification sweeps")
    layouts = list(cfg.get("layouts", [])) or [str(cfg.get("layout_id", "cramped_room"))]
    depths = [int(x) for x in cfg.get("max_depths", [int(cfg.get("max_depth", 2))])]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for layout_id in layouts:
        for max_depth in depths:
            run_cfg = dict(cfg)
            run_cfg["layout_id"] = layout_id
            run_cfg["max_depth"] = max_depth
            run_cfg.pop("layouts", None)
            run_cfg.pop("max_depths", None)
            try:
                rows.extend(_run_layout_depth_tau_family(run_cfg, out_root))
            except Exception as exc:  # noqa: BLE001 - sweep-level explicit failure record only
                failure = {
                    "layout_id": layout_id,
                    "max_depth": max_depth,
                    "status": "GATE3_SWEEP_LAYOUT_DEPTH_FAILED_BEFORE_CERTIFICATE_WRITE",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(failure)
                dump_json(os.path.join(out_root, f"FAILED_{layout_id}_d{max_depth}.json"), failure)
    aggregate = _aggregate(rows, failures)
    dump_json(os.path.join(out_root, "aggregate_summary.json"), aggregate)
    with open(os.path.join(out_root, "AGGREGATE_GATE3_SWEEP_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(_render_aggregate_report(aggregate, rows, failures))
    return aggregate


def _run_layout_depth_tau_family(cfg: dict[str, Any], out_root: str) -> list[dict[str, Any]]:
    layout_id = str(cfg["layout_id"])
    max_depth = int(cfg["max_depth"])
    max_steps = int(cfg.get("max_steps", max(8, max_depth + 3)))
    max_nodes = int(cfg.get("max_nodes", 5000))
    seed = int(cfg.get("seed", 0))
    audited_agent_i = int(cfg.get("audited_agent_i", 0))
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    step_mode = str(cfg.get("step_mode", "vectorized"))
    batch_size = int(cfg.get("batch_size", 512))
    backend = JaxmarlOvercookedV2Backend(
        layout_id=layout_id,
        max_steps=max_steps,
        env_kwargs=env_kwargs,
        jit_batch_step=bool(cfg.get("jit_batch_step", True)),
    )
    codec = PublicStateCodec(layout_id=layout_id, config_snapshot=backend.config_snapshot, reset_domain_tag=str(cfg.get("reset_domain_tag", "RESET_DEFAULT")))
    checker = CertifiedEdgeChecker(codec=codec, backend=backend)
    graph = CertifiedGraphBuilder(
        backend=backend,
        codec=codec,
        checker=checker,
        max_depth=max_depth,
        max_nodes=max_nodes,
        seed=seed,
        step_mode=step_mode,
        batch_size=batch_size,
    ).build_from_reset()
    initial_hash = next((h for h, d in graph.depths.items() if d == 0), None)
    if initial_hash is None:
        raise GraphBuildError("reset state not present in graph")
    task_specs = _task_specs_from_config(cfg, graph.nodes[initial_hash], audited_agent_i)
    rows: list[dict[str, Any]] = []
    for task_spec in task_specs:
        run_dir = os.path.join(out_root, _safe_name(f"{layout_id}_d{max_depth}_{task_spec.tau_id}"))
        row = _analyze_graph_for_tau(
            cfg=cfg,
            out_dir=run_dir,
            backend_name=str(cfg.get("backend", "jaxmarl_overcooked_v2")),
            backend=backend,
            graph=graph,
            task_spec=task_spec,
            audited_agent_i=audited_agent_i,
            max_depth=max_depth,
        )
        rows.append(row)
    return rows


def _task_specs_from_config(cfg: dict[str, Any], initial_public, audited_agent_i: int) -> list[TaskSpec]:
    if "tau" in cfg and cfg["tau"]:
        tau = cfg["tau"]
        return [TaskSpec(tau_id=str(tau["tau_id"]), task_type=str(tau["task_type"]), params=dict(tau.get("params", {})))]
    requested = cfg.get("tau_family", {}).get("include") if isinstance(cfg.get("tau_family"), dict) else cfg.get("tau_family")
    return TauFamilyGenerator(audited_agent_i=audited_agent_i).from_initial_state(initial_public, requested=requested)


def _analyze_graph_for_tau(
    *,
    cfg: dict[str, Any],
    out_dir: str,
    backend_name: str,
    backend: Any,
    graph,
    task_spec: TaskSpec,
    audited_agent_i: int,
    max_depth: int,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    qc = ContinuationQuotientComputer(graph=graph, task_spec=task_spec, audited_agent_i=audited_agent_i)
    quotient = qc.compute()
    kappa = KappaComputer(graph=graph, task_spec=task_spec)
    matcher = ExactKappaMatcher(quotient_computer=qc, quotient=quotient, kappa=kappa, audited_agent_i=audited_agent_i, action_count=backend.num_actions)
    probes, kappa_signatures = matcher.build_probe_records()
    probes, mccs, socs = matcher.exact_match(probes)
    run_id = canonical_hash({"cfg": cfg, "tau": asdict(task_spec), "nodes": len(graph.nodes), "edges": len(graph.edges)}, prefix="run")
    summary = make_summary(run_id=run_id, backend=backend_name, layout_id=str(cfg["layout_id"]), tau_id=task_spec.tau_id, max_depth=max_depth, graph=graph, quotient=quotient, probes=probes, matched_controls=mccs, same_override=socs)
    diagnostics = diagnose_exact_matching(probes=probes, matched_controls=mccs, kappa_signatures=kappa_signatures, num_tau_states=len(quotient.target_states), max_depth=max_depth)
    write_run_artifacts(out_dir, summary=summary, graph=graph, quotient=quotient, probes=probes, matched_controls=mccs, same_override=socs, kappa_signatures=kappa_signatures, diagnostics=diagnostics)
    dump_json(os.path.join(out_dir, "config_effective.json"), {**cfg, "tau": asdict(task_spec)})
    dump_json(os.path.join(out_dir, "env_config_snapshot.json"), asdict(backend.config_snapshot))
    return {
        **asdict(summary),
        "diagnostic_primary_label": diagnostics.get("diagnostic_primary_label"),
        "diagnostic_notes": diagnostics.get("notes"),
        "run_dir": out_dir,
    }


def _aggregate(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    exact_nonempty = [r for r in rows if int(r.get("num_exact_matched_controls", 0)) > 0]
    same_override_nonempty = [r for r in rows if int(r.get("num_same_override_counterexamples", 0)) > 0]
    labels: dict[str, int] = {}
    for r in rows:
        label = str(r.get("diagnostic_primary_label"))
        labels[label] = labels.get(label, 0) + 1
    status = "GATE3_SWEEP_COMPLETED"
    if failures and not rows:
        status = "GATE3_SWEEP_FAILED_BEFORE_ANY_RUN"
    elif not exact_nonempty:
        status = "GATE3_SWEEP_COMPLETED_EXACT_MATCHING_EMPTY_ACROSS_RUNS"
    return {
        "status": status,
        "num_runs": len(rows),
        "num_failures": len(failures),
        "num_runs_with_exact_matches": len(exact_nonempty),
        "num_runs_with_certified_same_override": len(same_override_nonempty),
        "diagnostic_label_histogram": labels,
        "claim_boundary": {
            "matched_control_pool_certified": bool(exact_nonempty),
            "valid_semantic_recovery_probe_claimed": False,
            "pilot_or_benchmark_claimed": False,
        },
    }


def _render_aggregate_report(aggregate: dict[str, Any], rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    lines = [
        "# TCASO-CRITIC Gate 3 Depth-2/3 Vectorized Sweep Report",
        "",
        "## Status",
        "",
        "```text",
        str(aggregate["status"]),
        "```",
        "",
        f"- runs completed: `{aggregate['num_runs']}`",
        f"- layout/depth failures: `{aggregate['num_failures']}`",
        f"- runs with exact matches: `{aggregate['num_runs_with_exact_matches']}`",
        f"- runs with certified same-override records: `{aggregate['num_runs_with_certified_same_override']}`",
        "",
        "## Diagnostic labels",
        "",
    ]
    if aggregate.get("diagnostic_label_histogram"):
        for k, v in aggregate["diagnostic_label_histogram"].items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")
    if rows:
        lines += ["", "## Run table", ""]
        for r in rows:
            lines.append(
                f"- `{r['layout_id']}` depth `{r['max_depth']}` tau `{r['tau_id']}`: "
                f"nodes={r['num_nodes']}, edges={r['num_edges']}, positives={r['num_positive_candidates']}, "
                f"exact={r['num_exact_matched_controls']}, label=`{r.get('diagnostic_primary_label')}`"
            )
    if failures:
        lines += ["", "## Failures", ""]
        for f in failures:
            lines.append(f"- `{f['layout_id']}` depth `{f['max_depth']}`: `{f['exception_type']}` {f['message']}")
    lines += [
        "",
        "## Claim boundary",
        "",
        "No policy result, pilot result, benchmark result, or ICLR-ready evidence is claimed by this sweep.",
        "",
    ]
    return "\n".join(lines)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)[:180]
