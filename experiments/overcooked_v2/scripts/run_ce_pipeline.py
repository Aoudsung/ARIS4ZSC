"""Step 2-3: CE collect → estimate → graph build → coverage check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.overcooked_v2.ce_sampler import (
    collect_option_replay,
    estimate_empirical_ce,
    option_kind_stats,
    refine_empirical_ce,
    replay_coverage,
    replay_coverage_gate,
    save_replay_npz,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.event_extractor import EVENT_SEMANTICS_VERSION
from experiments.overcooked_v2.graph_builder import (
    build_support_graph,
    validate_task_stage_coverage,
)
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    runtime_provenance,
    sha256_file,
    stamp_graph_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ocv2_step4.yaml"
LEGACY_P0_OUTPUT = REPO_ROOT / "outputs" / "p0_verify"


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)
    layout = str(config["layout"])
    train_cfg = config.get("training", {})
    cost_coef = float(train_cfg["cost_coef"])
    shaped_reward_coef = float(train_cfg["shaped_reward_coef"])
    cost_per_step = float(train_cfg["cost_per_step"])
    reward_metadata = {
        "layout": layout,
        "cost_coef": cost_coef,
        "cost_per_step": cost_per_step,
        "shaped_reward_coef": shaped_reward_coef,
        "reward_scale_source": "config.training",
        "event_semantics_version": int(EVENT_SEMANTICS_VERSION),
    }

    output_dir = Path(args.output_dir)
    if output_dir.resolve() == LEGACY_P0_OUTPUT.resolve():
        raise ValueError(
            "Refusing to write CE outputs to outputs/p0_verify; run7 depends on that "
            "legacy directory. Pass --output_dir with a new versioned path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Setup ===")
    env_cfg = config.get("env", {})
    env = OCV2Adapter(
        layout,
        max_steps=int(env_cfg.get("max_steps", 200)),
        observation_type=str(env_cfg.get("observation_type", "default")),
        force_path_planning=bool(env_cfg.get("force_path_planning", False)),
    )
    lg = parse_layout(env.env, layout)
    lib = OCV2OptionLibrary(
        lg,
        max_option_steps=int(config.get("options", {}).get("max_option_steps", 6)),
    )
    partners = make_training_partners(lib)

    print(f"Options: {lib.num_options}")
    for opt in lib.options:
        print(f"  {opt.id}: {opt.kind}")

    ep_per_partner = 100
    print(f"\n=== Phase 2: CE Collect ({ep_per_partner} episodes × {len(partners)} partners, sequential) ===")
    rows = collect_option_replay(
        env,
        partners,
        lib,
        layout_name=layout,
        episodes=ep_per_partner,
        max_options_per_episode=200,
        seed=42,
        gamma=0.99,
        horizon_options=5,
        cost_per_step=cost_per_step,
        cost_coef=cost_coef,
        shaped_reward_coef=shaped_reward_coef,
    )
    print(f"Collected {len(rows)} option replay rows")

    # Coverage check
    coverage = replay_coverage(rows, lib.options)
    print("\n--- Replay Coverage ---")
    for key, value in coverage.items():
        marker = "OK" if value > 0 else "MISSING"
        print(f"  {key}: {value} [{marker}]")

    try:
        gate = replay_coverage_gate(coverage, require_full_task_coverage=True)
        print(f"Coverage gate: {gate['status']}")
    except RuntimeError as exc:
        print(f"Coverage gate FAILED: {exc}")
        gate = {"status": "failed", "error": str(exc)}

    replay_path = output_dir / "replay.npz"
    replay_metadata = {
        **reward_metadata,
        "num_rows": len(rows),
        "episodes_per_partner": ep_per_partner,
        "partners": [partner.name for partner in partners],
        "coverage": coverage,
        "coverage_gate": gate,
        "option_kind_stats": option_kind_stats(rows, lib.options),
    }
    save_replay_npz(replay_path, rows, replay_metadata)
    replay_sha256 = sha256_file(replay_path)
    if gate.get("status") != "passed":
        sys.exit(1)

    # Option kind stats
    stats = replay_metadata["option_kind_stats"]
    print("\n--- Option Kind Stats ---")
    for kind, info in sorted(stats.items()):
        print(
            f"  {kind:30s}: attempts={info['attempt_count']:5d} "
            f"success={info['success_count']:5d} "
            f"rate={info['success_rate']:.2%} "
            f"timeout={info['timeout_count']:5d}"
        )

    print("\n=== Phase 3: CE Estimate ===")
    ce = estimate_empirical_ce(rows, lib.num_options, min_weight=20.0)
    ce_path = output_dir / "ce_matrix.npy"
    np.save(ce_path, ce)
    ce_sha256 = sha256_file(ce_path)

    print("CE matrix (non-zero entries):")
    for i in range(ce.shape[0]):
        for j in range(ce.shape[1]):
            if ce[i, j] > 0.01:
                oi = lib.options[i].kind if i < lib.num_options else f"opt{i}"
                oj = lib.options[j].kind if j < lib.num_options else f"opt{j}"
                print(f"  CE({i}:{oi}, {j}:{oj}) = {ce[i, j]:.4f}")

    # Refine
    refined, refine_meta = refine_empirical_ce(ce, rows, lib.num_options, top_k=32, min_weight=20.0)
    refined_path = output_dir / "ce_refined.npy"
    np.save(refined_path, refined)
    refined_sha256 = sha256_file(refined_path)
    sidecar_path = output_dir / "ce_refined.meta.json"
    sidecar_path.write_text(
        json.dumps(
            {
                **reward_metadata,
                **refine_meta,
                "provenance": {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "replay_sha256": replay_sha256,
                    "ce_matrix_input_sha256": ce_sha256,
                    "ce_matrix_sha256": refined_sha256,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved CE metadata sidecar to {sidecar_path}")

    print("\n=== Phase 4: Graph Build ===")
    graph = build_support_graph(
        layout,
        lib.options,
        refined,
        eta=0.05,
        max_factors=16,
    )
    graph.metadata = {
        **(graph.metadata or {}),
        **reward_metadata,
        "coverage_gate": "pending",
        "formal_graph": True,
        "provenance": {
            **((graph.metadata or {}).get("provenance", {})),
            **runtime_provenance(
                config=config,
                layout_graph=lg,
                option_lib=lib,
                partners=partners,
                repo_root=REPO_ROOT,
                ce_path=refined_path,
                replay_path=replay_path,
            ),
        },
    }
    print(f"Graph: {len(graph.factors)} factors")
    factor_kinds_in_graph = set()
    for factor in graph.factors:
        oi = lib.options[factor.option_i] if factor.option_i < lib.num_options else None
        oj = lib.options[factor.option_j] if factor.option_j < lib.num_options else None
        oi_kind = oi.kind if oi else f"opt{factor.option_i}"
        oj_kind = oj.kind if oj else f"opt{factor.option_j}"
        factor_kinds_in_graph.add(oi_kind)
        factor_kinds_in_graph.add(oj_kind)
        print(
            f"  F{factor.id}: ({factor.option_i}:{oi_kind}, {factor.option_j}:{oj_kind}) "
            f"CE={factor.ce_score:.4f} kind={factor.factor_kind} modes={factor.num_modes}"
        )

    print(f"\nOption kinds in graph: {sorted(factor_kinds_in_graph)}")

    # Task stage coverage
    print("\n=== Phase 5: Task Stage Coverage Gate ===")
    required = {"deliver_ingredient_to_pot", "plate_soup", "serve_soup"}
    present = required & factor_kinds_in_graph
    missing = required - factor_kinds_in_graph
    print(f"Required: {sorted(required)}")
    print(f"Present:  {sorted(present)}")
    print(f"Missing:  {sorted(missing)}")

    task_stage_coverage_ok = True
    if args.require_task_stage_coverage:
        try:
            validate_task_stage_coverage(graph)
            print("PASS: Task stage coverage validated")
            graph.metadata["coverage_gate"] = "passed"
        except RuntimeError as exc:
            task_stage_coverage_ok = False
            graph.metadata["coverage_gate"] = "failed"
            print(f"FAIL: {exc}")
    else:
        graph.metadata["coverage_gate"] = "not_required"

    if not task_stage_coverage_ok:
        if args.allow_incomplete_graph:
            debug_graph = graph.to_json_dict()
            debug_graph.setdefault("metadata", {}).update(
                {"formal_graph": False, "coverage_gate": "failed"}
            )
            debug_path = output_dir / "debug_incomplete_graph.json"
            debug_path.write_text(json.dumps(debug_graph, indent=2), encoding="utf-8")
            print(f"Saved DEBUG-ONLY graph to {debug_path}")
        sys.exit(1)

    # Save graph
    graph_path = output_dir / "graph.json"
    graph_json = stamp_graph_hash(graph.to_json_dict())
    graph.metadata = graph_json["metadata"]
    graph_path.write_text(json.dumps(graph_json, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"Replay rows: {len(rows)}")
    print(f"CE factors: {len(graph.factors)}")
    print(f"Replay coverage gate: {gate.get('status', 'unknown')}")
    print(f"plate_soup in CE graph: {'plate_soup' in factor_kinds_in_graph}")
    print(f"serve_soup in CE graph: {'serve_soup' in factor_kinds_in_graph}")
    print(f"deliver_ingredient_to_pot in CE graph: {'deliver_ingredient_to_pot' in factor_kinds_in_graph}")
    print(f"Outputs saved to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_dir", default="outputs/p1_verify")
    parser.add_argument("--allow_incomplete_graph", action="store_true")
    parser.add_argument(
        "--require_task_stage_coverage",
        dest="require_task_stage_coverage",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_require_task_stage_coverage",
        dest="require_task_stage_coverage",
        action="store_false",
    )
    return parser


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


if __name__ == "__main__":
    main()
