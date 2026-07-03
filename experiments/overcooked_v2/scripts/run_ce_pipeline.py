"""Step 2-3: CE collect → estimate → graph build → coverage check."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.overcooked_v2.ce_sampler import (
    collect_option_replay,
    estimate_empirical_ce_with_support,
    load_replay_npz,
    option_kind_stats,
    refine_empirical_ce,
    replay_coverage,
    replay_coverage_gate,
    save_replay_npz,
)
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.event_extractor import (
    EVENT_SEMANTICS_VERSION,
    PARTNER_OPTION_EVIDENCE_POLICY,
    sparse_credit_params,
)
from experiments.overcooked_v2.graph_builder import (
    build_support_graph,
    validate_task_stage_coverage,
)
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.reward_design import terminal_progress_params
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
    graph_cfg = config.get("graph", {}) or {}
    cost_coef = float(train_cfg["cost_coef"])
    shaped_reward_coef = float(train_cfg["shaped_reward_coef"])
    cost_per_step = float(train_cfg["cost_per_step"])
    # RC root-cause fix: the actor-specific sparse-credit mode is part of the CE
    # objective. Recorded into the graph metadata so train_aris rejects a graph
    # built under a different credit objective (see _enforce_graph_objective_metadata).
    credit_params = sparse_credit_params(train_cfg)
    if (
        credit_params["mode"] == "role_contrib_team"
        and not bool(train_cfg.get("oracle_role_conditioned_ablation", False))
    ):
        raise ValueError(
            "P5: CE graph construction with sparse_credit='role_contrib_team' "
            "is oracle role-conditioned and is allowed only for explicitly labeled "
            "oracle ablations, not the main black-box method path."
        )
    terminal_progress_cfg = terminal_progress_params(train_cfg)
    ce_gamma = float(graph_cfg.get("ce_gamma", train_cfg.get("gamma", 0.99)))
    ce_horizon = int(
        graph_cfg.get(
            "local_return_horizon_options",
            graph_cfg.get(
                "horizon_options",
                train_cfg.get("local_return_horizon_options", 5),
            ),
        )
    )
    ce_min_weight = float(graph_cfg.get("ce_min_weight", 20.0))
    ce_refine_top_k = int(graph_cfg.get("ce_refine_top_k", 32))
    ce_max_options_per_episode = int(
        graph_cfg.get(
            "ce_max_options_per_episode",
            train_cfg.get("max_options_per_episode", train_cfg.get("max_episode_options", 200)),
        )
    )
    min_actor_delivery_support = int(graph_cfg.get("min_actor_delivery_support", 1))
    support_objective = (
        "sparse_excluding_terminal_progress"
        if bool(args.sparse_ce_support)
        else "training_reward_sum"
    )
    reward_metadata = {
        "layout": layout,
        "cost_coef": cost_coef,
        "cost_per_step": cost_per_step,
        "shaped_reward_coef": shaped_reward_coef,
        "sparse_credit": credit_params["mode"],
        "partner_set": str(train_cfg.get("partner_set", "standard7")),
        "contribution_credit": {
            "contrib_scale": float(
                (train_cfg.get("contrib_team") or {}).get("contrib_scale", 1.0)
            ),
        },
        "reward_scale_source": "config.training",
        "event_semantics_version": int(EVENT_SEMANTICS_VERSION),
        "terminal_progress_shaping": terminal_progress_cfg,
        "ce_support_objective": {
            "gamma": ce_gamma,
            "horizon_options": ce_horizon,
            "min_weight": ce_min_weight,
            "min_actor_delivery_support": min_actor_delivery_support,
            "support_objective": support_objective,
            "sparse_ce_support": bool(args.sparse_ce_support),
        },
    }
    # The ego_correct_delivery mode's reward magnitude depends on explicit constants;
    # record them so the objective gate rejects a graph built with different constants.
    # team / ego_delivery do not use them, so they are omitted to keep the gate clean.
    if credit_params["mode"] == "ego_correct_delivery":
        reward_metadata["ego_delivery_reward"] = float(credit_params["ego_delivery_reward"])
        reward_metadata["ego_wrong_delivery_penalty"] = float(
            credit_params["ego_wrong_delivery_penalty"]
        )
    if credit_params["mode"] == "role_contrib_team":
        reward_metadata["role_contrib_team"] = {
            "ego_terminal_penalty_under_claim": float(
                (train_cfg.get("role_contrib_team") or {}).get(
                    "ego_terminal_penalty_under_claim", 0.3
                )
            ),
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
    _opt_cfg = config.get("options", {}) or {}
    lib = OCV2OptionLibrary(
        lg,
        max_option_steps=int(_opt_cfg.get("max_option_steps", 6)),
        strict_preconditions=bool(_opt_cfg.get("strict_preconditions", False)),
        dynamic_budget=bool(_opt_cfg.get("dynamic_budget", False)),
    )
    partners_all = make_training_partners(
        lib,
        partner_set=str(train_cfg.get("partner_set", "standard7")),
    )
    _train_names = (config.get("training", {}) or {}).get("train_partners")
    if _train_names:
        by_name = {p.name: p for p in partners_all}
        missing = sorted(set(_train_names) - set(by_name))
        if missing:
            raise ValueError(
                f"train_partners not found for CE collection: {missing}; "
                f"available={sorted(by_name)}"
            )
        # Preserve config order but collapse duplicates for CE graph construction; partner_sampling
        # affects online training distribution, not the CE support estimate.
        seen = set()
        partners = []
        for name in _train_names:
            if name in seen:
                continue
            seen.add(name)
            partners.append(by_name[str(name)])
        print(f"CE collected on TRAIN partners only (held-out excluded): {[p.name for p in partners]}")
    else:
        if not bool(train_cfg.get("allow_all_partners_for_no_split", False)):
            raise ValueError(
                "S23/P3: CE graph construction for formal split claims requires "
                "training.train_partners. Set training.allow_all_partners_for_no_split=true "
                "only for explicitly labeled no-split smoke runs."
            )
        partners = partners_all

    print(f"Options: {lib.num_options}")
    for opt in lib.options:
        print(f"  {opt.id}: {opt.kind}")

    ep_per_partner = int(args.episodes_per_partner)
    if args.reuse_replay:
        reuse_path = Path(args.reuse_replay)
        print(f"\n=== Phase 2: REUSE Replay ({reuse_path}) ===")
        rows, prior_meta = load_replay_npz(reuse_path)
        # Compatibility guard: reward objective + partners must match the current
        # config, otherwise we would silently be running eta/min_weight tests
        # against a replay from a different substrate. Fail loudly instead.
        prior_partners = tuple(prior_meta.get("partners", ()) or ())
        current_partners = tuple(p.name for p in partners)
        if prior_partners and prior_partners != current_partners:
            raise SystemExit(
                f"reuse_replay partners {prior_partners} do not match config partners "
                f"{current_partners}. Rerun collection or align the config."
            )
        prior_credit = str(prior_meta.get("sparse_credit", "team"))
        current_credit = str((reward_metadata or {}).get("sparse_credit", "team"))
        if prior_credit != current_credit:
            raise SystemExit(
                f"reuse_replay sparse_credit {prior_credit!r} != config {current_credit!r}. "
                "CE reward objective mismatch; rerun collection."
            )
        print(f"Reused {len(rows)} option replay rows (from {reuse_path.parent.name})")
    else:
        print(f"\n=== Phase 2: CE Collect ({ep_per_partner} episodes × {len(partners)} partners, sequential) ===")
        rows = collect_option_replay(
            env,
            partners,
            lib,
            layout_name=layout,
            episodes=ep_per_partner,
            max_options_per_episode=ce_max_options_per_episode,
            seed=int(args.seed),
            gamma=ce_gamma,
            horizon_options=ce_horizon,
            cost_per_step=cost_per_step,
            cost_coef=cost_coef,
            shaped_reward_coef=shaped_reward_coef,
            credit_params=credit_params,
            terminal_progress=terminal_progress_cfg,
            exclude_terminal_progress_from_reward_sum=bool(args.sparse_ce_support),
        )
        print(f"Collected {len(rows)} option replay rows")

    # Coverage check
    coverage = replay_coverage(rows, lib.options)
    print("\n--- Replay Coverage ---")
    for key, value in coverage.items():
        marker = "OK" if value > 0 else "MISSING"
        print(f"  {key}: {value} [{marker}]")

    try:
        gate = replay_coverage_gate(
            coverage,
            require_full_task_coverage=True,
            min_actor_delivery_support=min_actor_delivery_support,
        )
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
        "gamma": ce_gamma,
        "horizon_options": ce_horizon,
        "ce_min_weight": ce_min_weight,
        "ce_max_options_per_episode": ce_max_options_per_episode,
        "partner_option_evidence_policy": PARTNER_OPTION_EVIDENCE_POLICY,
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
    ce, ce_support_audit = estimate_empirical_ce_with_support(
        rows,
        lib.num_options,
        min_weight=ce_min_weight,
        gamma=ce_gamma,
        horizon_options=ce_horizon,
        reward_objective=str(credit_params["mode"]),
        support_objective=support_objective,
    )
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
    refined, refine_meta = refine_empirical_ce(
        ce, rows, lib.num_options, top_k=ce_refine_top_k, min_weight=ce_min_weight
    )
    estimable_mask = np.asarray(ce_support_audit.get("estimable_mask", []), dtype=bool)
    if estimable_mask.shape != refined.shape:
        raise ValueError(
            f"CE support audit shape mismatch: mask={estimable_mask.shape} refined={refined.shape}"
        )
    measured_zero_mask = np.asarray(ce_support_audit.get("measured_zero_mask", []), dtype=bool)
    skipped_mask = np.asarray(ce_support_audit.get("skipped_mask", []), dtype=bool)
    weight_sum = np.asarray(ce_support_audit.get("weight_sum", []), dtype=float)
    refined_for_graph = np.asarray(refined, dtype=np.float32).copy()
    # Unsupported/skipped cells are not zero-effect evidence.  The formal ce_refined
    # artifact consumed by train_aris is support-masked; the raw refined scores are
    # kept separately for audit only.
    refined_for_graph[~estimable_mask] = 0.0
    refined_unmasked_path = output_dir / "ce_refined_unmasked.npy"
    np.save(refined_unmasked_path, refined)
    refined_unmasked_sha256 = sha256_file(refined_unmasked_path)
    refined_path = output_dir / "ce_refined.npy"
    np.save(refined_path, refined_for_graph)
    refined_sha256 = sha256_file(refined_path)
    support_sidecar_path = output_dir / "ce_support_audit.json"
    support_sidecar_path.write_text(
        json.dumps(ce_support_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    support_sidecar_sha256 = sha256_file(support_sidecar_path)

    sidecar_path = output_dir / "ce_refined.meta.json"
    sidecar_path.write_text(
        json.dumps(
            {
                **reward_metadata,
                **refine_meta,
                "ce_support_audit": ce_support_audit,
                "ce_support_sidecar": str(support_sidecar_path),
                "ce_support_sidecar_sha256": support_sidecar_sha256,
                "ce_refined_unmasked": str(refined_unmasked_path),
                "ce_refined_unmasked_sha256": refined_unmasked_sha256,
                "ce_unsupported_cells_masked": True,
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
    print(f"Saved CE support sidecar to {support_sidecar_path}")

    print("\n=== Phase 4: Graph Build ===")
    # Coverage-constrained selection: graph_cfg may carry selection / required_option_*_coverage
    # / diversity. With selection="coverage_constrained_ce" task-critical options (serve_soup, …)
    # are reserved before CE fill, so the task-stage gate is a postcondition, not a late surprise.
    graph = build_support_graph(
        layout,
        lib.options,
        refined_for_graph,
        eta=float(graph_cfg.get("ce_eta", 0.05)),
        max_factors=int(graph_cfg.get("max_factors", 16)),
        selection_cfg=graph_cfg,
    )
    graph.factors = [
        replace(
            factor,
            metadata={
                **(factor.metadata or {}),
                "ce_weight_sum": float(weight_sum[factor.option_i, factor.option_j]),
                "ce_estimable": bool(estimable_mask[factor.option_i, factor.option_j]),
                "ce_skipped": bool(skipped_mask[factor.option_i, factor.option_j]),
                "ce_measured_zero": bool(measured_zero_mask[factor.option_i, factor.option_j]),
                "ce_support_min_weight": ce_min_weight,
            },
        )
        for factor in graph.factors
    ]
    graph.metadata = {
        **(graph.metadata or {}),
        **reward_metadata,
        "coverage_gate": "pending",
        "formal_graph": True,
        "ce_support_audit": ce_support_audit,
        "ce_support_sidecar": str(support_sidecar_path),
        "ce_support_sidecar_sha256": support_sidecar_sha256,
        "ce_unsupported_cells_masked": True,
        "ce_refine_top_k": ce_refine_top_k,
        "ce_max_options_per_episode": ce_max_options_per_episode,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for CE collection rollouts; vary to separate CE stochasticity from "
        "training stochasticity (default 42 reproduces the original collection).",
    )
    parser.add_argument(
        "--sparse_ce_support",
        action="store_true",
        help="Exclude the dense terminal_progress_bonus from the CE return used for graph "
        "support selection (G3). TD-time progression shaping is unaffected; this only "
        "changes which factors the support graph is selected on.",
    )
    parser.add_argument("--allow_incomplete_graph", action="store_true")
    parser.add_argument("--episodes-per-partner", dest="episodes_per_partner", type=int, default=100)
    parser.add_argument(
        "--reuse_replay",
        default=None,
        help="Reuse an existing replay.npz (from a prior pipeline run) instead of "
        "re-collecting. Skips Phase 2 collection and reads the saved rows. Use to "
        "cheaply re-run estimation/graph-build with different eta/min_weight/config "
        "flags. Ledger: R1-C follow-up (was 'wontfix/defer' — enabled by demand).",
    )
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
