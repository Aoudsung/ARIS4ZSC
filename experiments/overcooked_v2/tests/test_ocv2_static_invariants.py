from __future__ import annotations

import ast
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_ce_local_return_source_uses_shaped_reward_and_training_cost_scale():
    src = _read("experiments/overcooked_v2/ce_sampler.py")
    assert "shaped_reward_coef" in src
    assert "next_row.shaped_reward_sum" in src
    assert "- float(cost_coef) * next_row.realized_cost" in src


def test_graph_builder_source_excludes_noop_from_interaction_factors():
    src = _read("experiments/overcooked_v2/graph_builder.py")
    assert "EXCLUDED_FACTOR_OPTION_KINDS = {\"noop\"}" in src
    assert "_factorable_option_pair" in src
    assert "options[option_i].kind not in EXCLUDED_FACTOR_OPTION_KINDS" in src
    assert "option_i == option_j" in src


def test_train_script_has_no_gtvoi_or_mi_selector_logic():
    src = _read("experiments/overcooked_v2/train_aris.py")
    tree = ast.parse(src)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "gtvoi_selector" not in src
    assert "mi_selector" not in src
    assert "active_gain" not in src
    assert "G_TVOI" not in names


def test_train_requires_explicit_accepted_preflight_without_smoke_bypass():
    src = _read("experiments/overcooked_v2/train_aris.py")
    assert "OvercookedV2 training requires an accepted preflight report" in src
    assert "failed preflight; refusing formal experiment" in src
    assert "smoke_without_preflight" not in src
    assert "allow_rejected_layout_smoke" not in src


def test_train_records_task_progress_counts():
    src = _read("experiments/overcooked_v2/train_aris.py")
    assert "task_progress_counts" in src
    assert "event_summary=event_summary" in src
    assert "delivery_event_count" in src
