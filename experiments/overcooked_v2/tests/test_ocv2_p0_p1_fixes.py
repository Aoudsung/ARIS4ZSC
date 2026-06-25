from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject, StaticObject

from src.aris_bellman.specs import FactorSpec, GraphSpec, OptionSpec

from experiments.overcooked_v2 import ce_sampler
from experiments.overcooked_v2 import evaluate_aris, layout_diagnostics, train_aris
from experiments.overcooked_v2.batched_rollout import (
    BatchedRolloutUnsupported,
    assert_batched_shape,
    batched_reset,
)
from experiments.overcooked_v2.event_extractor import extract_event
from experiments.overcooked_v2.evidence_router import EVIDENCE_INDEX, OCV2EvidenceRouter
from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.graph_builder import build_graph_variant
from experiments.overcooked_v2.layout_parser import LayoutGraph, all_pairs_shortest_paths
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer
from experiments.overcooked_v2.obs_encoder import OCV2ObsEncoder
from experiments.overcooked_v2.option_inferencer import PartnerOptionInferencer
from experiments.overcooked_v2.option_termination import OptionRuntime, option_terminated
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.partner_option_classifier import PartnerOptionClassifier
from experiments.overcooked_v2.partner_pool import ProtocolSpec, ScriptedProtocolPartner


def _option(option_id: int, entity_ids=(), region_ids=(), kind="fetch_ingredient"):
    return OptionSpec(
        id=option_id,
        name=f"option:{option_id}",
        kind=kind,
        target_id=None,
        target_pos=None,
        entity_ids=tuple(entity_ids),
        region_ids=tuple(region_ids),
        max_steps=10,
        metadata={},
    )


def _graph(options, factors, route_map=None):
    num_factors = len(factors)
    num_options = len(options)
    return GraphSpec(
        layout_name="unit",
        options=list(options),
        factors=list(factors),
        relevance=np.ones((num_factors, num_options), dtype=bool),
        option_mask=np.ones((num_options,), dtype=bool),
        factor_mask=np.ones((num_factors,), dtype=bool),
        mode_mask=np.ones((num_factors, 2), dtype=bool),
        route_map=route_map or {idx: (factor.option_i, factor.option_j) for idx, factor in enumerate(factors)},
        metadata={},
    )


def _factor(factor_id: int, option_i: int, option_j: int, entity_ids=(), region_ids=(), score=1.0):
    return FactorSpec(
        id=factor_id,
        option_i=option_i,
        option_j=option_j,
        ce_score=float(score),
        num_modes=2,
        entity_ids=tuple(entity_ids),
        region_ids=tuple(region_ids),
    )


def _event(**overrides):
    data = {
        "ego_pos_before": (0, 0),
        "ego_pos_after": (0, 0),
        "partner_pos_before": (9, 9),
        "partner_pos_after": (9, 9),
        "ego_inventory_before": 0,
        "ego_inventory_after": 0,
        "partner_inventory_before": 0,
        "partner_inventory_after": 0,
        "ego_action": 0,
        "partner_action": 0,
        "partner_option": None,
        "partner_option_dist": None,
        "partner_option_confidence": 0.0,
        "ego_waited": False,
        "partner_waited": True,
        "ego_interacted": False,
        "partner_interacted": False,
        "collision_or_block": False,
        "delivery_event": False,
        "wrong_delivery_event": False,
        "pot_changed": False,
        "object_pickup_or_drop": False,
        "recipe_indicator_event": False,
        "button_pressed": False,
        "changed_cells": (),
        "changed_object_bits": (),
        "changed_object_before": (),
        "changed_object_after": (),
        "pot_changed_cells": (),
        "pot_became_full": False,
        "pot_became_cooked": False,
        "pot_became_ready": False,
        "plate_picked": False,
        "soup_picked": False,
        "correct_delivery": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _state(
    agent0_pos,
    inventory0=0,
    grid=None,
    agent1_pos=(5, 5),
    inventory1=0,
    dir0=0,
    dir1=0,
    recipe=None,
):
    if grid is None:
        grid = np.zeros((6, 6, 2), dtype=np.int32)
    if recipe is None:
        recipe = _recipe(0, 0, 0)
    agents = SimpleNamespace(
        pos=SimpleNamespace(
            x=np.asarray([agent0_pos[0], agent1_pos[0]]),
            y=np.asarray([agent0_pos[1], agent1_pos[1]]),
        ),
        dir=np.asarray([dir0, dir1]),
        inventory=np.asarray([inventory0, inventory1]),
    )
    return SimpleNamespace(agents=agents, grid=grid, recipe=np.asarray(recipe))


def _layout_graph_for_entities(entities):
    passable = np.ones((4, 4), dtype=bool)
    cell_to_entity = {}
    entity_map = {}
    entities_by_kind = {}
    interaction_cells = {}
    for entity_id, kind, pos in entities:
        x, y = pos
        passable[y, x] = False
        entity = SimpleNamespace(id=entity_id, kind=kind, pos=pos)
        entity_map[entity_id] = entity
        entities_by_kind.setdefault(kind, []).append(entity_id)
        cell_to_entity[pos] = entity_id
        interaction_cells[entity_id] = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        interaction_cells[entity_id] = [
            cell
            for cell in interaction_cells[entity_id]
            if 0 <= cell[0] < 4 and 0 <= cell[1] < 4 and passable[cell[1], cell[0]]
        ]
    return LayoutGraph(
        layout_name="unit",
        width=4,
        height=4,
        passable=passable,
        entities=entity_map,
        entities_by_kind=entities_by_kind,
        interaction_cells=interaction_cells,
        bottlenecks=[],
        region_cells={},
        cell_to_entity=cell_to_entity,
        shortest_path_dist=all_pairs_shortest_paths(passable),
    )


def _grid_with_static(pos, static_obj, dynamic_obj=0):
    grid = np.zeros((4, 4, 2), dtype=np.int32)
    x, y = pos
    grid[y, x, 0] = int(static_obj)
    grid[y, x, 1] = int(dynamic_obj)
    return grid


def _grid_with_cells(cells, *, channels=3):
    grid = np.zeros((4, 4, channels), dtype=np.int32)
    for item in cells:
        pos, static_obj, dynamic_obj, *rest = item
        timer = rest[0] if rest else 0
        x, y = pos
        grid[y, x, 0] = int(static_obj)
        grid[y, x, 1] = int(dynamic_obj)
        if channels > 2:
            grid[y, x, 2] = int(timer)
    return grid


def _recipe(*ingredients):
    return sum(int(DynamicObject.ingredient(idx)) for idx in ingredients)


def test_numpy_featurizer_outputs_fixed_dim_and_agent_order():
    layout = _layout_graph_for_entities([])
    state = _state(
        (1, 1),
        agent1_pos=(2, 2),
        grid=np.zeros((4, 4, 3), dtype=np.int32),
        dir0=2,
        dir1=3,
        recipe=_recipe(0, 0, 1),
    )

    obs = NumpyFeaturizer(layout)(state)

    assert set(obs) == {"agent_0", "agent_1"}
    assert obs["agent_0"].shape == (96,)
    assert obs["agent_1"].shape == (96,)
    assert obs["agent_0"].dtype == np.float32
    np.testing.assert_array_equal(obs["agent_0"][0:4], np.asarray([0, 0, 1, 0]))
    np.testing.assert_array_equal(obs["agent_0"][46:50], np.asarray([0, 0, 0, 1]))
    np.testing.assert_array_equal(obs["agent_0"][92:94], np.asarray([1, 1]))
    np.testing.assert_array_equal(obs["agent_0"][94:96], np.asarray([1, 1]))


def test_numpy_featurizer_uses_interaction_distance_for_targets():
    layout = _layout_graph_for_entities(
        [
            ("ingredient_pile:0:2:1", "ingredient_pile:0", (2, 1)),
            ("delivery:1:0", "delivery", (1, 0)),
            ("counter:0:1", "counter", (0, 1)),
            ("counter:3:1", "counter", (3, 1)),
        ]
    )
    grid = _grid_with_cells(
        [
            ((2, 1), int(StaticObject.INGREDIENT_PILE_BASE), 0),
            ((1, 0), StaticObject.GOAL, 0),
            ((0, 1), StaticObject.WALL, 0),
            ((3, 1), StaticObject.WALL, DynamicObject.PLATE),
        ]
    )
    state = _state((1, 1), agent1_pos=(1, 2), grid=grid)

    obs0 = NumpyFeaturizer(layout)(state)["agent_0"]

    np.testing.assert_array_equal(obs0[8:10], np.asarray([1, 0]))
    np.testing.assert_array_equal(obs0[12:14], np.asarray([2, 0]))
    np.testing.assert_array_equal(obs0[18:20], np.asarray([0, -1]))
    np.testing.assert_array_equal(obs0[20:22], np.asarray([-1, 0]))


def test_numpy_featurizer_encodes_recipe_soup_and_pot_features():
    recipe = _recipe(0, 0, 1)
    soup = recipe | int(DynamicObject.COOKED) | int(DynamicObject.PLATE)
    layout = _layout_graph_for_entities(
        [
            ("pot:2:1", "pot", (2, 1)),
            ("counter:1:0", "counter", (1, 0)),
        ]
    )
    grid = _grid_with_cells(
        [
            ((2, 1), StaticObject.POT, recipe | int(DynamicObject.COOKED), 0),
            ((1, 0), StaticObject.WALL, soup, 0),
        ]
    )
    state = _state((1, 1), agent1_pos=(1, 2), grid=grid, recipe=recipe)

    obs0 = NumpyFeaturizer(layout)(state)["agent_0"]

    np.testing.assert_array_equal(obs0[14:16], np.asarray([0, -1]))
    np.testing.assert_array_equal(obs0[16:18], np.asarray([2, 1]))
    np.testing.assert_array_equal(
        obs0[22:32],
        np.asarray([1, 0, 1, 0, 1, 2, 1, 0, 1, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(obs0[32:42], np.zeros(10, dtype=np.float32))


def test_numpy_featurizer_defaults_missing_fake_grid_timer_to_zero():
    recipe = _recipe(0, 0, 0)
    layout = _layout_graph_for_entities([("pot:2:1", "pot", (2, 1))])
    grid = _grid_with_static((2, 1), StaticObject.POT, recipe)
    state = _state((1, 1), agent1_pos=(1, 2), grid=grid, recipe=recipe)

    obs0 = NumpyFeaturizer(layout)(state)["agent_0"]

    assert obs0[22 + 7] == 0.0


def test_numpy_featurizer_rejects_non_two_agent_contract():
    layout = _layout_graph_for_entities([])
    with pytest.raises(ValueError, match="exactly 2 agents"):
        NumpyFeaturizer(layout, num_agents=3)


def test_ocv2_adapter_applies_optional_featurizer_only_when_present():
    adapter = OCV2Adapter.__new__(OCV2Adapter)
    adapter.featurizer = None
    raw = {"agent_0": np.asarray([1]), "agent_1": np.asarray([2])}

    passthrough = adapter._apply_featurizer(raw, object())
    np.testing.assert_array_equal(passthrough["agent_0"], np.asarray([1]))

    adapter.set_featurizer(
        lambda _state: {
            "agent_0": np.ones(96, dtype=np.float32),
            "agent_1": np.zeros(96, dtype=np.float32),
        }
    )
    featurized = adapter._apply_featurizer(raw, object())
    assert featurized["agent_0"].shape == (96,)
    assert featurized["agent_0"].dtype == np.float32


def test_train_eval_and_preflight_use_default_env_without_path_planning():
    train_build_env = inspect.getsource(train_aris._build_env)
    assert 'observation_type="default"' in train_build_env
    assert "force_path_planning=False" in train_build_env

    preflight_make_env = inspect.getsource(layout_diagnostics._make_env)
    assert 'observation_type="default"' in preflight_make_env
    assert "force_path_planning=False" in preflight_make_env

    assert "NumpyFeaturizer(layout_graph)" in inspect.getsource(evaluate_aris._load_context)
    assert "NumpyFeaturizer(ctx.layout_graph)" in inspect.getsource(
        evaluate_aris._evaluate_partner
    )
    assert "NumpyFeaturizer(ctx.layout_graph)" in inspect.getsource(
        evaluate_aris._factor_deletion_q_proxy
    )


def test_option_library_uses_python_shortest_paths_not_jax_path_planner():
    layout = _layout_graph_for_entities([])
    option_lib = OCV2OptionLibrary(layout, max_option_steps=6)

    assert not hasattr(option_lib, "path_planner")
    assert option_lib._closest_target_cell(
        _state((0, 0), grid=np.zeros((4, 4, 2), dtype=np.int32)),
        agent_id=0,
        targets=((3, 3), (1, 0)),
    ) == (1, 0)


def test_closest_target_cell_returns_none_when_targets_unreachable():
    layout = _layout_graph_for_entities([])
    layout.shortest_path_dist = {}
    option_lib = OCV2OptionLibrary(layout, max_option_steps=6)

    assert option_lib._closest_target_cell(
        _state((0, 0), grid=np.zeros((4, 4, 2), dtype=np.int32)),
        agent_id=0,
        targets=((1, 0),),
    ) is None


def test_layout_preflight_forwards_ce_max_options_per_episode(monkeypatch):
    calls = []

    def fake_collect_option_replay(*args, **kwargs):
        calls.append(kwargs)
        return [
            ce_sampler.OptionReplayRow(
                layout="unit",
                episode_id=0,
                t_option=0,
                ego_option=0,
                partner_option=0,
                partner_option_dist=None,
                partner_option_confidence=1.0,
                state_key="s",
                duration=1,
                reward_sum=1.0,
                shaped_reward_sum=0.0,
                realized_cost=1.0,
                local_return_h=1.0,
                reward_to_go=1.0,
                event_summary={},
                partner_name="p",
                partner_id=0,
            )
        ]

    monkeypatch.setattr(layout_diagnostics, "collect_option_replay", fake_collect_option_replay)
    monkeypatch.setattr(
        layout_diagnostics,
        "estimate_empirical_ce",
        lambda replay, num_options, min_weight: np.ones((num_options, num_options)),
    )
    option_lib = SimpleNamespace(num_options=2)
    config = {
        "graph": {"ce_episodes": 1, "ce_max_options_per_episode": 7},
        "training": {"cost_per_step": 1.0, "cost_coef": 0.02},
    }

    layout_diagnostics._ce_matrix_and_replay(
        env=SimpleNamespace(),
        option_lib=option_lib,
        layout_name="unit",
        config=config,
        gamma=0.99,
        horizon_options=3,
        min_weight=1.0,
    )
    assert calls[-1]["max_options_per_episode"] == 7

    layout_diagnostics._partner_return_proxy_stats(
        env=SimpleNamespace(),
        option_lib=option_lib,
        layout_name="unit",
        config={"graph": {"ce_max_options_per_episode": 5}, "diagnostics": {"proxy_episodes": 1}},
        replay_rows=None,
        gamma=0.99,
        horizon_options=3,
    )
    assert calls[-1]["max_options_per_episode"] == 5


def test_shuffled_routes_changes_routed_evidence():
    options = [_option(0, entity_ids=("a",)), _option(1, entity_ids=("b",))]
    factors = [
        _factor(0, 0, 0, entity_ids=("a",)),
        _factor(1, 1, 1, entity_ids=("b",)),
    ]
    full = _graph(options, factors, route_map={0: (0,), 1: (1,)})
    shuffled = _graph(options, factors, route_map={0: (1,), 1: (0,)})
    cell_to_entity = {(0, 0): "a", (5, 5): "b"}
    event = _event(ego_pos_after=(0, 0))

    full_x = OCV2EvidenceRouter(full, cell_to_entity, {}).route(event)
    shuffled_x = OCV2EvidenceRouter(shuffled, cell_to_entity, {}).route(event)

    assert not np.allclose(full_x, shuffled_x)
    assert full_x[0, EVIDENCE_INDEX["ego_near_entity"]] == 1.0
    assert shuffled_x[0, EVIDENCE_INDEX["ego_near_entity"]] == 0.0


def test_event_extractor_and_router_expose_pot_semantic_evidence():
    pot_pos = (1, 1)
    before_grid = _grid_with_static(pot_pos, StaticObject.POT, 0)
    after_grid = _grid_with_static(
        pot_pos,
        StaticObject.POT,
        int(DynamicObject.COOKED) | (1 << 2),
    )
    event = extract_event(
        _state((0, 1), grid=before_grid),
        int(Actions.stay),
        int(Actions.stay),
        _state((0, 1), grid=after_grid),
        {"wrong_delivery_event": False},
        partner_option=None,
        partner_option_dist=None,
    )
    assert event.pot_changed is True
    assert event.pot_became_ready is True
    assert event.wrong_delivery_event is False

    options = [_option(0, entity_ids=("pot:1:1",))]
    factor = _factor(0, 0, 0, entity_ids=("pot:1:1",))
    graph = _graph(options, [factor], route_map={0: (0,)})
    router = OCV2EvidenceRouter(graph, {pot_pos: "pot:1:1"}, {})
    routed = router.route(event)

    assert routed[0, EVIDENCE_INDEX["pot_changed"]] == 1.0
    assert routed[0, EVIDENCE_INDEX["pot_became_ready"]] == 1.0
    assert routed[0, EVIDENCE_INDEX["pot_changed_near_entity"]] == 1.0


def test_partner_fixed_within_episode_reset_only_resets_selected_partner():
    class FakeEnv:
        def reset(self, seed):
            return {"agent_0": np.zeros(1)}, SimpleNamespace(seed=seed)

    class FakeBuffer:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class FakeRouter:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class FakePartner:
        def __init__(self, name):
            self.name = name
            self.reset_count = 0

        def reset(self, seed):
            self.reset_count += 1

    partners = [FakePartner("a"), FakePartner("b"), FakePartner("c")]
    buffer = FakeBuffer()
    router = FakeRouter()

    _obs, _state_obj, selected = train_aris._reset_episode(
        FakeEnv(),
        buffer,
        partners,
        np.random.default_rng(0),
        123,
        router,
    )

    assert selected in partners
    assert buffer.reset_count == 1
    assert router.reset_count == 1
    assert sum(partner.reset_count for partner in partners) == 1


def test_cross_bottleneck_terminates_only_after_crossing():
    opt = OptionSpec(
        id=0,
        name="cross",
        kind="cross_bottleneck",
        target_id="b",
        target_pos=(1, 0),
        entity_ids=(),
        region_ids=("bottleneck:0",),
        max_steps=10,
        metadata={"region_cells": ((1, 0),)},
    )
    runtime = OptionRuntime(option_id=0, start_pos=(0, 0))
    event = _event()

    terminated, reason = option_terminated(
        opt,
        _state((0, 0)),
        _state((1, 0)),
        event,
        agent_id=0,
        elapsed=1,
        runtime=runtime,
    )
    assert (terminated, reason) == (False, "running")

    terminated, reason = option_terminated(
        opt,
        _state((1, 0)),
        _state((2, 0)),
        event,
        agent_id=0,
        elapsed=2,
        runtime=runtime,
    )
    assert (terminated, reason) == (True, "crossed_bottleneck")


def test_wait_bottleneck_waits_after_arrival():
    opt = OptionSpec(
        id=0,
        name="wait",
        kind="wait_at_bottleneck",
        target_id="b",
        target_pos=(1, 0),
        entity_ids=(),
        region_ids=("bottleneck:0",),
        max_steps=20,
        metadata={"region_cells": ((1, 0),), "wait_duration": 2},
    )
    runtime = OptionRuntime(option_id=0, start_pos=(0, 0))
    move_event = _event(ego_waited=False, partner_action=0, partner_pos_after=(5, 5))
    wait_event = _event(ego_waited=True, partner_action=0, partner_pos_after=(5, 5))

    terminated, reason = option_terminated(
        opt,
        _state((0, 0)),
        _state((0, 0)),
        move_event,
        agent_id=0,
        elapsed=5,
        runtime=runtime,
    )
    assert (terminated, reason) == (False, "running")

    terminated, reason = option_terminated(
        opt,
        _state((0, 0)),
        _state((1, 0)),
        move_event,
        agent_id=0,
        elapsed=6,
        runtime=runtime,
    )
    assert (terminated, reason) == (False, "running")
    assert runtime.wait_elapsed_after_arrival == 0

    terminated, reason = option_terminated(
        opt,
        _state((1, 0)),
        _state((1, 0)),
        wait_event,
        agent_id=0,
        elapsed=7,
        runtime=runtime,
    )
    assert (terminated, reason) == (False, "running")
    assert runtime.wait_elapsed_after_arrival == 1

    terminated, reason = option_terminated(
        opt,
        _state((1, 0)),
        _state((1, 0)),
        wait_event,
        agent_id=0,
        elapsed=8,
        runtime=runtime,
    )
    assert (terminated, reason) == (True, "wait_duration_after_arrival")


def test_scripted_partner_wait_runtime_counts_only_post_arrival_stay():
    opt = OptionSpec(
        id=0,
        name="wait",
        kind="wait_at_bottleneck",
        target_id="b",
        target_pos=(1, 0),
        entity_ids=(),
        region_ids=("bottleneck:0",),
        max_steps=20,
        metadata={"region_cells": ((1, 0),), "wait_duration": 2},
    )
    partner = ScriptedProtocolPartner(
        name="unit",
        option_library=SimpleNamespace(options=[opt]),
        protocol=ProtocolSpec(),
    )
    partner.current_option = 0
    partner.option_runtime = OptionRuntime(option_id=0, start_pos=(0, 0))
    partner.last_state = _state((5, 5), agent1_pos=(0, 0))
    partner.last_primitive_action = int(Actions.right)

    partner._update_option_runtime(_state((5, 5), agent1_pos=(1, 0)), agent_id=1)
    assert partner.option_runtime.reached_region is True
    assert partner.option_runtime.wait_elapsed_after_arrival == 0

    partner.last_state = _state((5, 5), agent1_pos=(1, 0))
    partner.last_primitive_action = int(Actions.stay)
    partner._update_option_runtime(_state((5, 5), agent1_pos=(1, 0)), agent_id=1)
    assert partner.option_runtime.wait_elapsed_after_arrival == 1


def test_train_requires_precomputed_ce_for_formal_mode():
    args = SimpleNamespace(seed=0, graph_variant="full_support")
    with pytest.raises(ValueError, match="Formal training requires"):
        train_aris._build_graph(
            env=SimpleNamespace(),
            layout_graph=SimpleNamespace(layout_name="unit"),
            option_lib=SimpleNamespace(options=[], num_options=0),
            config={"graph": {}, "training": {"gamma": 0.99}},
            args=args,
        )


def test_option_validity_filters_task_inapplicable_pot_options_and_builds_button():
    layout = _layout_graph_for_entities(
        [
            ("pot:1:1", "pot", (1, 1)),
            ("button_recipe_indicator:2:1", "button_recipe_indicator", (2, 1)),
        ]
    )
    option_lib = OCV2OptionLibrary(layout, max_option_steps=6)
    cooked_soup = int(DynamicObject.COOKED) | (1 << 2)
    state = _state(
        (0, 1),
        inventory0=int(DynamicObject.PLATE),
        grid=_grid_with_static((1, 1), StaticObject.POT, cooked_soup),
    )

    valid = option_lib.valid_options(state, agent_id=0)
    valid_kinds = {
        option_lib.options[idx].kind
        for idx in np.flatnonzero(valid)
    }

    assert "plate_soup" in valid_kinds
    assert any(opt.kind == "press_recipe_button" for opt in option_lib.options)

    ingredient_state = _state(
        (0, 1),
        inventory0=(1 << 2),
        grid=_grid_with_static((1, 1), StaticObject.POT, cooked_soup),
    )
    ingredient_valid = option_lib.valid_options(ingredient_state, agent_id=0)
    ingredient_valid_kinds = {
        option_lib.options[idx].kind
        for idx in np.flatnonzero(ingredient_valid)
    }
    assert "deliver_ingredient_to_pot" not in ingredient_valid_kinds


def test_minus_critical_requires_scores_and_minus_high_ce_is_debug_path():
    options = [_option(0, entity_ids=("a",)), _option(1, entity_ids=("b",))]
    ce = np.asarray([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="minus_critical requires"):
        build_graph_variant(
            "minus_critical",
            "unit",
            options,
            ce,
            eta=-1.0,
            max_factors=2,
        )

    graph = build_graph_variant(
        "minus_high_ce",
        "unit",
        options,
        ce,
        eta=-1.0,
        max_factors=2,
    )
    assert graph.metadata["graph_variant"] == "minus_high_ce"
    assert graph.metadata["criticality_source"] == "ce_highest_debug"


def test_overcomplete_has_extra_factors_when_budgeted():
    options = [
        _option(0, entity_ids=("a",)),
        _option(1, entity_ids=("b",)),
        _option(2, entity_ids=("c",)),
    ]
    ce = np.asarray(
        [
            [0.0, 3.0, 2.0],
            [1.0, 0.0, 0.5],
            [0.25, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    graph = build_graph_variant(
        "overcomplete",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=1,
        full_max_factors=1,
        overcomplete_extra_factors=2,
    )
    assert graph.num_factors == 3
    assert graph.metadata["full_count"] == 1
    assert graph.metadata["extra_count"] == 2


def test_full_support_and_derived_variants_honor_full_max_factors():
    options = [
        _option(0, entity_ids=("a",)),
        _option(1, entity_ids=("b",)),
        _option(2, entity_ids=("c",)),
    ]
    ce = np.asarray(
        [
            [0.0, 5.0, 4.0],
            [3.0, 0.0, 2.0],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    full = build_graph_variant(
        "full_support",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=10,
        full_max_factors=2,
    )
    shuffled = build_graph_variant(
        "shuffled_routes",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=10,
        full_max_factors=2,
    )
    random_graph = build_graph_variant(
        "random_same_size",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=10,
        full_max_factors=2,
    )

    assert full.num_factors == 2
    assert shuffled.num_factors == 2
    assert random_graph.metadata["full_support_count"] == 2
    assert full.metadata["full_max_factors"] == 2


def test_factor_deletion_rollout_and_q_proxy_are_separate_outputs():
    source = inspect.getsource(evaluate_aris.evaluate)
    assert "factor_deletion_q_proxy" in source
    assert "factor_deletion_return_drop" in source
    assert "\"factor_deletion_return_drop\"" not in inspect.getsource(
        evaluate_aris._factor_deletion_q_proxy_diagnostics
    )


def test_partner_id_q_consumes_partner_id():
    graph = _graph([_option(0), _option(1)], [_factor(0, 0, 1)])
    q_net = train_aris._build_q_network(
        "partner_id_q",
        4,
        graph,
        {"training": {"hidden_dim": 8, "num_partners": 3, "obs_encoder": "mlp"}},
    )
    obs = torch.zeros(2, 4)
    belief = torch.zeros(2, 1, 2)
    graph_batch = train_aris._graph_tensors(graph, 2, torch.device("cpu"))
    q_values = q_net(
        obs,
        belief,
        **train_aris._q_forward_kwargs(graph_batch),
        partner_id=torch.tensor([0, 1]),
    )
    assert q_values.shape == (2, graph.num_options)


def test_obs_encoder_supports_mlp_and_cnn_shapes():
    mlp = OCV2ObsEncoder(5, hidden_dim=8, encoder_type="mlp")
    cnn = OCV2ObsEncoder((4, 4, 3), hidden_dim=8, encoder_type="cnn")

    assert mlp(torch.zeros(2, 5)).shape == (2, 8)
    assert cnn(torch.zeros(2, 4, 4, 3)).shape == (2, 8)


def test_partner_option_classifier_checkpoint_path(tmp_path):
    model = PartnerOptionClassifier(input_dim=16, num_options=2, hidden_dim=8)
    path = tmp_path / "classifier.pt"
    torch.save(
        {
            "input_dim": 16,
            "num_options": 2,
            "hidden_dim": 8,
            "state_dict": model.state_dict(),
        },
        path,
    )
    inferencer = PartnerOptionInferencer(
        option_library=SimpleNamespace(num_options=2),
        classifier_checkpoint=str(path),
    )
    action = inferencer.update(
        _state((0, 0)),
        int(Actions.stay),
        _state((0, 0)),
        _event(partner_action=int(Actions.stay)),
    )

    assert action.source == "classifier"
    assert action.option_dist.shape == (2,)
    with pytest.raises(ValueError, match="classifier_checkpoint"):
        PartnerOptionInferencer(SimpleNamespace(num_options=2), allow_heuristic=False)


def test_interventional_ce_refine_updates_only_topk_and_records_metadata():
    rows = [
        ce_sampler.OptionReplayRow(
            layout="unit",
            episode_id=0,
            t_option=0,
            ego_option=0,
            partner_option=1,
            partner_option_dist=None,
            partner_option_confidence=1.0,
            state_key="s",
            duration=1,
            reward_sum=1.0,
            shaped_reward_sum=0.0,
            realized_cost=1.0,
            local_return_h=0.0,
            reward_to_go=0.0,
            event_summary={},
            partner_name="p",
            partner_id=0,
        )
    ]
    ce = np.asarray([[0.0, 5.0], [1.0, 0.0]], dtype=np.float32)

    refined, metadata = ce_sampler.refine_interventional_ce(
        ce,
        rows,
        num_options=2,
        top_k=1,
        samples_per_pair=2,
        cost_coef=0.5,
        intervention_runner=lambda ego, partner, seed, sample_idx: 10.0 + sample_idx,
    )

    assert metadata["forced_intervention"] is True
    assert metadata["changed_pairs"] == [[0, 1]]
    assert refined[0, 1] != ce[0, 1]
    assert refined[1, 0] == ce[1, 0]


def test_batched_rollout_shape_helper_and_visible_unsupported_failure():
    assert_batched_shape({"agent_0": np.zeros((3, 2))}, batch_size=3)
    with pytest.raises(BatchedRolloutUnsupported, match="adapter exposing"):
        batched_reset(SimpleNamespace(), [1, 2])


def test_no_gtvoi_selector_in_train():
    source = inspect.getsource(train_aris)
    assert "gtvoi_selector" not in source
    assert "mi_selector" not in source


def test_ce_local_return_uses_shaped_reward_with_training_scale():
    row = ce_sampler.OptionReplayRow(
        layout="unit",
        episode_id=0,
        t_option=0,
        ego_option=0,
        partner_option=1,
        partner_option_dist=None,
        partner_option_confidence=1.0,
        state_key="s",
        duration=1,
        reward_sum=1.0,
        shaped_reward_sum=3.0,
        realized_cost=2.0,
        local_return_h=0.0,
        reward_to_go=0.0,
        event_summary={},
        partner_name="p",
        partner_id=0,
    )
    ce_sampler.compute_local_returns([row], gamma=0.99, horizon=1, cost_coef=0.5, shaped_reward_coef=0.25)
    assert row.local_return_h == pytest.approx(1.0 + 0.25 * 3.0 - 0.5 * 2.0)


def test_graph_builder_excludes_noop_from_factor_support():
    options = [
        _option(0, entity_ids=("a",), kind="noop"),
        _option(1, entity_ids=("b",), kind="fetch_ingredient"),
        _option(2, entity_ids=("c",), kind="pick_plate"),
    ]
    ce = np.asarray(
        [
            [0.0, 100.0, 100.0],
            [100.0, 0.0, 2.0],
            [100.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph = build_graph_variant(
        "full_support",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=4,
    )
    assert graph.num_factors > 0
    for factor in graph.factors:
        assert options[factor.option_i].kind != "noop"
        assert options[factor.option_j].kind != "noop"


def test_preflight_gate_rejects_rejected_layout_without_bypass(tmp_path):
    report = [{"layout_name": "unit", "accepted": False, "num_valid_options": 1}]
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(preflight_path=str(path))
    with pytest.raises(RuntimeError, match="failed preflight"):
        train_aris._enforce_preflight_gate("unit", {"preflight": {}}, args)


def test_preflight_gate_requires_report_path():
    args = SimpleNamespace(preflight_path=None)
    with pytest.raises(ValueError, match="requires an accepted preflight"):
        train_aris._enforce_preflight_gate("unit", {"preflight": {}}, args)


def test_preflight_gate_accepts_only_accepted_report(tmp_path):
    report = [{"layout_name": "unit", "accepted": True, "num_valid_options": 9}]
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(preflight_path=str(path))
    result = train_aris._enforce_preflight_gate("unit", {"preflight": {}}, args)
    assert result["accepted"] is True
    assert result["formal_experiment"] is True


def test_local_return_uses_shaped_reward_and_training_cost_scale():
    rows = [
        ce_sampler.OptionReplayRow(
            layout="unit",
            episode_id=0,
            t_option=0,
            ego_option=0,
            partner_option=1,
            partner_option_dist=None,
            partner_option_confidence=1.0,
            state_key="s0",
            duration=1,
            reward_sum=1.0,
            shaped_reward_sum=4.0,
            realized_cost=10.0,
            local_return_h=0.0,
            reward_to_go=0.0,
            event_summary={},
            partner_name="p",
            partner_id=0,
        )
    ]
    ce_sampler.compute_local_returns(
        rows,
        gamma=0.99,
        horizon=1,
        cost_coef=0.02,
        shaped_reward_coef=0.5,
    )
    assert rows[0].local_return_h == pytest.approx(1.0 + 0.5 * 4.0 - 0.02 * 10.0)


def test_graph_builder_excludes_noop_from_factor_support():
    options = [
        _option(0, kind="noop"),
        _option(1, entity_ids=("a",)),
        _option(2, entity_ids=("b",)),
    ]
    ce = np.asarray(
        [
            [0.0, 10.0, 9.0],
            [8.0, 0.0, 7.0],
            [6.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph = build_graph_variant(
        "complete_option_graph",
        "unit",
        options,
        ce,
        eta=0.0,
        max_factors=10,
    )
    assert graph.num_factors > 0
    assert all(options[f.option_i].kind != "noop" for f in graph.factors)
    assert all(options[f.option_j].kind != "noop" for f in graph.factors)


def test_training_progress_summary_counts_task_events():
    summary = train_aris._empty_progress_summary()
    ingredient_event = _event(
        ego_inventory_before=0,
        ego_inventory_after=(1 << 2),
    )
    event = _event(
        pot_became_full=True,
        pot_became_ready=True,
        plate_picked=False,
        soup_picked=True,
        correct_delivery=True,
        collision_or_block=True,
    )
    train_aris._accumulate_progress_summary(summary, ingredient_event)
    train_aris._accumulate_progress_summary(summary, event)
    assert summary["picked_ingredient"] == 1
    assert summary["ingredient_delivered_to_pot"] == 1
    assert summary["pot_became_ready"] == 1
    assert summary["soup_picked"] == 1
    assert summary["correct_delivery"] == 1
    assert summary["collision_or_block"] == 1
