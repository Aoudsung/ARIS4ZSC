from types import SimpleNamespace

from experiments.overcooked_v2.event_extractor import EVENT_SEMANTICS_VERSION
from experiments.overcooked_v2.train_aris import _graph_objective_metadata_status


def test_graph_objective_metadata_status_verifies_reward_scale_metadata() -> None:
    config = {
        "layout": "unit_layout",
        "training": {
            "cost_coef": 0.2,
            "cost_per_step": 0.01,
            "shaped_reward_coef": 1.5,
        },
    }
    metadata = {
        "layout": "unit_layout",
        "cost_coef": 0.2,
        "cost_per_step": 0.01,
        "shaped_reward_coef": 1.5,
        "event_semantics_version": int(EVENT_SEMANTICS_VERSION),
    }

    verified_graph = SimpleNamespace(layout_name="unit_layout", metadata=metadata)
    missing_metadata_graph = SimpleNamespace(layout_name="unit_layout", metadata={})

    assert _graph_objective_metadata_status(verified_graph, config)["reward_scale_verified"] is True
    assert _graph_objective_metadata_status(missing_metadata_graph, config)["reward_scale_verified"] is False
