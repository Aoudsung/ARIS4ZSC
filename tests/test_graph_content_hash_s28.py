"""S28 regression: graph identity compared content-to-content.

Incident (2026-07-04, E1-rev stage-1 launch): the LDS-B2 hard reward-scale gate
refused every eval because the checkpoint-embedded graph's provenance record is
stamped AFTER runtime metadata additions (formal_experiment / graph_source /
preflight_gate), so the recorded graph_json_sha256 can never equal the on-disk
file's hash on the eval path. Forensics showed the graphs were content-identical
(same hash after stripping runtime bookkeeping, file mtime predating every
checkpoint). Fix: _graph_provenance_status recomputes BOTH sides with
graph_content_hash (runtime keys stripped); real factor/CE/semantic-metadata
drift still mismatches.
"""
from __future__ import annotations

import json

import pytest

try:
    from experiments.overcooked_v2.provenance import (
        GRAPH_RUNTIME_METADATA_KEYS,
        graph_content_hash,
    )

    _PROV_OK = True
except Exception:  # pragma: no cover - import guard
    _PROV_OK = False

try:
    from experiments.overcooked_v2.train_aris import _graph_provenance_status

    _TRAIN_OK = True
except Exception:  # pragma: no cover - import guard
    _TRAIN_OK = False


def _graph_dict(**metadata_extra):
    return {
        "factors": [{"option_i": 1, "option_j": 2, "ce": 3.5}],
        "metadata": {
            "layout": "asymm_advantages",
            "sparse_credit": "contrib_team",
            "provenance": {"graph_json_sha256": "stale-record"},
            **metadata_extra,
        },
    }


@pytest.mark.skipif(not _PROV_OK, reason="provenance stack unavailable")
def test_content_hash_ignores_runtime_bookkeeping_keys():
    base = graph_content_hash(_graph_dict())
    stamped = graph_content_hash(
        _graph_dict(
            formal_experiment=True,
            graph_source="graph_path",
            preflight_gate={"accepted": True},
            provenance={"graph_json_sha256": "totally-different-record"},
        )
    )
    assert base == stamped  # runtime stamps must not change graph identity


@pytest.mark.skipif(not _PROV_OK, reason="provenance stack unavailable")
def test_content_hash_catches_semantic_drift():
    base = graph_content_hash(_graph_dict())
    assert graph_content_hash(
        {**_graph_dict(), "factors": [{"option_i": 1, "option_j": 2, "ce": 9.9}]}
    ) != base  # CE drift
    assert graph_content_hash(_graph_dict(sparse_credit="team")) != base  # semantic metadata drift


class _StubGraph:
    """Duck-typed GraphSpec: _graph_provenance_status only touches .metadata
    and .to_json_dict()."""

    def __init__(self, payload):
        self._payload = payload
        self.metadata = payload.get("metadata", {})

    def to_json_dict(self):
        return self._payload


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_provenance_status_passes_for_runtime_stamped_checkpoint_graph(tmp_path):
    """The incident scenario: file on disk vs checkpoint graph that differs
    ONLY by runtime-stamped metadata + a stale provenance record ⇒ no
    graph_json_sha256 mismatch."""
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_graph_dict()))
    ckpt_graph = _StubGraph(
        _graph_dict(
            formal_experiment=True,
            graph_source="graph_path",
            preflight_gate={"accepted": True},
            provenance={"graph_json_sha256": "stale-runtime-restamp"},
        )
    )
    status = _graph_provenance_status(
        ckpt_graph,
        {"layout": "asymm_advantages",
         "training": {"cost_coef": 0.02, "cost_per_step": 1.0, "shaped_reward_coef": 1.0},
         "graph": {"graph_path": str(graph_file)}},
    )
    assert "graph_json_sha256" not in status["mismatches"]


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_provenance_status_still_catches_real_graph_drift(tmp_path):
    """Preservation of the gate's teeth: a file whose CONTENT drifted from the
    in-use graph must still mismatch."""
    graph_file = tmp_path / "graph.json"
    drifted = _graph_dict()
    drifted["factors"] = [{"option_i": 7, "option_j": 9, "ce": 1.1}]
    graph_file.write_text(json.dumps(drifted))
    ckpt_graph = _StubGraph(_graph_dict(formal_experiment=True))
    status = _graph_provenance_status(
        ckpt_graph,
        {"layout": "asymm_advantages",
         "training": {"cost_coef": 0.02, "cost_per_step": 1.0, "shaped_reward_coef": 1.0},
         "graph": {"graph_path": str(graph_file)}},
    )
    assert "graph_json_sha256" in status["mismatches"]


@pytest.mark.skipif(not _PROV_OK, reason="provenance stack unavailable")
def test_runtime_key_set_is_frozen_and_minimal():
    assert GRAPH_RUNTIME_METADATA_KEYS == {
        "provenance",
        "formal_experiment",
        "graph_source",
        "preflight_gate",
    }
