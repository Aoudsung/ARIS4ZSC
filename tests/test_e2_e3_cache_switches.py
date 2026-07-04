"""Phase-2 static-engineering regression tests (E2 zeroed channel, E3 belief
persistence switch, reference-baseline cache).

Write-only per the project convention (like ``test_actor_sparse_credit.py``):
the belief/eval stack pulls JaxMARL, so the JaxMARL-dependent cases are guarded
with import flags — they COLLECT everywhere and RUN where the env stack exists
(the remote venv). The pure-logic assertions (gate admit/reject, cache key
determinism, persistence-flag reading) carry the load.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# --- guarded imports (heavy env stack) -------------------------------------
try:
    from experiments.overcooked_v2.option_inferencer import (
        PartnerOptionInferencer,
        make_behavior_option_inferencer,
    )

    _INFER_OK = True
except Exception:  # pragma: no cover - import guard
    _INFER_OK = False

try:
    from experiments.overcooked_v2.evaluate_aris import (
        _baseline_cache_target,
        _evidence_policy_for_config,
        _read_baseline_cache,
        _validate_eval_integrity,
        _write_baseline_cache,
    )

    _EVAL_OK = True
except Exception:  # pragma: no cover - import guard
    _EVAL_OK = False

try:
    from src.aris_bellman.replay import EvidenceBuffer
    from experiments.overcooked_v2.evidence_router import D_EVID
    from experiments.overcooked_v2.train_aris import (
        _advance_persistent_belief,
        _belief_persistence_enabled,
        _initialise_persistent_belief,
    )

    _TRAIN_OK = True
except Exception:  # pragma: no cover - import guard
    _TRAIN_OK = False


# ============================ E2: zeroed channel ============================


@pytest.mark.skipif(not _INFER_OK, reason="option_inferencer stack unavailable")
def test_zeroed_inferencer_withholds_option_intent():
    """mode='zeroed' emits the observable primitive action but NO option intent."""
    inf = PartnerOptionInferencer(option_library=SimpleNamespace(), mode="zeroed")
    action = inf.update(prev_state=None, primitive_action=3, next_state=None, event=None)
    assert action.option_id is None
    assert action.option_dist is None
    assert float(action.option_confidence) == 0.0
    assert action.source == "zeroed_partner_option"
    assert int(action.primitive_action) == 3  # observable behavior preserved


@pytest.mark.skipif(not _INFER_OK, reason="option_inferencer stack unavailable")
def test_inferencer_rejects_bad_mode():
    with pytest.raises(ValueError):
        PartnerOptionInferencer(option_library=SimpleNamespace(), mode="bogus")


@pytest.mark.skipif(not _INFER_OK, reason="option_inferencer stack unavailable")
def test_zeroed_is_eval_only_on_training_path():
    """require_inferred=True (training/CE) must REJECT a mode:zeroed config so the
    ablation can never silently zero evidence during training (review BLOCK fix)."""
    zeroed_cfg = {"evidence": {"partner_option_inference": {"mode": "zeroed"}}}
    with pytest.raises(ValueError):
        make_behavior_option_inferencer(SimpleNamespace(), zeroed_cfg, require_inferred=True)
    # eval path (require_inferred=False, default) still allows zeroed
    inf = make_behavior_option_inferencer(SimpleNamespace(), zeroed_cfg)
    assert inf.mode == "zeroed"


def _aggregate(policy: str, **evidence_counts):
    ev = {
        "evidence_policy": policy,
        "observed_dist_count": 0,
        "missing_count": 0,
        "oracle_source_count": 0,
        "zeroed_count": 0,
    }
    ev.update(evidence_counts)
    return {"forced_noop_count": 0, "partner_option_evidence": ev}


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_gate_admits_zeroed_ablation_policy():
    """A zeroed run (withheld intent → zeroed_count>0, no oracle) must PASS the gate."""
    agg = _aggregate("behavior_inferred_v1_zeroed_ablation", zeroed_count=40)
    _validate_eval_integrity(agg, collect_diagnostics=False, allow_diag_skip=False)


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_gate_still_rejects_oracle_under_zeroed_policy():
    """Admitting the ablation string must NOT weaken the oracle-leak hard check."""
    agg = _aggregate(
        "behavior_inferred_v1_zeroed_ablation", zeroed_count=40, oracle_source_count=1
    )
    with pytest.raises(RuntimeError):
        _validate_eval_integrity(agg, collect_diagnostics=False, allow_diag_skip=False)


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_gate_rejects_unknown_policy():
    with pytest.raises(RuntimeError):
        _validate_eval_integrity(
            _aggregate("something_else"), collect_diagnostics=False, allow_diag_skip=False
        )


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_gate_normal_policy_still_passes():
    _validate_eval_integrity(
        _aggregate("behavior_inferred_v1"), collect_diagnostics=False, allow_diag_skip=False
    )


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_evidence_policy_for_config():
    assert _evidence_policy_for_config({}) == "behavior_inferred_v1"
    zeroed = {"evidence": {"partner_option_inference": {"mode": "zeroed"}}}
    assert _evidence_policy_for_config(zeroed) == "behavior_inferred_v1_zeroed_ablation"
    inferred = {"evidence": {"partner_option_inference": {"mode": "inferred"}}}
    assert _evidence_policy_for_config(inferred) == "behavior_inferred_v1"


# ======================== E3: belief persistence switch =====================


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_belief_persistence_flag_default_and_off():
    assert _belief_persistence_enabled({}) is True  # default = current P4 behavior
    assert _belief_persistence_enabled({"training": {}}) is True
    assert _belief_persistence_enabled({"training": {"belief_persistence": False}}) is False


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_persistence_off_never_carries_hidden():
    """persistent=False must leave the buffer stateless (pre-P4 zero-reencode)."""
    buf = EvidenceBuffer(num_factors=2, window=4, evidence_dim=int(D_EVID))
    # init off: no hidden set even for an ARIS method
    _initialise_persistent_belief(buf, "aris_bellman", None, None, None, False)
    assert buf.belief_hidden_snapshot() is None
    # advance off: no window base recorded → snapshot stays None → zeros re-encode
    row = np.zeros((2, int(D_EVID)), dtype=np.float32)
    _advance_persistent_belief(buf, "aris_bellman", None, None, None, row, False)
    assert buf.belief_window_base_snapshot() is None


# ============================ reference-baseline cache ======================


def _cfg(**over):
    cfg = {
        "layout": "asymm_advantages",  # reward_config_payload requires config['layout']
        "env": {"max_steps": 200, "negative_rewards": True, "force_path_planning": False},
        "options": {"max_option_steps": 16, "strict_preconditions": True, "dynamic_budget": True},
        "training": {
            "partner_set": "role_conditioned_v2",
            "cost_coef": 0.02,
            "cost_per_step": 1.0,
            "shaped_reward_coef": 1.0,
        },
    }
    cfg.update(over)
    return cfg


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_cache_key_deterministic_and_sensitive(tmp_path: Path):
    common = dict(
        kind="random_policy",
        config=_cfg(),
        partner="p1",
        episodes=5,
        seed=0,
        max_episode_options=20,
    )
    p1, k1 = _baseline_cache_target(tmp_path, layout="asymm", **common)
    p2, k2 = _baseline_cache_target(tmp_path, layout="asymm", **common)
    assert p1 == p2 and k1 == k2  # deterministic

    def _flip(**changes):
        c = dict(common)
        c.update(changes)
        return _baseline_cache_target(tmp_path, layout="asymm", **c)[0]

    # every result-affecting input must flip the key
    assert _flip(config={**_cfg(), "env": {**_cfg()["env"], "max_steps": 400}}) != p1
    assert _flip(config={**_cfg(), "options": {**_cfg()["options"], "max_option_steps": 8}}) != p1
    assert _flip(config=_cfg(training={"partner_set": "standard7", "cost_coef": 0.02,
                                       "cost_per_step": 1.0, "shaped_reward_coef": 1.0})) != p1
    assert _flip(max_episode_options=40) != p1
    assert _flip(partner="p2") != p1
    assert _flip(seed=1) != p1
    assert _baseline_cache_target(tmp_path, layout="cramped", **common)[0] != p1
    # disabled cache → (None, None)
    assert _baseline_cache_target(None, layout="asymm", **common) == (None, None)


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_cache_key_sensitive_to_reward_variant(tmp_path: Path):
    """LDS-C2 (latent-defect sweep): sparse-credit and terminal-progress reward
    variants must NOT share baseline cache entries — a contrib_team+scaffold
    config (E1-rev) and a team/no-scaffold config differ in rollout return
    accounting."""
    common = dict(
        kind="random_policy",
        config=_cfg(),
        partner="p1",
        episodes=5,
        seed=0,
        max_episode_options=20,
    )
    base_path, _ = _baseline_cache_target(tmp_path, layout="asymm", **common)

    def _with_training(**extra):
        cfg = _cfg()
        cfg["training"] = {**cfg["training"], **extra}
        c = dict(common)
        c["config"] = cfg
        return _baseline_cache_target(tmp_path, layout="asymm", **c)[0]

    # sparse-credit mode flips the key
    assert _with_training(sparse_credit="contrib_team") != base_path
    # contrib_scale (inside the resolved sparse-credit signature) flips the key
    assert _with_training(
        sparse_credit="contrib_team", contrib_team={"contrib_scale": 0.5}
    ) != _with_training(sparse_credit="contrib_team")
    # terminal-progress shaping flips the key even though eval returns exclude
    # it since LDS-B1 (over-keying is deliberate: misses are safe, aliasing not)
    assert _with_training(
        terminal_progress_shaping={
            "enabled": True,
            "ego_plate_pick_bonus": 0.5,
            "ego_plate_soup_bonus": 1.5,
            "ego_serve_bonus": 0.0,
            "max_bonus_per_step": 1.5,
        }
    ) != base_path


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_cache_roundtrip_key_and_corruption(tmp_path: Path):
    path = tmp_path / "random_policy_abc.json"
    key = "abc"
    assert _read_baseline_cache(path, key) is None  # miss on absent
    entry = {"base_kind": "random_policy", "mean_return": 1.25}
    _write_baseline_cache(path, key, entry)
    assert _read_baseline_cache(path, key) == entry  # hit, value preserved
    assert _read_baseline_cache(path, "other_key") is None  # wrong key → miss (stale guard)
    path.write_text("{ not json", encoding="utf-8")
    assert _read_baseline_cache(path, key) is None  # corrupted → recompute
    assert _read_baseline_cache(None, key) is None  # disabled → None
