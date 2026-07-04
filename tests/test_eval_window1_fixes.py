"""Window-1 eval-fix regressions (latent-defect sweep, SWEEP_SUMMARY.md).

Covers:
  F1 / LDS-B2 — ``_enforce_reward_scale`` hard gate (fail closed, escape hatch);
  F2 / LDS-B1 — eval return accounting excludes terminal_progress_shaping while
                training call sites (default) are bit-unchanged;
  F4 / LDS-C3 — canonical throughput fields with frozen denominator and
                None-not-zero semantics for undefined values.

Write-only per project convention: heavy-stack imports are guarded so the file
COLLECTS everywhere and RUNS where the env stack exists (remote venv).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

# --- guarded imports (heavy env stack) -------------------------------------
try:
    from experiments.overcooked_v2.evaluate_aris import (
        _enforce_reward_scale,
        _throughput_fields,
    )

    _EVAL_OK = True
except Exception:  # pragma: no cover - import guard
    _EVAL_OK = False

try:
    from experiments.overcooked_v2 import train_aris as _ta
    from experiments.overcooked_v2.train_aris import _training_reward

    _TRAIN_OK = True
except Exception:  # pragma: no cover - import guard
    _TRAIN_OK = False


# ===================== F1 / LDS-B2: reward-scale hard gate ==================


def _status(verified: bool) -> dict:
    return {"reward_scale_verified": verified, "detail": "synthetic"}


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_reward_scale_gate_passes_when_all_verified():
    _enforce_reward_scale(
        {"full_support": _status(True), "ablation": _status(True)},
        allow_unverified=False,
    )  # must not raise


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_reward_scale_gate_hard_fails_on_any_unverified_variant():
    with pytest.raises(RuntimeError, match="full_support"):
        _enforce_reward_scale(
            {"full_support": _status(False), "ablation": _status(True)},
            allow_unverified=False,
        )


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_reward_scale_gate_escape_hatch_downgrades_to_warning(capsys):
    _enforce_reward_scale(
        {"full_support": _status(False)},
        allow_unverified=True,
    )  # must not raise
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_reward_scale_gate_fails_closed_on_missing_flag():
    # A status dict WITHOUT the boolean must be treated as unverified.
    with pytest.raises(RuntimeError):
        _enforce_reward_scale({"full_support": {}}, allow_unverified=False)


# ============ F2 / LDS-B1: eval return excludes terminal shaping ============


def _reward_inputs():
    """Minimal synthetic (step, config, event) isolating the terminal bonus."""
    step = SimpleNamespace(rewards={"agent_0": 0.0}, info={})
    config = {
        "layout": "synthetic",
        "training": {
            "sparse_credit": "team",  # passthrough: sparse term = team reward = 0
            "shaped_reward_coef": 0.0,  # isolate: no dense shaping term
            "terminal_progress_shaping": {
                "enabled": True,
                "ego_plate_pick_bonus": 0.5,
                "ego_plate_soup_bonus": 1.5,
                "ego_serve_bonus": 0.0,
                "max_bonus_per_step": 1.5,
            },
        },
    }
    event = SimpleNamespace()  # attributes resolved via getattr defaults
    return step, config, event


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_eval_return_excludes_terminal_shaping(monkeypatch):
    """include_terminal_shaping=False must drop EXACTLY the terminal bonus."""
    step, config, event = _reward_inputs()
    monkeypatch.setattr(_ta, "terminal_progress_bonus", lambda *a, **k: 7.0)
    train_side = _training_reward(step, config, "agent_0", event)
    eval_side = _training_reward(
        step, config, "agent_0", event, include_terminal_shaping=False
    )
    assert train_side == pytest.approx(eval_side + 7.0)
    assert eval_side == pytest.approx(0.0)


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_training_default_includes_terminal_shaping(monkeypatch):
    """Preservation: the default (train call sites) still adds the bonus."""
    step, config, event = _reward_inputs()
    monkeypatch.setattr(_ta, "terminal_progress_bonus", lambda *a, **k: 7.0)
    assert _training_reward(step, config, "agent_0", event) == pytest.approx(7.0)


@pytest.mark.skipif(not _TRAIN_OK, reason="train_aris stack unavailable")
def test_shaping_disabled_configs_bitwise_unaffected():
    """Golden: with shaping disabled (E1-family configs) both accounting modes
    agree bit-for-bit — the LDS-B1 fix cannot move any E1 number."""
    step, config, event = _reward_inputs()
    config["training"]["terminal_progress_shaping"]["enabled"] = False
    on = _training_reward(step, config, "agent_0", event)
    off = _training_reward(
        step, config, "agent_0", event, include_terminal_shaping=False
    )
    assert on == off


# =============== F4 / LDS-C3: canonical throughput fields ===================


def _counts(correct=6, ego=2, partner=4) -> dict:
    return {
        "correct_delivery": correct,
        "ego_correct_delivery": ego,
        "partner_correct_delivery": partner,
    }


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_throughput_fields_values_and_denominator():
    fields = _throughput_fields(_counts(), n_episodes=4)
    assert fields["team_correct_delivery_throughput_per_episode"] == pytest.approx(1.5)
    assert fields["ego_correct_delivery_throughput_per_episode"] == pytest.approx(0.5)
    assert fields["ego_serve_share"] == pytest.approx(2 / 6)


@pytest.mark.skipif(not _EVAL_OK, reason="evaluate_aris stack unavailable")
def test_throughput_undefined_is_none_not_zero():
    # No deliveries at all: serve share is UNDEFINED (None), never 0.0.
    fields = _throughput_fields(_counts(correct=0, ego=0, partner=0), n_episodes=4)
    assert fields["ego_serve_share"] is None
    assert fields["team_correct_delivery_throughput_per_episode"] == 0.0  # measured zero
    # Zero episodes: every rate is undefined.
    empty = _throughput_fields(_counts(), n_episodes=0)
    assert empty["team_correct_delivery_throughput_per_episode"] is None
    assert empty["ego_correct_delivery_throughput_per_episode"] is None
