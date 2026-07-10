from __future__ import annotations

import copy
import numpy as np
from pathlib import Path
import pytest
import torch
import yaml

from experiments.overcooked_v2.baselines.belief_filter import (
    BayesianHMMBeliefFilter,
    HMMFilterConfig,
)
from experiments.overcooked_v2.path_c_config import normalize_path_c_config
from experiments.overcooked_v2.residual_signature import (
    residual_control_signature,
    residual_signature_disagreement,
)
from experiments.overcooked_v2.scripts.diag_d1_train import (
    _path_c_g_value_measurement,
    path_c_go_no_go_rule,
)
from src.aris_bellman.td import aris_td_loss


def _write_frozen_prereg(path, *, sections=()):
    del sections
    template = Path(__file__).resolve().parents[1] / "configs" / "path_c_preregistration.yaml"
    body = yaml.safe_load(template.read_text(encoding="utf-8"))
    body["status"] = "frozen"
    body["version"] = "test-path-c"
    body["freeze_timestamp"] = "2026-07-09T00:00:00Z"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _active_runtime_config(prereg_path: Path) -> dict:
    payload = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    runtime = {"preregistration_path": str(prereg_path)}
    fields = {
        "ensemble": ("n_heads", "bootstrap_p", "prior_scale", "disagreement_stat", "tie_atol"),
        "probe": (
            "enable", "eval_enable", "collection_enable", "rule",
            "disagreement_threshold", "return_floor", "base_checkpoint_sha256",
            "require_public_residual_baseline",
            "allow_raw_q_fallback", "min_selected_probes", "min_probe_opportunities",
            "min_context_coverage", "min_action_coverage",
        ),
        "rv_summary_spec": ("enable",),
        "fingerprint": ("enable", "kind", "vocab", "positive_control_kind", "negative_control_kind"),
        "cross_identity_split": ("enable", "group_key", "identity_key", "split", "folds"),
        "pass_af": ("enable",),
    }
    for section, names in fields.items():
        runtime[section] = {name: copy.deepcopy(payload[section][name]) for name in names}
    runtime["probe"]["base_checkpoint"] = str(prereg_path.parent / "base_only.pt")
    return {"path_c": runtime}


def test_path_c_default_config_is_inactive():
    cfg = {"training": {}}
    section = normalize_path_c_config(cfg)
    assert section["active_sections"] == []
    assert cfg["path_c"]["ensemble"]["n_heads"] == 1
    assert not cfg["path_c"]["probe"]["enable"]
    assert not cfg["path_c"]["probe"].get("eval_enable", False)


def test_residual_signature_subtracts_public_baseline_for_all_control_stats():
    q = torch.tensor([[10.0, 9.0, -1.0]])
    base = torch.tensor([[9.0, 0.0, -1.0]])
    out = residual_control_signature(q, base)
    assert torch.equal(out["residual_q"], torch.tensor([[1.0, 9.0, 0.0]]))
    assert torch.equal(out["best_option"], torch.tensor([1]))
    assert torch.equal(out["gap"], torch.tensor([8.0]))
    assert torch.equal(out["advantage"], torch.tensor([[-8.0, 0.0, -9.0]]))


def test_residual_signature_disagreement_can_differ_from_raw_q_variance():
    heads = torch.tensor(
        [[
            [-3.3643987, -6.0135937, -12.196077, 4.912762],
            [5.892592, 7.3932333, -2.436348, 1.8451736],
            [0.78176594, -5.334225, 1.6146172, -6.6889353],
        ]]
    )
    base = torch.tensor([[-5.588356, -2.5692625, 3.932838, -7.768317]])
    raw_argmax = int(heads.var(dim=1, unbiased=False).squeeze(0).argmax().item())
    residual = residual_signature_disagreement(heads, base)
    residual_argmax = int(residual["per_option"].squeeze(0).argmax().item())
    assert raw_argmax == 1
    assert residual_argmax == 2


def test_bayesian_hmm_filter_updates_toward_matching_mode():
    filt = BayesianHMMBeliefFilter(
        HMMFilterConfig(mode_names=("yield", "claim"), transition_stay_prob=0.9),
        {"partner_waited": {"yield": 0.9, "claim": 0.1}},
    )
    prior = filt.reset()
    post = filt.update("partner_waited")
    assert np.isclose(prior.sum(), 1.0)
    assert np.isclose(post.sum(), 1.0)
    assert post[0] > post[1]


def test_preregistration_required_when_path_c_feature_is_enabled():
    with pytest.raises(ValueError, match="preregistration_path"):
        normalize_path_c_config({"path_c": {"ensemble": {"n_heads": 2, "bootstrap_p": 0.5}}})


def test_draft_preregistration_is_rejected_when_feature_is_enabled(tmp_path):
    prereg = tmp_path / "path_c_preregistration.yaml"
    _write_frozen_prereg(prereg)
    payload = yaml.safe_load(prereg.read_text(encoding="utf-8"))
    payload["status"] = "proposed_defaults_pending_user_signoff"
    prereg.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="status: frozen"):
        normalize_path_c_config(_active_runtime_config(prereg))


def test_frozen_rv_summary_spec_binds_preregistration_without_yaml_path(tmp_path):
    prereg = _write_frozen_prereg(tmp_path / "path_c_preregistration.yaml", sections=("rv_summary_spec",))
    cfg = _active_runtime_config(prereg)
    section = normalize_path_c_config(cfg)
    assert section["rv_summary_spec"]["enable"] is True
    assert section["rv_summary_spec"]["path"] is None
    assert section["preregistration"]["status"] == "frozen"
    assert section["preregistration"]["sha256"]



def test_path_c_active_ensemble_requires_bootstrap_and_randomized_prior(tmp_path):
    prereg = _write_frozen_prereg(tmp_path / "path_c_preregistration.yaml", sections=("ensemble",))
    config = _active_runtime_config(prereg)
    config["path_c"]["ensemble"]["prior_scale"] = 0.0
    with pytest.raises(ValueError, match="bootstrap_p < 1.0 and prior_scale > 0.0"):
        normalize_path_c_config(config)
    config = _active_runtime_config(prereg)
    config["path_c"]["ensemble"]["bootstrap_p"] = 1.0
    with pytest.raises(ValueError, match="bootstrap_p < 1.0 and prior_scale > 0.0"):
        normalize_path_c_config(config)

def test_probe_enable_requires_public_residual_baseline_checkpoint(tmp_path):
    prereg = _write_frozen_prereg(tmp_path / "path_c_preregistration.yaml", sections=("ensemble", "probe"))
    config = _active_runtime_config(prereg)
    config["path_c"]["probe"]["base_checkpoint"] = None
    with pytest.raises(ValueError, match="base_checkpoint"):
        normalize_path_c_config(config)


def test_probe_choice_noops_when_probe_is_disabled():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.train_aris import _path_c_probe_choice

    class RaisingQ(torch.nn.Module):
        def forward_heads(self, *args, **kwargs):
            raise AssertionError("probe-disabled path must not inspect heads")

    choice = _path_c_probe_choice(
        RaisingQ(),
        torch.zeros(1, 2),
        torch.zeros(1, 1, 1),
        {},
        torch.zeros(3),
        torch.ones(3, dtype=torch.bool),
        {"path_c": {"probe": {"enable": False}, "ensemble": {"n_heads": 1}}},
        partner_id=None,
        device=torch.device("cpu"),
        selection_stats={},
    )
    assert choice is None


def test_probe_choice_noops_for_single_head_even_when_enabled():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.train_aris import _path_c_probe_choice

    class RaisingQ(torch.nn.Module):
        def forward_heads(self, *args, **kwargs):
            raise AssertionError("single-head path must not inspect heads")

    stats: dict[str, int] = {}
    choice = _path_c_probe_choice(
        RaisingQ(),
        torch.zeros(1, 2),
        torch.zeros(1, 1, 1),
        {},
        torch.zeros(3),
        torch.ones(3, dtype=torch.bool),
        {"path_c": {"probe": {"enable": True}, "ensemble": {"n_heads": 1}}},
        partner_id=None,
        device=torch.device("cpu"),
        selection_stats=stats,
    )
    assert choice is None
    assert stats["path_c_probe_skipped_single_head_count"] == 1
    assert stats["path_c_probe_last_decision"]["reason"] == "disabled"


def test_probe_choice_uses_residual_signature_disagreement_target():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.train_aris import _path_c_probe_choice

    heads = torch.tensor(
        [[
            [-3.3643987, -6.0135937, -12.196077, 4.912762],
            [5.892592, 7.3932333, -2.436348, 1.8451736],
            [0.78176594, -5.334225, 1.6146172, -6.6889353],
        ]]
    )
    base = torch.tensor([[-5.588356, -2.5692625, 3.932838, -7.768317]])

    class FakeQ(torch.nn.Module):
        def forward_heads(self, *args, **kwargs):
            return heads

    stats: dict[str, object] = {}
    choice = _path_c_probe_choice(
        FakeQ(),
        torch.zeros(1, 2),
        torch.zeros(1, 1, 1),
        {},
        torch.zeros(4),
        torch.ones(4, dtype=torch.bool),
        {
            "path_c": {
                "probe": {
                    "enable": True,
                    "disagreement_threshold": 0.0,
                    "return_floor": -100.0,
                    "rule": "max_residual_signature_disagreement",
                },
                "ensemble": {"n_heads": 3, "disagreement_stat": "variance", "tie_atol": 1.0e-6},
            }
        },
        partner_id=None,
        device=torch.device("cpu"),
        selection_stats=stats,
        base_q_values=base,
    )
    assert choice == 2
    assert stats["path_c_probe_last_decision"]["selected"] is True
    assert stats["path_c_probe_last_decision"]["target"] == "residual_control_signature_disagreement"


def test_randomized_prior_has_independent_frozen_encoder_and_head():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.train_aris import EnsembleArisBellmanQNetwork
    from src.aris_bellman.specs import FactorSpec, GraphSpec, OptionSpec

    graph = GraphSpec(
        layout_name="test",
        options=[OptionSpec(0, "noop", "noop", None, None, (), (), 1)],
        factors=[FactorSpec(0, 0, 0, 0.0, 1, (), ())],
        relevance=np.ones((1, 1), dtype=bool),
        option_mask=np.ones(1, dtype=bool),
        factor_mask=np.ones(1, dtype=bool),
        mode_mask=np.ones((1, 1), dtype=bool),
        route_map={0: (0,)},
    )
    model = EnsembleArisBellmanQNetwork(
        4,
        8,
        graph,
        n_heads=2,
        prior_scale=0.1,
    )
    assert model.prior_encoder is not model.encoder
    assert all(not parameter.requires_grad for parameter in model.prior_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.prior_heads.parameters())
    model.train()
    assert model.prior_encoder.training is False
    assert model.prior_heads.training is False


def test_path_c_synthetic_protocols_counterbalance_fingerprint_within_mechanism():
    try:
        from experiments.overcooked_v2.partner_pool import (
            PATH_C_FINGERPRINT_NEGATIVE_PROTOCOLS,
            PATH_C_SYNTHETIC_PROTOCOLS,
        )
    except ModuleNotFoundError as exc:  # optional JaxMARL dependency in local static test env
        pytest.skip(str(exc))

    by_mechanism: dict[tuple[str, int], list[tuple[str, object]]] = {}
    for name, spec in PATH_C_SYNTHETIC_PROTOCOLS:
        key = (spec.mode.family, int(spec.mode.param))
        by_mechanism.setdefault(key, []).append((name, spec))
    assert by_mechanism
    for specs in by_mechanism.values():
        fingerprints = {int(spec.fingerprint_id) for _name, spec in specs}
        identities = {
            (str(spec.geometry_profile), str(spec.base_protocol.role), str(spec.base_protocol.pot_preference))
            for _name, spec in specs
        }
        assert fingerprints == {0, 1}
        assert len(identities) >= 2
        for fid in fingerprints:
            assert sum(int(spec.fingerprint_id) == fid for _name, spec in specs) >= 2
            assert all(spec.fingerprint_bias == "retreat_order_rotate_candidate" for _name, spec in specs)
    assert PATH_C_FINGERPRINT_NEGATIVE_PROTOCOLS
    assert all(
        spec.fingerprint_bias == "metadata_only_value_null"
        for _name, spec in PATH_C_FINGERPRINT_NEGATIVE_PROTOCOLS
    )


def test_rv_summary_spec_excludes_absolute_coordinates_and_source_labels():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.scripts.diag_d1_dataset import RV_COLUMNS, _default_rv_summary_spec

    spec = _default_rv_summary_spec()
    assert "resp_rv_value_event_target_x" not in RV_COLUMNS
    assert "resp_rv_value_event_target_y" not in RV_COLUMNS
    forbidden_terms = {"identity", "seed", "style", "trajectory_source", "absolute_target_coordinates"}
    assert forbidden_terms.issubset(set(spec["excludes"]))
    assert all(not str(col).endswith(("_x", "_y")) for col in spec["rv_columns"])


def test_effective_episode_count_uses_records_not_requested_intent():
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.scripts.diag_d1_dataset import _effective_episode_count

    arrays = {"episode_id": np.asarray([0, 0, 2, 2, 2], dtype=np.int32)}
    assert _effective_episode_count(arrays) == 2
    assert _effective_episode_count({"episode_id": np.asarray([], dtype=np.int32)}) == 0


def test_path_c_go_no_go_rejects_unbound_threshold_mapping():
    passed = {"pass": True, "value": 1.0}
    measurements = {
        "effective_episodes": 2000,
        "effective_transitions": 40000,
        "path_c_baseline_completeness": {"pass": True},
        "C_resp^V": passed,
        "G_value": passed,
        "C_transfer": passed,
        "C_leak": passed,
        "C_null": passed,
    }
    with pytest.raises(TypeError, match="FrozenPathCPreregistration"):
        path_c_go_no_go_rule(
            measurements,
            {"epsilon_F": 0.1, "epsilon_R": 0.4, "_preregistration_status": "frozen"},
        )


def test_path_c_g_value_rejects_rv_derived_proxy_columns():
    data = {
        "episode_id": np.asarray([0, 1, 2], dtype=np.int32),
        "heldout_value_gap_advantage": np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        "action_ranking_advantage": np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    }
    out = _path_c_g_value_measurement(data, np.asarray([True, True, True]))
    assert out["pass"] is False
    assert out["available"] is False
    assert "heldout_value_gap_advantage" in out["rejected_proxy_columns_present"]
    assert "true held-out value-control" in out["reason"]


def test_single_head_ensemble_td_loss_matches_plain_td_loss():
    class PlainQ(torch.nn.Module):
        def __init__(self, q_values: torch.Tensor):
            super().__init__()
            self.register_buffer("q_values", q_values)

        def forward(self, obs_feat, belief, option_mask=None, **kwargs):
            del belief, kwargs
            out = self.q_values[: obs_feat.shape[0]].clone()
            if option_mask is not None:
                out = out.masked_fill(~option_mask.bool(), -1e9)
            return out

    class SingleHeadQ(PlainQ):
        def forward_heads(self, obs_feat, belief, option_mask=None, **kwargs):
            return self.forward(
                obs_feat,
                belief,
                option_mask=option_mask,
                **kwargs,
            ).unsqueeze(1)

    q_values = torch.tensor([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]])
    graph_batch = {
        "option_mask": torch.tensor([[True, True, False], [True, False, True]]),
        "option_mask_next": torch.tensor([[True, True, False], [True, False, True]]),
        "factor_mask": torch.ones(2, 1, dtype=torch.bool),
        "mode_mask": torch.ones(2, 1, dtype=torch.bool),
        "relevance_mask": torch.ones(2, 3, 1, dtype=torch.bool),
    }
    kwargs = dict(
        obs_feat_t=torch.zeros(2, 4),
        belief_t=torch.zeros(2, 1, 1),
        option_id=torch.tensor([1, 2]),
        reward_sum=torch.tensor([1.0, 0.5]),
        realized_cost=torch.tensor([0.0, 0.25]),
        duration=torch.tensor([1.0, 2.0]),
        obs_feat_next=torch.zeros(2, 4),
        belief_next=torch.zeros(2, 1, 1),
        done=torch.tensor([0.0, 1.0]),
        graph_batch=graph_batch,
        gamma=0.9,
        cost_coef=0.1,
        td_loss="mse",
        double_q=True,
    )
    plain_loss = aris_td_loss(PlainQ(q_values), PlainQ(q_values), **kwargs)
    head_loss = aris_td_loss(
        SingleHeadQ(q_values),
        SingleHeadQ(q_values),
        bootstrap_mask=torch.ones(2, 1, dtype=torch.bool),
        **kwargs,
    )
    assert torch.allclose(head_loss, plain_loss)
