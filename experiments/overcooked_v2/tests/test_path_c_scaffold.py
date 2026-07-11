from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from experiments.overcooked_v2.baselines.belief_filter import (
    BayesianHMMBeliefFilter,
    HMMFilterConfig,
)
from experiments.overcooked_v2.path_c_config import (
    PATH_C_RUNTIME_SCHEMA_VERSION,
    default_path_c_config,
    normalize_path_c_config,
)
from experiments.overcooked_v2.path_c_split import (
    SPLIT_ROLES,
    SplitGroupV1,
    SplitManifestV1,
)
from experiments.overcooked_v2.partner_pool import (
    PATH_C_SYNTHETIC_IDENTITIES,
    PATH_C_SYNTHETIC_MECHANISMS,
    PATH_C_SYNTHETIC_PROTOCOLS,
)
from experiments.overcooked_v2.residual_signature import (
    normalized_advantage_disagreement,
    normalized_advantage_signature,
)


def _independent_groups(count: int = 4) -> tuple[SplitGroupV1, ...]:
    return tuple(
        SplitGroupV1(
            group_id=f"mechanism-0-group-{index}",
            mechanism="mechanism-0",
            identity_group=f"identity-{index}",
            style_group=f"style-{index}",
            seed_group=f"seed-{index}",
            layout_group=f"layout-template-{index}",
            layout_stratum="shared-layout-control",
        )
        for index in range(count)
    )


def test_path_c_version_3_default_config_is_inactive():
    config = {"training": {}}
    section = normalize_path_c_config(config)
    assert section["schema_version"] == PATH_C_RUNTIME_SCHEMA_VERSION
    assert section["active_sections"] == []
    assert section["resolved_path_c_sha256"] is None
    assert section["ensemble"]["architecture"] == "legacy_default_off"
    assert section["ensemble"]["n_heads"] == 1
    assert section["ensemble"]["disagreement_target"] == "normalized_advantage"
    assert section["probe"]["enable"] is False
    assert section["probe"]["collection_enable"] is False
    assert section["belief_kernel_audit"]["enable"] is False
    assert section["primary_endpoint"]["enable"] is False


def test_path_c_normalization_is_idempotent_for_checkpoint_reload():
    config = {"path_c": default_path_c_config()}
    first = normalize_path_c_config(config)
    first_copy = copy.deepcopy(first)
    second = normalize_path_c_config(config)
    assert second == first_copy

    config["path_c"]["resolved_path_c_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match recomputation"):
        normalize_path_c_config(config)


def test_default_split_contract_has_four_roles_and_four_group_minimum():
    section = default_path_c_config()
    assert tuple(section["split"]["roles"]) == SPLIT_ROLES
    assert section["split"]["minimum_groups_per_mechanism"] == 4
    assert section["split"]["preferred_groups_per_mechanism"] == 5
    assert section["split"]["primary_shift"] == "identity"
    assert section["split"]["secondary_shift"] == "layout"


@pytest.mark.parametrize(
    "legacy_key",
    ("rv_summary_spec", "cross_identity_split", "pass_af"),
)
def test_legacy_runtime_keys_are_rejected_without_aliases(legacy_key):
    with pytest.raises(ValueError, match="explicit version-3 migration"):
        normalize_path_c_config({"path_c": {legacy_key: {"enable": True}}})


def test_active_path_c_requires_a_frozen_preregistration_path():
    with pytest.raises(ValueError, match="preregistration_path"):
        normalize_path_c_config(
            {
                "path_c": {
                    "evidence_spec": {"enable": True},
                    "ensemble": {"architecture": "recurrent_sequence_v1"},
                }
            }
        )


def test_belief_audit_config_separates_exact_and_bounded_approximate_modes():
    with pytest.raises(ValueError, match="zero posterior and reset bias"):
        normalize_path_c_config(
            {
                "path_c": {
                    "belief_kernel_audit": {
                        "exact_mode": True,
                        "posterior_bias_bound": 0.1,
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="positive posterior bias bound"):
        normalize_path_c_config(
            {
                "path_c": {
                    "belief_kernel_audit": {
                        "exact_mode": False,
                        "tier1_hypothesis_prune": 0.01,
                        "posterior_bias_bound": 0.0,
                    }
                }
            }
        )


def test_normalized_advantage_is_the_primary_offset_invariant_code():
    q_values = torch.tensor(
        [
            [[5.0, 4.0, 100.0], [5.0, 1.0, -100.0]],
        ]
    )
    valid = torch.tensor([[True, True, False]])
    signature = normalized_advantage_signature(q_values, option_mask=valid)
    head_offsets = torch.tensor([[[100.0], [-50.0]]])
    shifted = normalized_advantage_signature(
        q_values + head_offsets,
        option_mask=valid,
    )
    assert torch.equal(
        signature["normalized_advantage"],
        shifted["normalized_advantage"],
    )
    assert torch.equal(
        signature["normalized_advantage"],
        torch.tensor([[[0.0, -1.0, 0.0], [0.0, -4.0, 0.0]]]),
    )


def test_probe_disagreement_uses_normalized_advantage_and_valid_support():
    heads = torch.tensor(
        [
            [[5.0, 4.0, 100.0], [5.0, 1.0, -100.0]],
        ]
    )
    valid = torch.tensor([[True, True, False]])
    disagreement = normalized_advantage_disagreement(
        heads,
        option_mask=valid,
        stat="variance",
    )
    assert int(disagreement["per_option"].argmax(dim=-1).item()) == 1
    assert disagreement["per_option"][0, 2].item() == 0.0
    assert disagreement["normalized_advantage"][0, 0, 0].item() == 0.0
    assert disagreement["normalized_advantage"][0, 1, 0].item() == 0.0


def test_four_independent_identity_groups_cover_the_four_roles():
    manifest = SplitManifestV1.build(_independent_groups(4), manifest_seed=7)
    assert set(manifest.assignment_by_group.values()) == set(SPLIT_ROLES)
    assert manifest.independent_group_count_by_mechanism == {"mechanism-0": 4}
    assert manifest.preferred_spare_satisfied_by_mechanism == {
        "mechanism-0": False
    }
    role_by_identity: dict[str, set[str]] = {}
    for group in manifest.groups:
        role_by_identity.setdefault(group.identity_group, set()).add(
            manifest.role_for(group.group_id)
        )
    assert all(len(roles) == 1 for roles in role_by_identity.values())


def test_synthetic_factorial_crosses_three_mechanisms_five_styles_and_two_fingerprints():
    assert len(PATH_C_SYNTHETIC_MECHANISMS) == 3
    assert len(PATH_C_SYNTHETIC_IDENTITIES) == 5
    assert len(PATH_C_SYNTHETIC_PROTOCOLS) == 3 * 5 * 2
    cells: dict[tuple[str, int | None], set[tuple[str | None, int | None]]] = {}
    for _name, spec in PATH_C_SYNTHETIC_PROTOCOLS:
        mechanism = (spec.mode.family, spec.mode.param)
        cells.setdefault(mechanism, set()).add((spec.style_id, spec.fingerprint_id))
    assert len(cells) == 3
    assert all(len(realizations) == 5 * 2 for realizations in cells.values())


def test_split_fails_closed_below_four_independent_identity_groups():
    with pytest.raises(ValueError, match="at least four"):
        SplitManifestV1.build(_independent_groups(3))


def test_bayesian_filter_updates_toward_the_matching_observation_mode():
    belief_filter = BayesianHMMBeliefFilter(
        HMMFilterConfig(
            mode_names=("yield", "claim"),
            transition_stay_prob=0.9,
        ),
        {"partner_waited": {"yield": 0.9, "claim": 0.1}},
    )
    prior = belief_filter.reset()
    posterior = belief_filter.update("partner_waited")
    assert np.isclose(prior.sum(), 1.0)
    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[0] > posterior[1]
