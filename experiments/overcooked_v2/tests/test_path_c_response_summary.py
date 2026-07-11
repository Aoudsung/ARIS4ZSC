from __future__ import annotations

import pytest

from experiments.overcooked_v2.path_c_response_summary import (
    RESPONSE_SUMMARY_SCHEMA_VERSION,
    STRUCTURED_MULTILABEL_ROLE,
    ResponseSummarySpecV1,
)


def _spec() -> ResponseSummarySpecV1:
    return ResponseSummarySpecV1(
        response_classes=("wait", "help", "block"),
        latency_bin_upper_bounds=(0, 2, 5),
    )


def test_response_summary_has_stable_finite_vocabulary_and_hash():
    spec = _spec()
    assert spec.vocabulary[:4] == (
        "terminal",
        "censored",
        "invalid_script",
        "support_violation",
    )
    assert spec.q == 4 + 3 * 4
    assert len(spec.vocabulary) == len(set(spec.vocabulary))
    assert hash(spec) == hash(_spec())
    assert spec.sha256 == _spec().sha256


def test_response_summary_mapping_round_trip_is_strict_and_hash_stable():
    spec = _spec()
    restored = ResponseSummarySpecV1.from_mapping(spec.to_mapping())
    assert restored == spec
    assert restored.sha256 == spec.sha256

    payload = spec.to_mapping()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown key"):
        ResponseSummarySpecV1.from_mapping(payload)


def test_response_summary_rejects_missing_keys_and_wrong_version():
    payload = _spec().to_mapping()
    del payload["latency_bin_upper_bounds"]
    with pytest.raises(ValueError, match="missing required key"):
        ResponseSummarySpecV1.from_mapping(payload)

    payload = _spec().to_mapping()
    payload["schema_version"] = "path_c_response_summary_v2"
    with pytest.raises(ValueError, match="schema_version"):
        ResponseSummarySpecV1.from_mapping(payload)


def test_response_summary_latency_bins_use_preregistered_inclusive_edges():
    spec = _spec()
    assert spec.decode(spec.encode("wait", 0)).endswith("latency_le_0")
    assert spec.decode(spec.encode("wait", 1)).endswith("latency_le_2")
    assert spec.decode(spec.encode("wait", 2)).endswith("latency_le_2")
    assert spec.decode(spec.encode("wait", 5)).endswith("latency_le_5")
    assert spec.decode(spec.encode("wait", 6)).endswith("latency_gt_5")


@pytest.mark.parametrize(
    ("flag", "expected_token"),
    [
        ("terminal", "terminal"),
        ("censored", "censored"),
        ("invalid_script", "invalid_script"),
        ("support_violation", "support_violation"),
    ],
)
def test_response_summary_special_outcomes_have_explicit_token_ids(
    flag: str,
    expected_token: str,
):
    spec = _spec()
    token_id = spec.encode(**{flag: True})
    assert spec.decode(token_id) == expected_token
    assert token_id == spec.token_ids[expected_token]


def test_response_summary_rejects_ambiguous_or_unregistered_responses():
    spec = _spec()
    with pytest.raises(ValueError, match="at most one"):
        spec.encode(terminal=True, censored=True)
    with pytest.raises(ValueError, match="cannot also carry"):
        spec.encode("wait", 1, terminal=True)
    with pytest.raises(ValueError, match="requires both"):
        spec.encode(response_class="wait")
    with pytest.raises(ValueError, match="Unregistered"):
        spec.encode("escalate", 1)
    with pytest.raises(ValueError, match="non-negative"):
        spec.encode("wait", -1)


def test_response_summary_rejects_unfrozen_or_invalid_latency_bins():
    with pytest.raises(TypeError, match="sequence of labels"):
        ResponseSummarySpecV1("wait", (1,))
    with pytest.raises(ValueError, match="non-empty"):
        ResponseSummarySpecV1(("wait",), ())
    with pytest.raises(ValueError, match="strictly increasing"):
        ResponseSummarySpecV1(("wait",), (1, 1))
    with pytest.raises(ValueError, match="integers"):
        ResponseSummarySpecV1(("wait",), (1, 2.5))


def test_structured_multilabel_role_cannot_be_promoted_to_primary():
    with pytest.raises(ValueError, match="secondary-only"):
        ResponseSummarySpecV1(
            response_classes=("wait",),
            latency_bin_upper_bounds=(1,),
            schema_version=RESPONSE_SUMMARY_SCHEMA_VERSION,
            structured_multilabel_role="primary",
        )
    assert _spec().structured_multilabel_role == STRUCTURED_MULTILABEL_ROLE
    assert _spec().canonical_payload()[
        "structured_multilabel_theorem_eligible"
    ] is False


def test_response_summary_hash_binds_classes_and_latency_bins():
    base = _spec()
    changed_classes = ResponseSummarySpecV1(
        response_classes=("wait", "help", "defer"),
        latency_bin_upper_bounds=(0, 2, 5),
    )
    changed_bins = ResponseSummarySpecV1(
        response_classes=("wait", "help", "block"),
        latency_bin_upper_bounds=(0, 3, 5),
    )
    assert len({base.sha256, changed_classes.sha256, changed_bins.sha256}) == 3
