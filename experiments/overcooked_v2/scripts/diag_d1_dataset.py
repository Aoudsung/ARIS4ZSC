"""D1 dataset generator — held-out response-predictability diagnostic (prereg D1 §2-§3).

One partner/controller pair is collected per invocation. Version-3 Path C data
is written as one content-addressed Parquet shard plus metadata; the historical
NPZ output remains available only for non-Path-C diagnostics.
Substrate diagnostic ONLY: no training loss touches the main method; rollouts ride the
exact eval execution semantics (E._select_option / option_primitive_step / behavior-only
inferencer), so the logged partner-event stream is the same public channel
(behavior_inferred_v1, oracle_source_count must stay 0) the method itself consumes.

Per ego option-decision point we record:
  - public state features (featurizer agent_0 view) + pot/inventory opportunity signals
  - partner public event history (run-length-encoded inferred option kinds + delivery
    punctuation events), truncated to the most recent --hist-window events
  - the ego option kind actually executed
  - labels (D1-rev, prereg §9): first TERMINAL-CHAIN INITIATION attribution
    (no_initiation / ego_first / partner_first) within the next K in {3,5,8} ego option
    decisions. Initiation = the agent ACQUIRES the soup (inventory transitions to
    carrying a plated soup — raw state bits, NOT inferred kinds) or correct-delivers
    (strict D1 attribution). Same-step double initiation is AMBIGUOUS and the affected
    windows are flagged for exclusion (fraction reported).
  - opportunity-onset gates (D1-rev): main = (any pot ready OR cooking) AND neither
    agent carries soup; strict = any pot ready AND neither carries soup (co-report)
  - step-level audit arrays (audit_*) are stored per chunk for offline relabeling

Ego modes (all share ONE anchor checkpoint context; trained weights influence rollouts
only in argmax mode):
  argmax    : anchor checkpoint policy as-is (use a base_only ckpt per prereg)
  fullchain : scripted priority full task chain (probe_behavior semantics), noop tail
  random    : uniform over valid options
  prepchain : full chain minus terminal kinds (deferring competent ego; Link-A r2 —
              reaction-family defer triggers only appear under a deferring ego)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import yaml
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.baselines.belief_filter import (  # noqa: E402
    BayesianHMMBeliefFilter,
    HMMFilterConfig,
)
from experiments.overcooked_v2.option_executor import option_primitive_step  # noqa: E402
from experiments.overcooked_v2.path_c_config import path_c_metadata  # noqa: E402
from experiments.overcooked_v2.path_c_evaluation import (  # noqa: E402
    PATH_C_PROBE_PROVENANCE_IDS,
    PATH_C_PROBE_RULE_IDS,
    canonical_sha256,
    load_frozen_preregistration,
    make_episode_uid,
    make_run_id,
)
from experiments.overcooked_v2.path_c_response_summary import (  # noqa: E402
    ResponseSummarySpecV1,
)
from experiments.overcooked_v2.path_c_sequence import (  # noqa: E402
    DecisionEvidenceBuffer,
    EgoEvidenceSpecV1,
    EpisodeEvidenceBuffer,
    ensemble_diversity_telemetry,
)
from experiments.overcooked_v2.partner_modes import (  # noqa: E402
    ID_TO_FAMILY,
    ID_TO_POLICY,
    ID_TO_TRIGGER,
)
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402
from experiments.overcooked_v2.residual_signature import (  # noqa: E402
    compute_residual_control_signature,
    normalized_advantage_signature,
    residual_control_signature,
)
from experiments.overcooked_v2.state_utils import (  # noqa: E402
    get_cell_extra,
    get_inventory,
    has_plate,
    is_cooked,
    is_pot_cooking,
    is_pot_ready_for_plate,
)

# Same deliberate-play order as probe_behavior.py (sec18.14 certification ego).
KNOWN_ORDER = [
    "serve_soup", "plate_soup", "pick_plate", "deliver_ingredient_to_pot",
    "fetch_ingredient", "wait_duration_after_arrival", "wait_duration",
    "wait_at_bottleneck", "cross_bottleneck", "handoff_counter",
    "press_recipe_button", "clear_interaction_cell", "drop_item_to_counter",
]

PAD, UNK = "<PAD>", "<UNK>"
EV_PARTNER_DELIV_OK = "PARTNER_DELIVERY_CORRECT"
EV_PARTNER_DELIV_WRONG = "PARTNER_DELIVERY_WRONG"
ORACLE_LIKE = ("oracle", "scripted", "terminal_policy", "protocol", "partner_action")
K_WINDOWS = (3, 5, 8)
RV_RAW_COLUMNS = (
    "resp_raw_ego_commit_count",
    "resp_raw_partner_commit_count",
    "resp_raw_ambig_count",
    "resp_raw_response_primitive_len",
    "resp_raw_partner_commitment_seq",
    "resp_raw_delivery_outcome_seq",
)
RV_COLUMNS = (
    "resp_rv_value_event_partner_commitment",
    "resp_rv_value_event_target_object",
    "resp_rv_value_event_response_latency",
    "resp_rv_value_event_defer",
    "resp_rv_value_event_escalate",
    "resp_rv_value_event_wait",
    "resp_rv_value_event_help",
    "resp_rv_value_event_block",
    "resp_rv_value_event_delivery_outcome",
)
# step delivery codes
D_NONE, D_EGO, D_PARTNER, D_AMBIG = 0, 1, 2, 3
RV_DELIVERY_NONE = 0
RV_DELIVERY_EGO_CORRECT = 1
RV_DELIVERY_PARTNER_CORRECT = 2
RV_DELIVERY_AMBIG_CORRECT = 3
RV_DELIVERY_EGO_WRONG = 4
RV_DELIVERY_PARTNER_WRONG = 5
RV_DELIVERY_AMBIG_WRONG = 6

RV_COMMITMENT_TO_ID = {
    "none": 0,
    "terminal": 1,
    "prep": 2,
    "support": 3,
    "wait": 4,
    "block": 5,
}
RV_TARGET_OBJECT_TO_ID = {
    "none": 0,
    "ingredient": 1,
    "pot": 2,
    "plate": 3,
    "soup": 4,
    "delivery": 5,
    "button": 6,
    "counter": 7,
    "bottleneck": 8,
}
RV_DELIVERY_OUTCOME_TO_ID = {
    "none": RV_DELIVERY_NONE,
    "ego_correct": RV_DELIVERY_EGO_CORRECT,
    "partner_correct": RV_DELIVERY_PARTNER_CORRECT,
    "ambiguous_correct": RV_DELIVERY_AMBIG_CORRECT,
    "ego_wrong": RV_DELIVERY_EGO_WRONG,
    "partner_wrong": RV_DELIVERY_PARTNER_WRONG,
    "ambiguous_wrong": RV_DELIVERY_AMBIG_WRONG,
}
RV_WAIT_KINDS = frozenset({
    "wait_duration_after_arrival",
    "wait_duration",
    "wait_at_bottleneck",
    "noop",
})
RV_PREP_KINDS = frozenset({
    "fetch_ingredient",
    "deliver_ingredient_to_pot",
    "pick_plate",
    "plate_soup",
})
RV_TERMINAL_KINDS = frozenset({"serve_soup"})
RV_SUPPORT_KINDS = frozenset({
    "handoff_counter",
    "drop_item_to_counter",
    "clear_interaction_cell",
})
RV_BLOCK_KINDS = frozenset({"cross_bottleneck", "wait_at_bottleneck"})
PATH_C_PROBE_VARIANT = "probing_ego"
PATH_C_Q_BASELINE_VARIANTS = (
    "global_gru",
    "base_only",
    "partner_id",
    "rnn_residualized",
    "random_probe",
    "no_probe",
    "no_admission",
    "unpruned_ensemble",
)
PATH_C_REPR_VARIANTS = (
    PATH_C_PROBE_VARIANT,
    *PATH_C_Q_BASELINE_VARIANTS,
    "belief_filter",
)
PATH_C_SIGNATURE_SCALARS = (
    "gap",
    "advantage_action",
    "residual_q_action",
    "residual_q_best",
    "residual_q_gap",
    "best_option",
    "action_rank_score",
    "ensemble_effective_rank",
    "head_correlation",
    "head_correlation_absolute",
    "prior_contribution_ratio",
)
PATH_C_VALUE_ADVANTAGE_COLUMNS = (
    "heldout_value_gap_advantage",
    "action_ranking_advantage",
    "residual_q_advantage",
)
PATH_C_VALUE_QUALITY_KINDS = (
    "heldout_value_gap",
    "action_ranking",
    "residual_q",
)
PATH_C_CORRECT_DELIVERY_CODES = frozenset({
    RV_DELIVERY_EGO_CORRECT,
    RV_DELIVERY_PARTNER_CORRECT,
    RV_DELIVERY_AMBIG_CORRECT,
})
PATH_C_BELIEF_FILTER_MODES = ("terminal", "prep", "support", "wait", "block")
PATH_C_PROBE_REASON_TO_ID = PATH_C_PROBE_PROVENANCE_IDS
PROBE_RULE_TO_ID = PATH_C_PROBE_RULE_IDS
FINGERPRINT_CONTROL_KIND_TO_ID = {
    "none": -1,
    "metadata_only": 0,
    "raw_response_value_null": 1,
}


def _probe_reason_id(reason: str) -> int:
    key = str(reason)
    if key not in PATH_C_PROBE_REASON_TO_ID:
        raise ValueError(f"unknown Path C probe provenance reason: {key!r}")
    return int(PATH_C_PROBE_REASON_TO_ID[key])


def _probe_rule_id(rule: str) -> int:
    key = str(rule)
    if key not in PROBE_RULE_TO_ID:
        raise ValueError(f"unknown Path C probe rule: {key!r}")
    return int(PROBE_RULE_TO_ID[key])


def _stable_small_id(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:6], 16) % 32767


def _surface_identity_key(partner: Any) -> str:
    spec = getattr(partner, "spec", None)
    if spec is None:
        return canonical_sha256({"partner_class": type(partner).__name__})[:24]
    explicit = getattr(spec, "surface_identity_key", None)
    if explicit not in {None, ""}:
        return str(explicit)
    payload = _jsonable_dataclass(spec)
    if isinstance(payload, dict):
        payload = {
            "geometry_profile": payload.get("geometry_profile"),
            "base_protocol": payload.get("base_protocol"),
        }
    return canonical_sha256(payload)[:24]


def _frozen_split_assignment(
    config: dict[str, Any],
    partner: Any,
    *,
    layout_name: str,
    split_group_id: str | None,
    numeric_seed: int,
) -> dict[str, Any]:
    """Resolve the collection role and group fields from the frozen manifest."""

    if split_group_id in {None, ""}:
        raise ValueError(
            "Version-3 Path C collection requires --path-c-split-group-id."
        )
    preregistration_path = (config.get("path_c") or {}).get(
        "preregistration_path"
    )
    if preregistration_path in {None, ""}:
        raise ValueError(
            "Version-3 Path C collection requires a frozen preregistration path."
        )
    preregistration = load_frozen_preregistration(preregistration_path)
    manifest = preregistration.split_manifest
    groups = {group.group_id: group for group in manifest.groups}
    group_id = str(split_group_id)
    if group_id not in groups:
        raise ValueError(
            f"Split group {group_id!r} is absent from the frozen split manifest."
        )
    group = groups[group_id]
    manifest.validate_numeric_seed(group_id, int(numeric_seed))
    partner_spec = getattr(partner, "spec", None)
    mechanism = getattr(partner_spec, "value_class_id", None)
    style_group = getattr(partner_spec, "style_id", None)
    identity_group = _surface_identity_key(partner)
    if mechanism in {None, ""}:
        raise ValueError(
            "Version-3 Path C collection partner lacks a frozen value mechanism."
        )
    if style_group in {None, ""}:
        raise ValueError(
            "Version-3 Path C collection partner lacks a frozen style group."
        )
    observed = {
        "mechanism": str(mechanism),
        "identity_group": identity_group,
        "style_group": str(style_group),
        "layout_group": str(layout_name),
        "numeric_seed": int(numeric_seed),
    }
    expected = {
        "mechanism": group.mechanism,
        "identity_group": group.identity_group,
        "style_group": group.style_group,
        "layout_group": group.layout_group,
        "numeric_seed": int(numeric_seed),
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, observed {observed[name]!r}"
        for name in expected
        if observed[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "The selected split group does not describe the collection target; "
            + "; ".join(mismatches)
        )
    return {
        "split_manifest_sha256": manifest.sha256,
        "split_group_id": group.group_id,
        "collection_role": manifest.role_for(group.group_id),
        "mechanism": group.mechanism,
        "identity_group": group.identity_group,
        "style_group": group.style_group,
        "seed_group": group.seed_group,
        "layout_group": group.layout_group,
        "numeric_seed": int(numeric_seed),
    }


def _surface_action_frequency_bin(hist_kind: np.ndarray) -> int:
    vals = np.asarray(hist_kind, dtype=np.int64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    counts = np.bincount(vals)
    return int(counts.argmax())


def _full_priority(option_lib) -> list[str]:
    present = list(dict.fromkeys(str(o.kind) for o in option_lib.options))
    ordered = [k for k in KNOWN_ORDER if k in present]
    extras = [k for k in present if k not in KNOWN_ORDER and k != "noop"]
    return ordered + extras + (["noop"] if "noop" in present else [])


def _kind_vocab(option_lib) -> list[str]:
    kinds = sorted(set(str(o.kind) for o in option_lib.options))
    return [PAD, UNK, EV_PARTNER_DELIV_OK, EV_PARTNER_DELIV_WRONG] + kinds


def _carries_soup(inv: int) -> bool:
    return is_cooked(inv) and has_plate(inv)


def _mode_diag(partner) -> dict[str, int]:
    fn = getattr(partner, "diagnostic_mode_state", None)
    if not callable(fn):
        return {
            "mode_policy_id": -1,
            "mode_family_id": -1,
            "mode_param": -1,
            "mode_age": -1,
            "mode_opportunity_count": -1,
            "mode_last_trigger_id": -1,
            "mode_fingerprint_id": -1,
            "mode_fingerprint_control_kind_id": -1,
        }
    diag = fn()
    return {
        "mode_policy_id": int(diag.get("mode_policy_id", -1)),
        "mode_family_id": int(diag.get("mode_family_id", -1)),
        "mode_param": int(diag.get("mode_param", -1)),
        "mode_age": int(diag.get("mode_age", -1)),
        "mode_opportunity_count": int(diag.get("mode_opportunity_count", -1)),
        "mode_last_trigger_id": int(diag.get("mode_last_trigger_id", -1)),
        "mode_fingerprint_id": int(diag.get("mode_fingerprint_id", -1)),
        "mode_fingerprint_control_kind_id": int(FINGERPRINT_CONTROL_KIND_TO_ID.get(
            str(diag.get("mode_fingerprint_control_kind", "none")),
            -1,
        )),
    }


def _jsonable_dataclass(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return None


def _pot_signals(state, pot_positions) -> tuple[int, int, float]:
    # D1-rev (prereg §9): gate readiness is ANY-recipe ready — wrong-recipe soup
    # acquisition counts as initiation, so wrong-recipe-ready pots must gate too
    # (codex D1-rev review, blocker 1). Correct-recipe count is a separate feature.
    n_ready = sum(
        1 for p in pot_positions
        if is_pot_ready_for_plate(state, p, require_correct_recipe=False)
    )
    n_cooking = sum(1 for p in pot_positions if is_pot_cooking(state, p))
    timers = [int(get_cell_extra(state, p)) for p in pot_positions]
    pos_timers = [t for t in timers if t > 0]
    min_timer = float(np.log1p(min(pos_timers))) if pos_timers else 0.0
    return int(n_ready), int(n_cooking), min_timer


def _step_delivery_code(event) -> int:
    """Strict first-delivery attribution (codex finding 1).

    ego counts only when the delivery is unambiguously ego's
    (ego_sole_correct_delivery = ego delivered AND partner did not AND correct);
    partner counts only when ego did not deliver in the same step. A same-step
    double delivery with any correct component is AMBIGUOUS.
    """
    ego_deliv = bool(event.ego_delivery_event)
    partner_deliv = bool(event.partner_delivery_event)
    any_correct = bool(event.ego_correct_delivery) or bool(event.partner_correct_delivery)
    if ego_deliv and partner_deliv and any_correct:
        return D_AMBIG
    if bool(event.ego_sole_correct_delivery):
        return D_EGO
    if bool(event.partner_correct_delivery) and not ego_deliv:
        return D_PARTNER
    return D_NONE


def _step_delivery_outcome_code(event, delivery_code: int) -> int:
    if delivery_code == D_AMBIG:
        return RV_DELIVERY_AMBIG_CORRECT
    if delivery_code == D_EGO:
        return RV_DELIVERY_EGO_CORRECT
    if delivery_code == D_PARTNER:
        return RV_DELIVERY_PARTNER_CORRECT
    ego_wrong = bool(getattr(event, "ego_wrong_delivery_event", False))
    partner_wrong = bool(getattr(event, "partner_wrong_delivery_event", False))
    if ego_wrong and partner_wrong:
        return RV_DELIVERY_AMBIG_WRONG
    if ego_wrong:
        return RV_DELIVERY_EGO_WRONG
    if partner_wrong:
        return RV_DELIVERY_PARTNER_WRONG
    return RV_DELIVERY_NONE


def _rv_commitment_id(kind: str | None) -> int:
    if kind is None:
        return RV_COMMITMENT_TO_ID["none"]
    if kind in RV_TERMINAL_KINDS:
        return RV_COMMITMENT_TO_ID["terminal"]
    if kind in RV_PREP_KINDS:
        return RV_COMMITMENT_TO_ID["prep"]
    if kind in RV_SUPPORT_KINDS:
        return RV_COMMITMENT_TO_ID["support"]
    if kind in RV_WAIT_KINDS:
        return RV_COMMITMENT_TO_ID["wait"]
    if kind in RV_BLOCK_KINDS:
        return RV_COMMITMENT_TO_ID["block"]
    return RV_COMMITMENT_TO_ID["support"]


def _rv_target_object_id(option) -> int:
    if option is None:
        return RV_TARGET_OBJECT_TO_ID["none"]
    kind = str(option.kind)
    if kind == "fetch_ingredient":
        return RV_TARGET_OBJECT_TO_ID["ingredient"]
    if kind == "deliver_ingredient_to_pot":
        return RV_TARGET_OBJECT_TO_ID["pot"]
    if kind == "pick_plate":
        return RV_TARGET_OBJECT_TO_ID["plate"]
    if kind == "plate_soup":
        return RV_TARGET_OBJECT_TO_ID["soup"]
    if kind == "serve_soup":
        return RV_TARGET_OBJECT_TO_ID["delivery"]
    if kind == "press_recipe_button":
        return RV_TARGET_OBJECT_TO_ID["button"]
    if kind in {"handoff_counter", "drop_item_to_counter", "clear_interaction_cell"}:
        return RV_TARGET_OBJECT_TO_ID["counter"]
    if kind in RV_BLOCK_KINDS:
        return RV_TARGET_OBJECT_TO_ID["bottleneck"]
    return RV_TARGET_OBJECT_TO_ID["none"]


def _rv_step_value_event(option, event) -> tuple[int, int, int, int, int]:
    """Return the frozen identity/style-free value-event tuple for one step.

    Absolute target coordinates are deliberately excluded from R^V because they can
    become layout/surface-style proxies. The retained target object is a public
    semantic bucket fixed by the grammar.
    """
    kind = None if option is None else str(option.kind)
    commitment = _rv_commitment_id(kind)
    target_object = _rv_target_object_id(option)
    wait = int(bool(getattr(event, "partner_waited", False)) or kind in RV_WAIT_KINDS)
    help_event = int(kind in RV_PREP_KINDS or kind in RV_SUPPORT_KINDS)
    block = int(bool(getattr(event, "collision_or_block", False)) or kind in RV_BLOCK_KINDS)
    return commitment, target_object, wait, help_event, block

def _summarize_rv_value_event_window(
    commitments: list[int],
    target_objects: list[int],
    waits: list[int],
    helps: list[int],
    blocks: list[int],
    delivery_outcomes: list[int],
) -> dict[str, int]:
    c = np.asarray(commitments, dtype=np.int16)
    d = np.asarray(delivery_outcomes, dtype=np.int16)
    event_mask = (c > 0) | (d > 0)
    first_event_hits = np.flatnonzero(event_mask)
    first_event = int(first_event_hits[0]) if first_event_hits.size else -1
    commit_hits = np.flatnonzero(c > 0)
    first_commit = int(commit_hits[0]) if commit_hits.size else -1
    delivery_hits = np.flatnonzero(d > 0)
    first_delivery = int(delivery_hits[0]) if delivery_hits.size else -1
    wait_arr = np.asarray(waits, dtype=bool)
    help_arr = np.asarray(helps, dtype=bool)
    block_arr = np.asarray(blocks, dtype=bool)
    delivery_outcome = int(d[first_delivery]) if first_delivery >= 0 else RV_DELIVERY_NONE
    if first_commit >= 0:
        target_object = int(target_objects[first_commit])
        commitment = int(c[first_commit])
    else:
        target_object = RV_TARGET_OBJECT_TO_ID["none"]
        commitment = RV_COMMITMENT_TO_ID["none"]
    wait_before_commit = bool(wait_arr[:first_commit].any()) if first_commit > 0 else False
    has_partner_commit = first_commit >= 0 and commitment in {
        RV_COMMITMENT_TO_ID["terminal"],
        RV_COMMITMENT_TO_ID["prep"],
    }
    return {
        "resp_rv_value_event_partner_commitment": commitment,
        "resp_rv_value_event_target_object": target_object,
        "resp_rv_value_event_response_latency": first_event,
        "resp_rv_value_event_defer": int(not has_partner_commit and bool(wait_arr.any())),
        "resp_rv_value_event_escalate": int(has_partner_commit and wait_before_commit),
        "resp_rv_value_event_wait": int(bool(wait_arr.any())),
        "resp_rv_value_event_help": int(bool(help_arr.any())),
        "resp_rv_value_event_block": int(bool(block_arr.any())),
        "resp_rv_value_event_delivery_outcome": delivery_outcome,
    }


def _canonical_response_token_id(
    spec: ResponseSummarySpecV1,
    summary: dict[str, int],
    *,
    terminal: bool,
    censored: bool,
    invalid_script: bool,
    support_violation: bool,
) -> int:
    # Special outcomes are absorbing and mutually exclusive by registered
    # precedence. They cannot be combined with a regular response token.
    if support_violation:
        return spec.encode(support_violation=True)
    if invalid_script:
        return spec.encode(invalid_script=True)
    if terminal:
        return spec.encode(terminal=True)
    if censored:
        return spec.encode(censored=True)
    delivery = int(summary["resp_rv_value_event_delivery_outcome"])
    if delivery in {
        RV_DELIVERY_EGO_CORRECT,
        RV_DELIVERY_PARTNER_CORRECT,
        RV_DELIVERY_AMBIG_CORRECT,
    }:
        response_class = "delivery_success"
    elif delivery in {
        RV_DELIVERY_EGO_WRONG,
        RV_DELIVERY_PARTNER_WRONG,
        RV_DELIVERY_AMBIG_WRONG,
    }:
        response_class = "delivery_failure"
    elif int(summary["resp_rv_value_event_block"]):
        response_class = "block"
    elif int(summary["resp_rv_value_event_escalate"]):
        response_class = "escalate"
    elif int(summary["resp_rv_value_event_defer"]):
        response_class = "defer"
    elif int(summary["resp_rv_value_event_help"]):
        response_class = "help"
    elif int(summary["resp_rv_value_event_wait"]):
        response_class = "wait"
    elif int(summary["resp_rv_value_event_partner_commitment"]):
        response_class = "commitment"
    elif int(summary["resp_rv_value_event_target_object"]):
        response_class = "progress"
    else:
        response_class = "no_response"
    latency = max(0, int(summary["resp_rv_value_event_response_latency"]))
    return spec.encode(response_class=response_class, latency_steps=latency)


def _raw_value_event_sequences(
    i: int,
    n_dp: int,
    total_steps: int,
    dp_first_step: list[int],
    commitments: list[int],
    delivery_outcomes: list[int],
    *,
    window_decisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    commitment_seq = np.zeros(window_decisions, dtype=np.int16)
    delivery_seq = np.zeros(window_decisions, dtype=np.int16)
    for j in range(window_decisions):
        dp = i + j
        if dp >= n_dp:
            break
        start = dp_first_step[dp]
        end = dp_first_step[dp + 1] if dp + 1 < n_dp else total_steps
        c = np.asarray(commitments[start:end], dtype=np.int16)
        d = np.asarray(delivery_outcomes[start:end], dtype=np.int16)
        c_hits = np.flatnonzero(c > 0)
        d_hits = np.flatnonzero(d > 0)
        if c_hits.size:
            commitment_seq[j] = int(c[c_hits[0]])
        if d_hits.size:
            delivery_seq[j] = int(d[d_hits[0]])
    return commitment_seq, delivery_seq


def _default_rv_summary_spec() -> dict[str, object]:
    return {
        "source": "frozen_value_event_grammar",
        "window_decisions": int(K_WINDOWS[1]),
        "events": [
            "partner_option_commitment",
            "target_object",
            "response_latency",
            "defer",
            "escalate",
            "wait",
            "help",
            "block",
            "delivery_outcome",
        ],
        "commitment_vocab": RV_COMMITMENT_TO_ID,
        "target_object_vocab": RV_TARGET_OBJECT_TO_ID,
        "delivery_outcome_vocab": RV_DELIVERY_OUTCOME_TO_ID,
        "raw_columns": list(RV_RAW_COLUMNS),
        "rv_columns": list(RV_COLUMNS),
        "excludes": ["identity", "seed", "style", "layout_style", "trajectory_source", "absolute_target_coordinates"],
        "coordinate_policy": "absolute_target_x_y_excluded",
    }


def _rv_summary_spec(config: dict[str, Any] | None = None) -> dict[str, object]:
    """Load the structured response fields used only by secondary readouts.

    R^V must be an identity/seed/style-free deterministic value-event summary.
    The code refuses specs that include absolute target coordinates or other known
    nuisance/source labels.
    """
    spec = _default_rv_summary_spec()
    source_path = None
    if config is not None:
        path_c = config.get("path_c") or {}
        spec_path = ((path_c.get("response_summary_spec") or {}).get("path"))
        prereg_path = path_c.get("preregistration_path")
        source_path = spec_path or prereg_path
        if source_path not in {None, ""}:
            payload = yaml.safe_load(Path(str(source_path)).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"R^V spec/preregistration file must be a YAML mapping: {source_path}")
            status = str(payload.get("status", "")).strip().lower()
            if "response_summary_spec" in payload and status not in {"frozen", "locked"}:
                raise ValueError(
                    f"Path C R^V preregistration must be frozen before dataset collection; "
                    f"got status={payload.get('status')!r}."
                )
            loaded = payload.get("response_summary_spec", payload)
            if not isinstance(loaded, dict):
                raise ValueError("rv_summary_spec section must be a YAML mapping.")
            spec = dict(spec)
            for key, value in loaded.items():
                spec[key] = value
    rv_cols = [str(c) for c in spec.get("rv_columns", [])]
    forbidden = {
        "resp_rv_value_event_target_x",
        "resp_rv_value_event_target_y",
        "identity",
        "seed",
        "style",
        "layout_style",
        "trajectory_source",
    }
    bad = sorted(c for c in rv_cols if c in forbidden or c.endswith("_x") or c.endswith("_y"))
    if bad:
        raise ValueError(
            "R^V summary spec contains forbidden identity/style/coordinate column(s): "
            + ", ".join(bad)
        )
    if rv_cols and tuple(rv_cols) != tuple(RV_COLUMNS):
        raise ValueError(
            "R^V rv_columns must match the implementation columns exactly; got "
            f"{rv_cols}, expected {list(RV_COLUMNS)}."
        )
    spec["rv_columns"] = list(RV_COLUMNS)
    spec["raw_columns"] = list(RV_RAW_COLUMNS)
    spec.setdefault("excludes", [])
    excludes = set(str(x) for x in spec["excludes"])
    excludes.update({"identity", "seed", "style", "layout_style", "trajectory_source", "absolute_target_coordinates"})
    spec["excludes"] = sorted(excludes)
    if source_path not in {None, ""}:
        raw = Path(str(source_path)).read_bytes()
        spec["source_path"] = str(Path(str(source_path)).resolve())
        spec["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return spec


def _primary_response_summary_spec(
    config: dict[str, Any],
) -> tuple[ResponseSummarySpecV1, int]:
    path_c = config.get("path_c") or {}
    source_path = (
        (path_c.get("response_summary_spec") or {}).get("path")
        or path_c.get("preregistration_path")
    )
    if source_path in {None, ""}:
        raise ValueError(
            "Canonical response collection requires a frozen Path C preregistration."
        )
    payload = yaml.safe_load(Path(str(source_path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Path C response-summary source must be a YAML mapping.")
    if str(payload.get("status", "")).strip().lower() != "frozen":
        raise ValueError("Canonical response-summary collection requires status: frozen.")
    raw = payload.get("response_summary_spec")
    if not isinstance(raw, dict):
        raise ValueError("Frozen preregistration lacks response_summary_spec.")
    spec = ResponseSummarySpecV1.from_mapping({
        "schema_version": raw.get("schema_version"),
        "response_classes": raw.get("response_classes"),
        "latency_bin_upper_bounds": raw.get("latency_bin_upper_bounds"),
        "structured_multilabel_role": raw.get("structured_multilabel_role"),
    })
    window = int(raw.get("window_decisions", 0))
    if window <= 0:
        raise ValueError("response_summary_spec.window_decisions must be positive.")
    return spec, window


def _primary_evidence_spec(config: dict[str, Any]) -> EgoEvidenceSpecV1:
    path_c = config.get("path_c") or {}
    runtime_spec = (path_c.get("evidence_spec") or {}).get("frozen_spec")
    if isinstance(runtime_spec, dict):
        return EgoEvidenceSpecV1.from_dict(runtime_spec)
    source_path = path_c.get("preregistration_path")
    if source_path in {None, ""}:
        raise ValueError("EgoEvidenceSpecV1 requires a frozen preregistration.")
    return load_frozen_preregistration(source_path).evidence_spec


def _frozen_probe_cost_from_config(config: dict[str, Any]) -> float:
    """Read the only admissible Path C per-probe cost from frozen metadata."""

    frozen = (config.get("path_c") or {}).get("preregistration", {}).get(
        "probe_cost_per_use"
    )
    if frozen is None:
        raise ValueError(
            "Version-3 Path C collection requires a frozen per-probe cost."
        )
    if isinstance(frozen, (bool, np.bool_)):
        raise ValueError(
            "The frozen Path C per-probe cost must be finite and non-negative."
        )
    try:
        cost = float(frozen)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The frozen Path C per-probe cost must be finite and non-negative."
        ) from exc
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "The frozen Path C per-probe cost must be finite and non-negative."
        )
    return cost


def _progress_event_vector(
    spec: EgoEvidenceSpecV1,
    event: Any,
) -> np.ndarray:
    values = []
    for name in spec.progress_event_names:
        raw = getattr(event, name, 0.0)
        if isinstance(raw, (bool, np.bool_)):
            raw = float(raw)
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"Progress event {name!r} is not finite.")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _effective_episode_count(arrays: dict[str, np.ndarray]) -> int:
    episode_ids = arrays.get("episode_id")
    if episode_ids is None or len(episode_ids) == 0:
        return 0
    return int(np.unique(episode_ids).size)


def _backfill_held_out_episode_return(
    rows: dict[str, list],
    row_start: int,
    row_stop: int,
) -> float:
    """Validate decision rewards and fill one ecological outcome per episode."""

    start = int(row_start)
    stop = int(row_stop)
    if start < 0 or stop <= start:
        raise ValueError("An ecological episode must contain at least one decision.")
    decision_rewards = rows.get("decision_reward")
    episode_returns = rows.get("held_out_episode_return")
    if decision_rewards is None or episode_returns is None:
        raise ValueError("Ecological return columns are missing from dataset rows.")
    if len(decision_rewards) < stop or len(episode_returns) < stop:
        raise ValueError("Ecological return columns are shorter than the episode.")
    rewards = np.asarray(decision_rewards[start:stop], dtype=np.float64)
    if rewards.shape != (stop - start,) or not np.isfinite(rewards).all():
        raise ValueError("Decision rewards must be finite and aligned.")
    held_out_return = float(np.sum(rewards, dtype=np.float64))
    for row_index in range(start, stop):
        episode_returns[row_index] = np.float64(held_out_return)
    return held_out_return


def _resolve_checkpoint(path_value: str | None) -> Path | None:
    if path_value in {None, ""}:
        return None
    path = Path(str(path_value))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_path_c_context(path_value: str | None, variant: str):
    path = _resolve_checkpoint(path_value)
    if path is None:
        return None
    return E._load_context(path, variant)


def _path_c_value_contexts(args: argparse.Namespace, anchor_ctx) -> dict[str, Any] | None:
    paths = {
        PATH_C_PROBE_VARIANT: getattr(args, "path_c_probing_checkpoint", None),
        "base_only": getattr(args, "path_c_base_checkpoint", None),
        "global_gru": getattr(args, "path_c_global_gru_checkpoint", None),
        "partner_id": getattr(args, "path_c_partner_id_checkpoint", None),
        "rnn_residualized": getattr(args, "path_c_rnn_residualized_checkpoint", None),
        "random_probe": getattr(args, "path_c_random_probe_checkpoint", None),
        "no_probe": getattr(args, "path_c_no_probe_checkpoint", None),
        "no_admission": getattr(args, "path_c_no_admission_checkpoint", None),
        "unpruned_ensemble": getattr(args, "path_c_unpruned_ensemble_checkpoint", None),
    }
    requested = bool(
        any(value not in {None, ""} for value in paths.values())
        or getattr(args, "path_c_belief_filter_artifact", None) not in {None, ""}
    )
    require = bool(getattr(args, "path_c_require_value_plumbing", False))
    if not requested and not require:
        return None

    collection_variant = str(getattr(args, "path_c_collection_variant", "diagnostic_anchor"))
    if collection_variant != "diagnostic_anchor":
        contexts: dict[str, Any] = {collection_variant: anchor_ctx}
        base_ctx = _load_path_c_context(paths["base_only"], "base_only")
        if collection_variant == "base_only":
            base_ctx = anchor_ctx
        if base_ctx is not None:
            if str(base_ctx.method) != "base_only":
                raise ValueError("The optional residual diagnostic requires method='base_only'.")
            contexts["base_only"] = base_ctx
        return contexts

    probe_ctx = _load_path_c_context(paths[PATH_C_PROBE_VARIANT], PATH_C_PROBE_VARIANT)
    if probe_ctx is None:
        probe_ctx = anchor_ctx
    if str(probe_ctx.method) != "aris_bellman":
        raise ValueError(
            "Path C value plumbing requires the probing ego checkpoint to use "
            f"method='aris_bellman', got {probe_ctx.method!r}."
        )

    base_ctx = _load_path_c_context(paths["base_only"], "base_only")
    if base_ctx is None and str(anchor_ctx.method) == "base_only":
        base_ctx = anchor_ctx
    if base_ctx is not None and str(base_ctx.method) != "base_only":
        raise ValueError(
            "The optional Path C residual diagnostic requires a base_only checkpoint, got "
            f"{base_ctx.method!r}."
        )

    contexts: dict[str, Any] = {
        PATH_C_PROBE_VARIANT: probe_ctx,
    }
    if base_ctx is not None:
        contexts["base_only"] = base_ctx
    optional_methods = {
        "global_gru": {"global_gru"},
        "partner_id": {"partner_id_q"},
        # These are matched-budget ablation/baseline artifacts.  Their training
        # configuration, not a new method string, defines the variant, so any Q
        # checkpoint method that exposes the standard q_net interface is accepted
        # and recorded in metadata for audit.
        "rnn_residualized": {"global_gru"},
        "random_probe": None,
        "no_probe": None,
        "no_admission": None,
        "unpruned_ensemble": None,
    }
    for variant, expected_methods in optional_methods.items():
        loaded = _load_path_c_context(paths[variant], variant)
        if loaded is None:
            if require:
                raise ValueError(f"Path C value plumbing missing --path-c-{variant.replace('_', '-')}-checkpoint.")
            continue
        if expected_methods is not None and str(loaded.method) not in expected_methods:
            raise ValueError(
                f"Path C {variant} baseline requires method in {sorted(expected_methods)!r}, "
                f"got {loaded.method!r}."
            )
        contexts[variant] = loaded
    return contexts


def _validate_path_c_collection_args(args: argparse.Namespace, anchor: Path) -> None:
    variant = str(getattr(args, "path_c_collection_variant", "diagnostic_anchor"))
    if variant == "diagnostic_anchor":
        return
    if variant == "belief_filter":
        raise ValueError(
            "belief_filter is a secondary offline readout, not an independent "
            "acting controller. Use the hash-bound formal acting-policy factory "
            "for a belief-filter benchmark policy."
        )
    checkpoint_args = {
        "probing_ego": "path_c_probing_checkpoint",
        "base_only": "path_c_base_checkpoint",
        "global_gru": "path_c_global_gru_checkpoint",
        "partner_id": "path_c_partner_id_checkpoint",
        "rnn_residualized": "path_c_rnn_residualized_checkpoint",
        "random_probe": "path_c_random_probe_checkpoint",
        "no_probe": "path_c_no_probe_checkpoint",
        "no_admission": "path_c_no_admission_checkpoint",
        "unpruned_ensemble": "path_c_unpruned_ensemble_checkpoint",
    }
    selected_arg = checkpoint_args.get(variant)
    if selected_arg is not None:
        selected_path = _resolve_checkpoint(getattr(args, selected_arg, None))
        if selected_path is None:
            raise ValueError(
                f"Independent Path C {variant} collection requires --{selected_arg.replace('_', '-')}"
            )
        if selected_path.resolve() != anchor.resolve():
            raise ValueError(
                f"Path C {variant} collection must execute its own checkpoint as the anchor."
            )
    allowed = {selected_arg, "path_c_base_checkpoint"}
    extras = [
        name
        for name in checkpoint_args.values()
        if name not in allowed and getattr(args, name, None) not in {None, ""}
    ]
    if extras:
        raise ValueError(
            "Independent Path C collection cannot load other controller variants: "
            + ", ".join(extras)
        )


def _path_c_belief_filter(
    artifact_path: str | None = None,
    *,
    preregistration_sha256: str | None = None,
    resolved_path_c_sha256: str | None = None,
) -> BayesianHMMBeliefFilter:
    if artifact_path not in {None, ""}:
        source = Path(str(artifact_path)).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("preregistration_sha256") != preregistration_sha256:
            raise ValueError("Belief-filter artifact uses a different preregistration.")
        if payload.get("resolved_path_c_sha256") != resolved_path_c_sha256:
            raise ValueError("Belief-filter artifact uses a different runtime config.")
        if not payload.get("train_split_ids"):
            raise ValueError("Belief-filter artifact must record its training split ids.")
        mode_names = tuple(map(str, payload.get("mode_names", [])))
        likelihood = payload.get("observation_likelihood")
        if not mode_names or not isinstance(likelihood, dict):
            raise ValueError("Belief-filter artifact lacks fitted likelihood parameters.")
        return BayesianHMMBeliefFilter(
            HMMFilterConfig(
                mode_names=mode_names,
                transition_stay_prob=float(payload["transition_stay_prob"]),
                observation_smoothing=float(payload["observation_smoothing"]),
            ),
            likelihood,
        )
    modes = PATH_C_BELIEF_FILTER_MODES
    off_prob = 0.15 / float(max(len(modes) - 1, 1))
    likelihood = {
        obs: {
            mode: (0.85 if obs == mode else off_prob)
            for mode in modes
        }
        for obs in modes
    }
    likelihood["none"] = {mode: 1.0 / float(len(modes)) for mode in modes}
    return BayesianHMMBeliefFilter(
        HMMFilterConfig(mode_names=modes, transition_stay_prob=0.98),
        likelihood,
    )


def _belief_filter_observation(commitment_id: int) -> str:
    if int(commitment_id) == RV_COMMITMENT_TO_ID["terminal"]:
        return "terminal"
    if int(commitment_id) == RV_COMMITMENT_TO_ID["prep"]:
        return "prep"
    if int(commitment_id) == RV_COMMITMENT_TO_ID["support"]:
        return "support"
    if int(commitment_id) == RV_COMMITMENT_TO_ID["wait"]:
        return "wait"
    if int(commitment_id) == RV_COMMITMENT_TO_ID["block"]:
        return "block"
    return "none"


def _path_c_state_repr(ctx, obs: dict[str, np.ndarray], evidence_buffer, device):
    graph_batch = E._graph_tensors(ctx.graph, 1, device)
    evidence = E._tensor(evidence_buffer.snapshot()[None, ...], device)
    evidence_mask = E.torch.as_tensor(
        evidence_buffer.snapshot_mask()[None, ...],
        dtype=E.torch.bool,
        device=device,
    )
    evidence_lengths = E.torch.as_tensor(
        [evidence_buffer.length()],
        dtype=E.torch.float32,
        device=device,
    )
    hidden_np = evidence_buffer.belief_window_base_snapshot()
    belief_hidden = E._tensor(hidden_np[None, ...], device) if hidden_np is not None else None
    state_repr = E._state_repr(
        ctx.method,
        ctx.belief_model,
        evidence,
        graph_batch,
        evidence_lengths=evidence_lengths,
        evidence_mask=evidence_mask,
        belief_hidden=belief_hidden,
    )
    obs_tensor = E._tensor(E._obs_vector(obs, "agent_0")[None, ...], device)
    return obs_tensor, state_repr, graph_batch


def _path_c_signature_row(
    variant: str,
    ctx,
    base_ctx,
    obs: dict[str, np.ndarray],
    evidence_buffer,
    valid: np.ndarray,
    option_id: int,
    partner_id: int | None,
) -> tuple[np.ndarray, dict[str, np.float32]]:
    """Collect the primary normalized-advantage code and secondary residuals."""

    device = E.torch.device("cpu")
    valid_tensor = E.torch.as_tensor(
        valid, dtype=E.torch.bool, device=device
    ).unsqueeze(0)
    sequence_primary = E._uses_sequence_q(ctx.q_net)
    obs_tensor = None
    state_repr = None
    graph_kwargs = None
    q_extra = None
    if not sequence_primary:
        obs_tensor, state_repr, graph_batch = _path_c_state_repr(
            ctx, obs, evidence_buffer, device
        )
        graph_kwargs = E._q_forward_kwargs(graph_batch)
        graph_kwargs["option_mask"] = graph_kwargs["option_mask"] & valid_tensor
        q_extra = {
            "partner_id": E._partner_id_tensor(partner_id, 1, device),
        }
    with E.torch.no_grad():
        sequence_batch = None
        if sequence_primary:
            episode = evidence_buffer.sequence_episode()
            if not isinstance(episode, EpisodeEvidenceBuffer):
                raise RuntimeError(
                    "Path C dataset sequence readout requires a started episode history."
                )
            sequence_batch = episode.evidence_batch(device)
            head_trace, _ = ctx.q_net.forward_sequence(sequence_batch)
            head_values = head_trace[:, -1]
            q_values = head_values.mean(dim=1)
            prior_head_values = None
            if float(ctx.q_net.prior_scale) > 0.0:
                prior_head_values = ctx.q_net.forward_prior_sequence(sequence_batch)[:, -1]
        else:
            if hasattr(ctx.q_net, "forward_mean"):
                q_values = ctx.q_net.forward_mean(
                    obs_tensor, state_repr, **graph_kwargs, **q_extra
                )
                head_values = q_values.unsqueeze(1)
            else:
                raw_q_values = ctx.q_net(
                    obs_tensor, state_repr, **graph_kwargs, **q_extra
                )
                if raw_q_values.ndim == 3:
                    head_values = raw_q_values
                    q_values = raw_q_values.mean(dim=1)
                else:
                    q_values = raw_q_values
                    head_values = raw_q_values.unsqueeze(1)
            prior_head_values = None
        if q_values.ndim != 2:
            raise ValueError(
                f"Path C checkpoint {variant!r} did not return [B,A] Q values."
            )
        ensemble_telemetry = ensemble_diversity_telemetry(
            head_values,
            valid_tensor,
            prior_q_values=prior_head_values,
            prior_scale=float(getattr(ctx.q_net, "prior_scale", 0.0)),
        )
        primary = normalized_advantage_signature(
            q_values, option_mask=valid_tensor
        )
        normalized = primary["normalized_advantage"].detach()
        best_option = primary["best_option"].detach()
        best_option_set = primary["best_option_set"].detach()
        valid_1d = valid_tensor.squeeze(0)
        ranked_q = q_values.masked_fill(~valid_tensor, -E.torch.inf)
        sorted_q = E.torch.sort(ranked_q, dim=1, descending=True).values
        gap = (
            sorted_q[:, 0] - sorted_q[:, 1]
            if int(valid_1d.sum().item()) >= 2
            else E.torch.zeros(1, dtype=q_values.dtype, device=device)
        )
        action = int(option_id)
        action_advantage = normalized[0, action]
        valid_count = valid_1d.to(dtype=normalized.dtype).sum().clamp(min=1.0)
        rank_score = (
            ((normalized[0] <= action_advantage) & valid_1d)
            .to(dtype=normalized.dtype)
            .sum()
            / valid_count
        )

        nan = np.float32(np.nan)
        action_residual = nan
        residual_best = nan
        residual_gap = nan
        if base_ctx is not None:
            if E._uses_sequence_q(base_ctx.q_net):
                if sequence_batch is None:
                    episode = evidence_buffer.sequence_episode()
                    if not isinstance(episode, EpisodeEvidenceBuffer):
                        raise RuntimeError(
                            "A recurrent base diagnostic requires a started episode history."
                        )
                    sequence_batch = episode.evidence_batch(device)
                base_trace, _ = base_ctx.q_net.forward_sequence(sequence_batch)
                base_values = base_trace[:, -1].mean(dim=1)
                residual = residual_control_signature(
                    q_values,
                    base_values,
                    option_mask=valid_tensor,
                )
            elif E._uses_sequence_q(ctx.q_net):
                base_obs, base_state, base_graph_batch = _path_c_state_repr(
                    base_ctx, obs, evidence_buffer, device
                )
                base_kwargs = E._q_forward_kwargs(base_graph_batch)
                base_kwargs["option_mask"] = base_kwargs["option_mask"] & valid_tensor
                base_values = base_ctx.q_net(base_obs, base_state, **base_kwargs)
                if base_values.ndim == 3:
                    base_values = base_values.mean(dim=1)
                residual = residual_control_signature(
                    q_values,
                    base_values,
                    option_mask=valid_tensor,
                )
            else:
                residual = compute_residual_control_signature(
                    ctx.q_net,
                    base_ctx.q_net,
                    obs_tensor,
                    state_repr,
                    graph_kwargs=graph_kwargs,
                    q_extra=q_extra,
                )
            residual_q = residual["residual_q"].detach()
            valid_residual = residual_q.masked_fill(~valid_tensor, -E.torch.inf)
            sorted_residual = E.torch.sort(
                valid_residual, dim=1, descending=True
            ).values
            action_residual = np.float32(residual_q[0, action].cpu().item())
            residual_best = np.float32(
                valid_residual.max(dim=1).values.cpu().item()
            )
            residual_gap = np.float32(
                (sorted_residual[:, 0] - sorted_residual[:, 1]).cpu().item()
                if int(valid_1d.sum().item()) >= 2
                else 0.0
            )

        repr_vec = E.torch.cat(
            [
                normalized.reshape(-1),
                best_option_set.to(dtype=normalized.dtype).reshape(-1),
            ],
            dim=0,
        ).cpu().numpy().astype(np.float32)
        scalars = {
            f"gamma_c_{variant}_gap": np.float32(gap.cpu().item()),
            f"gamma_c_{variant}_advantage_action": np.float32(
                action_advantage.cpu().item()
            ),
            f"gamma_c_{variant}_residual_q_action": action_residual,
            f"gamma_c_{variant}_residual_q_best": residual_best,
            f"gamma_c_{variant}_residual_q_gap": residual_gap,
            f"gamma_c_{variant}_best_option": np.float32(
                best_option.cpu().item()
            ),
            f"gamma_c_{variant}_action_rank_score": np.float32(
                rank_score.cpu().item()
            ),
            f"gamma_c_{variant}_ensemble_effective_rank": np.float32(
                ensemble_telemetry["effective_rank"]
            ),
            f"gamma_c_{variant}_head_correlation": np.float32(
                np.nan
                if ensemble_telemetry["head_correlation_mean"] is None
                else ensemble_telemetry["head_correlation_mean"]
            ),
            f"gamma_c_{variant}_head_correlation_absolute": np.float32(
                np.nan
                if ensemble_telemetry["head_correlation_absolute_mean"] is None
                else ensemble_telemetry["head_correlation_absolute_mean"]
            ),
            f"gamma_c_{variant}_prior_contribution_ratio": np.float32(
                ensemble_telemetry["prior_contribution_ratio"]
            ),
            f"path_c_value_gap_{variant}": np.float32(gap.cpu().item()),
            f"path_c_action_rank_score_{variant}": np.float32(
                rank_score.cpu().item()
            ),
            f"path_c_residual_q_action_{variant}": action_residual,
        }
        if variant == PATH_C_PROBE_VARIANT:
            scalars["gamma_c_gap"] = scalars[f"gamma_c_{variant}_gap"]
            scalars["residual_q_gap"] = residual_gap
    return repr_vec, scalars

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _path_c_finalize_episode_value_advantages(
    rows: dict[str, list],
    row_start: int,
    row_stop: int,
    baseline_variants: tuple[str, ...],
) -> None:
    """Fill legacy proxy G-value columns with NaN placeholders.

    Earlier scaffold code synthesized value-control advantages from R^V event labels.
    That confounded C_resp^V with C_value. Path C now requires real held-out value
    artifacts (TD/residual-Q error, action-value ranking, value-gap calibration, or
    return gain) read back by diag_d1_train. These legacy columns remain only as
    diagnostics and are never allowed to contribute to go/no-go.
    """
    target_len = max(0, int(row_stop))
    for metric in PATH_C_VALUE_QUALITY_KINDS:
        key = f"{metric}_advantage"
        values = rows.setdefault(key, [])
        if len(values) < target_len:
            values.extend([np.float32(np.nan)] * (target_len - len(values)))
        for baseline in baseline_variants:
            bkey = f"{metric}_advantage_{baseline}"
            bvalues = rows.setdefault(bkey, [])
            if len(bvalues) < target_len:
                bvalues.extend([np.float32(np.nan)] * (target_len - len(bvalues)))


class _HistoryTracker:
    """Run-length encode the per-primitive-step inferred partner option kind stream,
    punctuated by partner delivery events (public, from the event extractor)."""

    def __init__(self, kind_to_id: dict[str, int]):
        self.kind_to_id = kind_to_id
        self.events: list[tuple[int, int, int]] = []  # (kind_id, duration, end_step)
        self._run_kind: int | None = None
        self._run_len = 0
        self._step = 0

    def _flush(self) -> None:
        if self._run_kind is not None and self._run_len > 0:
            self.events.append((self._run_kind, self._run_len, self._step))
        self._run_kind, self._run_len = None, 0

    def push(self, inferred_kind: str | None, deliv_ok: bool, deliv_wrong: bool) -> None:
        self._step += 1
        kid = self.kind_to_id.get(str(inferred_kind), self.kind_to_id[UNK])
        if kid != self._run_kind:
            self._flush()
            self._run_kind = kid
        self._run_len += 1
        if deliv_ok or deliv_wrong:
            self._flush()
            ev = EV_PARTNER_DELIV_OK if deliv_ok else EV_PARTNER_DELIV_WRONG
            self.events.append((self.kind_to_id[ev], 1, self._step))

    def snapshot(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        # Include the in-flight run so "partner has been doing X for d steps" is visible.
        evs = list(self.events)
        if self._run_kind is not None and self._run_len > 0:
            evs.append((self._run_kind, self._run_len, self._step))
        evs = evs[-window:]
        kind = np.zeros(window, dtype=np.int16)
        dur = np.zeros(window, dtype=np.float32)
        ago = np.zeros(window, dtype=np.float32)
        for i, (kid, d, end) in enumerate(evs):
            kind[i] = kid
            dur[i] = np.log1p(float(d))
            ago[i] = np.log1p(float(max(0, self._step - end)))
        return kind, dur, ago, len(evs)


def run(args: argparse.Namespace) -> None:
    ckpt = Path(args.anchor_checkpoint)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    _validate_path_c_collection_args(args, ckpt)
    ctx = E._load_context(ckpt, "d1_anchor")
    ctx.qaudit = None
    ctx.scripted_fsm = None
    rv_summary_spec = _rv_summary_spec(ctx.config)
    response_summary_enabled = bool(
        ((ctx.config.get("path_c") or {}).get("response_summary_spec") or {}).get(
            "enable", False
        )
    )
    primary_response_spec = None
    primary_evidence_spec = None
    if response_summary_enabled:
        primary_response_spec, rv_window_decisions = _primary_response_summary_spec(
            ctx.config
        )
        primary_evidence_spec = _primary_evidence_spec(ctx.config)
    else:
        rv_window_decisions = int(rv_summary_spec["window_decisions"])
    if args.ego == "fullchain":
        ctx.scripted_priority = _full_priority(ctx.option_lib)
    elif args.ego == "prepchain":
        # deferring competent ego (Link-A r2): reaction-family triggers
        # (escalate/tit-for-tat punish paths) only fire under ego deferrals
        terminal = ("serve_soup", "plate_soup", "pick_plate")
        ctx.scripted_priority = [
            k for k in _full_priority(ctx.option_lib) if k not in terminal]
    else:
        ctx.scripted_priority = None
    random_policy = args.ego == "random"

    policy = E._evidence_policy_for_config(ctx.config)
    if policy != "behavior_inferred_v1":
        raise RuntimeError(f"D1 requires inferred evidence mode, got {policy}")

    vocab = _kind_vocab(ctx.option_lib)
    kind_to_id = {k: i for i, k in enumerate(vocab)}
    graph = ctx.graph
    env = E._build_env(graph.layout_name, ctx.config)
    env.set_featurizer(E.NumpyFeaturizer(ctx.layout_graph))
    partners = {
        p.name: p
        for p in make_training_partners(ctx.option_lib, partner_set=args.partner_set)
    }
    if args.partner not in partners:
        raise KeyError(f"unknown partner {args.partner!r}; choices={sorted(partners)}")
    partner = partners[args.partner]
    split_assignment = None
    if response_summary_enabled:
        split_assignment = _frozen_split_assignment(
            ctx.config,
            partner,
            layout_name=str(graph.layout_name),
            split_group_id=getattr(args, "path_c_split_group_id", None),
            numeric_seed=int(args.seed),
        )
    checkpoint_sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    path_c_meta = path_c_metadata(ctx.config)
    resolved_path_c_sha256 = path_c_meta.get("resolved_path_c_sha256")
    if not resolved_path_c_sha256:
        resolved_path_c_sha256 = canonical_sha256(path_c_meta)
    collection_variant = str(args.path_c_collection_variant)
    belief_filter_artifact = getattr(args, "path_c_belief_filter_artifact", None)
    collect_belief_filter = bool(
        collection_variant == "diagnostic_anchor"
        or belief_filter_artifact not in {None, ""}
    )
    if collection_variant == "belief_filter" and not collect_belief_filter:
        raise ValueError(
            "Independent belief-filter collection requires --path-c-belief-filter-artifact."
        )
    run_id = make_run_id(
        checkpoint_sha256=checkpoint_sha256,
        resolved_path_c_sha256=str(resolved_path_c_sha256),
        collection_variant=collection_variant,
        partner=str(args.partner),
        ego=str(args.ego),
        base_seed=int(args.seed),
    )
    surface_identity_key = _surface_identity_key(partner)
    seed_group = (
        split_assignment["seed_group"]
        if split_assignment is not None
        else str(args.seed)
    )
    layout_style = str(graph.layout_name)
    matching_group_sha256 = canonical_sha256({
        "partner": str(args.partner),
        "ego": str(args.ego),
        "base_seed": int(args.seed),
        "episodes": int(args.episodes),
        "max_episode_options": int(args.max_episode_options),
        "layout_style": layout_style,
    })
    pot_positions = [e.pos for e in ctx.layout_graph.entities.values() if e.kind == "pot"]
    patience = int((ctx.config.get("options") or {}).get("block_patience", 0))
    path_c_value_contexts = _path_c_value_contexts(args, ctx)
    sequence_history_context = ctx if E._uses_sequence_q(ctx.q_net) else None
    if path_c_value_contexts is not None:
        sequence_contexts = [
            value_context
            for value_context in path_c_value_contexts.values()
            if E._uses_sequence_q(value_context.q_net)
        ]
        if sequence_contexts:
            sequence_history_context = sequence_contexts[0]
            evidence_hashes = {
                E._sequence_spec(value_context.q_net).sha256()
                for value_context in sequence_contexts
            }
            if len(evidence_hashes) != 1:
                raise ValueError(
                    "Path C sequence checkpoints do not share one evidence spec."
                )
            anchor_reward_hash = E.sha256_json(E.reward_config_payload(ctx.config))
            if any(
                E.sha256_json(E.reward_config_payload(value_context.config))
                != anchor_reward_hash
                for value_context in sequence_contexts
            ):
                raise ValueError(
                    "Path C value checkpoints changed the dataset reward contract."
                )
    sequence_history_config = (
        sequence_history_context.config
        if sequence_history_context is not None
        else ctx.config
    )
    if response_summary_enabled or path_c_value_contexts is not None:
        ctx.path_c_probe_cost_per_use = _frozen_probe_cost_from_config(ctx.config)
    if path_c_value_contexts is not None:
        active_probe_collection = str(args.path_c_collection_variant) in {
            PATH_C_PROBE_VARIANT,
            "random_probe",
        }
        ctx.active_probe_collection = active_probe_collection
        ctx.config.setdefault("path_c", {}).setdefault("probe", {})[
            "collection_selection_mode"
        ] = (
            "random"
            if str(args.path_c_collection_variant) == "random_probe"
            else "normalized_advantage"
        )
    path_c_q_variants = tuple(path_c_value_contexts or ())
    path_c_baseline_variants = tuple(
        variant for variant in path_c_q_variants if variant != PATH_C_PROBE_VARIANT
    )

    rng = np.random.default_rng(args.seed)
    rows: dict[str, list] = {k: [] for k in (
        "state_feat", "extra_feat", "hist_kind", "hist_dur", "hist_ago", "hist_len",
        "ego_opt_kind", "valid_kinds", "gate_main", "gate_strict",
        "episode_id", "episode_seed", "layout_style_id", "trajectory_source_id",
        "episode_uid", "run_id", "collection_variant", "option_transition",
        "collection_role", "split_group_id", "mechanism", "style_group",
        "surface_identity_key", "seed_group", "layout_style", "mechanism_key",
        "seed",
        "synthetic_registry_value_class_id", "style_id",
        "decision_reward", "held_out_episode_return",
        "public_context_stratum",
        "surface_action_frequency_bin", "dp_index", "dp_step",
        "mode_policy_id", "mode_family_id", "mode_param", "mode_age",
        "mode_opportunity_count", "mode_last_trigger_id", "mode_fingerprint_id",
        "mode_fingerprint_control_kind_id",
        "probe_action_id", "probe_selected", "probe_skip_reason", "probe_rule_id",
        "probe_candidate_scores", "probe_candidate_mask", "probe_propensity",
        "probe_cost_per_use", "probe_realized_cost",
        "probe_support_violation", "probe_invalid_script", "response_token_id",
        "resp_raw_ego_commit_count",
        "resp_raw_partner_commit_count", "resp_raw_ambig_count",
        "resp_raw_response_primitive_len",
        "resp_raw_partner_commitment_seq", "resp_raw_delivery_outcome_seq",
        *RV_COLUMNS,
    )}
    if primary_evidence_spec is not None:
        rows["ego_evidence_v1"] = []
    if path_c_value_contexts is not None:
        for variant in path_c_q_variants:
            rows[f"path_c_repr_{variant}"] = []
            for scalar in PATH_C_SIGNATURE_SCALARS:
                rows[f"gamma_c_{variant}_{scalar}"] = []
            rows[f"path_c_value_gap_{variant}"] = []
            rows[f"path_c_action_rank_score_{variant}"] = []
            rows[f"path_c_residual_q_action_{variant}"] = []
        rows["gamma_c_gap"] = []
        rows["residual_q_gap"] = []
        if collect_belief_filter:
            rows["path_c_repr_belief_filter"] = []
        for metric in PATH_C_VALUE_QUALITY_KINDS:
            rows[f"{metric}_advantage"] = []
            for baseline in path_c_baseline_variants:
                rows[f"{metric}_advantage_{baseline}"] = []
    audit: dict[str, list] = {k: [] for k in (
        "ep", "code", "inv0b", "inv0a", "inv1b", "inv1a", "n_ready", "n_cooking")}
    label_cols: dict[str, list] = {}
    for k in K_WINDOWS:
        label_cols[f"label_k{k}"] = []
        label_cols[f"censored_k{k}"] = []
        label_cols[f"ambig_k{k}"] = []
    source_counts: dict[str, int] = {}
    oracle_source_count = 0
    ambiguous_steps = 0
    golden_lines: list[str] = []

    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        evidence_buffer = E.EvidenceBuffer(
            num_factors=graph.num_factors,
            window=int(ctx.config["training"]["evidence_window"]),
            evidence_dim=E.D_EVID,
        )
        evidence_buffer.reset()
        path_c_filter = (
            _path_c_belief_filter(
                belief_filter_artifact,
                preregistration_sha256=(path_c_meta.get("preregistration") or {}).get("sha256"),
                resolved_path_c_sha256=str(resolved_path_c_sha256),
            )
            if collect_belief_filter
            else None
        )
        if path_c_filter is not None:
            path_c_filter.reset()
        router = E.OCV2EvidenceRouter(
            graph,
            ctx.layout_graph.cell_to_entity,
            ctx.layout_graph.region_cells,
            evidence_policy=policy,
        )
        router.reset()
        obs, state0 = env.reset(seed)
        partner.reset(seed)
        inferencer = E.make_behavior_option_inferencer(ctx.option_lib, ctx.config)
        inferencer.reset(state0)
        E._initialise_persistent_belief(
            evidence_buffer, ctx.method, ctx.belief_model, graph,
            E.torch.device("cpu"), E._belief_persistence_enabled(ctx.config),
        )
        E._start_sequence_episode(
            evidence_buffer,
            (
                sequence_history_context.q_net
                if sequence_history_context is not None
                else ctx.q_net
            ),
            obs,
            state0,
            ctx.option_lib,
            sequence_history_config,
            episode_id=f"path-c-dataset:{run_id}:{episode_idx}",
            manifest_seed=seed,
        )
        contribution_ledger = E.ContributionLedger.from_config(
            ctx.config.get("training")
        )
        tracker = _HistoryTracker(kind_to_id)
        selection_stats = {
            "option_selection_count": 0, "forced_noop_count": 0,
            "no_valid_option_count": 0,
        }
        step_deliv: list[int] = []
        step_delivery_outcome: list[int] = []
        step_partner_commitment: list[int] = []
        step_target_object: list[int] = []
        step_wait: list[int] = []
        step_help: list[int] = []
        step_block: list[int] = []
        dp_first_step: list[int] = []
        ep_row_start = len(rows["episode_id"])
        done = False
        option_count = 0
        primitive_steps = 0
        previous_ego_actions: list[int] = []
        previous_partner_actions: list[int] = []
        previous_option_id: int | None = None
        previous_reward = 0.0
        previous_progress = (
            np.zeros(primary_evidence_spec.progress_event_dim, dtype=np.float32)
            if primary_evidence_spec is not None
            else None
        )

        while not done and option_count < args.max_episode_options:
            # ---- decision-point record (state BEFORE selecting/executing) ----
            state = env.state
            n_ready, n_cooking, min_timer = _pot_signals(state, pot_positions)
            inv0, inv1 = int(get_inventory(state, 0)), int(get_inventory(state, 1))
            ego_soup, partner_soup = _carries_soup(inv0), _carries_soup(inv1)
            # D1-rev opportunity-onset gates (prereg §9): soup pending, nobody holds it
            clean_hands = not ego_soup and not partner_soup
            gate_main = bool((n_ready > 0 or n_cooking > 0) and clean_hands)
            gate_strict = bool(n_ready > 0 and clean_hands)
            valid = ctx.option_lib.valid_options(state, 0)
            valid_ids = np.flatnonzero(valid)
            valid_kind_hot = np.zeros(len(vocab), dtype=np.uint8)
            for vid in valid_ids:
                valid_kind_hot[kind_to_id.get(
                    str(ctx.option_lib.options[int(vid)].kind), kind_to_id[UNK])] = 1
            hk, hd, ha, hl = tracker.snapshot(args.hist_window)
            sync_public_state = getattr(partner, "sync_public_state", None)
            if callable(sync_public_state):
                sync_public_state(state)
            md = _mode_diag(partner)

            if primary_evidence_spec is not None:
                encoded_evidence = primary_evidence_spec.encode_decision(
                    observation=E._obs_vector(obs, "agent_0"),
                    ego_primitive_actions=previous_ego_actions,
                    partner_primitive_actions=previous_partner_actions,
                    ego_option_id=previous_option_id,
                    duration=len(previous_ego_actions),
                    reward=previous_reward,
                    progress_events=previous_progress,
                    valid_actions=valid,
                    terminated=False,
                    truncated=False,
                )
                sequence_episode = evidence_buffer.sequence_episode()
                if isinstance(sequence_episode, EpisodeEvidenceBuffer):
                    sequence_latest = sequence_episode.evidence_batch().evidence[0, -1]
                    if not E.torch.equal(encoded_evidence, sequence_latest):
                        raise RuntimeError(
                            "Dataset evidence row diverges from the acting sequence history."
                        )
                    encoded_evidence = sequence_latest
                rows["ego_evidence_v1"].append(
                    encoded_evidence.cpu().numpy().astype(np.float32)
                )

            option_id = E._select_option(
                ctx, obs, state, evidence_buffer, graph, rng, random_policy,
                partner_id=int(getattr(partner, "partner_id", 0)),
                selection_stats=selection_stats,
            )
            opt = ctx.option_lib.options[int(option_id)]
            if path_c_value_contexts is not None:
                for variant, variant_ctx in path_c_value_contexts.items():
                    repr_vec, scalar_values = _path_c_signature_row(
                        variant,
                        variant_ctx,
                        path_c_value_contexts.get("base_only"),
                        obs,
                        evidence_buffer,
                        valid,
                        int(option_id),
                        int(getattr(partner, "partner_id", 0)),
                    )
                    rows[f"path_c_repr_{variant}"].append(repr_vec)
                    for key, value in scalar_values.items():
                        rows[key].append(value)
                if path_c_filter is not None:
                    rows["path_c_repr_belief_filter"].append(
                        path_c_filter.belief.astype(np.float32).copy()
                    )
                for metric in PATH_C_VALUE_QUALITY_KINDS:
                    rows[f"{metric}_advantage"].append(np.float32(np.nan))
                    for baseline in path_c_baseline_variants:
                        rows[f"{metric}_advantage_{baseline}"].append(np.float32(np.nan))

            rows["state_feat"].append(np.asarray(obs["agent_0"], dtype=np.float32))
            rows["extra_feat"].append(np.asarray([
                float(n_ready), float(n_cooking), min_timer,
                float(ego_soup), float(partner_soup),
                float(has_plate(inv0)), float(has_plate(inv1)),
                float(option_count) / float(args.max_episode_options),
                np.log1p(float(primitive_steps)),
            ], dtype=np.float32))
            rows["hist_kind"].append(hk)
            rows["hist_dur"].append(hd)
            rows["hist_ago"].append(ha)
            rows["hist_len"].append(np.int16(hl))
            rows["ego_opt_kind"].append(
                np.int16(kind_to_id.get(str(opt.kind), kind_to_id[UNK])))
            rows["valid_kinds"].append(valid_kind_hot)
            rows["gate_main"].append(np.uint8(gate_main))
            rows["gate_strict"].append(np.uint8(gate_strict))
            rows["episode_id"].append(np.int32(episode_idx))
            rows["episode_seed"].append(np.int32(seed))
            rows["episode_uid"].append(make_episode_uid(run_id, args.seed, episode_idx))
            rows["run_id"].append(run_id)
            rows["collection_variant"].append(collection_variant)
            rows["collection_role"].append(
                split_assignment["collection_role"]
                if split_assignment is not None
                else "data_collection_only"
            )
            rows["split_group_id"].append(
                split_assignment["split_group_id"]
                if split_assignment is not None
                else "legacy_unassigned"
            )
            rows["mechanism"].append(
                split_assignment["mechanism"]
                if split_assignment is not None
                else "legacy_unassigned"
            )
            rows["style_group"].append(
                split_assignment["style_group"]
                if split_assignment is not None
                else "legacy_unassigned"
            )
            rows["decision_reward"].append(np.float64(np.nan))
            rows["held_out_episode_return"].append(np.float64(np.nan))
            rows["option_transition"].append(np.uint8(1))
            rows["surface_identity_key"].append(surface_identity_key)
            rows["seed_group"].append(seed_group)
            rows["seed"].append(np.uint64(
                split_assignment["numeric_seed"]
                if split_assignment is not None
                else int(args.seed)
            ))
            rows["layout_style"].append(layout_style)
            rows["mechanism_key"].append(
                f"{int(md['mode_family_id'])}:{int(md['mode_param'])}"
            )
            partner_spec = getattr(partner, "spec", None)
            rows["synthetic_registry_value_class_id"].append(str(
                getattr(partner_spec, "value_class_id", None)
                or f"{int(md['mode_family_id'])}:{int(md['mode_param'])}"
            ))
            rows["style_id"].append(str(
                getattr(partner_spec, "style_id", None) or "unspecified"
            ))
            rows["layout_style_id"].append(np.int16(_stable_small_id(graph.layout_name)))
            rows["trajectory_source_id"].append(np.int16(_stable_small_id(args.ego)))
            rows["surface_action_frequency_bin"].append(np.int16(_surface_action_frequency_bin(hk)))
            rows["dp_index"].append(np.int16(option_count))
            rows["dp_step"].append(np.int32(primitive_steps))
            rows["mode_policy_id"].append(np.int16(md["mode_policy_id"]))
            rows["mode_family_id"].append(np.int16(md["mode_family_id"]))
            rows["mode_param"].append(np.int16(md["mode_param"]))
            rows["mode_age"].append(np.int16(md["mode_age"]))
            rows["mode_opportunity_count"].append(np.int16(md["mode_opportunity_count"]))
            rows["mode_last_trigger_id"].append(np.int16(md["mode_last_trigger_id"]))
            rows["mode_fingerprint_id"].append(np.int16(md.get("mode_fingerprint_id", -1)))
            rows["mode_fingerprint_control_kind_id"].append(
                np.int16(md.get("mode_fingerprint_control_kind_id", -1))
            )
            probe_decision = dict(selection_stats.get("path_c_probe_last_decision") or {})
            probe_selected = bool(probe_decision.get("selected", False))
            probe_reason = str(probe_decision.get("reason", "none"))
            if probe_reason == "none" and random_policy:
                probe_reason = "exploration"
            elif probe_reason == "none" and ctx.scripted_priority is not None:
                probe_reason = "scripted"
            elif probe_reason == "none" and not bool(
                ((ctx.config.get("path_c") or {}).get("probe") or {}).get("enable", False)
            ):
                probe_reason = "disabled"
            probe_rule = str(probe_decision.get("rule", "none"))
            rows["probe_action_id"].append(
                np.int16(option_id if probe_selected else -1)
            )
            rows["probe_support_violation"].append(
                np.uint8(bool(probe_decision.get("support_violation", False)))
            )
            rows["probe_invalid_script"].append(
                np.uint8(bool(probe_decision.get("invalid_script", False)))
            )
            rows["probe_selected"].append(np.uint8(probe_selected))
            rows["probe_skip_reason"].append(np.int16(_probe_reason_id(probe_reason)))
            rows["probe_rule_id"].append(np.int16(_probe_rule_id(probe_rule)))
            raw_candidate_scores = probe_decision.get("candidate_scores")
            raw_candidate_mask = probe_decision.get("candidate_mask")
            candidate_scores = (
                np.full(graph.num_options, np.nan, dtype=np.float32)
                if raw_candidate_scores is None
                else np.asarray(raw_candidate_scores, dtype=np.float32)
            )
            candidate_mask = (
                np.asarray(valid, dtype=np.uint8)
                if raw_candidate_mask is None
                else np.asarray(raw_candidate_mask, dtype=np.uint8)
            )
            if candidate_scores.shape != (graph.num_options,):
                raise ValueError(
                    "Path C probe candidate scores must cover every registered option."
                )
            if candidate_mask.shape != (graph.num_options,):
                raise ValueError(
                    "Path C probe candidate mask must cover every registered option."
                )
            propensity = probe_decision.get("propensity")
            probe_cost_per_use = float(
                probe_decision.get(
                    "probe_cost_per_use",
                    ctx.path_c_probe_cost_per_use,
                )
            )
            probe_realized_cost = float(
                probe_decision.get("realized_probe_cost", 0.0)
            )
            if (
                not np.isfinite(probe_cost_per_use)
                or probe_cost_per_use < 0.0
                or not np.isfinite(probe_realized_cost)
                or probe_realized_cost < 0.0
            ):
                raise ValueError(
                    "Path C probe costs must be finite and non-negative."
                )
            expected_probe_cost = probe_cost_per_use if probe_selected else 0.0
            if probe_realized_cost != expected_probe_cost:
                raise ValueError(
                    "Path C realized probe cost must equal the frozen per-use cost "
                    "exactly when a probe is selected, and zero otherwise."
                )
            rows["probe_candidate_scores"].append(candidate_scores)
            rows["probe_candidate_mask"].append(candidate_mask)
            rows["probe_propensity"].append(
                np.float32(np.nan if propensity is None else float(propensity))
            )
            rows["probe_cost_per_use"].append(
                np.float32(probe_cost_per_use)
            )
            rows["probe_realized_cost"].append(
                np.float32(probe_realized_cost)
            )
            public_task_stage = "ready" if n_ready > 0 else ("cooking" if n_cooking > 0 else "prep")
            public_inventory_stage = (
                f"ego_soup={int(ego_soup)}:partner_soup={int(partner_soup)}:"
                f"ego_plate={int(has_plate(inv0))}:partner_plate={int(has_plate(inv1))}"
            )
            rows["public_context_stratum"].append(
                f"layout={layout_style}|task={public_task_stage}|inventory={public_inventory_stage}|"
                f"probe={int(option_id) if probe_selected else 'none'}"
            )
            dp_first_step.append(primitive_steps)

            # ---- execute the option (mirrors _execute_eval_option semantics) ----
            current_ego_actions: list[int] = []
            current_partner_actions: list[int] = []
            reward_scale = float(
                (ctx.config.get("training", {}).get("value_bound") or {}).get(
                    "reward_scale", 1.0
                )
            )
            if not np.isfinite(reward_scale) or reward_scale <= 0.0:
                raise ValueError(
                    "training.value_bound.reward_scale must be positive and finite."
                )
            current_reward = -float(probe_realized_cost) / reward_scale
            decision_evaluation_return = -float(probe_realized_cost)
            current_progress = (
                np.zeros(primary_evidence_spec.progress_event_dim, dtype=np.float32)
                if primary_evidence_spec is not None
                else None
            )
            sequence_episode = evidence_buffer.sequence_episode()
            sequence_decision = (
                DecisionEvidenceBuffer(sequence_episode.spec)
                if isinstance(sequence_episode, EpisodeEvidenceBuffer)
                else None
            )
            runtime = E.OptionRuntime(
                option_id=int(option_id), start_pos=E.get_agent_pos(state, 0))
            budget = ctx.option_lib.option_budget(state, 0, int(option_id))
            duration = 0
            termination_reason = "running"
            _spd = ctx.option_lib.layout_graph.shortest_path_dist
            _targets = tuple(ctx.option_lib._target_cells(opt))

            def _dist_to_target(_st) -> int | None:
                if not _targets:
                    return None
                _a = E.get_agent_pos(_st, 0)
                _ds = [d for d in (_spd.get((_a, _t)) for _t in _targets) if d is not None]
                return min(_ds) if _ds else None

            best_dist = _dist_to_target(state)
            stuck = 0
            while duration < budget:
                ostep = option_primitive_step(
                    env, ctx.option_lib, int(option_id), partner, obs, rng,
                    partner_option_inferencer=inferencer,
                )
                event = ostep.event
                current_ego_actions.append(int(ostep.ego_action))
                current_partner_actions.append(
                    int(ostep.partner_action.primitive_action)
                )
                contribution_ledger.update(event, ego_option_kind=str(opt.kind))
                ego_contributed = contribution_ledger.query_and_reset_on_delivery(
                    event
                )
                model_step_reward = E._training_reward(
                    ostep.step,
                    ctx.config,
                    "agent_0",
                    event,
                    ego_contributed=ego_contributed,
                    include_terminal_shaping=True,
                )
                evaluation_step_reward = E._training_reward(
                    ostep.step,
                    ctx.config,
                    "agent_0",
                    event,
                    ego_contributed=ego_contributed,
                    include_terminal_shaping=False,
                )
                current_reward += E._sequence_step_return(
                    model_step_reward,
                    ctx.config,
                )
                decision_evaluation_return += float(evaluation_step_reward) - float(
                    ctx.config["training"].get("cost_coef", 0.0)
                ) * float(ctx.config["training"].get("cost_per_step", 1.0))
                if primary_evidence_spec is not None:
                    current_progress += _progress_event_vector(
                        primary_evidence_spec, event
                    )
                if sequence_decision is not None:
                    evidence_step_return = E._sequence_step_return(
                        model_step_reward,
                        sequence_history_config,
                    )
                    if duration == 0 and probe_realized_cost > 0.0:
                        evidence_step_return -= float(probe_realized_cost) / reward_scale
                    sequence_decision.append_primitive(
                        int(ostep.ego_action),
                        int(ostep.partner_action.primitive_action),
                        reward=evidence_step_return,
                        progress_event=E._path_c_progress_event_vector(
                            event,
                            sequence_decision.spec,
                        ),
                    )
                duration += 1
                primitive_steps += 1
                src = str(event.partner_option_source)
                source_counts[src] = source_counts.get(src, 0) + 1
                if any(tok in src for tok in ORACLE_LIKE):
                    oracle_source_count += 1
                inferred_kind = None
                inferred_option = None
                if event.partner_option is not None:
                    inferred_option = ctx.option_lib.options[int(event.partner_option)]
                    inferred_kind = str(
                        inferred_option.kind)
                tracker.push(
                    inferred_kind,
                    bool(event.partner_correct_delivery),
                    bool(getattr(event, "partner_wrong_delivery_event", False)),
                )
                # D1-rev initiation coding (prereg §9): acquisition from RAW inventory
                # bits + strict delivery attribution; inferred kinds never touch labels.
                dcode = _step_delivery_code(event)
                inv0_b = int(get_inventory(ostep.prev_state, 0))
                inv1_b = int(get_inventory(ostep.prev_state, 1))
                inv0_a = int(get_inventory(ostep.step.state, 0))
                inv1_a = int(get_inventory(ostep.step.state, 1))
                ego_init = (_carries_soup(inv0_a) and not _carries_soup(inv0_b)) \
                    or dcode == D_EGO
                partner_init = (_carries_soup(inv1_a) and not _carries_soup(inv1_b)) \
                    or dcode == D_PARTNER
                if dcode == D_AMBIG or (ego_init and partner_init):
                    code = D_AMBIG
                    ambiguous_steps += 1
                elif ego_init:
                    code = D_EGO
                elif partner_init:
                    code = D_PARTNER
                else:
                    code = D_NONE
                step_deliv.append(code)
                step_delivery_outcome.append(_step_delivery_outcome_code(event, code))
                (
                    rv_commitment,
                    rv_target_object,
                    rv_wait,
                    rv_help,
                    rv_block,
                ) = _rv_step_value_event(inferred_option, event)
                step_partner_commitment.append(rv_commitment)
                step_target_object.append(rv_target_object)
                step_wait.append(rv_wait)
                step_help.append(rv_help)
                step_block.append(rv_block)
                if path_c_value_contexts is not None:
                    if path_c_filter is not None:
                        path_c_filter.update(_belief_filter_observation(rv_commitment))
                sr, sc, _ = _pot_signals(ostep.step.state, pot_positions)
                audit["ep"].append(episode_idx)
                audit["code"].append(code)
                audit["inv0b"].append(inv0_b)
                audit["inv0a"].append(inv0_a)
                audit["inv1b"].append(inv1_b)
                audit["inv1a"].append(inv1_a)
                audit["n_ready"].append(sr)
                audit["n_cooking"].append(sc)
                x_f = router.route(
                    event, ego_option_id=int(option_id),
                    ego_option_elapsed=duration, ego_option_max_steps=opt.max_steps,
                )
                evidence_buffer.append(x_f)
                E._advance_persistent_belief(
                    evidence_buffer, ctx.method, ctx.belief_model, graph,
                    E.torch.device("cpu"), x_f, E._belief_persistence_enabled(ctx.config),
                )
                done = bool(ostep.step.dones.get("__all__", False))
                if patience and best_dist is not None and not done:
                    cur = _dist_to_target(ostep.step.state)
                    if cur is not None and cur < best_dist:
                        best_dist, stuck = cur, 0
                    elif cur is not None and cur > 0:
                        stuck += 1
                    if stuck >= patience:
                        obs = ostep.step.obs
                        termination_reason = "blocked_no_progress"
                        break
                terminated, termination_reason = ctx.option_lib.option_terminated(
                    opt, ostep.prev_state, ostep.step.state, event,
                    agent_id=0, elapsed=duration, runtime=runtime,
                )
                if done and not terminated:
                    termination_reason = "env_max_steps"
                obs = ostep.step.obs
                if done or terminated:
                    break
            if not done and termination_reason == "running":
                termination_reason = "budget_exhausted"
            if termination_reason in {
                "budget_exhausted", "max_steps", "env_max_steps", "blocked_no_progress"
            } and duration > 0:
                x_fail = router.route_failure_boundary(
                    ego_option_id=int(option_id),
                    ego_option_elapsed=duration, ego_option_max_steps=opt.max_steps,
                )
                evidence_buffer.append(x_fail)
                E._advance_persistent_belief(
                    evidence_buffer, ctx.method, ctx.belief_model, graph,
                    E.torch.device("cpu"), x_fail,
                    E._belief_persistence_enabled(ctx.config),
                )
            if sequence_decision is not None:
                if not isinstance(sequence_episode, EpisodeEvidenceBuffer):
                    raise RuntimeError("Sequence decision has no episode buffer.")
                valid_options_next = np.asarray(
                    ctx.option_lib.valid_options(env.state, 0),
                    dtype=bool,
                )
                sequence_truncated = bool(
                    not done and option_count + 1 >= int(args.max_episode_options)
                )
                next_evidence = sequence_decision.encode_boundary(
                    observation=E._obs_vector(obs, "agent_0"),
                    ego_option_id=int(option_id),
                    valid_actions=valid_options_next,
                    terminated=bool(done),
                    truncated=sequence_truncated,
                )
                sequence_episode.append_transition(
                    action=int(option_id),
                    reward=float(current_reward),
                    discount=float(
                        sequence_history_config["training"]["gamma"]
                    ) ** int(duration),
                    done=bool(done),
                    truncated=sequence_truncated,
                    next_evidence=next_evidence,
                    next_valid_actions=valid_options_next,
                )
            previous_ego_actions = current_ego_actions
            previous_partner_actions = current_partner_actions
            previous_option_id = int(option_id)
            previous_reward = float(current_reward)
            rows["decision_reward"][ep_row_start + option_count] = np.float64(
                decision_evaluation_return
            )
            previous_progress = current_progress
            option_count += 1

        # ---- labels: first correct-delivery attribution within next K decisions ----
        n_dp = len(dp_first_step)
        if n_dp != len(rows["decision_reward"]) - ep_row_start:
            raise RuntimeError("Decision rewards are not aligned with episode rows.")
        _backfill_held_out_episode_return(
            rows,
            ep_row_start,
            ep_row_start + n_dp,
        )
        total_steps = len(step_deliv)
        deliv = np.asarray(step_deliv, dtype=np.int8)
        for k in K_WINDOWS:
            for i in range(n_dp):
                start = dp_first_step[i]
                end = dp_first_step[i + k] if i + k < n_dp else total_steps
                window = deliv[start:end]
                hits = np.flatnonzero(window)
                first = int(window[hits[0]]) if hits.size else D_NONE
                ambig = first == D_AMBIG
                label = 0 if ambig else first
                censored = bool(i + k >= n_dp and hits.size == 0)
                label_cols[f"label_k{k}"].append(np.int8(label))
                label_cols[f"censored_k{k}"].append(np.uint8(censored))
                label_cols[f"ambig_k{k}"].append(np.uint8(ambig))
        for i in range(n_dp):
            start = dp_first_step[i]
            end = (
                dp_first_step[i + rv_window_decisions]
                if i + rv_window_decisions < n_dp
                else total_steps
            )
            window = deliv[start:end]
            hits = np.flatnonzero(window)
            first = int(window[hits[0]]) if hits.size else D_NONE
            rows["resp_raw_ego_commit_count"].append(np.int16((window == D_EGO).sum()))
            rows["resp_raw_partner_commit_count"].append(np.int16((window == D_PARTNER).sum()))
            rows["resp_raw_ambig_count"].append(np.int16((window == D_AMBIG).sum()))
            rows["resp_raw_response_primitive_len"].append(np.int16(max(0, end - start)))
            commitment_seq, delivery_seq = _raw_value_event_sequences(
                i,
                n_dp,
                total_steps,
                dp_first_step,
                step_partner_commitment,
                step_delivery_outcome,
                window_decisions=rv_window_decisions,
            )
            rows["resp_raw_partner_commitment_seq"].append(commitment_seq)
            rows["resp_raw_delivery_outcome_seq"].append(delivery_seq)
            rv = _summarize_rv_value_event_window(
                step_partner_commitment[start:end],
                step_target_object[start:end],
                step_wait[start:end],
                step_help[start:end],
                step_block[start:end],
                step_delivery_outcome[start:end],
            )
            for col in RV_COLUMNS:
                rows[col].append(np.int16(rv[col]))
            if primary_response_spec is None:
                rows["response_token_id"].append(np.int16(-1))
            else:
                incomplete = bool(i + rv_window_decisions > n_dp)
                rows["response_token_id"].append(np.int16(
                    _canonical_response_token_id(
                        primary_response_spec,
                        rv,
                        terminal=bool(incomplete and done),
                        censored=bool(incomplete and not done),
                        invalid_script=bool(
                            rows["probe_invalid_script"][ep_row_start + i]
                        ),
                        support_violation=bool(
                            rows["probe_support_violation"][ep_row_start + i]
                        ),
                    )
                ))
        if path_c_value_contexts is not None:
            _path_c_finalize_episode_value_advantages(
                rows,
                ep_row_start,
                ep_row_start + n_dp,
                path_c_baseline_variants,
            )
        if args.golden and episode_idx == 0:
            for i in range(n_dp):
                s = dp_first_step[i]
                e = dp_first_step[i + 1] if i + 1 < n_dp else total_steps
                golden_lines.append(
                    f"dp{i:03d} step[{s}:{e}) gate_main={rows['gate_main'][ep_row_start + i]} "
                    f"gate_strict={rows['gate_strict'][ep_row_start + i]} "
                    f"mode_policy={int(rows['mode_policy_id'][ep_row_start + i])} "
                    f"mode_family={int(rows['mode_family_id'][ep_row_start + i])} "
                    f"mode_param={int(rows['mode_param'][ep_row_start + i])} "
                    f"mode_age={int(rows['mode_age'][ep_row_start + i])} "
                    f"mode_opp={int(rows['mode_opportunity_count'][ep_row_start + i])} "
                    f"mode_trigger={int(rows['mode_last_trigger_id'][ep_row_start + i])} "
                    f"ego_opt={vocab[int(rows['ego_opt_kind'][ep_row_start + i])]} "
                    f"deliv_in_opt={deliv[s:e].tolist()} "
                    f"label_k5={int(label_cols['label_k5'][ep_row_start + i])} "
                    f"ambig_k5={int(label_cols['ambig_k5'][ep_row_start + i])} "
                    f"censored_k5={int(label_cols['censored_k5'][ep_row_start + i])} "
                    f"hist_len={int(rows['hist_len'][ep_row_start + i])}"
                )

    n = len(rows["episode_id"])
    arrays = {k: np.stack(v) if k in ("state_feat", "extra_feat", "hist_kind",
                                      "hist_dur", "hist_ago", "valid_kinds",
                                      "probe_candidate_scores", "probe_candidate_mask",
                                      "resp_raw_partner_commitment_seq",
                                      "resp_raw_delivery_outcome_seq")
              else np.asarray(v) for k, v in rows.items()}
    if response_summary_enabled:
        preregistration_meta = (path_c_meta.get("preregistration") or {})
        semantic_bindings = preregistration_meta.get("semantic_bindings") or {}
        arrays["preregistration_sha256"] = np.asarray(
            [str(preregistration_meta.get("sha256"))] * n
        )
        arrays["resolved_path_c_sha256"] = np.asarray(
            [str(resolved_path_c_sha256)] * n
        )
        arrays["semantic_bindings_json"] = np.asarray(
            [json.dumps(semantic_bindings, sort_keys=True, separators=(",", ":"))] * n
        )
    for k, v in label_cols.items():
        arrays[k] = np.asarray(v)
    assert all(len(a) == n for a in arrays.values()), "ragged record arrays"
    for k, v in audit.items():
        arrays[f"audit_{k}"] = np.asarray(
            v, dtype=np.int32 if k == "ep" else np.int16)

    gm = arrays["gate_main"].astype(bool)
    lab5 = arrays["label_k5"]
    probe_reason_ids = arrays["probe_skip_reason"].astype(np.int64)
    candidate_reason_ids = {
        int(PATH_C_PROBE_REASON_TO_ID[name])
        for name in ("return_floor", "selected")
    }
    probe_candidate_count = int(
        np.count_nonzero(np.isin(probe_reason_ids, list(candidate_reason_ids)))
    )
    return_floor_count = int(
        np.count_nonzero(
            probe_reason_ids == int(PATH_C_PROBE_REASON_TO_ID["return_floor"])
        )
    )
    effective_episodes = int(np.unique(arrays["episode_uid"].astype(str)).size)
    effective_transitions = int(np.count_nonzero(arrays["option_transition"].astype(bool)))
    meta = {
        "diag": "D1",
        "partner": args.partner,
        "partner_set": args.partner_set,
        "ego": args.ego,
        "episodes": int(args.episodes),
        "base_seed": int(args.seed),
        "max_episode_options": int(args.max_episode_options),
        "hist_window": int(args.hist_window),
        "k_windows": list(K_WINDOWS),
        "kind_vocab": vocab,
        "anchor_checkpoint": str(ckpt),
        "anchor_sha256": checkpoint_sha256,
        "anchor_method": str(ctx.method),
        "evidence_policy": policy,
        "oracle_source_count": int(oracle_source_count),
        "partner_option_source_counts": source_counts,
        "ambiguous_delivery_steps": int(ambiguous_steps),
        "ambig_record_counts": {
            f"k{k}": int(arrays[f"ambig_k{k}"].sum()) for k in K_WINDOWS},
        "label_version": "rev1_initiation (prereg §9)",
        "n_records": int(n),
        "requested_episodes": int(args.episodes),
        "effective_episodes": effective_episodes,
        "effective_transitions": effective_transitions,
        "run_id": run_id,
        "collection_variant": collection_variant,
        "controller_checkpoint_key": collection_variant,
        "controller_checkpoint_sha256": checkpoint_sha256,
        "collection_role": (
            split_assignment["collection_role"]
            if split_assignment is not None
            else "data_collection_only"
        ),
        "split_manifest_sha256": (
            split_assignment["split_manifest_sha256"]
            if split_assignment is not None
            else None
        ),
        "split_group_id": (
            split_assignment["split_group_id"]
            if split_assignment is not None
            else None
        ),
        "mechanism": (
            split_assignment["mechanism"]
            if split_assignment is not None
            else None
        ),
        "style_group": (
            split_assignment["style_group"]
            if split_assignment is not None
            else None
        ),
        "split_group_ids": (
            [split_assignment["split_group_id"]]
            if split_assignment is not None
            else []
        ),
        "mechanisms": (
            [split_assignment["mechanism"]]
            if split_assignment is not None
            else []
        ),
        "style_groups": (
            [split_assignment["style_group"]]
            if split_assignment is not None
            else []
        ),
        "collection_mode": (
            "offline_anchor_diagnostic_only"
            if collection_variant == "diagnostic_anchor"
            else "on_policy_independent"
        ),
        "matching_group_sha256": matching_group_sha256,
        "preregistration_sha256": (
            (path_c_meta.get("preregistration") or {}).get("sha256")
        ),
        "resolved_path_c_sha256": str(resolved_path_c_sha256),
        "surface_identity_key": surface_identity_key,
        "seed_group": seed_group,
        "layout_style": layout_style,
        "n_gate_main": int(gm.sum()),
        "n_gate_strict": int(arrays["gate_strict"].sum()),
        "class_counts_k5_gate_main": {
            "no_initiation": int(((lab5 == 0) & gm).sum()),
            "ego_first": int(((lab5 == 1) & gm).sum()),
            "partner_first": int(((lab5 == 2) & gm).sum()),
        },
        "selection_stats_last_episode": selection_stats,
        "probe_selection_telemetry": {
            "floor_binding_definition": (
                "return-floor rejections divided by candidates that passed the "
                "disagreement threshold and reached return-floor adjudication"
            ),
            "return_floor_rejection_count": return_floor_count,
            "candidate_decision_count": probe_candidate_count,
            "floor_binding_rate": (
                float(return_floor_count / probe_candidate_count)
                if probe_candidate_count > 0
                else None
            ),
            "source_column": "probe_skip_reason",
        },
        "probe_cost_contract": {
            "probe_cost_per_use": float(ctx.path_c_probe_cost_per_use),
            "realized_cost_rule": (
                "selected probes pay the frozen per-use cost exactly once; "
                "all other decisions pay zero"
            ),
            "ego_evidence_reward_includes_realized_probe_cost": True,
        },
        "ecological_value_outcome": {
            "decision_reward_column": "decision_reward",
            "episode_outcome_column": "held_out_episode_return",
            "episode_outcome_rule": "sum_decision_reward_within_episode",
            "decision_reward_semantics": (
                "evaluation_return_without_terminal_shaping_minus_frozen_"
                "primitive_and_probe_cost"
            ),
            "probe_cost_included_exactly_once": True,
            "value_class_assignment": (
                "offline_out_of_fold_with_frozen_split_manifest"
            ),
            "synthetic_registry_label_column": (
                "synthetic_registry_value_class_id"
            ),
            "synthetic_registry_label_role": "secondary_oracle_diagnostic_only",
        },
        "opportunity_gate_def": {
            "main": "(any pot ready OR cooking) AND neither agent carries soup",
            "strict": "any pot ready AND neither agent carries soup",
        },
        "mode_policy_vocab": [ID_TO_POLICY[i] for i in sorted(ID_TO_POLICY)],
        "mode_family_vocab": [ID_TO_FAMILY[i] for i in sorted(ID_TO_FAMILY)],
        "mode_trigger_vocab": [ID_TO_TRIGGER[i] for i in sorted(ID_TO_TRIGGER)],
        "latent_partner_spec": _jsonable_dataclass(getattr(partner, "spec", None)),
        "path_c": path_c_metadata(ctx.config),
        "path_c_value_plumbing": {
            "integration_point": "diag_d1_dataset decision-point collection before option execution",
            "q_variants": list(path_c_q_variants),
            "q_variant_checkpoints": {
                variant: str(variant_ctx.checkpoint_path)
                for variant, variant_ctx in (path_c_value_contexts or {}).items()
            },
            "q_variant_methods": {
                variant: str(variant_ctx.method)
                for variant, variant_ctx in (path_c_value_contexts or {}).items()
            },
            "representation_variants": (
                list(path_c_q_variants)
                + (["belief_filter"] if collect_belief_filter else [])
                if path_c_value_contexts is not None else []
            ),
            "belief_filter_artifact": (
                str(Path(str(belief_filter_artifact)).resolve())
                if belief_filter_artifact not in {None, ""}
                else None
            ),
            "base_residualization_variant": (
                "base_only" if path_c_value_contexts is not None else None
            ),
            "single_td_loss_preserved": True,
            "frozen_outputs_no_grad": True,
        },
        "rv_summary_spec": rv_summary_spec,
        "rv_summary_spec_sha256": hashlib.sha256(
            json.dumps(rv_summary_spec, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "frozen_rv_summary_spec_sha256": (
            (path_c_meta.get("preregistration") or {}).get("response_vocabulary_sha256")
        ),
        "response_summary_spec": (
            primary_response_spec.canonical_payload()
            if primary_response_spec is not None
            else None
        ),
        "response_vocabulary_sha256": (
            primary_response_spec.sha256 if primary_response_spec is not None else None
        ),
        "response_vocabulary_q": (
            primary_response_spec.q if primary_response_spec is not None else None
        ),
        "structured_response_role": "secondary_only",
        "layout_name": str(graph.layout_name),
        "probe_family": "path_c_default_off" if not (ctx.config.get("path_c") or {}).get("active_sections") else "path_c",
        "fingerprint_vocab": sorted(
            {
                int(value)
                for value in rows["mode_fingerprint_id"]
                if int(value) >= 0
            }
        ),
    }
    if oracle_source_count != 0:
        raise RuntimeError(
            f"WIRING VIOLATION: oracle-like partner-option sources seen "
            f"({oracle_source_count}); D1 must ride behavior_inferred_v1 only.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if response_summary_enabled:
        shard_path, shard_sha256 = _write_content_addressed_parquet_shard(
            out, arrays
        )
        meta["storage"] = "content_addressed_append_only_shards"
        meta["format"] = "parquet"
        meta["shard_path"] = str(shard_path)
        meta["shard_sha256"] = shard_sha256
        meta_path = shard_path.with_suffix(".meta.json")
    else:
        # Historical diagnostic output only. Version-3 manifests reject this
        # format as primary or locked-audit evidence.
        np.savez_compressed(out, **arrays)
        meta["storage"] = "legacy_diagnostic_single_npz"
        meta["format"] = "npz"
        meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1))
    if args.golden:
        out.with_suffix(".golden.txt").write_text("\n".join(golden_lines))
    print(json.dumps({k: meta[k] for k in (
        "partner", "ego", "n_records", "n_gate_main", "n_gate_strict",
        "class_counts_k5_gate_main", "ambiguous_delivery_steps",
        "oracle_source_count")}, indent=1), flush=True)


def _write_content_addressed_parquet_shard(
    requested_path: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[Path, str]:
    if requested_path.suffix.lower() != ".parquet":
        raise ValueError(
            "Version-3 Path C collection requires --out ending in .parquet."
        )
    if requested_path.exists():
        raise FileExistsError(
            "Append-only Path C collection refuses to overwrite an existing shard."
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Version-3 Path C collection requires pyarrow for Parquet shards."
        ) from exc
    table = pa.table({
        name: pa.array(np.asarray(value).tolist())
        for name, value in arrays.items()
    })
    pq.write_table(table, requested_path, compression="zstd")
    digest = hashlib.sha256(requested_path.read_bytes()).hexdigest()
    content_path = requested_path.with_name(
        f"{requested_path.stem}.{digest[:16]}{requested_path.suffix}"
    )
    if content_path.exists():
        raise FileExistsError(f"Content-addressed shard already exists: {content_path}")
    requested_path.replace(content_path)
    return content_path, digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_checkpoint", required=True,
                    help="checkpoint providing env/option_lib/graph; also the argmax ego")
    ap.add_argument("--partner", required=True)
    ap.add_argument("--partner_set", default="role_conditioned_v2",
                    choices=(
                        "role_conditioned_v2",
                        "blind_v1",
                        "latent_v3_dev",
                        "path_c_synthetic",
                        "path_c_overlap_posterior",
                        "path_c_fingerprint_negative",
                    ))
    ap.add_argument("--ego", required=True,
                    choices=("argmax", "fullchain", "random", "prepchain"))
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-episode-options", type=int, default=40)
    ap.add_argument("--hist-window", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden", action="store_true",
                    help="dump a human-checkable per-decision timeline for episode 0")
    ap.add_argument(
        "--path-c-collection-variant",
        default="diagnostic_anchor",
        choices=("diagnostic_anchor", *PATH_C_REPR_VARIANTS),
        help="Controller variant that generated this on-policy chunk.",
    )
    ap.add_argument(
        "--path-c-split-group-id",
        default=None,
        help=(
            "Frozen split-manifest group ID. Required for version-3 Path C; "
            "the collection role and group metadata are derived from it."
        ),
    )
    ap.add_argument("--path-c-probing-checkpoint", default=None,
                    help="Optional aris_bellman probing ego checkpoint for frozen Path C readouts")
    ap.add_argument("--path-c-base-checkpoint", default=None,
                    help="Optional base_only checkpoint used as the public residual baseline")
    ap.add_argument("--path-c-global-gru-checkpoint", default=None,
                    help="Optional global_gru hard-baseline checkpoint for Path C readouts")
    ap.add_argument("--path-c-partner-id-checkpoint", default=None,
                    help="Optional partner_id_q hard-baseline checkpoint for Path C readouts")
    ap.add_argument("--path-c-rnn-residualized-checkpoint", default=None,
                    help="Optional RNN + same-residualization hard-baseline checkpoint for Path C readouts")
    ap.add_argument("--path-c-random-probe-checkpoint", default=None,
                    help="Optional random-probe hard-baseline/ablation checkpoint for Path C readouts")
    ap.add_argument("--path-c-no-probe-checkpoint", default=None,
                    help="Optional ordinary/no-probe hard-baseline checkpoint for Path C readouts")
    ap.add_argument("--path-c-no-admission-checkpoint", default=None,
                    help="Optional no-admission hard-baseline/ablation checkpoint for Path C readouts")
    ap.add_argument("--path-c-unpruned-ensemble-checkpoint", default=None,
                    help="Optional unpruned ensemble hard-baseline checkpoint for Path C readouts")
    ap.add_argument(
        "--path-c-belief-filter-artifact",
        default=None,
        help="Training-split-fitted HMM likelihood and matched value-head artifact.",
    )
    ap.add_argument("--path-c-require-value-plumbing", action="store_true",
                    help="Require the full Path C frozen-Q value/readout plumbing inputs")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
