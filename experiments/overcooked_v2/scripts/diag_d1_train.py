"""D1 supervised trainer and secondary response readout.

Offline models over diag_d1_dataset chunks; ZERO parameter sharing with the main method.

Stages (run in order; later stages refuse to peek earlier):
  gate    : merge chunks -> dataset_gate_report.json (sufficiency floors, class rates,
            ambiguity drop rates, cross-partner history-variance wiring check).
            Sign-B material. Never looks at any transfer metric.
            Floors are HARD on the two judged splits (indist_val, blind_terminal) and
            reported (soft) on co-report splits (indist_train, dev, blind_offaxis).
  tune    : train FULL / NOHIST (x K in {3,5,8}) + IDORACLE (K=5) x 3 seeds on training
            partners only (main-gate onset rows, ambiguous-window rows dropped). Reports
            in-distribution metrics ONLY; dev/blind rows are physically excluded from
            the tensors this stage builds.
  readout : SINGLE-LOOK. Requires dataset_gate_report.json PASS. Loads frozen models,
            computes all splits once, writes readout_frozen.json (refuses to overwrite).

The response-area-under-the-curve gains, leakage checks, and sensitivity analyses
below are secondary diagnostics only. They cannot replace or veto the version-3
normalized net-return versus probe-budget primary endpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments.overcooked_v2.path_c_evaluation import (
    FrozenPathCPreregistration,
    assign_group_disjoint_folds,
    build_path_c_readout_features,
    empirical_kernel_distance_audit,
    assemble_path_c_measurements,
    evaluate_path_c_decision,
    load_and_validate_path_c_inputs,
    load_frozen_preregistration,
    phase_b_go_no_go_rule,
)
from experiments.overcooked_v2.path_c_protocol import (
    simulate_cluster_operating_characteristics,
)
from experiments.overcooked_v2.path_c_value_classes import (
    ECOLOGICAL_VALUE_OUTCOME_NAME,
    CrossFittedEcologicalValueClassesV2,
    CrossFittedValueClassesV1,
    cluster_stratified_permutation,
    fit_cross_fitted_ecological_value_estimates,
)

JUDGED_K = 5
K_SET = (3, 5, 8)
N_SEEDS = 3
BOOT_ITERS = 10_000
FLOOR_GATED = 5000
FLOOR_MINCLASS = 500
RESPONSE_GAIN_FLOOR = 0.10
BLIND_RESPONSE_RETENTION = 0.5
LABEL_NAMES = ("no_initiation", "ego_first", "partner_first")  # D1-rev (prereg §9)
HARD_FLOOR_SPLITS = ("indist_val", "blind_terminal")
PATH_C_READOUT_FOLDS = 5
PATH_C_READOUT_HIDDEN = 64
PATH_C_READOUT_EPOCHS = 40
PATH_C_READOUT_BATCH = 2048
PATH_C_EQ_LABEL_TV = 0.05
PATH_C_EQ_RV_TV = 0.05
PATH_C_EQ_GAMMA_L1 = 0.05
PATH_C_LEAK_ADVANTAGE_EQ = 0.01
PATH_C_LEAK_BAL_ACC_MARGIN = 0.05
PATH_C_NULL_EQ_GAIN = 0.05
PATH_C_PERM_ITERS = 499
PATH_C_DEFAULT_PREREGISTRATION = (
    Path(__file__).resolve().parents[1] / "configs" / "path_c_preregistration.yaml"
)
PATH_C_DEFAULT_THRESHOLDS = {
    "response_advantage": 0.0,
    "value_advantage": 0.0,
    "transfer_advantage": 0.0,
    "conditional_leakage_max": PATH_C_LEAK_ADVANTAGE_EQ,
    "null_equivalence_gain": PATH_C_NULL_EQ_GAIN,
    "value_label_tv_equivalence_threshold": PATH_C_EQ_LABEL_TV,
    "rv_tv_equivalence_threshold": PATH_C_EQ_RV_TV,
    "gamma_c_l1_equivalence_threshold": PATH_C_EQ_GAMMA_L1,
    "fingerprint_visibility_l1_threshold": 0.01,
    "leakage_logloss_advantage_equivalence": PATH_C_LEAK_ADVANTAGE_EQ,
    "leakage_balanced_accuracy_margin": PATH_C_LEAK_BAL_ACC_MARGIN,
    "null_equivalence_gain": PATH_C_NULL_EQ_GAIN,
    "synthetic_power_min_mechanism_advantage": 0.0,
    "effective_episode_floor": 2000,
    "effective_transition_floor": 40000,
}
PATH_C_PRIMARY_VARIANT = "probing_ego"
PATH_C_COLLECTED_REPR_PREFIX = "path_c_repr_"
PATH_C_HARD_BASELINE_VARIANTS = (
    "global_gru",
    "base_only",
    "partner_id",
    "belief_filter",
    "rnn_residualized",
    "random_probe",
    "no_probe",
    "no_admission",
    "unpruned_ensemble",
)
PATH_C_REQUIRED_HARD_BASELINES = PATH_C_HARD_BASELINE_VARIANTS
PATH_C_STRONG_HISTORY_BASELINES = (
    "global_gru",
    "rnn_residualized",
    "belief_filter",
)
PATH_C_TRUE_VALUE_ADVANTAGE_STEMS = (
    "heldout_td_error_advantage",
    "td_error_advantage",
    "action_value_ranking_advantage",
    "value_gap_calibration_advantage",
    "residual_q_error_advantage",
    "cross_identity_return_gain",
)
# Accepted value-control columns are baseline-specific: <stem>_<baseline>. A
# single aggregate column is intentionally insufficient because the secondary
# readout requires advantage over every required baseline at the same split and budget.
PATH_C_TRUE_VALUE_ADVANTAGE_COLUMNS = tuple(
    f"{stem}_{baseline}"
    for stem in PATH_C_TRUE_VALUE_ADVANTAGE_STEMS
    for baseline in PATH_C_REQUIRED_HARD_BASELINES
)
PATH_C_REJECTED_PROXY_VALUE_COLUMNS = (
    "heldout_value_gap_advantage",
    "action_ranking_advantage",
    "residual_q_advantage",
)
PATH_C_FORBIDDEN_RV_COLUMNS = {
    "resp_rv_value_event_target_x",
    "resp_rv_value_event_target_y",
}
PATH_C_REQUIRED_LEAKAGE_CHANNELS = (
    "fingerprint",
    "identity",
    "seed",
    "layout_style",
    "surface_action_frequency",
    "trajectory_source",
)
PATH_C_VALUE_CLASS_ARTIFACT_DIR = "path_c_value_classes"
PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN = "ecological_value_class_id"
PATH_C_SYNTHETIC_VALUE_ORACLE_COLUMN = "synthetic_registry_value_class_id"
PATH_C_VALIDATED_VALUE_CLASS_OBJECTS = "_validated_ecological_value_class_artifacts"


def family_of(partner: str) -> str:
    if partner.startswith("latent-") or partner.startswith("blind-cert-"):
        return "latent"
    if "claim" in partner:
        return "claim"
    if "yield" in partner:
        return "yield"
    return "offaxis"


def split_of(partner: str) -> str:
    if partner.startswith("blind-"):
        return "blind"
    if partner.startswith("heldout-"):
        return "dev"
    return "train"


def _path_c_threshold(
    thresholds: dict[str, Any] | None,
    key: str,
    default: float | int | None = None,
) -> Any:
    if thresholds is not None and key in thresholds:
        return thresholds[key]
    if key in PATH_C_DEFAULT_THRESHOLDS:
        return PATH_C_DEFAULT_THRESHOLDS[key]
    return default


def _normalize_path_c_thresholds(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    flat: dict[str, Any] = {}

    def absorb(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        for key, value in mapping.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[str(key)] = value

    absorb(payload)
    for section in (
        payload.get("thresholds"),
        (payload.get("secondary_endpoints") or {}).get("thresholds"),
        payload.get("fingerprint_admission"),
        (payload.get("fingerprint_admission") or {}).get("thresholds"),
        payload.get("leakage"),
        (payload.get("leakage") or {}).get("thresholds"),
        payload.get("power_null"),
        (payload.get("power_null") or {}).get("thresholds"),
        payload.get("power_analysis"),
        payload.get("budget"),
    ):
        absorb(section)
    baseline_payload = payload.get("baselines") if isinstance(payload.get("baselines"), dict) else {}
    required_variants = (
        payload.get("required_baseline_variants")
        or payload.get("required_hard_baselines")
        or baseline_payload.get("required_variants")
        or baseline_payload.get("variants")
    )
    if isinstance(required_variants, (list, tuple)):
        flat["required_baseline_variants"] = tuple(str(v) for v in required_variants)
    return flat


# ---------------------------------------------------------------- data loading

def _load_chunk_columns(path: Path) -> dict[str, np.ndarray]:
    """Read one immutable dataset shard without changing its stored schema."""

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {str(key): np.asarray(archive[key]) for key in archive.files}
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"unsupported Path C shard format: {path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading version-3 Path C shards requires pyarrow."
        ) from exc
    table = pq.read_table(path)
    required_seed_types = {
        "seed": pa.uint64(),
        "episode_seed": pa.uint64(),
        "execution_seed": pa.uint32(),
    }
    result: dict[str, np.ndarray] = {}
    for name in table.column_names:
        field = table.schema.field(name)
        expected_type = required_seed_types.get(str(name))
        if expected_type is not None and field.type != expected_type:
            raise RuntimeError(
                f"Path C Parquet column {name!r} must use Arrow "
                f"{expected_type}; observed {field.type}."
            )
        if expected_type == pa.uint64():
            result[str(name)] = np.asarray(
                table[name].to_pylist(),
                dtype=np.uint64,
            )
        elif expected_type == pa.uint32():
            result[str(name)] = np.asarray(
                table[name].to_pylist(),
                dtype=np.uint32,
            )
        else:
            result[str(name)] = np.asarray(table[name].to_pylist())
    return result


def load_chunks(chunk_dir: str) -> dict[str, np.ndarray]:
    parquet_files = sorted(glob.glob(str(Path(chunk_dir) / "*.parquet")))
    legacy_files = sorted(glob.glob(str(Path(chunk_dir) / "*.npz")))
    if parquet_files and legacy_files:
        raise RuntimeError(
            "Path C refuses to mix version-3 Parquet shards with legacy NPZ diagnostics."
        )
    files = parquet_files or legacy_files
    if not files:
        raise FileNotFoundError(f"no chunks under {chunk_dir}")
    cols: dict[str, list] = {}
    partners, egos, partner_sets, vocab = [], [], [], None
    seed_keys, trajectory_sources, surface_profiles = [], [], []
    rv_summary_spec = None
    cross_identity_config = None
    fingerprint_vocab: set[int] = set()
    expected_columns: tuple[str, ...] | None = None
    seen_chunk_hashes: set[str] = set()
    seen_episode_uids: set[str] = set()
    observed_episode_uids: set[str] = set()
    effective_transitions = 0
    for f in files:
        chunk_sha256 = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        if chunk_sha256 in seen_chunk_hashes:
            raise RuntimeError(f"duplicate chunk SHA-256 under {chunk_dir}: {chunk_sha256}")
        seen_chunk_hashes.add(chunk_sha256)
        meta = json.loads(Path(f).with_suffix(".meta.json").read_text())
        chunk = _load_chunk_columns(Path(f))
        row_columns = tuple(sorted(
            str(key) for key in chunk if not str(key).startswith("audit_")
        ))
        columns = row_columns
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            missing = sorted(set(expected_columns).difference(columns))
            extra = sorted(set(columns).difference(expected_columns))
            raise RuntimeError(
                f"chunk schema mismatch in {f}; missing={missing}, extra={extra}"
            )
        column_lengths = {int(np.asarray(chunk[key]).shape[0]) for key in row_columns}
        if len(column_lengths) != 1:
            raise RuntimeError(f"chunk {f} has misaligned column lengths")
        forbidden_rv = sorted(PATH_C_FORBIDDEN_RV_COLUMNS.intersection(set(chunk)))
        if forbidden_rv:
            raise RuntimeError(
                f"chunk {f} contains forbidden R^V nuisance coordinate columns: {forbidden_rv}"
            )
        if Path(f).suffix.lower() == ".parquet" and "value_class_id" in chunk:
            raise RuntimeError(
                "Version-3 Path C shards reject ambiguous value_class_id; "
                "registry labels must use synthetic_registry_value_class_id and "
                "remain secondary oracle diagnostics."
            )
        n = int(chunk["episode_id"].shape[0])
        if column_lengths != {n}:
            raise RuntimeError(f"chunk {f} row count does not match episode_id")
        if int(meta["oracle_source_count"]) != 0:
            raise RuntimeError(f"chunk {f} has oracle-like sources — wiring violation")
        if vocab is None:
            vocab = meta["kind_vocab"]
        elif vocab != meta["kind_vocab"]:
            raise RuntimeError(f"kind vocab mismatch in {f}")
        spec = meta.get("rv_summary_spec")
        if rv_summary_spec is None:
            rv_summary_spec = spec
        elif spec != rv_summary_spec:
            raise RuntimeError(f"R^V summary spec mismatch in {f}")
        split_config = ((meta.get("path_c") or {}).get("split") or {})
        if cross_identity_config is None:
            cross_identity_config = split_config
        elif split_config != cross_identity_config:
            raise RuntimeError(f"cross-identity split config mismatch in {f}")
        fingerprint_vocab.update(int(v) for v in meta.get("fingerprint_vocab", []))
        if "episode_uid" in chunk:
            chunk_episode_uids = set(np.asarray(chunk["episode_uid"]).astype(str).tolist())
        else:
            chunk_episode_uids = {
                f"legacy:{meta.get('partner')}:{meta.get('ego')}:{meta.get('base_seed')}:{int(value)}"
                for value in np.unique(chunk["episode_id"]).tolist()
            }
        overlap = seen_episode_uids.intersection(chunk_episode_uids)
        if overlap:
            raise RuntimeError(f"duplicate episode UID across chunks: {sorted(overlap)[:3]}")
        seen_episode_uids.update(chunk_episode_uids)
        observed_episode_uids.update(chunk_episode_uids)
        chunk_transitions = (
            int(np.count_nonzero(np.asarray(chunk["option_transition"]).astype(bool)))
            if "option_transition" in chunk
            else n
        )
        if meta.get("effective_episodes") is not None and int(meta["effective_episodes"]) != len(chunk_episode_uids):
            raise RuntimeError(f"chunk {f} metadata effective_episodes disagrees with data")
        if meta.get("effective_transitions") is not None and int(meta["effective_transitions"]) != chunk_transitions:
            raise RuntimeError(f"chunk {f} metadata effective_transitions disagrees with data")
        effective_transitions += chunk_transitions
        for k in row_columns:
            values = np.asarray(chunk[k])
            if k in {"seed", "episode_seed"}:
                if values.dtype.kind not in {"u", "i"} or np.any(values < 0):
                    raise RuntimeError(
                        f"chunk {f} column {k!r} must contain unsigned integers"
                    )
                values = values.astype(np.uint64, copy=False)
            elif k == "execution_seed":
                if values.dtype.kind not in {"u", "i"} or np.any(values < 0):
                    raise RuntimeError(
                        f"chunk {f} column execution_seed must contain unsigned integers"
                    )
                if np.any(values.astype(np.uint64) > np.iinfo(np.uint32).max):
                    raise RuntimeError(
                        f"chunk {f} column execution_seed exceeds unsigned 32-bit range"
                    )
                values = values.astype(np.uint32, copy=False)
            cols.setdefault(k, []).append(values)
        partners.append(np.array([meta["partner"]] * n))
        egos.append(np.array([meta["ego"]] * n))
        partner_sets.append(np.array([meta.get("partner_set", "unknown")] * n))
        seed_keys.append(np.array([str(meta.get("base_seed", "unknown"))] * n))
        trajectory_sources.append(np.array([str(meta.get("ego", "unknown"))] * n))
        spec_meta = meta.get("latent_partner_spec") or {}
        surface_profile = ":".join(
            str(spec_meta.get(key, "none"))
            for key in ("geometry_profile", "base_protocol", "role", "pot_preference")
        )
        surface_profiles.append(np.array([surface_profile] * n))
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["partner"] = np.concatenate(partners)
    data["ego"] = np.concatenate(egos)
    data["partner_set"] = np.concatenate(partner_sets)
    data["seed_key"] = np.concatenate(seed_keys)
    data["trajectory_source_key"] = np.concatenate(trajectory_sources)
    data["surface_profile_key"] = np.concatenate(surface_profiles)
    base_seed_labels, _ = _encode_labels(data["seed_key"].astype(str))
    data["base_seed_id"] = base_seed_labels.astype(np.int16)
    try:
        base_seed_numeric = data["seed_key"].astype(np.int64)
        data["episode_seed"] = (base_seed_numeric + data["episode_id"].astype(np.int64)).astype(np.int64)
    except Exception:
        data["episode_seed"] = base_seed_labels.astype(np.int64)
    traj_labels, _ = _encode_labels(data["trajectory_source_key"].astype(str))
    data["trajectory_source_id"] = traj_labels.astype(np.int16)
    style_labels, _ = _encode_labels(data["surface_profile_key"].astype(str))
    data["layout_style_id"] = style_labels.astype(np.int16)
    if "surface_action_frequency_bin" not in data:
        freq_cols = [
            col for col in (
                "resp_raw_partner_commit_count",
                "resp_raw_ego_commit_count",
                "resp_rv_value_event_response_latency",
                "resp_rv_value_event_wait_count",
                "resp_rv_value_event_help_count",
                "resp_rv_value_event_block_count",
            ) if col in data
        ]
        if freq_cols:
            freq = np.zeros(data["episode_id"].shape[0], dtype=np.float64)
            for col in freq_cols:
                freq += np.asarray(data[col], dtype=np.float64)
            q1, q2 = np.quantile(freq, [1.0 / 3.0, 2.0 / 3.0])
            data["surface_action_frequency_bin"] = np.digitize(freq, [q1, q2]).astype(np.int16)
        else:
            data["surface_action_frequency_bin"] = np.zeros(data["episode_id"].shape[0], dtype=np.int16)
    data["_vocab"] = np.array(vocab)
    data["_rv_summary_spec"] = np.array(
        [json.dumps(rv_summary_spec or {}, sort_keys=True)]
    )
    data["_fingerprint_vocab"] = np.asarray(sorted(fingerprint_vocab), dtype=np.int16)
    data["_effective_episodes"] = np.asarray([len(observed_episode_uids)], dtype=np.int64)
    data["_effective_transitions"] = np.asarray([effective_transitions], dtype=np.int64)
    data["family"] = np.array([family_of(p) for p in data["partner"]])
    data["split"] = np.array([split_of(p) for p in data["partner"]])
    if "mechanism_key" in data:
        data["mechanism_key"] = data["mechanism_key"].astype(str)
    elif "mode_family_id" in data and "mode_param" in data:
        data["mechanism_key"] = np.array([
            f"{int(fam)}:{int(param)}"
            for fam, param in zip(data["mode_family_id"], data["mode_param"], strict=False)
        ])
    else:
        data["mechanism_key"] = data["family"].copy()
    data["identity_key"] = (
        data["surface_identity_key"].astype(str)
        if "surface_identity_key" in data
        else data["surface_profile_key"].astype(str)
    )
    if bool((cross_identity_config or {}).get("enable", False)):
        for key in ("surface_identity_key", "seed_group", "layout_style"):
            if key not in data:
                raise RuntimeError(f"cross-identity split requires column {key!r}")
        data["cross_identity_fold"] = assign_group_disjoint_folds(
            data["surface_identity_key"],
            data["seed_group"],
            data["layout_style"],
            n_folds=int((cross_identity_config or {}).get("folds", PATH_C_READOUT_FOLDS)),
        )
    # deterministic in-distribution episode split: 20% val by episode index
    data["indist_val"] = (data["episode_id"] % 5 == 4) & (data["split"] == "train")
    return data


def _validate_ecological_return_columns(
    data: dict[str, np.ndarray],
    row_mask: np.ndarray,
) -> None:
    required = {
        "episode_uid",
        "decision_reward",
        "held_out_episode_return",
        "collection_role",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            "Ecological value-class fitting is missing dataset column(s): "
            + ", ".join(missing)
        )
    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != np.asarray(data["episode_uid"]).shape:
        raise ValueError("Ecological value-class row mask is not aligned.")
    episodes = np.asarray(data["episode_uid"]).astype(str)
    decision_rewards = np.asarray(data["decision_reward"], dtype=np.float64)
    episode_returns = np.asarray(
        data["held_out_episode_return"], dtype=np.float64
    )
    if not np.isfinite(decision_rewards[mask]).all() or not np.isfinite(
        episode_returns[mask]
    ).all():
        raise ValueError("Ecological decision and episode returns must be finite.")
    for episode in sorted(np.unique(episodes[mask]).tolist()):
        episode_rows = mask & (episodes == episode)
        outcomes = np.unique(episode_returns[episode_rows])
        if outcomes.size != 1:
            raise ValueError(
                "Every episode must carry one held_out_episode_return."
            )
        recomputed = float(np.sum(
            decision_rewards[episode_rows],
            dtype=np.float64,
        ))
        if not np.isclose(
            recomputed,
            float(outcomes[0]),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "held_out_episode_return does not equal the sum of decision_reward."
            )


def build_ecological_value_class_artifacts(
    data: dict[str, np.ndarray],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, CrossFittedEcologicalValueClassesV2]:
    """Fit one frozen-manifest, out-of-fold ecological artifact per role."""

    if not isinstance(preregistration, FrozenPathCPreregistration):
        raise TypeError("A frozen Path C preregistration is required.")
    if "collection_role" not in data:
        raise ValueError("Ecological value classes require collection_role.")
    if "split_group_id" not in data:
        raise ValueError("Ecological value classes require split_group_id.")
    if "seed" not in data:
        raise ValueError("Ecological value classes require numeric seed.")
    if "public_context_stratum" not in data:
        raise ValueError("Ecological value classes require public_context_stratum.")
    roles = np.asarray(data["collection_role"]).astype(str)
    episodes = np.asarray(data.get("episode_uid", ())).astype(str)
    group_ids = np.asarray(data["split_group_id"]).astype(str)
    numeric_seeds = np.asarray(data["seed"])
    if not (roles.shape == episodes.shape == group_ids.shape == numeric_seeds.shape):
        raise ValueError(
            "collection_role, split_group_id, episode_uid, and seed are not row-aligned."
        )
    manifest_groups = {
        group.group_id: group for group in preregistration.split_manifest.groups
    }
    unknown_groups = sorted(set(group_ids.tolist()).difference(manifest_groups))
    if unknown_groups:
        raise ValueError(
            "Ecological value-class fitting found unknown split group(s): "
            + ", ".join(unknown_groups)
        )
    for group_id, role, numeric_seed in zip(
        group_ids.tolist(),
        roles.tolist(),
        numeric_seeds.tolist(),
        strict=True,
    ):
        if preregistration.split_manifest.role_for(group_id) != role:
            raise ValueError(
                "Ecological value-class role differs from the frozen split manifest."
            )
        preregistration.split_manifest.validate_numeric_seed(
            group_id,
            int(numeric_seed),
        )
    registered_roles = tuple(
        role for role in preregistration.split_manifest.assignment_by_group.values()
        if role in {"train", "design", "calibration", "locked_audit"}
    )
    unknown_roles = sorted(set(roles.tolist()).difference(set(registered_roles)))
    if unknown_roles:
        raise ValueError(
            "Ecological value-class fitting rejects unregistered role(s): "
            + ", ".join(unknown_roles)
        )
    observed_roles = tuple(sorted(
        set(roles.tolist()).intersection(set(registered_roles))
    ))
    if not observed_roles:
        raise ValueError("No frozen Path C role is available for ecological fitting.")
    episode_roles: dict[str, set[str]] = {}
    episode_groups: dict[str, set[str]] = {}
    episode_seeds: dict[str, set[int]] = {}
    for episode, role, group_id, numeric_seed in zip(
        episodes.tolist(),
        roles.tolist(),
        group_ids.tolist(),
        numeric_seeds.tolist(),
        strict=True,
    ):
        if role in observed_roles:
            episode_roles.setdefault(episode, set()).add(role)
            episode_groups.setdefault(episode, set()).add(group_id)
            episode_seeds.setdefault(episode, set()).add(int(numeric_seed))
    if any(len(values) != 1 for values in episode_roles.values()):
        raise ValueError("An episode cannot appear in more than one frozen role.")
    if any(len(values) != 1 for values in episode_groups.values()):
        raise ValueError("An episode cannot appear in more than one split group.")
    if any(len(values) != 1 for values in episode_seeds.values()):
        raise ValueError("An episode cannot use more than one numeric seed.")
    n_classes = int(
        preregistration.payload["synthetic_factorial"]["value_mechanisms"]
    )
    if n_classes < 2:
        raise ValueError("Frozen ecological value-class count must be at least two.")
    artifacts: dict[str, CrossFittedEcologicalValueClassesV2] = {}
    for role in observed_roles:
        role_mask = roles == role
        _validate_ecological_return_columns(data, role_mask)
        artifacts[role] = fit_cross_fitted_ecological_value_estimates(
            np.asarray(data["held_out_episode_return"])[role_mask],
            episodes[role_mask],
            np.asarray(data["public_context_stratum"])[role_mask],
            split_manifest=preregistration.split_manifest,
            role=role,
            n_classes=n_classes,
        )
    return artifacts


def write_ecological_value_class_artifacts(
    artifacts: Mapping[
        str,
        CrossFittedEcologicalValueClassesV2 | CrossFittedValueClassesV1,
    ],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Write immutable, content-addressed role artifacts for later analysis."""

    if not artifacts:
        raise ValueError("At least one ecological value-class artifact is required.")
    artifact_dir = Path(out_dir) / PATH_C_VALUE_CLASS_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    references: list[dict[str, Any]] = []
    for role, artifact in sorted(artifacts.items()):
        if not isinstance(
            artifact,
            (CrossFittedEcologicalValueClassesV2, CrossFittedValueClassesV1),
        ):
            raise TypeError(
                "Ecological value-class outputs must be "
                "registered cross-fitted ecological artifacts."
            )
        if role != artifact.role:
            raise ValueError("Value-class artifact mapping key changed its role.")
        payload = artifact.to_mapping()
        raw = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        file_sha256 = hashlib.sha256(raw).hexdigest()
        path = artifact_dir / f"{role}.{artifact.sha256}.json"
        if path.exists():
            raise FileExistsError(
                f"Ecological value-class artifact already exists: {path}"
            )
        path.write_bytes(raw)
        references.append({
            "role": role,
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "artifact_sha256": artifact.sha256,
            "split_manifest_sha256": artifact.split_manifest_sha256,
        })
    return references


def attach_validated_ecological_value_classes(
    data: dict[str, Any],
    references: Sequence[Mapping[str, Any]],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    """Validate role artifacts and attach their labels; no registry fallback exists."""

    if not isinstance(preregistration, FrozenPathCPreregistration):
        raise TypeError("A frozen Path C preregistration is required.")
    expected_reference_keys = {
        "role",
        "path",
        "file_sha256",
        "artifact_sha256",
        "split_manifest_sha256",
    }
    by_role: dict[str, Mapping[str, Any]] = {}
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != expected_reference_keys:
            raise ValueError("Ecological value-class reference has the wrong schema.")
        role = str(reference["role"])
        if role in by_role:
            raise ValueError("Ecological value-class references repeat a role.")
        by_role[role] = reference
    roles = np.asarray(data.get("collection_role", ())).astype(str)
    episodes = np.asarray(data.get("episode_uid", ())).astype(str)
    group_ids = np.asarray(data.get("split_group_id", ())).astype(str)
    outcomes = np.asarray(
        data.get("held_out_episode_return", ()), dtype=np.float64
    )
    if not (roles.shape == episodes.shape == group_ids.shape == outcomes.shape):
        raise ValueError("Ecological artifact inputs are not row-aligned.")
    manifest_group_ids = {
        group.group_id for group in preregistration.split_manifest.groups
    }
    if not set(group_ids.tolist()).issubset(manifest_group_ids):
        raise ValueError("Ecological attachment found an unknown split group.")
    if any(
        preregistration.split_manifest.role_for(group_id) != role
        for group_id, role in zip(
            group_ids.tolist(), roles.tolist(), strict=True
        )
    ):
        raise ValueError(
            "Ecological attachment role differs from the frozen split manifest."
        )
    episode_roles: dict[str, set[str]] = {}
    episode_groups: dict[str, set[str]] = {}
    for episode, role, group_id in zip(
        episodes.tolist(), roles.tolist(), group_ids.tolist(), strict=True
    ):
        episode_roles.setdefault(episode, set()).add(role)
        episode_groups.setdefault(episode, set()).add(group_id)
    if any(len(values) != 1 for values in episode_roles.values()):
        raise ValueError("An episode cannot appear in more than one frozen role.")
    if any(len(values) != 1 for values in episode_groups.values()):
        raise ValueError("An episode cannot appear in more than one split group.")
    ecological_ids = np.full(roles.shape, -1, dtype=np.int16)
    fold_ids = np.full(roles.shape, -1, dtype=np.int16)
    threshold_distances = np.full(roles.shape, np.nan, dtype=np.float64)
    value_estimates = np.full(roles.shape, np.nan, dtype=np.float64)
    value_standard_errors = np.full(roles.shape, np.nan, dtype=np.float64)
    artifact_hashes = np.full(roles.shape, "", dtype=object)
    validated_artifacts: dict[str, CrossFittedEcologicalValueClassesV2] = {}
    observed_roles = sorted(set(roles.tolist()).intersection({
        "train", "design", "calibration", "locked_audit"
    }))
    unknown_roles = sorted(set(roles.tolist()).difference(observed_roles))
    if unknown_roles:
        raise ValueError(
            "Ecological value-class attachment rejects unregistered role(s): "
            + ", ".join(unknown_roles)
        )
    if set(by_role) != set(observed_roles):
        raise ValueError(
            "Ecological value-class references must exactly cover observed roles."
        )
    for role in observed_roles:
        reference = by_role[role]
        path = Path(str(reference["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Ecological value-class artifact is missing: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(reference["file_sha256"]):
            raise ValueError("Ecological value-class artifact file SHA-256 mismatch.")
        payload = json.loads(raw.decode("utf-8"))
        artifact = CrossFittedEcologicalValueClassesV2.from_mapping(
            payload,
            split_manifest=preregistration.split_manifest,
        )
        expected_n_classes = int(
            preregistration.payload["synthetic_factorial"]["value_mechanisms"]
        )
        if (
            artifact.role != role
            or artifact.n_classes != expected_n_classes
            or artifact.sha256 != str(reference["artifact_sha256"])
            or artifact.split_manifest_sha256
            != str(reference["split_manifest_sha256"])
        ):
            raise ValueError("Ecological value-class reference changed artifact content.")
        role_mask = roles == role
        _validate_ecological_return_columns(data, role_mask)
        ecological_ids[role_mask] = artifact.labels_for_rows(
            episodes[role_mask],
            held_out_episode_return=outcomes[role_mask],
            require_complete=True,
        )
        assignment_by_episode = artifact.assignment_by_episode
        fold_ids[role_mask] = np.asarray(
            [assignment_by_episode[item].fold_id for item in episodes[role_mask]],
            dtype=np.int16,
        )
        threshold_distances[role_mask] = np.asarray(
            [
                assignment_by_episode[item].distance_to_nearest_threshold
                for item in episodes[role_mask]
            ],
            dtype=np.float64,
        )
        value_estimates[role_mask] = np.asarray(
            [assignment_by_episode[item].value_estimate for item in episodes[role_mask]],
            dtype=np.float64,
        )
        value_standard_errors[role_mask] = np.asarray(
            [
                assignment_by_episode[item].value_standard_error
                for item in episodes[role_mask]
            ],
            dtype=np.float64,
        )
        artifact_hashes[role_mask] = artifact.sha256
        validated_artifacts[role] = artifact
    if np.any(ecological_ids < 0) or np.any(fold_ids < 0):
        raise ValueError("Ecological value-class artifacts did not label every row.")
    data["ecological_value_class_id"] = ecological_ids
    data["ecological_value_class_fold"] = fold_ids
    data["ecological_value_class_threshold_distance"] = threshold_distances
    data["ecological_value_estimate"] = value_estimates
    data["ecological_value_standard_error"] = value_standard_errors
    data["ecological_value_class_artifact_sha256"] = artifact_hashes.astype(str)
    data[PATH_C_VALIDATED_VALUE_CLASS_OBJECTS] = validated_artifacts
    return data


def _validated_ecological_artifact_for_rows(
    data: dict[str, Any],
    idx: np.ndarray,
) -> tuple[
    str,
    CrossFittedEcologicalValueClassesV2 | CrossFittedValueClassesV1,
]:
    """Recheck attached ecological labels against the validated artifact object."""

    roles = np.unique(data["collection_role"][idx].astype(str))
    if roles.size != 1:
        raise ValueError("Ecological analysis cannot pool collection roles.")
    role = str(roles[0])
    artifacts = data.get(PATH_C_VALIDATED_VALUE_CLASS_OBJECTS)
    if not isinstance(artifacts, Mapping):
        raise ValueError(
            "Ecological analysis requires an artifact validated in this process."
        )
    artifact = artifacts.get(role)
    if not isinstance(
        artifact,
        (CrossFittedEcologicalValueClassesV2, CrossFittedValueClassesV1),
    ):
        raise ValueError("Ecological analysis lacks its validated role artifact.")
    episodes = data["episode_uid"][idx].astype(str)
    expected_classes = artifact.labels_for_rows(
        episodes,
        held_out_episode_return=data["held_out_episode_return"][idx],
        require_complete=False,
    )
    if not np.array_equal(
        expected_classes,
        data[PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN][idx].astype(np.int16),
    ):
        raise ValueError("Ecological value-class labels changed after validation.")
    assignments = artifact.assignment_by_episode
    expected_folds = np.asarray(
        [assignments[episode].fold_id for episode in episodes], dtype=np.int16
    )
    expected_distances = np.asarray(
        [
            assignments[episode].distance_to_nearest_threshold
            for episode in episodes
        ],
        dtype=np.float64,
    )
    if not np.array_equal(
        expected_folds,
        data["ecological_value_class_fold"][idx].astype(np.int16),
    ) or not np.allclose(
        expected_distances,
        data["ecological_value_class_threshold_distance"][idx],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Ecological fold or threshold distance changed after validation.")
    if set(data["ecological_value_class_artifact_sha256"][idx].astype(str)) != {
        artifact.sha256
    }:
        raise ValueError("Ecological artifact hash changed after validation.")
    return role, artifact


def ok_rows(data, k: int) -> np.ndarray:
    """Rows usable for window K: ambiguous-first-delivery windows are excluded."""
    return ~data[f"ambig_k{k}"].astype(bool)


def ep_keys(data, mask) -> np.ndarray:
    if "episode_uid" in data:
        return data["episode_uid"][mask].astype(str)
    return np.char.add(
        np.char.add(data["partner"][mask].astype(str),
                    np.char.add("|", data["ego"][mask].astype(str))),
        np.char.add("|", data["episode_id"][mask].astype(str)),
    )


def _path_c_fingerprint_checks(
    data,
    mask,
    labels,
    thresholds: dict[str, Any] | None = None,
) -> dict:
    """Admission test for value-null fingerprint candidates.

    A fingerprint candidate is admitted only when value labels, Γ_C audit columns,
    and the frozen R^V value-event columns are equivalent across fingerprint ids
    at fixed mechanism. Raw response visibility is reported separately; it is not
    allowed to rescue a candidate that moves value-relevant quantities.
    """
    if "mode_fingerprint_id" not in data or "mechanism_key" not in data:
        return {"available": False, "reason": "missing_fingerprint_or_mechanism_columns"}
    fp = data["mode_fingerprint_id"].astype(int)
    mechanism = data["mechanism_key"].astype(str)
    active = mask & (fp >= 0)
    if not bool(active.any()):
        return {"available": False, "reason": "no_fingerprint_rows_under_gate"}
    label_tv_threshold = float(_path_c_threshold(
        thresholds, "value_label_tv_equivalence_threshold", PATH_C_EQ_LABEL_TV))
    rv_tv_threshold = float(_path_c_threshold(
        thresholds, "rv_tv_equivalence_threshold", PATH_C_EQ_RV_TV))
    gamma_l1_threshold = float(_path_c_threshold(
        thresholds, "gamma_c_l1_equivalence_threshold", PATH_C_EQ_GAMMA_L1))
    raw_l1_threshold = float(_path_c_threshold(
        thresholds, "fingerprint_visibility_l1_threshold", 0.01))
    raw_cols = [
        c for c in (
            "resp_raw_ego_commit_count",
            "resp_raw_partner_commit_count",
            "resp_raw_ambig_count",
            "resp_raw_response_primitive_len",
        )
        if c in data
    ]
    gamma_cols = _path_c_value_audit_columns(data)
    rv_cols = _path_c_rv_columns(data)
    joint_rv_labels = None
    if rv_cols:
        joint_rv_text = np.asarray([
            "|".join(str(data[col][row]) for col in rv_cols)
            for row in range(data[rv_cols[0]].shape[0])
        ])
        joint_rv_labels, _ = _encode_labels(joint_rv_text)
    positive_control = bool(
        "mode_fingerprint_control_kind_id" in data
        and set(data["mode_fingerprint_control_kind_id"][active].astype(int).tolist()) == {1}
    )
    pair_reports = []
    max_label_tv = 0.0
    max_label_tv_bound = 0.0
    max_rv_tv = 0.0
    max_rv_tv_bound = 0.0
    max_raw_l1 = 0.0
    max_gamma_l1 = 0.0
    max_gamma_l1_bound = 0.0
    admitted_by_mechanism: dict[str, set[int]] = {}
    for mech in sorted(np.unique(mechanism[active]).tolist()):
        mech_mask = active & (mechanism == mech)
        ids = sorted(int(x) for x in np.unique(fp[mech_mask]).tolist())
        if len(ids) < 2:
            continue
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                lm = mech_mask & (fp == left)
                rm = mech_mask & (fp == right)
                if not bool(lm.any()) or not bool(rm.any()):
                    continue
                label_tv, label_bound = _categorical_tv_bound(
                    labels[lm].astype(int),
                    labels[rm].astype(int),
                    n_classes=max(3, int(np.max(labels[active])) + 1),
                )
                rv_tv, rv_bound = (
                    _categorical_tv_bound(joint_rv_labels[lm], joint_rv_labels[rm])
                    if joint_rv_labels is not None
                    else (0.0, 0.0)
                )
                raw_l1 = float(
                    sum(abs(float(data[col][lm].mean()) - float(data[col][rm].mean()))
                        for col in raw_cols)
                )
                gamma_l1, gamma_bound = _numeric_mean_l1_bound(data, gamma_cols, lm, rm)
                label_pass = bool(label_tv + 2.0 * label_bound <= label_tv_threshold)
                rv_pass = bool(rv_cols and rv_tv + 2.0 * rv_bound <= rv_tv_threshold)
                gamma_pass = bool(gamma_cols and gamma_l1 + 2.0 * gamma_bound <= gamma_l1_threshold)
                max_label_tv = max(max_label_tv, label_tv)
                max_label_tv_bound = max(max_label_tv_bound, label_bound)
                max_rv_tv = max(max_rv_tv, rv_tv)
                max_rv_tv_bound = max(max_rv_tv_bound, rv_bound)
                max_raw_l1 = max(max_raw_l1, raw_l1)
                max_gamma_l1 = max(max_gamma_l1, gamma_l1)
                max_gamma_l1_bound = max(max_gamma_l1_bound, gamma_bound)
                if label_pass and rv_pass and gamma_pass:
                    admitted_by_mechanism.setdefault(str(mech), set()).update({int(left), int(right)})
                pair_reports.append({
                    "mechanism_key": str(mech),
                    "fingerprint_pair": [int(left), int(right)],
                    "n_left": int(lm.sum()),
                    "n_right": int(rm.sum()),
                    "value_label_tv": label_tv,
                    "value_label_confidence_radius": label_bound,
                    "value_label_equivalence_pass": label_pass,
                    "joint_rv_tv": rv_tv if rv_cols else None,
                    "joint_rv_confidence_radius": rv_bound if rv_cols else None,
                    "joint_rv_equivalence_pass": rv_pass,
                    "raw_response_l1": raw_l1,
                    "gamma_c_l1": gamma_l1 if gamma_cols else None,
                    "gamma_c_confidence_radius": gamma_bound if gamma_cols else None,
                    "gamma_c_equivalence_pass": gamma_pass,
                    "admitted": bool(label_pass and rv_pass and gamma_pass),
                })
    available = bool(pair_reports)
    value_label_pass = bool(
        available and max_label_tv + 2.0 * max_label_tv_bound <= label_tv_threshold
    )
    rv_pass = bool(
        available and bool(rv_cols) and max_rv_tv + 2.0 * max_rv_tv_bound <= rv_tv_threshold
    )
    gamma_pass = bool(
        available
        and bool(gamma_cols)
        and max_gamma_l1 + 2.0 * max_gamma_l1_bound <= gamma_l1_threshold
    )
    visibility_pass = bool(available and max_raw_l1 >= raw_l1_threshold)
    admitted_sets = list(admitted_by_mechanism.values())
    counterbalance_pass = bool(
        len(admitted_sets) >= 2
        and min(len(values) for values in admitted_sets) >= 2
        and all(values == admitted_sets[0] for values in admitted_sets[1:])
    )
    admission_pass = bool(
        positive_control
        and visibility_pass
        and value_label_pass
        and rv_pass
        and gamma_pass
        and counterbalance_pass
    )
    return {
        "available": available,
        "value_label_tv_equivalence_threshold": label_tv_threshold,
        "rv_tv_equivalence_threshold": rv_tv_threshold,
        "gamma_c_l1_equivalence_threshold": gamma_l1_threshold,
        "fingerprint_visibility_l1_threshold": raw_l1_threshold,
        "gamma_c_columns": gamma_cols,
        "rv_columns": rv_cols,
        "raw_response_columns": raw_cols,
        "value_label_max_tv": max_label_tv,
        "value_label_max_confidence_radius": max_label_tv_bound,
        "joint_rv_max_tv": max_rv_tv if rv_cols else None,
        "rv_max_confidence_radius": max_rv_tv_bound if rv_cols else None,
        "raw_response_max_l1": max_raw_l1,
        "gamma_c_max_l1": max_gamma_l1 if gamma_cols else None,
        "gamma_c_max_confidence_radius": max_gamma_l1_bound if gamma_cols else None,
        "raw_visibility_admission_required": True,
        "positive_control_kind_pass": positive_control,
        "value_label_preservation_pass": value_label_pass,
        "rv_preservation_pass": rv_pass,
        "fingerprint_visibility_pass": visibility_pass,
        "gamma_c_preservation_pass": gamma_pass,
        "counterbalance_pass": counterbalance_pass,
        "admission_pass": admission_pass,
        "admitted_fingerprint_ids_by_mechanism": {
            key: sorted(values) for key, values in admitted_by_mechanism.items()
        },
        "pass": admission_pass,
        "pairs": pair_reports,
    }


def _path_c_rv_columns(data) -> list[str]:
    keys = {str(key) for key in data}
    forbidden_present = sorted(PATH_C_FORBIDDEN_RV_COLUMNS.intersection(keys))
    if forbidden_present:
        raise RuntimeError(
            "R^V contains forbidden identity/style/coordinate proxy column(s): "
            + ", ".join(forbidden_present)
        )
    cols = [
        str(key) for key in data
        if str(key).startswith("resp_rv_value_event_")
    ]
    return sorted(cols)


def _categorical_tv_bound(
    left: np.ndarray,
    right: np.ndarray,
    n_classes: int | None = None,
) -> tuple[float, float]:
    left = np.asarray(left).astype(int)
    right = np.asarray(right).astype(int)
    if left.size == 0 or right.size == 0:
        return float("inf"), float("inf")
    if n_classes is None:
        min_label = int(min(left.min(), right.min()))
        max_label = int(max(left.max(), right.max()))
        offset = -min_label if min_label < 0 else 0
        left = left + offset
        right = right + offset
        n_classes = max_label + offset + 1
    else:
        offset = 0
        if int(left.min()) < 0 or int(right.min()) < 0:
            min_label = int(min(left.min(), right.min()))
            offset = -min_label
            left = left + offset
            right = right + offset
            n_classes += offset
    lc = np.bincount(left, minlength=int(n_classes)).astype(float)
    rc = np.bincount(right, minlength=int(n_classes)).astype(float)
    ld = lc / max(float(lc.sum()), 1.0)
    rd = rc / max(float(rc.sum()), 1.0)
    tv = float(0.5 * np.abs(ld - rd).sum())
    radius = float(np.sqrt(0.5 * (1.0 / max(left.size, 1) + 1.0 / max(right.size, 1))))
    return tv, radius


def _numeric_mean_l1_bound(
    data,
    cols: list[str],
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[float, float]:
    if not cols:
        return 0.0, 0.0
    l1 = 0.0
    radius = 0.0
    for col in cols:
        left = np.asarray(data[col][left_mask], dtype=np.float64)
        right = np.asarray(data[col][right_mask], dtype=np.float64)
        if left.size == 0 or right.size == 0:
            return float("inf"), float("inf")
        l1 += abs(float(left.mean()) - float(right.mean()))
        lvar = float(left.var(ddof=1)) if left.size > 1 else 0.0
        rvar = float(right.var(ddof=1)) if right.size > 1 else 0.0
        radius += 1.96 * float(np.sqrt(lvar / max(left.size, 1) + rvar / max(right.size, 1)))
    return l1, radius


def _path_c_value_audit_columns(data) -> list[str]:
    explicit = {
        "residual_gap",
        "gamma_c_gap",
        "residual_q_gap",
        "resp_residual_gap",
    }
    prefixes = (
        "gamma_c_",
        "residual_value_",
        "residual_q_",
        "value_response_kernel_",
        "kv_",
    )
    cols: list[str] = []
    for key, value in data.items():
        if key not in explicit and not any(str(key).startswith(p) for p in prefixes):
            continue
        arr = np.asarray(value)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            cols.append(str(key))
    return sorted(set(cols))


# ------------------------------------------------------------------- metrics

def auc_binary(scores: np.ndarray, positives: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling (average ranks)."""
    pos, neg = scores[positives], scores[~positives]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    sorted_c = np.sort(combined, kind="mergesort")
    uniq, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    starts = np.searchsorted(sorted_c, uniq, side="left") + 1
    avg = starts + (counts - 1) / 2.0
    ranks = avg[inv]
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def metric_block(p_full, p_nohist, labels) -> dict:
    pos = labels == 2
    a_f = auc_binary(p_full[:, 2], pos)
    a_n = auc_binary(p_nohist[:, 2], pos)
    return {
        "n": int(labels.size),
        "class_counts": {LABEL_NAMES[i]: int((labels == i).sum()) for i in range(3)},
        "auc_ps_full": a_f,
        "auc_ps_nohist": a_n,
        "gain": a_f - a_n,
        "logloss_full": logloss(p_full, labels),
        "logloss_nohist": logloss(p_nohist, labels),
    }


def bootstrap_ci(stat_fn, keys: np.ndarray, iters=BOOT_ITERS, seed=0) -> dict:
    """Episode-block bootstrap, stratified by partner (first key field)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    strata: dict[str, list] = {}
    for k in uniq:
        strata.setdefault(str(k).split("|")[0], []).append(k)
    key_to_rows = {k: np.flatnonzero(keys == k) for k in uniq}
    vals = np.empty(iters, dtype=np.float64)
    for it in range(iters):
        idx_parts = []
        for _, ks in strata.items():
            pick = rng.integers(0, len(ks), size=len(ks))
            idx_parts.extend(key_to_rows[ks[j]] for j in pick)
        idx = np.concatenate(idx_parts)
        vals[it] = stat_fn(idx)
    return {"lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "mean": float(vals.mean())}


# -------------------------------------------------------------------- models

class D1Net(nn.Module):
    def __init__(self, variant: str, n_vocab: int, state_dim: int, n_partners: int):
        super().__init__()
        self.variant = variant
        hid = 64
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hid), nn.ReLU())
        self.opt_emb = nn.Embedding(n_vocab, 16)
        if variant == "full":
            self.kind_emb = nn.Embedding(n_vocab, 24, padding_idx=0)
            self.gru = nn.GRU(24 + 2, hid, batch_first=True)
        elif variant == "idoracle":
            self.partner_emb = nn.Embedding(max(n_partners, 1), hid)
        self.head = nn.Sequential(
            nn.Linear(hid + hid + 16, 128), nn.ReLU(), nn.Linear(128, 3))

    def hist_repr(self, hk, hd, ha, hl):
        if self.variant != "full":
            raise RuntimeError("hist_repr only defined for FULL")
        x = torch.cat([self.kind_emb(hk), hd.unsqueeze(-1), ha.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)
        idx = torch.clamp(hl.long() - 1, min=0)
        rep = out[torch.arange(out.shape[0]), idx]
        return torch.where((hl > 0).unsqueeze(-1), rep, torch.zeros_like(rep))

    def forward(self, state, hk, hd, ha, hl, ego_opt, partner_idx):
        return self.head(self.representation(state, hk, hd, ha, hl, ego_opt, partner_idx))

    def representation(self, state, hk, hd, ha, hl, ego_opt, partner_idx):
        if self.variant == "full":
            h = self.hist_repr(hk, hd, ha, hl)
        elif self.variant == "idoracle":
            h = self.partner_emb(torch.clamp(partner_idx.long(), min=0))
        else:  # nohist
            h = torch.zeros(state.shape[0], 64, device=state.device)
        return torch.cat([h, self.state_mlp(state), self.opt_emb(ego_opt.long())], dim=-1)


def build_state(data, idx, norm=None):
    x = np.concatenate([
        data["state_feat"][idx], data["extra_feat"][idx],
        data["valid_kinds"][idx].astype(np.float32),
    ], axis=1).astype(np.float32)
    if norm is not None:
        x = (x - norm[0]) / norm[1]
    return x


def tensors_for(data, idx, norm, partner_to_idx, device, k: int):
    pid = np.array([partner_to_idx.get(p, -1) for p in data["partner"][idx]])
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    return dict(
        state=t(build_state(data, idx, norm), torch.float32),
        hk=t(data["hist_kind"][idx], torch.long),
        hd=t(data["hist_dur"][idx], torch.float32),
        ha=t(data["hist_ago"][idx], torch.float32),
        hl=t(data["hist_len"][idx], torch.long),
        ego_opt=t(data["ego_opt_kind"][idx], torch.long),
        partner_idx=t(pid, torch.long),
        labels=t(data[f"label_k{k}"][idx], torch.long),
    )


def predict(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            logits = model(batch["state"][sl], batch["hk"][sl], batch["hd"][sl],
                           batch["ha"][sl], batch["hl"][sl], batch["ego_opt"][sl],
                           batch["partner_idx"][sl])
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outs)


def hidden_of(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            outs.append(model.hist_repr(batch["hk"][sl], batch["hd"][sl],
                                        batch["ha"][sl], batch["hl"][sl]).cpu().numpy())
    return np.concatenate(outs)


def representation_of(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            outs.append(model.representation(
                batch["state"][sl],
                batch["hk"][sl],
                batch["hd"][sl],
                batch["ha"][sl],
                batch["hl"][sl],
                batch["ego_opt"][sl],
                batch["partner_idx"][sl],
            ).cpu().numpy())
    return np.concatenate(outs)


class PathCReadoutNet(nn.Module):
    """Offline-only readout; gradients never touch the frozen ego representation."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = PATH_C_READOUT_HIDDEN):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )
        else:
            self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.net(x)


def _path_c_readout_capacity() -> dict[str, Any]:
    return {
        "model": "mlp",
        "hidden": PATH_C_READOUT_HIDDEN,
        "epochs": PATH_C_READOUT_EPOCHS,
        "batch": PATH_C_READOUT_BATCH,
        "folds": PATH_C_READOUT_FOLDS,
        "split_unit": "episode_uid",
        "rv_likelihood": "autoregressive_joint",
    }


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32), (mean, std)


def _stable_fold_ids(keys: np.ndarray, n_folds: int = PATH_C_READOUT_FOLDS) -> np.ndarray:
    keys = np.asarray(keys).astype(str)
    numeric = []
    for value in keys.tolist():
        try:
            numeric.append(int(value))
        except (TypeError, ValueError):
            numeric = []
            break
    if numeric and min(numeric) >= 0 and max(numeric) < int(n_folds):
        return np.asarray(numeric, dtype=np.int16)
    uniq, inv = np.unique(keys, return_inverse=True)
    if len(uniq) <= int(n_folds):
        fold_values = np.arange(len(uniq), dtype=np.int16)
    else:
        fold_values = np.asarray([
            int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % int(n_folds)
            for key in uniq
        ], dtype=np.int16)
    return fold_values[inv]


def _encode_labels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    text = np.asarray(values).astype(str)
    vocab, encoded = np.unique(text, return_inverse=True)
    return encoded.astype(np.int64), [str(v) for v in vocab.tolist()]


def _fit_predict_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    fold_keys: np.ndarray,
    *,
    device: str,
    seed: int,
    hidden: int = PATH_C_READOUT_HIDDEN,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.size)
    n_classes = int(labels.max()) + 1 if labels.size else 0
    if n == 0 or n_classes < 2:
        return {
            "available": False,
            "reason": "empty_or_single_class",
            "probs": np.zeros((n, max(n_classes, 1)), dtype=np.float32),
        }
    folds = _stable_fold_ids(fold_keys, PATH_C_READOUT_FOLDS)
    probs = np.zeros((n, n_classes), dtype=np.float32)
    covered = np.zeros(n, dtype=bool)
    used_folds = []
    dev = torch.device(device)
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        test = folds == fold
        if not bool(train.any()) or not bool(test.any()):
            continue
        if np.unique(labels[train]).size < 2:
            continue
        x_all, _ = _standardize(features[train], features)
        x_train = torch.as_tensor(x_all[train], dtype=torch.float32, device=dev)
        y_train = torch.as_tensor(labels[train], dtype=torch.long, device=dev)
        x_test = torch.as_tensor(x_all[test], dtype=torch.float32, device=dev)
        torch.manual_seed(int(seed) + int(fold))
        model = PathCReadoutNet(features.shape[1], n_classes, hidden=hidden).to(dev)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        order = torch.arange(x_train.shape[0], device=dev)
        for _epoch in range(PATH_C_READOUT_EPOCHS):
            model.train()
            perm = order[torch.randperm(order.shape[0], device=dev)]
            for start in range(0, perm.shape[0], PATH_C_READOUT_BATCH):
                sl = perm[start:start + PATH_C_READOUT_BATCH]
                logits = model(x_train[sl])
                loss = nn.functional.cross_entropy(logits, y_train[sl])
                optim.zero_grad()
                loss.backward()
                optim.step()
        model.eval()
        out_parts = []
        with torch.no_grad():
            for start in range(0, x_test.shape[0], PATH_C_READOUT_BATCH):
                logits = model(x_test[start:start + PATH_C_READOUT_BATCH])
                out_parts.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs[test] = np.concatenate(out_parts, axis=0)
        covered[test] = True
        used_folds.append(int(fold))
    if not used_folds or not bool(covered.all()):
        return {
            "available": False,
            "reason": "no_valid_crossfit_fold" if not used_folds else "incomplete_crossfit_coverage",
            "probs": probs,
        }
    return {
        "available": True,
        "probs": probs,
        "used_folds": used_folds,
        "n_classes": n_classes,
    }


def _row_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    return -np.log(p)


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if labels.size == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if bool(m.any()):
            ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def balanced_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    pred = probs.argmax(axis=1)
    recalls = []
    for cls in np.unique(labels):
        m = labels == cls
        recalls.append(float((pred[m] == cls).mean()) if bool(m.any()) else float("nan"))
    return float(np.nanmean(recalls)) if recalls else float("nan")


def multiclass_ovr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    aucs = []
    for cls in np.unique(labels):
        pos = labels == cls
        if bool(pos.any()) and bool((~pos).any()):
            aucs.append(auc_binary(probs[:, int(cls)], pos))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def _onehot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n_classes = int(labels.max()) + 1
    out = np.zeros((labels.size, n_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def _path_c_context_features(data, idx: np.ndarray) -> np.ndarray:
    pieces = [
        data["state_feat"][idx].astype(np.float32),
        data["extra_feat"][idx].astype(np.float32),
        data["valid_kinds"][idx].astype(np.float32),
    ]
    if "probe_action_id" in data:
        probe_labels, _ = _encode_labels(data["probe_action_id"][idx])
        pieces.append(_onehot(probe_labels))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _path_c_variant_features(
    data,
    idx: np.ndarray,
    models: dict[str, list],
    norm,
    partner_to_idx,
    device: str,
    k: int,
) -> dict[str, np.ndarray]:
    del models, norm, partner_to_idx, device, k
    # Path C measurements must read the dataset-collected Path C representations.
    # Falling back to legacy D1 full/no-history models would detach the GO rule from
    # the training/probe implementation being certified. Missing collected features
    # therefore make the measurement unavailable and NO-GO.
    return _path_c_collected_variant_features(data, idx)


def _path_c_collected_variant_features(data, idx: np.ndarray) -> dict[str, np.ndarray]:
    representations: dict[str, np.ndarray] = {}
    for key, value in data.items():
        name = str(key)
        if not name.startswith(PATH_C_COLLECTED_REPR_PREFIX):
            continue
        variant = name[len(PATH_C_COLLECTED_REPR_PREFIX):]
        arr = np.asarray(value)
        if arr.ndim != 2 or arr.shape[0] != data["episode_id"].shape[0]:
            continue
        representations[variant] = arr[idx].astype(np.float32)
    identity = data.get("surface_identity_key", data.get("identity_key", data["partner"]))
    features = build_path_c_readout_features(
        _path_c_context_features(data, idx),
        representations,
        surface_identity=np.asarray(identity)[idx],
    )
    ordered: dict[str, np.ndarray] = {}
    if PATH_C_PRIMARY_VARIANT in features:
        ordered[PATH_C_PRIMARY_VARIANT] = features[PATH_C_PRIMARY_VARIANT]
    for variant in PATH_C_HARD_BASELINE_VARIANTS:
        if variant in features:
            ordered[variant] = features[variant]
    for variant, arr in features.items():
        if variant not in ordered:
            ordered[variant] = arr
    return ordered


def _path_c_primary_variant(features: dict[str, np.ndarray]) -> str | None:
    # Path C GO/readout must be about the collected probing representation, never
    # a legacy D1 full-history fallback. Missing probe features make the Path C
    # measurement unavailable/failed instead of silently rebranding D1 as Path C.
    return PATH_C_PRIMARY_VARIANT if PATH_C_PRIMARY_VARIANT in features else None


def _path_c_required_baselines(thresholds: dict[str, Any] | None = None) -> tuple[str, ...]:
    configured = None if thresholds is None else thresholds.get("required_baseline_variants")
    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",") if item.strip()]
    if isinstance(configured, (list, tuple)) and configured:
        return tuple(str(v) for v in configured)
    return PATH_C_REQUIRED_HARD_BASELINES


def _path_c_baseline_completeness(
    features: dict[str, np.ndarray],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = _path_c_required_baselines(thresholds)
    missing = [variant for variant in required if variant not in features]
    return {
        "available": True,
        "role": "legacy_secondary_readout_only",
        "decision_eligible": False,
        "required_hard_baselines": list(required),
        "present_variants": sorted(features),
        "missing_required_hard_baselines": missing,
        "pass": not missing and PATH_C_PRIMARY_VARIANT in features,
    }


def _readout_score_for_columns(
    features: np.ndarray,
    data,
    idx: np.ndarray,
    columns: list[str],
    fold_keys: np.ndarray,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    nlls = []
    column_reports = {}
    eces = []
    autoregressive_features = np.asarray(features, dtype=np.float32)
    for offset, col in enumerate(columns):
        labels, vocab = _encode_labels(data[col][idx])
        pred = _fit_predict_classifier(
            autoregressive_features,
            labels,
            fold_keys,
            device=device,
            seed=seed + offset * 101,
        )
        if not pred.get("available"):
            column_reports[col] = {
                "available": False,
                "reason": pred.get("reason", "unknown"),
                "vocab": vocab,
            }
            return {
                "available": False,
                "reason": "incomplete_autoregressive_factorization",
                "failed_column": col,
                "columns": column_reports,
            }
        probs = pred["probs"]
        nll = _row_nll(probs, labels)
        nlls.append(nll)
        ece = multiclass_ece(probs, labels)
        eces.append(ece)
        column_reports[col] = {
            "available": True,
            "classes": vocab,
            "logloss": float(nll.mean()),
            "calibration_ece": ece,
            "conditioned_on_previous": list(columns[:offset]),
        }
        autoregressive_features = np.concatenate(
            [autoregressive_features, _onehot(labels)],
            axis=1,
        )
    if not nlls:
        return {"available": False, "reason": "no_rv_column_readout", "columns": column_reports}
    if len(nlls) != len(columns):
        return {
            "available": False,
            "reason": "incomplete_autoregressive_factorization",
            "columns": column_reports,
        }
    row_nll = np.sum(np.stack(nlls, axis=1), axis=1)
    return {
        "available": True,
        "row_nll": row_nll,
        "logloss": float(row_nll.mean()),
        "calibration_ece": float(np.nanmean(eces)) if eces else float("nan"),
        "likelihood": "autoregressive_joint",
        "factor_order": list(columns),
        "columns": column_reports,
    }


def _path_c_advantage_block(
    row_scores: dict[str, np.ndarray],
    keys: np.ndarray,
    *,
    primary: str | None = None,
    threshold: float = 0.0,
    required_baselines: tuple[str, ...] | None = None,
    power_cluster_keys: np.ndarray | None = None,
) -> dict[str, Any]:
    if primary is None:
        primary = PATH_C_PRIMARY_VARIANT
    if required_baselines is None:
        required_baselines = PATH_C_REQUIRED_HARD_BASELINES
    if primary not in row_scores:
        return {"available": False, "reason": f"missing_primary_variant:{primary}", "pass": False}
    missing = [variant for variant in required_baselines if variant not in row_scores]
    if missing:
        return {
            "available": False,
            "reason": "missing_required_hard_baselines",
            "missing_required_hard_baselines": missing,
            "required_hard_baselines": list(required_baselines),
            "present_variants": sorted(row_scores),
            "pass": False,
        }
    primary_scores = np.asarray(row_scores[primary], dtype=np.float64)
    bootstrap_keys = np.asarray(keys).astype(str)
    if primary_scores.ndim != 1 or bootstrap_keys.shape != primary_scores.shape:
        raise ValueError("Path C readout scores and episode keys must be row-aligned.")
    cluster_keys = None
    cluster_ids: list[str] = []
    if power_cluster_keys is not None:
        cluster_keys = np.asarray(power_cluster_keys).astype(str)
        if cluster_keys.shape != primary_scores.shape:
            raise ValueError(
                "Path C identity-group keys must be row-aligned with readout scores."
            )
        cluster_ids = sorted(np.unique(cluster_keys).tolist())
    advantages = {}
    passes = []
    for variant, baseline_nll in row_scores.items():
        if variant == primary:
            continue
        baseline_scores = np.asarray(baseline_nll, dtype=np.float64)
        if baseline_scores.shape != primary_scores.shape:
            raise ValueError(
                f"Path C baseline {variant!r} scores are not row-aligned."
            )
        delta = baseline_scores - primary_scores
        if not bool(np.all(np.isfinite(delta))):
            raise ValueError(f"Path C baseline {variant!r} effects must be finite.")
        ci = bootstrap_ci(
            lambda ii, d=delta: float(d[ii].mean()),
            bootstrap_keys,
        )
        passed = bool(float(delta.mean()) >= threshold and ci["lo"] > threshold)
        passes.append(passed)
        advantage = {
            "value": float(delta.mean()),
            "threshold": threshold,
            "ci": ci,
            "pass": passed,
        }
        if cluster_keys is not None:
            advantage.update({
                "power_cluster_unit": "identity_group",
                "power_cluster_ids": cluster_ids,
                "power_cluster_effects": [
                    float(delta[cluster_keys == cluster_id].mean())
                    for cluster_id in cluster_ids
                ],
            })
        advantages[variant] = advantage
    if not advantages:
        return {"available": False, "reason": "no_baseline_variants", "pass": False}
    min_adv = min(float(item["value"]) for item in advantages.values())
    min_ci_lo = min(float(item["ci"]["lo"]) for item in advantages.values())
    report = {
        "available": True,
        "value": min_adv,
        "min_ci_lo": min_ci_lo,
        "threshold": threshold,
        "pass": bool(all(passes)),
        "advantages_over_baseline": advantages,
        "required_hard_baselines": list(required_baselines),
    }
    if cluster_keys is not None:
        reference_baseline = min(
            required_baselines,
            key=lambda variant: (float(advantages[variant]["value"]), variant),
        )
        reference = advantages[reference_baseline]
        report.update({
            "power_cluster_unit": "identity_group",
            "power_reference_baseline": reference_baseline,
            "power_cluster_ids": list(reference["power_cluster_ids"]),
            "power_cluster_effects": list(reference["power_cluster_effects"]),
        })
    return report

def _path_c_rv_readout(
    data,
    mask: np.ndarray,
    models: dict[str, list],
    norm,
    partner_to_idx,
    device: str,
    *,
    split_name: str,
    fold_keys: np.ndarray | None = None,
    threshold: float = 0.0,
    required_baselines: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    rv_cols = _path_c_rv_columns(data)
    if not rv_cols:
        return {"available": False, "reason": "missing_value_event_rv_columns"}
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"available": False, "reason": "empty_mask", "split": split_name}
    keys = ep_keys(data, mask)
    folds = keys if fold_keys is None else np.asarray(fold_keys)[idx].astype(str)
    identity_source = next(
        (
            name
            for name in ("identity_key", "surface_identity_key", "partner")
            if name in data
        ),
        None,
    )
    if identity_source is None:
        return {
            "available": False,
            "reason": "missing_identity_group_for_power_analysis",
            "split": split_name,
        }
    power_cluster_keys = np.asarray(data[identity_source])[idx].astype(str)
    features = _path_c_variant_features(
        data, idx, models, norm, partner_to_idx, device, JUDGED_K)
    variant_scores = {}
    row_scores: dict[str, np.ndarray] = {}
    for variant, x in features.items():
        score = _readout_score_for_columns(
            x,
            data,
            idx,
            rv_cols,
            folds,
            device=device,
            seed=7000 + len(variant),
        )
        variant_scores[variant] = {
            key: value for key, value in score.items() if key != "row_nll"
        }
        if score.get("available"):
            row_scores[variant] = score["row_nll"]
    signal = _path_c_advantage_block(
        row_scores,
        keys,
        threshold=threshold,
        required_baselines=required_baselines,
        power_cluster_keys=power_cluster_keys,
    )
    return {
        "available": bool(signal.get("available")),
        "role": "legacy_secondary_readout_only",
        "decision_eligible": False,
        "split": split_name,
        "n": int(idx.size),
        "rv_columns": rv_cols,
        "readout_capacity": _path_c_readout_capacity(),
        "power_cluster_source_column": identity_source,
        "variant_scores": variant_scores,
        "signal": signal,
        "measurement": {
            "name": "held-out R^V readout advantage over every baseline",
            "value": signal.get("value"),
            "threshold": signal.get("threshold", threshold),
            "ci_lo": signal.get("min_ci_lo"),
            "pass": bool(signal.get("pass", False)),
            "decision_eligible": False,
        },
    }


def _conditional_linear_dependence(
    x: np.ndarray,
    labels: np.ndarray,
    cond: np.ndarray,
    episode_uid: np.ndarray,
    registered_stratum: np.ndarray,
    *,
    seed: int,
    iters: int = PATH_C_PERM_ITERS,
) -> dict[str, Any]:
    encoded_labels, vocab = _encode_labels(labels)
    if encoded_labels.size == 0 or len(vocab) < 2:
        return {"available": False, "reason": "empty_or_single_class", "classes": vocab}
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond, dtype=np.float64)
    if x.shape[0] != encoded_labels.size or cond.shape[0] != encoded_labels.size:
        raise ValueError("Conditional-dependence arrays must be row-aligned.")
    cond_aug = np.concatenate(
        [cond, np.ones((cond.shape[0], 1), dtype=np.float64)], axis=1
    )
    beta_x = np.linalg.pinv(cond_aug) @ x
    rx = x - cond_aug @ beta_x
    rx = rx - rx.mean(axis=0, keepdims=True)
    denom = max(float(rx.shape[0] - 1), 1.0)

    def statistic(target_labels: np.ndarray) -> float:
        target = _onehot(target_labels).astype(np.float64)
        beta_y = np.linalg.pinv(cond_aug) @ target
        residual_y = target - cond_aug @ beta_y
        residual_y = residual_y - residual_y.mean(axis=0, keepdims=True)
        return float(np.linalg.norm((rx.T @ residual_y) / denom, ord="fro"))

    stat = statistic(encoded_labels)
    perm_stats = np.empty(int(iters), dtype=np.float64)
    try:
        for iteration in range(int(iters)):
            permuted = cluster_stratified_permutation(
                encoded_labels,
                episode_uid,
                registered_stratum,
                seed=int(seed) + iteration,
            ).astype(np.int64)
            perm_stats[iteration] = statistic(permuted)
    except ValueError as exc:
        return {
            "available": False,
            "reason": "episode_cluster_permutation_infeasible",
            "detail": str(exc),
            "classes": vocab,
        }
    p_value = float(
        (1.0 + np.count_nonzero(perm_stats >= stat)) / (int(iters) + 1.0)
    )
    return {
        "available": True,
        "test": "linear_kernel_conditional_dependence_episode_cluster_permutation",
        "statistic": stat,
        "permutation_p": p_value,
        "permutation_iters": int(iters),
        "permutation_unit": "episode_uid",
        "permutation_scope": "within_registered_stratum",
        "classes": vocab,
    }

def _joint_categorical_codes(data, cols: list[str], idx: np.ndarray) -> np.ndarray:
    if not cols:
        return np.zeros(idx.size, dtype=np.int64)
    text = np.asarray(["|".join(str(data[col][row]) for col in cols) for row in idx], dtype=str)
    labels, _ = _encode_labels(text)
    return labels.astype(np.int64)


def _path_c_public_probe_context_codes(data, idx: np.ndarray) -> np.ndarray:
    """Deterministic public-context/probe bucket used for K^V conditioning.

    The audit intentionally hashes only public state features, valid-option masks,
    and logged probe provenance. It never includes identity/fingerprint/seed/style
    labels. Continuous public features are rounded to make exact empirical buckets
    stable across serialization.
    """
    rows: list[str] = []
    for row in idx:
        parts: list[str] = []
        for col in ("state_feat", "extra_feat", "valid_kinds"):
            if col not in data:
                continue
            arr = np.asarray(data[col][row])
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.round(arr.astype(np.float64), 4)
            parts.append(col + "=" + ",".join(map(str, np.ravel(arr).tolist())))
        for col in ("probe_action_id", "probe_selected", "probe_skip_reason", "probe_skip_reason_id"):
            if col in data:
                parts.append(f"{col}={data[col][row]}")
        rows.append("||".join(parts))
    labels, _ = _encode_labels(np.asarray(rows, dtype=str))
    return labels.astype(np.int64)


def _conditional_kernel_tv(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
    left_context: np.ndarray,
    right_context: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    """Maximum conditional TV across shared public/probe contexts.

    For same-W_C pairs this upper-bounds within-mechanism variation. For different
    W_C pairs it is an observable separation witness: at least one shared context
    / probe bucket must separate the value-response kernel.
    """
    left_context = np.asarray(left_context).astype(int)
    right_context = np.asarray(right_context).astype(int)
    common = sorted(set(left_context.tolist()).intersection(set(right_context.tolist())))
    if not common:
        return 1.0, 1.0, {
            "common_contexts": 0,
            "support_overlap": 0.0,
            "context_distances": [],
            "reason": "no_shared_public_probe_context",
        }
    distances: list[dict[str, Any]] = []
    best_tv = 0.0
    best_radius = 0.0
    overlap_mass_left = 0
    overlap_mass_right = 0
    for context in common:
        lm = left_context == context
        rm = right_context == context
        overlap_mass_left += int(lm.sum())
        overlap_mass_right += int(rm.sum())
        tv, radius = _tv_between_codes(left_codes[lm], right_codes[rm])
        best_tv = max(best_tv, tv)
        best_radius = max(best_radius, radius)
        distances.append({
            "context_code": int(context),
            "tv": float(tv),
            "confidence_radius": float(radius),
            "n_left": int(lm.sum()),
            "n_right": int(rm.sum()),
        })
    support_overlap = float(
        0.5 * (overlap_mass_left / max(left_context.size, 1)
               + overlap_mass_right / max(right_context.size, 1))
    )
    return best_tv, best_radius, {
        "common_contexts": int(len(common)),
        "support_overlap": support_overlap,
        "context_distances": distances,
    }


def _tv_between_codes(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    return _categorical_tv_bound(left.astype(int), right.astype(int))


def _path_c_public_probe_bucket(data, idx: np.ndarray) -> np.ndarray:
    """Coarse public-context/intervention bucket for observable K^V audit.

    The bucket intentionally uses only public decision-time fields and the logged
    probe intervention id. If exact public buckets are too sparse, C_fact becomes
    unavailable/NO-GO rather than collapsing across public fibers.
    """
    pieces: list[np.ndarray] = []
    for key in ("probe_action_id", "gate_main", "gate_strict", "dp_index"):
        if key in data:
            pieces.append(np.asarray(data[key][idx]).astype(str))
    if "valid_kinds" in data:
        vk = np.asarray(data["valid_kinds"][idx])
        pieces.append(np.asarray(["".join(row.astype(int).astype(str).tolist()) for row in vk], dtype=str))
    if "state_feat" in data:
        # Deterministic public-state fingerprint, rounded to avoid floating noise.
        sf = np.asarray(data["state_feat"][idx], dtype=np.float32)
        pieces.append(np.asarray([hashlib.sha1(np.round(row, 3).tobytes()).hexdigest()[:12] for row in sf], dtype=str))
    if not pieces:
        return np.asarray(["public_bucket:all"] * idx.size, dtype=str)
    text = pieces[0]
    for piece in pieces[1:]:
        text = np.char.add(np.char.add(text, "|"), piece)
    return text.astype(str)


def _path_c_kernel_distance_audit(
    data,
    mask: np.ndarray,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Secondary distributional-cell diagnostic over semantic strata.

    The version-3 instrument uses canonical outer-replica rows. This diagnostic
    cannot enter instrument validity or the primary decision.
    """
    source = None if thresholds is None else thresholds.get("_source_path")
    required = {
        "mechanism_key", "surface_identity_key", "public_context_stratum",
        "episode_uid",
    }
    missing = sorted(required.difference(data))
    if source in {None, ""}:
        return {"available": False, "reason": "missing_frozen_preregistration"}
    if missing:
        return {"available": False, "reason": "missing_kernel_columns", "missing": missing}
    rv_cols = _path_c_rv_columns(data)
    if not rv_cols:
        return {"available": False, "reason": "missing_rv_columns"}
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"available": False, "reason": "empty_mask"}
    preregistration = load_frozen_preregistration(source)
    joint_rv = np.stack(
        [np.asarray(data[column])[idx] for column in rv_cols],
        axis=1,
    )
    audit = empirical_kernel_distance_audit(
        joint_rv,
        np.asarray(data["mechanism_key"])[idx],
        np.asarray(data["surface_identity_key"])[idx],
        np.asarray(data["public_context_stratum"])[idx],
        np.asarray(data["episode_uid"])[idx],
        preregistration,
        seed=8302,
    )
    audit.update({
        "rv_columns": rv_cols,
        "theorem_eligible": False,
        "decision_eligible": False,
        "role": "secondary_distributional_diagnostic",
    })
    return audit


def _synthetic_registry_value_oracle_diagnostic(
    data: dict[str, np.ndarray],
    idx: np.ndarray,
) -> dict[str, Any]:
    """Expose registry labels only as a clearly non-ecological oracle diagnostic."""

    if PATH_C_SYNTHETIC_VALUE_ORACLE_COLUMN not in data:
        return {
            "available": False,
            "role": "secondary_oracle_diagnostic_only",
            "decision_eligible": False,
        }
    labels = np.asarray(data[PATH_C_SYNTHETIC_VALUE_ORACLE_COLUMN])[idx].astype(str)
    return {
        "available": bool(labels.size),
        "role": "secondary_oracle_diagnostic_only",
        "source": PATH_C_SYNTHETIC_VALUE_ORACLE_COLUMN,
        "classes": sorted(np.unique(labels).tolist()),
        "decision_eligible": False,
        "may_define_ecological_value_class": False,
    }


def _path_c_conditional_factorization_audit(
    data,
    mask: np.ndarray,
    models: dict[str, list],
    norm,
    partner_to_idx,
    device: str,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Secondary value/identity readout using validated ecological classes."""

    idx = np.flatnonzero(mask)
    required = {
        PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN,
        "ecological_value_class_fold",
        "ecological_value_class_artifact_sha256",
        "ecological_value_class_threshold_distance",
        "collection_role",
        "held_out_episode_return",
        PATH_C_VALIDATED_VALUE_CLASS_OBJECTS,
        "identity_key",
        "episode_uid",
    }
    missing = sorted(required.difference(data))
    if missing:
        return {
            "available": False,
            "reason": "missing_value_or_identity_fields",
            "missing": missing,
            "synthetic_registry_value_oracle": (
                _synthetic_registry_value_oracle_diagnostic(data, idx)
            ),
        }
    if idx.size == 0:
        return {"available": False, "reason": "empty_mask"}
    analysis_roles = np.unique(data["collection_role"][idx].astype(str))
    if analysis_roles.size != 1:
        return {
            "available": False,
            "reason": "cross_role_ecological_value_class_pooling_forbidden",
            "roles": analysis_roles.tolist(),
        }
    analysis_role, _value_class_artifact = _validated_ecological_artifact_for_rows(
        data,
        idx,
    )
    features_by_variant = _path_c_variant_features(
        data, idx, models, norm, partner_to_idx, device, JUDGED_K
    )
    primary = _path_c_primary_variant(features_by_variant)
    if primary is None:
        return {"available": False, "reason": "missing_primary_variant"}
    features = features_by_variant[primary]
    context = _path_c_context_features(data, idx)
    ecological_values = data[PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN][idx]
    value_labels, value_vocab = _encode_labels(ecological_values)
    identity_labels, identity_vocab = _encode_labels(data["identity_key"][idx])
    value_onehot = _onehot(value_labels)
    keys = ep_keys(data, mask)

    value_context = _fit_predict_classifier(
        context, value_labels, keys, device=device, seed=8101
    )
    value_joint = _fit_predict_classifier(
        np.concatenate([context, features], axis=1),
        value_labels,
        keys,
        device=device,
        seed=8102,
    )
    identity_condition = np.concatenate([context, value_onehot], axis=1)
    identity_context = _fit_predict_classifier(
        identity_condition, identity_labels, keys, device=device, seed=8201
    )
    identity_joint = _fit_predict_classifier(
        np.concatenate([identity_condition, features], axis=1),
        identity_labels,
        keys,
        device=device,
        seed=8202,
    )
    value_advantage = None
    identity_advantage = None
    if value_context.get("available") and value_joint.get("available"):
        value_advantage = float(
            logloss(value_context["probs"], value_labels)
            - logloss(value_joint["probs"], value_labels)
        )
    if identity_context.get("available") and identity_joint.get("available"):
        identity_advantage = float(
            logloss(identity_context["probs"], identity_labels)
            - logloss(identity_joint["probs"], identity_labels)
        )
    public_probe = _path_c_public_probe_context_codes(data, idx).astype(str)
    registered_stratum = np.char.add(
        ecological_values.astype(str),
        np.char.add("|", public_probe),
    )
    dependence = _conditional_linear_dependence(
        features,
        data["identity_key"][idx],
        identity_condition,
        data["episode_uid"][idx],
        registered_stratum,
        seed=8301,
    )
    leakage_limit = float(_path_c_threshold(
        thresholds,
        "conditional_leakage_max",
        PATH_C_LEAK_ADVANTAGE_EQ,
    ))
    mechanism_proxy = None
    if "mechanism_key" in data:
        mechanism_proxy = {
            "role": "secondary_proxy_only",
            "classes": sorted(
                np.unique(data["mechanism_key"][idx].astype(str)).tolist()
            ),
        }
    return {
        "available": True,
        "primary_variant": primary,
        "collection_role": analysis_role,
        "value_class_source": "validated_cross_fitted_ecological_value_class_artifact",
        "value_class_artifact_sha256": sorted(np.unique(
            data["ecological_value_class_artifact_sha256"][idx].astype(str)
        ).tolist()),
        "cross_fit_folds": sorted(np.unique(
            data["ecological_value_class_fold"][idx].astype(int)
        ).tolist()),
        "value_class_uncertainty": {
            "measure": "distance_to_nearest_out_of_fold_threshold",
            "minimum": float(np.min(
                data["ecological_value_class_threshold_distance"][idx]
            )),
            "median": float(np.median(
                data["ecological_value_class_threshold_distance"][idx]
            )),
        },
        "value_classes": value_vocab,
        "identity_classes": identity_vocab,
        "value_readout_conditioned_on_public_context": {
            "logloss_context": (
                logloss(value_context["probs"], value_labels)
                if value_context.get("available") else None
            ),
            "logloss_context_plus_representation": (
                logloss(value_joint["probs"], value_labels)
                if value_joint.get("available") else None
            ),
            "advantage": value_advantage,
        },
        "identity_given_value_readout": {
            "logloss_context_value": (
                logloss(identity_context["probs"], identity_labels)
                if identity_context.get("available") else None
            ),
            "logloss_context_value_plus_representation": (
                logloss(identity_joint["probs"], identity_labels)
                if identity_joint.get("available") else None
            ),
            "advantage": identity_advantage,
            "equivalence_threshold": leakage_limit,
        },
        "conditional_independence_test": dependence,
        "mechanism_label": mechanism_proxy,
        "synthetic_registry_value_oracle": (
            _synthetic_registry_value_oracle_diagnostic(data, idx)
        ),
        "decision_eligible": False,
    }

def _nearest_neighbor_leakage(
    features: np.ndarray,
    labels: np.ndarray,
    cond: np.ndarray,
    *,
    max_n: int = 2048,
) -> dict[str, Any]:
    labels, vocab = _encode_labels(labels)
    if labels.size == 0 or len(vocab) < 2:
        return {"available": False, "reason": "empty_or_single_class", "classes": vocab}
    x = np.asarray(features, dtype=np.float64)
    cond = np.asarray(cond, dtype=np.float64)
    if x.shape[0] > max_n:
        keep = np.linspace(0, x.shape[0] - 1, max_n).round().astype(int)
        x = x[keep]
        cond = cond[keep]
        labels = labels[keep]
    cond_aug = np.concatenate([cond, np.ones((cond.shape[0], 1), dtype=np.float64)], axis=1)
    beta = np.linalg.pinv(cond_aug) @ x
    rx = x - cond_aug @ beta
    sq = np.sum(rx * rx, axis=1, keepdims=True)
    dist = sq + sq.T - 2.0 * (rx @ rx.T)
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(axis=1)
    acc = float((labels[nn] == labels).mean())
    chance = float(np.max(np.bincount(labels)) / labels.size)
    return {
        "available": True,
        "accuracy": acc,
        "majority_chance": chance,
        "n": int(labels.size),
        "classes": vocab,
    }


def _path_c_leakage_battery(
    data,
    mask: np.ndarray,
    models: dict[str, list],
    norm,
    partner_to_idx,
    device: str,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"available": False, "reason": "empty_mask", "pass": False}
    required_value_fields = {
        PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN,
        "ecological_value_class_fold",
        "ecological_value_class_artifact_sha256",
        "ecological_value_class_threshold_distance",
        "collection_role",
        "held_out_episode_return",
        PATH_C_VALIDATED_VALUE_CLASS_OBJECTS,
        "episode_uid",
    }
    if not required_value_fields.issubset(data):
        return {
            "available": False,
            "reason": "missing_validated_ecological_value_class_artifact",
            "synthetic_registry_value_oracle": (
                _synthetic_registry_value_oracle_diagnostic(data, idx)
            ),
            "pass": False,
        }
    analysis_roles = np.unique(data["collection_role"][idx].astype(str))
    if analysis_roles.size != 1:
        return {
            "available": False,
            "reason": "cross_role_ecological_value_class_pooling_forbidden",
            "roles": analysis_roles.tolist(),
            "pass": False,
        }
    analysis_role, _value_class_artifact = _validated_ecological_artifact_for_rows(
        data,
        idx,
    )
    features_by_variant = _path_c_variant_features(
        data, idx, models, norm, partner_to_idx, device, JUDGED_K)
    primary = _path_c_primary_variant(features_by_variant)
    if primary is None:
        return {"available": False, "reason": "missing_primary_variant", "pass": False}
    features = features_by_variant[primary]
    context = _path_c_context_features(data, idx)
    ecological_values = data[PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN][idx]
    value_labels, _ = _encode_labels(ecological_values)
    cond = np.concatenate([context, _onehot(value_labels)], axis=1)
    registered_stratum = np.char.add(
        ecological_values.astype(str),
        np.char.add(
            "|",
            _path_c_public_probe_context_codes(data, idx).astype(str),
        ),
    )
    keys = ep_keys(data, mask)
    leak_eq = float(_path_c_threshold(
        thresholds,
        "leakage_logloss_advantage_equivalence",
        PATH_C_LEAK_ADVANTAGE_EQ,
    ))
    bal_margin = float(_path_c_threshold(
        thresholds,
        "leakage_balanced_accuracy_margin",
        PATH_C_LEAK_BAL_ACC_MARGIN,
    ))
    nuisance_sources: dict[str, np.ndarray | None] = {
        "fingerprint": data["mode_fingerprint_id"].astype(int) if "mode_fingerprint_id" in data else None,
        "identity": data["identity_key"].astype(str) if "identity_key" in data else None,
        "seed": data["episode_seed"].astype(int) if "episode_seed" in data else None,
        "layout_style": data["layout_style_id"].astype(int) if "layout_style_id" in data else None,
        "surface_action_frequency": data["surface_action_frequency_bin"].astype(int) if "surface_action_frequency_bin" in data else None,
        "trajectory_source": data["trajectory_source_id"].astype(int) if "trajectory_source_id" in data else None,
    }
    channel_reports = {}
    missing = []
    pass_items = []
    primary_advantages = []
    for offset, (name, values) in enumerate(nuisance_sources.items()):
        if values is None:
            missing.append(name)
            channel_reports[name] = {"available": False, "reason": "missing_required_label"}
            pass_items.append(False)
            continue
        active_idx = idx
        if name == "fingerprint":
            active_idx = idx[values[idx].astype(int) >= 0]
        if active_idx.size == 0:
            channel_reports[name] = {"available": False, "reason": "no_rows"}
            pass_items.append(False)
            continue
        pos = np.asarray([np.flatnonzero(idx == row)[0] for row in active_idx], dtype=np.int64)
        labels, vocab = _encode_labels(values[active_idx])
        if len(vocab) < 2:
            channel_reports[name] = {"available": False, "reason": "single_class", "classes": vocab}
            pass_items.append(False)
            continue
        chan_cond = cond[pos]
        chan_features = features[pos]
        chan_keys = ep_keys(data, np.isin(np.arange(data["episode_id"].shape[0]), active_idx))
        base = _fit_predict_classifier(chan_cond, labels, chan_keys, device=device, seed=9101 + offset * 17)
        joint = _fit_predict_classifier(
            np.concatenate([chan_cond, chan_features], axis=1),
            labels,
            chan_keys,
            device=device,
            seed=9102 + offset * 17,
        )
        if not base.get("available") or not joint.get("available"):
            channel_reports[name] = {
                "available": False,
                "reason": "classifier_unavailable",
                "base": {k: v for k, v in base.items() if k != "probs"},
                "joint": {k: v for k, v in joint.items() if k != "probs"},
            }
            pass_items.append(False)
            continue
        base_ll = logloss(base["probs"], labels)
        joint_ll = logloss(joint["probs"], labels)
        advantage = float(base_ll - joint_ll)
        bal = balanced_accuracy(joint["probs"], labels)
        chance = float(1.0 / max(len(vocab), 1))
        dep = _conditional_linear_dependence(
            chan_features,
            values[active_idx],
            chan_cond,
            data["episode_uid"][active_idx],
            registered_stratum[pos],
            seed=9103 + offset * 17,
        )
        nn = _nearest_neighbor_leakage(chan_features, values[active_idx], chan_cond)
        ovr_auc = multiclass_ovr_auc(joint["probs"], labels)
        auc_pass = bool(np.isnan(ovr_auc) or ovr_auc <= 0.55)
        nn_pass = bool(
            not nn.get("available")
            or nn["accuracy"] <= nn["majority_chance"] + bal_margin
        )
        channel_pass = bool(
            advantage <= leak_eq
            and bal <= chance + bal_margin
            and auc_pass
            and nn_pass
            and (not dep.get("available") or dep["permutation_p"] >= 0.05)
        )
        primary_advantages.append(advantage)
        pass_items.append(channel_pass)
        channel_reports[name] = {
            "available": True,
            "classes": vocab,
            "conditioned_on": [
                PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN,
                "public_context",
                "probe_action_id",
            ],
            "logloss_context": base_ll,
            "logloss_context_plus_representation": joint_ll,
            "representation_advantage": advantage,
            "equivalence_threshold": leak_eq,
            "balanced_accuracy": {"value": bal, "chance": chance, "threshold": chance + bal_margin},
            "one_vs_rest_auc": {"value": ovr_auc, "threshold": 0.55, "pass": auc_pass},
            "calibration_ece": multiclass_ece(joint["probs"], labels),
            "nearest_neighbor_leakage": nn,
            "nearest_neighbor_equivalence_pass": nn_pass,
            "conditional_dependence": dep,
            "pass": channel_pass,
        }
    max_adv = max(primary_advantages) if primary_advantages else None
    return {
        "available": not missing and bool(channel_reports),
        "primary_variant": primary,
        "collection_role": analysis_role,
        "required_channels": list(nuisance_sources),
        "missing_required_channels": missing,
        "channels": channel_reports,
        "conditioned_on": [
            PATH_C_ECOLOGICAL_VALUE_CLASS_COLUMN,
            "public_context",
            "probe_action_id",
        ],
        "value_class_source": "validated_cross_fitted_ecological_value_class_artifact",
        "value_class_artifact_sha256": sorted(np.unique(
            data["ecological_value_class_artifact_sha256"][idx].astype(str)
        ).tolist()),
        "value_class_uncertainty": {
            "measure": "distance_to_nearest_out_of_fold_threshold",
            "minimum": float(np.min(
                data["ecological_value_class_threshold_distance"][idx]
            )),
            "median": float(np.median(
                data["ecological_value_class_threshold_distance"][idx]
            )),
        },
        "synthetic_registry_value_oracle": (
            _synthetic_registry_value_oracle_diagnostic(data, idx)
        ),
        "calibrated_multiclass_logloss": {
            "representation_advantage": max_adv,
            "equivalence_threshold": leak_eq,
        },
        "pass": bool(not missing and pass_items and all(pass_items)),
    }

def _path_c_value_control_measurement(
    data,
    mask: np.ndarray,
    threshold: float = 0.0,
    *,
    required_baselines: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if required_baselines is None:
        required_baselines = PATH_C_REQUIRED_HARD_BASELINES
    rejected_proxy_cols = [col for col in PATH_C_REJECTED_PROXY_VALUE_COLUMNS if col in data]
    cols: list[str] = []
    missing_by_baseline: dict[str, list[str]] = {}
    for baseline in required_baselines:
        baseline_cols = [
            f"{stem}_{baseline}"
            for stem in PATH_C_TRUE_VALUE_ADVANTAGE_STEMS
            if f"{stem}_{baseline}" in data
        ]
        if baseline_cols:
            cols.extend(baseline_cols)
        else:
            missing_by_baseline[str(baseline)] = [
                f"{stem}_{baseline}" for stem in PATH_C_TRUE_VALUE_ADVANTAGE_STEMS
            ]
    if missing_by_baseline:
        return {
            "available": False,
            "role": "legacy_secondary_readout_only",
            "decision_eligible": False,
            "reason": "missing true held-out value-control artifact columns for required baseline(s)",
            "accepted_column_patterns": [f"<stem>_<baseline>" for baseline in required_baselines],
            "metric_stems": list(PATH_C_TRUE_VALUE_ADVANTAGE_STEMS),
            "missing_by_baseline": missing_by_baseline,
            "rejected_proxy_columns_present": rejected_proxy_cols,
            "note": "Response-derived proxy columns and aggregate intent are not valid held-out value-control evidence.",
            "value": None,
            "threshold": threshold,
            "pass": False,
        }
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            "available": False,
            "reason": "empty_mask",
            "role": "legacy_secondary_readout_only",
            "decision_eligible": False,
            "pass": False,
        }
    keys = ep_keys(data, mask)
    items = {}
    passes = []
    values = []
    for col in cols:
        arr = np.asarray(data[col][idx], dtype=np.float64)
        finite = np.isfinite(arr)
        if not bool(finite.all()):
            arr = arr[finite]
            col_keys = keys[finite]
        else:
            col_keys = keys
        if arr.size == 0:
            items[col] = {"available": False, "reason": "no_finite_rows", "threshold": threshold, "pass": False}
            passes.append(False)
            values.append(-1.0e30)
            continue
        ci = bootstrap_ci(lambda ii, a=arr: float(a[ii].mean()), col_keys)
        passed = bool(float(arr.mean()) >= threshold and ci["lo"] > threshold)
        passes.append(passed)
        values.append(float(arr.mean()))
        items[col] = {"value": float(arr.mean()), "ci": ci, "threshold": threshold, "pass": passed}
    return {
        "available": True,
        "role": "legacy_secondary_readout_only",
        "decision_eligible": False,
        "name": "true held-out value-control advantage over every baseline",
        "value": min(values),
        "threshold": threshold,
        "metrics": items,
        "provenance_required": "held-out TD/residual-Q/action-value-ranking/value-gap-calibration/return artifact",
        "rejected_proxy_columns_present": rejected_proxy_cols,
        "pass": bool(all(passes)),
    }

def _path_c_power_null_measurement(
    res: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """C4 design readout with observed effects, intervals, TOST, and joint power."""

    normalized = _normalize_path_c_thresholds(dict(thresholds or {}))
    fingerprint = (
        (res.get("legacy_d1_gate_report") or {}).get("path_c_fingerprint_checks")
        or {}
    )
    target_keys = (
        "minimum_positive_power",
        "minimum_null_equivalence_power",
        "minimum_joint_power",
    )
    required_settings = (
        "positive_and_null_cluster_simulation_required",
        "joint_operating_characteristics_required",
        "equivalence_test",
        "secondary_multiplicity",
        "selection_role",
        "primary_alpha",
        "clusters_per_trial",
        "simulation_repetitions",
        "bootstrap_iterations",
        "seed",
        "equivalence_margin",
        "synthetic_power_min_advantage",
        *target_keys,
        "cluster_unit",
    )
    power_targets = {key: normalized.get(key) for key in target_keys}
    base_report = {
        "schema_version": "path_c_positive_null_secondary_v2",
        "available": False,
        "selection_role": normalized.get("selection_role"),
        "secondary_multiplicity": normalized.get("secondary_multiplicity"),
        "cluster_unit": normalized.get("cluster_unit"),
        "effect": {"positive": None, "null": None},
        "confidence_intervals": {"positive": None, "null": None},
        "tost": {"available": False},
        "positive": None,
        "null": None,
        "joint": None,
        "power_targets": power_targets,
        "power_targets_met": {
            "positive": False,
            "null_equivalence": False,
            "joint": False,
            "all": False,
        },
        "power_effect": None,
        "power_confidence_interval": None,
        "null_effect": None,
        "null_confidence_interval": None,
        "two_one_sided_equivalence_test": {"available": False},
        "joint_cluster_power_simulation": None,
        "preregistration": {
            "status": normalized.get("_preregistration_status"),
            "path": normalized.get("_source_path"),
            "sha256": normalized.get("_source_sha256"),
        },
        "fingerprint_admission": fingerprint,
        "decision_eligible": False,
        "pass": False,
    }
    preregistration_status = normalized.get("_preregistration_status")
    if preregistration_status != "frozen":
        return {
            **base_report,
            "reason": "power_analysis_not_loaded_from_frozen_preregistration",
            "preregistration_status": preregistration_status,
        }
    missing_settings = [
        key for key in required_settings if normalized.get(key) is None
    ]
    if missing_settings:
        return {
            **base_report,
            "reason": "missing_frozen_power_analysis_settings",
            "missing_settings": missing_settings,
        }
    if normalized["cluster_unit"] != "identity_group":
        return {
            **base_report,
            "reason": "power_analysis_cluster_unit_must_be_identity_group",
        }
    if (
        normalized["positive_and_null_cluster_simulation_required"] is not True
        or normalized["joint_operating_characteristics_required"] is not True
        or normalized["equivalence_test"] != "two_one_sided_tests"
        or normalized["secondary_multiplicity"] != "hierarchical_gatekeeping"
        or normalized["selection_role"] != "design"
    ):
        return {
            **base_report,
            "reason": "frozen_power_analysis_contract_mismatch",
        }

    positive_signal = (res.get("path_c_rv_readout") or {}).get("signal") or {}
    null_signal = (
        (res.get("path_c_terminal_axis_null_readout") or {}).get("signal") or {}
    )
    positive_effects = positive_signal.get("power_cluster_effects")
    null_effects = null_signal.get("power_cluster_effects")
    missing_effects = []
    if positive_effects is None:
        missing_effects.append("positive_identity_group_effects")
    if null_effects is None:
        missing_effects.append("null_identity_group_effects")
    if missing_effects:
        return {
            **base_report,
            "reason": "missing_cluster_effects",
            "missing_effects": missing_effects,
        }

    try:
        simulation = simulate_cluster_operating_characteristics(
            positive_effects,
            null_effects,
            clusters_per_trial=normalized["clusters_per_trial"],
            simulation_repetitions=normalized["simulation_repetitions"],
            bootstrap_iterations=normalized["bootstrap_iterations"],
            alpha=normalized["primary_alpha"],
            efficacy_margin=normalized["synthetic_power_min_advantage"],
            equivalence_margin=normalized["equivalence_margin"],
            seed=normalized["seed"],
        )
        targets = {key: float(normalized[key]) for key in target_keys}
        if any(not 0.0 < value <= 1.0 for value in targets.values()):
            raise ValueError("Power targets must lie in (0, 1].")
        if targets["minimum_joint_power"] > min(
            targets["minimum_positive_power"],
            targets["minimum_null_equivalence_power"],
        ):
            raise ValueError(
                "Minimum joint power cannot exceed either component target."
            )
    except (TypeError, ValueError) as exc:
        return {
            **base_report,
            "reason": "invalid_power_analysis_input",
            "detail": str(exc),
        }

    positive_met = bool(
        simulation["positive"]["power"] >= targets["minimum_positive_power"]
    )
    null_met = bool(
        simulation["null"]["power"]
        >= targets["minimum_null_equivalence_power"]
    )
    joint_met = bool(
        simulation["joint"]["power"] >= targets["minimum_joint_power"]
    )
    all_targets_met = bool(positive_met and null_met and joint_met)
    tost = {"available": True, **simulation["null"]["two_one_sided_tests"]}
    sources = {
        "positive_reference_baseline": positive_signal.get(
            "power_reference_baseline"
        ),
        "null_reference_baseline": null_signal.get("power_reference_baseline"),
        "positive_cluster_ids": positive_signal.get("power_cluster_ids"),
        "null_cluster_ids": null_signal.get("power_cluster_ids"),
    }
    return {
        **base_report,
        "available": True,
        "effect": {
            "positive": simulation["positive"]["effect"],
            "null": simulation["null"]["effect"],
        },
        "confidence_intervals": {
            "positive": simulation["positive"]["confidence_interval"],
            "null": simulation["null"]["confidence_interval"],
        },
        "tost": tost,
        "positive": simulation["positive"],
        "null": simulation["null"],
        "joint": simulation["joint"],
        "power_targets": targets,
        "power_targets_met": {
            "positive": positive_met,
            "null_equivalence": null_met,
            "joint": joint_met,
            "all": all_targets_met,
        },
        "power_effect": simulation["positive"]["effect"],
        "power_confidence_interval": simulation["positive"][
            "confidence_interval"
        ],
        "null_effect": simulation["null"]["effect"],
        "null_confidence_interval": simulation["null"]["confidence_interval"],
        "two_one_sided_equivalence_test": tost,
        "joint_cluster_power_simulation": simulation,
        "source": sources,
        "pass": all_targets_met,
    }
def path_c_decision_rule(
    measurements: dict[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    """Apply the strict version-3 staged decision."""
    if not isinstance(preregistration, FrozenPathCPreregistration):
        raise TypeError(
            "Path C judgment requires FrozenPathCPreregistration, not a threshold mapping."
        )
    return phase_b_go_no_go_rule(measurements, preregistration)


def _legacy_cross_identity_transfer_summary(
    transfer: Mapping[str, Any],
    conditional_factorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Report old readouts without manufacturing a joint decision gate."""

    measurement = transfer.get("measurement") or {}
    return {
        "name": "cross-identity R^V transfer and conditional factorization",
        "value": measurement.get("value"),
        "threshold": measurement.get("threshold", 0.0),
        "reported_transfer_threshold_check": bool(measurement.get("pass", False)),
        "conditional_factorization_available": bool(
            conditional_factorization.get("available", False)
        ),
        "joint_boolean_gate": None,
        "role": "legacy_secondary_readout_only",
        "decision_eligible": False,
    }


# -------------------------------------------------------------------- stages

def stage_gate(data, out_dir: Path, path_c_thresholds: str | None = None) -> None:
    path_c_threshold_values = _load_path_c_thresholds(path_c_thresholds)
    value_class_references: list[dict[str, Any]] = []
    preregistration_source = path_c_threshold_values.get("_source_path")
    if preregistration_source not in {None, ""}:
        preregistration = load_frozen_preregistration(preregistration_source)
        value_class_references = write_ecological_value_class_artifacts(
            build_ecological_value_class_artifacts(data, preregistration),
            out_dir,
        )
    rep: dict = {"floors": {"gated_rows": FLOOR_GATED, "min_class": FLOOR_MINCLASS,
                            "hard_floor_splits": list(HARD_FLOOR_SPLITS)},
                 "splits": {}, "per_partner": {},
                 "path_c_thresholds": path_c_threshold_values,
                 "path_c_ecological_value_class_artifacts": value_class_references,
                 "path_c_ecological_value_class_preregistration": (
                     str(preregistration_source)
                     if preregistration_source not in {None, ""}
                     else None
                 )}
    lab = data[f"label_k{JUDGED_K}"]
    gm = data["gate_main"].astype(bool) & ok_rows(data, JUDGED_K)
    named = {
        "indist_val": gm & data["indist_val"],
        "blind_terminal": gm & (data["split"] == "blind") & (data["family"] != "offaxis"),
        "indist_train": gm & (data["split"] == "train") & ~data["indist_val"],
        "dev": gm & (data["split"] == "dev"),
        "blind_offaxis": gm & (data["split"] == "blind") & (data["family"] == "offaxis"),
    }
    for name, mask in named.items():
        counts = {LABEL_NAMES[i]: int((lab[mask] == i).sum()) for i in range(3)}
        entry = {
            "n_gate_main": int(mask.sum()),
            "class_counts": counts,
            "floor_rows_pass": bool(mask.sum() >= FLOOR_GATED),
            "floor_class_pass": bool(min(counts.values()) >= FLOOR_MINCLASS),
            "hard_floor": name in HARD_FLOOR_SPLITS,
            "co_report_only": name not in HARD_FLOOR_SPLITS,
        }
        rep["splits"][name] = entry
    rep["ambig_drop_rate"] = {
        f"k{k}": float(data[f"ambig_k{k}"].astype(bool).mean()) for k in K_SET}
    rep["censoring_rate_k5"] = float(data[f"censored_k{JUDGED_K}"].astype(bool).mean())
    for p in np.unique(data["partner"]):
        m = (data["partner"] == p) & gm
        rep["per_partner"][str(p)] = {
            "n_gate_main": int(m.sum()),
            "episodes": int(np.unique(ep_keys(data, data["partner"] == p)).size),
            "class_counts": {LABEL_NAMES[i]: int((lab[m] == i).sum()) for i in range(3)},
        }
    # wiring check: does the partner-history channel actually vary across partners?
    # Criterion (amended 2026-07-07, gate round 2): positional twin partners (e.g.
    # server-left-claim vs server-right-claim, differing only in delivery position)
    # are IDENTICAL at option-kind granularity by construction, so pairwise-min TV
    # is the wrong operationalization. The prereg intent ("通道随伙伴变化") is that
    # each partner's stream differs from at least one other; kind-collapsed twin
    # pairs are recorded informationally (they share family/response semantics).
    names = sorted(np.unique(data["partner"][data["split"] == "train"]).tolist())
    n_vocab = len(data["_vocab"])
    hists = {}
    for p in names:
        m = data["partner"] == p
        h = np.bincount(data["hist_kind"][m].ravel(), minlength=n_vocab).astype(float)
        h[0] = 0.0  # drop PAD
        hists[p] = h / max(h.sum(), 1.0)
    tv_pairs = {(a, b): float(0.5 * np.abs(hists[a] - hists[b]).sum())
                for i, a in enumerate(names) for b in names[i + 1:]}
    tv = list(tv_pairs.values())
    per_partner_max = {
        a: max(tv_pairs.get((a, b), tv_pairs.get((b, a), 0.0))
               for b in names if b != a) for a in names}
    rep["history_channel_cross_partner_tv"] = {
        "min": min(tv), "max": max(tv), "mean": float(np.mean(tv))}
    rep["kind_collapsed_pairs"] = [
        {"pair": list(k), "tv": v} for k, v in tv_pairs.items() if v < 0.01]
    rep["per_partner_max_tv"] = per_partner_max
    rep["history_channel_varies"] = bool(min(per_partner_max.values()) > 0.05)
    path_c_required = bool(
        "partner_set" in data and np.any(data["partner_set"].astype(str) == "path_c_synthetic")
    )
    fingerprint_checks = _path_c_fingerprint_checks(data, gm, lab, path_c_threshold_values)
    rep["path_c_fingerprint_required"] = path_c_required
    rep["path_c_fingerprint_checks"] = fingerprint_checks
    rep["path_c_preregistration_status"] = path_c_threshold_values.get("_preregistration_status")
    rep["path_c_preregistration_frozen"] = bool(
        path_c_threshold_values.get("_preregistration_status") == "frozen"
    )
    base_pass = bool(
        all(rep["splits"][s]["floor_rows_pass"] and rep["splits"][s]["floor_class_pass"]
            for s in HARD_FLOOR_SPLITS) and rep["history_channel_varies"])
    rep["PASS"] = bool(
        base_pass
        and (not path_c_required or bool(fingerprint_checks.get("pass", False)))
        and (not path_c_required or bool(rep["path_c_preregistration_frozen"]))
    )
    rep["soft_floor_failures"] = [
        s for s, e in rep["splits"].items()
        if e["co_report_only"] and not (e["floor_rows_pass"] and e["floor_class_pass"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_gate_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"PASS": rep["PASS"],
                      "soft_floor_failures": rep["soft_floor_failures"],
                      **{k: {kk: v[kk] for kk in ("n_gate_main", "class_counts")}
                         for k, v in rep["splits"].items()}}, indent=1), flush=True)


def _train_one(variant, k, seed, b_tr, b_va, n_vocab, state_dim, n_partners, dev):
    torch.manual_seed(seed)
    model = D1Net(variant, n_vocab, state_dim, n_partners).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = b_tr["state"].shape[0]
    best, best_state, patience = float("inf"), None, 0
    for _epoch in range(40):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            sl = perm[i:i + 1024]
            logits = model(b_tr["state"][sl], b_tr["hk"][sl], b_tr["hd"][sl],
                           b_tr["ha"][sl], b_tr["hl"][sl], b_tr["ego_opt"][sl],
                           b_tr["partner_idx"][sl])
            loss = nn.functional.cross_entropy(logits, b_tr["labels"][sl])
            optim.zero_grad(); loss.backward(); optim.step()
        p_va = predict(model, b_va)
        ll = logloss(p_va, b_va["labels"].cpu().numpy())
        if ll < best - 1e-4:
            best, patience = ll, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    return model, best


def stage_tune(data, out_dir: Path, device: str) -> None:
    # physically restrict to training partners before anything else (no-peek guarantee)
    rows = np.flatnonzero(data["split"] == "train")
    n_total = data["episode_id"].shape[0]
    sub = {k: (v[rows] if isinstance(v, np.ndarray) and v.shape[:1] == (n_total,) else v)
           for k, v in data.items()}
    train_partners = sorted(np.unique(sub["partner"]).tolist())
    partner_to_idx = {p: i for i, p in enumerate(train_partners)}
    gmain = sub["gate_main"].astype(bool)
    dev = torch.device(device)
    n_vocab = len(sub["_vocab"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"train_partners": train_partners, "k_blocks": {}}

    for k in K_SET:
        ok = ok_rows(sub, k)
        tr = ~sub["indist_val"] & gmain & ok
        va = sub["indist_val"] & gmain & ok
        va_main = va  # D1-rev: training gate == judged gate (strict gate is co-report)
        # norm stats from exactly this K's training rows (ambig-filtered; codex re-review)
        norm_x = build_state(sub, np.flatnonzero(tr))
        norm = (norm_x.mean(0), norm_x.std(0) + 1e-6)
        state_dim = norm_x.shape[1]
        b_tr = tensors_for(sub, np.flatnonzero(tr), norm, partner_to_idx, dev, k)
        b_va = tensors_for(sub, np.flatnonzero(va), norm, partner_to_idx, dev, k)
        b_va_main = b_va
        variants = ("full", "nohist", "idoracle") if k == JUDGED_K else ("full", "nohist")
        blk: dict = {"n_train": int(tr.sum()), "n_val_main": int(va.sum()),
                     "variants": {}}
        probs_main: dict[str, list[np.ndarray]] = {}
        for variant in variants:
            blk["variants"][variant] = []
            for seed in range(N_SEEDS):
                model, best_ll = _train_one(variant, k, seed, b_tr, b_va, n_vocab,
                                            state_dim, len(train_partners), dev)
                path = out_dir / f"model_{variant}_k{k}_s{seed}.pt"
                torch.save({"state_dict": model.state_dict(), "variant": variant,
                            "k": k, "n_vocab": n_vocab, "state_dim": state_dim,
                            "n_partners": len(train_partners),
                            "norm_mean": norm[0], "norm_std": norm[1],
                            "train_partners": train_partners}, path)
                pm = predict(model, b_va_main)
                probs_main.setdefault(variant, []).append(pm)
                blk["variants"][variant].append({
                    "seed": seed, "val_logloss": best_ll,
                    "val_main_logloss": logloss(pm, b_va_main["labels"].cpu().numpy()),
                    "val_main_auc_ps": auc_binary(
                        pm[:, 2], b_va_main["labels"].cpu().numpy() == 2),
                })
        lab_main = b_va_main["labels"].cpu().numpy()
        ens = {v: np.mean(probs_main[v], axis=0) for v in probs_main}
        blk["indist_val_main_ensemble"] = metric_block(
            ens["full"], ens["nohist"], lab_main)
        if "idoracle" in ens:
            blk["indist_val_main_ensemble"]["auc_ps_idoracle"] = auc_binary(
                ens["idoracle"][:, 2], lab_main == 2)
        report["k_blocks"][f"k{k}"] = blk
    (out_dir / "tune_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v["indist_val_main_ensemble"]
                      for k, v in report["k_blocks"].items()}, indent=1), flush=True)


def _load_models(out_dir: Path, device: str, k: int):
    models: dict[str, list] = {"full": [], "nohist": []}
    norm = partner_to_idx = None
    shas = {}
    for variant in ("full", "nohist"):
        for seed in range(N_SEEDS):
            path = out_dir / f"model_{variant}_k{k}_s{seed}.pt"
            ck = torch.load(path, map_location=device, weights_only=False)
            m = D1Net(variant, ck["n_vocab"], ck["state_dim"], ck["n_partners"])
            m.load_state_dict(ck["state_dict"])
            m.to(device)
            models[variant].append(m)
            shas[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            norm = (ck["norm_mean"], ck["norm_std"])
            partner_to_idx = {p: i for i, p in enumerate(ck["train_partners"])}
    return models, norm, partner_to_idx, shas


def _eval_pack(data, mask, models, norm, partner_to_idx, device, k):
    idx = np.flatnonzero(mask)
    if idx.size == 0:  # strict onset gates can empty a co-report split (codex rev)
        empty = np.zeros((0, 3), dtype=np.float64)
        return idx, None, np.zeros(0, dtype=np.int64), {v: empty for v in models}
    batch = tensors_for(data, idx, norm, partner_to_idx, torch.device(device), k)
    labels = batch["labels"].cpu().numpy()
    probs = {v: np.mean([predict(m, batch) for m in models[v]], axis=0)
             for v in models}
    return idx, batch, labels, probs


def _load_path_c_thresholds(path: str | None) -> dict[str, Any]:
    if path in {None, ""}:
        return {
            "_preregistration_status": "missing",
            "_source_path": None,
            "_source_sha256": None,
        }
    preregistration = load_frozen_preregistration(path)
    payload = preregistration.payload
    thresholds = dict(preregistration.thresholds)
    thresholds["required_baseline_variants"] = preregistration.required_baselines
    thresholds["effective_episode_floor"] = payload["budget"]["effective_episode_floor"]
    thresholds["effective_transition_floor"] = payload["budget"]["effective_transition_floor"]
    power_analysis = payload["power_analysis"]
    for key in (
        "positive_and_null_cluster_simulation_required",
        "joint_operating_characteristics_required",
        "equivalence_test",
        "secondary_multiplicity",
        "selection_role",
        "primary_alpha",
        "cluster_unit",
        "clusters_per_trial",
        "simulation_repetitions",
        "bootstrap_iterations",
        "seed",
        "equivalence_margin",
        "minimum_positive_power",
        "minimum_null_equivalence_power",
        "minimum_joint_power",
    ):
        thresholds[key] = power_analysis[key]
    thresholds["_source_path"] = str(preregistration.path)
    thresholds["_source_sha256"] = preregistration.sha256
    thresholds["_preregistration_status"] = "frozen"
    thresholds["_preregistration_status_raw"] = payload.get("status")
    thresholds["_preregistration_version"] = payload.get("version")
    thresholds["_preregistration_freeze_timestamp"] = payload.get("freeze_timestamp")
    return thresholds


def _stage_path_c_readout(
    out_dir: Path,
    preregistration_path: str,
    artifact_manifest_path: str,
) -> None:
    """Recompute Path C decisions from hash-bound raw artifacts only."""
    out_path = out_dir / "readout_frozen.json"
    if out_path.exists():
        raise RuntimeError("readout_frozen.json exists - single look already consumed")
    inputs = load_and_validate_path_c_inputs(
        preregistration_path,
        artifact_manifest_path,
    )
    measurements = assemble_path_c_measurements(inputs)
    decision = evaluate_path_c_decision(measurements, inputs.preregistration)
    result = {
        "schema_version": "path_c_readout_v3",
        "decision_scope": {
            "order": [
                "software_conformance",
                "instrument_validity",
                "design_and_calibration_freeze",
                "locked_primary_efficacy",
                "secondary_mechanisms",
            ],
            "type_b_review_required": True,
        },
        "preregistration": {
            "path": str(inputs.preregistration.path),
            "sha256": inputs.preregistration.sha256,
            "version": inputs.preregistration.payload["version"],
            "freeze_timestamp": inputs.preregistration.payload["freeze_timestamp"],
        },
        "artifact_manifest": {
            "path": str(inputs.manifest.path),
            "sha256": inputs.manifest.sha256,
            "resolved_path_c_sha256": inputs.manifest.resolved_path_c_sha256,
        },
        "validated_evidence_contract": measurements["artifact_contract"],
        "path_c_decision": asdict(decision),
        "artifact_read_back_only": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "path_c_status": decision.status,
        "allowed_claim": decision.allowed_claim,
    }, indent=2), flush=True)


def stage_readout(
    data,
    out_dir: Path,
    device: str,
    path_c_thresholds: str | None = None,
    path_c_artifact_manifest: str | None = None,
) -> None:
    path_c_mode = bool(path_c_thresholds or path_c_artifact_manifest)
    if path_c_mode:
        if not path_c_thresholds or not path_c_artifact_manifest:
            raise ValueError(
                "Path C readout requires both --path-c-preregistration and "
                "--path-c-artifact-manifest."
            )
        _stage_path_c_readout(
            out_dir,
            path_c_thresholds,
            path_c_artifact_manifest,
        )
        return
    out_path = out_dir / "readout_frozen.json"
    if out_path.exists():
        raise RuntimeError("readout_frozen.json exists — single look already consumed")
    gate_path = out_dir / "dataset_gate_report.json"
    if not gate_path.exists():
        raise RuntimeError("dataset_gate_report.json missing — run --stage gate first")
    gate = json.loads(gate_path.read_text())
    path_c_threshold_values = _load_path_c_thresholds(path_c_thresholds)
    if not gate.get("PASS"):
        raise RuntimeError(
            "sufficiency gate FAILED — no readout permitted on insufficient samples "
            f"(prereg §5); gate says: { {s: gate['splits'][s] for s in HARD_FLOOR_SPLITS} }")
    value_class_references = gate.get(
        "path_c_ecological_value_class_artifacts", []
    )
    if value_class_references:
        preregistration_source = gate.get(
            "path_c_ecological_value_class_preregistration"
        )
        if preregistration_source in {None, ""}:
            raise RuntimeError(
                "Ecological value-class artifacts lack their frozen preregistration."
            )
        attach_validated_ecological_value_classes(
            data,
            value_class_references,
            load_frozen_preregistration(preregistration_source),
        )

    models, norm, partner_to_idx, shas = _load_models(out_dir, device, JUDGED_K)
    ok5 = ok_rows(data, JUDGED_K)
    gm = data["gate_main"].astype(bool) & ok5
    gs = data["gate_strict"].astype(bool) & ok5
    path_c_threshold_source = path_c_threshold_values.get("_source_path")
    rv_spec = json.loads(str(data["_rv_summary_spec"][0])) if "_rv_summary_spec" in data else {}
    res: dict = {
        "judged_k": JUDGED_K,
        "model_sha256": shas,
        "gate_report_pass": bool(gate.get("PASS")),
        "legacy_d1_gate_report": gate,
        "path_c_readout_budget_source": "artifact_effective_counts_not_legacy_d1_floor",
        "rv_summary_spec": rv_spec,
        "effective_episodes": (
            int(data["_effective_episodes"][0]) if "_effective_episodes" in data else None
        ),
        "effective_transitions": (
            int(data["_effective_transitions"][0]) if "_effective_transitions" in data else None
        ),
        "path_c_thresholds": path_c_threshold_values,
        "path_c_thresholds_path": str(path_c_threshold_source) if path_c_threshold_source else None,
        "path_c_ecological_value_class_artifacts": value_class_references,
        "splits": {},
    }

    def add(name, mask, boot=False):
        idx, batch, labels, probs = _eval_pack(
            data, mask, models, norm, partner_to_idx, device, JUDGED_K)
        if idx.size == 0:
            blk = {"n": 0, "note": "empty mask under onset gate"}
            res["splits"][name] = blk
            return blk
        blk = metric_block(probs["full"], probs["nohist"], labels)
        blk["censoring_rate"] = float(data[f"censored_k{JUDGED_K}"][idx].mean()) \
            if idx.size else float("nan")
        if boot and labels.size:
            keys = ep_keys(data, mask)
            pos = labels == 2
            blk["gain_ci"] = bootstrap_ci(
                lambda ii: auc_binary(probs["full"][ii, 2], pos[ii])
                - auc_binary(probs["nohist"][ii, 2], pos[ii]), keys)
            blk["dlogloss_ci"] = bootstrap_ci(
                lambda ii: logloss(probs["nohist"][ii], labels[ii])
                - logloss(probs["full"][ii], labels[ii]), keys)
        res["splits"][name] = blk
        return blk

    iv = add("indist_val_main", gm & data["indist_val"], boot=True)
    add("indist_val_strict", gs & data["indist_val"])
    add("dev_main", gm & (data["split"] == "dev"))
    bt = add("blind_terminal_main",
             gm & (data["split"] == "blind") & (data["family"] != "offaxis"), boot=True)
    add("blind_terminal_strict",
        gs & (data["split"] == "blind") & (data["family"] != "offaxis"))
    add("blind_offaxis_main",
        gm & (data["split"] == "blind") & (data["family"] == "offaxis"))
    for fam in ("yield", "claim"):
        add(f"blind_{fam}_main", gm & (data["split"] == "blind") & (data["family"] == fam))
    for p in sorted(np.unique(data["partner"][data["split"] == "blind"]).tolist()):
        add(f"partner::{p}", gm & (data["partner"] == p))
    for ego in sorted(np.unique(data["ego"]).tolist()):
        add(f"blind_terminal_ego::{ego}",
            gm & (data["split"] == "blind") & (data["family"] != "offaxis")
            & (data["ego"] == ego))

    # K-sensitivity: independent ensembles per K (codex finding 5)
    res["k_sensitivity"] = {}
    for k in K_SET:
        if k == JUDGED_K:
            res["k_sensitivity"][f"k{k}"] = {
                "indistinguishable_value_gain": iv["gain"],
                "blind_terminal_gain": bt["gain"],
            }
            continue
        mk, nk, pk, _ = _load_models(out_dir, device, k)
        okk = ok_rows(data, k)
        gmk = data["gate_main"].astype(bool) & okk
        _, _, lab_i, pr_i = _eval_pack(
            data, gmk & data["indist_val"], mk, nk, pk, device, k)
        _, _, lab_b, pr_b = _eval_pack(
            data, gmk & (data["split"] == "blind") & (data["family"] != "offaxis"),
            mk, nk, pk, device, k)
        res["k_sensitivity"][f"k{k}"] = {
            "indistinguishable_value_gain": (auc_binary(pr_i["full"][:, 2], lab_i == 2)
                         - auc_binary(pr_i["nohist"][:, 2], lab_i == 2))
            if lab_i.size else float("nan"),
            "blind_terminal_gain": (auc_binary(pr_b["full"][:, 2], lab_b == 2)
                                 - auc_binary(pr_b["nohist"][:, 2], lab_b == 2))
            if lab_b.size else float("nan"),
        }
    signs_i = {
        np.sign(v["indistinguishable_value_gain"])
        for v in res["k_sensitivity"].values()
    }
    signs_b = {
        np.sign(v["blind_terminal_gain"])
        for v in res["k_sensitivity"].values()
    }
    res["k_sensitivity"]["direction_consistent"] = bool(
        len(signs_i) == 1 and len(signs_b) == 1)

    # R3: semantics vs fingerprint on blind gate_main points (K=5 FULL models)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score as nmi
        bm = gm & (data["split"] == "blind")
        idx = np.flatnonzero(bm)
        batch = tensors_for(data, idx, norm, partner_to_idx,
                            torch.device(device), JUDGED_K)
        mechanism = data.get("mechanism_key", data["family"])
        identity = data.get("identity_key", data["partner"])
        fam_lab = np.searchsorted(np.unique(mechanism[bm]), mechanism[bm])
        pid_lab = np.searchsorted(np.unique(identity[bm]), identity[bm])
        nmis_fam, nmis_pid = [], []
        for m in models["full"]:
            h = hidden_of(m, batch)
            cl = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(h)
            nmis_fam.append(float(nmi(fam_lab, cl)))
            nmis_pid.append(float(nmi(pid_lab, cl)))
        res["r3_nmi"] = {"family_median": float(np.median(nmis_fam)),
                         "partner_id_median": float(np.median(nmis_pid)),
                         "per_seed_family": nmis_fam, "per_seed_partner": nmis_pid,
                         "visualization_only": True,
                         "pass_relevant": False}
    except Exception as exc:  # noqa: BLE001
        res["r3_nmi"] = {"error": str(exc)}

    path_c_mask = gm
    if "partner_set" in data and np.any(data["partner_set"].astype(str) == "path_c_synthetic"):
        path_c_mask = gm & (data["partner_set"].astype(str) == "path_c_synthetic")
    path_c_idx = np.flatnonzero(path_c_mask)
    path_c_features = _path_c_collected_variant_features(data, path_c_idx) if path_c_idx.size else {}
    required_baselines = _path_c_required_baselines(path_c_threshold_values)
    res["path_c_baseline_completeness"] = _path_c_baseline_completeness(
        path_c_features, path_c_threshold_values
    )
    rv_readout = _path_c_rv_readout(
        data,
        path_c_mask,
        models,
        norm,
        partner_to_idx,
        device,
        split_name="episode_crossfit",
        threshold=float(_path_c_threshold(
            path_c_threshold_values, "response_advantage", 0.0
        )),
        required_baselines=required_baselines,
    )
    res["path_c_rv_readout"] = rv_readout
    res["response_readout_secondary"] = rv_readout.get(
        "measurement", {"pass": False}
    )
    transfer_mask = path_c_mask
    if "mode_fingerprint_id" in data:
        transfer_mask = transfer_mask & (data["mode_fingerprint_id"].astype(int) >= 0)
    transfer = _path_c_rv_readout(
        data,
        transfer_mask,
        models,
        norm,
        partner_to_idx,
        device,
        split_name="cross_identity",
        fold_keys=data["identity_key"] if "identity_key" in data else data["partner"],
        threshold=float(_path_c_threshold(
            path_c_threshold_values, "transfer_advantage", 0.0
        )),
        required_baselines=required_baselines,
    )
    res["path_c_transfer_readout"] = transfer
    terminal_null_mask = (
        gm
        & (data["split"] == "blind")
        & (data["family"] == "offaxis")
    )
    terminal_null = _path_c_rv_readout(
        data,
        terminal_null_mask,
        models,
        norm,
        partner_to_idx,
        device,
        split_name="terminal_axis_null",
        threshold=0.0,
        required_baselines=required_baselines,
    )
    res["path_c_terminal_axis_null_readout"] = terminal_null
    c_fact = _path_c_conditional_factorization_audit(
        data,
        transfer_mask,
        models,
        norm,
        partner_to_idx,
        device,
        path_c_threshold_values,
    )
    res["conditional_factorization_secondary"] = c_fact
    res["cross_identity_transfer_secondary"] = (
        _legacy_cross_identity_transfer_summary(transfer, c_fact)
    )
    leakage = _path_c_leakage_battery(
        data,
        transfer_mask,
        models,
        norm,
        partner_to_idx,
        device,
        path_c_threshold_values,
    )
    res["path_c_leakage_battery"] = leakage
    res["conditioned_leakage_secondary"] = {
        "name": "conditioned nuisance leakage exclusion",
        "value": (
            (leakage.get("calibrated_multiclass_logloss") or {}).get(
                "representation_advantage"
            )
            if leakage.get("available") else None
        ),
        "threshold": float(_path_c_threshold(
            path_c_threshold_values,
            "leakage_logloss_advantage_equivalence",
            PATH_C_LEAK_ADVANTAGE_EQ,
        )),
        "direction": "upper",
        "pass": bool(leakage.get("pass", False)),
    }
    res["value_control_secondary"] = _path_c_value_control_measurement(
        data,
        path_c_mask,
        threshold=float(_path_c_threshold(
            path_c_threshold_values, "value_advantage", 0.0
        )),
        required_baselines=required_baselines,
    )
    res["power_and_null_secondary"] = _path_c_power_null_measurement(
        res, path_c_threshold_values
    )
    res["path_c_joint_power_simulation"] = res["power_and_null_secondary"].get(
        "joint_cluster_power_simulation"
    )

    g_ind, g_bld = iv["gain"], bt["gain"]
    res["legacy_response_diagnostics_secondary"] = {
        "indistinguishable_value_gain": g_ind,
        "blind_terminal_gain": g_bld,
        "indistinguishable_value_gain_check": bool(
            g_ind >= RESPONSE_GAIN_FLOOR and iv["dlogloss_ci"]["lo"] > 0
        ),
        "blind_terminal_retention_check": bool(
            g_bld >= BLIND_RESPONSE_RETENTION * g_ind and bt["gain_ci"]["lo"] > 0
        ),
        "k_direction_consistent": res["k_sensitivity"]["direction_consistent"],
        "thresholds": {
            "response_gain_floor": RESPONSE_GAIN_FLOOR,
            "blind_response_retention": BLIND_RESPONSE_RETENTION,
        },
        "decision_eligible": False,
    }
    out_path.write_text(json.dumps(res, indent=1))
    print(
        json.dumps(res["legacy_response_diagnostics_secondary"], indent=1),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=None, help="directory of diag_d1 npz chunks")
    ap.add_argument("--out", required=True, help="output dir (models + reports)")
    ap.add_argument("--stage", required=True, choices=("gate", "tune", "readout"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--path-c-preregistration",
        default=None,
        help="Frozen Path C preregistration; thresholds are never accepted separately.",
    )
    ap.add_argument("--path-c-artifact-manifest", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    strict_path_c_readout = bool(
        args.stage == "readout"
        and (args.path_c_preregistration or args.path_c_artifact_manifest)
    )
    if strict_path_c_readout:
        stage_readout(
            None,
            out_dir,
            args.device,
            args.path_c_preregistration,
            args.path_c_artifact_manifest,
        )
        return
    if not args.chunks:
        raise ValueError("--chunks is required for legacy gate, tune, and readout stages.")
    data = load_chunks(args.chunks)
    if args.stage == "gate":
        stage_gate(data, out_dir, args.path_c_preregistration)
    elif args.stage == "tune":
        stage_tune(data, out_dir, args.device)
    else:
        stage_readout(
            data,
            out_dir,
            args.device,
            args.path_c_preregistration,
            args.path_c_artifact_manifest,
        )


if __name__ == "__main__":
    main()
