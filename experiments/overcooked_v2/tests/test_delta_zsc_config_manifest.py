from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.path_c.experiment import (
    CONFIG_VERSION,
    MANIFEST_VERSION,
    METHOD_VERSION,
    OFFICIAL_PROTOCOL_VERSION,
    PartnerManifest,
    PartnerRun,
    RUN_BUDGETS,
    load_config,
    load_partner_manifest,
    official_training_key,
    validate_partner_manifest,
    validate_seed_training_manifest,
    validate_config,
)
from src.path_c.storage import sha256_path
from src.path_c.response_targets import official_partner_observation_planes


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments" / "overcooked_v2" / "configs"


def test_response_planes_match_pinned_official_default_observation_order() -> None:
    # With 39 channels the pinned shape equation gives three ingredients. Each
    # agent block is position[1], direction[4], inventory[5], with ego first.
    planes = official_partner_observation_planes(39)
    assert planes.visibility_channel == 10
    assert planes.direction_channels == (11, 12, 13, 14)
    assert planes.inventory_channels == (15, 16, 17, 18, 19)
    assert planes.interaction_channels == planes.inventory_channels
    with pytest.raises(ValueError, match="Official DEFAULT"):
        official_partner_observation_planes(41)


def _run(index: int, role: str, *, parent: str | None = None) -> PartnerRun:
    return PartnerRun(
        run_id=f"run-{index}",
        role=role,
        checkpoint=Path(f"/tmp/checkpoint-{index}"),
        checkpoint_sha256=f"{index:064x}",
        parent_training_run_id=parent or f"parent-{index}",
        generation_mechanism="fixture",
        seed=index,
        seed_index=None,
        jax_prng_key=(0, 10_000 + index),
        owner_seed_index=None,
        co_training_group_id=f"group-{index}",
        partner_type_id=None,
    )


def _valid_manifest() -> PartnerManifest:
    roles = (
        "owner_source",
        "generator_init_source",
        "generator_init_source",
        "development_support",
        "development_support",
        "calibration",
        "calibration",
        "confirmatory",
        "confirmatory",
    )
    return PartnerManifest(
        layout="test_time_simple",
        runs=tuple(_run(index + 1, role) for index, role in enumerate(roles)),
    )


def test_registered_versions_and_run_budgets() -> None:
    assert CONFIG_VERSION == 9
    assert MANIFEST_VERSION == 2
    assert METHOD_VERSION == "delta_zsc_v6_end_to_end_bayes_coordination"
    assert OFFICIAL_PROTOCOL_VERSION == "overcooked_v2_iclr2025_5ce1707_v1"
    assert RUN_BUDGETS["mechanical"].num_envs == 4
    assert RUN_BUDGETS["development"].environment_steps == 1_228_800
    assert RUN_BUDGETS["formal"].num_envs == 256
    assert RUN_BUDGETS["formal"].environment_steps == 29_949_952


@pytest.mark.parametrize(
    ("filename", "run_kind", "expected_envs"),
    (
        ("delta_zsc_simple_development.yaml", "development", 32),
        ("delta_zsc_wide_development.yaml", "mechanical", 4),
        ("delta_zsc_simple_mechanical_e2e.yaml", "mechanical", 4),
        ("delta_zsc_simple_formal.yaml", "formal", 256),
        ("delta_zsc_wide_formal.yaml", "formal", 256),
    ),
)
def test_v6_configs_load_with_registered_budget(
    filename: str, run_kind: str, expected_envs: int
) -> None:
    config = load_config(CONFIGS / filename, run_kind=run_kind)
    assert config.environment.num_envs == expected_envs
    assert config.training.rollout_length == (
        16 if "mechanical_e2e" in filename else 256
    )
    assert len(config.fingerprint) == 64


def test_v6_formal_signal_and_schedule_values_are_frozen() -> None:
    config = load_config(
        CONFIGS / "delta_zsc_simple_formal.yaml", run_kind="formal"
    )
    assert config.model.latent_dim == 8
    assert config.model.posterior_particles == 16
    assert config.model.log_standard_deviation_minimum == -5.0
    assert config.model.log_standard_deviation_maximum == 2.0
    assert config.loss.raw_q_weight == 1.0
    assert config.loss.counterfactual_weight == 1.0
    assert config.loss.response_weight == 1.0
    assert config.loss.decision_equivalence_weight == 0.1
    assert config.loss.information_bottleneck_weight == 0.001
    assert config.loss.q_policy_weight == 0.25
    assert config.loss.robust_generalist_weight == 0.1
    assert config.loss.policy_belief_gradient_scale == 0.1
    assert config.training.context_dropout_initial == 0.30
    assert config.training.context_dropout_final == 0.10
    assert config.anchors.interval_updates == 16
    assert config.anchors.replay_capacity == 512
    assert config.partner_generator.target_polyak_coefficient == 0.005
    assert config.partner_generator.maximum_generator_probability == 0.75
    invalid = replace(
        config,
        loss=replace(config.loss, q_policy_weight=0.5),
    )
    with pytest.raises(ValueError, match="q_policy_weight"):
        validate_config(invalid)


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    import yaml

    source = CONFIGS / "delta_zsc_simple_development.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    candidate = tmp_path / "invalid.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(candidate, run_kind="mechanical")


def test_manifest_accepts_run_disjoint_roles() -> None:
    validate_partner_manifest(_valid_manifest())


def test_manifest_requires_real_prng_provenance_for_every_partner() -> None:
    manifest = _valid_manifest()
    missing = replace(manifest.runs[1], jax_prng_key=None)
    with pytest.raises(ValueError, match="real two-word JAX key"):
        validate_partner_manifest(
            replace(manifest, runs=(manifest.runs[0], missing, *manifest.runs[2:]))
        )


def _formal_seed_manifest() -> PartnerManifest:
    mechanisms = ("sp", "op", "sa", "fcp")
    rows: list[PartnerRun] = [
        PartnerRun(
            run_id="owner-0",
            role="owner_source",
            checkpoint=Path("/tmp/formal-owner-0"),
            checkpoint_sha256=f"{1:064x}",
            parent_training_run_id="formal-parent-1",
            generation_mechanism="rnn-sp",
            seed=42,
            seed_index=0,
            jax_prng_key=official_training_key(0),
            owner_seed_index=0,
            co_training_group_id="formal-group-1",
            partner_type_id=None,
        )
    ]
    index = 2
    for role, repetitions in (
        ("generator_init_source", 1),
        ("development_support", 4),
    ):
        for repetition in range(repetitions):
            for mechanism_index, mechanism in enumerate(mechanisms):
                rows.append(
                    PartnerRun(
                        run_id=f"{role}-{mechanism}-{repetition}",
                        role=role,
                        checkpoint=Path(f"/tmp/formal-source-{index}"),
                        checkpoint_sha256=f"{index:064x}",
                        parent_training_run_id=f"formal-parent-{index}",
                        generation_mechanism=mechanism,
                        seed=10_000 + index,
                        seed_index=None,
                        jax_prng_key=(mechanism_index + 1, 10_000 + index),
                        owner_seed_index=0,
                        co_training_group_id=f"formal-group-{index}",
                        partner_type_id=None,
                    )
                )
                index += 1
    return PartnerManifest(layout="test_time_simple", runs=tuple(rows))


def test_formal_seed_manifest_requires_exact_balanced_v6_support() -> None:
    manifest = _formal_seed_manifest()
    validate_seed_training_manifest(manifest, owner_seed_index=0, formal=True)

    missing_fcp = next(
        index
        for index, run in enumerate(manifest.runs)
        if run.role == "development_support" and run.generation_mechanism == "fcp"
    )
    reduced = replace(
        manifest,
        runs=manifest.runs[:missing_fcp] + manifest.runs[missing_fcp + 1 :],
    )
    with pytest.raises(ValueError, match="development_support"):
        validate_seed_training_manifest(reduced, owner_seed_index=0, formal=True)


def test_manifest_rejects_checkpoint_parent_and_group_leakage() -> None:
    manifest = _valid_manifest()
    duplicate_hash = replace(
        manifest.runs[-1], checkpoint_sha256=manifest.runs[0].checkpoint_sha256
    )
    with pytest.raises(ValueError, match="checkpoint overlap"):
        validate_partner_manifest(
            replace(manifest, runs=(*manifest.runs[:-1], duplicate_hash))
        )

    shared_parent = replace(
        manifest.runs[5], parent_training_run_id=manifest.runs[0].parent_training_run_id
    )
    with pytest.raises(ValueError, match="parent runs overlap"):
        validate_partner_manifest(
            replace(manifest, runs=(*manifest.runs[:5], shared_parent, *manifest.runs[6:]))
        )

    shared_group = replace(
        manifest.runs[5], co_training_group_id=manifest.runs[0].co_training_group_id
    )
    with pytest.raises(ValueError, match="co-training group"):
        validate_partner_manifest(
            replace(manifest, runs=(*manifest.runs[:5], shared_group, *manifest.runs[6:]))
        )


def test_manifest_file_hash_is_verified(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    runs = []
    for index, run in enumerate(manifest.runs):
        checkpoint = tmp_path / f"checkpoint-{index}.bin"
        checkpoint.write_bytes(f"checkpoint-{index}".encode("utf-8"))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        runs.append(replace(run, checkpoint=checkpoint, checkpoint_sha256=digest))
    materialized = replace(manifest, runs=tuple(runs))
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(materialized.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_partner_manifest(path, expected_layout="test_time_simple")
    assert loaded.to_mapping() == materialized.to_mapping()

    runs[0].checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_partner_manifest(path, expected_layout="test_time_simple")


def test_manifest_directory_hash_uses_the_canonical_fingerprint(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    runs = []
    for index, run in enumerate(manifest.runs):
        checkpoint = tmp_path / f"checkpoint-{index}"
        checkpoint.mkdir()
        (checkpoint / "metadata").write_text(
            f"metadata-{index}", encoding="utf-8"
        )
        data = checkpoint / "nested" / "data.bin"
        data.parent.mkdir()
        data.write_bytes(bytes((index, index + 1)))
        runs.append(
            replace(
                run,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_path(checkpoint),
            )
        )
    materialized = replace(manifest, runs=tuple(runs))
    path = tmp_path / "directory-manifest.json"
    path.write_text(
        json.dumps(materialized.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_partner_manifest(path, expected_layout="test_time_simple")
    assert loaded.to_mapping() == materialized.to_mapping()
