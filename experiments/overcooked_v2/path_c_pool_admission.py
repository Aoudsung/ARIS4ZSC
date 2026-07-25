"""Fail-closed admission for the R015 opportunity audit's two-family support."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from experiments.overcooked_v2.batched_rollout import BatchedEnvPool
from experiments.overcooked_v2.path_c_backbone_ppo import (
    BACKBONE_CONFIG_SCHEMA_VERSION,
    IPPO_PARTNER_TRAINING_DEPENDENCIES,
    IPPO_PARTNER_FAMILY_DEFINITION,
    IPPO_PARTNER_FAMILY_SPEC_SHA256,
    RecurrentIPPOBackbone,
    ippo_partner_architecture_from_config,
    sample_actor_actions,
)
from experiments.overcooked_v2.path_c_seed import derive_ocv2_execution_seed
from experiments.overcooked_v2.path_c_official_artifact import (
    OFFICIAL_FAMILY_DEFINITIONS,
    OFFICIAL_FAMILY_HASHES,
    OFFICIAL_MODEL_CLASS,
    checkpoint_artifact_sha256,
    validate_official_artifact_manifest,
)
from experiments.overcooked_v2.path_c_standard import (
    PrimitiveRecurrentEnsembleQ,
    STANDARD_CHECKPOINT_SCHEMA_VERSION,
    StandardEnvConfig,
    choose_primitive_actions,
    load_standard_checkpoint,
    model_weights_sha256,
    shared_team_reward,
    single_step_batch,
)
from experiments.overcooked_v2.path_c_standard_diagnostics import (
    decompose_raw_reward_events,
)
from experiments.overcooked_v2.path_c_standard_training import (
    Q_PARTNER_TRAINING_DEPENDENCIES,
    Q_PARTNER_FAMILY_DEFINITION,
    Q_PARTNER_FAMILY_SPEC_SHA256,
    STANDARD_TRAINING_SCHEMA_VERSION,
    canonical_mapping_sha256,
    derive_standard_seed,
    partner_training_run_id,
    q_partner_architecture_from_config,
    repository_source_dependency_closure,
)


ADMISSION_CONFIG_SCHEMA_VERSION = "path_c_pool_admission_v1"
R015_SUPPORT_CONFIG_SCHEMA_VERSION = "path_c_r015_partner_support_v1"
R015_EPISODE_EVIDENCE_SCHEMA_VERSION = (
    "path_c_r015_partner_support_episode_evidence_v1"
)

# This explicit repository-source closure determines support admission.  The
# same path-and-content hashing algorithm is used for both partner trainers and
# this evaluator, so provenance does not depend on a second ad-hoc digest rule.
R015_ADMISSION_IMPLEMENTATION_DEPENDENCIES = (
    "experiments/overcooked_v2/batched_rollout.py",
    "experiments/overcooked_v2/env_adapter.py",
    "experiments/overcooked_v2/path_c_backbone_ppo.py",
    "experiments/overcooked_v2/path_c_pool_admission.py",
    "experiments/overcooked_v2/path_c_official_artifact.py",
    "experiments/overcooked_v2/path_c_official_evidence.py",
    "experiments/overcooked_v2/path_c_flax_policy.py",
    "experiments/overcooked_v2/official/overcooked_v2_experiments_adapter.py",
    "experiments/overcooked_v2/path_c_seed.py",
    "experiments/overcooked_v2/path_c_sequence.py",
    "experiments/overcooked_v2/path_c_standard.py",
    "experiments/overcooked_v2/path_c_standard_diagnostics.py",
    "experiments/overcooked_v2/path_c_standard_training.py",
    "experiments/overcooked_v2/residual_signature.py",
)

R015_REGISTERED_FAMILIES = OFFICIAL_FAMILY_DEFINITIONS
R015_REGISTERED_FAMILY_HASHES = OFFICIAL_FAMILY_HASHES


def _registered_training_implementation_dependencies(
    family: "R015FamilySpec",
) -> dict[str, Any]:
    """Recompute the source closure registered for one partner family."""

    if family.model_class == OFFICIAL_MODEL_CLASS:
        raise ValueError(
            "Official trainer dependencies are recomputed from the wrapped remote "
            "source closure, not from this repository."
        )
    if family.model_class == "RecurrentIPPOBackbone":
        paths = IPPO_PARTNER_TRAINING_DEPENDENCIES
    elif family.model_class == "PrimitiveRecurrentEnsembleQ":
        paths = Q_PARTNER_TRAINING_DEPENDENCIES
    else:
        raise ValueError("R015 family declares an unsupported model class.")
    return repository_source_dependency_closure(paths)


def _action_distribution_jsd(
    left_counts: Sequence[int],
    right_counts: Sequence[int],
) -> float:
    """Return descriptive Jensen-Shannon divergence between two action marginals."""

    left = np.asarray(left_counts, dtype=np.float64)
    right = np.asarray(right_counts, dtype=np.float64)
    if left.shape != (6,) or right.shape != (6,) or left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("Action-distance inputs must be positive six-action counts.")
    left /= left.sum()
    right /= right.sum()
    mixture = 0.5 * (left + right)

    def kl_divergence(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(np.sum(values[positive] * np.log(values[positive] / mixture[positive])))

    return 0.5 * kl_divergence(left) + 0.5 * kl_divergence(right)


@dataclass(frozen=True)
class AdmissionCandidate:
    candidate_id: str
    training_seed: int
    snapshot_environment_steps: int
    path: Path


@dataclass(frozen=True)
class AdmissionSpec:
    evaluation_seed: int
    episodes_per_pairing: int
    evaluation_batch_size: int
    admission_floor: float
    minimum_admitted_members: int
    minimum_distinct_training_seeds: int
    members_per_training_seed: int
    policy_action_selection: str
    checkpoint_paths: tuple[Path, ...]
    output_dir: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AdmissionSpec":
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("Admission config requires an evaluation mapping.")
        episodes = int(evaluation["episodes_per_pairing"])
        batch_size = int(evaluation["evaluation_batch_size"])
        if episodes != 100:
            raise ValueError("Admission requires exactly 100 episodes per pairing.")
        if batch_size <= 0 or episodes % batch_size:
            raise ValueError("Admission batch size must divide 100 episodes.")
        floor = float(evaluation["admission_floor"])
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("admission_floor must be a finite non-negative numeric slot.")
        action_selection = str(evaluation.get("policy_action_selection"))
        if action_selection != "stochastic":
            raise ValueError(
                "Published Overcooked v2 PPO admission requires stochastic actions."
            )
        minimum_members = int(evaluation.get("minimum_admitted_members", 0))
        if minimum_members < 2:
            raise ValueError("Admission requires at least two capable partner policies.")
        minimum_training_seeds = int(
            evaluation.get("minimum_distinct_training_seeds", 2)
        )
        if minimum_training_seeds < 2:
            raise ValueError(
                "Partner diversity requires at least two independent training seeds."
            )
        members_per_training_seed = int(
            evaluation.get("members_per_training_seed", 1)
        )
        if members_per_training_seed <= 0:
            raise ValueError("members_per_training_seed must be positive.")
        paths = tuple(Path(item) for item in payload["checkpoint_paths"])
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Admission requires unique candidate checkpoint paths.")
        return cls(
            evaluation_seed=int(payload["evaluation_seed"]),
            episodes_per_pairing=episodes,
            evaluation_batch_size=batch_size,
            admission_floor=floor,
            minimum_admitted_members=minimum_members,
            minimum_distinct_training_seeds=minimum_training_seeds,
            members_per_training_seed=members_per_training_seed,
            policy_action_selection=action_selection,
            checkpoint_paths=paths,
            output_dir=Path(payload["output_dir"]),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    """Return a lowercase SHA-256 digest or reject the artifact binding."""

    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"R015 {name} must be a lowercase SHA-256 digest.")
    return digest


def _resolved_declared_path(value: Any, *, relative_to: Path) -> Path:
    """Resolve a path declared by a training configuration or manifest."""

    path = Path(str(value))
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


@dataclass(frozen=True)
class R015FamilySpec:
    family_id: str
    training_algorithm: str
    training_objective: str
    reward_objective: str
    convention_generation: str
    observation_contract: str
    action_space: str
    model_class: str
    checkpoint_phase: str
    action_rule: str
    family_spec_sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "R015FamilySpec":
        definition = {str(key): str(value) for key, value in payload.items()}
        family_id = definition.get("family_id", "")
        expected = R015_REGISTERED_FAMILIES.get(family_id)
        if expected is None or definition != expected:
            raise ValueError(
                "R015 family semantics must match one registered family exactly."
            )
        family_spec_sha256 = canonical_mapping_sha256(definition)
        if family_spec_sha256 != R015_REGISTERED_FAMILY_HASHES[family_id]:
            raise ValueError("R015 registered family hash is internally inconsistent.")
        return cls(
            family_id=family_id,
            training_algorithm=definition["training_algorithm"],
            training_objective=definition["training_objective"],
            reward_objective=definition["reward_objective"],
            convention_generation=definition["convention_generation"],
            observation_contract=definition["observation_contract"],
            action_space=definition["action_space"],
            model_class=definition["model_class"],
            checkpoint_phase=definition["checkpoint_phase"],
            action_rule=definition["action_rule"],
            family_spec_sha256=family_spec_sha256,
        )

    def definition(self) -> dict[str, str]:
        return {
            "family_id": self.family_id,
            "training_algorithm": self.training_algorithm,
            "training_objective": self.training_objective,
            "reward_objective": self.reward_objective,
            "convention_generation": self.convention_generation,
            "observation_contract": self.observation_contract,
            "action_space": self.action_space,
            "model_class": self.model_class,
            "checkpoint_phase": self.checkpoint_phase,
            "action_rule": self.action_rule,
        }


@dataclass(frozen=True)
class R015CandidateSpec:
    candidate_id: str
    family_id: str
    training_seed: int
    expected_environment_steps: int
    checkpoint_path: Path
    training_config_path: Path
    training_manifest_path: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "R015CandidateSpec":
        candidate_id = str(payload["candidate_id"])
        family_id = str(payload["family_id"])
        if not candidate_id or not family_id:
            raise ValueError("R015 candidate and family identifiers cannot be empty.")
        training_seed = int(payload["training_seed"])
        environment_steps = int(payload["expected_environment_steps"])
        if training_seed < 0 or environment_steps <= 0:
            raise ValueError("R015 candidate seed and step count are invalid.")
        return cls(
            candidate_id=candidate_id,
            family_id=family_id,
            training_seed=training_seed,
            expected_environment_steps=environment_steps,
            checkpoint_path=Path(payload["checkpoint_path"]),
            training_config_path=Path(payload["training_config_path"]),
            training_manifest_path=Path(payload["training_manifest_path"]),
        )


@dataclass(frozen=True)
class R015PartnerSupportSpec:
    registration_status: str
    scientific_readout_allowed: bool
    support_registration_sha256: str
    admission_implementation_sha256: str
    admission_implementation_dependencies: Mapping[str, Any]
    environment_config_sha256: str
    evaluation_seed: int
    episodes_per_pairing: int
    evaluation_batch_size: int
    admission_floor: float
    families: tuple[R015FamilySpec, ...]
    candidates: tuple[R015CandidateSpec, ...]
    output_dir: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "R015PartnerSupportSpec":
        if payload.get("registration_status") != "configured":
            raise ValueError("R015 static registration must have status configured.")
        if payload.get("scientific_readout_allowed") is not False:
            raise ValueError("R015 support formation is not a scientific readout.")
        raw_registration_sha256 = payload.get("_source_config_sha256")
        if raw_registration_sha256 is None:
            raw_registration_sha256 = canonical_mapping_sha256(
                {
                    str(key): value
                    for key, value in payload.items()
                    if not str(key).startswith("_")
                }
            )
        support_registration_sha256 = _require_sha256(
            raw_registration_sha256,
            name="support-registration content hash",
        )
        current_admission_dependencies = repository_source_dependency_closure(
            R015_ADMISSION_IMPLEMENTATION_DEPENDENCIES
        )
        raw_admission_dependencies = payload.get(
            "_admission_implementation_dependencies",
            current_admission_dependencies,
        )
        if not isinstance(raw_admission_dependencies, Mapping) or dict(
            raw_admission_dependencies
        ) != current_admission_dependencies:
            raise ValueError(
                "R015 admission dependency closure differs from the registered "
                "repository sources."
            )
        expected_admission_sha256 = canonical_mapping_sha256(
            current_admission_dependencies
        )
        admission_implementation_sha256 = _require_sha256(
            payload.get(
                "_admission_implementation_sha256",
                expected_admission_sha256,
            ),
            name="admission-implementation content hash",
        )
        if admission_implementation_sha256 != expected_admission_sha256:
            raise ValueError(
                "R015 admission implementation hash does not match its dependency "
                "closure."
            )
        environment_config_sha256 = _require_sha256(
            payload.get(
                "_environment_config_sha256",
                canonical_mapping_sha256(
                    payload.get("environment")
                    if isinstance(payload.get("environment"), Mapping)
                    else {}
                ),
            ),
            name="support environment-config content hash",
        )
        raw_families = payload.get("families")
        if not isinstance(raw_families, Sequence) or isinstance(
            raw_families, (str, bytes)
        ):
            raise TypeError("R015 support requires a family list.")
        families = tuple(R015FamilySpec.from_mapping(item) for item in raw_families)
        if len(families) != 2 or len({item.family_id for item in families}) != 2:
            raise ValueError("R015 requires exactly two distinct partner families.")
        if {item.family_id for item in families} != set(R015_REGISTERED_FAMILIES):
            raise ValueError(
                "R015 support must contain the registered official self-play and "
                "Other-Play families."
            )
        generation_signatures = {
            (
                item.training_algorithm,
                item.training_objective,
                item.reward_objective,
                item.convention_generation,
                item.observation_contract,
                item.action_space,
            )
            for item in families
        }
        if len(generation_signatures) != 2:
            raise ValueError(
                "The two R015 families must differ in their generation semantics."
            )
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes)
        ):
            raise TypeError("R015 support requires a candidate list.")
        candidates = tuple(
            R015CandidateSpec.from_mapping(item) for item in raw_candidates
        )
        if len(candidates) != 4:
            raise ValueError("R015 support requires exactly four candidate checkpoints.")
        if len({item.candidate_id for item in candidates}) != 4:
            raise ValueError("R015 candidate identifiers must be unique.")
        if len({item.checkpoint_path for item in candidates}) != 4:
            raise ValueError("R015 candidate checkpoints must be unique.")
        if len({item.training_config_path for item in candidates}) != 4:
            raise ValueError(
                "R015 independent runs require four distinct training configs."
            )
        if len({item.training_manifest_path for item in candidates}) != 4:
            raise ValueError(
                "R015 independent runs require four distinct training manifests."
            )
        family_ids = {item.family_id for item in families}
        if any(item.family_id not in family_ids for item in candidates):
            raise ValueError("R015 candidate refers to an unregistered family.")
        for family_id in family_ids:
            family_candidates = [
                item for item in candidates if item.family_id == family_id
            ]
            if len(family_candidates) != 2:
                raise ValueError(
                    "R015 requires exactly two candidates from each family."
                )
            if len({item.training_seed for item in family_candidates}) != 2:
                raise ValueError(
                    "Each R015 family requires two independent training runs."
                )
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("R015 support requires an evaluation mapping.")
        episodes = int(evaluation["episodes_per_pairing"])
        batch_size = int(evaluation["evaluation_batch_size"])
        floor = float(evaluation["admission_floor"])
        if episodes != 100:
            raise ValueError("R015 ability admission uses exactly 100 episodes.")
        if batch_size <= 0 or episodes % batch_size:
            raise ValueError("R015 evaluation batch size must divide 100 episodes.")
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("R015 admission_floor must be finite and non-negative.")
        return cls(
            registration_status="configured",
            scientific_readout_allowed=False,
            support_registration_sha256=support_registration_sha256,
            admission_implementation_sha256=admission_implementation_sha256,
            admission_implementation_dependencies=current_admission_dependencies,
            environment_config_sha256=environment_config_sha256,
            evaluation_seed=int(payload["evaluation_seed"]),
            episodes_per_pairing=episodes,
            evaluation_batch_size=batch_size,
            admission_floor=floor,
            families=families,
            candidates=candidates,
            output_dir=Path(payload["output_dir"]),
        )

    def family(self, family_id: str) -> R015FamilySpec:
        matches = [item for item in self.families if item.family_id == family_id]
        if len(matches) != 1:
            raise ValueError("R015 family lookup is not unique.")
        return matches[0]


def _registered_candidate_training_contract(
    candidate: R015CandidateSpec,
    family: R015FamilySpec,
    training_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate the frozen schedule and return its normalized architecture."""

    training = training_payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("R015 candidate training schedule is malformed.")
    model_config = training_payload.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("R015 candidate model configuration is malformed.")
    observation_shape = tuple(
        int(item) for item in model_config.get("observation_shape", ())
    )
    if observation_shape[:2] != (5, 5):
        raise ValueError(
            "R015 candidate changed the official local 5-by-5 observation contract."
        )
    if family.model_class == "RecurrentIPPOBackbone":
        if training_payload.get("schema_version") != BACKBONE_CONFIG_SCHEMA_VERSION:
            raise ValueError("R015 IPPO candidate uses an unexpected config schema.")
        if (
            int(training.get("total_environment_steps", -1))
            != candidate.expected_environment_steps
            or candidate.expected_environment_steps
            not in tuple(int(item) for item in training.get("snapshot_steps", ()))
            or int(training.get("shaping_horizon_environment_steps", -1))
            != 15_000_000
        ):
            raise ValueError("R015 IPPO candidate changed its frozen budget.")
        return (
            ippo_partner_architecture_from_config(model_config),
            "path_c_backbone_artifact_v2",
        )
    checkpoint_binding_field: str
    if family.model_class == "PrimitiveRecurrentEnsembleQ":
        if training_payload.get("schema_version") != STANDARD_TRAINING_SCHEMA_VERSION:
            raise ValueError("R015 Q candidate uses an unexpected config schema.")
        if training_payload.get("run_kind") != "partner_family" or (
            training_payload.get("training_mode") != "self_play_only"
        ):
            raise ValueError("R015 Q candidate changed its formation mode.")
        if (
            int(training.get("total_environment_steps", -1))
            != candidate.expected_environment_steps
            or int(training.get("self_play_environment_steps", -1))
            != candidate.expected_environment_steps
            or int(training.get("path_c_environment_steps", -1)) != 0
            or candidate.expected_environment_steps
            not in tuple(
                int(item)
                for item in training.get("partner_pool_snapshot_steps", ())
            )
        ):
            raise ValueError("R015 Q candidate changed its frozen budget.")
        return (
            q_partner_architecture_from_config(model_config),
            "path_c_partner_family_training_artifact_v2",
        )
    raise ValueError("R015 family declares an unsupported model class.")


def validate_r015_candidate_artifact_provenance(
    spec: R015PartnerSupportSpec,
    candidate: R015CandidateSpec,
    *,
    reported: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute one candidate's provenance without running an environment.

    The evaluator and the later R015 adjudicator can call this same helper.  It
    reads the registered config, training manifest and checkpoint metadata;
    it does not train, act, or use reported scores as ground truth.
    """

    family = spec.family(candidate.family_id)
    for name, path in (
        ("checkpoint", candidate.checkpoint_path),
        ("training configuration", candidate.training_config_path),
        ("training manifest", candidate.training_manifest_path),
    ):
        present = path.is_file() or (
            name == "checkpoint"
            and family.model_class == OFFICIAL_MODEL_CLASS
            and path.is_dir()
        )
        if not present:
            raise FileNotFoundError(f"R015 {name} is still pending: {path}")
    if family.model_class == OFFICIAL_MODEL_CLASS:
        computed = validate_official_artifact_manifest(
            candidate.training_config_path,
            candidate.training_manifest_path,
            expected_checkpoint_path=candidate.checkpoint_path,
        )
        if (
            computed["family_id"] != candidate.family_id
            or int(computed["training_seed"]) != candidate.training_seed
            or int(computed["snapshot_environment_steps"])
            != candidate.expected_environment_steps
            or computed["family_spec_sha256"] != family.family_spec_sha256
        ):
            raise ValueError(
                "Official candidate artifact differs from its support registration."
            )
        if reported is not None:
            for field, expected in computed.items():
                if field == "layout":
                    continue
                actual = reported.get(field)
                if field.endswith("_path"):
                    if Path(str(actual)).resolve() != Path(str(expected)).resolve():
                        raise ValueError(
                            f"R015 reported candidate changed its {field}."
                        )
                elif actual != expected:
                    raise ValueError(
                        f"R015 reported candidate changed its {field}."
                    )
        return computed
    training_config_bytes = candidate.training_config_path.read_bytes()
    training_config_sha256 = hashlib.sha256(training_config_bytes).hexdigest()
    training_payload = yaml.safe_load(training_config_bytes.decode("utf-8"))
    if not isinstance(training_payload, Mapping):
        raise ValueError("R015 candidate training configuration is malformed.")
    if int(training_payload.get("seed", -1)) != candidate.training_seed:
        raise ValueError("R015 training config seed differs from registration.")
    if dict(training_payload.get("partner_family") or {}) != family.definition():
        raise ValueError(
            "R015 training config does not bind its registered partner family."
        )
    declared_environment = training_payload.get("environment_config")
    if not isinstance(declared_environment, str) or not declared_environment:
        raise ValueError("R015 training config does not identify its environment.")
    training_environment_path = _resolved_declared_path(
        declared_environment,
        relative_to=candidate.training_config_path.parent,
    )
    if not training_environment_path.is_file():
        raise FileNotFoundError(
            "R015 candidate environment configuration is missing: "
            f"{training_environment_path}"
        )
    environment_config_sha256 = _file_sha256(training_environment_path)
    if environment_config_sha256 != spec.environment_config_sha256:
        raise ValueError("R015 candidate binds a different environment config.")
    expected_architecture, expected_manifest_schema = (
        _registered_candidate_training_contract(
            candidate,
            family,
            training_payload,
        )
    )
    training_implementation_dependencies = (
        _registered_training_implementation_dependencies(family)
    )
    training_implementation_sha256 = canonical_mapping_sha256(
        training_implementation_dependencies
    )
    training_run_id = partner_training_run_id(
        family_spec_sha256=family.family_spec_sha256,
        training_seed=candidate.training_seed,
        training_config_sha256=training_config_sha256,
        environment_config_sha256=environment_config_sha256,
        training_implementation_sha256=training_implementation_sha256,
    )
    expected_binding = {
        "partner_family_spec_sha256": family.family_spec_sha256,
        "training_config_sha256": training_config_sha256,
        "environment_config_sha256": environment_config_sha256,
        "training_implementation_sha256": training_implementation_sha256,
        "training_run_id": training_run_id,
    }

    def verify_bound_payload(payload: Mapping[str, Any], *, source: str) -> None:
        if dict(payload.get("partner_family") or {}) != family.definition():
            raise ValueError(
                f"R015 {source} does not bind the registered partner family."
            )
        for field, expected in expected_binding.items():
            actual = _require_sha256(
                payload.get(field),
                name=f"{source} {field}",
            )
            if actual != expected:
                raise ValueError(
                    f"R015 {source} {field} differs from the registered run."
                )
        raw_dependencies = payload.get("training_implementation_dependencies")
        if not isinstance(raw_dependencies, Mapping) or dict(
            raw_dependencies
        ) != training_implementation_dependencies:
            raise ValueError(
                f"R015 {source} changed the training dependency closure."
            )

    training_manifest_bytes = candidate.training_manifest_path.read_bytes()
    training_manifest_sha256 = hashlib.sha256(training_manifest_bytes).hexdigest()
    manifest = json.loads(training_manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("R015 candidate training manifest is malformed.")
    if manifest.get("schema_version") != expected_manifest_schema:
        raise ValueError("R015 candidate training manifest schema changed.")
    if manifest.get("run_status") != "completed" or int(
        manifest.get("effective_environment_steps", -1)
    ) != candidate.expected_environment_steps:
        raise ValueError("R015 candidate training did not complete its fixed budget.")
    manifest_architecture = manifest.get("architecture")
    if not isinstance(manifest_architecture, Mapping) or dict(
        manifest_architecture
    ) != expected_architecture:
        raise ValueError(
            "R015 training manifest architecture differs from its model config."
        )
    verify_bound_payload(manifest, source="training manifest")
    if family.model_class == "PrimitiveRecurrentEnsembleQ":
        if int(manifest.get("seed", -1)) != candidate.training_seed:
            raise ValueError("R015 Q manifest seed differs from registration.")
        declared_checkpoint = manifest.get("final_candidate_checkpoint")
        if declared_checkpoint is None or _resolved_declared_path(
            declared_checkpoint,
            relative_to=candidate.training_manifest_path.parent,
        ) != candidate.checkpoint_path.resolve():
            raise ValueError("R015 Q manifest identifies another checkpoint.")
        checkpoint_binding_field = "candidate_checkpoint_bindings"
    else:
        declared_snapshots = manifest.get("snapshots")
        if not isinstance(declared_snapshots, Sequence) or isinstance(
            declared_snapshots, (str, bytes)
        ):
            raise ValueError("R015 IPPO manifest has no snapshot list.")
        resolved_snapshots = {
            _resolved_declared_path(
                item,
                relative_to=candidate.training_manifest_path.parent,
            )
            for item in declared_snapshots
        }
        if candidate.checkpoint_path.resolve() not in resolved_snapshots:
            raise ValueError("R015 IPPO manifest identifies another checkpoint.")
        checkpoint_binding_field = "snapshot_bindings"

    raw_checkpoint_bindings = manifest.get(checkpoint_binding_field)
    if not isinstance(raw_checkpoint_bindings, Sequence) or isinstance(
        raw_checkpoint_bindings, (str, bytes)
    ):
        raise ValueError("R015 training manifest lacks checkpoint weight bindings.")
    matching_checkpoint_bindings = []
    for raw_binding in raw_checkpoint_bindings:
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
            "path",
            "environment_steps",
            "checkpoint_sha256",
            "model_weights_sha256",
        }:
            raise ValueError("R015 checkpoint binding has the wrong schema.")
        if _resolved_declared_path(
            raw_binding["path"],
            relative_to=candidate.training_manifest_path.parent,
        ) == candidate.checkpoint_path.resolve():
            matching_checkpoint_bindings.append(raw_binding)
    if len(matching_checkpoint_bindings) != 1:
        raise ValueError("R015 manifest must bind the selected checkpoint exactly once.")
    manifest_checkpoint_binding = matching_checkpoint_bindings[0]
    if int(manifest_checkpoint_binding["environment_steps"]) != (
        candidate.expected_environment_steps
    ):
        raise ValueError("R015 manifest checkpoint binding changed its step count.")

    checkpoint_sha256 = _file_sha256(candidate.checkpoint_path)
    checkpoint_payload = torch.load(
        candidate.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint_payload, Mapping) or not isinstance(
        checkpoint_payload.get("model_state_dict"), Mapping
    ):
        raise ValueError("R015 checkpoint is missing model weights.")
    checkpoint_weights_sha256 = model_weights_sha256(
        checkpoint_payload["model_state_dict"]
    )
    metadata = checkpoint_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("R015 checkpoint metadata is malformed.")
    if metadata.get("schema_version") != STANDARD_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("R015 checkpoint schema changed.")
    if metadata.get("model_weights_sha256") != checkpoint_weights_sha256:
        raise ValueError("R015 checkpoint model-weight hash differs from its tensors.")
    if manifest_checkpoint_binding.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("R015 manifest checkpoint file hash differs from the file.")
    if manifest_checkpoint_binding.get(
        "model_weights_sha256"
    ) != checkpoint_weights_sha256:
        raise ValueError("R015 manifest model-weight hash differs from the checkpoint.")
    verify_bound_payload(metadata, source="checkpoint metadata")
    metadata_architecture = metadata.get("architecture")
    if not isinstance(metadata_architecture, Mapping) or dict(
        metadata_architecture
    ) != expected_architecture:
        raise ValueError(
            "R015 checkpoint architecture differs from its model config."
        )
    if metadata.get("partner_family_id") != family.family_id:
        raise ValueError("R015 checkpoint family identifier changed.")
    if metadata.get("phase") != family.checkpoint_phase:
        raise ValueError("R015 checkpoint phase differs from its family contract.")
    if int(metadata.get("seed", -1)) != candidate.training_seed or int(
        metadata.get("environment_steps", -1)
    ) != candidate.expected_environment_steps:
        raise ValueError("R015 checkpoint seed or step count changed.")
    layout = str(metadata.get("layout", ""))
    if not layout:
        raise ValueError("R015 checkpoint layout is missing.")
    loaded_model, loaded_metadata = load_standard_checkpoint(
        candidate.checkpoint_path,
        device="cpu",
    )
    if dict(loaded_metadata) != dict(metadata):
        raise ValueError("R015 checkpoint metadata changed while loading the model.")
    if family.model_class == "RecurrentIPPOBackbone":
        if not isinstance(loaded_model, RecurrentIPPOBackbone):
            raise ValueError("R015 IPPO checkpoint contains another model class.")
    elif not isinstance(loaded_model, PrimitiveRecurrentEnsembleQ):
        raise ValueError("R015 Q checkpoint contains another model class.")
    if loaded_model.architecture_manifest() != expected_architecture:
        raise ValueError(
            "R015 loaded model architecture differs from config and checkpoint."
        )

    computed = {
        "artifact_verified": True,
        "family_id": candidate.family_id,
        "family_spec_sha256": family.family_spec_sha256,
        "training_seed": candidate.training_seed,
        "snapshot_environment_steps": candidate.expected_environment_steps,
        "checkpoint_path": str(candidate.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "model_weights_sha256": checkpoint_weights_sha256,
        "training_config_path": str(candidate.training_config_path),
        "training_config_sha256": training_config_sha256,
        "training_manifest_path": str(candidate.training_manifest_path),
        "training_manifest_sha256": training_manifest_sha256,
        "environment_config_sha256": environment_config_sha256,
        "training_implementation_sha256": training_implementation_sha256,
        "training_implementation_dependencies": (
            training_implementation_dependencies
        ),
        "training_run_id": training_run_id,
        "architecture": expected_architecture,
        "architecture_sha256": canonical_mapping_sha256(expected_architecture),
        "layout": layout,
    }
    if reported is not None:
        for field, expected in computed.items():
            if field == "layout":
                continue
            actual = reported.get(field)
            if field.endswith("_path"):
                if Path(str(actual)).resolve() != Path(str(expected)).resolve():
                    raise ValueError(
                        f"R015 reported candidate changed its {field}."
                    )
            elif actual != expected:
                raise ValueError(
                    f"R015 reported candidate changed its {field}."
                )
    return computed


def configured_r015_support_status(
    spec: R015PartnerSupportSpec,
) -> dict[str, Any]:
    """Describe a valid registration without promoting absent artifacts."""

    candidates: dict[str, dict[str, Any]] = {}
    missing_checkpoints: list[str] = []
    missing_training_configs: list[str] = []
    missing_training_manifests: list[str] = []
    for candidate in spec.candidates:
        family = spec.family(candidate.family_id)
        checkpoint_present = candidate.checkpoint_path.is_file() or (
            family.model_class == OFFICIAL_MODEL_CLASS
            and candidate.checkpoint_path.is_dir()
        )
        config_present = candidate.training_config_path.is_file()
        manifest_present = candidate.training_manifest_path.is_file()
        if not checkpoint_present:
            missing_checkpoints.append(candidate.candidate_id)
        if not config_present:
            missing_training_configs.append(candidate.candidate_id)
        if not manifest_present:
            missing_training_manifests.append(candidate.candidate_id)
        candidates[candidate.candidate_id] = {
            "artifact_verified": False,
            "family_id": candidate.family_id,
            "training_seed": candidate.training_seed,
            "expected_environment_steps": candidate.expected_environment_steps,
            "checkpoint_path": str(candidate.checkpoint_path),
            "checkpoint_status": "present" if checkpoint_present else "pending",
            "checkpoint_sha256": (
                checkpoint_artifact_sha256(candidate.checkpoint_path)
                if checkpoint_present
                else None
            ),
            "training_config_path": str(candidate.training_config_path),
            "training_config_status": "present" if config_present else "pending",
            "training_config_sha256": (
                _file_sha256(candidate.training_config_path)
                if config_present
                else None
            ),
            "training_manifest_path": str(candidate.training_manifest_path),
            "training_manifest_status": (
                "present" if manifest_present else "pending"
            ),
            "training_manifest_sha256": (
                _file_sha256(candidate.training_manifest_path)
                if manifest_present
                else None
            ),
        }
    return {
        "schema_version": "path_c_r015_partner_support_status_v1",
        "registration_status": "configured",
        "support_status": "pending",
        "artifact_verification_status": "pending",
        "scientific_readout_allowed": False,
        "support_registration_sha256": spec.support_registration_sha256,
        "admission_implementation_sha256": (
            spec.admission_implementation_sha256
        ),
        "admission_implementation_dependencies": dict(
            spec.admission_implementation_dependencies
        ),
        "evaluation_seed": spec.evaluation_seed,
        "environment_config_sha256": spec.environment_config_sha256,
        "families": {
            family.family_id: {
                **family.definition(),
                "family_spec_sha256": family.family_spec_sha256,
            }
            for family in spec.families
        },
        "candidates": candidates,
        "missing_checkpoint_candidates": sorted(missing_checkpoints),
        "missing_training_config_candidates": sorted(missing_training_configs),
        "missing_training_manifest_candidates": sorted(
            missing_training_manifests
        ),
        "support_complete": False,
        "members": {},
    }


def select_r015_support_members(
    spec: R015PartnerSupportSpec,
    candidate_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select all four fixed candidates only when every ability check passes."""

    expected_ids = {item.candidate_id for item in spec.candidates}
    if set(candidate_results) != expected_ids:
        raise ValueError("R015 candidate results do not match the fixed support.")
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in spec.candidates
    }
    checkpoint_hashes: list[str] = []
    model_weight_hashes: list[str] = []
    training_config_hashes: list[str] = []
    training_manifest_hashes: list[str] = []
    training_run_ids: list[str] = []
    environment_hashes: list[str] = []
    implementation_hashes_by_family: dict[str, set[str]] = {
        family.family_id: set() for family in spec.families
    }
    for candidate_id, result in candidate_results.items():
        candidate = candidates_by_id[candidate_id]
        if result.get("family_id") != candidate.family_id or int(
            result.get("training_seed", -1)
        ) != candidate.training_seed:
            raise ValueError(
                "R015 candidate result changed its family or training run."
            )
        if int(result.get("snapshot_environment_steps", -1)) != (
            candidate.expected_environment_steps
        ):
            raise ValueError("R015 candidate result changed its registered snapshot.")
        for field, expected_path in (
            ("checkpoint_path", candidate.checkpoint_path),
            ("training_config_path", candidate.training_config_path),
            ("training_manifest_path", candidate.training_manifest_path),
        ):
            if Path(str(result.get(field, ""))).resolve() != expected_path.resolve():
                raise ValueError(f"R015 candidate result changed its {field}.")
        if result.get("artifact_verified") is not True:
            raise ValueError(
                "R015 cannot select a candidate whose artifacts were not verified."
            )
        family = spec.family(candidate.family_id)
        if result.get("family_spec_sha256") != family.family_spec_sha256:
            raise ValueError("R015 candidate family hash differs from registration.")
        architecture = result.get("architecture")
        if not isinstance(architecture, Mapping) or architecture.get(
            "model_class"
        ) != family.model_class:
            raise ValueError("R015 candidate architecture is missing or changed.")
        architecture_sha256 = _require_sha256(
            result.get("architecture_sha256"),
            name="candidate architecture hash",
        )
        if architecture_sha256 != canonical_mapping_sha256(architecture):
            raise ValueError(
                "R015 candidate architecture hash does not match its content."
            )
        checkpoint_hashes.append(
            _require_sha256(
                result.get("checkpoint_sha256"),
                name="checkpoint content hash",
            )
        )
        model_weight_hashes.append(
            _require_sha256(
                result.get("model_weights_sha256"),
                name="model-weight content hash",
            )
        )
        training_config_hashes.append(
            _require_sha256(
                result.get("training_config_sha256"),
                name="training-config content hash",
            )
        )
        training_manifest_hashes.append(
            _require_sha256(
                result.get("training_manifest_sha256"),
                name="training-manifest content hash",
            )
        )
        environment_config_sha256 = _require_sha256(
            result.get("environment_config_sha256"),
            name="environment-config content hash",
        )
        environment_hashes.append(environment_config_sha256)
        if environment_config_sha256 != spec.environment_config_sha256:
            raise ValueError(
                "R015 candidate environment hash differs from support registration."
            )
        raw_implementation_dependencies = result.get(
            "training_implementation_dependencies"
        )
        if not isinstance(raw_implementation_dependencies, Mapping) or not isinstance(
            raw_implementation_dependencies.get("files"), Mapping
        ) or not raw_implementation_dependencies.get("files"):
            raise ValueError(
                "R015 candidate training dependency closure is missing or malformed."
            )
        implementation_dependencies = dict(raw_implementation_dependencies)
        training_implementation_sha256 = _require_sha256(
            result.get("training_implementation_sha256"),
            name="training-implementation content hash",
        )
        if training_implementation_sha256 != canonical_mapping_sha256(
            implementation_dependencies
        ):
            raise ValueError(
                "R015 candidate implementation hash does not match its dependency "
                "closure."
            )
        implementation_hashes_by_family[candidate.family_id].add(
            training_implementation_sha256
        )
        training_run_id = _require_sha256(
            result.get("training_run_id"),
            name="training run identifier",
        )
        expected_training_run_id = partner_training_run_id(
            family_spec_sha256=family.family_spec_sha256,
            training_seed=candidate.training_seed,
            training_config_sha256=training_config_hashes[-1],
            environment_config_sha256=environment_config_sha256,
            training_implementation_sha256=training_implementation_sha256,
        )
        if training_run_id != expected_training_run_id:
            raise ValueError(
                "R015 candidate training run identifier does not match provenance."
            )
        training_run_ids.append(training_run_id)
    for name, values in (
        ("checkpoint contents", checkpoint_hashes),
        ("model weight states", model_weight_hashes),
        ("training configurations", training_config_hashes),
        ("training manifests", training_manifest_hashes),
        ("training run identifiers", training_run_ids),
    ):
        if len(set(values)) != 4:
            raise ValueError(f"R015 requires four distinct {name}.")
    if len(set(environment_hashes)) != 1:
        raise ValueError(
            "R015 requires one shared environment configuration across both families."
        )
    if any(len(values) != 1 for values in implementation_hashes_by_family.values()):
        raise ValueError(
            "Each R015 family must bind one training implementation closure."
        )
    if len(
        {next(iter(values)) for values in implementation_hashes_by_family.values()}
    ) != 2:
        raise ValueError(
            "The two R015 families must bind different training implementation "
            "closures."
        )
    qualified = {
        candidate_id: dict(result)
        for candidate_id, result in candidate_results.items()
        if result.get("admitted") is True
    }
    complete = len(qualified) == 4
    members = qualified if complete else {}
    members_by_family = {
        family.family_id: sorted(
            candidate_id
            for candidate_id, result in members.items()
            if result.get("family_id") == family.family_id
        )
        for family in spec.families
    }
    if complete and any(len(values) != 2 for values in members_by_family.values()):
        raise ValueError("A complete R015 support must contain two members per family.")
    return {
        "support_complete": complete,
        "support_status": "admitted" if complete else "not_admitted",
        "qualified_candidate_ids": sorted(qualified),
        "members_by_family": members_by_family,
        "members": members,
    }


def validate_r015_support_report_evidence(
    spec: R015PartnerSupportSpec,
    report: Mapping[str, Any],
    *,
    candidate_provenance_validator: Any = validate_r015_candidate_artifact_provenance,
) -> Mapping[str, Mapping[str, Any]]:
    """Recompute support admission from bound episode-level evidence.

    This validator performs no environment rollout.  It verifies the static
    registration, the admission source closure, all candidate provenance, and
    every recorded episode before reconstructing ability admission.
    """

    if report.get("schema_version") != "path_c_r015_partner_support_report_v2":
        raise ValueError("Unsupported R015 partner-support report.")
    if (
        report.get("registration_status") != "configured"
        or report.get("scientific_readout_allowed") is not False
        or report.get("artifact_verification_status") != "verified"
    ):
        raise ValueError("R015 support report changed its registration role.")
    for field, expected in (
        ("support_registration_sha256", spec.support_registration_sha256),
        (
            "admission_implementation_sha256",
            spec.admission_implementation_sha256,
        ),
        ("environment_config_sha256", spec.environment_config_sha256),
    ):
        actual = _require_sha256(report.get(field), name=f"report {field}")
        if actual != expected:
            raise ValueError(f"R015 support report changed its {field}.")
    raw_admission_dependencies = report.get(
        "admission_implementation_dependencies"
    )
    if not isinstance(raw_admission_dependencies, Mapping) or dict(
        raw_admission_dependencies
    ) != dict(spec.admission_implementation_dependencies):
        raise ValueError("R015 report changed the admission dependency closure.")
    if int(report.get("evaluation_seed", -1)) != spec.evaluation_seed:
        raise ValueError("R015 support report changed its evaluation seed.")
    if int(report.get("episodes_per_pairing", -1)) != spec.episodes_per_pairing:
        raise ValueError("R015 support report changed episodes per pairing.")
    if not math.isclose(
        float(report.get("admission_floor", float("nan"))),
        spec.admission_floor,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R015 support report changed its admission floor.")
    layout = str(report.get("layout", ""))
    if layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError("R015 support report layout is invalid.")
    raw_families = report.get("families")
    if not isinstance(raw_families, Mapping) or set(raw_families) != {
        family.family_id for family in spec.families
    }:
        raise ValueError("R015 support report changed its family set.")
    for family in spec.families:
        raw_family = raw_families[family.family_id]
        expected_family = {
            **family.definition(),
            "family_spec_sha256": family.family_spec_sha256,
        }
        if not isinstance(raw_family, Mapping) or dict(raw_family) != expected_family:
            raise ValueError("R015 support report changed family semantics.")

    expected_candidates = {
        candidate.candidate_id: candidate for candidate in spec.candidates
    }
    raw_candidates = report.get("candidates")
    if not isinstance(raw_candidates, Mapping) or set(raw_candidates) != set(
        expected_candidates
    ):
        raise ValueError("R015 report candidate set differs from registration.")
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate in expected_candidates.items():
        raw_candidate = raw_candidates[candidate_id]
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("R015 reported candidate is malformed.")
        computed = candidate_provenance_validator(
            spec,
            candidate,
            reported=raw_candidate,
        )
        if raw_candidate.get("action_rule") != spec.family(
            candidate.family_id
        ).action_rule:
            raise ValueError("R015 reported candidate changed its action rule.")
        if computed["layout"] != layout:
            raise ValueError("R015 candidate checkpoint layout differs from report.")
        candidates[candidate_id] = dict(raw_candidate)

    evidence = report.get("episode_evidence")
    if not isinstance(evidence, Mapping) or evidence.get(
        "schema_version"
    ) != R015_EPISODE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("R015 report lacks bound episode-level evidence.")
    evidence_path = Path(str(evidence.get("path", "")))
    expected_evidence_path = (spec.output_dir / "episode_returns.jsonl").resolve()
    if evidence_path.resolve() != expected_evidence_path:
        raise ValueError("R015 report changed the episode-evidence path.")
    if not evidence_path.is_file():
        raise FileNotFoundError("R015 episode-level evidence is missing.")
    evidence_sha256 = _require_sha256(
        evidence.get("sha256"),
        name="episode-evidence content hash",
    )
    if _file_sha256(evidence_path) != evidence_sha256:
        raise ValueError("R015 episode-level evidence content changed.")
    lines = evidence_path.read_text(encoding="utf-8").splitlines()
    expected_row_count = (
        len(expected_candidates) ** 2 * spec.episodes_per_pairing
    )
    if int(evidence.get("row_count", -1)) != expected_row_count or len(
        lines
    ) != expected_row_count:
        raise ValueError("R015 episode-evidence row count is incomplete.")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in lines:
        raw_row = json.loads(line)
        if not isinstance(raw_row, Mapping):
            raise ValueError("R015 episode-evidence row is malformed.")
        row = dict(raw_row)
        if row.get("schema_version") != (
            "path_c_r015_partner_support_episode_v1"
        ) or row.get("scientific_readout_allowed") is not False:
            raise ValueError("R015 episode-evidence row changed its schema or role.")
        left_id = str(row.get("policy_0_candidate_id", ""))
        right_id = str(row.get("policy_1_candidate_id", ""))
        if left_id not in expected_candidates or right_id not in expected_candidates:
            raise ValueError("R015 episode row names an unregistered candidate.")
        left = expected_candidates[left_id]
        right = expected_candidates[right_id]
        if row.get("policy_0_family_id") != left.family_id or row.get(
            "policy_1_family_id"
        ) != right.family_id:
            raise ValueError("R015 episode row changed a candidate family.")
        episode_index = int(row.get("episode_index", -1))
        if not 0 <= episode_index < spec.episodes_per_pairing:
            raise ValueError("R015 episode index lies outside the fixed pairing.")
        identity = (left_id, right_id, episode_index)
        if identity in seen:
            raise ValueError("R015 episode evidence contains a duplicate row.")
        seen.add(identity)
        expected_seed = derive_standard_seed(
            spec.evaluation_seed,
            "r015_partner_support",
            layout,
            left_id,
            right_id,
            episode_index,
        )
        if int(row.get("canonical_episode_seed", -1)) != expected_seed:
            raise ValueError("R015 episode evidence changed its canonical seed.")
        if row.get("layout") != layout or int(
            row.get("environment_steps", -1)
        ) != 400:
            raise ValueError("R015 episode evidence changed its environment contract.")
        raw_return = float(row.get("raw_episode_return", float("nan")))
        if not math.isfinite(raw_return):
            raise ValueError("R015 episode return must be finite.")
        for field in (
            "correct_delivery_count",
            "wrong_delivery_count",
            "indicator_activation_count",
            "ambiguous_reward_step_count",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"R015 episode {field} must be non-negative.")
        for field in ("policy_0_action_counts", "policy_1_action_counts"):
            values = row.get(field)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("R015 action counts are malformed.")
            counts = list(values)
            if len(counts) != 6 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts
            ) or sum(counts) != 400:
                raise ValueError(
                    "R015 action counts must cover 400 six-action decisions."
                )
        rows.append(row)
        grouped.setdefault((left_id, right_id), []).append(row)
    expected_pairings = {
        (left_id, right_id)
        for left_id in expected_candidates
        for right_id in expected_candidates
    }
    if set(grouped) != expected_pairings or any(
        len(pairing_rows) != spec.episodes_per_pairing
        or {int(row["episode_index"]) for row in pairing_rows}
        != set(range(spec.episodes_per_pairing))
        for pairing_rows in grouped.values()
    ):
        raise ValueError("R015 episode evidence does not cover every fixed pairing.")

    def require_reported_float(
        actual: Any,
        expected: float,
        *,
        name: str,
    ) -> None:
        try:
            value = float(actual)
        except (TypeError, ValueError) as error:
            raise ValueError(f"R015 {name} is not numeric.") from error
        if not math.isfinite(value) or not math.isclose(
            value,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"R015 {name} differs from episode evidence.")

    for candidate_id, candidate_result in candidates.items():
        self_play_rows = grouped[(candidate_id, candidate_id)]
        correct_delivery_rate = float(
            np.mean([row["correct_delivery_count"] for row in self_play_rows])
        )
        mean_return = float(
            np.mean([row["raw_episode_return"] for row in self_play_rows])
        )
        require_reported_float(
            candidate_result.get("correct_delivery_rate"),
            correct_delivery_rate,
            name=f"candidate {candidate_id} correct-delivery rate",
        )
        require_reported_float(
            candidate_result.get("mean_raw_episode_return"),
            mean_return,
            name=f"candidate {candidate_id} mean return",
        )
        admitted = correct_delivery_rate >= spec.admission_floor
        if candidate_result.get("admitted") is not admitted:
            raise ValueError("R015 candidate admission differs from episode evidence.")

    selection = select_r015_support_members(spec, candidates)
    if report.get("support_complete") is not selection["support_complete"] or (
        report.get("support_status") != selection["support_status"]
    ):
        raise ValueError("R015 support verdict differs from recomputed admission.")
    if set(report.get("qualified_candidate_ids", ())) != set(
        selection["qualified_candidate_ids"]
    ):
        raise ValueError("R015 qualified candidate list differs from evidence.")
    if report.get("members_by_family") != selection["members_by_family"] or report.get(
        "members"
    ) != selection["members"]:
        raise ValueError("R015 support membership differs from recomputed admission.")

    pairing_matrix = report.get("pairing_matrix")
    if not isinstance(pairing_matrix, Sequence) or isinstance(
        pairing_matrix, (str, bytes)
    ) or len(pairing_matrix) != len(expected_pairings):
        raise ValueError("R015 pairing matrix is incomplete.")
    reported_pairings: set[tuple[str, str]] = set()
    for raw_pairing in pairing_matrix:
        if not isinstance(raw_pairing, Mapping):
            raise ValueError("R015 pairing matrix row is malformed.")
        left_id = str(raw_pairing.get("policy_0_candidate_id", ""))
        right_id = str(raw_pairing.get("policy_1_candidate_id", ""))
        pairing = (left_id, right_id)
        if pairing not in expected_pairings or pairing in reported_pairings:
            raise ValueError("R015 pairing matrix identity is invalid or duplicated.")
        reported_pairings.add(pairing)
        pairing_rows = grouped[pairing]
        require_reported_float(
            raw_pairing.get("mean_raw_episode_return"),
            float(np.mean([row["raw_episode_return"] for row in pairing_rows])),
            name=f"pairing {left_id}/{right_id} mean return",
        )
        require_reported_float(
            raw_pairing.get("correct_delivery_rate"),
            float(np.mean([row["correct_delivery_count"] for row in pairing_rows])),
            name=f"pairing {left_id}/{right_id} correct-delivery rate",
        )
        expected_jsd = _action_distribution_jsd(
            np.sum(
                [row["policy_0_action_counts"] for row in pairing_rows],
                axis=0,
            ),
            np.sum(
                [row["policy_1_action_counts"] for row in pairing_rows],
                axis=0,
            ),
        )
        require_reported_float(
            raw_pairing.get("action_distribution_jsd"),
            expected_jsd,
            name=f"pairing {left_id}/{right_id} action distance",
        )
    return selection["members"]


class PoolAdmissionEvaluator:
    """Evaluate stochastic self-play and every directed cross-play pairing."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = AdmissionSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.device = torch.device(str(config.get("device", "cuda")))
        self.models: dict[str, RecurrentIPPOBackbone] = {}
        self.candidates: list[AdmissionCandidate] = []
        for path in self.spec.checkpoint_paths:
            model, metadata = load_standard_checkpoint(path, device=self.device)
            if not isinstance(model, RecurrentIPPOBackbone):
                raise ValueError("Admission accepts backbone checkpoints only.")
            if metadata.get("phase") != "backbone_snapshot":
                raise ValueError("Admission candidates must be backbone snapshots.")
            if metadata.get("layout") != self.env_config.layout:
                raise ValueError("Admission checkpoint layout mismatch.")
            training_seed = int(metadata["seed"])
            snapshot_environment_steps = int(metadata["environment_steps"])
            candidate_id = (
                f"seed_{training_seed}_step_{snapshot_environment_steps}"
            )
            if candidate_id in self.models:
                raise ValueError("Admission candidate identities must be unique.")
            self.models[candidate_id] = model
            self.candidates.append(
                AdmissionCandidate(
                    candidate_id=candidate_id,
                    training_seed=training_seed,
                    snapshot_environment_steps=snapshot_environment_steps,
                    path=path,
                )
            )
        self.candidates.sort(
            key=lambda item: (item.training_seed, item.snapshot_environment_steps)
        )
        self.adapter = self.env_config.make_adapter()
        self.pools: dict[int, BatchedEnvPool] = {}

    def _pool(self, batch_size: int) -> BatchedEnvPool:
        if batch_size not in self.pools:
            self.pools[batch_size] = BatchedEnvPool(self.adapter.env, batch_size)
        return self.pools[batch_size]

    def _evaluate_chunk(
        self,
        left_seed: str,
        right_seed: str,
        episode_indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        batch_size = len(episode_indices)
        pool = self._pool(batch_size)
        canonical_seeds = [
            derive_standard_seed(
                self.spec.evaluation_seed,
                "admission",
                self.env_config.layout,
                left_seed,
                right_seed,
                index,
            )
            for index in episode_indices
        ]
        pool.reset(
            np.asarray(
                [derive_ocv2_execution_seed(seed) for seed in canonical_seeds],
                dtype=np.uint32,
            )
        )
        observations = pool.snapshot_obs()
        models = (self.models[left_seed], self.models[right_seed])
        states = {
            slot: models[slot].initial_state(batch_size, device=self.device)
            for slot in (0, 1)
        }
        previous_actions = {
            slot: np.full(batch_size, 6, dtype=np.int64) for slot in (0, 1)
        }
        previous_rewards = {
            slot: np.zeros(batch_size, dtype=np.float32) for slot in (0, 1)
        }
        episode_starts = np.ones(batch_size, dtype=bool)
        totals = np.zeros(batch_size, dtype=np.float64)
        action_rngs = {
            slot: np.random.default_rng(
                derive_ocv2_execution_seed(
                    derive_standard_seed(
                        self.spec.evaluation_seed,
                        "admission_action",
                        left_seed,
                        right_seed,
                        int(episode_indices[0]),
                        slot,
                    )
                )
            )
            for slot in (0, 1)
        }
        correct = np.zeros(batch_size, dtype=np.int64)
        wrong = np.zeros(batch_size, dtype=np.int64)
        indicator = np.zeros(batch_size, dtype=np.int64)
        ambiguous = np.zeros(batch_size, dtype=np.int64)
        action_counts = {
            slot: np.zeros((batch_size, 6), dtype=np.int64) for slot in (0, 1)
        }
        for _ in range(self.env_config.max_steps):
            actions: dict[int, np.ndarray] = {}
            for slot in (0, 1):
                batch = single_step_batch(
                    observations[f"agent_{slot}"],
                    previous_actions[slot],
                    previous_rewards[slot],
                    episode_starts,
                    n_actions=6,
                    device=self.device,
                )
                with torch.no_grad():
                    logits, _, _, states[slot] = models[slot].forward_step(
                        batch, states[slot]
                    )
                actions[slot] = sample_actor_actions(
                    logits,
                    action_rngs[slot].random(batch_size),
                )
                action_counts[slot][np.arange(batch_size), actions[slot]] += 1
            observations, _, rewards, dones, _ = pool.step_joint(actions[0], actions[1])
            reward = shared_team_reward(rewards).astype(np.float64)
            totals += reward
            for row, value in enumerate(reward):
                events = decompose_raw_reward_events(np.asarray([value]))
                correct[row] += events.correct_delivery_count
                wrong[row] += events.wrong_delivery_count
                indicator[row] += events.indicator_activation_count
                ambiguous[row] += events.ambiguous_step_count
            episode_starts = np.asarray(dones["__all__"], dtype=bool)
            for slot in (0, 1):
                previous_actions[slot] = actions[slot]
                previous_rewards[slot] = reward.astype(np.float32)
            observations = {key: np.asarray(value) for key, value in observations.items()}
        if not bool(episode_starts.all()):
            raise ValueError("Admission episode did not reach the fixed boundary.")
        return [
            {
                "schema_version": "path_c_pool_admission_episode_v2",
                "run_kind": "admission",
                "scientific_readout_allowed": False,
                "layout": self.env_config.layout,
                "policy_0_candidate_id": left_seed,
                "policy_1_candidate_id": right_seed,
                "episode_index": int(episode_index),
                "canonical_episode_seed": int(canonical_seed),
                "environment_steps": self.env_config.max_steps,
                "raw_episode_return": float(totals[row]),
                "correct_delivery_count": int(correct[row]),
                "wrong_delivery_count": int(wrong[row]),
                "indicator_activation_count": int(indicator[row]),
                "ambiguous_reward_step_count": int(ambiguous[row]),
                "policy_0_action_counts": action_counts[0][row].tolist(),
                "policy_1_action_counts": action_counts[1][row].tolist(),
            }
            for row, (episode_index, canonical_seed) in enumerate(
                zip(episode_indices, canonical_seeds, strict=True)
            )
        ]

    def run(self) -> dict[str, Any]:
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        seeds = [candidate.candidate_id for candidate in self.candidates]
        pairings = [(seed, seed) for seed in seeds] + [
            pair for pair in ((a, b) for a in seeds for b in seeds) if pair[0] != pair[1]
        ]
        for left_seed, right_seed in pairings:
            for offset in range(0, self.spec.episodes_per_pairing, self.spec.evaluation_batch_size):
                rows.extend(
                    self._evaluate_chunk(
                        left_seed,
                        right_seed,
                        range(offset, offset + self.spec.evaluation_batch_size),
                    )
                )
        rows_path = self.spec.output_dir / "episode_returns.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        candidates = {}
        for candidate in self.candidates:
            self_play_rows = [
                row for row in rows
                if row["policy_0_candidate_id"] == candidate.candidate_id
                and row["policy_1_candidate_id"] == candidate.candidate_id
            ]
            correct_rate = float(
                np.mean([row["correct_delivery_count"] for row in self_play_rows])
            )
            candidates[candidate.candidate_id] = {
                "checkpoint_path": str(candidate.path),
                "training_seed": candidate.training_seed,
                "snapshot_environment_steps": candidate.snapshot_environment_steps,
                "correct_delivery_rate": correct_rate,
                "mean_raw_episode_return": float(
                    np.mean([row["raw_episode_return"] for row in self_play_rows])
                ),
                "admitted": correct_rate >= self.spec.admission_floor,
            }
        admitted_by_training_seed: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        for candidate_id, candidate in candidates.items():
            if candidate["admitted"]:
                admitted_by_training_seed.setdefault(
                    int(candidate["training_seed"]), []
                ).append((candidate_id, candidate))
        members: dict[str, dict[str, Any]] = {}
        for training_seed, seed_candidates in admitted_by_training_seed.items():
            ranked = sorted(
                seed_candidates,
                key=lambda item: (
                    -float(item[1]["correct_delivery_rate"]),
                    -int(item[1]["snapshot_environment_steps"]),
                    item[0],
                ),
            )
            for candidate_id, candidate in ranked[
                : self.spec.members_per_training_seed
            ]:
                members[candidate_id] = candidate
        distinct_training_seed_count = len({
            int(member["training_seed"]) for member in members.values()
        })
        report = {
            "schema_version": "path_c_pool_admission_report_v2",
            "run_kind": "admission",
            "scientific_readout_allowed": False,
            "layout": self.env_config.layout,
            "admission_floor": self.spec.admission_floor,
            "minimum_admitted_members": self.spec.minimum_admitted_members,
            "minimum_distinct_training_seeds": (
                self.spec.minimum_distinct_training_seeds
            ),
            "members_per_training_seed": self.spec.members_per_training_seed,
            "distinct_training_seed_count": distinct_training_seed_count,
            "policy_action_selection": self.spec.policy_action_selection,
            "episodes_per_pairing": self.spec.episodes_per_pairing,
            "candidates": candidates,
            "members": members,
            "pairing_matrix": [
                {
                    "policy_0_candidate_id": left,
                    "policy_1_candidate_id": right,
                    "mean_raw_episode_return": float(np.mean([
                        row["raw_episode_return"] for row in rows
                        if row["policy_0_candidate_id"] == left
                        and row["policy_1_candidate_id"] == right
                    ])),
                    "correct_delivery_rate": float(np.mean([
                        row["correct_delivery_count"] for row in rows
                        if row["policy_0_candidate_id"] == left
                        and row["policy_1_candidate_id"] == right
                    ])),
                    "action_distribution_jsd": _action_distribution_jsd(
                        np.sum([
                            row["policy_0_action_counts"] for row in rows
                            if row["policy_0_candidate_id"] == left
                            and row["policy_1_candidate_id"] == right
                        ], axis=0),
                        np.sum([
                            row["policy_1_action_counts"] for row in rows
                            if row["policy_0_candidate_id"] == left
                            and row["policy_1_candidate_id"] == right
                        ], axis=0),
                    ),
                }
                for left, right in pairings
            ],
            "admitted": (
                len(members) >= self.spec.minimum_admitted_members
                and distinct_training_seed_count
                >= self.spec.minimum_distinct_training_seeds
            ),
        }
        (self.spec.output_dir / "pool_admission_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report


@dataclass
class _LoadedR015Policy:
    candidate: R015CandidateSpec
    family: R015FamilySpec
    model: Any
    metadata: dict[str, Any]
    checkpoint_sha256: str
    model_weights_sha256: str
    training_config_sha256: str
    training_manifest_sha256: str
    environment_config_sha256: str
    training_implementation_sha256: str
    training_implementation_dependencies: Mapping[str, Any]
    training_run_id: str
    artifact_verified: bool


class R015PartnerSupportEvaluator:
    """Apply one ability rule to two registered partner-generation families."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.spec = R015PartnerSupportSpec.from_mapping(config)
        self.env_config = StandardEnvConfig.from_mapping(config["environment"])
        self.environment_config_sha256 = _require_sha256(
            config.get("_environment_config_sha256"),
            name="support environment-config content hash",
        )
        self.device = torch.device(str(config.get("device", "cuda")))
        self.policies: dict[str, _LoadedR015Policy] = {}
        for candidate in self.spec.candidates:
            self.policies[candidate.candidate_id] = self._load_policy(candidate)
        self.adapter = self.env_config.make_adapter()
        self.pools: dict[int, BatchedEnvPool] = {}

    def _load_policy(self, candidate: R015CandidateSpec) -> _LoadedR015Policy:
        provenance = validate_r015_candidate_artifact_provenance(
            self.spec,
            candidate,
        )
        if not candidate.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"R015 checkpoint is still pending: {candidate.checkpoint_path}"
            )
        if not candidate.training_config_path.is_file():
            raise FileNotFoundError(
                "R015 training configuration is still pending: "
                f"{candidate.training_config_path}"
            )
        if not candidate.training_manifest_path.is_file():
            raise FileNotFoundError(
                "R015 training manifest is still pending: "
                f"{candidate.training_manifest_path}"
            )
        family = self.spec.family(candidate.family_id)
        training_config_bytes = candidate.training_config_path.read_bytes()
        training_config_sha256 = hashlib.sha256(training_config_bytes).hexdigest()
        training_payload = yaml.safe_load(training_config_bytes.decode("utf-8"))
        if not isinstance(training_payload, Mapping):
            raise ValueError("R015 candidate training configuration is malformed.")
        if int(training_payload.get("seed", -1)) != candidate.training_seed:
            raise ValueError(
                "R015 training config seed differs from its registered run."
            )
        if dict(training_payload.get("partner_family") or {}) != family.definition():
            raise ValueError(
                "R015 training config does not bind its registered partner family."
            )
        declared_environment = training_payload.get("environment_config")
        if not isinstance(declared_environment, str) or not declared_environment:
            raise ValueError("R015 training config does not identify its environment.")
        training_environment_path = _resolved_declared_path(
            declared_environment,
            relative_to=candidate.training_config_path.parent,
        )
        if not training_environment_path.is_file():
            raise FileNotFoundError(
                "R015 candidate environment configuration is missing: "
                f"{training_environment_path}"
            )
        environment_config_sha256 = _file_sha256(training_environment_path)
        if environment_config_sha256 != self.environment_config_sha256:
            raise ValueError(
                "R015 candidate was trained on a different environment config."
            )

        training = training_payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("R015 candidate training schedule is malformed.")
        model_config = training_payload.get("model")
        if not isinstance(model_config, Mapping):
            raise ValueError("R015 candidate model configuration is malformed.")
        if tuple(int(item) for item in model_config.get("observation_shape", ()))[:2] != (
            5,
            5,
        ):
            raise ValueError(
                "R015 candidate changed the official local 5-by-5 observation contract."
            )
        if family.model_class == "RecurrentIPPOBackbone":
            if training_payload.get("schema_version") != BACKBONE_CONFIG_SCHEMA_VERSION:
                raise ValueError("R015 IPPO candidate uses an unexpected config schema.")
            if (
                int(training.get("total_environment_steps", -1))
                != candidate.expected_environment_steps
                or candidate.expected_environment_steps
                not in tuple(int(item) for item in training.get("snapshot_steps", ()))
                or int(training.get("shaping_horizon_environment_steps", -1))
                != 15_000_000
            ):
                raise ValueError("R015 IPPO candidate changed its frozen budget.")
            expected_architecture = ippo_partner_architecture_from_config(
                model_config
            )
            expected_manifest_schema = "path_c_backbone_artifact_v2"
        elif family.model_class == "PrimitiveRecurrentEnsembleQ":
            if training_payload.get("schema_version") != STANDARD_TRAINING_SCHEMA_VERSION:
                raise ValueError("R015 Q candidate uses an unexpected config schema.")
            if training_payload.get("run_kind") != "partner_family" or (
                training_payload.get("training_mode") != "self_play_only"
            ):
                raise ValueError("R015 Q candidate changed its formation mode.")
            if (
                int(training.get("total_environment_steps", -1))
                != candidate.expected_environment_steps
                or int(training.get("self_play_environment_steps", -1))
                != candidate.expected_environment_steps
                or int(training.get("path_c_environment_steps", -1)) != 0
                or candidate.expected_environment_steps
                not in tuple(
                    int(item)
                    for item in training.get("partner_pool_snapshot_steps", ())
                )
            ):
                raise ValueError("R015 Q candidate changed its frozen budget.")
            expected_architecture = q_partner_architecture_from_config(
                model_config
            )
            expected_manifest_schema = "path_c_partner_family_training_artifact_v2"
        else:
            raise ValueError("R015 family declares an unsupported model class.")
        training_implementation_dependencies = (
            _registered_training_implementation_dependencies(family)
        )
        training_implementation_sha256 = canonical_mapping_sha256(
            training_implementation_dependencies
        )
        training_run_id = partner_training_run_id(
            family_spec_sha256=family.family_spec_sha256,
            training_seed=candidate.training_seed,
            training_config_sha256=training_config_sha256,
            environment_config_sha256=environment_config_sha256,
            training_implementation_sha256=training_implementation_sha256,
        )

        training_manifest_bytes = candidate.training_manifest_path.read_bytes()
        training_manifest_sha256 = hashlib.sha256(
            training_manifest_bytes
        ).hexdigest()
        manifest = json.loads(training_manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("R015 candidate training manifest is malformed.")
        if manifest.get("schema_version") != expected_manifest_schema:
            raise ValueError("R015 candidate training manifest schema changed.")
        if manifest.get("run_status") != "completed" or int(
            manifest.get("effective_environment_steps", -1)
        ) != candidate.expected_environment_steps:
            raise ValueError("R015 candidate training did not complete its fixed budget.")
        manifest_architecture = manifest.get("architecture")
        if not isinstance(manifest_architecture, Mapping) or dict(
            manifest_architecture
        ) != expected_architecture:
            raise ValueError(
                "R015 training manifest architecture differs from its model config."
            )

        expected_binding = {
            "partner_family_spec_sha256": family.family_spec_sha256,
            "training_config_sha256": training_config_sha256,
            "environment_config_sha256": environment_config_sha256,
            "training_implementation_sha256": training_implementation_sha256,
            "training_run_id": training_run_id,
        }

        def verify_binding(payload: Mapping[str, Any], *, source: str) -> None:
            if dict(payload.get("partner_family") or {}) != family.definition():
                raise ValueError(
                    f"R015 {source} does not bind the registered partner family."
                )
            for field, expected in expected_binding.items():
                actual = _require_sha256(
                    payload.get(field),
                    name=f"{source} {field}",
                )
                if actual != expected:
                    raise ValueError(
                        f"R015 {source} {field} differs from the registered run."
                    )
            raw_dependencies = payload.get("training_implementation_dependencies")
            if not isinstance(raw_dependencies, Mapping) or dict(
                raw_dependencies
            ) != training_implementation_dependencies:
                raise ValueError(
                    f"R015 {source} changed the training dependency closure."
                )

        verify_binding(manifest, source="training manifest")
        if family.model_class == "PrimitiveRecurrentEnsembleQ":
            if int(manifest.get("seed", -1)) != candidate.training_seed:
                raise ValueError("R015 Q manifest seed differs from registration.")
            declared_checkpoint = manifest.get("final_candidate_checkpoint")
            if declared_checkpoint is None or _resolved_declared_path(
                declared_checkpoint,
                relative_to=candidate.training_manifest_path.parent,
            ) != candidate.checkpoint_path.resolve():
                raise ValueError("R015 Q manifest identifies another checkpoint.")
        else:
            declared_snapshots = manifest.get("snapshots")
            if not isinstance(declared_snapshots, Sequence) or isinstance(
                declared_snapshots, (str, bytes)
            ):
                raise ValueError("R015 IPPO manifest has no snapshot list.")
            resolved_snapshots = {
                _resolved_declared_path(
                    item,
                    relative_to=candidate.training_manifest_path.parent,
                )
                for item in declared_snapshots
            }
            if candidate.checkpoint_path.resolve() not in resolved_snapshots:
                raise ValueError("R015 IPPO manifest identifies another checkpoint.")

        model, metadata = load_standard_checkpoint(
            candidate.checkpoint_path,
            device=self.device,
        )
        if not isinstance(metadata, Mapping):
            raise ValueError("R015 checkpoint metadata is malformed.")
        verify_binding(metadata, source="checkpoint metadata")
        metadata_architecture = metadata.get("architecture")
        if not isinstance(metadata_architecture, Mapping) or dict(
            metadata_architecture
        ) != expected_architecture:
            raise ValueError(
                "R015 checkpoint architecture differs from its model config."
            )
        if model.architecture_manifest() != expected_architecture:
            raise ValueError(
                "R015 loaded model architecture differs from config and checkpoint."
            )
        if metadata.get("partner_family_id") != family.family_id:
            raise ValueError("R015 checkpoint family identifier changed.")
        if family.model_class == "RecurrentIPPOBackbone":
            if not isinstance(model, RecurrentIPPOBackbone):
                raise ValueError("R015 IPPO family requires a RecurrentIPPOBackbone.")
        elif family.model_class == "PrimitiveRecurrentEnsembleQ":
            if not isinstance(model, PrimitiveRecurrentEnsembleQ):
                raise ValueError(
                    "R015 Q family requires a PrimitiveRecurrentEnsembleQ."
                )
        if metadata.get("phase") != family.checkpoint_phase:
            raise ValueError("R015 checkpoint phase differs from its family contract.")
        if int(getattr(model, "n_actions", -1)) != 6:
            raise ValueError("R015 checkpoint changed the six-action contract.")
        if metadata.get("layout") != self.env_config.layout:
            raise ValueError("R015 checkpoint layout differs from the support config.")
        if int(metadata.get("seed", -1)) != candidate.training_seed:
            raise ValueError("R015 checkpoint training seed differs from registration.")
        if int(metadata.get("environment_steps", -1)) != (
            candidate.expected_environment_steps
        ):
            raise ValueError("R015 checkpoint step count differs from registration.")
        return _LoadedR015Policy(
            candidate=candidate,
            family=family,
            model=model,
            metadata=dict(metadata),
            checkpoint_sha256=str(provenance["checkpoint_sha256"]),
            model_weights_sha256=str(provenance["model_weights_sha256"]),
            training_config_sha256=str(provenance["training_config_sha256"]),
            training_manifest_sha256=str(
                provenance["training_manifest_sha256"]
            ),
            environment_config_sha256=str(
                provenance["environment_config_sha256"]
            ),
            training_implementation_sha256=str(
                provenance["training_implementation_sha256"]
            ),
            training_implementation_dependencies=(
                provenance["training_implementation_dependencies"]
            ),
            training_run_id=str(provenance["training_run_id"]),
            artifact_verified=True,
        )

    def _pool(self, batch_size: int) -> BatchedEnvPool:
        if batch_size not in self.pools:
            self.pools[batch_size] = BatchedEnvPool(
                self.adapter.env,
                batch_size,
            )
        return self.pools[batch_size]

    def _act(
        self,
        policy: _LoadedR015Policy,
        batch: Any,
        state: Any,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, Any]:
        with torch.no_grad():
            if policy.family.action_rule == "categorical_actor_logits":
                logits, _, _, next_state = policy.model.forward_step(batch, state)
                actions = sample_actor_actions(
                    logits,
                    rng.random(logits.shape[0]),
                )
            elif policy.family.action_rule == "greedy_mean_q":
                q_values, _, next_state = policy.model.forward_step(batch, state)
                actions = choose_primitive_actions(
                    q_values,
                    batch.valid_actions[:, 0],
                    rng=rng,
                    epsilon=0.0,
                    probe_enabled=False,
                ).actions
            else:
                raise ValueError("R015 partner action rule is unsupported.")
        return actions, next_state

    def _evaluate_chunk(
        self,
        left_id: str,
        right_id: str,
        episode_indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        batch_size = len(episode_indices)
        pool = self._pool(batch_size)
        canonical_seeds = [
            derive_standard_seed(
                self.spec.evaluation_seed,
                "r015_partner_support",
                self.env_config.layout,
                left_id,
                right_id,
                episode_index,
            )
            for episode_index in episode_indices
        ]
        pool.reset(
            np.asarray(
                [derive_ocv2_execution_seed(seed) for seed in canonical_seeds],
                dtype=np.uint32,
            )
        )
        observations = pool.snapshot_obs()
        policies = (self.policies[left_id], self.policies[right_id])
        states = {
            slot: policies[slot].model.initial_state(
                batch_size,
                device=self.device,
            )
            for slot in (0, 1)
        }
        previous_actions = {
            slot: np.full(batch_size, 6, dtype=np.int64) for slot in (0, 1)
        }
        previous_rewards = {
            slot: np.zeros(batch_size, dtype=np.float32) for slot in (0, 1)
        }
        episode_starts = np.ones(batch_size, dtype=bool)
        totals = np.zeros(batch_size, dtype=np.float64)
        correct = np.zeros(batch_size, dtype=np.int64)
        wrong = np.zeros(batch_size, dtype=np.int64)
        indicator = np.zeros(batch_size, dtype=np.int64)
        ambiguous = np.zeros(batch_size, dtype=np.int64)
        action_counts = {
            slot: np.zeros((batch_size, 6), dtype=np.int64) for slot in (0, 1)
        }
        action_rngs = {
            slot: np.random.default_rng(
                derive_ocv2_execution_seed(
                    derive_standard_seed(
                        self.spec.evaluation_seed,
                        "r015_partner_support_action",
                        left_id,
                        right_id,
                        int(episode_indices[0]),
                        slot,
                    )
                )
            )
            for slot in (0, 1)
        }
        for _ in range(self.env_config.max_steps):
            actions: dict[int, np.ndarray] = {}
            for slot, policy in enumerate(policies):
                batch = single_step_batch(
                    observations[f"agent_{slot}"],
                    previous_actions[slot],
                    previous_rewards[slot],
                    episode_starts,
                    n_actions=6,
                    device=self.device,
                )
                actions[slot], states[slot] = self._act(
                    policy,
                    batch,
                    states[slot],
                    action_rngs[slot],
                )
                action_counts[slot][np.arange(batch_size), actions[slot]] += 1
            observations, _, rewards, dones, _ = pool.step_joint(
                actions[0],
                actions[1],
            )
            reward = shared_team_reward(rewards).astype(np.float64)
            totals += reward
            for row, value in enumerate(reward):
                events = decompose_raw_reward_events(np.asarray([value]))
                correct[row] += events.correct_delivery_count
                wrong[row] += events.wrong_delivery_count
                indicator[row] += events.indicator_activation_count
                ambiguous[row] += events.ambiguous_step_count
            episode_starts = np.asarray(dones["__all__"], dtype=bool)
            for slot in (0, 1):
                previous_actions[slot] = actions[slot]
                previous_rewards[slot] = reward.astype(np.float32)
            observations = {
                key: np.asarray(value) for key, value in observations.items()
            }
        if not bool(episode_starts.all()):
            raise ValueError("R015 admission episode did not reach 400 steps.")
        return [
            {
                "schema_version": "path_c_r015_partner_support_episode_v1",
                "scientific_readout_allowed": False,
                "layout": self.env_config.layout,
                "policy_0_candidate_id": left_id,
                "policy_1_candidate_id": right_id,
                "policy_0_family_id": policies[0].family.family_id,
                "policy_1_family_id": policies[1].family.family_id,
                "episode_index": int(episode_index),
                "canonical_episode_seed": int(canonical_seed),
                "environment_steps": self.env_config.max_steps,
                "raw_episode_return": float(totals[row]),
                "correct_delivery_count": int(correct[row]),
                "wrong_delivery_count": int(wrong[row]),
                "indicator_activation_count": int(indicator[row]),
                "ambiguous_reward_step_count": int(ambiguous[row]),
                "policy_0_action_counts": action_counts[0][row].tolist(),
                "policy_1_action_counts": action_counts[1][row].tolist(),
            }
            for row, (episode_index, canonical_seed) in enumerate(
                zip(episode_indices, canonical_seeds, strict=True)
            )
        ]

    def run(self) -> dict[str, Any]:
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = [item.candidate_id for item in self.spec.candidates]
        pairings = [(item, item) for item in candidate_ids] + [
            (left, right)
            for left in candidate_ids
            for right in candidate_ids
            if left != right
        ]
        rows: list[dict[str, Any]] = []
        for left_id, right_id in pairings:
            for offset in range(
                0,
                self.spec.episodes_per_pairing,
                self.spec.evaluation_batch_size,
            ):
                rows.extend(
                    self._evaluate_chunk(
                        left_id,
                        right_id,
                        range(
                            offset,
                            offset + self.spec.evaluation_batch_size,
                        ),
                    )
                )
        rows_path = self.spec.output_dir / "episode_returns.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        candidate_results: dict[str, dict[str, Any]] = {}
        for candidate in self.spec.candidates:
            self_play_rows = [
                row
                for row in rows
                if row["policy_0_candidate_id"] == candidate.candidate_id
                and row["policy_1_candidate_id"] == candidate.candidate_id
            ]
            correct_delivery_rate = float(
                np.mean([row["correct_delivery_count"] for row in self_play_rows])
            )
            loaded = self.policies[candidate.candidate_id]
            candidate_results[candidate.candidate_id] = {
                "artifact_verified": loaded.artifact_verified,
                "family_id": candidate.family_id,
                "family_spec_sha256": loaded.family.family_spec_sha256,
                "training_seed": candidate.training_seed,
                "snapshot_environment_steps": candidate.expected_environment_steps,
                "checkpoint_path": str(candidate.checkpoint_path),
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "model_weights_sha256": loaded.model_weights_sha256,
                "training_config_path": str(candidate.training_config_path),
                "training_config_sha256": loaded.training_config_sha256,
                "training_manifest_path": str(candidate.training_manifest_path),
                "training_manifest_sha256": loaded.training_manifest_sha256,
                "environment_config_sha256": (
                    loaded.environment_config_sha256
                ),
                "training_implementation_sha256": (
                    loaded.training_implementation_sha256
                ),
                "training_implementation_dependencies": dict(
                    loaded.training_implementation_dependencies
                ),
                "training_run_id": loaded.training_run_id,
                "architecture": loaded.model.architecture_manifest(),
                "architecture_sha256": canonical_mapping_sha256(
                    loaded.model.architecture_manifest()
                ),
                "action_rule": loaded.family.action_rule,
                "correct_delivery_rate": correct_delivery_rate,
                "mean_raw_episode_return": float(
                    np.mean([row["raw_episode_return"] for row in self_play_rows])
                ),
                "admitted": correct_delivery_rate >= self.spec.admission_floor,
            }
        selection = select_r015_support_members(self.spec, candidate_results)
        report = {
            "schema_version": "path_c_r015_partner_support_report_v2",
            "registration_status": "configured",
            "support_status": selection["support_status"],
            "artifact_verification_status": "verified",
            "support_complete": selection["support_complete"],
            "scientific_readout_allowed": False,
            "support_registration_sha256": (
                self.spec.support_registration_sha256
            ),
            "admission_implementation_sha256": (
                self.spec.admission_implementation_sha256
            ),
            "admission_implementation_dependencies": dict(
                self.spec.admission_implementation_dependencies
            ),
            "evaluation_seed": self.spec.evaluation_seed,
            "environment_config_sha256": self.environment_config_sha256,
            "layout": self.env_config.layout,
            "episodes_per_pairing": self.spec.episodes_per_pairing,
            "admission_floor": self.spec.admission_floor,
            "episode_evidence": {
                "schema_version": R015_EPISODE_EVIDENCE_SCHEMA_VERSION,
                "path": str(rows_path),
                "sha256": _file_sha256(rows_path),
                "row_count": len(rows),
            },
            "families": {
                family.family_id: {
                    **family.definition(),
                    "family_spec_sha256": family.family_spec_sha256,
                }
                for family in self.spec.families
            },
            "candidates": candidate_results,
            "qualified_candidate_ids": selection["qualified_candidate_ids"],
            "members_by_family": selection["members_by_family"],
            "members": selection["members"],
            "pairing_matrix": [
                {
                    "policy_0_candidate_id": left_id,
                    "policy_1_candidate_id": right_id,
                    "mean_raw_episode_return": float(
                        np.mean(
                            [
                                row["raw_episode_return"]
                                for row in rows
                                if row["policy_0_candidate_id"] == left_id
                                and row["policy_1_candidate_id"] == right_id
                            ]
                        )
                    ),
                    "correct_delivery_rate": float(
                        np.mean(
                            [
                                row["correct_delivery_count"]
                                for row in rows
                                if row["policy_0_candidate_id"] == left_id
                                and row["policy_1_candidate_id"] == right_id
                            ]
                        )
                    ),
                    "action_distribution_jsd": _action_distribution_jsd(
                        np.sum(
                            [
                                row["policy_0_action_counts"]
                                for row in rows
                                if row["policy_0_candidate_id"] == left_id
                                and row["policy_1_candidate_id"] == right_id
                            ],
                            axis=0,
                        ),
                        np.sum(
                            [
                                row["policy_1_action_counts"]
                                for row in rows
                                if row["policy_0_candidate_id"] == left_id
                                and row["policy_1_candidate_id"] == right_id
                            ],
                            axis=0,
                        ),
                    ),
                }
                for left_id, right_id in pairings
            ],
        }
        validate_r015_support_report_evidence(self.spec, report)
        (self.spec.output_dir / "partner_support_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report


def load_pool_admission_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != ADMISSION_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported pool admission configuration.")
    config = dict(payload)
    for key in ("environment_config",):
        target = Path(config[key])
        if not target.is_absolute():
            target = (config_path.parent / target).resolve()
        config["environment"] = yaml.safe_load(target.read_text(encoding="utf-8"))
    config["checkpoint_paths"] = [
        str((config_path.parent / Path(item)).resolve()) if not Path(item).is_absolute() else str(item)
        for item in config["checkpoint_paths"]
    ]
    output = Path(config["output_dir"])
    if not output.is_absolute():
        config["output_dir"] = str((config_path.parent / output).resolve())
    return config


def load_r015_partner_support_config(path: str | Path) -> dict[str, Any]:
    """Load the independent R015 support registration without loading models."""

    config_path = Path(path)
    source_bytes = config_path.read_bytes()
    payload = yaml.safe_load(source_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != (
        R015_SUPPORT_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported R015 partner-support configuration.")
    config = dict(payload)
    config["_source_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    admission_dependencies = repository_source_dependency_closure(
        R015_ADMISSION_IMPLEMENTATION_DEPENDENCIES
    )
    config["_admission_implementation_dependencies"] = admission_dependencies
    config["_admission_implementation_sha256"] = canonical_mapping_sha256(
        admission_dependencies
    )
    environment_path = Path(config["environment_config"])
    if not environment_path.is_absolute():
        environment_path = (config_path.parent / environment_path).resolve()
    environment_bytes = environment_path.read_bytes()
    config["environment"] = yaml.safe_load(environment_bytes.decode("utf-8"))
    config["_environment_config_path"] = str(environment_path)
    config["_environment_config_sha256"] = hashlib.sha256(
        environment_bytes
    ).hexdigest()
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        config["output_dir"] = str((config_path.parent / output_dir).resolve())
    candidates: list[dict[str, Any]] = []
    for raw_candidate in config["candidates"]:
        candidate = dict(raw_candidate)
        for field in (
            "checkpoint_path",
            "training_config_path",
            "training_manifest_path",
        ):
            target = Path(candidate[field])
            if not target.is_absolute():
                target = (config_path.parent / target).resolve()
            candidate[field] = str(target)
        candidates.append(candidate)
    config["candidates"] = candidates
    config["_source_config_path"] = str(config_path.resolve())
    return config


def run_pool_admission(config_path: str | Path) -> dict[str, Any]:
    return PoolAdmissionEvaluator(load_pool_admission_config(config_path)).run()


def run_r015_partner_support(config_path: str | Path) -> dict[str, Any]:
    """Remain pending until all four registered artifacts can be evaluated."""

    config = load_r015_partner_support_config(config_path)
    spec = R015PartnerSupportSpec.from_mapping(config)
    status = configured_r015_support_status(spec)
    if status["missing_checkpoint_candidates"] or status[
        "missing_training_config_candidates"
    ] or status["missing_training_manifest_candidates"]:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "partner_support_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return status
    try:
        return R015PartnerSupportEvaluator(config).run()
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        # Presence is not evidence of provenance.  Old or malformed artifacts
        # remain outside the support instead of being promoted by path alone.
        status["artifact_verification_status"] = "failed"
        status["artifact_verification_error"] = str(error)
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "partner_support_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return status
