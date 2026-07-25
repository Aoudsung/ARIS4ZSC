"""Formal outer-training-unit identities for standard OvercookedV2 evaluation.

An outer training unit owns one official self-play backbone, four official
training partners, and all Path C randomness derived from that unit.  The
manifest is deliberately separate from condition YAML files so the four
conditions can share sources within one unit without permitting any reuse
between units.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTER_UNITS_SCHEMA_VERSION = "path_c_outer_units_v1"
FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION = "path_c_outer_units_v2"
FORMAL_SEED_ROOT = 2_026_072_101
FAMILY_POOL_FORMAL_SEED_ROOT = 2_026_072_302
FORMAL_OUTER_UNIT_COUNT = 10
HISTORY_CHECKPOINT_COUNT = 3
ABILITY_ADMISSION_SCHEMA_VERSION = "path_c_official_ability_admission_v1"
ABILITY_ADMISSION_RULE_ID = "official_final_quarter_ability_v1"
ABILITY_ADMISSION_METRIC = "returned_episode_returns"
ABILITY_ADMISSION_MINIMUM_FINAL_MEAN = 100.0
ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO = 0.9
ABILITY_ADMISSION_EXPECTED_UPDATES = {"rnn-sp": 457, "rnn-op": 1_831}
ABILITY_OBSERVATION_SELECTION_EFFECT = "audit_only_include_all_fixed_seeds_v1"
SELF_PLAY_FAMILY = "official_rnn_sp_ippo_v1"
OTHER_PLAY_FAMILY = "official_rnn_op_other_play_v1"
PARTNER_MEMBER_IDS = ("self_play_0", "self_play_1", "other_play_0", "other_play_1")
DEVELOPMENT_SEEDS = frozenset(
    {
        100,
        101,
        102,
        201,
        202,
        10_101,
        10_102,
        10_103,
        20_201,
        20_202,
        20_203,
    }
)
_HEX = frozenset("0123456789abcdef")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _fields(
    payload: Mapping[str, Any], *, allowed: set[str], required: set[str], name: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown or missing:
        raise ValueError(
            f"{name} fields are invalid; unknown={unknown}, missing={missing}."
        )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX):
        raise ValueError(f"{name} must be a lowercase SHA-256 string.")
    return value


def _path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else base_dir / path).resolve()


def _validated_ability_admission(
    admission: Mapping[str, Any],
    *,
    base_dir: Path,
    name: str,
    family_id: str,
    training_seed: int,
    training_run_id: str,
    launch_config_sha256: str,
    checkpoint_path: Path,
    flax_weights_sha256: str,
) -> None:
    """Recompute the registered ability observation from its bound raw curve."""

    variant = "rnn-sp" if family_id == SELF_PLAY_FAMILY else "rnn-op"
    expected_updates = ABILITY_ADMISSION_EXPECTED_UPDATES[variant]
    expected_steps = 29_949_952 if variant == "rnn-sp" else 29_999_104
    if (
        admission.get("schema_version") != ABILITY_ADMISSION_SCHEMA_VERSION
        or admission.get("rule_id") != ABILITY_ADMISSION_RULE_ID
        or admission.get("run_kind") != "formal"
        or admission.get("scientific_readout_allowed") is not False
        or admission.get("history_selection_consulted_return") is not False
        or admission.get("experiment_variant") != variant
        or admission.get("training_seed") != training_seed
        or admission.get("effective_environment_steps") != expected_steps
        or admission.get("training_run_id") != training_run_id
        or admission.get("launch_config_sha256") != launch_config_sha256
        or Path(str(admission.get("checkpoint_path", ""))).resolve()
        != checkpoint_path
        or admission.get("model_weights_sha256") != flax_weights_sha256
        or admission.get("metric") != ABILITY_ADMISSION_METRIC
        or admission.get("metric_meaning")
        != "official_wrapper_raw_episode_return"
        or admission.get("metric_record_count") != expected_updates
        or admission.get("quarter_partition")
        != "contiguous_integer_update_quarters_v1"
        or admission.get("minimum_final_quarter_mean_raw_return")
        != ABILITY_ADMISSION_MINIMUM_FINAL_MEAN
        or admission.get("minimum_final_to_peak_quarter_ratio")
        != ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO
        or admission.get("selection_effect")
        != ABILITY_OBSERVATION_SELECTION_EFFECT
        or admission.get("included_in_population") is not True
        or admission.get("seed_replacement_allowed") is not False
        or admission.get("retraining_until_pass_allowed") is not False
    ):
        raise ValueError(f"{name} ability-admission report changed identity.")
    _sha256(admission.get("checkpoint_sha256"), f"{name}.checkpoint_sha256")
    metrics_path = _path(
        admission.get("training_metrics_path"),
        base_dir=base_dir,
        name=f"{name}.training_metrics_path",
    )
    metrics_sha256 = _sha256(
        admission.get("training_metrics_sha256"),
        f"{name}.training_metrics_sha256",
    )
    if not metrics_path.is_file() or _file_sha256(metrics_path) != metrics_sha256:
        raise ValueError(f"{name} ability-admission raw metric evidence changed.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_by_update = (
        metrics.get("metrics_by_update") if isinstance(metrics, Mapping) else None
    )
    values = (
        metrics_by_update.get(ABILITY_ADMISSION_METRIC)
        if isinstance(metrics_by_update, Mapping)
        else None
    )
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("schema_version") != "path_c_official_training_metrics_v1"
        or metrics.get("experiment_variant") != variant
        or metrics.get("seed") != training_seed
        or metrics.get("effective_environment_steps") != expected_steps
        or metrics.get("metric_record_count") != expected_updates
        or not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(values) != expected_updates
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError(f"{name} ability-admission raw metric evidence is invalid.")
    boundaries = [index * expected_updates // 4 for index in range(5)]
    quarter_means = [
        sum(float(value) for value in values[boundaries[index] : boundaries[index + 1]])
        / (boundaries[index + 1] - boundaries[index])
        for index in range(4)
    ]
    final_mean = quarter_means[-1]
    peak_mean = max(quarter_means)
    final_to_peak = final_mean / peak_mean if peak_mean > 0.0 else float("-inf")
    performance_pass = final_mean >= ABILITY_ADMISSION_MINIMUM_FINAL_MEAN
    no_collapse_pass = (
        final_to_peak >= ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO
    )
    reported_means = admission.get("quarter_means")
    numerical_fields_match = (
        isinstance(reported_means, Sequence)
        and not isinstance(reported_means, (str, bytes, bytearray))
        and len(reported_means) == 4
        and all(
            math.isclose(float(reported), expected, rel_tol=0.0, abs_tol=1.0e-12)
            for reported, expected in zip(reported_means, quarter_means, strict=True)
        )
        and admission.get("quarter_boundaries_zero_based") == boundaries
        and math.isclose(
            float(admission.get("final_quarter_mean_raw_return")),
            final_mean,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(admission.get("peak_quarter_mean_raw_return")),
            peak_mean,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(admission.get("final_to_peak_quarter_ratio")),
            final_to_peak,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if (
        not numerical_fields_match
        or admission.get("performance_threshold_pass") is not performance_pass
        or admission.get("no_collapse_threshold_pass") is not no_collapse_pass
        or admission.get("passed") is not (performance_pass and no_collapse_pass)
    ):
        raise ValueError(f"{name} ability observation does not recompute.")


def _derive_seed(
    *,
    domain: str,
    seed_root: int,
    layout: str,
    outer_unit_id: int,
    role: str,
    member_index: int = 0,
    checkpoint_index: int = 0,
    stage: str = "",
) -> int:
    unit = _nonnegative_int(outer_unit_id, "outer_unit_id")
    member = _nonnegative_int(member_index, "member_index")
    checkpoint = _nonnegative_int(checkpoint_index, "checkpoint_index")
    if not isinstance(layout, str) or not layout:
        raise ValueError("layout must be a non-empty string.")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string.")
    if not isinstance(stage, str):
        raise ValueError("stage must be a string.")
    payload = (
        f"{domain}\0{seed_root}\0{layout}\0{unit}\0{role}\0"
        f"{member}\0{checkpoint}\0{stage}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def derive_formal_seed(
    *, layout: str, outer_unit_id: int, role: str, member_index: int = 0
) -> int:
    """Derive one unsigned 32-bit seed for historical formal manifests."""

    return _derive_seed(
        domain="path_c_formal_seed_v1",
        seed_root=FORMAL_SEED_ROOT,
        layout=layout,
        outer_unit_id=outer_unit_id,
        role=role,
        member_index=member_index,
    )


def derive_family_pool_seed(
    *,
    layout: str,
    outer_unit_id: int,
    role: str,
    member_index: int = 0,
    checkpoint_index: int = 0,
    stage: str = "",
) -> int:
    """Derive a domain-separated seed for the family-level partner-pool run."""

    return _derive_seed(
        domain="path_c_family_pool_formal_seed_v1",
        seed_root=FAMILY_POOL_FORMAL_SEED_ROOT,
        layout=layout,
        outer_unit_id=outer_unit_id,
        role=role,
        member_index=member_index,
        checkpoint_index=checkpoint_index,
        stage=stage,
    )


@dataclass(frozen=True, slots=True)
class OfficialCheckpointSnapshot:
    """One mechanically scheduled checkpoint from an official training run."""

    checkpoint_index: int
    update_step: int
    effective_environment_steps: int
    checkpoint_path: Path
    checkpoint_sha256: str
    flax_weights_sha256: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
        name: str,
    ) -> "OfficialCheckpointSnapshot":
        payload = _mapping(payload, name)
        fields = {
            "checkpoint_index",
            "update_step",
            "effective_environment_steps",
            "checkpoint_path",
            "checkpoint_sha256",
            "flax_weights_sha256",
        }
        _fields(payload, allowed=fields, required=fields, name=name)
        return cls(
            checkpoint_index=_nonnegative_int(
                payload["checkpoint_index"], f"{name}.checkpoint_index"
            ),
            update_step=_nonnegative_int(payload["update_step"], f"{name}.update_step"),
            effective_environment_steps=_nonnegative_int(
                payload["effective_environment_steps"],
                f"{name}.effective_environment_steps",
            ),
            checkpoint_path=_path(
                payload["checkpoint_path"],
                base_dir=base_dir,
                name=f"{name}.checkpoint_path",
            ),
            checkpoint_sha256=_sha256(
                payload["checkpoint_sha256"], f"{name}.checkpoint_sha256"
            ),
            flax_weights_sha256=_sha256(
                payload["flax_weights_sha256"], f"{name}.flax_weights_sha256"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "checkpoint_index": self.checkpoint_index,
            "update_step": self.update_step,
            "effective_environment_steps": self.effective_environment_steps,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "flax_weights_sha256": self.flax_weights_sha256,
        }


@dataclass(frozen=True, slots=True)
class OfficialPolicySource:
    member_id: str
    family_id: str
    training_seed: int
    checkpoint_path: Path
    flax_weights_sha256: str
    training_run_id: str
    launch_config_path: Path
    launch_config_sha256: str
    checkpoint_history: tuple[OfficialCheckpointSnapshot, ...] = ()
    ability_admission_path: Path | None = None
    ability_admission_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
        name: str,
        verify_files: bool,
        require_history: bool = False,
    ) -> "OfficialPolicySource":
        payload = _mapping(payload, name)
        allowed = {
            "member_id",
            "family_id",
            "training_seed",
            "checkpoint_path",
            "flax_weights_sha256",
            "training_run_id",
            "launch_config_path",
            "launch_config_sha256",
            "checkpoint_history",
            "ability_admission_path",
            "ability_admission_sha256",
        }
        required = (
            allowed
            if require_history
            else allowed
            - {
                "checkpoint_history",
                "ability_admission_path",
                "ability_admission_sha256",
            }
        )
        _fields(payload, allowed=allowed, required=required, name=name)
        member_id = str(payload["member_id"])
        family_id = str(payload["family_id"])
        if family_id not in {SELF_PLAY_FAMILY, OTHER_PLAY_FAMILY}:
            raise ValueError(f"{name}.family_id is not registered.")
        run_id = str(payload["training_run_id"])
        if not member_id or not run_id:
            raise ValueError(f"{name} member and training-run identifiers must be non-empty.")
        launch_path = _path(
            payload["launch_config_path"],
            base_dir=base_dir,
            name=f"{name}.launch_config_path",
        )
        launch_hash = _sha256(
            payload["launch_config_sha256"], f"{name}.launch_config_sha256"
        )
        if verify_files:
            if not launch_path.is_file():
                raise FileNotFoundError(f"Official launch configuration is missing: {launch_path}")
            if _file_sha256(launch_path) != launch_hash:
                raise ValueError(f"{name} launch configuration hash does not match its file.")
        history = tuple(
            OfficialCheckpointSnapshot.from_mapping(
                item,
                base_dir=base_dir,
                name=f"{name}.checkpoint_history[{index}]",
            )
            for index, item in enumerate(
                _sequence(payload.get("checkpoint_history", ()), f"{name}.checkpoint_history")
            )
        )
        if require_history:
            if len(history) != HISTORY_CHECKPOINT_COUNT:
                raise ValueError(f"{name} requires exactly three scheduled checkpoints.")
            if tuple(item.checkpoint_index for item in history) != tuple(
                range(HISTORY_CHECKPOINT_COUNT)
            ):
                raise ValueError(f"{name} checkpoint history must be ordered 0, 1, 2.")
            if any(
                current.update_step >= following.update_step
                or current.effective_environment_steps
                >= following.effective_environment_steps
                for current, following in zip(history, history[1:])
            ):
                raise ValueError(f"{name} checkpoint history must have increasing progress.")
            if history[-1].flax_weights_sha256 != payload["flax_weights_sha256"]:
                raise ValueError(
                    f"{name} final history checkpoint must equal the admitted final weights."
                )
            if len({item.checkpoint_path for item in history}) != HISTORY_CHECKPOINT_COUNT:
                raise ValueError(f"{name} checkpoint history repeats a path.")
        if verify_files:
            missing = [item.checkpoint_path for item in history if not item.checkpoint_path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    f"Official checkpoint history is missing: {', '.join(map(str, missing))}"
                )
        if ("ability_admission_path" in payload) != (
            "ability_admission_sha256" in payload
        ):
            raise ValueError(
                f"{name} ability-admission path and hash must appear together."
            )
        admission_path = (
            _path(
                payload["ability_admission_path"],
                base_dir=base_dir,
                name=f"{name}.ability_admission_path",
            )
            if "ability_admission_path" in payload
            else None
        )
        admission_sha256 = (
            _sha256(
                payload["ability_admission_sha256"],
                f"{name}.ability_admission_sha256",
            )
            if "ability_admission_sha256" in payload
            else None
        )
        if verify_files and admission_path is not None:
            if not admission_path.is_file():
                raise FileNotFoundError(
                    f"Official ability-admission report is missing: {admission_path}"
                )
            if _file_sha256(admission_path) != admission_sha256:
                raise ValueError(
                    f"{name} ability-admission report hash does not match its file."
                )
            admission = json.loads(admission_path.read_text(encoding="utf-8"))
            if not isinstance(admission, Mapping):
                raise ValueError(
                    f"{name} ability-admission report must be a mapping."
                )
            _validated_ability_admission(
                admission,
                base_dir=base_dir,
                name=name,
                family_id=family_id,
                training_seed=_nonnegative_int(
                    payload["training_seed"], f"{name}.training_seed"
                ),
                training_run_id=run_id,
                launch_config_sha256=launch_hash,
                checkpoint_path=_path(
                    payload["checkpoint_path"],
                    base_dir=base_dir,
                    name=f"{name}.checkpoint_path",
                ),
                flax_weights_sha256=_sha256(
                    payload["flax_weights_sha256"],
                    f"{name}.flax_weights_sha256",
                ),
            )
        return cls(
            member_id=member_id,
            family_id=family_id,
            training_seed=_nonnegative_int(
                payload["training_seed"], f"{name}.training_seed"
            ),
            checkpoint_path=_path(
                payload["checkpoint_path"],
                base_dir=base_dir,
                name=f"{name}.checkpoint_path",
            ),
            flax_weights_sha256=_sha256(
                payload["flax_weights_sha256"], f"{name}.flax_weights_sha256"
            ),
            training_run_id=run_id,
            launch_config_path=launch_path,
            launch_config_sha256=launch_hash,
            checkpoint_history=history,
            ability_admission_path=admission_path,
            ability_admission_sha256=admission_sha256,
        )

    def checkpoint_mapping(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "flax_weights_sha256": self.flax_weights_sha256,
            "training_run_id": self.training_run_id,
        }

    def partner_mapping(self) -> dict[str, Any]:
        return {**self.checkpoint_mapping(), "family_id": self.family_id}

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "member_id": self.member_id,
            "family_id": self.family_id,
            "training_seed": self.training_seed,
            "checkpoint_path": str(self.checkpoint_path),
            "flax_weights_sha256": self.flax_weights_sha256,
            "training_run_id": self.training_run_id,
            "launch_config_path": str(self.launch_config_path),
            "launch_config_sha256": self.launch_config_sha256,
        }
        if self.checkpoint_history:
            payload["checkpoint_history"] = [
                item.to_mapping() for item in self.checkpoint_history
            ]
        if self.ability_admission_path is not None:
            payload["ability_admission_path"] = str(self.ability_admission_path)
            payload["ability_admission_sha256"] = self.ability_admission_sha256
        return payload


@dataclass(frozen=True, slots=True)
class OuterUnitSeeds:
    model_seed: int
    environment_seed: int
    evaluation_seed: int

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        layout: str,
        outer_unit_id: int,
        family_pool: bool = False,
    ) -> "OuterUnitSeeds":
        payload = _mapping(payload, "outer unit seeds")
        fields = {"model_seed", "environment_seed", "evaluation_seed"}
        _fields(payload, allowed=fields, required=fields, name="outer unit seeds")
        values = {
            name: _nonnegative_int(payload[name], f"outer unit seeds.{name}")
            for name in fields
        }
        derive = derive_family_pool_seed if family_pool else derive_formal_seed
        expected = {
            "model_seed": derive(
                layout=layout, outer_unit_id=outer_unit_id, role="model"
            ),
            "environment_seed": derive(
                layout=layout, outer_unit_id=outer_unit_id, role="environment"
            ),
            "evaluation_seed": derive(
                layout=layout, outer_unit_id=outer_unit_id, role="evaluation"
            ),
        }
        if values != expected:
            raise ValueError("Outer-unit random seeds do not match the frozen SHA-256 schedule.")
        if len(set(values.values())) != 3:
            raise ValueError("Outer-unit model, environment, and evaluation seeds collide.")
        return cls(**values)

    def to_mapping(self) -> dict[str, int]:
        return {
            "model_seed": self.model_seed,
            "environment_seed": self.environment_seed,
            "evaluation_seed": self.evaluation_seed,
        }


@dataclass(frozen=True, slots=True)
class OuterTrainingUnit:
    outer_unit_id: int
    backbone: OfficialPolicySource
    partners: tuple[OfficialPolicySource, ...]
    seeds: OuterUnitSeeds

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
        layout: str,
        verify_files: bool,
        family_pool: bool = False,
    ) -> "OuterTrainingUnit":
        payload = _mapping(payload, "outer unit")
        fields = {"outer_unit_id", "backbone", "partners", "seeds"}
        _fields(payload, allowed=fields, required=fields, name="outer unit")
        unit_id = _nonnegative_int(payload["outer_unit_id"], "outer_unit_id")
        backbone = OfficialPolicySource.from_mapping(
            payload["backbone"],
            base_dir=base_dir,
            name=f"units[{unit_id}].backbone",
            verify_files=verify_files,
            require_history=family_pool,
        )
        partners = tuple(
            OfficialPolicySource.from_mapping(
                item,
                base_dir=base_dir,
                name=f"units[{unit_id}].partners[{index}]",
                verify_files=verify_files,
                require_history=family_pool,
            )
            for index, item in enumerate(
                _sequence(payload["partners"], f"units[{unit_id}].partners")
            )
        )
        if backbone.member_id != "backbone" or backbone.family_id != SELF_PLAY_FAMILY:
            raise ValueError("Each outer unit requires one self-play backbone named backbone.")
        if len(partners) != 4 or tuple(item.member_id for item in partners) != PARTNER_MEMBER_IDS:
            raise ValueError(
                "Each outer unit requires ordered partners self_play_0, self_play_1, "
                "other_play_0, and other_play_1."
            )
        expected_families = (
            SELF_PLAY_FAMILY,
            SELF_PLAY_FAMILY,
            OTHER_PLAY_FAMILY,
            OTHER_PLAY_FAMILY,
        )
        if tuple(item.family_id for item in partners) != expected_families:
            raise ValueError("Each outer unit requires two self-play and two Other-Play partners.")
        derive = derive_family_pool_seed if family_pool else derive_formal_seed
        expected_training_seeds = {
            "backbone": derive(
                layout=layout, outer_unit_id=unit_id, role="official_backbone"
            ),
            "self_play_0": derive(
                layout=layout,
                outer_unit_id=unit_id,
                role="official_partner_self_play",
                member_index=0,
            ),
            "self_play_1": derive(
                layout=layout,
                outer_unit_id=unit_id,
                role="official_partner_self_play",
                member_index=1,
            ),
            "other_play_0": derive(
                layout=layout,
                outer_unit_id=unit_id,
                role="official_partner_other_play",
                member_index=0,
            ),
            "other_play_1": derive(
                layout=layout,
                outer_unit_id=unit_id,
                role="official_partner_other_play",
                member_index=1,
            ),
        }
        sources = (backbone, *partners)
        observed_training_seeds = {item.member_id: item.training_seed for item in sources}
        if observed_training_seeds != expected_training_seeds:
            raise ValueError("Official training seeds do not match the frozen SHA-256 schedule.")
        return cls(
            outer_unit_id=unit_id,
            backbone=backbone,
            partners=partners,
            seeds=OuterUnitSeeds.from_mapping(
                payload["seeds"],
                layout=layout,
                outer_unit_id=unit_id,
                family_pool=family_pool,
            ),
        )

    @property
    def sources(self) -> tuple[OfficialPolicySource, ...]:
        return (self.backbone, *self.partners)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "outer_unit_id": self.outer_unit_id,
            "backbone": self.backbone.to_mapping(),
            "partners": [item.to_mapping() for item in self.partners],
            "seeds": self.seeds.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class OuterUnitsManifest:
    schema_version: str
    layout: str
    seed_root: int
    units: tuple[OuterTrainingUnit, ...]
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
        path: str | Path | None = None,
        manifest_sha256: str | None = None,
        verify_files: bool = True,
    ) -> "OuterUnitsManifest":
        payload = _mapping(payload, "Path C outer-units manifest")
        fields = {"schema_version", "layout", "seed_root", "units"}
        _fields(payload, allowed=fields, required=fields, name=OUTER_UNITS_SCHEMA_VERSION)
        schema_version = str(payload["schema_version"])
        if schema_version not in {
            OUTER_UNITS_SCHEMA_VERSION,
            FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
        }:
            raise ValueError("The outer-units manifest schema is not registered.")
        family_pool = schema_version == FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION
        layout = str(payload["layout"])
        if layout not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("The formal outer-unit layout is not registered.")
        expected_seed_root = (
            FAMILY_POOL_FORMAL_SEED_ROOT if family_pool else FORMAL_SEED_ROOT
        )
        if _nonnegative_int(payload["seed_root"], "seed_root") != expected_seed_root:
            raise ValueError("Formal outer units must use the frozen seed root.")
        root = Path(base_dir).resolve()
        units = tuple(
            OuterTrainingUnit.from_mapping(
                item,
                base_dir=root,
                layout=layout,
                verify_files=verify_files,
                family_pool=family_pool,
            )
            for item in _sequence(payload["units"], "units")
        )
        if len(units) != FORMAL_OUTER_UNIT_COUNT:
            raise ValueError("A formal manifest requires exactly ten outer training units.")
        if tuple(item.outer_unit_id for item in units) != tuple(range(FORMAL_OUTER_UNIT_COUNT)):
            raise ValueError("Outer training units must be ordered and numbered 0 through 9.")
        resolved_path = Path(path).resolve() if path is not None else root / "<in-memory>"
        if manifest_sha256 is None:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        manifest = cls(
            schema_version=schema_version,
            layout=layout,
            seed_root=expected_seed_root,
            units=units,
            path=resolved_path,
            sha256=_sha256(manifest_sha256, "outer-units manifest SHA-256"),
        )
        validate_outer_unit_manifests((manifest,))
        return manifest

    def unit(self, outer_unit_id: int) -> OuterTrainingUnit:
        unit_id = _nonnegative_int(outer_unit_id, "outer_unit_id")
        if unit_id >= len(self.units):
            raise ValueError("outer_unit must be between 0 and 9.")
        return self.units[unit_id]


def load_outer_units_manifest(
    path: str | Path, *, verify_files: bool = True
) -> OuterUnitsManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return OuterUnitsManifest.from_mapping(
        payload,
        base_dir=manifest_path.parent,
        path=manifest_path,
        manifest_sha256=_file_sha256(manifest_path),
        verify_files=verify_files,
    )


def validate_outer_unit_manifests(
    manifests: Sequence[OuterUnitsManifest],
) -> tuple[OuterUnitsManifest, ...]:
    """Reject source or random-stream reuse within or across layout manifests."""

    values = tuple(manifests)
    if not values:
        raise ValueError("At least one outer-unit manifest is required.")
    source_fields: dict[str, set[Any]] = {
        "checkpoint path": set(),
        "weight hash": set(),
        "training run": set(),
        "launch configuration path": set(),
        "official training seed": set(),
    }
    path_hashes: dict[Path, str] = {}
    history_paths: set[Path] = set()
    history_coordinates: set[tuple[str, int]] = set()
    ability_admission_paths: set[Path] = set()
    all_random_seeds: set[int] = set()
    seen_layouts: set[str] = set()
    for manifest in values:
        if manifest.layout in seen_layouts:
            raise ValueError("Only one outer-unit manifest may define a layout.")
        seen_layouts.add(manifest.layout)
        for unit in manifest.units:
            for source in unit.sources:
                identities = {
                    "checkpoint path": source.checkpoint_path,
                    "weight hash": source.flax_weights_sha256,
                    "training run": source.training_run_id,
                    "launch configuration path": source.launch_config_path,
                    "official training seed": source.training_seed,
                }
                for name, identity in identities.items():
                    if identity in source_fields[name]:
                        raise ValueError(
                            f"Outer training units reuse an official {name}: {identity}."
                        )
                    source_fields[name].add(identity)
                previous_hash = path_hashes.setdefault(
                    source.launch_config_path, source.launch_config_sha256
                )
                if previous_hash != source.launch_config_sha256:
                    raise ValueError("One launch configuration path has conflicting hashes.")
                for snapshot in source.checkpoint_history:
                    coordinate = (source.training_run_id, snapshot.checkpoint_index)
                    if snapshot.checkpoint_path in history_paths:
                        raise ValueError(
                            "Outer training units reuse a checkpoint-history path."
                        )
                    if coordinate in history_coordinates:
                        raise ValueError(
                            "Outer training units reuse a checkpoint-history coordinate."
                        )
                    history_paths.add(snapshot.checkpoint_path)
                    history_coordinates.add(coordinate)
                if source.ability_admission_path is not None:
                    if source.ability_admission_path in ability_admission_paths:
                        raise ValueError(
                            "Outer training units reuse an ability-admission report."
                        )
                    ability_admission_paths.add(source.ability_admission_path)
            for seed in unit.seeds.to_mapping().values():
                if seed in all_random_seeds:
                    raise ValueError("Outer training units reuse a Path C random seed.")
                all_random_seeds.add(seed)
    every_seed = set(source_fields["official training seed"]) | all_random_seeds
    if len(every_seed) != len(source_fields["official training seed"]) + len(
        all_random_seeds
    ):
        raise ValueError("Official and Path C random streams collide.")
    overlap = every_seed & DEVELOPMENT_SEEDS
    if overlap:
        raise ValueError(
            "Formal outer-unit seeds overlap historical development seeds: "
            + ", ".join(str(value) for value in sorted(overlap))
            + "."
        )
    return values


__all__ = [
    "ABILITY_ADMISSION_EXPECTED_UPDATES",
    "ABILITY_ADMISSION_METRIC",
    "ABILITY_ADMISSION_MINIMUM_FINAL_MEAN",
    "ABILITY_ADMISSION_MINIMUM_FINAL_TO_PEAK_RATIO",
    "ABILITY_ADMISSION_RULE_ID",
    "ABILITY_ADMISSION_SCHEMA_VERSION",
    "ABILITY_OBSERVATION_SELECTION_EFFECT",
    "DEVELOPMENT_SEEDS",
    "FORMAL_OUTER_UNIT_COUNT",
    "FORMAL_SEED_ROOT",
    "FAMILY_POOL_FORMAL_SEED_ROOT",
    "FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION",
    "HISTORY_CHECKPOINT_COUNT",
    "OTHER_PLAY_FAMILY",
    "OUTER_UNITS_SCHEMA_VERSION",
    "OfficialPolicySource",
    "OfficialCheckpointSnapshot",
    "OuterTrainingUnit",
    "OuterUnitSeeds",
    "OuterUnitsManifest",
    "SELF_PLAY_FAMILY",
    "derive_formal_seed",
    "derive_family_pool_seed",
    "load_outer_units_manifest",
    "validate_outer_unit_manifests",
]
