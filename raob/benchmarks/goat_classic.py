"""GOAT/MAPBT policy adapters for classic Overcooked-AI experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import sys
import types
from typing import Any

import numpy as np
import torch

from raob.benchmarks.partners import PartnerSpec


def _prepend_once(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def install_goat_paths(external_root: str | Path = "external/goat_overcooked") -> Path:
    """Expose the GOAT repository's ``mapbt`` package for lazy imports."""

    root = Path(external_root).resolve()
    if not (root / "mapbt").exists():
        raise FileNotFoundError(f"GOAT mapbt package not found under {root}")
    _prepend_once(root)
    os.environ.setdefault("POLICY_POOL", str(root / "mapbt" / "scripts" / "overcooked_population"))
    return root


def install_goat_import_stubs() -> None:
    """Provide no-op modules for GOAT imports unused by policy inference."""

    try:
        import setproctitle  # noqa: F401
    except Exception:
        module = types.ModuleType("setproctitle")
        module.setproctitle = lambda *_args, **_kwargs: None
        sys.modules["setproctitle"] = module

    try:
        import slackweb  # noqa: F401
    except Exception:
        module = types.ModuleType("slackweb")
        module.Slack = lambda *_args, **_kwargs: types.SimpleNamespace(
            notify=lambda *_a, **_k: None
        )
        sys.modules["slackweb"] = module

    try:
        import icecream  # noqa: F401
    except Exception:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **_kwargs: args[0] if len(args) == 1 else args
        sys.modules["icecream"] = module

    try:
        import tensorboardX  # noqa: F401
    except Exception:
        module = types.ModuleType("tensorboardX")

        class SummaryWriter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self.scalars: dict[str, Any] = {}

            def add_scalars(self, tag: str, scalars: Mapping[str, Any], step: int = 0) -> None:
                for key, value in dict(scalars).items():
                    full_key = f"{tag}/{key}"
                    self.scalars.setdefault(full_key, []).append([0.0, int(step), float(value)])

            def export_scalars_to_json(self, path: str) -> None:
                Path(path).write_text(json.dumps(self.scalars), encoding="utf-8")

            def close(self) -> None:
                return None

        module.SummaryWriter = SummaryWriter
        sys.modules["tensorboardX"] = module

    try:
        import h5py  # noqa: F401
    except Exception:
        module = types.ModuleType("h5py")
        module.Group = object
        sys.modules["h5py"] = module

    try:
        import torchvision.models  # noqa: F401
    except Exception:
        tv_module = types.ModuleType("torchvision")
        models_module = types.ModuleType("torchvision.models")
        models_module.resnet18 = lambda *_args, **_kwargs: None
        tv_module.models = models_module
        sys.modules["torchvision"] = tv_module
        sys.modules["torchvision.models"] = models_module


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - server dependency
        raise RuntimeError("PyYAML is required to read GOAT population YAML files") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise RuntimeError(f"GOAT population file must contain a mapping: {path}")
    return data


def _resolve_manifest_path(value: str | Path, *, manifest_path: Path, base_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    manifest_relative = manifest_path.parent / candidate
    if manifest_relative.exists():
        return manifest_relative
    return base_path / candidate


def load_goat_partner_specs(
    population_path: str | Path,
    *,
    external_root: str | Path = "external/goat_overcooked",
    layout: str = "cramped_room",
) -> tuple[list[PartnerSpec], list[PartnerSpec]]:
    """Read a GOAT population YAML/JSON into source and held-out partner specs."""

    root = install_goat_paths(external_root)
    manifest = Path(population_path)
    population = _load_mapping(manifest)
    base_path = root / "mapbt" / "scripts" / "overcooked_population"
    source: list[PartnerSpec] = []
    target: list[PartnerSpec] = []
    for partner_id, raw_info in population.items():
        if not isinstance(raw_info, Mapping):
            raise RuntimeError(f"GOAT partner entry must be a mapping: {partner_id}")
        model_path = raw_info.get("model_path", {})
        if not isinstance(model_path, Mapping) or "actor" not in model_path:
            continue
        actor_path = _resolve_manifest_path(
            str(model_path["actor"]),
            manifest_path=manifest,
            base_path=base_path,
        )
        policy_config_path = _resolve_manifest_path(
            str(raw_info["policy_config_path"]),
            manifest_path=manifest,
            base_path=base_path,
        )
        split = "target" if bool(raw_info.get("held_out", False)) else "source"
        spec = PartnerSpec(
            benchmark="overcooked_classic_goat",
            partner_id=f"goat:{layout}:{partner_id}",
            split=split,
            artifact=str(actor_path),
            layout=layout,
            group=str(raw_info.get("featurize_type", "ppo")),
            metadata={
                "policy_name": str(partner_id),
                "policy_config_path": str(policy_config_path),
                "featurize_type": str(raw_info.get("featurize_type", "ppo")),
                "held_out": bool(raw_info.get("held_out", False)),
            },
            source=str(manifest),
        )
        if split == "target":
            target.append(spec)
        else:
            source.append(spec)
    if not source:
        raise RuntimeError(f"no source GOAT partners found in {manifest}")
    if not target:
        raise RuntimeError(f"no held-out GOAT partners found in {manifest}")
    return source, target


def select_goat_partner_specs(
    population_path: str | Path,
    *,
    external_root: str | Path = "external/goat_overcooked",
    layout: str = "cramped_room",
    source_count: int | None = None,
    target_count: int | None = None,
) -> tuple[list[PartnerSpec], list[PartnerSpec]]:
    source, target = load_goat_partner_specs(
        population_path,
        external_root=external_root,
        layout=layout,
    )
    if source_count is not None:
        source = source[: int(source_count)]
    if target_count is not None:
        target = target[: int(target_count)]
    if not source or not target:
        raise ValueError("selected GOAT source and target partner sets must be nonempty")
    return source, target


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class GOATPolicyState:
    rnn_state: np.ndarray


class GOATClassicPartnerPolicy:
    """RAOB ``PartnerPolicy`` wrapper around one GOAT/MAPBT actor checkpoint."""

    def __init__(
        self,
        spec: PartnerSpec,
        *,
        external_root: str | Path = "external/goat_overcooked",
        device: str | torch.device = "cpu",
        agent_index: int = 1,
        deterministic: bool = True,
    ) -> None:
        self.spec = spec
        self.external_root = install_goat_paths(external_root)
        self.device = torch.device(device)
        self.agent_index = int(agent_index)
        if self.agent_index not in (0, 1):
            raise ValueError("agent_index must be 0 or 1")
        self.deterministic = bool(deterministic)
        self.featurize_type = str(spec.metadata.get("featurize_type", "ppo"))
        if self.featurize_type != "ppo":
            raise NotImplementedError(
                "GOATClassicPartnerPolicy currently supports PPO actors only"
            )
        self._policy = self._load_policy()
        self._policy.prep_rollout()
        self._recurrent_n = int(getattr(self._policy_args, "recurrent_N", 1))
        self._hidden_size = int(getattr(self._policy_args, "hidden_size", 64))

    def _load_policy(self) -> Any:
        if not hasattr(np, "Inf"):
            setattr(np, "Inf", np.inf)
        install_goat_import_stubs()
        from mapbt.runner.shared.base_runner import make_trainer_policy_cls

        policy_config_path = Path(str(self.spec.metadata["policy_config_path"]))
        with policy_config_path.open("rb") as handle:
            policy_config = list(pickle.load(handle))
        self._policy_args = policy_config[0]
        _trainer_cls, policy_cls = make_trainer_policy_cls(
            self._policy_args.algorithm_name,
            use_single_network=bool(getattr(self._policy_args, "use_single_network", False)),
        )
        policy = policy_cls(*policy_config, device=self.device)
        policy.load_checkpoint({"actor": str(self.spec.artifact)})
        policy.to(self.device)
        return policy

    def reset(self, seed: int | None = None) -> GOATPolicyState:
        _ = seed
        return GOATPolicyState(
            rnn_state=np.zeros(
                (1, self._recurrent_n, self._hidden_size),
                dtype=np.float32,
            )
        )

    def _obs_from_public_observation(self, observation: Mapping[str, Any]) -> np.ndarray:
        mdp = observation["mdp"]
        state = observation["state"]
        horizon = int(observation.get("horizon", 400))
        encoded = mdp.lossless_state_encoding(state, horizon)[self.agent_index]
        return (np.asarray(encoded, dtype=np.float32) * 255.0)[np.newaxis, ...]

    def act(
        self,
        observation: Mapping[str, Any],
        state: Any = None,
        rng: Any = None,
    ) -> tuple[int, GOATPolicyState]:
        _ = rng
        policy_state = state if isinstance(state, GOATPolicyState) else self.reset(None)
        obs = self._obs_from_public_observation(observation)
        masks = np.ones((1, 1), dtype=np.float32)
        available = np.ones((1, 6), dtype=np.float32)
        with torch.no_grad():
            action, next_rnn = self._policy.act(
                obs,
                policy_state.rnn_state,
                masks,
                available_actions=available,
                deterministic=self.deterministic,
            )
        action_value = int(_to_numpy(action).reshape(-1)[0])
        return action_value, GOATPolicyState(rnn_state=_to_numpy(next_rnn).astype(np.float32))


class GOATClassicEgoPolicy:
    """Partner-blind GOAT/MAPBT actor wrapper for ego rollout collection."""

    partner_blind = True

    def __init__(
        self,
        spec: PartnerSpec,
        *,
        external_root: str | Path = "external/goat_overcooked",
        device: str | torch.device = "cpu",
        agent_index: int = 0,
        deterministic: bool = True,
    ) -> None:
        self._policy = GOATClassicPartnerPolicy(
            spec,
            external_root=external_root,
            device=device,
            agent_index=agent_index,
            deterministic=deterministic,
        )
        self.spec = spec
        self.agent_index = int(agent_index)
        self.deterministic = bool(deterministic)
        self.metadata = {
            "kind": "goat_classic_ego_policy",
            "partner_blind": True,
            "policy_id": spec.partner_id,
            "policy_name": str(spec.metadata.get("policy_name", spec.partner_id)),
            "source_split": spec.split,
            "artifact": str(spec.artifact),
            "policy_config_path": str(spec.metadata.get("policy_config_path", "")),
            "featurize_type": str(spec.metadata.get("featurize_type", "")),
            "agent_index": self.agent_index,
            "deterministic": self.deterministic,
            "requested_device": str(device),
            "effective_device": str(self._policy.device),
            "no_partner_id": True,
            "no_beta": True,
            "no_future_labels": True,
            "fixed_before_factor_learning": True,
            "role": "continuation_policy_not_teacher_label",
            "observation_source": "public_overcooked_state_lossless_encoding",
            "reward_source": "not_used_by_policy",
            "uses_teacher_value": False,
            "uses_shaped_reward": False,
        }

    def reset(self, seed: int | None = None) -> GOATPolicyState:
        return self._policy.reset(seed)

    def act(
        self,
        observation: Mapping[str, Any],
        state_g: torch.Tensor,
        state: Any = None,
        rng: Any = None,
    ) -> tuple[int, GOATPolicyState]:
        _ = state_g
        return self._policy.act(observation, state, rng)


def make_goat_partners(
    specs: Sequence[PartnerSpec],
    *,
    external_root: str | Path = "external/goat_overcooked",
    device: str | torch.device = "cpu",
    agent_index: int = 1,
    deterministic: bool = True,
) -> list[tuple[str, GOATClassicPartnerPolicy]]:
    return [
        (
            spec.partner_id,
            GOATClassicPartnerPolicy(
                spec,
                external_root=external_root,
                device=device,
                agent_index=agent_index,
                deterministic=deterministic,
            ),
        )
        for spec in specs
    ]
