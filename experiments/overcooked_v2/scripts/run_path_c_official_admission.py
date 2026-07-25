#!/usr/bin/env python3
"""Run official-policy R015 admission after the remote adapter is verified."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from experiments.overcooked_v2.path_c_official_evidence import (
    OfficialR015RolloutBackend,
    generate_official_r015_support_report,
)
from experiments.overcooked_v2.path_c_pool_admission import (
    R015PartnerSupportSpec,
    load_r015_partner_support_config,
)


def _load_backend(
    module_path: Path,
    *,
    factory_name: str,
    config: dict,
    spec: R015PartnerSupportSpec,
) -> OfficialR015RolloutBackend:
    module_spec = importlib.util.spec_from_file_location(
        "path_c_official_r015_remote_adapter",
        module_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("The official rollout adapter cannot be loaded.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_spec.name, None)
        raise
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError("The official rollout adapter factory is missing.")
    backend = factory(config=config, support_spec=spec)
    if not isinstance(backend, OfficialR015RolloutBackend):
        raise TypeError("The official rollout adapter returned the wrong interface.")
    return backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend-module", required=True)
    parser.add_argument(
        "--factory-name",
        default="build_official_r015_rollout_backend",
    )
    arguments = parser.parse_args()
    config = load_r015_partner_support_config(arguments.config)
    spec = R015PartnerSupportSpec.from_mapping(config)
    backend = _load_backend(
        Path(arguments.backend_module).resolve(),
        factory_name=arguments.factory_name,
        config=config,
        spec=spec,
    )
    generate_official_r015_support_report(spec, backend)


if __name__ == "__main__":
    main()
