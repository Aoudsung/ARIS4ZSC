#!/usr/bin/env python3
"""Run legacy pool admission or the registered two-family R015 admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.overcooked_v2.path_c_pool_admission import (
    ADMISSION_CONFIG_SCHEMA_VERSION,
    R015_SUPPORT_CONFIG_SCHEMA_VERSION,
    run_pool_admission,
    run_r015_partner_support,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    payload = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Partner admission config must contain a mapping.")
    schema_version = payload.get("schema_version")
    if schema_version == ADMISSION_CONFIG_SCHEMA_VERSION:
        result = run_pool_admission(arguments.config)
    elif schema_version == R015_SUPPORT_CONFIG_SCHEMA_VERSION:
        result = run_r015_partner_support(arguments.config)
    else:
        raise ValueError("Unsupported partner admission config schema.")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
