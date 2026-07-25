"""Launch the WP-C Path C adaptation trainer from one resolved YAML config."""

from __future__ import annotations

import argparse
import json

from experiments.overcooked_v2.path_c_adaptation import run_adaptation_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_adaptation_training(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
