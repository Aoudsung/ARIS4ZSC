#!/usr/bin/env python3
"""执行登记的 seed 999 官方训练器机械验收，不生成正式伙伴产物。"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
    run_official_reference_acceptance,
    validate_official_reference_acceptance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()
    config = Path(arguments.config).resolve()
    report = Path(arguments.report).resolve()
    result = run_official_reference_acceptance(config, report)
    print(f"reference_acceptance_report={report}")
    print(f"completed_environment_steps={result['completed_environment_steps']}")
    print(f"final_quarter_mean_raw_return={result['final_quarter_mean_raw_return']}")
    print(f"final_to_peak_quarter_ratio={result['final_to_peak_quarter_ratio']}")
    print(f"acceptance_pass={str(result['acceptance_pass']).lower()}")
    if result["acceptance_pass"] is not True:
        raise SystemExit(3)
    validate_official_reference_acceptance_report(config, report)


if __name__ == "__main__":
    main()
