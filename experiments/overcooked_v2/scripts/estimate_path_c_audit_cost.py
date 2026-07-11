"""Static Path C audit-cost preflight; this command never launches rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.overcooked_v2.path_c_evaluation import load_frozen_preregistration
from experiments.overcooked_v2.path_c_protocol import estimate_audit_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preregistration = load_frozen_preregistration(args.preregistration)
    payload = preregistration.payload
    belief = payload["belief_kernel_audit"]
    audit_units = payload["belief_kernel_audit"]["audit_information_states"]
    if not isinstance(audit_units, list) or not audit_units:
        raise ValueError("Frozen audit_information_states must be a non-empty list.")
    estimate = estimate_audit_cost(
        audit_units=len(audit_units),
        probes=int(payload["audit_battery"]["core_probe_count"]),
        outer_replicas_M=int(belief["outer_replicas_M"]),
        inner_forks_L_inner=int(belief["inner_forks_L_inner"]),
        horizon_T_probe=int(belief["probe_horizon_T_probe"]),
        max_primitive_steps=int(
            payload["evidence_spec"]["max_primitive_actions_per_decision"]
        ),
        maximum_primitive_step_budget=int(
            payload["budget"]["maximum_audit_primitive_steps"]
        ),
        design_audit_units=(
            len(audit_units)
            if preregistration.audit_battery.design_only_scripts
            else 0
        ),
        design_only_probes=len(preregistration.audit_battery.design_only_scripts),
    )
    artifact = {
        **estimate,
        "measurement_schema_version": "path_c_measurement_v3",
        "preregistration_sha256": preregistration.sha256,
        "resolved_path_c_sha256": preregistration.runtime_contract_sha256,
        "semantic_bindings": dict(preregistration.semantic_bindings),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Audit cost artifact is immutable and cannot be overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(estimate, indent=2, sort_keys=True))
    if not estimate["within_budget"]:
        raise SystemExit("Path C audit cost exceeds the frozen primitive-step budget.")


if __name__ == "__main__":
    main()
