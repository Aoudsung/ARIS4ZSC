"""Phase 0 golden-trace equivalence harness (RC-2b executor extraction).

Runs a deterministic CE option-replay collect on a fixed layout/seed and prints a stable hash of the
replay rows. Run it with the NEW (extracted) `ce_sampler` and with the HEAD (inline) `ce_sampler`; the
hashes must be identical, proving the shared-executor extraction is behavior-preserving (no-op).

The CE collect path (`ce_sampler._rollout_option`) is independent of value_bound / F2 / eval hooks
(it samples valid options, uses no Q-net), so this isolates the executor extraction. train/eval use the
identical extraction pattern.

Usage:
    python experiments/overcooked_v2/scripts/executor_golden_trace.py --tag new \
        --out results/ocv2_rc1_bounded/executor_v0_equivalence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.ce_sampler import collect_option_replay  # noqa: E402
from experiments.overcooked_v2.layout_parser import parse_layout  # noqa: E402
from experiments.overcooked_v2.obs_featurizer import NumpyFeaturizer  # noqa: E402
from experiments.overcooked_v2.options import OCV2OptionLibrary  # noqa: E402
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402


def _ser(row):
    d = asdict(row) if is_dataclass(row) else dict(getattr(row, "__dict__", {}))
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(v).round(6).tolist()
        elif isinstance(v, float):
            out[k] = round(v, 6)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--partners", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-options", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "experiments/overcooked_v2/configs/ocv2_step4.yaml").read_text())
    env = E._build_env(args.layout, cfg)
    lg = parse_layout(env, args.layout)
    env.set_featurizer(NumpyFeaturizer(lg))
    ol = OCV2OptionLibrary(lg, max_option_steps=int(cfg["options"]["max_option_steps"]))
    partners = make_training_partners(ol)[: args.partners]

    rows = collect_option_replay(
        env, partners, ol, layout_name=args.layout,
        episodes=int(args.episodes), seed=0, max_options_per_episode=int(args.max_options),
    )
    payload = json.dumps([_ser(r) for r in rows], sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    result = {"tag": args.tag, "layout": args.layout, "partners": args.partners,
              "episodes": args.episodes, "n_rows": len(rows), "hash": digest}
    print(json.dumps(result))
    if args.out:
        outp = (REPO / args.out).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
