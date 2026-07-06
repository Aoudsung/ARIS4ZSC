"""Override gate — does forcing the terminal (serve) chain on a *yielding* held-out
partner raise return vs the trained argmax(Q) policy?

Purpose (rev3 §6.1-1 / codex round-3 [09] discriminator): the held-out deadlock
(heldout-handoff-alternate-yield: mutual-wait, teamCCR=0) could be either
  (a) CONTROL/CREDIT-fixable — ego *can* take over but argmax defaults to wait
      (belief shifts Q but never out-ranks wait), OR
  (b) TASK-BROKEN — taking over genuinely does not pay against this partner,
      in which case no belief-filter/interface change helps and the route is
      population training.
Existing data is OBSERVATIONAL (takeover pays where the model *chose* it: seed2
1.00/52 deliveries). This gate supplies the missing INTERVENTIONAL bit: force the
takeover in the states where argmax chooses wait, and measure the return lift.

Design (minimal, disciplined):
  * Rides the EXISTING eval-only `scripted_priority` hook (RC-2 mechanism,
    evaluate_aris._select_option): when a valid option's kind is in the priority
    list it is forced (bypassing Q); otherwise selection falls back to argmax(Q).
    Setting the priority to ONLY the terminal chain [serve_soup, plate_soup,
    pick_plate] = "advance the serving chain whenever feasible, else behave
    normally" = the minimal takeover intervention on top of the trained policy.
  * No edits to evaluate_aris.py; evidence policy is untouched (the integrity
    gate sees the same behavior_inferred_v1 policy and passes, as in RC-2).
  * Two arms per (seed, partner), SAME eval-seed so episode i is matched
    (identical env reset + partner reset; only ego option-selection differs):
        argmax   : scripted_priority = None            (trained policy)
        override : scripted_priority = TERMINAL_CHAIN   (forced takeover)
    Paired lift  Δ_i = return_override_i − return_argmax_i.
  * Primary partner = heldout-handoff-alternate-yield (the deadlock case).
    Contrast partner = heldout-resource-server-claim (a CLAIMING partner, where
    forcing takeover is *expected* to be neutral/harmful — guards against the
    "doing more = better" reading; a role-adaptive metric must NOT reward
    bulldozing a claimer).

Wiring self-verification (memory: verify-experiment-condition-wiring): the run
reads `ctx.scripted_priority` back from the context per arm, confirms the
terminal-chain kinds exist in the graph, records the evidence_policy actually
stamped (must stay behavior_inferred_v1), and records terminal-chain firing
(override must attempt materially more terminal options than argmax — else the
comparison is inert and is reported as such, not as "takeover doesn't pay").

Usage (remote, on zsc-customer):
    JAX_PLATFORMS=cpu python experiments/overcooked_v2/scripts/override_gate.py \
        --checkpoints results_e1rev/.../seed0,.../seed1,.../seed2,.../seed4 \
        --partners heldout-handoff-alternate-yield,heldout-resource-server-claim \
        --episodes 25 --max-episode-options 60 --eval-seed 0 \
        --out results_override_gate/override_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402

# The takeover intervention: force the terminal serving chain when any of its
# options is valid, else fall back to argmax(Q). Order = most-terminal first so a
# ready-to-serve state serves rather than re-plating.
TERMINAL_CHAIN = ["serve_soup", "plate_soup", "pick_plate"]


def _resolve_ckpt(path_str: str) -> Path:
    ckpt = Path(path_str)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    if ckpt.is_dir():
        for name in ("checkpoint.pt", "checkpoint_final.pt"):
            if (ckpt / name).exists():
                return ckpt / name
        raise FileNotFoundError(f"no checkpoint.pt/checkpoint_final.pt under {ckpt}")
    return ckpt


def _terminal_attempts(row: dict[str, Any]) -> int:
    oks = row.get("option_kind_stats", {}) or {}
    return int(sum(int((oks.get(k, {}) or {}).get("attempt_count", 0)) for k in TERMINAL_CHAIN))


def _run_arm(
    ctx: Any,
    env: Any,
    partner_obj: Any,
    router: Any,
    *,
    priority: list[str] | None,
    episode_seeds: list[int],
    max_episode_options: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one arm over a prebuilt (env, partner, router), one row per episode.

    priority=None -> argmax(Q); priority=list -> forced terminal chain when valid.

    Pairing (codex round-2 finding 1): the partner consumes the episode RNG
    (`rng.choice` in partner_pool.act), so a single RNG threaded across episodes
    would desync the arms after episode 0 diverges. We give each episode its OWN
    RNG `default_rng(eval_seed+i)` and call `_run_episode` with `seed=eval_seed+i`
    directly — so episode i starts identically in BOTH arms (same env.reset(seed),
    same partner.reset(seed), same fresh RNG); only ego selection differs. Within
    an episode the arms diverge — that divergence IS the causal effect measured.

    Speed: env/partner/router are built ONCE per (ckpt, partner) by the caller and
    reused here (each `_run_episode` internally resets them). This replaces the
    prior design that rebuilt the env per episode (~100x fewer JAX env builds).
    """
    ctx.scripted_fsm = None          # never use the FSM path in this gate
    ctx.scripted_priority = priority # None => argmax; list => forced-chain override
    ctx.qaudit = None
    # Read the effective condition back off the context (do not trust intent):
    effective = getattr(ctx, "scripted_priority", "<<missing>>")
    if effective != priority:
        raise RuntimeError(
            f"wiring: scripted_priority did not stick (set {priority!r}, read {effective!r})"
        )
    rows: list[dict[str, Any]] = []
    for es in episode_seeds:
        rng = np.random.default_rng(int(es))
        row = E._run_episode(
            ctx, env, partner_obj, router, ctx.graph, rng, int(es),
            int(max_episode_options),
            random_policy=False,
            collect_diagnostics=False,
            allow_diag_skip=True,
        )
        rows.append(row)
    wiring = {
        "scripted_priority_effective": effective,
        # evidence policy is fixed by the router we were built with (unchanged here):
        "evidence_policy": E._evidence_policy_for_config(ctx.config),
    }
    return rows, wiring


def _bootstrap_ci(diffs: list[float], n_boot: int, boot_seed: int) -> dict[str, float]:
    arr = np.asarray([float(d) for d in diffs], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot_means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n": int(arr.size),
    }


def _partner_block(
    checkpoints: list[Path],
    variant: str,
    partner: str,
    *,
    episodes: int,
    eval_seed: int,
    max_episode_options: int,
    n_boot: int,
    boot_seed: int,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    paired_diffs: list[float] = []
    evidence_policies: set[str] = set()
    terminal_kinds_present: list[str] | None = None
    episode_seeds = [int(eval_seed) + i for i in range(int(episodes))]

    for ckpt in checkpoints:
        ctx = E._load_context(ckpt, variant)
        seed_name = ckpt.parent.name

        # Wiring: the override can only force kinds that exist in this graph. If NONE
        # of the terminal-chain kinds is present, an inert override would be
        # mis-read as "deadlock upstream" — that is a config/wiring failure, so stop.
        present = [str(o.kind) for o in ctx.graph.options]
        kinds_present = [k for k in TERMINAL_CHAIN if k in present]
        if not kinds_present:
            raise RuntimeError(
                f"wiring: none of the terminal-chain kinds {TERMINAL_CHAIN} exist in graph "
                f"{ckpt} (present kinds={sorted(set(present))}) — override cannot fire."
            )
        terminal_kinds_present = kinds_present

        # Build env/partner/router ONCE per (ckpt, partner) — mirrors _evaluate_partner's
        # setup (evaluate_aris.py:382-397) but reused across both arms and all episodes.
        env = E._build_env(ctx.graph.layout_name, ctx.config)
        env.set_featurizer(E.NumpyFeaturizer(ctx.layout_graph))
        router = E.OCV2EvidenceRouter(
            ctx.graph,
            ctx.layout_graph.cell_to_entity,
            ctx.layout_graph.region_cells,
            evidence_policy=E._evidence_policy_for_config(ctx.config),
        )
        partners = {p.name: p for p in E.make_training_partners(
            ctx.option_lib,
            partner_set=str(ctx.config.get("training", {}).get("partner_set", "standard7")),
        )}
        if partner not in partners:
            raise KeyError(f"Unknown partner {partner!r}; choices={sorted(partners)}")
        partner_obj = partners[partner]

        argmax_rows, w_arg = _run_arm(
            ctx, env, partner_obj, router, priority=None,
            episode_seeds=episode_seeds, max_episode_options=max_episode_options,
        )
        override_rows, w_ovr = _run_arm(
            ctx, env, partner_obj, router, priority=list(TERMINAL_CHAIN),
            episode_seeds=episode_seeds, max_episode_options=max_episode_options,
        )
        evidence_policies.add(w_arg["evidence_policy"])
        evidence_policies.add(w_ovr["evidence_policy"])
        print(f"  [{partner}] {seed_name}: argmax_ret="
              f"{float(np.mean([r['return'] for r in argmax_rows])):.2f} "
              f"override_ret={float(np.mean([r['return'] for r in override_rows])):.2f}",
              flush=True)

        n = min(len(argmax_rows), len(override_rows))
        diffs = [float(override_rows[i]["return"]) - float(argmax_rows[i]["return"]) for i in range(n)]
        paired_diffs.extend(diffs)

        def _mean(rows: list[dict[str, Any]], key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in rows]
            return float(np.mean(vals)) if vals else float("nan")

        def _rate(rows: list[dict[str, Any]], key: str) -> float:
            vals = [1.0 if bool(r.get(key)) else 0.0 for r in rows]
            return float(np.mean(vals)) if vals else float("nan")

        per_seed.append({
            "seed": seed_name,
            "checkpoint": str(ckpt),
            "argmax": {
                "return_mean": _mean(argmax_rows, "return"),
                "ego_correct_rate": _rate(argmax_rows, "ego_correct_completed"),
                "team_rate": _rate(argmax_rows, "team_completed"),
                "terminal_attempts_mean": float(np.mean([_terminal_attempts(r) for r in argmax_rows])),
            },
            "override": {
                "return_mean": _mean(override_rows, "return"),
                "ego_correct_rate": _rate(override_rows, "ego_correct_completed"),
                "team_rate": _rate(override_rows, "team_completed"),
                "terminal_attempts_mean": float(np.mean([_terminal_attempts(r) for r in override_rows])),
            },
            "paired_lift_mean": float(np.mean(diffs)) if diffs else float("nan"),
            "n_paired": int(n),
        })

    ci = _bootstrap_ci(paired_diffs, n_boot=n_boot, boot_seed=boot_seed)
    # Firing check: did the override actually force materially more terminal options?
    arg_term = float(np.mean([s["argmax"]["terminal_attempts_mean"] for s in per_seed])) if per_seed else float("nan")
    ovr_term = float(np.mean([s["override"]["terminal_attempts_mean"] for s in per_seed])) if per_seed else float("nan")
    fired = bool(ovr_term - arg_term >= 1.0)

    return {
        "partner": partner,
        "paired_lift_ci": ci,
        "argmax_terminal_attempts_mean": arg_term,
        "override_terminal_attempts_mean": ovr_term,
        "override_fired": fired,
        "terminal_kinds_present": terminal_kinds_present,
        "evidence_policies_seen": sorted(evidence_policies),
        "per_seed": per_seed,
    }


def _verdict(primary: dict[str, Any]) -> str:
    ci = primary["paired_lift_ci"]
    if not primary["override_fired"]:
        return (
            "OVERRIDE_INERT: the terminal chain never became valid under forced play "
            "(override attempted no more terminal options than argmax). The deadlock is "
            "UPSTREAM of serving — prep/cook is not happening — so neither U2 nor a "
            "serve-side interface change addresses it. Investigate the prep/cook chain."
        )
    lo, hi = ci["ci_low"], ci["ci_high"]
    if lo > 0.0:
        return (
            f"TAKEOVER_PAYS: forcing takeover raises return (paired lift mean={ci['mean']:.3f}, "
            f"95% CI [{lo:.3f}, {hi:.3f}] > 0). The capability exists and the return signal is "
            "there — the deadlock is CONTROL/CREDIT-fixable; a belief/interface change can help."
        )
    if hi < 0.0:
        return (
            f"TAKEOVER_HURTS: forcing takeover lowers return (95% CI [{lo:.3f}, {hi:.3f}] < 0). "
            "Taking over is genuinely wrong against this partner — deference is correct; the "
            "deadlock is not fixed by making ego serve. Route = population / keep deferring."
        )
    return (
        f"TAKEOVER_NEUTRAL: forced takeover does not change return (95% CI [{lo:.3f}, {hi:.3f}] "
        "spans 0) even though it fired. Takeover does not pay even when forced — this leans "
        "TASK-BROKEN rather than control-fixable; the population route is favored over an "
        "interface change."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Override gate (rev3 §6.1-1 / codex [09] discriminator).")
    ap.add_argument("--checkpoints", required=True,
                    help="comma-separated seed dirs or .pt files (deployable aris seeds, e.g. s0,s1,s2,s4)")
    ap.add_argument("--partners",
                    default="heldout-handoff-alternate-yield,heldout-resource-server-claim",
                    help="comma-separated; FIRST is the primary (deadlock) partner for the verdict")
    ap.add_argument("--variant", default="role_v2", help="descriptive label only (graph comes from ckpt)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--max-episode-options", type=int, default=60)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoints = [_resolve_ckpt(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
    partners = [p.strip() for p in args.partners.split(",") if p.strip()]
    if not checkpoints:
        raise SystemExit("no checkpoints specified")
    if not partners:
        raise SystemExit("no partners specified")

    print("=== override gate ===")
    print(f"checkpoints ({len(checkpoints)}):")
    for c in checkpoints:
        print(f"  {c}")
    print(f"partners: {partners}  (primary={partners[0]})")
    print(f"override terminal chain: {TERMINAL_CHAIN}")
    print(f"episodes={args.episodes} max_episode_options={args.max_episode_options} eval_seed={args.eval_seed}\n")

    blocks: dict[str, Any] = {}
    for partner in partners:
        block = _partner_block(
            checkpoints, args.variant, partner,
            episodes=int(args.episodes), eval_seed=int(args.eval_seed),
            max_episode_options=int(args.max_episode_options),
            n_boot=int(args.bootstrap), boot_seed=int(args.bootstrap_seed),
        )
        blocks[partner] = block
        ci = block["paired_lift_ci"]
        print(f"[{partner}] lift mean={ci['mean']:.3f} CI[{ci['ci_low']:.3f},{ci['ci_high']:.3f}] "
              f"n={ci['n']} fired={block['override_fired']} "
              f"(term attempts argmax={block['argmax_terminal_attempts_mean']:.2f} "
              f"override={block['override_terminal_attempts_mean']:.2f})")

    primary = blocks[partners[0]]
    verdict = _verdict(primary)
    print(f"\nVERDICT (primary={partners[0]}): {verdict}")

    # Wiring guard: evidence policy must be the formal one on every arm (gate untouched).
    all_policies = sorted({p for b in blocks.values() for p in b["evidence_policies_seen"]})
    if all_policies != ["behavior_inferred_v1"]:
        print(f"\n[WIRING WARNING] unexpected evidence policies: {all_policies} "
              "(expected only 'behavior_inferred_v1' — the gate must not perturb the evidence path)")

    out = Path(args.out)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gate": "override_gate",
        "override_terminal_chain": TERMINAL_CHAIN,
        "primary_partner": partners[0],
        "episodes": int(args.episodes),
        "max_episode_options": int(args.max_episode_options),
        "eval_seed": int(args.eval_seed),
        "verdict": verdict,
        "evidence_policies_seen": all_policies,
        "partners": blocks,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
