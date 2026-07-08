"""D1 dataset generator — held-out response-predictability diagnostic (prereg D1 §2-§3).

One (partner, ego-mode) pair per invocation; writes one .npz chunk + meta.
Substrate diagnostic ONLY: no training loss touches the main method; rollouts ride the
exact eval execution semantics (E._select_option / option_primitive_step / behavior-only
inferencer), so the logged partner-event stream is the same public channel
(behavior_inferred_v1, oracle_source_count must stay 0) the method itself consumes.

Per ego option-decision point we record:
  - public state features (featurizer agent_0 view) + pot/inventory opportunity signals
  - partner public event history (run-length-encoded inferred option kinds + delivery
    punctuation events), truncated to the most recent --hist-window events
  - the ego option kind actually executed
  - labels (D1-rev, prereg §9): first TERMINAL-CHAIN INITIATION attribution
    (no_initiation / ego_first / partner_first) within the next K in {3,5,8} ego option
    decisions. Initiation = the agent ACQUIRES the soup (inventory transitions to
    carrying a plated soup — raw state bits, NOT inferred kinds) or correct-delivers
    (strict D1 attribution). Same-step double initiation is AMBIGUOUS and the affected
    windows are flagged for exclusion (fraction reported).
  - opportunity-onset gates (D1-rev): main = (any pot ready OR cooking) AND neither
    agent carries soup; strict = any pot ready AND neither carries soup (co-report)
  - step-level audit arrays (audit_*) are stored per chunk for offline relabeling

Ego modes (all share ONE anchor checkpoint context; trained weights influence rollouts
only in argmax mode):
  argmax    : anchor checkpoint policy as-is (use a base_only ckpt per prereg)
  fullchain : scripted priority full task chain (probe_behavior semantics), noop tail
  random    : uniform over valid options
  prepchain : full chain minus terminal kinds (deferring competent ego; Link-A r2 —
              reaction-family defer triggers only appear under a deferring ego)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2 import evaluate_aris as E  # noqa: E402
from experiments.overcooked_v2.option_executor import option_primitive_step  # noqa: E402
from experiments.overcooked_v2.partner_modes import (  # noqa: E402
    ID_TO_FAMILY,
    ID_TO_POLICY,
    ID_TO_TRIGGER,
)
from experiments.overcooked_v2.partner_pool import make_training_partners  # noqa: E402
from experiments.overcooked_v2.state_utils import (  # noqa: E402
    get_cell_extra,
    get_inventory,
    has_plate,
    is_cooked,
    is_pot_cooking,
    is_pot_ready_for_plate,
)

# Same deliberate-play order as probe_behavior.py (sec18.14 certification ego).
KNOWN_ORDER = [
    "serve_soup", "plate_soup", "pick_plate", "deliver_ingredient_to_pot",
    "fetch_ingredient", "wait_duration_after_arrival", "wait_duration",
    "wait_at_bottleneck", "cross_bottleneck", "handoff_counter",
    "press_recipe_button", "clear_interaction_cell", "drop_item_to_counter",
]

PAD, UNK = "<PAD>", "<UNK>"
EV_PARTNER_DELIV_OK = "PARTNER_DELIVERY_CORRECT"
EV_PARTNER_DELIV_WRONG = "PARTNER_DELIVERY_WRONG"
ORACLE_LIKE = ("oracle", "scripted", "terminal_policy", "protocol", "partner_action")
K_WINDOWS = (3, 5, 8)
# step delivery codes
D_NONE, D_EGO, D_PARTNER, D_AMBIG = 0, 1, 2, 3


def _full_priority(option_lib) -> list[str]:
    present = list(dict.fromkeys(str(o.kind) for o in option_lib.options))
    ordered = [k for k in KNOWN_ORDER if k in present]
    extras = [k for k in present if k not in KNOWN_ORDER and k != "noop"]
    return ordered + extras + (["noop"] if "noop" in present else [])


def _kind_vocab(option_lib) -> list[str]:
    kinds = sorted(set(str(o.kind) for o in option_lib.options))
    return [PAD, UNK, EV_PARTNER_DELIV_OK, EV_PARTNER_DELIV_WRONG] + kinds


def _carries_soup(inv: int) -> bool:
    return is_cooked(inv) and has_plate(inv)


def _mode_diag(partner) -> dict[str, int]:
    fn = getattr(partner, "diagnostic_mode_state", None)
    if not callable(fn):
        return {
            "mode_policy_id": -1,
            "mode_family_id": -1,
            "mode_param": -1,
            "mode_age": -1,
            "mode_opportunity_count": -1,
            "mode_last_trigger_id": -1,
        }
    diag = fn()
    return {
        "mode_policy_id": int(diag.get("mode_policy_id", -1)),
        "mode_family_id": int(diag.get("mode_family_id", -1)),
        "mode_param": int(diag.get("mode_param", -1)),
        "mode_age": int(diag.get("mode_age", -1)),
        "mode_opportunity_count": int(diag.get("mode_opportunity_count", -1)),
        "mode_last_trigger_id": int(diag.get("mode_last_trigger_id", -1)),
    }


def _jsonable_dataclass(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return None


def _pot_signals(state, pot_positions) -> tuple[int, int, float]:
    # D1-rev (prereg §9): gate readiness is ANY-recipe ready — wrong-recipe soup
    # acquisition counts as initiation, so wrong-recipe-ready pots must gate too
    # (codex D1-rev review, blocker 1). Correct-recipe count is a separate feature.
    n_ready = sum(
        1 for p in pot_positions
        if is_pot_ready_for_plate(state, p, require_correct_recipe=False)
    )
    n_cooking = sum(1 for p in pot_positions if is_pot_cooking(state, p))
    timers = [int(get_cell_extra(state, p)) for p in pot_positions]
    pos_timers = [t for t in timers if t > 0]
    min_timer = float(np.log1p(min(pos_timers))) if pos_timers else 0.0
    return int(n_ready), int(n_cooking), min_timer


def _step_delivery_code(event) -> int:
    """Strict first-delivery attribution (codex finding 1).

    ego counts only when the delivery is unambiguously ego's
    (ego_sole_correct_delivery = ego delivered AND partner did not AND correct);
    partner counts only when ego did not deliver in the same step. A same-step
    double delivery with any correct component is AMBIGUOUS.
    """
    ego_deliv = bool(event.ego_delivery_event)
    partner_deliv = bool(event.partner_delivery_event)
    any_correct = bool(event.ego_correct_delivery) or bool(event.partner_correct_delivery)
    if ego_deliv and partner_deliv and any_correct:
        return D_AMBIG
    if bool(event.ego_sole_correct_delivery):
        return D_EGO
    if bool(event.partner_correct_delivery) and not ego_deliv:
        return D_PARTNER
    return D_NONE


class _HistoryTracker:
    """Run-length encode the per-primitive-step inferred partner option kind stream,
    punctuated by partner delivery events (public, from the event extractor)."""

    def __init__(self, kind_to_id: dict[str, int]):
        self.kind_to_id = kind_to_id
        self.events: list[tuple[int, int, int]] = []  # (kind_id, duration, end_step)
        self._run_kind: int | None = None
        self._run_len = 0
        self._step = 0

    def _flush(self) -> None:
        if self._run_kind is not None and self._run_len > 0:
            self.events.append((self._run_kind, self._run_len, self._step))
        self._run_kind, self._run_len = None, 0

    def push(self, inferred_kind: str | None, deliv_ok: bool, deliv_wrong: bool) -> None:
        self._step += 1
        kid = self.kind_to_id.get(str(inferred_kind), self.kind_to_id[UNK])
        if kid != self._run_kind:
            self._flush()
            self._run_kind = kid
        self._run_len += 1
        if deliv_ok or deliv_wrong:
            self._flush()
            ev = EV_PARTNER_DELIV_OK if deliv_ok else EV_PARTNER_DELIV_WRONG
            self.events.append((self.kind_to_id[ev], 1, self._step))

    def snapshot(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        # Include the in-flight run so "partner has been doing X for d steps" is visible.
        evs = list(self.events)
        if self._run_kind is not None and self._run_len > 0:
            evs.append((self._run_kind, self._run_len, self._step))
        evs = evs[-window:]
        kind = np.zeros(window, dtype=np.int16)
        dur = np.zeros(window, dtype=np.float32)
        ago = np.zeros(window, dtype=np.float32)
        for i, (kid, d, end) in enumerate(evs):
            kind[i] = kid
            dur[i] = np.log1p(float(d))
            ago[i] = np.log1p(float(max(0, self._step - end)))
        return kind, dur, ago, len(evs)


def run(args: argparse.Namespace) -> None:
    ckpt = Path(args.anchor_checkpoint)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    ctx = E._load_context(ckpt, "d1_anchor")
    ctx.qaudit = None
    ctx.scripted_fsm = None
    if args.ego == "fullchain":
        ctx.scripted_priority = _full_priority(ctx.option_lib)
    elif args.ego == "prepchain":
        # deferring competent ego (Link-A r2): reaction-family triggers
        # (escalate/tit-for-tat punish paths) only fire under ego deferrals
        terminal = ("serve_soup", "plate_soup", "pick_plate")
        ctx.scripted_priority = [
            k for k in _full_priority(ctx.option_lib) if k not in terminal]
    else:
        ctx.scripted_priority = None
    random_policy = args.ego == "random"

    policy = E._evidence_policy_for_config(ctx.config)
    if policy != "behavior_inferred_v1":
        raise RuntimeError(f"D1 requires inferred evidence mode, got {policy}")

    vocab = _kind_vocab(ctx.option_lib)
    kind_to_id = {k: i for i, k in enumerate(vocab)}
    graph = ctx.graph
    env = E._build_env(graph.layout_name, ctx.config)
    env.set_featurizer(E.NumpyFeaturizer(ctx.layout_graph))
    partners = {
        p.name: p
        for p in make_training_partners(ctx.option_lib, partner_set=args.partner_set)
    }
    if args.partner not in partners:
        raise KeyError(f"unknown partner {args.partner!r}; choices={sorted(partners)}")
    partner = partners[args.partner]
    pot_positions = [e.pos for e in ctx.layout_graph.entities.values() if e.kind == "pot"]
    patience = int((ctx.config.get("options") or {}).get("block_patience", 0))

    rng = np.random.default_rng(args.seed)
    rows: dict[str, list] = {k: [] for k in (
        "state_feat", "extra_feat", "hist_kind", "hist_dur", "hist_ago", "hist_len",
        "ego_opt_kind", "valid_kinds", "gate_main", "gate_strict",
        "episode_id", "dp_index", "dp_step",
        "mode_policy_id", "mode_family_id", "mode_param", "mode_age",
        "mode_opportunity_count", "mode_last_trigger_id",
    )}
    audit: dict[str, list] = {k: [] for k in (
        "ep", "code", "inv0b", "inv0a", "inv1b", "inv1a", "n_ready", "n_cooking")}
    label_cols: dict[str, list] = {}
    for k in K_WINDOWS:
        label_cols[f"label_k{k}"] = []
        label_cols[f"censored_k{k}"] = []
        label_cols[f"ambig_k{k}"] = []
    source_counts: dict[str, int] = {}
    oracle_source_count = 0
    ambiguous_steps = 0
    golden_lines: list[str] = []

    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        evidence_buffer = E.EvidenceBuffer(
            num_factors=graph.num_factors,
            window=int(ctx.config["training"]["evidence_window"]),
            evidence_dim=E.D_EVID,
        )
        evidence_buffer.reset()
        router = E.OCV2EvidenceRouter(
            graph,
            ctx.layout_graph.cell_to_entity,
            ctx.layout_graph.region_cells,
            evidence_policy=policy,
        )
        router.reset()
        obs, state0 = env.reset(seed)
        partner.reset(seed)
        inferencer = E.make_behavior_option_inferencer(ctx.option_lib, ctx.config)
        inferencer.reset(state0)
        E._initialise_persistent_belief(
            evidence_buffer, ctx.method, ctx.belief_model, graph,
            E.torch.device("cpu"), E._belief_persistence_enabled(ctx.config),
        )
        tracker = _HistoryTracker(kind_to_id)
        selection_stats = {
            "option_selection_count": 0, "forced_noop_count": 0,
            "no_valid_option_count": 0,
        }
        step_deliv: list[int] = []
        dp_first_step: list[int] = []
        ep_row_start = len(rows["episode_id"])
        done = False
        option_count = 0
        primitive_steps = 0

        while not done and option_count < args.max_episode_options:
            # ---- decision-point record (state BEFORE selecting/executing) ----
            state = env.state
            n_ready, n_cooking, min_timer = _pot_signals(state, pot_positions)
            inv0, inv1 = int(get_inventory(state, 0)), int(get_inventory(state, 1))
            ego_soup, partner_soup = _carries_soup(inv0), _carries_soup(inv1)
            # D1-rev opportunity-onset gates (prereg §9): soup pending, nobody holds it
            clean_hands = not ego_soup and not partner_soup
            gate_main = bool((n_ready > 0 or n_cooking > 0) and clean_hands)
            gate_strict = bool(n_ready > 0 and clean_hands)
            valid = ctx.option_lib.valid_options(state, 0)
            valid_ids = np.flatnonzero(valid)
            valid_kind_hot = np.zeros(len(vocab), dtype=np.uint8)
            for vid in valid_ids:
                valid_kind_hot[kind_to_id.get(
                    str(ctx.option_lib.options[int(vid)].kind), kind_to_id[UNK])] = 1
            hk, hd, ha, hl = tracker.snapshot(args.hist_window)
            sync_public_state = getattr(partner, "sync_public_state", None)
            if callable(sync_public_state):
                sync_public_state(state)
            md = _mode_diag(partner)

            option_id = E._select_option(
                ctx, obs, state, evidence_buffer, graph, rng, random_policy,
                partner_id=int(getattr(partner, "partner_id", 0)),
                selection_stats=selection_stats,
            )
            opt = ctx.option_lib.options[int(option_id)]

            rows["state_feat"].append(np.asarray(obs["agent_0"], dtype=np.float32))
            rows["extra_feat"].append(np.asarray([
                float(n_ready), float(n_cooking), min_timer,
                float(ego_soup), float(partner_soup),
                float(has_plate(inv0)), float(has_plate(inv1)),
                float(option_count) / float(args.max_episode_options),
                np.log1p(float(primitive_steps)),
            ], dtype=np.float32))
            rows["hist_kind"].append(hk)
            rows["hist_dur"].append(hd)
            rows["hist_ago"].append(ha)
            rows["hist_len"].append(np.int16(hl))
            rows["ego_opt_kind"].append(
                np.int16(kind_to_id.get(str(opt.kind), kind_to_id[UNK])))
            rows["valid_kinds"].append(valid_kind_hot)
            rows["gate_main"].append(np.uint8(gate_main))
            rows["gate_strict"].append(np.uint8(gate_strict))
            rows["episode_id"].append(np.int32(episode_idx))
            rows["dp_index"].append(np.int16(option_count))
            rows["dp_step"].append(np.int32(primitive_steps))
            rows["mode_policy_id"].append(np.int16(md["mode_policy_id"]))
            rows["mode_family_id"].append(np.int16(md["mode_family_id"]))
            rows["mode_param"].append(np.int16(md["mode_param"]))
            rows["mode_age"].append(np.int16(md["mode_age"]))
            rows["mode_opportunity_count"].append(np.int16(md["mode_opportunity_count"]))
            rows["mode_last_trigger_id"].append(np.int16(md["mode_last_trigger_id"]))
            dp_first_step.append(primitive_steps)

            # ---- execute the option (mirrors _execute_eval_option semantics) ----
            runtime = E.OptionRuntime(
                option_id=int(option_id), start_pos=E.get_agent_pos(state, 0))
            budget = ctx.option_lib.option_budget(state, 0, int(option_id))
            duration = 0
            termination_reason = "running"
            _spd = ctx.option_lib.layout_graph.shortest_path_dist
            _targets = tuple(ctx.option_lib._target_cells(opt))

            def _dist_to_target(_st) -> int | None:
                if not _targets:
                    return None
                _a = E.get_agent_pos(_st, 0)
                _ds = [d for d in (_spd.get((_a, _t)) for _t in _targets) if d is not None]
                return min(_ds) if _ds else None

            best_dist = _dist_to_target(state)
            stuck = 0
            while duration < budget:
                ostep = option_primitive_step(
                    env, ctx.option_lib, int(option_id), partner, obs, rng,
                    partner_option_inferencer=inferencer,
                )
                event = ostep.event
                duration += 1
                primitive_steps += 1
                src = str(event.partner_option_source)
                source_counts[src] = source_counts.get(src, 0) + 1
                if any(tok in src for tok in ORACLE_LIKE):
                    oracle_source_count += 1
                inferred_kind = None
                if event.partner_option is not None:
                    inferred_kind = str(
                        ctx.option_lib.options[int(event.partner_option)].kind)
                tracker.push(
                    inferred_kind,
                    bool(event.partner_correct_delivery),
                    bool(getattr(event, "partner_wrong_delivery_event", False)),
                )
                # D1-rev initiation coding (prereg §9): acquisition from RAW inventory
                # bits + strict delivery attribution; inferred kinds never touch labels.
                dcode = _step_delivery_code(event)
                inv0_b = int(get_inventory(ostep.prev_state, 0))
                inv1_b = int(get_inventory(ostep.prev_state, 1))
                inv0_a = int(get_inventory(ostep.step.state, 0))
                inv1_a = int(get_inventory(ostep.step.state, 1))
                ego_init = (_carries_soup(inv0_a) and not _carries_soup(inv0_b)) \
                    or dcode == D_EGO
                partner_init = (_carries_soup(inv1_a) and not _carries_soup(inv1_b)) \
                    or dcode == D_PARTNER
                if dcode == D_AMBIG or (ego_init and partner_init):
                    code = D_AMBIG
                    ambiguous_steps += 1
                elif ego_init:
                    code = D_EGO
                elif partner_init:
                    code = D_PARTNER
                else:
                    code = D_NONE
                step_deliv.append(code)
                sr, sc, _ = _pot_signals(ostep.step.state, pot_positions)
                audit["ep"].append(episode_idx)
                audit["code"].append(code)
                audit["inv0b"].append(inv0_b)
                audit["inv0a"].append(inv0_a)
                audit["inv1b"].append(inv1_b)
                audit["inv1a"].append(inv1_a)
                audit["n_ready"].append(sr)
                audit["n_cooking"].append(sc)
                x_f = router.route(
                    event, ego_option_id=int(option_id),
                    ego_option_elapsed=duration, ego_option_max_steps=opt.max_steps,
                )
                evidence_buffer.append(x_f)
                E._advance_persistent_belief(
                    evidence_buffer, ctx.method, ctx.belief_model, graph,
                    E.torch.device("cpu"), x_f, E._belief_persistence_enabled(ctx.config),
                )
                done = bool(ostep.step.dones.get("__all__", False))
                if patience and best_dist is not None and not done:
                    cur = _dist_to_target(ostep.step.state)
                    if cur is not None and cur < best_dist:
                        best_dist, stuck = cur, 0
                    elif cur is not None and cur > 0:
                        stuck += 1
                    if stuck >= patience:
                        obs = ostep.step.obs
                        termination_reason = "blocked_no_progress"
                        break
                terminated, termination_reason = ctx.option_lib.option_terminated(
                    opt, ostep.prev_state, ostep.step.state, event,
                    agent_id=0, elapsed=duration, runtime=runtime,
                )
                if done and not terminated:
                    termination_reason = "env_max_steps"
                obs = ostep.step.obs
                if done or terminated:
                    break
            if not done and termination_reason == "running":
                termination_reason = "budget_exhausted"
            if termination_reason in {
                "budget_exhausted", "max_steps", "env_max_steps", "blocked_no_progress"
            } and duration > 0:
                x_fail = router.route_failure_boundary(
                    ego_option_id=int(option_id),
                    ego_option_elapsed=duration, ego_option_max_steps=opt.max_steps,
                )
                evidence_buffer.append(x_fail)
                E._advance_persistent_belief(
                    evidence_buffer, ctx.method, ctx.belief_model, graph,
                    E.torch.device("cpu"), x_fail,
                    E._belief_persistence_enabled(ctx.config),
                )
            option_count += 1

        # ---- labels: first correct-delivery attribution within next K decisions ----
        n_dp = len(dp_first_step)
        total_steps = len(step_deliv)
        deliv = np.asarray(step_deliv, dtype=np.int8)
        for k in K_WINDOWS:
            for i in range(n_dp):
                start = dp_first_step[i]
                end = dp_first_step[i + k] if i + k < n_dp else total_steps
                window = deliv[start:end]
                hits = np.flatnonzero(window)
                first = int(window[hits[0]]) if hits.size else D_NONE
                ambig = first == D_AMBIG
                label = 0 if ambig else first
                censored = bool(i + k >= n_dp and hits.size == 0)
                label_cols[f"label_k{k}"].append(np.int8(label))
                label_cols[f"censored_k{k}"].append(np.uint8(censored))
                label_cols[f"ambig_k{k}"].append(np.uint8(ambig))
        if args.golden and episode_idx == 0:
            for i in range(n_dp):
                s = dp_first_step[i]
                e = dp_first_step[i + 1] if i + 1 < n_dp else total_steps
                golden_lines.append(
                    f"dp{i:03d} step[{s}:{e}) gate_main={rows['gate_main'][ep_row_start + i]} "
                    f"gate_strict={rows['gate_strict'][ep_row_start + i]} "
                    f"mode_policy={int(rows['mode_policy_id'][ep_row_start + i])} "
                    f"mode_family={int(rows['mode_family_id'][ep_row_start + i])} "
                    f"mode_param={int(rows['mode_param'][ep_row_start + i])} "
                    f"mode_age={int(rows['mode_age'][ep_row_start + i])} "
                    f"mode_opp={int(rows['mode_opportunity_count'][ep_row_start + i])} "
                    f"mode_trigger={int(rows['mode_last_trigger_id'][ep_row_start + i])} "
                    f"ego_opt={vocab[int(rows['ego_opt_kind'][ep_row_start + i])]} "
                    f"deliv_in_opt={deliv[s:e].tolist()} "
                    f"label_k5={int(label_cols['label_k5'][ep_row_start + i])} "
                    f"ambig_k5={int(label_cols['ambig_k5'][ep_row_start + i])} "
                    f"censored_k5={int(label_cols['censored_k5'][ep_row_start + i])} "
                    f"hist_len={int(rows['hist_len'][ep_row_start + i])}"
                )

    n = len(rows["episode_id"])
    arrays = {k: np.stack(v) if k in ("state_feat", "extra_feat", "hist_kind",
                                      "hist_dur", "hist_ago", "valid_kinds")
              else np.asarray(v) for k, v in rows.items()}
    for k, v in label_cols.items():
        arrays[k] = np.asarray(v)
    assert all(len(a) == n for a in arrays.values()), "ragged record arrays"
    for k, v in audit.items():
        arrays[f"audit_{k}"] = np.asarray(
            v, dtype=np.int32 if k == "ep" else np.int16)

    gm = arrays["gate_main"].astype(bool)
    lab5 = arrays["label_k5"]
    meta = {
        "diag": "D1",
        "partner": args.partner,
        "partner_set": args.partner_set,
        "ego": args.ego,
        "episodes": int(args.episodes),
        "base_seed": int(args.seed),
        "max_episode_options": int(args.max_episode_options),
        "hist_window": int(args.hist_window),
        "k_windows": list(K_WINDOWS),
        "kind_vocab": vocab,
        "anchor_checkpoint": str(ckpt),
        "anchor_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "anchor_method": str(ctx.method),
        "evidence_policy": policy,
        "oracle_source_count": int(oracle_source_count),
        "partner_option_source_counts": source_counts,
        "ambiguous_delivery_steps": int(ambiguous_steps),
        "ambig_record_counts": {
            f"k{k}": int(arrays[f"ambig_k{k}"].sum()) for k in K_WINDOWS},
        "label_version": "rev1_initiation (prereg §9)",
        "n_records": int(n),
        "n_gate_main": int(gm.sum()),
        "n_gate_strict": int(arrays["gate_strict"].sum()),
        "class_counts_k5_gate_main": {
            "no_initiation": int(((lab5 == 0) & gm).sum()),
            "ego_first": int(((lab5 == 1) & gm).sum()),
            "partner_first": int(((lab5 == 2) & gm).sum()),
        },
        "selection_stats_last_episode": selection_stats,
        "opportunity_gate_def": {
            "main": "(any pot ready OR cooking) AND neither agent carries soup",
            "strict": "any pot ready AND neither agent carries soup",
        },
        "mode_policy_vocab": [ID_TO_POLICY[i] for i in sorted(ID_TO_POLICY)],
        "mode_family_vocab": [ID_TO_FAMILY[i] for i in sorted(ID_TO_FAMILY)],
        "mode_trigger_vocab": [ID_TO_TRIGGER[i] for i in sorted(ID_TO_TRIGGER)],
        "latent_partner_spec": _jsonable_dataclass(getattr(partner, "spec", None)),
    }
    if oracle_source_count != 0:
        raise RuntimeError(
            f"WIRING VIOLATION: oracle-like partner-option sources seen "
            f"({oracle_source_count}); D1 must ride behavior_inferred_v1 only.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    if args.golden:
        out.with_suffix(".golden.txt").write_text("\n".join(golden_lines))
    print(json.dumps({k: meta[k] for k in (
        "partner", "ego", "n_records", "n_gate_main", "n_gate_strict",
        "class_counts_k5_gate_main", "ambiguous_delivery_steps",
        "oracle_source_count")}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_checkpoint", required=True,
                    help="checkpoint providing env/option_lib/graph; also the argmax ego")
    ap.add_argument("--partner", required=True)
    ap.add_argument("--partner_set", default="role_conditioned_v2",
                    choices=("role_conditioned_v2", "blind_v1", "latent_v3_dev"))
    ap.add_argument("--ego", required=True,
                    choices=("argmax", "fullchain", "random", "prepchain"))
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-episode-options", type=int, default=40)
    ap.add_argument("--hist-window", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden", action="store_true",
                    help="dump a human-checkable per-decision timeline for episode 0")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
