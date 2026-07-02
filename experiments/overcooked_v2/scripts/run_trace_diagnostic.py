"""Step 1: 200-step trace to verify deadlock fix (pot=0xc → deliver invalid)."""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from experiments.overcooked_v2.env_adapter import OCV2Adapter
from experiments.overcooked_v2.layout_parser import parse_layout
from experiments.overcooked_v2.options import OCV2OptionLibrary
from experiments.overcooked_v2.option_termination import (
    OptionRuntime,
    option_success,
    option_terminated,
)
from experiments.overcooked_v2.event_extractor import extract_event
from experiments.overcooked_v2.partner_pool import make_training_partners
from experiments.overcooked_v2.state_utils import (
    get_agent_pos,
    get_inventory,
    get_pot_contents,
    has_plate,
    ingredient_count_py,
    is_empty_inventory,
    is_ingredient,
    is_plated_cooked_soup,
    is_pot_full,
)


def main() -> None:
    env = OCV2Adapter(
        layout="cramped_room",
        max_steps=200,
        observation_type="default",
        force_path_planning=False,
    )
    lg = parse_layout(env.env, "cramped_room")
    lib = OCV2OptionLibrary(lg, max_option_steps=6)
    partners = make_training_partners(lib)
    rng = np.random.default_rng(42)

    print(f"Options ({lib.num_options}):")
    for opt in lib.options:
        print(f"  {opt.id}: {opt.kind} name={opt.name}")

    option_count: Counter[str] = Counter()
    option_real_success: Counter[str] = Counter()
    option_timeout: Counter[str] = Counter()
    delivery_count = 0
    total_options = 0
    deadlock_steps = 0
    pot_full_deliver_blocked = 0

    num_partners = min(len(partners), 3)
    num_episodes = 10

    for pidx in range(num_partners):
        partner = partners[pidx]
        for ep in range(num_episodes):
            obs, state = env.reset(seed=pidx * 1000 + ep)
            if hasattr(partner, "reset"):
                partner.reset(pidx * 1000 + ep)

            for t_opt in range(200):
                valid = lib.valid_options(state, 0)
                valid_ids = np.flatnonzero(valid)
                if not valid_ids.size:
                    break
                oid = int(rng.choice(valid_ids))
                opt = lib.options[oid]
                option_count[opt.kind] += 1
                total_options += 1

                valid_kinds = {lib.options[i].kind for i in valid_ids}
                if valid_kinds == {"noop"}:
                    deadlock_steps += 1

                # Check: if holding ingredient, is deliver_ingredient_to_pot valid?
                inv = get_inventory(state, 0)
                if is_ingredient(inv) and "deliver_ingredient_to_pot" not in valid_kinds:
                    # Check pot state
                    for eid in lib._entity_ids_by_kind("pot"):
                        pot_pos = lg.entities[eid].pos
                        contents = get_pot_contents(state, pot_pos)
                        if is_pot_full(contents):
                            pot_full_deliver_blocked += 1

                pos_start = get_agent_pos(state, 0)
                rt = OptionRuntime(option_id=oid, start_pos=pos_start)
                for step_i in range(opt.max_steps):
                    ego_act = lib.primitive_action(state, 0, oid)
                    p_obs = obs.get("agent_1") if isinstance(obs, dict) else None
                    pa = partner.act(p_obs, state, rng)
                    prev_state = state
                    step_result = env.step(ego_act, pa.primitive_action)
                    event = extract_event(
                        prev_state,
                        ego_act,
                        pa.primitive_action,
                        step_result.state,
                        step_result.info,
                        partner_option=None,
                        partner_option_dist=None,
                        partner_option_source="diagnostic_behavior_only",
                    )
                    state = step_result.state
                    obs = step_result.obs
                    terminated, reason = option_terminated(
                        opt, prev_state, state, event, 0, step_i + 1, rt
                    )
                    if event.delivery_event:
                        delivery_count += 1
                    if terminated:
                        if option_success(opt.kind, reason):
                            option_real_success[opt.kind] += 1
                        elif reason == "max_steps":
                            option_timeout[opt.kind] += 1
                        break
                    if step_result.dones.get("__all__", False):
                        break
                if step_result.dones.get("__all__", False):
                    break

    print(f"\n=== Option stats (total={total_options}) ===")
    for kind in sorted(set(list(option_count.keys()) + list(option_real_success.keys()))):
        cnt = option_count[kind]
        succ = option_real_success[kind]
        tmout = option_timeout[kind]
        print(f"  {kind:30s}: selected={cnt:5d} success={succ:5d} timeout={tmout:5d}")

    print(f"\nDeadlock steps (only noop valid): {deadlock_steps}")
    print(f"Pot-full deliver blocked: {pot_full_deliver_blocked}")
    print(f"Delivery events: {delivery_count}")

    # plate_soup success vs timeout
    ps_succ = option_real_success.get("plate_soup", 0)
    ps_tmout = option_timeout.get("plate_soup", 0)
    ps_total = option_count.get("plate_soup", 0)
    if ps_total > 0:
        print(f"\nplate_soup: {ps_succ}/{ps_total} real success ({ps_succ/ps_total:.1%}), {ps_tmout} timeout")

    ss_succ = option_real_success.get("serve_soup", 0)
    ss_total = option_count.get("serve_soup", 0)
    if ss_total > 0:
        print(f"serve_soup: {ss_succ}/{ss_total} real success ({ss_succ/ss_total:.1%})")

    print("\n=== VERDICT ===")
    if deadlock_steps == 0:
        print("PASS: No deadlocks detected")
    else:
        print(f"FAIL: {deadlock_steps} deadlock steps")

    if pot_full_deliver_blocked > 0:
        print(f"PASS: Pot-full correctly blocked deliver {pot_full_deliver_blocked} times")

    if delivery_count > 0:
        print(f"PASS: {delivery_count} deliveries (full task cycle)")
    else:
        print("WARN: No deliveries in trace (may need more episodes)")


if __name__ == "__main__":
    main()
