from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

import numpy as np

from jaxmarl.environments.overcooked_v2.common import Actions
from src.aris_bellman.specs import OptionSpec, PartnerAction

from .partner_modes import (
    LatentModeController,
    LatentModeSpec,
    LatentPartnerSpec,
)
from .option_termination import OptionRuntime, cross_bottleneck_terminated
from .state_utils import (
    get_agent_pos,
    get_inventory,
    is_empty_inventory,
    is_ingredient,
    pot_accepts_inventory_ingredient,
)

GridPos = tuple[int, int]

TERMINAL_KINDS = frozenset({"pick_plate", "plate_soup", "serve_soup"})
PREP_KINDS = frozenset({"fetch_ingredient", "deliver_ingredient_to_pot"})
SUPPORT_KINDS = frozenset({
    "drop_item_to_counter",
    "clear_interaction_cell",
    "wait_at_bottleneck",
    "cross_bottleneck",
})


class PartnerPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction: ...

    def get_state(self) -> Mapping[str, Any]: ...

    def set_state(self, state: Mapping[str, Any]) -> None: ...

    def exact_observed_action_branches(
        self,
        state: Any,
        observed_primitive_action: int,
    ) -> tuple[tuple[Mapping[str, Any], float], ...]: ...


class _ForcedOptionChoiceRNG:
    """Minimal RNG adapter that enumerates one registered option draw exactly."""

    def __init__(self, forced_choice: int | None) -> None:
        self.forced_choice = forced_choice

    def choice(self, values: Any, *, p: Any) -> int:
        if self.forced_choice is None:
            raise RuntimeError("An unexpected stochastic option boundary was reached.")
        support = np.asarray(values, dtype=np.int64).reshape(-1)
        probabilities = np.asarray(p, dtype=np.float64).reshape(-1)
        if support.shape != probabilities.shape:
            raise ValueError("Enumerated option support and probabilities are unaligned.")
        matches = np.flatnonzero(support == int(self.forced_choice))
        if matches.size != 1 or probabilities[int(matches[0])] <= 0.0:
            raise ValueError("Forced option choice is outside positive generation support.")
        return int(self.forced_choice)


@dataclass(frozen=True)
class ProtocolSpec:
    role: str | None = None
    bottleneck_policy: str | None = None
    pot_preference: str | None = None
    delivery_preference: str | None = None
    serving_style: str | None = None
    terminal_policy: str | None = None
    button_policy: str | None = None
    counter_preference: str | None = None
    curriculum_group: str | None = None


STANDARD7_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    (
        "ingredient-near",
        ProtocolSpec(role="ingredient_person", pot_preference="near"),
    ),
    (
        "ingredient-far",
        ProtocolSpec(role="ingredient_person", pot_preference="far"),
    ),
    (
        "dish-server",
        ProtocolSpec(role="dish_person", delivery_preference="nearest"),
    ),
    (
        "server-left",
        ProtocolSpec(role="server", delivery_preference="left"),
    ),
    (
        "bottleneck-yield",
        ProtocolSpec(role="flexible", bottleneck_policy="yield"),
    ),
    (
        "flexible-balanced",
        ProtocolSpec(role="flexible", bottleneck_policy="alternate"),
    ),
    (
        "terminal-yield",
        ProtocolSpec(
            role="prep_partner",
            pot_preference="near",
            bottleneck_policy="yield",
            terminal_policy="yield",
        ),
    ),
)

TRAINING_PROTOCOLS = STANDARD7_PROTOCOLS  # backward-compat alias

ROLE_CONDITIONED_V1_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    (
        "ingredient-near-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="near",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "ingredient-far-yield",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="far",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "server-left-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="left",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "server-right-claim",
        ProtocolSpec(
            role="server",
            delivery_preference="right",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "bottleneck-yield-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
    (
        "bottleneck-push-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-yield-terminal-claim",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="yield",
            terminal_policy="claim",
            curriculum_group="terminal_claim",
        ),
    ),
    (
        "heldout-push-terminal-yield",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="push",
            terminal_policy="yield",
            curriculum_group="terminal_yield",
        ),
    ),
)

ROLE_CONDITIONED_V2_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    *ROLE_CONDITIONED_V1_PROTOCOLS[:6],
    (
        "heldout-handoff-alternate-yield",
        ProtocolSpec(
            role="flexible",
            pot_preference="far",
            bottleneck_policy="alternate",
            terminal_policy="yield",
            counter_preference="handoff",
            curriculum_group="heldout_terminal_yield",
        ),
    ),
    (
        "heldout-resource-server-claim",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="far",
            delivery_preference="right",
            bottleneck_policy="push",
            terminal_policy="claim",
            counter_preference="clear",
            curriculum_group="heldout_terminal_claim",
        ),
    ),
)

BLIND_V1_PROTOCOLS: tuple[tuple[str, ProtocolSpec], ...] = (
    # FORMAL BLIND ROUND partner set (METHOD_LOCK sec18.14, signed 2026-07-06).
    # Constructed from parameter-space principles ONLY — no model behavior was
    # consulted. SINGLE-LOOK RULE: no trained checkpoint may be evaluated against
    # any "blind-*" partner before the formal preregistered run; certification
    # probes are checkpoint-free (scripted-ego only). This registry must NEVER be
    # merged into a training partner_set.
    (
        "blind-dish-yield",
        ProtocolSpec(
            role="dish_person",
            terminal_policy="yield",
            curriculum_group="blind_yield",
        ),
    ),
    (
        "blind-prep-alternate-yield",
        ProtocolSpec(
            role="prep_partner",
            pot_preference="far",
            bottleneck_policy="alternate",
            terminal_policy="yield",
            curriculum_group="blind_yield",
        ),
    ),
    (
        "blind-dish-claim",
        ProtocolSpec(
            role="dish_person",
            terminal_policy="claim",
            delivery_preference="nearest",
            curriculum_group="blind_claim",
        ),
    ),
    (
        "blind-server-nearest-claim-push",
        ProtocolSpec(
            role="server",
            delivery_preference="nearest",
            bottleneck_policy="push",
            terminal_policy="claim",
            counter_preference="clear",
            curriculum_group="blind_claim",
        ),
    ),
    (
        "blind-bottleneck-alternate-neutral",
        ProtocolSpec(
            role="flexible",
            bottleneck_policy="alternate",
            counter_preference="clear",
            curriculum_group="blind_offaxis",
        ),
    ),
    (
        "blind-ingredient-near-neutral",
        ProtocolSpec(
            role="ingredient_person",
            pot_preference="near",
            counter_preference="handoff",
            curriculum_group="blind_offaxis",
        ),
    ),
)


def _latent(
    geometry_profile: str,
    base_protocol: ProtocolSpec,
    family: str,
    param: int | None = None,
    *,
    curriculum_group: str,
    epsilon: float = 0.10,
    fingerprint_id: int | None = None,
    fingerprint_bias: str | None = None,
    value_class_id: str | None = None,
    surface_identity_key: str | None = None,
    style_id: str | None = None,
    overlap_posterior_group: str | None = None,
) -> LatentPartnerSpec:
    return LatentPartnerSpec(
        geometry_profile=geometry_profile,
        base_protocol=base_protocol,
        mode=LatentModeSpec(family=family, param=param, epsilon=epsilon),
        curriculum_group=curriculum_group,
        fingerprint_id=fingerprint_id,
        fingerprint_bias=fingerprint_bias,
        value_class_id=value_class_id,
        surface_identity_key=surface_identity_key,
        style_id=style_id,
        overlap_posterior_group=overlap_posterior_group,
    )


LATENT_V3_DEV_PROTOCOLS: tuple[tuple[str, LatentPartnerSpec], ...] = (
    (
        "latent-ingnear-patience2",
        _latent(
            "ingredient_near",
            ProtocolSpec(role="ingredient_person", pot_preference="near"),
            "patience",
            2,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-ingnear-escalate2",
        _latent(
            "ingredient_near",
            ProtocolSpec(role="ingredient_person", pot_preference="near"),
            "escalate_after_defer",
            2,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-ingfar-patience4",
        _latent(
            "ingredient_far",
            ProtocolSpec(role="ingredient_person", pot_preference="far"),
            "patience",
            4,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-ingfar-titfortat1",
        _latent(
            "ingredient_far",
            ProtocolSpec(role="ingredient_person", pot_preference="far"),
            "tit_for_tat",
            1,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-prepnear-block12",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="near", bottleneck_policy="yield"),
            "block_switch",
            12,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-prepnear-static-yield",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="near", bottleneck_policy="yield"),
            "static_yield",
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-prepfar-block18",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="far", bottleneck_policy="yield"),
            "block_switch",
            18,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-prepfar-static-claim",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="far", bottleneck_policy="yield"),
            "static_claim",
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-bneck-titfortat2",
        _latent(
            "bottleneck_zone",
            ProtocolSpec(role="flexible", bottleneck_policy="alternate"),
            "tit_for_tat",
            2,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-bneck-escalate3",
        _latent(
            "bottleneck_zone",
            ProtocolSpec(role="flexible", bottleneck_policy="alternate"),
            "escalate_after_defer",
            3,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-flex-patience6",
        _latent(
            "flexible",
            ProtocolSpec(role="flexible", bottleneck_policy="yield", counter_preference="handoff"),
            "patience",
            6,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "latent-flex-block25",
        _latent(
            "flexible",
            ProtocolSpec(role="flexible", bottleneck_policy="yield", counter_preference="handoff"),
            "block_switch",
            25,
            curriculum_group="latent_v3_train",
        ),
    ),
    (
        "blind-cert-ingnear-titfortat3",
        _latent(
            "ingredient_near",
            ProtocolSpec(role="ingredient_person", pot_preference="near"),
            "tit_for_tat",
            3,
            curriculum_group="latent_v3_cert",
        ),
    ),
    (
        "blind-cert-ingfar-escalate2",
        _latent(
            "ingredient_far",
            ProtocolSpec(role="ingredient_person", pot_preference="far"),
            "escalate_after_defer",
            2,
            curriculum_group="latent_v3_cert",
        ),
    ),
    (
        "blind-cert-prepnear-patience5",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="near", bottleneck_policy="yield"),
            "patience",
            5,
            curriculum_group="latent_v3_cert",
        ),
    ),
    (
        "blind-cert-prepfar-block20",
        _latent(
            "prep_zone",
            ProtocolSpec(role="prep_partner", pot_preference="far", bottleneck_policy="yield"),
            "block_switch",
            20,
            curriculum_group="latent_v3_cert",
        ),
    ),
    (
        "blind-cert-bneck-patience3",
        _latent(
            "bottleneck_zone",
            ProtocolSpec(role="flexible", bottleneck_policy="alternate"),
            "patience",
            3,
            curriculum_group="latent_v3_cert",
        ),
    ),
    (
        "blind-cert-flex-block16",
        _latent(
            "flexible",
            ProtocolSpec(role="flexible", bottleneck_policy="yield", counter_preference="handoff"),
            "block_switch",
            16,
            curriculum_group="latent_v3_cert",
        ),
    ),
)


PATH_C_SYNTHETIC_IDENTITIES: tuple[tuple[str, str, ProtocolSpec], ...] = (
    (
        "ingredient-near",
        "ingredient_near",
        ProtocolSpec(role="ingredient_person", pot_preference="near"),
    ),
    (
        "ingredient-far",
        "ingredient_far",
        ProtocolSpec(role="ingredient_person", pot_preference="far"),
    ),
    (
        "prep-near",
        "prep_zone",
        ProtocolSpec(
            role="prep_partner",
            pot_preference="near",
            bottleneck_policy="yield",
        ),
    ),
    (
        "flexible-far",
        "flexible_far",
        ProtocolSpec(
            role="flexible",
            pot_preference="far",
            bottleneck_policy="yield",
        ),
    ),
    (
        "dish-nearest",
        "dish_zone",
        ProtocolSpec(
            role="dish_person",
            delivery_preference="nearest",
            bottleneck_policy="alternate",
        ),
    ),
)

PATH_C_SYNTHETIC_MECHANISMS: tuple[tuple[str, str, int], ...] = (
    ("delayed_yield", "patience", 2),
    ("defer_escalation", "escalate_after_defer", 2),
    ("reciprocal_response", "tit_for_tat", 1),
)

# Three value mechanisms, five independent surface/style realizations per
# mechanism, and both planted fingerprints in every cell. Fingerprint admission
# remains an offline equivalence test and is never inferred from this declaration.
PATH_C_SYNTHETIC_PROTOCOLS: tuple[tuple[str, LatentPartnerSpec], ...] = tuple(
    (
        f"pathc-{identity_name}-{family}{param}-fp{fingerprint_id}",
        _latent(
            geometry_profile,
            protocol,
            family,
            param,
            curriculum_group="path_c_synthetic",
            fingerprint_id=fingerprint_id,
            fingerprint_bias="retreat_order_rotate_candidate",
            value_class_id=value_class_id,
            surface_identity_key=f"pathc-{identity_name}",
            style_id=f"style-{identity_index}",
        ),
    )
    for value_class_id, family, param in PATH_C_SYNTHETIC_MECHANISMS
    for identity_index, (identity_name, geometry_profile, protocol) in enumerate(
        PATH_C_SYNTHETIC_IDENTITIES
    )
    for fingerprint_id in (0, 1)
)

# A separate validity-only pair with overlapping prior support. It is not mixed
# into the main factorial pool and cannot be used for model or threshold selection.
PATH_C_OVERLAPPING_POSTERIOR_PROTOCOLS: tuple[tuple[str, LatentPartnerSpec], ...] = (
    (
        "pathc-overlap-patience2",
        _latent(
            "overlap_shared_surface",
            ProtocolSpec(role="flexible", bottleneck_policy="yield"),
            "patience",
            2,
            epsilon=0.35,
            curriculum_group="path_c_overlap_posterior",
            value_class_id="overlap_delayed_yield",
            surface_identity_key="pathc-overlap-shared",
            style_id="overlap-style",
            overlap_posterior_group="overlap-pair-v1",
        ),
    ),
    (
        "pathc-overlap-escalate2",
        _latent(
            "overlap_shared_surface",
            ProtocolSpec(role="flexible", bottleneck_policy="yield"),
            "escalate_after_defer",
            2,
            epsilon=0.35,
            curriculum_group="path_c_overlap_posterior",
            value_class_id="overlap_defer_escalation",
            surface_identity_key="pathc-overlap-shared",
            style_id="overlap-style",
            overlap_posterior_group="overlap-pair-v1",
        ),
    ),
)


PATH_C_FINGERPRINT_NEGATIVE_PROTOCOLS: tuple[tuple[str, LatentPartnerSpec], ...] = tuple(
    (
        name.replace("pathc-", "pathc-meta-", 1),
        replace(
            spec,
            curriculum_group="path_c_fingerprint_negative",
            fingerprint_bias="metadata_only_value_null",
        ),
    )
    for name, spec in PATH_C_SYNTHETIC_PROTOCOLS
)


PARTNER_REGISTRIES: dict[str, tuple[tuple[str, ProtocolSpec | LatentPartnerSpec], ...]] = {
    "standard7": STANDARD7_PROTOCOLS,
    "role_conditioned_v1": ROLE_CONDITIONED_V1_PROTOCOLS,
    "role_conditioned_v2": ROLE_CONDITIONED_V2_PROTOCOLS,
    # Eval-only blind set (sec18.14). Never a training partner_set.
    "blind_v1": BLIND_V1_PROTOCOLS,
    # Link-A certificate substrate only. This is not Link-C's blind_v3.
    "latent_v3_dev": LATENT_V3_DEV_PROTOCOLS,
    # Candidate positive control; artifact admission decides whether it is usable.
    "path_c_synthetic": PATH_C_SYNTHETIC_PROTOCOLS,
    # Validity-only posterior-overlap stress pair; never a training registry.
    "path_c_overlap_posterior": PATH_C_OVERLAPPING_POSTERIOR_PROTOCOLS,
    # Metadata-only negative control cannot satisfy raw fingerprint visibility.
    "path_c_fingerprint_negative": PATH_C_FINGERPRINT_NEGATIVE_PROTOCOLS,
}


@dataclass
class ScriptedProtocolPartner:
    name: str
    option_library: Any
    protocol: ProtocolSpec
    partner_id: int = 0
    current_option: int | None = None
    last_state: Any | None = None
    last_primitive_action: int | None = None
    option_runtime: OptionRuntime | None = None
    elapsed: int = 0
    bottleneck_alternate_phase: int = 0

    def reset(self, seed: int) -> None:
        self.current_option = None
        self.last_state = None
        self.last_primitive_action = None
        self.option_runtime = None
        self.elapsed = 0
        self.bottleneck_alternate_phase = 0
        if hasattr(self, "_behavior_option_inferencer"):
            setattr(self, "_behavior_option_inferencer", None)

    def get_state(self) -> Mapping[str, Any]:
        """Return every mutable field that can affect future partner actions."""

        return {
            "schema_version": "path_c_scripted_partner_state_v1",
            "current_option": self.current_option,
            "last_state": copy.deepcopy(self.last_state),
            "last_primitive_action": self.last_primitive_action,
            "option_runtime": _option_runtime_to_mapping(self.option_runtime),
            "elapsed": int(self.elapsed),
            "bottleneck_alternate_phase": int(self.bottleneck_alternate_phase),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by get_state without retaining caller aliases."""

        expected = {
            "schema_version",
            "current_option",
            "last_state",
            "last_primitive_action",
            "option_runtime",
            "elapsed",
            "bottleneck_alternate_phase",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Scripted partner state has the wrong fields.")
        if state["schema_version"] != "path_c_scripted_partner_state_v1":
            raise ValueError("Scripted partner state schema version changed.")
        current_option = state["current_option"]
        last_action = state["last_primitive_action"]
        if current_option is not None and (
            isinstance(current_option, bool) or int(current_option) < 0
        ):
            raise ValueError("Scripted partner current_option is invalid.")
        if last_action is not None and (
            isinstance(last_action, bool) or int(last_action) < 0
        ):
            raise ValueError("Scripted partner last primitive action is invalid.")
        elapsed = int(state["elapsed"])
        phase = int(state["bottleneck_alternate_phase"])
        if elapsed < 0 or phase not in {0, 1}:
            raise ValueError("Scripted partner elapsed or alternate phase is invalid.")
        self.current_option = None if current_option is None else int(current_option)
        self.last_state = copy.deepcopy(state["last_state"])
        self.last_primitive_action = None if last_action is None else int(last_action)
        self.option_runtime = _option_runtime_from_mapping(state["option_runtime"])
        self.elapsed = elapsed
        self.bottleneck_alternate_phase = phase
        if hasattr(self, "_behavior_option_inferencer"):
            setattr(self, "_behavior_option_inferencer", None)

    def exact_observed_action_branches(
        self,
        state: Any,
        observed_primitive_action: int,
    ) -> tuple[tuple[Mapping[str, Any], float], ...]:
        """Enumerate hidden option draws compatible with one observed action."""

        observed = int(observed_primitive_action)
        source = copy.deepcopy(self.get_state())
        try:
            self._update_option_runtime(state, agent_id=1)
            choose_new = self.current_option is None or self._option_done(
                state,
                agent_id=1,
            )
            if choose_new:
                valid = self.option_library.valid_options(state, agent_id=1)
                distribution = option_distribution(
                    self.protocol,
                    {
                        "elapsed": int(self.elapsed),
                        "bottleneck_alternate_phase": int(
                            self.bottleneck_alternate_phase
                        ),
                        "epsilon": float(getattr(self, "option_epsilon", 0.0)),
                        "valid_options": np.asarray(valid, dtype=bool),
                    },
                    state,
                    option_library=self.option_library,
                )
                choices = tuple(
                    (int(option_id), float(distribution[int(option_id)]))
                    for option_id in np.flatnonzero(distribution > 0.0)
                )
            else:
                choices = ((None, 1.0),)
        finally:
            self.set_state(source)

        branches: list[tuple[Mapping[str, Any], float]] = []
        try:
            for choice, probability in choices:
                self.set_state(source)
                action = self.act(
                    None,
                    state,
                    _ForcedOptionChoiceRNG(choice),
                )
                if int(action.primitive_action) == observed:
                    branches.append((copy.deepcopy(self.get_state()), probability))
        finally:
            self.set_state(source)
        return tuple(branches)

    def act(self, obs_partner: Any, state: Any, rng: np.random.Generator) -> PartnerAction:
        del obs_partner

        self._update_option_runtime(state, agent_id=1)
        if self.current_option is None or self._option_done(state, agent_id=1):
            valid = self.option_library.valid_options(state, agent_id=1)
            self.current_option = self._choose_option(state, valid, rng)
            self.option_runtime = OptionRuntime(
                option_id=int(self.current_option),
                start_pos=get_agent_pos(state, 1),
            )
            self.elapsed = 0

        primitive = self.option_library.primitive_action(state, 1, self.current_option)
        self.last_state = state
        self.last_primitive_action = int(primitive)
        self.elapsed += 1
        # P1: scripted true option labels are diagnostic truth and must not enter
        # the main train/eval/CE evidence path. The shared executor infers partner
        # options from primitive action and state deltas instead.
        return PartnerAction(
            primitive_action=int(primitive),
            option_id=None,
            option_confidence=0.0,
            option_dist=None,
            source="scripted_primitive_only",
        )

    def _choose_option(
        self,
        state: Any,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        distribution = option_distribution(
            self.protocol,
            {
                "elapsed": int(self.elapsed),
                "bottleneck_alternate_phase": int(self.bottleneck_alternate_phase),
                "epsilon": float(getattr(self, "option_epsilon", 0.0)),
                "valid_options": np.asarray(valid, dtype=bool),
            },
            state,
            option_library=self.option_library,
        )
        choice = int(rng.choice(np.arange(distribution.size), p=distribution))
        chosen_kind = str(self.option_library.options[choice].kind)
        if self.protocol.bottleneck_policy == "alternate" and chosen_kind in {
            "cross_bottleneck",
            "wait_at_bottleneck",
        }:
            self.bottleneck_alternate_phase = 1 - int(self.bottleneck_alternate_phase)
        return choice

    def _protocol_score(self, opt: OptionSpec, state: Any) -> float:
        return _protocol_score_for(
            self.protocol,
            opt,
            state,
            option_library=self.option_library,
            elapsed=int(self.elapsed),
            bottleneck_alternate_phase=int(self.bottleneck_alternate_phase),
        )

    def _option_done(self, state: Any, agent_id: int) -> bool:
        opt = self.option_library.options[self.current_option]
        if self.elapsed >= opt.max_steps:
            return True
        if not self.option_library.is_valid_for_state(state, agent_id, opt.id):
            return True
        if opt.kind == "noop":
            return True
        if opt.kind == "wait_at_bottleneck":
            wait_duration = (opt.metadata or {}).get("wait_duration", 2)
            if self.option_runtime is None:
                return False
            return self.option_runtime.wait_elapsed_after_arrival >= int(wait_duration)
        if opt.kind == "cross_bottleneck":
            if self.option_runtime is None or self.last_state is None:
                return False
            return cross_bottleneck_terminated(
                self.option_runtime,
                self.last_state,
                state,
                opt,
                agent_id,
            )
        return False

    def _update_option_runtime(self, state: Any, agent_id: int) -> None:
        if self.current_option is None or self.option_runtime is None:
            return
        opt = self.option_library.options[self.current_option]
        if opt.kind == "wait_at_bottleneck" and get_agent_pos(state, agent_id) in _region_cells(opt):
            self.option_runtime.reached_region = True
            if self.last_primitive_action == int(Actions.stay):
                self.option_runtime.wait_elapsed_after_arrival += 1
        elif opt.kind == "cross_bottleneck" and self.last_state is not None:
            cross_bottleneck_terminated(
                self.option_runtime,
                self.last_state,
                state,
                opt,
                agent_id,
            )


def _option_runtime_to_mapping(runtime: OptionRuntime | None) -> Mapping[str, Any] | None:
    if runtime is None:
        return None
    return {
        "option_id": int(runtime.option_id),
        "start_pos": tuple(map(int, runtime.start_pos)),
        "elapsed": int(runtime.elapsed),
        "reached_region": bool(runtime.reached_region),
        "wait_elapsed_after_arrival": int(runtime.wait_elapsed_after_arrival),
        "entry_side": (
            None if runtime.entry_side is None else tuple(map(int, runtime.entry_side))
        ),
    }


def _option_runtime_from_mapping(value: Any) -> OptionRuntime | None:
    if value is None:
        return None
    expected = {
        "option_id",
        "start_pos",
        "elapsed",
        "reached_region",
        "wait_elapsed_after_arrival",
        "entry_side",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Partner option runtime has the wrong fields.")
    start_pos = tuple(map(int, value["start_pos"]))
    entry_side = (
        None if value["entry_side"] is None else tuple(map(int, value["entry_side"]))
    )
    if len(start_pos) != 2 or (entry_side is not None and len(entry_side) != 2):
        raise ValueError("Partner option runtime positions must be two-dimensional.")
    elapsed = int(value["elapsed"])
    wait_elapsed = int(value["wait_elapsed_after_arrival"])
    if elapsed < 0 or wait_elapsed < 0:
        raise ValueError("Partner option runtime elapsed values must be non-negative.")
    return OptionRuntime(
        option_id=int(value["option_id"]),
        start_pos=start_pos,
        elapsed=elapsed,
        reached_region=bool(value["reached_region"]),
        wait_elapsed_after_arrival=wait_elapsed,
        entry_side=entry_side,
    )


def make_training_partners(
    option_library: Any,
    partner_set: str = "standard7",
) -> list[PartnerPolicy]:
    try:
        protocols = PARTNER_REGISTRIES[str(partner_set)]
    except KeyError as exc:
        choices = ", ".join(sorted(PARTNER_REGISTRIES))
        raise ValueError(
            f"unknown partner_set {partner_set!r}; expected one of {choices}"
        ) from exc
    partners: list[PartnerPolicy] = []
    for partner_id, (name, protocol) in enumerate(protocols):
        if isinstance(protocol, LatentPartnerSpec):
            partners.append(
                LatentModeController(
                    name=name,
                    option_library=option_library,
                    spec=protocol,
                    partner_cls=ScriptedProtocolPartner,
                    partner_id=partner_id,
                )
            )
        else:
            partners.append(
                ScriptedProtocolPartner(
                    name=name,
                    option_library=option_library,
                    protocol=protocol,
                    partner_id=partner_id,
                )
            )
    return partners


def sample_mode_spec(
    rng: np.random.Generator,
    family_quotas: dict[str, float] | None = None,
) -> LatentModeSpec:
    quotas = family_quotas or {
        "static_claim": 1.0,
        "static_yield": 1.0,
        "patience": 1.0,
        "block_switch": 1.0,
        "tit_for_tat": 1.0,
        "escalate_after_defer": 1.0,
    }
    families = [str(family) for family, weight in quotas.items() if float(weight) > 0.0]
    if not families:
        raise ValueError("family_quotas selected no positive-weight families")
    weights = np.asarray([float(quotas[family]) for family in families], dtype=float)
    weights = weights / weights.sum()
    family = str(rng.choice(families, p=weights))
    epsilon = float(rng.uniform(0.05, 0.15))
    if family in {"static_claim", "static_yield"}:
        return LatentModeSpec(family=family, epsilon=epsilon)
    if family == "patience":
        return LatentModeSpec(family=family, param=int(rng.integers(2, 7)), epsilon=epsilon)
    if family == "block_switch":
        return LatentModeSpec(family=family, param=int(rng.integers(12, 26)), epsilon=epsilon)
    if family == "tit_for_tat":
        return LatentModeSpec(family=family, param=int(rng.integers(1, 4)), epsilon=epsilon)
    if family == "escalate_after_defer":
        return LatentModeSpec(family=family, param=int(rng.choice([2, 3])), epsilon=epsilon)
    raise ValueError(f"unknown latent mode family {family!r}")


def option_distribution(
    theta: ProtocolSpec | LatentPartnerSpec,
    runtime_state: Mapping[str, Any],
    public_state: Any,
    *,
    option_library: Any,
) -> np.ndarray:
    """Pure option emission shared by generation and Tier-1 inference.

    The vector is defined over the complete option library. Argmax ties are
    uniform, epsilon exploration is mixed over valid support, and an empty valid
    set is rerouted to the registered noop. No input object or random stream is
    mutated here.
    """

    if isinstance(theta, LatentPartnerSpec):
        # LatentModeController always chooses the approach option under the
        # claim-shaped public protocol; latent yield is expressed later by the
        # primitive-action retreat transition. Tier-1 inference must use this
        # exact same option-emission rule.
        protocol = replace(theta.base_protocol, terminal_policy="claim")
        default_epsilon = float(theta.mode.epsilon)
    elif isinstance(theta, ProtocolSpec):
        protocol = theta
        default_epsilon = 0.0
    else:
        raise TypeError("theta must be ProtocolSpec or LatentPartnerSpec.")
    n_options = len(option_library.options)
    if n_options <= 0:
        raise ValueError("option_library must contain at least one option.")
    option_ids = tuple(int(option.id) for option in option_library.options)
    if option_ids != tuple(range(n_options)):
        raise ValueError(
            "The frozen option library must use contiguous ids aligned with its table."
        )
    raw_valid = runtime_state.get("valid_options")
    valid = (
        np.asarray(raw_valid, dtype=bool)
        if raw_valid is not None
        else np.asarray(option_library.valid_options(public_state, agent_id=1), dtype=bool)
    )
    if valid.shape != (n_options,):
        raise ValueError(
            f"valid_options must have shape {(n_options,)}, got {valid.shape}."
        )
    probabilities = np.zeros(n_options, dtype=np.float64)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        probabilities[_noop_option_id(option_library)] = 1.0
        return probabilities
    elapsed = int(runtime_state.get("elapsed", 0))
    alternate_phase = int(runtime_state.get("bottleneck_alternate_phase", 0))
    epsilon = float(runtime_state.get("epsilon", default_epsilon))
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Option-distribution epsilon must be in [0, 1].")
    scores = np.asarray(
        [
            _protocol_score_for(
                protocol,
                option_library.options[int(option_id)],
                public_state,
                option_library=option_library,
                elapsed=elapsed,
                bottleneck_alternate_phase=alternate_phase,
            )
            for option_id in valid_ids
        ],
        dtype=np.float64,
    )
    best_local = np.flatnonzero(scores == np.max(scores))
    best_ids = valid_ids[best_local]
    probabilities[valid_ids] += epsilon / float(valid_ids.size)
    probabilities[best_ids] += (1.0 - epsilon) / float(best_ids.size)
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"Option distribution did not normalize; total={total}.")
    return probabilities


def _protocol_score_for(
    protocol: ProtocolSpec,
    opt: OptionSpec,
    state: Any,
    *,
    option_library: Any,
    elapsed: int,
    bottleneck_alternate_phase: int,
) -> float:
    score = _task_progress_score(opt, state)
    if protocol.role == "dish_person":
        score += _role_bonus(opt.kind in {"pick_plate", "plate_soup", "serve_soup"})
    elif protocol.role == "ingredient_person":
        score += _role_bonus(
            opt.kind in {"fetch_ingredient", "deliver_ingredient_to_pot"}
        )
    elif protocol.role == "server":
        score += _role_bonus(opt.kind == "serve_soup")
    elif protocol.role == "prep_partner":
        score += _role_bonus(
            opt.kind in {
                "fetch_ingredient",
                "deliver_ingredient_to_pot",
                "drop_item_to_counter",
                "clear_interaction_cell",
                "wait_at_bottleneck",
            }
        )
    score += _terminal_policy_bonus(opt, protocol.terminal_policy)
    if opt.kind == "deliver_ingredient_to_pot":
        score += _positional_preference_bonus(
            opt.target_pos, state, protocol.pot_preference, agent_id=1
        )
    if opt.kind == "fetch_ingredient" and not _fetch_has_current_pot_sink(
        option_library, opt, state
    ):
        score -= 10.0
    if opt.kind == "drop_item_to_counter" and _carrying_unusable_item(
        option_library, state, agent_id=1
    ):
        score += 100.0
    if opt.kind == "clear_interaction_cell" and _blocking_critical_cell(
        option_library, state, agent_id=1
    ):
        score += 100.0
    if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
        score += _bottleneck_bonus(
            opt,
            protocol.bottleneck_policy,
            int(elapsed),
            int(bottleneck_alternate_phase),
        )
    score += _counter_preference_bonus(opt, protocol.counter_preference)
    if opt.kind == "serve_soup":
        score += _positional_preference_bonus(
            opt.target_pos, state, protocol.delivery_preference, agent_id=1
        )
    if opt.kind == "press_recipe_button":
        score += _role_bonus(protocol.button_policy == "check_first")
    score -= _path_length_penalty(option_library, state, opt, agent_id=1)
    score -= _congestion_penalty(opt, state)
    return float(score)



def _noop_option_id(option_library: Any) -> int:
    for opt in option_library.options:
        if str(opt.kind) == "noop":
            return int(opt.id)
    raise ValueError("The frozen option library has no registered noop fallback.")



def _fetch_has_current_pot_sink(option_library: Any, opt: OptionSpec, state: Any) -> bool:
    ingredient_obj = option_library._ingredient_object_for_pile(opt)
    if ingredient_obj is None:
        return False
    for entity_id in option_library._entity_ids_by_kind("pot"):
        pot_pos = option_library.layout_graph.entities[entity_id].pos
        if pot_accepts_inventory_ingredient(
            state,
            pot_pos,
            ingredient_obj,
            require_recipe_useful=True,
        ):
            return True
    return False


def _carrying_unusable_item(option_library: Any, state: Any, agent_id: int) -> bool:
    inv = get_inventory(state, agent_id)
    if is_empty_inventory(inv):
        return False
    if not is_ingredient(inv):
        return False
    for entity_id in option_library._entity_ids_by_kind("pot"):
        pot_pos = option_library.layout_graph.entities[entity_id].pos
        if pot_accepts_inventory_ingredient(
            state,
            pot_pos,
            inv,
            require_recipe_useful=True,
        ):
            return False
    return True


def _blocking_critical_cell(option_library: Any, state: Any, agent_id: int) -> bool:
    return bool(option_library._is_blocking_critical_interaction_cell(state, agent_id))


def _task_progress_score(opt: OptionSpec, state: Any) -> float:
    inventory = get_inventory(state, 1)
    if is_empty_inventory(inventory) and opt.kind in {"fetch_ingredient", "pick_plate"}:
        return 2.0
    if opt.kind in {"deliver_ingredient_to_pot", "plate_soup", "serve_soup"}:
        return 2.0
    if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
        return 0.5
    if opt.kind == "noop":
        return -2.0
    return 0.0


def _role_bonus(condition: bool) -> float:
    """Lexicographic role-identity tier (magnitude 10^3)."""
    return 4000.0 if condition else -1000.0


def _positional_preference_bonus(
    target_pos: GridPos | None,
    state: Any,
    preference: str | None,
    agent_id: int,
) -> float:
    if target_pos is None or preference is None:
        return 0.0

    agent_pos = get_agent_pos(state, agent_id)
    distance = _manhattan(agent_pos, target_pos)
    if preference in {"near", "nearest"}:
        return -0.25 * distance
    if preference == "far":
        return 0.25 * distance
    if preference == "left":
        return -0.25 * target_pos[0]
    if preference == "right":
        return 0.25 * target_pos[0]
    return 0.0


def _bottleneck_bonus(
    opt: OptionSpec,
    policy: str | None,
    elapsed: int,
    alternate_phase: int = 0,
) -> float:
    if policy == "yield":
        # W3: bottleneck policy must be behaviorally visible within the support/
        # terminal tier, but still below the lexicographic role/terminal tier.
        return 1500.0 if opt.kind == "wait_at_bottleneck" else -500.0
    if policy == "push":
        return 1500.0 if opt.kind == "cross_bottleneck" else -500.0
    if policy == "alternate":
        # W4: alternate must alternate across option choices, not within-option
        # elapsed ticks that reset whenever a new option is selected.
        del elapsed
        prefer_cross = int(alternate_phase) % 2 == 0
        if prefer_cross and opt.kind == "cross_bottleneck":
            return 1500.0
        if not prefer_cross and opt.kind == "wait_at_bottleneck":
            return 1500.0
        if opt.kind in {"cross_bottleneck", "wait_at_bottleneck"}:
            return -500.0
    return 0.0


def _counter_preference_bonus(opt: OptionSpec, preference: str | None) -> float:
    if preference == "handoff" and opt.kind in {"handoff_counter", "drop_item_to_counter"}:
        return 3.0
    if preference == "clear" and opt.kind == "clear_interaction_cell":
        return 3.0
    if preference == "drop" and opt.kind == "drop_item_to_counter":
        return 3.0
    return 0.0


def _terminal_policy_bonus(opt: OptionSpec, policy: str | None) -> float:
    """Lexicographic role-identity tier (magnitude 10^3 - 10^4)."""
    if policy == "yield":
        if opt.kind in TERMINAL_KINDS:
            return -30000.0
        if opt.kind in PREP_KINDS or opt.kind in SUPPORT_KINDS:
            return 5000.0
        return 0.0
    if policy == "claim":
        if opt.kind in TERMINAL_KINDS:
            return 8000.0
        if opt.kind in PREP_KINDS:
            return 1000.0
        if opt.kind == "wait_at_bottleneck":
            return -2000.0
        return 0.0
    return 0.0


def _path_length_penalty(
    option_library: Any,
    state: Any,
    opt: OptionSpec,
    agent_id: int,
) -> float:
    cost = option_library.expected_cost(state, agent_id, opt.id)
    if not np.isfinite(cost):
        return 10.0
    return 0.1 * cost


def _congestion_penalty(opt: OptionSpec, state: Any) -> float:
    if opt.target_pos is None:
        return 0.0
    ego_pos = get_agent_pos(state, 0)
    partner_pos = get_agent_pos(state, 1)
    target = opt.target_pos
    # W5: penalize true target/corridor contention, not the nearly-always-true
    # condition that the partner is simply not on the ego's cell.
    if partner_pos == target:
        return 1.0
    if _manhattan(ego_pos, target) <= 1 and _manhattan(partner_pos, target) <= 1:
        return 0.5
    return 0.0


def _region_cells(opt: OptionSpec) -> tuple[GridPos, ...]:
    return tuple((opt.metadata or {}).get("region_cells", ()))


def _manhattan(a: GridPos, b: GridPos) -> int:
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
