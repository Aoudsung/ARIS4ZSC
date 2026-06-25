from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


EVENT_FEATURE_DIM = 16


class PartnerOptionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_options: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_options = int(num_options)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.num_options),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [B, {self.input_dim}]; got "
                f"{tuple(features.shape)}."
            )
        return self.net(features.float())


def event_feature_vector(event: Any) -> np.ndarray:
    values = np.asarray(
        [
            float(getattr(event, "partner_action", 0)),
            float(bool(getattr(event, "partner_waited", False))),
            float(bool(getattr(event, "partner_interacted", False))),
            float(bool(getattr(event, "collision_or_block", False))),
            float(bool(getattr(event, "partner_inventory_before", 0) != getattr(event, "partner_inventory_after", 0))),
            float(getattr(event, "partner_option_confidence", 0.0)),
            float(bool(getattr(event, "pot_changed", False))),
            float(bool(getattr(event, "pot_became_ready", False))),
            float(bool(getattr(event, "plate_picked", False))),
            float(bool(getattr(event, "soup_picked", False))),
            float(bool(getattr(event, "delivery_event", False))),
            float(bool(getattr(event, "recipe_indicator_event", False))),
            float(bool(getattr(event, "button_pressed", False))),
            float(_delta_x(getattr(event, "partner_pos_before", (0, 0)), getattr(event, "partner_pos_after", (0, 0)))),
            float(_delta_y(getattr(event, "partner_pos_before", (0, 0)), getattr(event, "partner_pos_after", (0, 0)))),
            float(len(getattr(event, "changed_cells", ()))),
        ],
        dtype=np.float32,
    )
    return values


def load_partner_option_classifier(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PartnerOptionClassifier:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    input_dim = int(checkpoint.get("input_dim", EVENT_FEATURE_DIM))
    num_options = int(checkpoint["num_options"])
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    model = PartnerOptionClassifier(input_dim, num_options, hidden_dim)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise ValueError(
            "Partner option classifier checkpoint must contain 'state_dict' or "
            "'model_state_dict'."
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def classifier_action(
    model: PartnerOptionClassifier,
    event: Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[int, float, np.ndarray]:
    features = torch.as_tensor(
        event_feature_vector(event)[None, :],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        probs = torch.softmax(model(features), dim=-1).squeeze(0).cpu().numpy()
    option_id = int(np.argmax(probs))
    return option_id, float(probs[option_id]), probs.astype(np.float32)


def _delta_x(before: tuple[int, int], after: tuple[int, int]) -> int:
    return int(after[0]) - int(before[0])


def _delta_y(before: tuple[int, int], after: tuple[int, int]) -> int:
    return int(after[1]) - int(before[1])
