"""Link-A C-4 mode identifiability readout.

Judged target: current latent response policy (yield vs claim), evaluated on
opportunity rows with mode_opportunity_count in [1, 8]. True mode labels are
diagnostic-only labels emitted by diag_d1_dataset; they are not method evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.overcooked_v2.scripts import diag_d1_train as D1  # noqa: E402

JUDGED_K = 5
DEFAULT_SEEDS = (0, 1, 2)


class ModeNet(nn.Module):
    def __init__(self, variant: str, n_vocab: int, state_dim: int, num_classes: int):
        super().__init__()
        self.variant = variant
        hid = 64
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hid), nn.ReLU())
        if variant == "full":
            self.kind_emb = nn.Embedding(n_vocab, 24, padding_idx=0)
            self.gru = nn.GRU(24 + 2, hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid + hid, 128),
            nn.ReLU(),
            nn.Linear(128, int(num_classes)),
        )

    def hist_repr(self, hk, hd, ha, hl):
        if self.variant != "full":
            return torch.zeros(hk.shape[0], 64, device=hk.device)
        x = torch.cat([self.kind_emb(hk), hd.unsqueeze(-1), ha.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)
        idx = torch.clamp(hl.long() - 1, min=0)
        rep = out[torch.arange(out.shape[0], device=hk.device), idx]
        return torch.where((hl > 0).unsqueeze(-1), rep, torch.zeros_like(rep))

    def forward(self, state, hk, hd, ha, hl):
        z = torch.cat([self.state_mlp(state), self.hist_repr(hk, hd, ha, hl)], dim=-1)
        return self.head(z)


def _balanced_acc(pred: np.ndarray, labels: np.ndarray) -> float:
    vals = []
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        if mask.any():
            vals.append(float((pred[mask] == cls).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _batch(data, idx, norm, device, target_col: str):
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    return {
        "state": t(D1.build_state(data, idx, norm), torch.float32),
        "hk": t(data["hist_kind"][idx], torch.long),
        "hd": t(data["hist_dur"][idx], torch.float32),
        "ha": t(data["hist_ago"][idx], torch.float32),
        "hl": t(data["hist_len"][idx], torch.long),
        "labels": t(data[target_col][idx], torch.long),
    }


def _predict(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            logits = model(
                batch["state"][sl],
                batch["hk"][sl],
                batch["hd"][sl],
                batch["ha"][sl],
                batch["hl"][sl],
            )
            outs.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.int64)


def _train_one(
    data,
    train_idx,
    val_idx,
    variant: str,
    seed: int,
    device: torch.device,
    *,
    target_col: str,
    num_classes: int,
):
    torch.manual_seed(seed)
    norm_x = D1.build_state(data, train_idx)
    norm = (norm_x.mean(0), norm_x.std(0) + 1e-6)
    state_dim = norm_x.shape[1]
    n_vocab = len(data["_vocab"])
    b_tr = _batch(data, train_idx, norm, device, target_col)
    b_va = _batch(data, val_idx if val_idx.size else train_idx, norm, device, target_col)
    model = ModeNet(variant, n_vocab, state_dim, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = b_tr["state"].shape[0]
    best, best_state, patience = float("inf"), None, 0
    for _epoch in range(35):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            sl = perm[i:i + 1024]
            logits = model(
                b_tr["state"][sl],
                b_tr["hk"][sl],
                b_tr["hd"][sl],
                b_tr["ha"][sl],
                b_tr["hl"][sl],
            )
            loss = nn.functional.cross_entropy(logits, b_tr["labels"][sl])
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            logits = model(b_va["state"], b_va["hk"], b_va["hd"], b_va["ha"], b_va["hl"])
            va_loss = float(nn.functional.cross_entropy(logits, b_va["labels"]).item())
        if va_loss < best - 1e-4:
            best = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, best


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {str(int(k)): int((values == k).sum()) for k in sorted(np.unique(values))}


def _aux_history_readout(
    data,
    base_mask: np.ndarray,
    target_col: str,
    seeds: list[int],
    device: torch.device,
) -> dict[str, Any]:
    mask = base_mask & (data[target_col] >= 0)
    train_idx = np.flatnonzero(mask & (data["split"] == "train") & ~data["indist_val"])
    val_idx = np.flatnonzero(mask & data["indist_val"])
    blind_idx = np.flatnonzero(mask & (data["split"] == "blind"))
    if train_idx.size == 0 or blind_idx.size == 0:
        return {"status": "empty", "train": int(train_idx.size), "blind": int(blind_idx.size)}
    num_classes = int(np.max(data[target_col][train_idx])) + 1
    rows = []
    for seed in seeds:
        model, norm, best_loss = _train_one(
            data,
            train_idx,
            val_idx,
            "full",
            seed,
            device,
            target_col=target_col,
            num_classes=num_classes,
        )
        b_blind = _batch(data, blind_idx, norm, device, target_col)
        pred = _predict(model, b_blind)
        labels = b_blind["labels"].cpu().numpy()
        rows.append({
            "seed": int(seed),
            "val_loss": float(best_loss),
            "blind_balanced_acc": _balanced_acc(pred, labels),
            "blind_accuracy": float((pred == labels).mean()) if labels.size else float("nan"),
        })
    return {
        "status": "ok",
        "target_col": target_col,
        "class_counts_blind": _class_counts(data[target_col][blind_idx]),
        "per_seed": rows,
        "blind_balanced_acc_median": float(np.median([r["blind_balanced_acc"] for r in rows])),
        "blind_accuracy_median": float(np.median([r["blind_accuracy"] for r in rows])),
    }


def run(args: argparse.Namespace) -> None:
    data = D1.load_chunks(args.chunks)
    required = {"mode_policy_id", "mode_family_id", "mode_opportunity_count"}
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"missing Link-A diagnostic columns: {missing}")
    ok = D1.ok_rows(data, JUDGED_K)
    base = (
        data["gate_main"].astype(bool)
        & ok
        & (data["mode_policy_id"] >= 0)
        & (data["mode_opportunity_count"] >= 1)
        & (data["mode_opportunity_count"] <= int(args.max_observations))
    )
    train = base & (data["split"] == "train") & ~data["indist_val"]
    val = base & data["indist_val"]
    blind = base & (data["split"] == "blind")
    train_idx = np.flatnonzero(train)
    val_idx = np.flatnonzero(val)
    blind_idx = np.flatnonzero(blind)
    if train_idx.size == 0 or blind_idx.size == 0:
        raise RuntimeError(
            f"insufficient rows for mode cert: train={train_idx.size} blind={blind_idx.size}"
        )

    device = torch.device(args.device)
    out: dict = {
        "diag": "Link-A C-4 mode identifiability",
        "judged_target": "mode_policy_id (yield=0, claim=1)",
        "max_observations": int(args.max_observations),
        "row_counts": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "blind": int(blind_idx.size),
        },
        "class_counts": {
            "train_policy": _class_counts(data["mode_policy_id"][train_idx]),
            "blind_policy": _class_counts(data["mode_policy_id"][blind_idx]),
            "blind_family": _class_counts(data["mode_family_id"][blind_idx]),
            "blind_param": _class_counts(data["mode_param"][blind_idx])
            if "mode_param" in data
            else {},
        },
        "variants": {},
    }
    seeds = [int(s) for s in args.seeds.split(",") if str(s).strip()]
    for variant, label in (("full", "history_full"), ("nohist", "state_only")):
        rows = []
        for seed in seeds:
            model, norm, best_loss = _train_one(
                data,
                train_idx,
                val_idx,
                variant,
                seed,
                device,
                target_col="mode_policy_id",
                num_classes=2,
            )
            b_blind = _batch(data, blind_idx, norm, device, "mode_policy_id")
            pred = _predict(model, b_blind)
            labels = b_blind["labels"].cpu().numpy()
            rows.append({
                "seed": int(seed),
                "val_loss": float(best_loss),
                "blind_balanced_acc": _balanced_acc(pred, labels),
                "blind_accuracy": float((pred == labels).mean()) if labels.size else float("nan"),
            })
        out["variants"][label] = {
            "per_seed": rows,
            "blind_balanced_acc_mean": float(np.mean([r["blind_balanced_acc"] for r in rows])),
            "blind_balanced_acc_median": float(np.median([r["blind_balanced_acc"] for r in rows])),
        }

    full_acc = float(out["variants"]["history_full"]["blind_balanced_acc_median"])
    state_acc = float(out["variants"]["state_only"]["blind_balanced_acc_median"])
    out["thresholds"] = {
        "history_full_balanced_acc_min": 0.75,
        "state_only_balanced_acc_max": 0.60,
    }
    out["criteria"] = {
        "history_full_pass": bool(full_acc >= 0.75),
        "state_only_hidden_pass": bool(state_acc <= 0.60),
        "PASS": bool(full_acc >= 0.75 and state_acc <= 0.60),
    }
    out["co_report"] = {
        "family_and_param_are_not_judged_targets": True,
        "family_class_counts_blind": out["class_counts"]["blind_family"],
        "param_class_counts_blind": out["class_counts"]["blind_param"],
        "family_history_full": _aux_history_readout(
            data, base, "mode_family_id", seeds, device,
        ),
        "param_history_full": _aux_history_readout(
            data, base, "mode_param", seeds, device,
        ) if "mode_param" in data else {"status": "missing"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["criteria"], indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-observations", type=int, default=8)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
