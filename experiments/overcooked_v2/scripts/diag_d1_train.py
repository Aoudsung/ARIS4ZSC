"""D1 supervised trainer + frozen readout (prereg D1 §4-§6; codex-review revised).

Offline models over diag_d1_dataset chunks; ZERO parameter sharing with the main method.

Stages (run in order; later stages refuse to peek earlier):
  gate    : merge chunks -> dataset_gate_report.json (sufficiency floors, class rates,
            ambiguity drop rates, cross-partner history-variance wiring check).
            Sign-B material. Never looks at any transfer metric.
            Floors are HARD on the two judged splits (indist_val, blind_terminal) and
            reported (soft) on co-report splits (indist_train, dev, blind_offaxis).
  tune    : train FULL / NOHIST (x K in {3,5,8}) + IDORACLE (K=5) x 3 seeds on training
            partners only (main-gate onset rows, ambiguous-window rows dropped). Reports
            in-distribution metrics ONLY; dev/blind rows are physically excluded from
            the tensors this stage builds.
  readout : SINGLE-LOOK. Requires dataset_gate_report.json PASS. Loads frozen models,
            computes all splits once, writes readout_frozen.json (refuses to overwrite).

Frozen primary judgment (prereg §5): K=5, gate_main, ensemble(3 seeds) probabilities.
  AUC_ps = AUC(partner_serves vs rest);  G = AUC_ps(FULL) - AUC_ps(NOHIST)
  R1: G_indist >= 0.10 AND bootstrap CI(Δlogloss NOHIST-FULL) lower > 0   [indist_val]
  R2: G_blind_terminal >= 0.5 * G_indist AND bootstrap CI(G_blind) lower > 0
  R3 (secondary): NMI(hidden clusters, behavior family) > NMI(hidden clusters, partner id)
  K-sensitivity (prereg §8.3): independent ensembles per K; direction must agree.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

JUDGED_K = 5
K_SET = (3, 5, 8)
N_SEEDS = 3
BOOT_ITERS = 10_000
FLOOR_GATED = 5000
FLOOR_MINCLASS = 500
R1_GAIN_FLOOR = 0.10
R2_RETENTION = 0.5
LABEL_NAMES = ("no_initiation", "ego_first", "partner_first")  # D1-rev (prereg §9)
HARD_FLOOR_SPLITS = ("indist_val", "blind_terminal")


def family_of(partner: str) -> str:
    if partner.startswith("latent-") or partner.startswith("blind-cert-"):
        return "latent"
    if "claim" in partner:
        return "claim"
    if "yield" in partner:
        return "yield"
    return "offaxis"


def split_of(partner: str) -> str:
    if partner.startswith("blind-"):
        return "blind"
    if partner.startswith("heldout-"):
        return "dev"
    return "train"


# ---------------------------------------------------------------- data loading

def load_chunks(chunk_dir: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(chunk_dir) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks under {chunk_dir}")
    cols: dict[str, list] = {}
    partners, egos, vocab = [], [], None
    for f in files:
        meta = json.loads(Path(f).with_suffix(".meta.json").read_text())
        if int(meta["oracle_source_count"]) != 0:
            raise RuntimeError(f"chunk {f} has oracle-like sources — wiring violation")
        if vocab is None:
            vocab = meta["kind_vocab"]
        elif vocab != meta["kind_vocab"]:
            raise RuntimeError(f"kind vocab mismatch in {f}")
        z = np.load(f)
        n = int(z["episode_id"].shape[0])
        for k in z.files:
            cols.setdefault(k, []).append(z[k])
        partners.append(np.array([meta["partner"]] * n))
        egos.append(np.array([meta["ego"]] * n))
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["partner"] = np.concatenate(partners)
    data["ego"] = np.concatenate(egos)
    data["_vocab"] = np.array(vocab)
    data["family"] = np.array([family_of(p) for p in data["partner"]])
    data["split"] = np.array([split_of(p) for p in data["partner"]])
    # deterministic in-distribution episode split: 20% val by episode index
    data["indist_val"] = (data["episode_id"] % 5 == 4) & (data["split"] == "train")
    return data


def ok_rows(data, k: int) -> np.ndarray:
    """Rows usable for window K: ambiguous-first-delivery windows are excluded."""
    return ~data[f"ambig_k{k}"].astype(bool)


def ep_keys(data, mask) -> np.ndarray:
    return np.char.add(
        np.char.add(data["partner"][mask].astype(str),
                    np.char.add("|", data["ego"][mask].astype(str))),
        np.char.add("|", data["episode_id"][mask].astype(str)),
    )


# ------------------------------------------------------------------- metrics

def auc_binary(scores: np.ndarray, positives: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling (average ranks)."""
    pos, neg = scores[positives], scores[~positives]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    sorted_c = np.sort(combined, kind="mergesort")
    uniq, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    starts = np.searchsorted(sorted_c, uniq, side="left") + 1
    avg = starts + (counts - 1) / 2.0
    ranks = avg[inv]
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def metric_block(p_full, p_nohist, labels) -> dict:
    pos = labels == 2
    a_f = auc_binary(p_full[:, 2], pos)
    a_n = auc_binary(p_nohist[:, 2], pos)
    return {
        "n": int(labels.size),
        "class_counts": {LABEL_NAMES[i]: int((labels == i).sum()) for i in range(3)},
        "auc_ps_full": a_f,
        "auc_ps_nohist": a_n,
        "gain": a_f - a_n,
        "logloss_full": logloss(p_full, labels),
        "logloss_nohist": logloss(p_nohist, labels),
    }


def bootstrap_ci(stat_fn, keys: np.ndarray, iters=BOOT_ITERS, seed=0) -> dict:
    """Episode-block bootstrap, stratified by partner (first key field)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    strata: dict[str, list] = {}
    for k in uniq:
        strata.setdefault(str(k).split("|")[0], []).append(k)
    key_to_rows = {k: np.flatnonzero(keys == k) for k in uniq}
    vals = np.empty(iters, dtype=np.float64)
    for it in range(iters):
        idx_parts = []
        for _, ks in strata.items():
            pick = rng.integers(0, len(ks), size=len(ks))
            idx_parts.extend(key_to_rows[ks[j]] for j in pick)
        idx = np.concatenate(idx_parts)
        vals[it] = stat_fn(idx)
    return {"lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "mean": float(vals.mean())}


# -------------------------------------------------------------------- models

class D1Net(nn.Module):
    def __init__(self, variant: str, n_vocab: int, state_dim: int, n_partners: int):
        super().__init__()
        self.variant = variant
        hid = 64
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hid), nn.ReLU())
        self.opt_emb = nn.Embedding(n_vocab, 16)
        if variant == "full":
            self.kind_emb = nn.Embedding(n_vocab, 24, padding_idx=0)
            self.gru = nn.GRU(24 + 2, hid, batch_first=True)
        elif variant == "idoracle":
            self.partner_emb = nn.Embedding(max(n_partners, 1), hid)
        self.head = nn.Sequential(
            nn.Linear(hid + hid + 16, 128), nn.ReLU(), nn.Linear(128, 3))

    def hist_repr(self, hk, hd, ha, hl):
        if self.variant != "full":
            raise RuntimeError("hist_repr only defined for FULL")
        x = torch.cat([self.kind_emb(hk), hd.unsqueeze(-1), ha.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)
        idx = torch.clamp(hl.long() - 1, min=0)
        rep = out[torch.arange(out.shape[0]), idx]
        return torch.where((hl > 0).unsqueeze(-1), rep, torch.zeros_like(rep))

    def forward(self, state, hk, hd, ha, hl, ego_opt, partner_idx):
        if self.variant == "full":
            h = self.hist_repr(hk, hd, ha, hl)
        elif self.variant == "idoracle":
            h = self.partner_emb(torch.clamp(partner_idx.long(), min=0))
        else:  # nohist
            h = torch.zeros(state.shape[0], 64, device=state.device)
        z = torch.cat([h, self.state_mlp(state), self.opt_emb(ego_opt.long())], dim=-1)
        return self.head(z)


def build_state(data, idx, norm=None):
    x = np.concatenate([
        data["state_feat"][idx], data["extra_feat"][idx],
        data["valid_kinds"][idx].astype(np.float32),
    ], axis=1).astype(np.float32)
    if norm is not None:
        x = (x - norm[0]) / norm[1]
    return x


def tensors_for(data, idx, norm, partner_to_idx, device, k: int):
    pid = np.array([partner_to_idx.get(p, -1) for p in data["partner"][idx]])
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    return dict(
        state=t(build_state(data, idx, norm), torch.float32),
        hk=t(data["hist_kind"][idx], torch.long),
        hd=t(data["hist_dur"][idx], torch.float32),
        ha=t(data["hist_ago"][idx], torch.float32),
        hl=t(data["hist_len"][idx], torch.long),
        ego_opt=t(data["ego_opt_kind"][idx], torch.long),
        partner_idx=t(pid, torch.long),
        labels=t(data[f"label_k{k}"][idx], torch.long),
    )


def predict(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            logits = model(batch["state"][sl], batch["hk"][sl], batch["hd"][sl],
                           batch["ha"][sl], batch["hl"][sl], batch["ego_opt"][sl],
                           batch["partner_idx"][sl])
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outs)


def hidden_of(model, batch, bs=8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        n = batch["state"].shape[0]
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            outs.append(model.hist_repr(batch["hk"][sl], batch["hd"][sl],
                                        batch["ha"][sl], batch["hl"][sl]).cpu().numpy())
    return np.concatenate(outs)


# -------------------------------------------------------------------- stages

def stage_gate(data, out_dir: Path) -> None:
    rep: dict = {"floors": {"gated_rows": FLOOR_GATED, "min_class": FLOOR_MINCLASS,
                            "hard_floor_splits": list(HARD_FLOOR_SPLITS)},
                 "splits": {}, "per_partner": {}}
    lab = data[f"label_k{JUDGED_K}"]
    gm = data["gate_main"].astype(bool) & ok_rows(data, JUDGED_K)
    named = {
        "indist_val": gm & data["indist_val"],
        "blind_terminal": gm & (data["split"] == "blind") & (data["family"] != "offaxis"),
        "indist_train": gm & (data["split"] == "train") & ~data["indist_val"],
        "dev": gm & (data["split"] == "dev"),
        "blind_offaxis": gm & (data["split"] == "blind") & (data["family"] == "offaxis"),
    }
    for name, mask in named.items():
        counts = {LABEL_NAMES[i]: int((lab[mask] == i).sum()) for i in range(3)}
        entry = {
            "n_gate_main": int(mask.sum()),
            "class_counts": counts,
            "floor_rows_pass": bool(mask.sum() >= FLOOR_GATED),
            "floor_class_pass": bool(min(counts.values()) >= FLOOR_MINCLASS),
            "hard_floor": name in HARD_FLOOR_SPLITS,
            "co_report_only": name not in HARD_FLOOR_SPLITS,
        }
        rep["splits"][name] = entry
    rep["ambig_drop_rate"] = {
        f"k{k}": float(data[f"ambig_k{k}"].astype(bool).mean()) for k in K_SET}
    rep["censoring_rate_k5"] = float(data[f"censored_k{JUDGED_K}"].astype(bool).mean())
    for p in np.unique(data["partner"]):
        m = (data["partner"] == p) & gm
        rep["per_partner"][str(p)] = {
            "n_gate_main": int(m.sum()),
            "episodes": int(np.unique(ep_keys(data, data["partner"] == p)).size),
            "class_counts": {LABEL_NAMES[i]: int((lab[m] == i).sum()) for i in range(3)},
        }
    # wiring check: does the partner-history channel actually vary across partners?
    # Criterion (amended 2026-07-07, gate round 2): positional twin partners (e.g.
    # server-left-claim vs server-right-claim, differing only in delivery position)
    # are IDENTICAL at option-kind granularity by construction, so pairwise-min TV
    # is the wrong operationalization. The prereg intent ("通道随伙伴变化") is that
    # each partner's stream differs from at least one other; kind-collapsed twin
    # pairs are recorded informationally (they share family/response semantics).
    names = sorted(np.unique(data["partner"][data["split"] == "train"]).tolist())
    n_vocab = len(data["_vocab"])
    hists = {}
    for p in names:
        m = data["partner"] == p
        h = np.bincount(data["hist_kind"][m].ravel(), minlength=n_vocab).astype(float)
        h[0] = 0.0  # drop PAD
        hists[p] = h / max(h.sum(), 1.0)
    tv_pairs = {(a, b): float(0.5 * np.abs(hists[a] - hists[b]).sum())
                for i, a in enumerate(names) for b in names[i + 1:]}
    tv = list(tv_pairs.values())
    per_partner_max = {
        a: max(tv_pairs.get((a, b), tv_pairs.get((b, a), 0.0))
               for b in names if b != a) for a in names}
    rep["history_channel_cross_partner_tv"] = {
        "min": min(tv), "max": max(tv), "mean": float(np.mean(tv))}
    rep["kind_collapsed_pairs"] = [
        {"pair": list(k), "tv": v} for k, v in tv_pairs.items() if v < 0.01]
    rep["per_partner_max_tv"] = per_partner_max
    rep["history_channel_varies"] = bool(min(per_partner_max.values()) > 0.05)
    rep["PASS"] = bool(
        all(rep["splits"][s]["floor_rows_pass"] and rep["splits"][s]["floor_class_pass"]
            for s in HARD_FLOOR_SPLITS) and rep["history_channel_varies"])
    rep["soft_floor_failures"] = [
        s for s, e in rep["splits"].items()
        if e["co_report_only"] and not (e["floor_rows_pass"] and e["floor_class_pass"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_gate_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"PASS": rep["PASS"],
                      "soft_floor_failures": rep["soft_floor_failures"],
                      **{k: {kk: v[kk] for kk in ("n_gate_main", "class_counts")}
                         for k, v in rep["splits"].items()}}, indent=1), flush=True)


def _train_one(variant, k, seed, b_tr, b_va, n_vocab, state_dim, n_partners, dev):
    torch.manual_seed(seed)
    model = D1Net(variant, n_vocab, state_dim, n_partners).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = b_tr["state"].shape[0]
    best, best_state, patience = float("inf"), None, 0
    for _epoch in range(40):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            sl = perm[i:i + 1024]
            logits = model(b_tr["state"][sl], b_tr["hk"][sl], b_tr["hd"][sl],
                           b_tr["ha"][sl], b_tr["hl"][sl], b_tr["ego_opt"][sl],
                           b_tr["partner_idx"][sl])
            loss = nn.functional.cross_entropy(logits, b_tr["labels"][sl])
            optim.zero_grad(); loss.backward(); optim.step()
        p_va = predict(model, b_va)
        ll = logloss(p_va, b_va["labels"].cpu().numpy())
        if ll < best - 1e-4:
            best, patience = ll, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    return model, best


def stage_tune(data, out_dir: Path, device: str) -> None:
    # physically restrict to training partners before anything else (no-peek guarantee)
    rows = np.flatnonzero(data["split"] == "train")
    n_total = data["episode_id"].shape[0]
    sub = {k: (v[rows] if isinstance(v, np.ndarray) and v.shape[:1] == (n_total,) else v)
           for k, v in data.items()}
    train_partners = sorted(np.unique(sub["partner"]).tolist())
    partner_to_idx = {p: i for i, p in enumerate(train_partners)}
    gmain = sub["gate_main"].astype(bool)
    dev = torch.device(device)
    n_vocab = len(sub["_vocab"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"train_partners": train_partners, "k_blocks": {}}

    for k in K_SET:
        ok = ok_rows(sub, k)
        tr = ~sub["indist_val"] & gmain & ok
        va = sub["indist_val"] & gmain & ok
        va_main = va  # D1-rev: training gate == judged gate (strict gate is co-report)
        # norm stats from exactly this K's training rows (ambig-filtered; codex re-review)
        norm_x = build_state(sub, np.flatnonzero(tr))
        norm = (norm_x.mean(0), norm_x.std(0) + 1e-6)
        state_dim = norm_x.shape[1]
        b_tr = tensors_for(sub, np.flatnonzero(tr), norm, partner_to_idx, dev, k)
        b_va = tensors_for(sub, np.flatnonzero(va), norm, partner_to_idx, dev, k)
        b_va_main = b_va
        variants = ("full", "nohist", "idoracle") if k == JUDGED_K else ("full", "nohist")
        blk: dict = {"n_train": int(tr.sum()), "n_val_main": int(va.sum()),
                     "variants": {}}
        probs_main: dict[str, list[np.ndarray]] = {}
        for variant in variants:
            blk["variants"][variant] = []
            for seed in range(N_SEEDS):
                model, best_ll = _train_one(variant, k, seed, b_tr, b_va, n_vocab,
                                            state_dim, len(train_partners), dev)
                path = out_dir / f"model_{variant}_k{k}_s{seed}.pt"
                torch.save({"state_dict": model.state_dict(), "variant": variant,
                            "k": k, "n_vocab": n_vocab, "state_dim": state_dim,
                            "n_partners": len(train_partners),
                            "norm_mean": norm[0], "norm_std": norm[1],
                            "train_partners": train_partners}, path)
                pm = predict(model, b_va_main)
                probs_main.setdefault(variant, []).append(pm)
                blk["variants"][variant].append({
                    "seed": seed, "val_logloss": best_ll,
                    "val_main_logloss": logloss(pm, b_va_main["labels"].cpu().numpy()),
                    "val_main_auc_ps": auc_binary(
                        pm[:, 2], b_va_main["labels"].cpu().numpy() == 2),
                })
        lab_main = b_va_main["labels"].cpu().numpy()
        ens = {v: np.mean(probs_main[v], axis=0) for v in probs_main}
        blk["indist_val_main_ensemble"] = metric_block(
            ens["full"], ens["nohist"], lab_main)
        if "idoracle" in ens:
            blk["indist_val_main_ensemble"]["auc_ps_idoracle"] = auc_binary(
                ens["idoracle"][:, 2], lab_main == 2)
        report["k_blocks"][f"k{k}"] = blk
    (out_dir / "tune_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v["indist_val_main_ensemble"]
                      for k, v in report["k_blocks"].items()}, indent=1), flush=True)


def _load_models(out_dir: Path, device: str, k: int):
    models: dict[str, list] = {"full": [], "nohist": []}
    norm = partner_to_idx = None
    shas = {}
    for variant in ("full", "nohist"):
        for seed in range(N_SEEDS):
            path = out_dir / f"model_{variant}_k{k}_s{seed}.pt"
            ck = torch.load(path, map_location=device, weights_only=False)
            m = D1Net(variant, ck["n_vocab"], ck["state_dim"], ck["n_partners"])
            m.load_state_dict(ck["state_dict"])
            m.to(device)
            models[variant].append(m)
            shas[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            norm = (ck["norm_mean"], ck["norm_std"])
            partner_to_idx = {p: i for i, p in enumerate(ck["train_partners"])}
    return models, norm, partner_to_idx, shas


def _eval_pack(data, mask, models, norm, partner_to_idx, device, k):
    idx = np.flatnonzero(mask)
    if idx.size == 0:  # strict onset gates can empty a co-report split (codex rev)
        empty = np.zeros((0, 3), dtype=np.float64)
        return idx, None, np.zeros(0, dtype=np.int64), {v: empty for v in models}
    batch = tensors_for(data, idx, norm, partner_to_idx, torch.device(device), k)
    labels = batch["labels"].cpu().numpy()
    probs = {v: np.mean([predict(m, batch) for m in models[v]], axis=0)
             for v in models}
    return idx, batch, labels, probs


def stage_readout(data, out_dir: Path, device: str) -> None:
    out_path = out_dir / "readout_frozen.json"
    if out_path.exists():
        raise RuntimeError("readout_frozen.json exists — single look already consumed")
    gate_path = out_dir / "dataset_gate_report.json"
    if not gate_path.exists():
        raise RuntimeError("dataset_gate_report.json missing — run --stage gate first")
    gate = json.loads(gate_path.read_text())
    if not gate.get("PASS"):
        raise RuntimeError(
            "sufficiency gate FAILED — no readout permitted on insufficient samples "
            f"(prereg §5); gate says: { {s: gate['splits'][s] for s in HARD_FLOOR_SPLITS} }")

    models, norm, partner_to_idx, shas = _load_models(out_dir, device, JUDGED_K)
    ok5 = ok_rows(data, JUDGED_K)
    gm = data["gate_main"].astype(bool) & ok5
    gs = data["gate_strict"].astype(bool) & ok5
    res: dict = {"judged_k": JUDGED_K, "model_sha256": shas,
                 "gate_report_pass": True, "splits": {}}

    def add(name, mask, boot=False):
        idx, batch, labels, probs = _eval_pack(
            data, mask, models, norm, partner_to_idx, device, JUDGED_K)
        if idx.size == 0:
            blk = {"n": 0, "note": "empty mask under onset gate"}
            res["splits"][name] = blk
            return blk
        blk = metric_block(probs["full"], probs["nohist"], labels)
        blk["censoring_rate"] = float(data[f"censored_k{JUDGED_K}"][idx].mean()) \
            if idx.size else float("nan")
        if boot and labels.size:
            keys = ep_keys(data, mask)
            pos = labels == 2
            blk["gain_ci"] = bootstrap_ci(
                lambda ii: auc_binary(probs["full"][ii, 2], pos[ii])
                - auc_binary(probs["nohist"][ii, 2], pos[ii]), keys)
            blk["dlogloss_ci"] = bootstrap_ci(
                lambda ii: logloss(probs["nohist"][ii], labels[ii])
                - logloss(probs["full"][ii], labels[ii]), keys)
        res["splits"][name] = blk
        return blk

    iv = add("indist_val_main", gm & data["indist_val"], boot=True)
    add("indist_val_strict", gs & data["indist_val"])
    add("dev_main", gm & (data["split"] == "dev"))
    bt = add("blind_terminal_main",
             gm & (data["split"] == "blind") & (data["family"] != "offaxis"), boot=True)
    add("blind_terminal_strict",
        gs & (data["split"] == "blind") & (data["family"] != "offaxis"))
    add("blind_offaxis_main",
        gm & (data["split"] == "blind") & (data["family"] == "offaxis"))
    for fam in ("yield", "claim"):
        add(f"blind_{fam}_main", gm & (data["split"] == "blind") & (data["family"] == fam))
    for p in sorted(np.unique(data["partner"][data["split"] == "blind"]).tolist()):
        add(f"partner::{p}", gm & (data["partner"] == p))
    for ego in sorted(np.unique(data["ego"]).tolist()):
        add(f"blind_terminal_ego::{ego}",
            gm & (data["split"] == "blind") & (data["family"] != "offaxis")
            & (data["ego"] == ego))

    # K-sensitivity: independent ensembles per K (codex finding 5)
    res["k_sensitivity"] = {}
    for k in K_SET:
        if k == JUDGED_K:
            res["k_sensitivity"][f"k{k}"] = {
                "G_indist": iv["gain"], "G_blind_terminal": bt["gain"]}
            continue
        mk, nk, pk, _ = _load_models(out_dir, device, k)
        okk = ok_rows(data, k)
        gmk = data["gate_main"].astype(bool) & okk
        _, _, lab_i, pr_i = _eval_pack(
            data, gmk & data["indist_val"], mk, nk, pk, device, k)
        _, _, lab_b, pr_b = _eval_pack(
            data, gmk & (data["split"] == "blind") & (data["family"] != "offaxis"),
            mk, nk, pk, device, k)
        res["k_sensitivity"][f"k{k}"] = {
            "G_indist": (auc_binary(pr_i["full"][:, 2], lab_i == 2)
                         - auc_binary(pr_i["nohist"][:, 2], lab_i == 2))
            if lab_i.size else float("nan"),
            "G_blind_terminal": (auc_binary(pr_b["full"][:, 2], lab_b == 2)
                                 - auc_binary(pr_b["nohist"][:, 2], lab_b == 2))
            if lab_b.size else float("nan"),
        }
    signs_i = {np.sign(v["G_indist"]) for v in res["k_sensitivity"].values()}
    signs_b = {np.sign(v["G_blind_terminal"]) for v in res["k_sensitivity"].values()}
    res["k_sensitivity"]["direction_consistent"] = bool(
        len(signs_i) == 1 and len(signs_b) == 1)

    # R3: semantics vs fingerprint on blind gate_main points (K=5 FULL models)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score as nmi
        bm = gm & (data["split"] == "blind")
        idx = np.flatnonzero(bm)
        batch = tensors_for(data, idx, norm, partner_to_idx,
                            torch.device(device), JUDGED_K)
        fam_lab = np.searchsorted(np.unique(data["family"][bm]), data["family"][bm])
        pid_lab = np.searchsorted(np.unique(data["partner"][bm]), data["partner"][bm])
        nmis_fam, nmis_pid = [], []
        for m in models["full"]:
            h = hidden_of(m, batch)
            cl = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(h)
            nmis_fam.append(float(nmi(fam_lab, cl)))
            nmis_pid.append(float(nmi(pid_lab, cl)))
        res["r3_nmi"] = {"family_median": float(np.median(nmis_fam)),
                         "partner_id_median": float(np.median(nmis_pid)),
                         "per_seed_family": nmis_fam, "per_seed_partner": nmis_pid,
                         "pass": bool(np.median(nmis_fam) > np.median(nmis_pid))}
    except Exception as exc:  # noqa: BLE001
        res["r3_nmi"] = {"error": str(exc)}

    g_ind, g_bld = iv["gain"], bt["gain"]
    res["criteria"] = {
        "G_indist": g_ind,
        "G_blind_terminal": g_bld,
        "R1_pass": bool(g_ind >= R1_GAIN_FLOOR and iv["dlogloss_ci"]["lo"] > 0),
        "R2_pass": bool(g_bld >= R2_RETENTION * g_ind and bt["gain_ci"]["lo"] > 0),
        "k_direction_consistent": res["k_sensitivity"]["direction_consistent"],
        "thresholds": {"R1_gain_floor": R1_GAIN_FLOOR, "R2_retention": R2_RETENTION},
    }
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["criteria"], indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="directory of diag_d1 npz chunks")
    ap.add_argument("--out", required=True, help="output dir (models + reports)")
    ap.add_argument("--stage", required=True, choices=("gate", "tune", "readout"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    data = load_chunks(args.chunks)
    out_dir = Path(args.out)
    if args.stage == "gate":
        stage_gate(data, out_dir)
    elif args.stage == "tune":
        stage_tune(data, out_dir, args.device)
    else:
        stage_readout(data, out_dir, args.device)


if __name__ == "__main__":
    main()
