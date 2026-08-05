#!/usr/bin/env python3
"""L0 零成本维度定位：NN / k-分划映射判别分析.

现象 Φ："类型级结构不迁移但个体级机会存在"
  （S1 迁移差 -83.60、类型内 Kendall tau 0.062 ~= 跨类型 0.065、
   去 identity oracle 84.22）。

本脚本用已有数据对 Φ 做维度定位，判别竞争解释：
  H1 低信噪比 / H2 前缀不足 / H3 模式基底伪结构 / H4 布局混淆。

核心对比（同一前缀特征、同一 k-NN 估计器）：
  A. 签名可预测性：前缀特征 -> continuation 价值签名（k-NN 回归 + 分划分类）
  B. 身份可预测性：前缀特征 -> run 标签 / 类型标签（k-NN 分类）
两者都给出 run-disjoint 版本（排除同一 run 的近邻，防 identity 泄漏）。

输入：符合 DATA_SPEC.md 的 npz（--input）。
自检：--smoke 生成三种已知结构的合成数据并断言读数方向，验证脚本自身正确性。

依赖：仅 numpy（服务器可直接运行）。
用法：
  python l0_nn_partition_discriminant.py --input <l0_dataset.npz> \
      --output-dir runs/exploration/s1_panel/full/l0_dimension_localization/
  python l0_nn_partition_discriminant.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ACTION_DIM = 6  # Overcooked 动作 0-5


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def loo_knn_indices(x: np.ndarray, k: int, block: int = 256,
                    forbid_same_group: np.ndarray | None = None) -> np.ndarray:
    """留一 k 近邻索引。x: (N, D) 已标准化。

    forbid_same_group 为 (N,) 整数标签时，候选近邻排除与查询同组的样本
    （run-disjoint 口径）；排除后不足 k 个时用全体补位并打标记在返回外处理。
    返回 (N, k) 索引。
    """
    n = x.shape[0]
    k = min(k, n - 1)
    out = np.zeros((n, k), dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        xb = x[s:e]
        d2 = ((xb[:, None, :] - x[None, :, :]) ** 2).sum(-1)  # (b, N)
        for i in range(e - s):
            d2[i, s + i] = np.inf  # 留一：排除自身
        if forbid_same_group is not None:
            # same[b, j] = 查询 b（全局索引 s+b）与候选 j 同组 -> (b, N)
            same = forbid_same_group[s:e, None] == forbid_same_group[None, :]
            d2[same] = np.inf
        k_need = k
        if forbid_same_group is not None:
            # 若某查询同组样本过多导致有限候选不足 k，退化为全体（含同组）补位
            finite = np.isfinite(d2).sum(axis=1)
            shortfall = finite < k_need
            if shortfall.any():
                d2f = ((xb[shortfall][:, None, :] - x[None, :, :]) ** 2).sum(-1)
                for i_local, i_glob in enumerate(np.where(shortfall)[0]):
                    d2f[i_local, i_glob] = np.inf  # 排除自身
                d2[shortfall] = d2f
        idx = np.argpartition(d2, k_need, axis=1)[:, :k_need]
        out[s:e] = idx
    return out


def x_perm_knn(x: np.ndarray, k: int, rng: np.random.Generator,
               forbid_same_group: np.ndarray | None = None) -> np.ndarray:
    """特征行置换后的 k 近邻（破坏特征与一切标签的关联，
    保留近邻几何，用于构造无偏零分布）。"""
    return loo_knn_indices(x[rng.permutation(x.shape[0])], k,
                           forbid_same_group=forbid_same_group)


def classify_acc(y: np.ndarray, nn_idx: np.ndarray) -> float:
    """k-NN 多数类分类准确率（平票按近邻中先到者）。"""
    pred = np.empty(y.shape[0], dtype=y.dtype)
    for i in range(y.shape[0]):
        vals, counts = np.unique(y[nn_idx[i]], return_counts=True)
        pred[i] = vals[counts.argmax()]
    return float((pred == y).mean())


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 n_boot: int = 999, alpha: float = 0.05):
    n = values.shape[0]
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def spearman_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行 Spearman 相关。(N, M) x (N, M) -> (N,)"""
    ra = a.argsort(axis=1).argsort(axis=1).astype(float)
    rb = b.argsort(axis=1).argsort(axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator,
           n_iter: int = 100) -> np.ndarray:
    """朴素 k-means，返回标签。空簇重置到最远点。"""
    n = x.shape[0]
    k = min(k, n)
    centers = x[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                centers[c] = x[mask].mean(axis=0)
            else:
                far = d2.min(axis=1).argmax()
                centers[c] = x[far]
                new_labels[far] = c
        if (new_labels == labels).all():
            break
        labels = new_labels
    return labels


# ----------------------------------------------------------------------
# 核心分析
# ----------------------------------------------------------------------

def analyze_dataset(data: dict, k: int, n_perm: int, n_boot: int,
                    max_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    x3 = data["prefix_features"].astype(np.float64)  # (N, T, D)
    sig = data["signature"].astype(np.float64)       # (N, M)
    run = data["run_label"].astype(np.int64)
    typ = data["type_label"].astype(np.int64)
    prefix_t = np.asarray(data.get("prefix_t", np.arange(x3.shape[1])))

    n = x3.shape[0]
    # NaN 行剔除（规格禁止插补）
    good = np.isfinite(x3).all(axis=(1, 2)) & np.isfinite(sig).all(axis=1)
    if not good.all():
        print(f"[warn] 剔除含 NaN 样本 {int((~good).sum())} 行", file=sys.stderr)
        x3, sig, run, typ = x3[good], sig[good], run[good], typ[good]
        n = x3.shape[0]

    # 分层降采样（按 run 均衡），控制距离矩阵开销
    if n > max_samples:
        keep = []
        runs = np.unique(run)
        per = max(1, max_samples // len(runs))
        for r in runs:
            idx = np.where(run == r)[0]
            keep.append(rng.choice(idx, min(per, len(idx)), replace=False))
        keep = np.concatenate(keep)
        x3, sig, run, typ = x3[keep], sig[keep], run[keep], typ[keep]
        n = x3.shape[0]
        print(f"[info] 降采样到 N={n}（每 run ≤ {per}）", file=sys.stderr)

    sig_centered = sig - sig.mean(axis=1, keepdims=True)
    n_runs = int(run.max() + 1)

    # 签名 k-means 分划（K=2 与 K=min(n_runs, 20)），作为 k-分划判别的标签
    part2 = kmeans(sig_centered, 2, rng)
    partR = kmeans(sig_centered, min(n_runs, 20), rng)

    per_cutoff = []
    for ti in range(x3.shape[1]):
        x = standardize(x3[:, ti, :])

        # --- 身份可预测性（B 任务） ---
        nn_pooled = loo_knn_indices(x, k)
        nn_disjoint = loo_knn_indices(x, k, forbid_same_group=run)

        acc_run = classify_acc(run, nn_pooled)
        acc_type = classify_acc(typ, nn_pooled)
        acc_run_dj = classify_acc(run, nn_disjoint)
        acc_type_dj = classify_acc(typ, nn_disjoint)

        # 置换零分布：特征行置换重建近邻（标签不变），破坏特征-标签关联
        n_perm_eff = min(n_perm, 100)
        accs_run_null, accs_type_null = [], []
        accs_run_dj_null, accs_type_dj_null = [], []
        rho_null_list, rho_dj_null_list = [], []
        pred_sig_null = np.zeros((n_perm_eff, sig_centered.shape[0],
                                  sig_centered.shape[1]))
        pred_sig_dj_null = np.zeros_like(pred_sig_null)
        for p in range(n_perm_eff):
            nn_n = x_perm_knn(x, k, rng)
            nn_nd = x_perm_knn(x, k, rng, forbid_same_group=run)
            accs_run_null.append(classify_acc(run, nn_n))
            accs_type_null.append(classify_acc(typ, nn_n))
            accs_run_dj_null.append(classify_acc(run, nn_nd))
            accs_type_dj_null.append(classify_acc(typ, nn_nd))
            pred_sig_null[p] = sig_centered[nn_n].mean(axis=1)
            pred_sig_dj_null[p] = sig_centered[nn_nd].mean(axis=1)
        rho_null_list = [float(spearman_rowwise(pred_sig_null[p],
                                                sig_centered).mean())
                         for p in range(n_perm_eff)]
        rho_dj_null_list = [float(spearman_rowwise(pred_sig_dj_null[p],
                                                   sig_centered).mean())
                            for p in range(n_perm_eff)]
        null_run = float(np.mean(accs_run_null))
        null_type = float(np.mean(accs_type_null))
        null_run_dj = float(np.mean(accs_run_dj_null))
        null_type_dj = float(np.mean(accs_type_dj_null))
        rho_null = float(np.mean(rho_null_list))
        rho_dj_null = float(np.mean(rho_dj_null_list))

        # --- 签名可预测性（A 任务） ---
        # A1: k-NN 回归：预测签名 = k 近邻签名均值，逐行 Spearman
        pred_sig = sig_centered[nn_pooled].mean(axis=1)
        pred_sig_dj = sig_centered[nn_disjoint].mean(axis=1)
        rho = float(spearman_rowwise(pred_sig, sig_centered).mean())
        rho_dj = float(spearman_rowwise(pred_sig_dj, sig_centered).mean())

        # A2: k-分划分类：预测签名分划成员（K=2 / K=run 数）
        acc_p2 = classify_acc(part2, nn_pooled)
        acc_pR = classify_acc(partR, nn_pooled)
        acc_p2_dj = classify_acc(part2, nn_disjoint)
        acc_pR_dj = classify_acc(partR, nn_disjoint)
        null_p2 = float(np.mean([classify_acc(part2, x_perm_knn(x, k, rng))
                                 for _ in range(n_perm_eff)]))
        null_pR = float(np.mean([classify_acc(partR, x_perm_knn(x, k, rng))
                                 for _ in range(n_perm_eff)]))

        # A3: 路由回收代理：由预测签名选模式 vs 真签名最优模式的一致率
        mode_pick_pred = pred_sig.argmax(axis=1)
        mode_pick_true = sig_centered.argmax(axis=1)
        route_agree = float((mode_pick_pred == mode_pick_true).mean())
        route_agree_dj = float(
            (pred_sig_dj.argmax(axis=1) == mode_pick_true).mean())
        route_null = float(1.0 / sig_centered.shape[1])

        per_cutoff.append({
            "prefix_t": int(prefix_t[ti]),
            "acc_run": acc_run, "null_run": null_run,
            "acc_type": acc_type, "null_type": null_type,
            "acc_run_disjoint": acc_run_dj, "null_run_disjoint": null_run_dj,
            "acc_type_disjoint": acc_type_dj, "null_type_disjoint": null_type_dj,
            "sig_spearman": rho, "sig_spearman_disjoint": rho_dj,
            "sig_spearman_null": rho_null,
            "sig_spearman_disjoint_null": rho_dj_null,
            "acc_partition2": acc_p2, "null_partition2": null_p2,
            "acc_partition_run": acc_pR, "null_partition_run": null_pR,
            "acc_partition2_disjoint": acc_p2_dj,
            "acc_partition_run_disjoint": acc_pR_dj,
            "route_agreement": route_agree,
            "route_agreement_disjoint": route_agree_dj,
            "route_chance": route_null,
        })

    # 区间：对最终切点的逐样本正确性做样本级 bootstrap（基于降采样后数据）
    ti_last = x3.shape[1] - 1
    x = standardize(x3[:, ti_last, :])
    nn = loo_knn_indices(x, k)
    nn_labels_run = np.array([[run[j] for j in nn[i]] for i in range(n)])
    nn_labels_typ = np.array([[typ[j] for j in nn[i]] for i in range(n)])
    pred_run = np.array([np.bincount(row, minlength=n_runs).argmax()
                         for row in nn_labels_run])
    pred_typ = np.array([np.bincount(row, minlength=2).argmax()
                         for row in nn_labels_typ])
    hit_run = (pred_run == run).astype(float)
    hit_type = (pred_typ == typ).astype(float)
    br, br_lo, br_hi = bootstrap_ci(hit_run, rng, n_boot)
    bt, bt_lo, bt_hi = bootstrap_ci(hit_type, rng, n_boot)

    result = {
        "n_samples": n,
        "n_runs": n_runs,
        "n_types": int(typ.max() + 1),
        "n_modes": int(sig.shape[1]),
        "k": k,
        "per_cutoff": per_cutoff,
        "bootstrap_last_cutoff": {
            "acc_run": [br, br_lo, br_hi],
            "acc_type": [bt, bt_lo, bt_hi],
        },
        "note": "探索轨读数，scientific_readout_allowed=false；"
                "run-disjoint 口径排除同 run 近邻，防 identity 泄漏",
    }
    result["verdict"] = derive_verdict(result)
    return result


def derive_verdict(result: dict) -> dict:
    """按 READOUT_CRITERIA.md 的判读规则给 H1-H4 初步排除/保留结论。"""
    cuts = result["per_cutoff"]
    last, first = cuts[-1], cuts[0]

    def lift(val, null):
        return val - null

    sig_lift_dj = lift(last["sig_spearman_disjoint"],
                       last["sig_spearman_disjoint_null"])
    type_lift_dj = lift(last["acc_type_disjoint"], last["null_type_disjoint"])
    run_lift_dj = lift(last["acc_run_disjoint"], last["null_run_disjoint"])
    chance_run = 1.0 / result["n_runs"]
    chance_type = 1.0 / result["n_types"]

    # H2 形状：签名可预测性（对零线抬升）随前缀长度的增幅
    def sig_lift_at(c):
        return c["sig_spearman_disjoint"] - c["sig_spearman_disjoint_null"]

    if len(cuts) >= 2:
        rise = sig_lift_at(last) - sig_lift_at(first)
    else:
        rise = 0.0

    verdict = {}

    # H1 低信噪比：签名与身份任务在最终切点均无高于零的抬升
    if sig_lift_dj <= 0.02 and type_lift_dj <= 0.02 and run_lift_dj <= 0.02:
        verdict["H1_low_snr"] = ("RETAIN", "最终切点签名/身份预测均无显著高于"
                                 "置换零分布的抬升，低信噪比解释未被排除")
    else:
        verdict["H1_low_snr"] = ("EXCLUDE", "至少一项预测任务显著高于零分布，"
                                 "数据中存在可恢复信号，低信噪比解释初步排除")

    # H2 前缀不足：窗口内单调上升且未趋平 + 绝对抬升仍低
    if len(cuts) >= 2 and rise > 0.10 and sig_lift_dj < 0.3:
        verdict["H2_prefix_insufficient"] = (
            "RETAIN", f"签名可预测性在前缀窗口内持续上升（+{rise:.2f}）且绝对"
            "水平低，窗口内前缀信息不足解释未被排除（与 E4 的 425 步判读一致方向）")
    else:
        verdict["H2_prefix_insufficient"] = (
            "EXCLUDE_OR_WEAKEN", "签名可预测性在窗口内趋平或已达可观水平，"
            "前缀不足解释初步排除或弱化；若绝对水平仍低则转由 H1/H3 解释")

    # H3 模式基底伪结构：签名可预测但 run-disjoint 身份/分划不可预测，
    # 且路由一致率接近机会水平 => 机会结构在个体级而非模式基底级
    route_l = lift(last["route_agreement_disjoint"], last["route_chance"])
    if sig_lift_dj > 0.05 and run_lift_dj <= 0.02 and route_l > 0.05:
        verdict["H3_spurious_mode_structure"] = (
            "RETAIN_AS_INDIVIDUAL", "签名可预测但 run 身份不可跨 run 预测，"
            "且路由回收高于机会水平：结构存在于个体级签名而非类型级模式基底，"
            "与 E0 tau 读数一致方向")
    elif sig_lift_dj <= 0.02:
        verdict["H3_spurious_mode_structure"] = (
            "INCONCLUSIVE", "签名本身不可预测，H3 无法由本分析单独裁决")
    else:
        verdict["H3_spurious_mode_structure"] = (
            "EXCLUDE_OR_WEAKEN", "签名、身份、路由读数组合不符合伪结构形态，"
            "H3 初步弱化（需 L1 正对照复核）")

    # H4 布局混淆：本轮仅 Simple 布局，恒声明边界
    verdict["H4_layout_confound"] = (
        "BOUNDARY_ONLY", "本轮数据仅含 Simple 布局，H4 无法在本数据集内排除；"
        "结论边界：所有读数仅对 Simple 成立，Wide 后置（P6 口径）")

    return verdict


# ----------------------------------------------------------------------
# 合成 smoke test（验证脚本自身正确性）
# ----------------------------------------------------------------------

def _make_synthetic(kind: str, rng: np.random.Generator,
                    n_runs: int = 12, per_run: int = 120, m_modes: int = 24,
                    cutoffs=(50, 100, 200, 400)) -> dict:
    n = n_runs * per_run
    # 打乱 run 排布，避免 run 索引顺序与类型标签产生伪相关
    run = np.repeat(rng.permutation(n_runs), per_run)
    type_of_run = np.zeros(n_runs, dtype=np.int64)
    type_of_run[rng.permutation(n_runs)[: n_runs // 2]] = 1
    typ = type_of_run[run]
    d = 12
    t_len = len(cutoffs)

    if kind == "individual_structure":
        # 个体级结构：签名是 latent 的线性函数（可由前缀恢复），
        # 但 latent 连续、类型与 latent 无关 -> run/类型标签不可跨 run 预测
        latent = rng.normal(size=(n_runs, d))
        x = np.repeat(latent, per_run, axis=0)[:, None, :] + \
            rng.normal(scale=0.4, size=(n, t_len, d))
        b_proj = rng.normal(size=(d, m_modes))
        w = 2.0 * (latent @ b_proj)
        sig = np.repeat(w, per_run, axis=0) + rng.normal(scale=0.3, size=(n, m_modes))
    elif kind == "noise":
        x = rng.normal(size=(n, t_len, d))
        sig = rng.normal(size=(n, m_modes))
    elif kind == "prefix_insufficient":
        # 短切点特征无信息，长切点才有信息
        latent = rng.normal(size=(n_runs, d))
        b_proj = rng.normal(size=(d, m_modes))
        w = 2.0 * (latent @ b_proj)
        sig = np.repeat(w, per_run, axis=0) + rng.normal(scale=0.3, size=(n, m_modes))
        x = rng.normal(size=(n, t_len, d))
        x[:, -1, :] = np.repeat(latent, per_run, axis=0) + \
            rng.normal(scale=0.4, size=(n, d))
    else:
        raise ValueError(kind)

    return {
        "prefix_features": x.astype(np.float32),
        "signature": sig.astype(np.float32),
        "run_label": run,
        "type_label": typ,
        "layout": np.zeros(n, dtype=np.int64),
        "prefix_t": np.asarray(cutoffs),
        "mode_run_ids": np.arange(m_modes),
    }


def run_smoke(seed: int = 20260803) -> int:
    print("== smoke test：合成数据三形态自检 ==")
    ok = True

    # 形态 1：个体级结构（签名可预测、类型≈机会、run 可预测）
    data = _make_synthetic("individual_structure", np.random.default_rng(seed))
    r = analyze_dataset(data, k=5, n_perm=50, n_boot=99, max_samples=1600, seed=seed)
    last = r["per_cutoff"][-1]
    c1 = (last["sig_spearman_disjoint"] > last["sig_spearman_disjoint_null"] + 0.1
          and last["acc_type_disjoint"] < 0.5 + 0.10
          and last["acc_run_disjoint"] < last["null_run_disjoint"] + 0.05)
    print(f"[1] individual_structure: sig_rho_dj={last['sig_spearman_disjoint']:.3f}"
          f" (null {last['sig_spearman_disjoint_null']:.3f}),"
          f" acc_type_dj={last['acc_type_disjoint']:.3f},"
          f" acc_run_dj={last['acc_run_disjoint']:.3f}"
          f" (null {last['null_run_disjoint']:.3f}) -> {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    # 形态 2：纯噪声（所有任务≈机会/零）
    data = _make_synthetic("noise", np.random.default_rng(seed + 1))
    r = analyze_dataset(data, k=5, n_perm=50, n_boot=99, max_samples=1600, seed=seed)
    last = r["per_cutoff"][-1]
    c2 = (abs(last["sig_spearman_disjoint"] -
              last["sig_spearman_disjoint_null"]) < 0.08
          and last["acc_run"] < last["null_run"] + 0.05
          and r["verdict"]["H1_low_snr"][0] == "RETAIN")
    print(f"[2] noise: sig_rho_dj={last['sig_spearman_disjoint']:.3f}"
          f" (null {last['sig_spearman_disjoint_null']:.3f}),"
          f" H1={r['verdict']['H1_low_snr'][0]}"
          f" -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # 形态 3：前缀不足（短切点无信息、长切点有信息，曲线上升）
    data = _make_synthetic("prefix_insufficient", np.random.default_rng(seed + 2))
    r = analyze_dataset(data, k=5, n_perm=50, n_boot=99, max_samples=1600, seed=seed)
    first, last = r["per_cutoff"][0], r["per_cutoff"][-1]
    lift0 = first["sig_spearman_disjoint"] - first["sig_spearman_disjoint_null"]
    lift1 = last["sig_spearman_disjoint"] - last["sig_spearman_disjoint_null"]
    c3 = (lift1 - lift0 > 0.2 and lift1 > 0.1 and abs(lift0) < 0.15)
    print(f"[3] prefix_insufficient: rho_dj {first['sig_spearman_disjoint']:.3f} -> "
          f"{last['sig_spearman_disjoint']:.3f} -> {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    print("== smoke 总结:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def load_npz(path: str) -> dict:
    z = np.load(path)
    required = ["prefix_features", "signature", "run_label", "type_label"]
    missing = [f for f in required if f not in z.files]
    if missing:
        raise SystemExit(f"[error] npz 缺少必需字段 {missing}，见 DATA_SPEC.md")
    data = {f: z[f] for f in z.files}
    if "prefix_t" not in data:
        data["prefix_t"] = np.arange(data["prefix_features"].shape[1])
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default=None, help="l0_dataset.npz 路径")
    ap.add_argument("--output-dir", type=str, default=".",
                    help="结果输出目录")
    ap.add_argument("--k", type=int, default=5, help="k-NN 的 k")
    ap.add_argument("--n-perm", type=int, default=200, help="置换零分布次数")
    ap.add_argument("--n-boot", type=int, default=999, help="bootstrap 次数")
    ap.add_argument("--max-samples", type=int, default=2400,
                    help="降采样上限（按 run 均衡）")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--smoke", action="store_true", help="合成自检模式")
    args = ap.parse_args()

    if args.smoke:
        return run_smoke(args.seed)

    if not args.input:
        raise SystemExit("[error] 需要 --input <npz>，或使用 --smoke 自检")
    data = load_npz(args.input)
    result = analyze_dataset(data, k=args.k, n_perm=args.n_perm,
                             n_boot=args.n_boot, max_samples=args.max_samples,
                             seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "l0_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] 结果写入 {out_path}")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
