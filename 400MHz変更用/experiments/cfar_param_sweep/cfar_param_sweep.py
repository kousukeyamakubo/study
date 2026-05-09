"""CFAR パラメータグリッドサーチスクリプト

(n_train_a, n_guard_a) × (n_train_d, n_guard_d) × (n_train_r, n_guard_r) × pfa
の全組み合わせを sweep し、各設定の (Precision, Recall, F1) を集計する。

高速化: (n_train, n_guard) が同じならノイズ推定の畳み込みを一度だけ実行し、
       pfa sweep 分は閾値乗算のみで済ませる。
評価:   NMS で離散化した検出点と GT を一対一マッチング (class-agnostic)。
"""

import os, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Meiryo'
from datetime import datetime

# ===== 定数 =====
MIXED_META_CSV = "../../learn_dataset_fixed_angle/metadata.csv"
OUTPUT_DIR     = "./cfar_param_sweep_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

# ===== 現在の基準設定 (eval_score_based_pr.py と同一) =====
DEFAULT_N_TRAIN = (1, 1, 10)  # (angle, doppler, range)
DEFAULT_N_GUARD = (1, 1, 5)
DEFAULT_PFA     = 1e-3

"""
# ===== sweep 範囲 =====
SWEEP_N_TRAIN_A = [1, 2, 3]
SWEEP_N_GUARD_A = [1]
SWEEP_N_TRAIN_D = [1, 2, 3, 5]
SWEEP_N_GUARD_D = [1, 2]
SWEEP_N_TRAIN_R = [5]
SWEEP_N_GUARD_R = [2, 3]
SWEEP_PFA       = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
"""
# ===== sweep 範囲 =====
SWEEP_N_TRAIN_A = [1, 2, 3]
SWEEP_N_GUARD_A = [1]
SWEEP_N_TRAIN_D = [5,6,7,8,9]
SWEEP_N_GUARD_D = [1, 2]
SWEEP_N_TRAIN_R = [5]
SWEEP_N_GUARD_R = [2, 3]
SWEEP_PFA       = [1e-2]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"出力先: {OUTPUT_DIR}", flush=True)


# ===== データ読み込み =====
def load_rd_maps_linear(npz_path):
    d = np.load(npz_path)
    return np.abs(d["rd_maps"]).astype(np.float32)  # (A, H, W)


def preload_rd_data(df):
    """全サンプルの線形 RD マップと rd_db を事前ロードする（I/O を1回に抑える）"""
    data = []
    for i in range(len(df)):
        row    = df.iloc[i]
        npz    = row["file"] if os.path.isabs(row["file"]) \
                 else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(MIXED_META_CSV)), "..", row["file"]))
        rd_lin = load_rd_maps_linear(npz)
        rd_db  = 20. * np.log10(np.maximum(rd_lin, 1e-12))
        data.append({"x_np": rd_lin, "rd_db": rd_db})
    return data


# ===== GT 構築 =====
def build_gts(df):
    """
    サンプルインデックスをキーとする GT リストを構築する (class-agnostic)。
    gts[i] = [{"ch": int|None, "d": int, "r": int}, ...]
    """
    gts = {}
    for i in range(len(df)):
        row     = df.iloc[i]
        entries = []
        if "cyclist_true_d_idx" in row and pd.notna(row.get("cyclist_true_d_idx")):
            cy_ach = int(np.argmin(np.abs(FIXED_ANGLES - float(row["cyclist_true_angle_deg"])))) \
                     if "cyclist_true_angle_deg" in row else None
            entries.append({"ch": cy_ach,
                            "d":  int(row["cyclist_true_d_idx"]),
                            "r":  int(row["cyclist_true_r_idx"])})
        if "vehicle_true_d_idx" in row and pd.notna(row.get("vehicle_true_d_idx")):
            ve_ach = int(np.argmin(np.abs(FIXED_ANGLES - float(row["vehicle_true_angle_deg"])))) \
                     if "vehicle_true_angle_deg" in row else None
            entries.append({"ch": ve_ach,
                            "d":  int(row["vehicle_true_d_idx"]),
                            "r":  int(row["vehicle_true_r_idx"])})
        gts[i] = entries
    return gts


# ===== ノイズ推定キャッシュ =====
def build_noise_cache(preloaded_data, n_train, n_guard):
    """
    (n_train, n_guard) 固定でノイズ推定値をキャッシュする。
    pfa ループでは x > α×noise_est の乗算のみで判定できるようにする。
    """
    ta, td, tr = n_train
    ga, gd, gr = n_guard
    ka = 2*(ta+ga)+1; kd = 2*(td+gd)+1; kr = 2*(tr+gr)+1
    kernel = torch.ones((1, 1, ka, kd, kr))
    # 角度・ドップラー・レンジの順でガードセル領域をゼロにする
    kernel[:, :, ta:ta+2*ga+1, td:td+2*gd+1, tr:tr+2*gr+1] = 0
    n_train_cells = int(kernel.sum().item())
    pad_a, pad_d, pad_r = ka//2, kd//2, kr//2

    noise_cache = []
    for item in preloaded_data:
        rd_lin = item["x_np"]
        x      = torch.from_numpy(rd_lin).unsqueeze(0).unsqueeze(0)  # (1,1,A,H,W)
        # range: replicate, doppler: circular（速度軸は周期的）, angle: replicate
        x_pad  = F.pad(x,     (pad_r, pad_r, 0, 0, 0, 0),     mode="replicate")
        x_pad  = F.pad(x_pad, (0, 0, pad_d, pad_d, 0, 0),     mode="circular")
        x_pad  = F.pad(x_pad, (0, 0, 0, 0, pad_a, pad_a),     mode="replicate")
        noise_est = (F.conv3d(x_pad, kernel) / n_train_cells)[0, 0].numpy()  # (A,H,W)
        noise_cache.append({"x_np":      rd_lin,
                            "noise_est": noise_est,
                            "rd_db":     item["rd_db"]})
    return noise_cache, n_train_cells


# ===== NMS =====
def nms_on_cfar_mask(mask, rd_db):
    """
    CFARマスクに greedy NMS を適用し、離散的な検出点リストを返す。
    numpy ブロードキャストで近傍抑制を高速化する。
    戻り値: list of (a, d, r)
    """
    locs = np.argwhere(mask)  # (K, 3): [a, d, r]
    if len(locs) == 0:
        return []
    scores = rd_db[locs[:, 0], locs[:, 1], locs[:, 2]]
    order  = np.argsort(-scores)
    locs   = locs[order]
    alive  = np.ones(len(locs), dtype=bool)
    peaks  = []
    for i in range(len(locs)):
        if not alive[i]:
            continue
        a, d, r = int(locs[i, 0]), int(locs[i, 1]), int(locs[i, 2])
        peaks.append((a, d, r))
        suppress = ((np.abs(locs[:, 0] - a) <= A_TOL) &
                    (np.abs(locs[:, 1] - d) <= D_TOL) &
                    (np.abs(locs[:, 2] - r) <= R_TOL))
        alive[suppress] = False
    return peaks


# ===== 1設定の評価 =====
def evaluate_one(noise_cache, gts, alpha):
    """
    固定 alpha (= N_train × (pfa^{-1/N_train} - 1)) で全サンプルを評価する。
    戻り値: (precision, recall, f1, tp, fp, fn)
    """
    tp = fp = fn = 0
    for i, nc in enumerate(noise_cache):
        mask       = nc["x_np"] > alpha * nc["noise_est"]  # (A,H,W) bool
        peaks      = nms_on_cfar_mask(mask, nc["rd_db"])
        sample_gts = gts.get(i, [])
        matched_gt  = set()
        matched_det = set()
        for gi, gt in enumerate(sample_gts):
            for pi, (a, d, r) in enumerate(peaks):
                if pi in matched_det:
                    continue
                if gt["ch"] is not None and abs(a - gt["ch"]) > A_TOL:
                    continue
                if abs(d - gt["d"]) <= D_TOL and abs(r - gt["r"]) <= R_TOL:
                    matched_gt.add(gi)
                    matched_det.add(pi)
                    break
        tp += len(matched_gt)
        fn += len(sample_gts) - len(matched_gt)
        fp += len(peaks) - len(matched_det)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


# ===== メイン sweep =====
def run_sweep(preloaded_data, gts):
    configs = list(itertools.product(
        SWEEP_N_TRAIN_A, SWEEP_N_GUARD_A,
        SWEEP_N_TRAIN_D, SWEEP_N_GUARD_D,
        SWEEP_N_TRAIN_R, SWEEP_N_GUARD_R,
    ))
    n_configs = len(configs)
    print(f"(n_train, n_guard) 組み合わせ: {n_configs}  ×  pfa: {len(SWEEP_PFA)} "
          f"= {n_configs * len(SWEEP_PFA)} 評価", flush=True)

    records = []
    for ci, (ta, ga, td, gd, tr, gr) in enumerate(configs):
        noise_cache, n_train_cells = build_noise_cache(
            preloaded_data, (ta, td, tr), (ga, gd, gr))

        for pfa in SWEEP_PFA:
            # CA-CFAR の倍率 α = N_train × (pfa^{-1/N_train} - 1)
            alpha = n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0)
            p, r, f1, tp, fp, fn = evaluate_one(noise_cache, gts, alpha)
            records.append({
                "n_train_a": ta, "n_guard_a": ga,
                "n_train_d": td, "n_guard_d": gd,
                "n_train_r": tr, "n_guard_r": gr,
                "pfa": pfa, "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
            })

        if (ci + 1) % 20 == 0 or (ci + 1) == n_configs:
            print(f"  [{ci+1:3d}/{n_configs}] 完了", flush=True)

    return pd.DataFrame(records)


# ===== データ準備 =====
print("データ読み込み中...", flush=True)
mixed_df    = pd.read_csv(MIXED_META_CSV)
mixed_df    = mixed_df[mixed_df["valid_all"] == 1].reset_index(drop=True)
mixed_df    = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
holdout2obj = mixed_df.iloc[200:].reset_index(drop=True)
N           = len(holdout2obj)
print(f"holdout2obj: {N} 件", flush=True)

preloaded = preload_rd_data(holdout2obj)
gts       = build_gts(holdout2obj)
n_gt      = sum(len(v) for v in gts.values())
print(f"GT 総数: {n_gt}", flush=True)


# ===== sweep 実行 =====
print("\nsweep 開始...", flush=True)
t0     = datetime.now()
df_res = run_sweep(preloaded, gts)
elapsed = (datetime.now() - t0).total_seconds()
print(f"\nsweep 完了  ({elapsed:.1f} 秒)", flush=True)


# ===== CSV 保存 =====
csv_path = os.path.join(OUTPUT_DIR, "cfar_param_sweep.csv")
df_res.to_csv(csv_path, index=False)
print(f"CSV 保存: {csv_path}", flush=True)


# ===== サマリー表示 =====
top20 = df_res.nlargest(20, "f1")
print("\n--- F1 上位 20 件 ---")
print(f"{'ta':>3} {'ga':>3} {'td':>3} {'gd':>3} {'tr':>3} {'gr':>3} "
      f"{'pfa':>8}  {'P':>6}  {'R':>6}  {'F1':>6}   tp   fp   fn")
print("-" * 78)
for _, row in top20.iterrows():
    print(f"{int(row.n_train_a):>3} {int(row.n_guard_a):>3} "
          f"{int(row.n_train_d):>3} {int(row.n_guard_d):>3} "
          f"{int(row.n_train_r):>3} {int(row.n_guard_r):>3} "
          f"{row.pfa:>8.0e}  {row.precision:>6.4f}  {row.recall:>6.4f}  "
          f"{row.f1:>6.4f}  {int(row.tp):>4} {int(row.fp):>4} {int(row.fn):>4}")

# 現在の基準設定の結果を表示
ta0, td0, tr0 = DEFAULT_N_TRAIN
ga0, gd0, gr0 = DEFAULT_N_GUARD
default_row = df_res[
    (df_res.n_train_a == ta0) & (df_res.n_guard_a == ga0) &
    (df_res.n_train_d == td0) & (df_res.n_guard_d == gd0) &
    (df_res.n_train_r == tr0) & (df_res.n_guard_r == gr0) &
    np.isclose(df_res.pfa, DEFAULT_PFA)
]
if len(default_row) > 0:
    dr = default_row.iloc[0]
    print(f"\n現在の基準設定 "
          f"(ta={ta0},ga={ga0}, td={td0},gd={gd0}, tr={tr0},gr={gr0}, pfa={DEFAULT_PFA:.0e}):")
    print(f"  P={dr.precision:.4f}  R={dr.recall:.4f}  F1={dr.f1:.4f}  "
          f"tp={int(dr.tp)} fp={int(dr.fp)} fn={int(dr.fn)}")


# ===== プロット =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"CFAR パラメータ sweep  (holdout 2-object, {N} samples, GT={n_gt})\n"
    f"全 {len(df_res)} 評価点  sweep 時間: {elapsed:.1f} 秒"
)

# --- 左: PR 散布図（F1 で色付け）---
ax = axes[0]
sc = ax.scatter(df_res["recall"], df_res["precision"],
                c=df_res["f1"], cmap="viridis",
                s=8, alpha=0.35, zorder=2)
plt.colorbar(sc, ax=ax, label="F1 score")

# 上位5件を強調表示
for rank, (_, row) in enumerate(top20.head(5).iterrows()):
    ax.scatter(row["recall"], row["precision"],
               marker="*", s=180, color="red", zorder=5)
    ax.annotate(
        f"#{rank+1} F1={row.f1:.3f}\n"
        f"ta={int(row.n_train_a)},ga={int(row.n_guard_a)}\n"
        f"td={int(row.n_train_d)},gd={int(row.n_guard_d)}\n"
        f"tr={int(row.n_train_r)},gr={int(row.n_guard_r)}\n"
        f"pfa={row.pfa:.0e}",
        (row["recall"], row["precision"]),
        fontsize=5, textcoords="offset points", xytext=(6, 3),
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )

# 現在の基準設定を強調表示
if len(default_row) > 0:
    dr = default_row.iloc[0]
    ax.scatter(dr["recall"], dr["precision"],
               marker="X", s=140, color="orange", zorder=5,
               label=f"現在設定 F1={dr['f1']:.3f}")
    ax.legend(fontsize=8)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_title("PR 散布図（色 = F1）")
ax.grid(True, alpha=0.3)

# --- 右: pfa 別 F1 分布（box plot）---
ax2 = axes[1]
pfa_labels   = [f"{p:.0e}" for p in SWEEP_PFA]
data_by_pfa  = [df_res[np.isclose(df_res["pfa"], p)]["f1"].values for p in SWEEP_PFA]
bp = ax2.boxplot(data_by_pfa, labels=pfa_labels, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightblue")

# 基準設定の pfa での F1 をオーバーレイ
if len(default_row) > 0:
    pfa_idx = SWEEP_PFA.index(DEFAULT_PFA) + 1  # boxplot は 1-indexed
    ax2.scatter([pfa_idx], [default_row.iloc[0]["f1"]],
                marker="X", s=100, color="orange", zorder=5,
                label=f"現在設定 F1={default_row.iloc[0]['f1']:.3f}")
    ax2.legend(fontsize=8)

ax2.set_xlabel("pfa")
ax2.set_ylabel("F1 score")
ax2.set_title("pfa 別 F1 分布（(n_train, n_guard) 全組み合わせ）")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "cfar_param_sweep.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nプロット保存: {plot_path}", flush=True)
print(f"\n完了。結果は {OUTPUT_DIR}/ に保存されました。", flush=True)
