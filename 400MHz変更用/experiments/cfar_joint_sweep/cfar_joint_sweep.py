"""cfar_joint_sweep.py

(n_train, n_guard, pfa, tar_thresh) を同時に総当たりして最良 CFAR 設定を探す。
閾値なし最適設定 ≠ 閾値あり最適設定 となる可能性があるため，
tar_thresh もパラメータ空間に含める。

高速化:
  - noise_est (F.conv3d) は (n_train, n_guard) ごとに 1 回だけ実行
  - CFAR mask + NMS は pfa ごとに 1 回だけ実行
  - tar_thresh ループはピークフィルタリングのみで済む
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

# ===== 定数 =====
MIXED_META_CSV = "../../learn_dataset_fixed_angle/metadata.csv"
OUTPUT_DIR     = "./cfar_joint_sweep_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

# ===== 基準設定 (eval_score_based_pr.py と同一) =====
DEFAULT_N_TRAIN = (1, 1, 10)
DEFAULT_N_GUARD = (1, 1, 5)
DEFAULT_PFA     = 1e-3

# ===== sweep 範囲 =====
SWEEP_N_TRAIN_A  = [1, 2, 3]
SWEEP_N_GUARD_A  = [1]
SWEEP_N_TRAIN_D  = [1, 2, 3, 5]
SWEEP_N_GUARD_D  = [1, 2]
SWEEP_N_TRAIN_R  = [5]        # target masking を避けるため固定
SWEEP_N_GUARD_R  = [2, 3]
SWEEP_PFA        = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
# None = 閾値なし（全 CFAR 通過検出を使用）
SWEEP_TAR_THRESH = [None, -40, -35, -30, -25, -20, -15, -10, -5, 0]

# CSV に記録する際の None のセンチネル値
THRESH_NONE_SENTINEL = -9999.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"出力先: {OUTPUT_DIR}", flush=True)


# ===== データ読み込み =====
def load_rd_maps_linear(npz_path):
    d = np.load(npz_path)
    return np.abs(d["rd_maps"]).astype(np.float32)  # (A, H, W)


def preload_rd_data(df):
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) \
              else os.path.normpath(os.path.join(
                  os.path.dirname(os.path.abspath(MIXED_META_CSV)), "..", row["file"]))
        rd_lin = load_rd_maps_linear(npz)
        rd_db  = 20. * np.log10(np.maximum(rd_lin, 1e-12))
        data.append({"x_np": rd_lin, "rd_db": rd_db})
    return data


# ===== GT 構築 =====
def build_gts(df):
    """gts[i] = [{"ch": int|None, "d": int, "r": int}, ...]"""
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
    """(n_train, n_guard) 固定でノイズ推定値をキャッシュする"""
    ta, td, tr = n_train
    ga, gd, gr = n_guard
    ka = 2*(ta+ga)+1; kd = 2*(td+gd)+1; kr = 2*(tr+gr)+1
    kernel = torch.ones((1, 1, ka, kd, kr))
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
    """CFAR マスクに greedy NMS を適用し，スコア付き検出点リストを返す"""
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
        peaks.append((a, d, r, float(rd_db[a, d, r])))
        suppress = ((np.abs(locs[:, 0] - a) <= A_TOL) &
                    (np.abs(locs[:, 1] - d) <= D_TOL) &
                    (np.abs(locs[:, 2] - r) <= R_TOL))
        alive[suppress] = False
    return peaks  # list of (a, d, r, score)


# ===== CFAR + NMS キャッシュ（pfa 固定） =====
def collect_cfar_peaks(noise_cache, gts, alpha):
    """全サンプルで CFAR mask + NMS を実行し，スコア付きピークをキャッシュする"""
    result = []
    for i, nc in enumerate(noise_cache):
        mask  = nc["x_np"] > alpha * nc["noise_est"]  # (A,H,W)
        peaks = nms_on_cfar_mask(mask, nc["rd_db"])    # [(a,d,r,score), ...]
        result.append({"peaks": peaks, "gts": gts.get(i, [])})
    return result


# ===== tar_thresh を適用して評価 =====
def evaluate_with_thresh(cached, tar_thresh):
    """
    tar_thresh=None: 全ピークを使用
    tar_thresh=X:    rd_db >= X のピークのみ使用
    """
    tp = fp = fn = 0
    for item in cached:
        if tar_thresh is None:
            peaks = [(a, d, r) for a, d, r, _ in item["peaks"]]
        else:
            peaks = [(a, d, r) for a, d, r, s in item["peaks"] if s >= tar_thresh]
        sample_gts  = item["gts"]
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
    n_configs  = len(configs)
    n_total    = n_configs * len(SWEEP_PFA) * len(SWEEP_TAR_THRESH)
    print(f"(n_train,n_guard): {n_configs}  ×  pfa: {len(SWEEP_PFA)}"
          f"  ×  tar_thresh: {len(SWEEP_TAR_THRESH)}  =  {n_total} 評価", flush=True)

    records = []
    for ci, (ta, ga, td, gd, tr, gr) in enumerate(configs):
        noise_cache, n_train_cells = build_noise_cache(
            preloaded_data, (ta, td, tr), (ga, gd, gr))

        for pfa in SWEEP_PFA:
            alpha  = n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0)
            cached = collect_cfar_peaks(noise_cache, gts, alpha)

            for tar_thresh in SWEEP_TAR_THRESH:
                p, r, f1, tp, fp, fn = evaluate_with_thresh(cached, tar_thresh)
                records.append({
                    "n_train_a":  ta, "n_guard_a": ga,
                    "n_train_d":  td, "n_guard_d": gd,
                    "n_train_r":  tr, "n_guard_r": gr,
                    "pfa":        pfa,
                    "tar_thresh": THRESH_NONE_SENTINEL if tar_thresh is None else tar_thresh,
                    "precision":  p, "recall": r, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn,
                })

        pct = (ci + 1) / n_configs * 100
        best_f1 = max(rec["f1"] for rec in records[-(len(SWEEP_PFA)*len(SWEEP_TAR_THRESH)):])
        print(f"  [{pct:5.1f}%] config {ci+1}/{n_configs}"
              f"  ta={ta} td={td} tr={tr} gr={gr}"
              f"  best_f1_this_config={best_f1:.4f}", flush=True)

    return pd.DataFrame(records)


# ===== プロット =====
def plot_results(df, df_default_row):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    n_evals = len(df)
    fig.suptitle(
        f"CFAR 合同 sweep (n_train, n_guard, pfa, tar_thresh)  "
        f"holdout 2-object, 100 samples\n全 {n_evals} 評価点",
        fontsize=12
    )

    sc = axes[0].scatter(df["recall"], df["precision"],
                         c=df["f1"], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0], label="F1 score")
    # トップ5 をハイライト
    top5 = df.nlargest(5, "f1")
    axes[0].scatter(top5["recall"], top5["precision"],
                    marker="*", color="red", s=120, zorder=5, label="Top-5")
    if df_default_row is not None:
        axes[0].scatter([df_default_row["recall"]], [df_default_row["precision"]],
                        marker="X", color="darkorange", s=150, zorder=6,
                        label=f"基準設定 F1={df_default_row['f1']:.3f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("PR 散布図（色 = F1）")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    # tar_thresh 別 F1 box plot（None = -9999 は "なし" として表示）
    thresh_labels = []
    thresh_groups = []
    for t in SWEEP_TAR_THRESH:
        key = THRESH_NONE_SENTINEL if t is None else t
        sub = df[df["tar_thresh"] == key]["f1"].values
        thresh_groups.append(sub)
        thresh_labels.append("なし" if t is None else f"{t}")
    axes[1].boxplot(thresh_groups, tick_labels=thresh_labels)
    axes[1].set_xlabel("tar_thresh [dB]"); axes[1].set_ylabel("F1")
    axes[1].set_title("tar_thresh 別 F1 分布（全 (n_train,n_guard,pfa) 組み合わせ）")
    axes[1].grid(True, axis="y")
    if df_default_row is not None:
        axes[1].axhline(df_default_row["f1"], color="darkorange", ls="--", lw=1,
                        label=f"基準設定 F1={df_default_row['f1']:.3f}")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cfar_joint_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"プロット保存: {out}", flush=True)


# ===== メイン =====
def main():
    print("データ読み込み中...", flush=True)
    df_all = pd.read_csv(MIXED_META_CSV)
    df_all = df_all[df_all["valid_all"] == 1].reset_index(drop=True)
    np.random.seed(RANDOM_SEED)
    holdout2obj = df_all.tail(100).reset_index(drop=True)
    print(f"holdout2obj: {len(holdout2obj)} 件", flush=True)

    preloaded = preload_rd_data(holdout2obj)
    gts       = build_gts(holdout2obj)
    print(f"GT 総数: {sum(len(v) for v in gts.values())}", flush=True)

    # --- 基準設定の単一評価点（比較用） ---
    nc_def, ntc_def = build_noise_cache(preloaded, DEFAULT_N_TRAIN, DEFAULT_N_GUARD)
    alpha_def       = ntc_def * (DEFAULT_PFA ** (-1.0 / ntc_def) - 1.0)
    cached_def      = collect_cfar_peaks(nc_def, gts, alpha_def)
    p0, r0, f0, *_  = evaluate_with_thresh(cached_def, None)
    df_default_row  = {"precision": p0, "recall": r0, "f1": f0}
    print(f"基準設定 (thresh無し): P={p0:.4f}  R={r0:.4f}  F1={f0:.4f}", flush=True)

    # --- sweep 実行 ---
    print("\nsweep 開始...", flush=True)
    df = run_sweep(preloaded, gts)

    # --- 保存 ---
    csv_path = os.path.join(OUTPUT_DIR, "cfar_joint_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV 保存: {csv_path}", flush=True)

    # --- トップ5 表示 ---
    print("\n=== トップ 5 (F1 降順) ===")
    top5 = df.nlargest(5, "f1")
    for _, row in top5.iterrows():
        t = "なし" if row["tar_thresh"] == THRESH_NONE_SENTINEL else f"{row['tar_thresh']:.0f} dB"
        print(f"  ta={int(row.n_train_a)} td={int(row.n_train_d)} tr={int(row.n_train_r)}"
              f" gr={int(row.n_guard_r)} pfa={row.pfa:.0e} thresh={t}"
              f"  P={row.precision:.4f} R={row.recall:.4f} F1={row.f1:.4f}"
              f"  TP={int(row.tp)} FP={int(row.fp)}")

    plot_results(df, df_default_row)
    print("\n完了。結果は ./cfar_joint_sweep_results/ に保存されました。", flush=True)


if __name__ == "__main__":
    main()
