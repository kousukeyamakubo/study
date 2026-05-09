"""cfar_thresh_sweep.py

CFAR パラメータを cfar_param_sweep で求めた最良設定に固定し，
tar_thresh（信号強度下限）を連続的に sweep して PR 曲線を描く。

全ピークの rd_db 値を閾値候補として使うことで正確な PR 曲線を生成する。
"""

import os
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
OUTPUT_DIR     = "./cfar_thresh_sweep_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

# cfar_param_sweep で求めた最良設定
BEST_N_TRAIN = (3, 1, 5)   # (angle, doppler, range)
BEST_N_GUARD = (1, 1, 3)
BEST_PFA     = 1e-5

# 比較用: 現在の基準設定 (eval_score_based_pr.py と同一)
DEFAULT_N_TRAIN = (1, 1, 10)
DEFAULT_N_GUARD = (1, 1, 5)
DEFAULT_PFA     = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"出力先: {OUTPUT_DIR}", flush=True)


# ===== データ読み込み =====
def load_rd_maps_linear(npz_path):
    d = np.load(npz_path)
    return np.abs(d["rd_maps"]).astype(np.float32)  # (A, H, W)


def preload_rd_data(df):
    """全サンプルの線形 RD マップと rd_db を事前ロードする"""
    data = []
    for i in range(len(df)):
        row    = df.iloc[i]
        npz    = row["file"] if os.path.isabs(row["file"]) \
                 else os.path.normpath(os.path.join(
                     os.path.dirname(os.path.abspath(MIXED_META_CSV)), "..", row["file"]))
        rd_lin = load_rd_maps_linear(npz)
        rd_db  = 20. * np.log10(np.maximum(rd_lin, 1e-12))
        data.append({"x_np": rd_lin, "rd_db": rd_db})
    return data


# ===== GT 構築 =====
def build_gts(df):
    """
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
    """
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
    CFAR マスクに greedy NMS を適用し，離散的な検出点リストを返す。
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


# ===== CFAR + NMS キャッシュ =====
def collect_cfar_peaks(noise_cache, gts, alpha):
    """
    全サンプルで CFAR mask + NMS を実行し，スコア付きピークと GT をキャッシュする。
    戻り値: list of {"peaks": [(a, d, r, score), ...], "gts": [...]}
    """
    result = []
    for i, nc in enumerate(noise_cache):
        mask   = nc["x_np"] > alpha * nc["noise_est"]  # (A,H,W) bool
        peaks  = nms_on_cfar_mask(mask, nc["rd_db"])
        # rd_db スコアを付加
        peaks_scored = [(a, d, r, float(nc["rd_db"][a, d, r])) for a, d, r in peaks]
        result.append({"peaks": peaks_scored, "gts": gts.get(i, [])})
    return result


# ===== 1閾値での評価 =====
def evaluate_with_thresh(cached, tar_thresh):
    """
    tar_thresh 以上の rd_db を持つピークのみで P/R/F1 を計算する。
    """
    tp = fp = fn = 0
    for item in cached:
        peaks      = [(a, d, r) for a, d, r, s in item["peaks"] if s >= tar_thresh]
        sample_gts = item["gts"]
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


# ===== tar_thresh sweep =====
def sweep_tar_thresh(cached):
    """
    全ピークの rd_db 値を閾値候補として PR 曲線を計算する。
    高閾値（厳しい）→ 低閾値（緩い）の順に評価することで単調な PR 曲線が得られる。
    """
    all_scores = sorted(
        set(s for item in cached for _, _, _, s in item["peaks"]),
        reverse=True
    )
    if not all_scores:
        return pd.DataFrame(columns=["tar_thresh", "precision", "recall", "f1", "tp", "fp", "fn"])

    # 番兵: 最高スコア+1（全除外 → P=1,R=0）と最低スコア-1（全含む）
    thresholds = [all_scores[0] + 1.0] + all_scores + [all_scores[-1] - 1.0]

    records = []
    for tau in thresholds:
        p, r, f1, tp, fp, fn = evaluate_with_thresh(cached, tau)
        records.append({
            "tar_thresh": tau,
            "precision":  p,
            "recall":     r,
            "f1":         f1,
            "tp":         tp,
            "fp":         fp,
            "fn":         fn,
        })
    return pd.DataFrame(records)


# ===== プロット =====
def plot_results(df_best, df_default, n_gt_total):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"CFAR tar_thresh sweep  (holdout 2-object, 100 samples, GT={n_gt_total})\n"
        f"最良設定: n_train={BEST_N_TRAIN}, n_guard={BEST_N_GUARD}, pfa={BEST_PFA:.0e}",
        fontsize=12
    )

    # --- 左: PR 曲線 ---
    ax = axes[0]
    ax.set_title("PR 曲線（tar_thresh を変化）")
    ax.plot(df_best["recall"], df_best["precision"],
            color="steelblue", lw=2, label="最良CFAR設定")
    # F1 等高線
    for f1_val in [0.05, 0.10, 0.20, 0.30]:
        r_arr = np.linspace(0.01, 1.0, 200)
        p_arr = f1_val * r_arr / (2 * r_arr - f1_val)
        valid = p_arr > 0
        ax.plot(r_arr[valid], p_arr[valid], "k--", lw=0.6, alpha=0.4)
        ax.text(r_arr[valid][-1], p_arr[valid][-1], f"F1={f1_val:.2f}",
                fontsize=7, color="gray")
    # デフォルト設定の単一点（tar_thresh なし）
    if df_default is not None and len(df_default) > 0:
        row0 = df_default[df_default["tar_thresh"] == df_default["tar_thresh"].min()].iloc[0]
        ax.scatter([row0["recall"]], [row0["precision"]],
                   marker="X", color="darkorange", s=120, zorder=5,
                   label=f"基準設定(thresh無し) F1={row0['f1']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(True)

    # --- 右: F1 vs tar_thresh ---
    ax2 = axes[1]
    ax2.set_title("F1 vs tar_thresh [dB]")
    ax2.plot(df_best["tar_thresh"], df_best["f1"], color="steelblue", lw=2)
    best_row = df_best.loc[df_best["f1"].idxmax()]
    ax2.axvline(best_row["tar_thresh"], color="red", ls="--", lw=1,
                label=f"最良 tar_thresh={best_row['tar_thresh']:.1f} dB  F1={best_row['f1']:.4f}")
    ax2.set_xlabel("tar_thresh [dB]"); ax2.set_ylabel("F1")
    ax2.legend(fontsize=9); ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cfar_thresh_sweep.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"プロット保存: {out_path}", flush=True)


# ===== メイン =====
def main():
    # --- データ読み込み ---
    print("データ読み込み中...", flush=True)
    df_all = pd.read_csv(MIXED_META_CSV)
    df_all = df_all[df_all["valid_all"] == 1].reset_index(drop=True)
    np.random.seed(RANDOM_SEED)
    holdout2obj = df_all.tail(100).reset_index(drop=True)
    print(f"holdout2obj: {len(holdout2obj)} 件", flush=True)

    preloaded = preload_rd_data(holdout2obj)
    gts       = build_gts(holdout2obj)
    n_gt_total = sum(len(v) for v in gts.values())
    print(f"GT 総数: {n_gt_total}", flush=True)

    # --- 最良設定でノイズキャッシュ構築 ---
    print("\n[最良設定] ノイズキャッシュ構築中...", flush=True)
    noise_cache, n_train_cells = build_noise_cache(preloaded, BEST_N_TRAIN, BEST_N_GUARD)
    alpha_best = n_train_cells * (BEST_PFA ** (-1.0 / n_train_cells) - 1.0)
    print(f"  n_train={BEST_N_TRAIN}, n_guard={BEST_N_GUARD}, pfa={BEST_PFA:.0e}"
          f"  →  alpha={alpha_best:.3f}", flush=True)

    # --- CFAR + NMS キャッシュ ---
    print("[最良設定] CFAR + NMS 実行中...", flush=True)
    cached_best = collect_cfar_peaks(noise_cache, gts, alpha_best)
    n_peaks_total = sum(len(c["peaks"]) for c in cached_best)
    print(f"  NMS 後ピーク総数: {n_peaks_total}", flush=True)

    # --- tar_thresh sweep ---
    print("[最良設定] tar_thresh sweep 中...", flush=True)
    df_best = sweep_tar_thresh(cached_best)
    csv_path = os.path.join(OUTPUT_DIR, "cfar_thresh_sweep.csv")
    df_best.to_csv(csv_path, index=False)
    print(f"CSV 保存: {csv_path}", flush=True)

    # --- 基準設定（現在のデフォルト, tar_thresh なし）も計算して比較点に使う ---
    print("\n[基準設定] ノイズキャッシュ構築中...", flush=True)
    nc_default, ntc_default = build_noise_cache(preloaded, DEFAULT_N_TRAIN, DEFAULT_N_GUARD)
    alpha_default = ntc_default * (DEFAULT_PFA ** (-1.0 / ntc_default) - 1.0)
    cached_default = collect_cfar_peaks(nc_default, gts, alpha_default)
    df_default = sweep_tar_thresh(cached_default)

    # --- 結果サマリ ---
    best_row = df_best.loc[df_best["f1"].idxmax()]
    print(f"\n=== 最良 tar_thresh ===")
    print(f"  tar_thresh = {best_row['tar_thresh']:.2f} dB")
    print(f"  P={best_row['precision']:.4f}  R={best_row['recall']:.4f}  F1={best_row['f1']:.4f}")
    print(f"  TP={best_row['tp']}  FP={int(best_row['fp'])}  FN={best_row['fn']}")

    # --- プロット ---
    plot_results(df_best, df_default, n_gt_total)
    print("\n完了。結果は ./cfar_thresh_sweep_results/ に保存されました。", flush=True)


if __name__ == "__main__":
    main()
