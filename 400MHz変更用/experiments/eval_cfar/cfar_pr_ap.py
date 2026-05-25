"""cfar_pr_ap_narrow.py

cfar_pr_ap.py の narrow_angle_fixed 版。
データセットを learn_dataset_narrow_angle_fixed（角度 1°〜4°）に変更し，
sweep_pfa を 7 種類に拡張して実行する。
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
MIXED_META_CSV  = "../../learn_dataset_narrow_angle_fixed/metadata.csv"
OUTPUT_DIR      = "./for_paper"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)  # narrow_angle_fixed の角度グリッド
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

TAR_THRESH_MIN = -15.0   # PR 曲線の最も厳しい端点 [dB]

# ===== sweep 範囲 =====
SWEEP_N_TRAIN_A = [1, 2, 3]
SWEEP_N_GUARD_A = [1]
SWEEP_N_TRAIN_D = [1, 2, 3, 5]
SWEEP_N_GUARD_D = [1, 2]
SWEEP_N_TRAIN_R = [5]
SWEEP_N_GUARD_R = [2, 3]
SWEEP_PFA       = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

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
        x      = torch.from_numpy(rd_lin).unsqueeze(0).unsqueeze(0)
        x_pad  = F.pad(x,     (pad_r, pad_r, 0, 0, 0, 0),     mode="replicate")
        x_pad  = F.pad(x_pad, (0, 0, pad_d, pad_d, 0, 0),     mode="circular")
        x_pad  = F.pad(x_pad, (0, 0, 0, 0, pad_a, pad_a),     mode="replicate")
        noise_est = (F.conv3d(x_pad, kernel) / n_train_cells)[0, 0].numpy()
        noise_cache.append({"x_np": rd_lin, "noise_est": noise_est, "rd_db": item["rd_db"]})
    return noise_cache, n_train_cells


# ===== NMS =====
def nms_on_cfar_mask(mask, rd_db):
    locs = np.argwhere(mask)
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
    return peaks


# ===== CFAR + NMS キャッシュ =====
def collect_cfar_peaks(noise_cache, gts, alpha):
    result = []
    for i, nc in enumerate(noise_cache):
        mask  = nc["x_np"] > alpha * nc["noise_est"]
        peaks = nms_on_cfar_mask(mask, nc["rd_db"])
        result.append({"peaks": peaks, "gts": gts.get(i, [])})
    return result


# ===== 閾値適用評価 =====
def evaluate_with_thresh(cached, tar_thresh):
    """
    tar_thresh=None: 全ピーク使用
    tar_thresh=X:    rd_db >= X のピークのみ使用
    """
    tp = fp = fn = 0
    for item in cached:
        peaks = [(a, d, r) for a, d, r, s in item["peaks"]] if tar_thresh is None \
                else [(a, d, r) for a, d, r, s in item["peaks"] if s >= tar_thresh]
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


# ===== PR 曲線生成 + AP 計算 =====
def compute_pr_curve(cached):
    """
    tar_thresh を -15 dB（最厳） → None（全ピーク）方向にスイープして PR 曲線を生成。
    thresholds:
      1. TAR_THRESH_MIN = -15 dB （最も厳しい端点）
      2. -15 dB 未満の全ユニーク rd_db 値（降順）→ 弱いピークを1つずつ追加
      3. None （全ピーク, 最も許容的な端点）
    AP: 台形則で PR 曲線の面積を計算。
    """
    all_scores = sorted(
        set(s for item in cached for _, _, _, s in item["peaks"]),
        reverse=True,  # 高い（厳しい）から低い順
    )
    # 閾値候補: TAR_THRESH_MIN から None まで
    thresholds: list = [TAR_THRESH_MIN]
    thresholds += [s for s in all_scores if s < TAR_THRESH_MIN]  # 中間点
    thresholds.append(None)  # 全ピーク

    precisions, recalls = [], []
    for tau in thresholds:
        p, r, _, _, _, _ = evaluate_with_thresh(cached, tau)
        precisions.append(p)
        recalls.append(r)

    # AP: recall 昇順に並び替えて台形則
    pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    r_arr = np.array([x[0] for x in pairs])
    p_arr = np.array([x[1] for x in pairs])
    ap = float(np.trapz(p_arr, r_arr))

    return np.array(recalls), np.array(precisions), ap


# ===== メイン sweep =====
def run_sweep(preloaded_data, gts):
    configs = list(itertools.product(
        SWEEP_N_TRAIN_A, SWEEP_N_GUARD_A,
        SWEEP_N_TRAIN_D, SWEEP_N_GUARD_D,
        SWEEP_N_TRAIN_R, SWEEP_N_GUARD_R,
    ))
    n_configs = len(configs)
    n_total   = n_configs * len(SWEEP_PFA)
    print(f"(n_train,n_guard): {n_configs}  x  pfa: {len(SWEEP_PFA)}  =  {n_total} PR curves", flush=True)

    best_ap     = -1.0
    best_record = None
    best_curve  = None   # (recalls, precisions)
    ap_records  = []

    for ci, (ta, ga, td, gd, tr, gr) in enumerate(configs):
        noise_cache, n_train_cells = build_noise_cache(
            preloaded_data, (ta, td, tr), (ga, gd, gr))

        for pfa in SWEEP_PFA:
            alpha  = n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0)
            cached = collect_cfar_peaks(noise_cache, gts, alpha)
            recalls, precisions, ap = compute_pr_curve(cached)

            ap_records.append({
                "n_train_a": ta, "n_guard_a": ga,
                "n_train_d": td, "n_guard_d": gd,
                "n_train_r": tr, "n_guard_r": gr,
                "pfa": pfa, "ap": ap,
            })
            if ap > best_ap:
                best_ap     = ap
                best_record = ap_records[-1].copy()
                best_curve  = (recalls.copy(), precisions.copy())

        pct = (ci + 1) / n_configs * 100
        print(f"  [{pct:5.1f}%] config {ci+1}/{n_configs}"
              f"  ta={ta} td={td} tr={tr} gr={gr}", flush=True)

    return pd.DataFrame(ap_records), best_record, best_curve


# ===== プロット =====
def plot_results(best_record, best_curve, top_records, n_gt_total):
    r_best, p_best = best_curve

    FONT_LABEL  = 18
    FONT_TICK   = 16
    FONT_LEGEND = 14
    FONT_F1     = 12

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(r_best, p_best, color="steelblue", lw=2,
            label=f"Best config  AP={best_record['ap']:.4f}\n"
                  f"ta={int(best_record['n_train_a'])} "
                  f"td={int(best_record['n_train_d'])} "
                  f"tr={int(best_record['n_train_r'])} "
                  f"gr={int(best_record['n_guard_r'])} "
                  f"pfa={best_record['pfa']:.0e}")

    # F1 等高線（F1 値を一定にする PR 上の軌跡）
    for f1_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r_arr = np.linspace(0.01, 1.0, 300)
        p_arr = f1_val * r_arr / (2 * r_arr - f1_val)
        valid = (p_arr > 0) & (p_arr <= 1)
        ax.plot(r_arr[valid], p_arr[valid], "k--", lw=0.8, alpha=0.4)
        idx = np.argmin(np.abs(r_arr[valid] - 0.9))
        ax.text(r_arr[valid][idx], p_arr[valid][idx],
                f"F1={f1_val:.1f}", fontsize=FONT_F1, color="gray")

    ax.set_xlabel("Recall",    fontsize=FONT_LABEL)
    ax.set_ylabel("Precision", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(True)
    ax.set_title("Precision-Recall Curve (CFAR)", fontsize=FONT_LABEL)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cfar_pr_ap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"plot saved: {out}", flush=True)

    # --- 棒グラフ: AP 上位 10 設定 ---
    FONT_LABEL_B  = 18
    FONT_TICK_B   = 14
    FONT_VAL_B    = 13

    top10 = top_records.nlargest(10, "ap")
    labels = [
        f"ta={int(r.n_train_a)}, td={int(r.n_train_d)}, "
        f"tr={int(r.n_train_r)}, gr={int(r.n_guard_r)}, pfa={r.pfa:.0e}"
        for _, r in top10.iterrows()
    ]
    ap_max = top10["ap"].max()

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    bars = ax2.barh(range(len(top10)), top10["ap"].values, color="steelblue", alpha=0.7)
    ax2.set_yticks(range(len(top10)))
    ax2.set_yticklabels(labels, fontsize=FONT_TICK_B)
    ax2.invert_yaxis()
    ax2.set_xlabel("AP", fontsize=FONT_LABEL_B)
    ax2.tick_params(axis="x", labelsize=FONT_TICK_B)
    ax2.set_title("Top-10 CFAR Configurations by AP", fontsize=FONT_LABEL_B)
    ax2.grid(True, axis="x")
    for bar, val in zip(bars, top10["ap"].values):
        ax2.text(val - ap_max * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha="right",
                 fontsize=FONT_VAL_B, color="white", fontweight="bold")
    ax2.set_xlim(0, ap_max * 1.05)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "cfar_ap_bar.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"plot saved: {out2}", flush=True)


# ===== メイン =====
def main():
    print("loading data...", flush=True)
    df_all = pd.read_csv(MIXED_META_CSV)
    df_all = df_all[df_all["valid_all"] == 1].reset_index(drop=True)
    np.random.seed(RANDOM_SEED)
    holdout2obj = df_all.tail(100).reset_index(drop=True)
    print(f"holdout2obj: {len(holdout2obj)} samples", flush=True)

    preloaded  = preload_rd_data(holdout2obj)
    gts        = build_gts(holdout2obj)
    n_gt_total = sum(len(v) for v in gts.values())
    print(f"GT total: {n_gt_total}", flush=True)

    print("\nrunning sweep...", flush=True)
    df_ap, best_record, best_curve = run_sweep(preloaded, gts)

    csv_path = os.path.join(OUTPUT_DIR, "cfar_pr_ap.csv")
    df_ap.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}", flush=True)

    print("\n=== top-5 AP ===")
    for _, row in df_ap.nlargest(5, "ap").iterrows():
        print(f"  ta={int(row.n_train_a)} td={int(row.n_train_d)}"
              f" tr={int(row.n_train_r)} gr={int(row.n_guard_r)}"
              f" pfa={row.pfa:.0e}  AP={row.ap:.4f}")

    plot_results(best_record, best_curve, df_ap, n_gt_total)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
