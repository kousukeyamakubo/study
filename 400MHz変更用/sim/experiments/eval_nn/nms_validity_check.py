"""
nms_validity_check.py

NMS 導入の妥当性を確認する3つの分析：
  ① サンプルごとのクラス最大確率分布（ピークが sharp かどうか）
  ② ピーク voxel からの距離 vs 確率プロファイル（周囲への広がり）
  ③ ピーク voxel と GT の距離分布（ピークが正しい場所にあるか）

出力: tolerance_sweep_results/for_paper_300/nms_validity/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== パス =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_detector_narrow_angle.pt")
META_CSV   = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed", "metadata.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tolerance_sweep_results", "for_paper_300", "nms_validity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 定数 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")


# ===== モデル定義 =====
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch=32, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )
    def forward(self, x): return self.block(x)


class RadarUNet3DSoftmax(nn.Module):
    def __init__(self, n_angles=N_FIXED, ch=32, dropout=0.1):
        super().__init__()
        self.encoders   = nn.ModuleList([ConvBlock3D(1 if i == 0 else ch, ch, dropout) for i in range(4)])
        self.pools      = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck = ConvBlock3D(ch, ch, dropout)
        self.upsamples  = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders   = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head   = nn.Conv3d(ch, 3, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        skips, feat = [], x
        for enc, pool in zip(self.encoders, self.pools):
            feat = enc(feat); skips.append(feat); feat = pool(feat)
        feat = self.bottleneck(feat)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            feat = up(feat)
            if feat.shape[-3:] != skip.shape[-3:]:
                feat = F.interpolate(feat, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feat = dec(torch.cat([feat, skip], dim=1))
        return self.seg_head(feat)  # (B, 3, N_FIXED, H, W)


# ===== データ読み込み =====
def load_sample(path):
    data = np.load(path)
    x = np.stack([20*np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                  for i in range(data["rd_maps"].shape[0])], axis=0)
    return x, data


# ===== 推論・データ収集 =====
def collect_nms_stats(model, eval_df):
    """
    Returns:
        cy_peak_probs   : list[float] — サンプルごとの cyclist 最大確率
        ve_peak_probs   : list[float] — サンプルごとの vehicle  最大確率
        cy_dist_prob    : list[(dist, prob)] — cyclist: ピークからの距離と確率
        ve_dist_prob    : list[(dist, prob)] — vehicle:  ピークからの距離と確率
        cy_peak_gt_dist : list[float] — cyclist ピークと GT の L2 距離（voxel単位）
        ve_peak_gt_dist : list[float] — vehicle  ピークと GT の L2 距離（voxel単位）
    """
    cy_peak_probs,   ve_peak_probs   = [], []
    cy_dist_prob,    ve_dist_prob    = [], []
    cy_peak_gt_dist, ve_peak_gt_dist = [], []

    cls_configs = [
        (1, cy_peak_probs, cy_dist_prob, cy_peak_gt_dist,
         "valid_cyclist", "cyclist_true_angle_deg", "cyclist_true_d_idx", "cyclist_true_r_idx"),
        (2, ve_peak_probs, ve_dist_prob, ve_peak_gt_dist,
         "valid_vehicle", "vehicle_true_angle_deg", "vehicle_true_d_idx", "vehicle_true_r_idx"),
    ]

    model.eval()
    with torch.no_grad():
        for idx in range(len(eval_df)):
            if idx % 60 == 0:
                print(f"  {idx}/{len(eval_df)}", flush=True)
            row  = eval_df.iloc[idx]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))

            x, data = load_sample(path)
            fa = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES

            probs = F.softmax(
                model(torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)), dim=1
            )[0]  # (3, N_FIXED, H, W)
            pred = probs.argmax(dim=0)  # (N_FIXED, H, W)

            for (cls_idx, peak_list, dist_prob_list, gt_dist_list,
                 valid_col, angle_key, d_key, r_key) in cls_configs:

                valid_flag = int(str(row[valid_col]).strip() in ("1", "True"))
                mask = (pred == cls_idx)
                if not mask.any():
                    continue

                cls_prob = probs[cls_idx]  # (N_FIXED, H, W)

                # ① サンプルごとのピーク確率
                peak_val = cls_prob[mask].max().item()
                peak_list.append(peak_val)

                # ピーク voxel の座標（同クラス voxel 中で最大確率の位置）
                coords = mask.nonzero(as_tuple=False)  # (N, 3): ch, d, r
                prob_vals = cls_prob[mask]
                peak_idx = prob_vals.argmax().item()
                peak_coord = coords[peak_idx].cpu().numpy()  # (ch, d, r)

                # ② ピークからの距離 vs 確率
                coords_np   = coords.cpu().numpy()
                prob_vals_np = prob_vals.cpu().numpy()
                dists = np.sqrt(((coords_np - peak_coord)**2).sum(axis=1))
                for d, p in zip(dists, prob_vals_np):
                    dist_prob_list.append((float(d), float(p)))

                # ③ ピーク voxel と GT の距離
                if valid_flag:
                    tch   = int(np.argmin(np.abs(fa - float(data[angle_key]))))
                    tru_d = int(row[d_key])
                    tru_r = int(row[r_key])
                    gt_dist = np.sqrt(
                        (peak_coord[0] - tch  )**2 +
                        (peak_coord[1] - tru_d)**2 +
                        (peak_coord[2] - tru_r)**2
                    )
                    gt_dist_list.append(float(gt_dist))

    return (cy_peak_probs, ve_peak_probs,
            cy_dist_prob,  ve_dist_prob,
            cy_peak_gt_dist, ve_peak_gt_dist)


# ===== 可視化 =====
def plot_peak_prob_dist(cy_peak_probs, ve_peak_probs):
    """① サンプルごとのピーク確率ヒストグラム"""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, probs_list, label, color in [
        (axes[0], cy_peak_probs, "Cyclist", "tab:blue"),
        (axes[1], ve_peak_probs, "Vehicle", "tab:orange"),
    ]:
        arr = np.array(probs_list)
        ax.hist(arr, bins=40, range=(0, 1), color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0.9, color="red", linestyle="--", linewidth=1.2, label="p=0.9")
        n = len(arr)
        n90 = (arr >= 0.9).sum()
        n95 = (arr >= 0.95).sum()
        ax.text(0.02, 0.97,
                f"n={n}\n≥0.9:  {n90} ({100*n90/max(n,1):.1f}%)\n≥0.95: {n95} ({100*n95/max(n,1):.1f}%)",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlabel("Peak probability (per sample)", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(f"{label}: per-sample peak probability", fontsize=13)
        ax.legend(fontsize=11)
    plt.suptitle("(1) Per-sample peak probability distribution\n", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "peak_prob_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_dist_vs_prob(cy_dist_prob, ve_dist_prob):
    """② ピークからの距離 vs 確率（散布図 + ビン平均）"""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dp_list, label, color in [
        (axes[0], cy_dist_prob, "Cyclist", "tab:blue"),
        (axes[1], ve_dist_prob, "Vehicle", "tab:orange"),
    ]:
        dists = np.array([x[0] for x in dp_list])
        probs = np.array([x[1] for x in dp_list])

        ax.scatter(dists, probs, alpha=0.05, s=5, color=color)

        # ビン平均（距離 1 voxel 刻み）
        max_d = int(dists.max()) + 1
        bin_edges, bin_means = [], []
        for b in range(max_d):
            m = (dists >= b) & (dists < b + 1)
            if m.sum() > 0:
                bin_edges.append(b + 0.5)
                bin_means.append(probs[m].mean())
        ax.plot(bin_edges, bin_means, color="black", linewidth=2,
                marker="o", markersize=4, label="bin mean")

        ax.set_xlabel("Distance from peak voxel [voxels]", fontsize=13)
        ax.set_ylabel("Probability", fontsize=13)
        ax.set_title(f"{label}: prob vs distance from peak", fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
    plt.suptitle("(2) Probability profile around peak voxel\n", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dist_vs_prob.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_peak_gt_dist(cy_gt_dist, ve_gt_dist):
    """③ ピーク voxel と GT の距離ヒストグラム"""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dists_list, label, color in [
        (axes[0], cy_gt_dist, "Cyclist", "tab:blue"),
        (axes[1], ve_gt_dist, "Vehicle", "tab:orange"),
    ]:
        arr = np.array(dists_list)
        ax.hist(arr, bins=30, range=(0, min(arr.max() + 1, 20)),
                color=color, alpha=0.8, edgecolor="white")
        n = len(arr)
        n0 = (arr == 0).sum()
        n1 = (arr <= 1).sum()
        n3 = (arr <= 3).sum()
        ax.text(0.98, 0.97,
                f"n={n}\n=0:  {n0} ({100*n0/max(n,1):.1f}%)\n≤1: {n1} ({100*n1/max(n,1):.1f}%)\n≤3: {n3} ({100*n3/max(n,1):.1f}%)",
                transform=ax.transAxes, va="top", ha="right", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlabel("L2 distance: peak voxel → GT [voxels]", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(f"{label}: peak voxel to GT distance", fontsize=13)
    plt.suptitle("(3) Distance from peak voxel to GT\n", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "peak_gt_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ===== メイン =====
if __name__ == "__main__":
    print("Loading model ...")
    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("  done.")

    df = pd.read_csv(META_CSV)
    df = df[df["valid_all"] == 1].reset_index(drop=True)
    print(f"Samples: {len(df)}")

    print("Running inference ...")
    (cy_peak_probs, ve_peak_probs,
     cy_dist_prob,  ve_dist_prob,
     cy_peak_gt_dist, ve_peak_gt_dist) = collect_nms_stats(model, df)

    cy_arr = np.array(cy_peak_probs)
    ve_arr = np.array(ve_peak_probs)
    cy_gd  = np.array(cy_peak_gt_dist)
    ve_gd  = np.array(ve_peak_gt_dist)

    print(f"\n--- cyclist ---")
    print(f"  peak probs  : n={len(cy_arr)}, mean={cy_arr.mean():.3f}, median={np.median(cy_arr):.3f}")
    # ※ print文のUnicode記号をASCIIに置換済み（cp932環境対応）
    print(f"  peak->GT dist: n={len(cy_gd)},  mean={cy_gd.mean():.2f},  =0: {(cy_gd==0).sum()},  <=1: {(cy_gd<=1).sum()},  <=3: {(cy_gd<=3).sum()}")
    print(f"\n--- vehicle ---")
    print(f"  peak probs  : n={len(ve_arr)}, mean={ve_arr.mean():.3f}, median={np.median(ve_arr):.3f}")
    print(f"  peak->GT dist: n={len(ve_gd)},  mean={ve_gd.mean():.2f},  =0: {(ve_gd==0).sum()},  <=1: {(ve_gd<=1).sum()},  <=3: {(ve_gd<=3).sum()}")

    plot_peak_prob_dist(cy_peak_probs, ve_peak_probs)
    plot_dist_vs_prob(cy_dist_prob, ve_dist_prob)
    plot_peak_gt_dist(cy_peak_gt_dist, ve_peak_gt_dist)
    print("\nAll done.")
