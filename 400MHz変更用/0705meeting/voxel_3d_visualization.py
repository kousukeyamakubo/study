"""
voxel_3d_visualization.py

2026-07-05 ミーティングフィードバック1対応:
「統計量でまとめず、クラスごとに 3 次元の分布を可視化した方が良い」

argmax で各クラスに割り当てられたボクセル群を (角度ch, Doppler, Range) の
3D 散布図で表示する。色・サイズ＝確率値、GT は別マーカー（赤い星）で重ねる。

全 300 件を重ねると読めないため 2 案を両方出力する:
  (i)  代表サンプル＋外れ値サンプルの個別 3D 表示   → voxel_3d_samples.png
  (ii) GT 中心の相対座標に平行移動して全サンプル重ね描き → voxel_3d_relative.png

出力: 0705meeting/results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401（projection="3d" の登録に必要）
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== パス =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_detector_narrow_angle.pt")
META_CSV   = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed", "metadata.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 推論結果のキャッシュ（3D の視点調整など再描画のたびに推論し直さないため）
CACHE_PATH = os.path.join(OUTPUT_DIR, "voxel_records_cache.npz")
USE_CACHE  = True

# ===== 定数 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")

# クラスごとの逐次カラーマップ（確率＝濃さ）。既存図の tab:blue / tab:orange に合わせる
CLS_SPECS = [
    # (cls_idx, ラベル, カラーマップ, valid列, 角度キー, dキー, rキー)
    (1, "Cyclist", "Blues",
     "valid_cyclist", "cyclist_true_angle_deg", "cyclist_true_d_idx", "cyclist_true_r_idx"),
    (2, "Vehicle", "Oranges",
     "valid_vehicle", "vehicle_true_angle_deg", "vehicle_true_d_idx", "vehicle_true_r_idx"),
]


# ===== モデル定義（nms_validity_check.py と同一） =====
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


# ===== 推論・ボクセル座標収集 =====
def collect_voxel_records(model, eval_df):
    """
    全サンプルについて推論し、クラスごとに argmax 割り当てボクセルの座標・確率・GT を保持する。
    nms_validity_check.py の collect_nms_stats() を、統計量でなく生の座標を残す形に拡張したもの。

    Returns:
        records: {"cyclist": list[dict], "vehicle": list[dict]}
            dict = {
                "sample_idx": int,
                "coords":    (N, 3) int   — argmax 割り当てボクセル (ch, d, r)
                "probs":     (N,)   float — 各ボクセルの当該クラス確率
                "gt":        (3,)   int   — GT (ch, d, r)。invalid なら None
                "peak":      (3,)   int   — 確率最大ボクセル (ch, d, r)
                "peak_prob": float,
                "gt_dist":   float        — ピーク–GT の L2 距離 [voxel]。GT なしなら nan
            }
    """
    records = {"cyclist": [], "vehicle": []}

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

            for cls_idx, label, _, valid_col, angle_key, d_key, r_key in CLS_SPECS:
                mask = (pred == cls_idx)
                if not mask.any():
                    continue

                coords   = mask.nonzero(as_tuple=False).cpu().numpy()      # (N, 3): ch, d, r
                prob_arr = probs[cls_idx][mask].cpu().numpy()              # (N,)
                peak_i   = int(prob_arr.argmax())
                peak     = coords[peak_i]

                valid_flag = str(row[valid_col]).strip() in ("1", "True")
                if valid_flag:
                    tch = int(np.argmin(np.abs(fa - float(data[angle_key]))))
                    gt  = np.array([tch, int(row[d_key]), int(row[r_key])])
                    gt_dist = float(np.sqrt(((peak - gt)**2).sum()))
                else:
                    gt, gt_dist = None, float("nan")

                records[label.lower()].append({
                    "sample_idx": idx,
                    "coords":     coords,
                    "probs":      prob_arr,
                    "gt":         gt,
                    "peak":       peak,
                    "peak_prob":  float(prob_arr[peak_i]),
                    "gt_dist":    gt_dist,
                })
    return records


def save_cache(records):
    # dict の list は object 配列として pickle 保存する（研究用の再描画高速化目的）
    np.savez(CACHE_PATH,
             cyclist=np.array(records["cyclist"], dtype=object),
             vehicle=np.array(records["vehicle"], dtype=object))
    print(f"Cache saved: {CACHE_PATH}")


def load_cache():
    data = np.load(CACHE_PATH, allow_pickle=True)
    return {"cyclist": list(data["cyclist"]), "vehicle": list(data["vehicle"])}


# ===== サンプル選定 =====
def select_samples(recs):
    """
    代表サンプル1件＋外れ値2件を決定的に選ぶ。
      代表: ピークが GT に近く (<=1 voxel) 高確信 (>=0.9) なもののうち、
            ボクセル数が中央値に最も近いサンプル（「典型的な見え方」を代表させるため）
      外れ値: ピーク–GT 距離の大きい順に 2 件（vehicle の 4〜5 voxel ずれの原因調査対象）
    """
    valid = [r for r in recs if r["gt"] is not None]
    outliers = sorted(valid, key=lambda r: -r["gt_dist"])[:2]

    good = [r for r in valid if r["gt_dist"] <= 1.0 and r["peak_prob"] >= 0.9]
    med  = np.median([len(r["probs"]) for r in good])
    rep  = min(good, key=lambda r: (abs(len(r["probs"]) - med), r["sample_idx"]))
    return rep, outliers


# ===== 可視化 (i): 代表＋外れ値の個別 3D 表示 =====
def draw_voxel_panel(ax, rec, cmap, title):
    c, p = rec["coords"], rec["probs"]
    # x=Range, y=Doppler, z=角度ch。confidence を色と点サイズの両方に載せる
    sc = ax.scatter(c[:, 2], c[:, 1], c[:, 0],
                    c=p, cmap=cmap, vmin=0, vmax=1,
                    s=20 + 100*p, alpha=0.9, depthshade=False)
    gt, peak = rec["gt"], rec["peak"]
    if gt is not None:
        ax.scatter([gt[2]], [gt[1]], [gt[0]], marker="*", s=400,
                   color="red", edgecolors="black", linewidths=0.8,
                   depthshade=False, zorder=5)
    ax.scatter([peak[2]], [peak[1]], [peak[0]], marker="x", s=120,
               color="black", linewidths=2, depthshade=False, zorder=5)

    # 軸範囲は GT とボクセル群を含む範囲＋余白（サンプルごとに位置が違うため固定できない）
    pts = c if gt is None else np.vstack([c, gt[None, :]])
    pad = 3
    ax.set_xlim(pts[:, 2].min() - pad, pts[:, 2].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(max(pts[:, 0].min() - 1, -0.5), min(pts[:, 0].max() + 1, N_FIXED - 0.5))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # 角度chは整数のみ
    ax.set_xlabel("Range bin", fontsize=10, labelpad=2)
    ax.set_ylabel("Doppler bin", fontsize=10, labelpad=2)
    ax.set_zlabel("Angle ch", fontsize=10, labelpad=2)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11)
    return sc


def plot_representative_and_outliers(records):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             subplot_kw={"projection": "3d"})
    for row_i, (cls_idx, label, cmap, *_rest) in enumerate(CLS_SPECS):
        recs = records[label.lower()]
        rep, outliers = select_samples(recs)
        panels = [(rep, "representative")] + [(o, f"outlier #{k+1}") for k, o in enumerate(outliers)]
        sc = None
        for col_i, (rec, kind) in enumerate(panels):
            title = (f"{label} {kind} (sample {rec['sample_idx']})\n"
                     f"n={len(rec['probs'])} voxels, peak p={rec['peak_prob']:.2f}, "
                     f"peak-GT dist={rec['gt_dist']:.1f}")
            sc = draw_voxel_panel(axes[row_i, col_i], rec, cmap, title)
        fig.colorbar(sc, ax=list(axes[row_i, :]), shrink=0.6, pad=0.02,
                     label=f"{label} probability")

    # GT / peak マーカーの凡例（色に依存しない形状で識別できるようにする）
    legend_elems = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="red",
               markeredgecolor="black", markersize=15, label="GT"),
        Line2D([0], [0], marker="x", color="black", linestyle="none",
               markersize=10, markeredgewidth=2, label="peak voxel"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=11)
    fig.suptitle("Argmax-assigned voxels in (Angle ch, Doppler, Range) — "
                 "representative and outlier samples", fontsize=13)

    path = os.path.join(OUTPUT_DIR, "voxel_3d_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ===== 可視化 (ii): GT 中心の相対座標で全サンプル重ね描き =====
def plot_gt_relative_overlay(records):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw={"projection": "3d"})
    ZOOM = 5  # 下段のズーム範囲 [voxel]。減衰分析（2〜3 voxel で急減）が見える範囲

    for col_i, (cls_idx, label, cmap, *_rest) in enumerate(CLS_SPECS):
        recs  = [r for r in records[label.lower()] if r["gt"] is not None]
        # 全サンプルのボクセルを GT 中心の相対座標へ平行移動して連結
        rel   = np.vstack([r["coords"] - r["gt"][None, :] for r in recs])  # (M, 3): dch, dd, dr
        probs = np.concatenate([r["probs"] for r in recs])                 # (M,)

        n_near = int((np.sqrt((rel**2).sum(axis=1)) <= 3).sum())
        stats  = (f"samples: {len(recs)}\nvoxels:  {len(rel)}\n"
                  f"within 3 voxels of GT: {100*n_near/len(rel):.1f}%")

        for row_i, zoom in enumerate([False, True]):
            ax = axes[row_i, col_i]
            if zoom:
                m = np.all(np.abs(rel) <= ZOOM, axis=1)
                r_plot, p_plot = rel[m], probs[m]
            else:
                r_plot, p_plot = rel, probs

            sc = ax.scatter(r_plot[:, 2], r_plot[:, 1], r_plot[:, 0],
                            c=p_plot, cmap=cmap, vmin=0, vmax=1,
                            s=6 + 30*p_plot, alpha=0.35, depthshade=False)
            # GT は原点（全サンプル共通）
            ax.scatter([0], [0], [0], marker="*", s=400, color="red",
                       edgecolors="black", linewidths=0.8, depthshade=False, zorder=5)

            if zoom:
                ax.set_xlim(-ZOOM, ZOOM); ax.set_ylim(-ZOOM, ZOOM); ax.set_zlim(-ZOOM, ZOOM)
                ax.set_title(f"{label}: zoom (+/-{ZOOM} voxels)", fontsize=11)
            else:
                lim = np.abs(rel).max() + 2
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
                ax.set_title(f"{label}: all voxels (GT-centered)", fontsize=11)
                ax.text2D(0.02, 0.95, stats, transform=ax.transAxes,
                          va="top", ha="left", fontsize=9, family="monospace",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_xlabel("dRange [voxel]", fontsize=10, labelpad=2)
            ax.set_ylabel("dDoppler [voxel]", fontsize=10, labelpad=2)
            ax.set_zlabel("dAngle ch [voxel]", fontsize=10, labelpad=2)
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # 角度chは整数のみ
            ax.tick_params(labelsize=8)

        # 縦配置だと z 軸ラベルに重なるため、列ごとに下側へ水平配置する
        fig.colorbar(sc, ax=list(axes[:, col_i]), shrink=0.5, pad=0.06,
                     location="bottom", orientation="horizontal",
                     label=f"{label} probability")

    legend_elems = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="red",
               markeredgecolor="black", markersize=15, label="GT (origin)"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=11)
    fig.suptitle("Argmax-assigned voxels relative to GT (all 300 samples overlaid)",
                 fontsize=13)

    path = os.path.join(OUTPUT_DIR, "voxel_3d_relative.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ===== メイン =====
if __name__ == "__main__":
    if USE_CACHE and os.path.exists(CACHE_PATH):
        print(f"Loading cache: {CACHE_PATH}")
        records = load_cache()
    else:
        print("Loading model ...")
        model = RadarUNet3DSoftmax().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("  done.")

        df = pd.read_csv(META_CSV)
        df = df[df["valid_all"] == 1].reset_index(drop=True)
        print(f"Samples: {len(df)}")

        print("Running inference ...")
        records = collect_voxel_records(model, df)
        save_cache(records)

    for label in ("cyclist", "vehicle"):
        recs  = records[label]
        valid = [r for r in recs if r["gt"] is not None]
        n_vox = sum(len(r["probs"]) for r in recs)
        print(f"\n--- {label} ---")
        print(f"  samples with detection: {len(recs)}, total voxels: {n_vox}")
        dists = np.array([r["gt_dist"] for r in valid])
        print(f"  peak-GT dist: mean={dists.mean():.2f}, max={dists.max():.2f}")

    plot_representative_and_outliers(records)
    plot_gt_relative_overlay(records)
    print("\nAll done.")
