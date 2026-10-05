"""
angle_distribution.py

learn_dataset_narrow_angle_fixed (for_paper_300 と同一シミュレーション) 全300サンプルに
best_detector_narrow_angle.pt を適用し、cyclist・vehicle それぞれの予測角度チャネル
ヒストグラムを生成する。

出力: tolerance_sweep_results/for_paper_300/angle_distribution.png
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

MODEL_PATH     = os.path.join(ROOT_DIR, "models", "best_detector_narrow_angle.pt")
META_CSV       = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed", "metadata.csv")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "tolerance_sweep_results", "for_paper_300")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 定数 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)  # narrow_angle_fixed の角度グリッド [deg]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")


# ===== モデル定義（tolerance_sweep.py と同一） =====
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
    # 各角度チャネルの振幅を dB 変換
    x = np.stack([20*np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                  for i in range(data["rd_maps"].shape[0])], axis=0)
    return x, data


# ===== 推論・角度チャネル収集 =====
def collect_angle_channels(model, eval_df):
    """
    全サンプルについて推論し、各クラスに割り当てられた全ボクセルの
    角度チャネルインデックスと最大確率値を収集する。

    Returns:
        cy_pred_chs  : list[int]   — 全サンプル全ボクセルの cyclist 予測角度チャネル
        ve_pred_chs  : list[int]   — 全サンプル全ボクセルの vehicle  予測角度チャネル
        cy_gt_chs    : list[int]   — valid サンプルの cyclist GT 角度チャネル（1件/サンプル）
        ve_gt_chs    : list[int]   — valid サンプルの vehicle  GT 角度チャネル（1件/サンプル）
        cy_pred_probs: list[float] — argmax で cyclist と判定されたボクセルの確率値
        ve_pred_probs: list[float] — argmax で vehicle  と判定されたボクセルの確率値
    """
    cy_pred_chs, ve_pred_chs = [], []
    cy_gt_chs,   ve_gt_chs   = [], []
    cy_pred_probs, ve_pred_probs = [], []

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
            pred = probs.argmax(dim=0)  # (N_FIXED, H, W) — per-voxel argmax

            # 各クラスに割り当てられた全ボクセルの角度チャネルと確率値を記録
            for cls_idx, ch_list, prob_list in [(1, cy_pred_chs, cy_pred_probs),
                                                (2, ve_pred_chs, ve_pred_probs)]:
                mask = (pred == cls_idx)
                if mask.any():
                    chs = mask.nonzero(as_tuple=False)[:, 0].cpu().numpy()
                    ch_list.extend(chs.tolist())
                    # argmax で選ばれたクラスの確率値を取得
                    p = probs[cls_idx][mask].cpu().numpy()
                    prob_list.extend(p.tolist())

            # GT 角度チャネル（valid サンプルのみ）
            vcy = int(str(row["valid_cyclist"]).strip() in ("1", "True"))
            vve = int(str(row["valid_vehicle"]).strip() in ("1", "True"))
            if vcy:
                tch = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                cy_gt_chs.append(tch)
            if vve:
                tch = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                ve_gt_chs.append(tch)

    return cy_pred_chs, ve_pred_chs, cy_gt_chs, ve_gt_chs, cy_pred_probs, ve_pred_probs


# ===== 可視化 =====
def plot_angle_distribution(cy_pred_chs, ve_pred_chs, cy_gt_chs, ve_gt_chs):
    """
    cyclist / vehicle それぞれについて、予測角度チャネルのヒストグラムと
    GT 角度チャネルの分布を重ねて表示する。
    """
    xticks   = np.arange(N_FIXED)
    xlabels  = [f"{FIXED_ANGLES[i]:.2f}°" for i in range(N_FIXED)]
    bar_w    = 0.35

    # 両クラスを集計してから共通 y 上限を決める
    all_counts = []
    count_pairs = []
    for pred_chs, gt_chs in [(cy_pred_chs, cy_gt_chs), (ve_pred_chs, ve_gt_chs)]:
        pc = np.array([pred_chs.count(i) for i in range(N_FIXED)], dtype=float)
        gc = np.array([gt_chs.count(i)   for i in range(N_FIXED)], dtype=float)
        count_pairs.append((pc, gc))
        all_counts.extend([pc.max(), gc.max()])
    y_max = max(all_counts) * 1.15  # 上に余白

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    specs = [
        (axes[0], count_pairs[0][0], count_pairs[0][1], cy_pred_chs, cy_gt_chs, "Cyclist",  "tab:blue"),
        (axes[1], count_pairs[1][0], count_pairs[1][1], ve_pred_chs, ve_gt_chs, "Vehicle",  "tab:orange"),
    ]
    for ax, pred_counts, gt_counts, pred_chs, gt_chs, label, color in specs:
        ax.bar(xticks - bar_w/2, pred_counts, width=bar_w,
               color=color, alpha=0.8, label="Predicted (all voxels)")
        ax.bar(xticks + bar_w/2, gt_counts,   width=bar_w,
               color="gray", alpha=0.7, label="GT (valid samples)")

        ax.set_ylim(0, y_max)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("Angle channel [°]", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        ax.set_title(f"{label}: Predicted vs GT angle distribution", fontsize=14)
        ax.legend(fontsize=12)

        # 統計サマリ
        n_pred = len(pred_chs)
        n_gt   = len(gt_chs)
        ax.text(0.02, 0.97,
                f"pred voxels: {n_pred}\nGT samples:  {n_gt}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=11, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.suptitle(
        "Predicted angle distribution\n",
        fontsize=12
    )
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "angle_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.show()


def plot_prob_distribution(cy_pred_probs, ve_pred_probs):
    """
    argmax で各クラスに割り当てられたボクセルの確率値ヒストグラムを描画する。
    確率が低いボクセルが多ければ「閾値なし argmax では過剰検出になっている」ことが分かる。
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    specs = [
        (axes[0], cy_pred_probs, "Cyclist",  "tab:blue"),
        (axes[1], ve_pred_probs, "Vehicle",  "tab:orange"),
    ]
    for ax, probs_list, label, color in specs:
        arr = np.array(probs_list)
        ax.hist(arr, bins=50, range=(0, 1), color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="p=0.5")
        ax.set_xlabel("Predicted probability", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title(f"{label}: probability of argmax-selected voxels", fontsize=13)
        ax.legend(fontsize=11)

        # 閾値別のボクセル数を表示
        n_total = len(arr)
        n_above50 = (arr >= 0.5).sum()
        n_above80 = (arr >= 0.8).sum()
        ax.text(0.98, 0.97,
                f"total: {n_total}\n≥0.5: {n_above50} ({100*n_above50/max(n_total,1):.1f}%)\n≥0.8: {n_above80} ({100*n_above80/max(n_total,1):.1f}%)",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.suptitle("Probability distribution of argmax-selected voxels\n", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "argmax_prob_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ===== メイン =====
if __name__ == "__main__":
    # モデルロード
    print("Loading model ...")
    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("  done.")

    # データセット（valid_all == 1 のみ）
    df = pd.read_csv(META_CSV)
    df = df[df["valid_all"] == 1].reset_index(drop=True)
    print(f"Samples: {len(df)}")

    # 推論・収集
    print("Running inference ...")
    cy_pred, ve_pred, cy_gt, ve_gt, cy_probs, ve_probs = collect_angle_channels(model, df)
    print(f"  cyclist pred voxels: {len(cy_pred)},  GT samples: {len(cy_gt)}")
    print(f"  vehicle  pred voxels: {len(ve_pred)},  GT samples: {len(ve_gt)}")

    # 角度チャネル別集計の表示
    print("\n--- cyclist 予測チャネル分布 ---")
    for i, ang in enumerate(FIXED_ANGLES):
        cnt = cy_pred.count(i)
        print(f"  ch{i:2d} ({ang:.2f}°): {cnt:5d} voxels")

    print("\n--- vehicle 予測チャネル分布 ---")
    for i, ang in enumerate(FIXED_ANGLES):
        cnt = ve_pred.count(i)
        print(f"  ch{i:2d} ({ang:.2f}°): {cnt:5d} voxels")

    # プロット保存
    plot_angle_distribution(cy_pred, ve_pred, cy_gt, ve_gt)
    plot_prob_distribution(cy_probs, ve_probs)
