"""
PR曲線・AP評価スクリプト — 狭角度グリッド実験版

評価モデル : best_detector_narrow_angle.pt
評価データ : narrow_angle_fixed holdout (100件)
スコア     : クラスkのsoftmax確率の全ボクセル最大値
ヒット条件 : |Δch| <= A_TOL, |Δd| <= D_TOL, |Δr| <= R_TOL
AP算出     : PASCAL VOC方式（スコア降順ソート → PR曲線面積）
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== パス =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

# 狭角度グリッド実験用: パスと角度設定を変更
MODEL_PATH      = os.path.join(ROOT_DIR, "models", "best_detector_narrow_angle.pt")
SINGLE_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_single", "metadata.csv")
FIXED_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed",  "metadata.csv")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "pr_curve_eval_narrow_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 定数 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)  # 狭角度グリッド: 1°〜4°
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42

A_TOL = 1
D_TOL = 2
R_TOL = 3

print(f"DEVICE: {DEVICE}")
print(f"hit criterion: A_TOL={A_TOL}, D_TOL={D_TOL}, R_TOL={R_TOL}")


# ===== モデル定義（check.ipynb と同一） =====
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=32, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class RadarUNet3DSoftmax(nn.Module):
    def __init__(self, n_angles=N_FIXED, fixed_channels=32, dropout=0.1):
        super().__init__()
        self.n_angles = n_angles
        self.encoders = nn.ModuleList([
            ConvBlock3D(1,              fixed_channels, dropout),
            ConvBlock3D(fixed_channels, fixed_channels, dropout),
            ConvBlock3D(fixed_channels, fixed_channels, dropout),
            ConvBlock3D(fixed_channels, fixed_channels, dropout),
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) for _ in range(4)
        ])
        self.bottleneck = ConvBlock3D(fixed_channels, fixed_channels, dropout)
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            for _ in range(4)
        ])
        self.decoders = nn.ModuleList([
            ConvBlock3D(fixed_channels * 2, fixed_channels, dropout)
            for _ in range(4)
        ])
        self.seg_head = nn.Conv3d(fixed_channels, 3, kernel_size=1)

    def forward(self, x):
        # x: (B, N_FIXED, H, W) → (B, 1, N_FIXED, H, W)
        x_3d = x.unsqueeze(1)
        skips = []
        feat = x_3d
        for enc, pool in zip(self.encoders, self.pools):
            feat = enc(feat)
            skips.append(feat)
            feat = pool(feat)
        feat = self.bottleneck(feat)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            feat = up(feat)
            if feat.shape[-3:] != skip.shape[-3:]:
                feat = F.interpolate(feat, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feat = torch.cat([feat, skip], dim=1)
            feat = dec(feat)
        return self.seg_head(feat)  # (B, 3, N_FIXED, H, W)


# ===== データ準備 =====
def load_sample(npz_path):
    data = np.load(npz_path)
    rd_maps = data["rd_maps"]
    x = np.stack(
        [20.0 * np.log10(np.maximum(np.abs(rd_maps[i]).astype(np.float32), 1e-12))
         for i in range(rd_maps.shape[0])],
        axis=0,
    )  # (N_FIXED, H, W)
    return x, data


def build_eval_df():
    """narrow_angle_fixed holdout (100件) — 2物体シーンのみ（cy+ve 同時存在）"""
    fixed_df = pd.read_csv(FIXED_META_CSV)
    fixed_df = fixed_df[fixed_df["valid_all"] == 1].reset_index(drop=True)
    fixed_df = fixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return fixed_df.iloc[200:].reset_index(drop=True)


# ===== 推論・スコア収集 =====
def collect_detections(model, eval_df):
    """
    per-voxel argmax で各クラスに割り当てられた全ボクセルを検出として収集する。
    1サンプルから複数の検出が生じうる。GT 未検出サンプルは検出エントリなし。
    戻り値: cy_dets, ve_dets (各 list of {'score', 'is_hit', 'sample_idx'}), n_gt_cy, n_gt_ve
      is_hit : GT 許容範囲内かどうか（compute_ap 内で greedy matching を適用）
    """
    cy_dets, ve_dets = [], []
    n_gt_cy, n_gt_ve = 0, 0

    model.eval()
    with torch.no_grad():
        for idx in range(len(eval_df)):
            if idx % 50 == 0:
                print(f"  {idx}/{len(eval_df)}", flush=True)
            row = eval_df.iloc[idx]
            npz_path = row["file"]
            if not os.path.isabs(npz_path):
                npz_path = os.path.normpath(os.path.join(ROOT_DIR, npz_path))

            x, data = load_sample(npz_path)
            fa = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES

            valid_cy = int(str(row["valid_cyclist"]).strip() in ("1", "True"))
            valid_ve = int(str(row["valid_vehicle"]).strip() in ("1", "True"))

            x_t   = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
            probs = F.softmax(model(x_t), dim=1)[0]  # (3, N_FIXED, H, W)
            pred  = probs.argmax(dim=0)               # (N_FIXED, H, W) — per-voxel argmax

            def voxels_for_class(cls_idx):
                # decode_softmax_detections と同じ: argmax 割り当て済み全ボクセルを返す
                mask = (pred == cls_idx)
                if not mask.any():
                    return []
                return [(int(ij[0]), int(ij[1]), int(ij[2]),
                         probs[cls_idx, int(ij[0]), int(ij[1]), int(ij[2])].item())
                        for ij in mask.nonzero(as_tuple=False)]

            # --- cy ---
            if valid_cy:
                n_gt_cy += 1
                true_d  = int(row["cyclist_true_d_idx"])
                true_r  = int(row["cyclist_true_r_idx"])
                true_ch = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                for ch, d, r, sc in voxels_for_class(1):
                    is_hit = (abs(ch - true_ch) <= A_TOL and
                              abs(d  - true_d)  <= D_TOL and
                              abs(r  - true_r)  <= R_TOL)
                    cy_dets.append({"score": sc, "is_hit": is_hit, "sample_idx": idx})
            else:
                for ch, d, r, sc in voxels_for_class(1):
                    cy_dets.append({"score": sc, "is_hit": False, "sample_idx": idx})

            # --- ve ---
            if valid_ve:
                n_gt_ve += 1
                true_d  = int(row["vehicle_true_d_idx"])
                true_r  = int(row["vehicle_true_r_idx"])
                true_ch = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                for ch, d, r, sc in voxels_for_class(2):
                    is_hit = (abs(ch - true_ch) <= A_TOL and
                              abs(d  - true_d)  <= D_TOL and
                              abs(r  - true_r)  <= R_TOL)
                    ve_dets.append({"score": sc, "is_hit": is_hit, "sample_idx": idx})
            else:
                for ch, d, r, sc in voxels_for_class(2):
                    ve_dets.append({"score": sc, "is_hit": False, "sample_idx": idx})

    return cy_dets, ve_dets, n_gt_cy, n_gt_ve


# ===== AP計算（PASCAL VOC方式 + greedy matching）=====
def compute_ap(detections, n_gt):
    """
    detections : list of {'score': float, 'is_hit': bool, 'sample_idx': int}
    n_gt       : GT 総数
    スコア降順でソート後、各 GT に対して最初にヒットした検出のみ TP とする。
    同一サンプルの複数ボクセルが GT 範囲内に入っても TP は 1 回だけカウント。
    returns    : AP (float), precisions, recalls (各 np.ndarray, 先頭に (1.0, 0.0) を付加)
    """
    sorted_dets = sorted(detections, key=lambda x: -x["score"])
    matched = set()  # マッチ済み GT の sample_idx
    tp = 0
    precs, recs = [], []
    for i, det in enumerate(sorted_dets):
        if det["is_hit"] and det["sample_idx"] not in matched:
            tp += 1
            matched.add(det["sample_idx"])
        precs.append(tp / (i + 1))
        recs.append(tp / n_gt if n_gt > 0 else 0.0)

    # PR曲線の先頭に (precision=1, recall=0) を付加
    precs = np.concatenate([[1.0], precs])
    recs  = np.concatenate([[0.0], recs])

    # AP = Σ p_i * Δr_i
    ap = float(np.sum((recs[1:] - recs[:-1]) * precs[1:]))
    return ap, precs, recs


# ===== メイン =====
model = RadarUNet3DSoftmax(n_angles=N_FIXED).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"model loaded: {os.path.basename(MODEL_PATH)}")

eval_df = build_eval_df()
print(f"eval samples: {len(eval_df)}")

print("running inference...")
cy_dets, ve_dets, n_gt_cy, n_gt_ve = collect_detections(model, eval_df)
print(f"GT: cy={n_gt_cy}, ve={n_gt_ve}")

ap_cy, precs_cy, recs_cy = compute_ap(cy_dets, n_gt_cy)
ap_ve, precs_ve, recs_ve = compute_ap(ve_dets, n_gt_ve)
mAP = (ap_cy + ap_ve) / 2.0

print(f"\nAP_cy = {ap_cy:.4f}")
print(f"AP_ve = {ap_ve:.4f}")
print(f"mAP   = {mAP:.4f}")

# ===== PR曲線プロット =====
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, precs, recs, ap, label, n_gt, color in [
    (axes[0], precs_cy, recs_cy, ap_cy, "Cyclist", n_gt_cy, "tab:blue"),
    (axes[1], precs_ve, recs_ve, ap_ve, "Vehicle",  n_gt_ve, "tab:orange"),
]:
    ax.step(recs, precs, where="post", color=color, lw=2, label=f"AP={ap:.4f}")
    ax.fill_between(recs, precs, step="post", alpha=0.15, color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{label}  AP={ap:.4f}  (nGT={n_gt})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"PR Curve — Narrow Angle  mAP={mAP:.4f}  "
    f"(A_TOL={A_TOL}, D_TOL={D_TOL}, R_TOL={R_TOL})",
    y=1.02,
)
plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, "pr_curve.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"saved: {out_png}")

# ===== メトリクス保存 =====
metrics = {
    "model":    os.path.basename(MODEL_PATH),
    "n_eval":   len(eval_df),
    "n_gt_cy":  n_gt_cy,
    "n_gt_ve":  n_gt_ve,
    "A_TOL":    A_TOL,
    "D_TOL":    D_TOL,
    "R_TOL":    R_TOL,
    "AP_cy":    round(ap_cy, 4),
    "AP_ve":    round(ap_ve, 4),
    "mAP":      round(mAP,   4),
}
out_json = os.path.join(OUTPUT_DIR, "metrics.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"saved: {out_json}")
