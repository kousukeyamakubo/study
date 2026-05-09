"""
過学習チェック: train/holdout Precision 比較 + 学習曲線

対象モデル:
  - D_tk3_fp0.001_mixed  (top-K, K=3, mixed学習)
  - D_dw_fp0.001_single  (daware, single学習)
  - D_dw_fp0.001_mixed   (daware, mixed学習)

出力:
  eval_overfitting_results/
    learning_curves.png   -- train/val loss & recall の推移
    precision_comparison.png -- train vs holdout の Precision/Recall
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 定数 =====
SINGLE_META_CSV = "../../learn_dataset_single_object/metadata.csv"
MIXED_META_CSV  = "../../learn_dataset_fixed_angle/metadata.csv"
V3_RESULTS_DIR  = "../sweep_v3/sweep_experiment_results_v3"
OUTPUT_DIR      = "./eval_overfitting_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 8
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42
NMS_THR      = 0.5

TARGET_RUNS = [
    "D_tk3_fp0.001_mixed",
    "D_dw_fp0.001_single",
    "D_dw_fp0.001_mixed",
]
COLORS = {
    "D_tk3_fp0.001_mixed":  ("tab:blue",   "topK (K=3, mixed)"),
    "D_dw_fp0.001_single":  ("tab:orange", "daware (single)"),
    "D_dw_fp0.001_mixed":   ("tab:green",  "daware (mixed)"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)


# ===== モデル =====
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
        self.encoders   = nn.ModuleList([ConvBlock3D(1,ch,dropout), ConvBlock3D(ch,ch,dropout),
                                         ConvBlock3D(ch,ch,dropout), ConvBlock3D(ch,ch,dropout)])
        self.pools      = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck = ConvBlock3D(ch, ch, dropout)
        self.upsamples  = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear",
                                                      align_corners=False) for _ in range(4)])
        self.decoders   = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head   = nn.Conv3d(ch, 3, 1)

    def forward(self, x):
        # (B, N_FIXED, H, W) -> (B, 1, N_FIXED, H, W)
        x3 = x.unsqueeze(1)
        skips, feat = [], x3
        for enc, pool in zip(self.encoders, self.pools):
            feat = enc(feat); skips.append(feat); feat = pool(feat)
        feat = self.bottleneck(feat)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            feat = up(feat)
            if feat.shape[-3:] != skip.shape[-3:]:
                feat = F.interpolate(feat, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feat = dec(torch.cat([feat, skip], dim=1))
        return self.seg_head(feat)  # (B, 3, N_FIXED, H, W)


# スクリプトの2階層上がデータセットのルート
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

def resolve_path(file_col):
    if os.path.isabs(file_col):
        return file_col
    return os.path.normpath(os.path.join(ROOT_DIR, file_col))


# ===== データ読み込み =====
def load_rd_maps(npz_path):
    d = np.load(npz_path)
    return np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                     for i in range(d["rd_maps"].shape[0])], axis=0)

def decode_detections(logits, threshold=0.5):
    probs = F.softmax(logits, dim=1)
    def _nms(pmap, thr):
        p = pmap.clone(); N, H, W = p.shape; dets = []
        while p.max().item() >= thr:
            fi = torch.argmax(p).item()
            ch, rem = fi // (H*W), fi % (H*W)
            d, r = rem // W, rem % W
            dets.append((ch, d, r))
            p[max(ch-1,0):ch+2, max(d-3,0):d+4, max(r-1,0):r+2] = 0.
        return dets
    return _nms(probs[0,1], threshold), _nms(probs[0,2], threshold)


# ===== Precision/Recall 計算 =====
def compute_precision_recall(model, df, thr=NMS_THR):
    """
    モデルを df で推論し cy/ve の Precision と Recall を計算。
    各サンプルは GT が高々1個/クラスなので、マッチした最初の検出のみ TP とする。
    """
    cy_tp = cy_fp = cy_gt = 0
    ve_tp = ve_fp = ve_gt = 0

    model.eval()
    for i in range(len(df)):
        row = df.iloc[i]
        npz = resolve_path(row["file"])

        # 角度チャンネルのGTをnpzから取得
        d_npz = np.load(npz)
        fa = d_npz["fixed_angles"].astype(float) if "fixed_angles" in d_npz else FIXED_ANGLES
        cy_ach = int(np.argmin(np.abs(fa - float(d_npz["cyclist_true_angle_deg"])))) \
                 if "cyclist_true_angle_deg" in d_npz else None
        ve_ach = int(np.argmin(np.abs(fa - float(d_npz["vehicle_true_angle_deg"])))) \
                 if "vehicle_true_angle_deg" in d_npz else None

        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)

        vcy = 1 if str(row.get("valid_cyclist", "0")).strip() in ("1", "True") else 0
        vve = 1 if str(row.get("valid_vehicle",  "0")).strip() in ("1", "True") else 0

        def is_match(dets, true_d, true_r, true_ch):
            for ch, d, r in dets:
                if true_ch is not None and abs(ch - true_ch) > A_TOL:
                    continue
                if abs(d - true_d) <= D_TOL and abs(r - true_r) <= R_TOL:
                    return True
            return False

        # Cyclist
        if vcy:
            cy_gt += 1
            ctd, ctr = int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])
            matched = is_match(cy_dets, ctd, ctr, cy_ach)
            if matched:
                cy_tp += 1
                cy_fp += len(cy_dets) - 1
            else:
                cy_fp += len(cy_dets)
        else:
            cy_fp += len(cy_dets)

        # Vehicle
        if vve:
            ve_gt += 1
            vtd, vtr = int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])
            matched = is_match(ve_dets, vtd, vtr, ve_ach)
            if matched:
                ve_tp += 1
                ve_fp += len(ve_dets) - 1
            else:
                ve_fp += len(ve_dets)
        else:
            ve_fp += len(ve_dets)

    cy_prec = cy_tp / max(cy_tp + cy_fp, 1)
    cy_rec  = cy_tp / max(cy_gt, 1)
    ve_prec = ve_tp / max(ve_tp + ve_fp, 1)
    ve_rec  = ve_tp / max(ve_gt, 1)

    return {
        "cy_precision": cy_prec, "cy_recall": cy_rec,
        "ve_precision": ve_prec, "ve_recall": ve_rec,
        "cy_tp": cy_tp, "cy_fp": cy_fp, "cy_gt": cy_gt,
        "ve_tp": ve_tp, "ve_fp": ve_fp, "ve_gt": ve_gt,
    }


# ===== データ準備 =====
single_df = pd.read_csv(SINGLE_META_CSV)
single_df = single_df[single_df["valid_all"]==1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
single_train_df = single_df.iloc[:280].reset_index(drop=True)

mixed_df = pd.read_csv(MIXED_META_CSV)
mixed_df = mixed_df[mixed_df["valid_all"]==1].reset_index(drop=True)
mixed_df = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
mixed_train_df = mixed_df.iloc[:180].reset_index(drop=True)
holdout_df     = mixed_df.iloc[200:].reset_index(drop=True)

print(f"single_train={len(single_train_df)}, mixed_train={len(mixed_train_df)}, holdout={len(holdout_df)}", flush=True)

# モデルが使用した学習データを紐付け
TRAIN_DATA = {
    "D_tk3_fp0.001_mixed": mixed_train_df,
    "D_dw_fp0.001_single": single_train_df,
    "D_dw_fp0.001_mixed":  mixed_train_df,
}


# ===== 1) 学習曲線のプロット =====
print("\n[1] 学習曲線をプロット中...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Learning Curves: train vs val loss (30 epochs)")

for col, name in enumerate(TARGET_RUNS):
    json_path = os.path.join(V3_RESULTS_DIR, f"{name}.json")
    if not os.path.exists(json_path):
        print(f"  [{name}] JSON未検出", flush=True)
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    history = result["history"]
    epochs  = [h["epoch"]      for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"]   for h in history]

    col_c, label = COLORS[name]

    ax = axes[col]
    ax.plot(epochs, tr_loss, "-",  color=col_c, lw=1.5, label="train")
    ax.plot(epochs, va_loss, "--", color=col_c, lw=1.5, alpha=0.6, label="val")
    ax.set_title(label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
lc_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
fig.savefig(lc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: {lc_path}", flush=True)


# ===== 2) train vs holdout Precision の比較 =====
print("\n[2] train/holdout Precision を計算中...", flush=True)

results = {}
for name in TARGET_RUNS:
    pt_path = os.path.join(V3_RESULTS_DIR, f"{name}.pt")
    if not os.path.exists(pt_path):
        print(f"  [{name}] .pt未検出: {pt_path}", flush=True)
        continue

    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    train_df = TRAIN_DATA[name]
    print(f"  [{name}] train ({len(train_df)}件) 評価中...", flush=True)
    train_metrics = compute_precision_recall(model, train_df)

    print(f"  [{name}] holdout ({len(holdout_df)}件) 評価中...", flush=True)
    holdout_metrics = compute_precision_recall(model, holdout_df)

    results[name] = {"train": train_metrics, "holdout": holdout_metrics}

    print(f"  [{name}]"
          f"  train: cy_P={train_metrics['cy_precision']:.3f} cy_R={train_metrics['cy_recall']:.3f}"
          f"  ve_P={train_metrics['ve_precision']:.3f} ve_R={train_metrics['ve_recall']:.3f}",
          flush=True)
    print(f"  [{name}]"
          f"  holdout: cy_P={holdout_metrics['cy_precision']:.3f} cy_R={holdout_metrics['cy_recall']:.3f}"
          f"  ve_P={holdout_metrics['ve_precision']:.3f} ve_R={holdout_metrics['ve_recall']:.3f}",
          flush=True)


# ===== 2b) Precision比較プロット =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Train vs Holdout: Precision & Recall\n(NMS threshold=0.5, D_TOL=2, R_TOL=3)")

classes = [("cy", "Cyclist"), ("ve", "Vehicle")]
x = np.arange(len(TARGET_RUNS))
width = 0.3

for ax_idx, (cls, cls_label) in enumerate(classes):
    ax = axes[ax_idx]
    ax.set_title(f"{cls_label}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([COLORS[n][1] for n in TARGET_RUNS], fontsize=8, rotation=10)
    ax.grid(True, alpha=0.3, axis="y")

    train_prec  = [results[n]["train"][f"{cls}_precision"]  for n in TARGET_RUNS if n in results]
    holdout_prec = [results[n]["holdout"][f"{cls}_precision"] for n in TARGET_RUNS if n in results]
    train_rec   = [results[n]["train"][f"{cls}_recall"]     for n in TARGET_RUNS if n in results]
    holdout_rec  = [results[n]["holdout"][f"{cls}_recall"]  for n in TARGET_RUNS if n in results]

    n_valid = len(train_prec)
    xv = np.arange(n_valid)

    bars_tp = ax.bar(xv - width*1.5, train_prec,   width, label="train Precision",   color="tab:blue",   alpha=0.8)
    bars_hp = ax.bar(xv - width*0.5, holdout_prec, width, label="holdout Precision", color="tab:blue",   alpha=0.4, hatch="//")
    bars_tr = ax.bar(xv + width*0.5, train_rec,    width, label="train Recall",      color="tab:orange", alpha=0.8)
    bars_hr = ax.bar(xv + width*1.5, holdout_rec,  width, label="holdout Recall",    color="tab:orange", alpha=0.4, hatch="//")

    # 数値ラベル
    for bar in bars_tp + bars_hp + bars_tr + bars_hr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.legend(fontsize=7)

plt.tight_layout()
pc_path = os.path.join(OUTPUT_DIR, "precision_comparison.png")
fig.savefig(pc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n精度比較グラフ保存: {pc_path}", flush=True)


# ===== 3) サマリーテキスト出力 =====
print("\n" + "="*70, flush=True)
print(f"{'run':30s}  {'split':8s}  {'cy_P':>6} {'cy_R':>6} {'ve_P':>6} {'ve_R':>6}")
print("-"*70)
for name in TARGET_RUNS:
    if name not in results:
        continue
    for split in ["train", "holdout"]:
        m = results[name][split]
        print(f"{name:30s}  {split:8s}  "
              f"{m['cy_precision']:>6.3f} {m['cy_recall']:>6.3f} "
              f"{m['ve_precision']:>6.3f} {m['ve_recall']:>6.3f}", flush=True)
    print("-"*70)
print("="*70, flush=True)
print(f"\n完了。結果は {OUTPUT_DIR}/ に保存されました。", flush=True)
