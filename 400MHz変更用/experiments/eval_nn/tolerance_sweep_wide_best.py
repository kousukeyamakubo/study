"""
tolerance_sweep_narrow.py の wide_best 版。

変更点（4箇所のみ）:
  MODEL_PATH    : wide_loss_sweep/sweep_models/wide_gamma5.0_alpha500.pt
  FIXED_META_CSV: learn_dataset_fixed_angle/metadata.csv
  FIXED_ANGLES  : np.linspace(-5, 5, N_FIXED)
  FOR_PAPER_DIR : tolerance_sweep_results/for_paper_pm5deg/
  JSON出力      : tolerance_sweep_results_wide_best.json（narrowのJSONを上書きしない）
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

MODEL_PATH      = os.path.join(ROOT_DIR, "experiments", "sweep_wide", "wide_loss_sweep", "sweep_models", "wide_gamma5.0_alpha500.pt")
SINGLE_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_single_object", "metadata.csv")
FIXED_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_fixed_angle",   "metadata.csv")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "tolerance_sweep_results")
FOR_PAPER_DIR   = os.path.join(OUTPUT_DIR, "for_paper_pm5deg")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOR_PAPER_DIR, exist_ok=True)

# 論文用フォントサイズ
FP_LABEL  = 16
FP_TICK   = 14
FP_LEGEND = 12
FP_TITLE  = 14

# ===== 定数 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)  # wide (±5°) の角度グリッド
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42

DIAG_MAX  = 5   # Stage 1: tol = 0, 1, ..., DIAG_MAX
AXIS_MAX  = 7   # Stage 2: each axis sweeps 0, 1, ..., AXIS_MAX

print(f"DEVICE: {DEVICE}")


# ===== モデル定義（check.ipynb と同一） =====
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


# ===== データ準備 =====
def load_sample(path):
    data = np.load(path)
    x = np.stack([20*np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                  for i in range(data["rd_maps"].shape[0])], axis=0)
    return x, data


def build_eval_df():
    """learn_dataset_fixed_angle holdout (100件) — 2物体シーンのみ（cy+ve 同時存在）"""
    fixed = pd.read_csv(FIXED_META_CSV)
    fixed = fixed[fixed["valid_all"] == 1].reset_index(drop=True)
    fixed = fixed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return fixed.iloc[200:].reset_index(drop=True)


# ===== 推論（1回のみ）: スコア・検出位置をキャッシュ =====
def collect_raw(model, eval_df):
    """
    per-voxel argmax で各クラスに割り当てられた全ボクセルをキャッシュして返す。
    1サンプルから複数エントリが生じうる。GT 未検出サンプルはエントリなし。

    Returns:
        cy_raw, ve_raw        : 検出ボクセルリスト（Stage 1-6 の AP 計算用）
        cy_sample_info,
        ve_sample_info        : GT サンプルごとの情報（Stage 6-7 用）
            sample_idx        : サンプル番号
            r_diff            : |cy_true_r - ve_true_r|
            n_det             : そのクラスに割り当てられたボクセル数
            gt_pred_label     : GT ボクセル位置の argmax ラベル
        n_gt_cy, n_gt_ve      : GT 総数
    """
    cy_raw, ve_raw = [], []
    cy_sample_info, ve_sample_info = [], []
    n_gt_cy = n_gt_ve = 0

    model.eval()
    with torch.no_grad():
        for idx in range(len(eval_df)):
            if idx % 60 == 0:
                print(f"  {idx}/{len(eval_df)}", flush=True)
            row = eval_df.iloc[idx]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))

            x, data = load_sample(path)
            fa = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES

            vcy = int(str(row["valid_cyclist"]).strip() in ("1", "True"))
            vve = int(str(row["valid_vehicle"]).strip() in ("1", "True"))

            probs = F.softmax(model(torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)), dim=1)[0]
            # (3, N_FIXED, H, W)
            pred = probs.argmax(dim=0)  # (N_FIXED, H, W) — per-voxel argmax

            r_diff = (abs(int(row["cyclist_true_r_idx"]) - int(row["vehicle_true_r_idx"]))
                      if vcy and vve else -1)

            def voxels_for_class(cls_idx):
                mask = (pred == cls_idx)
                if not mask.any():
                    return []
                return [(int(ij[0]), int(ij[1]), int(ij[2]),
                         probs[cls_idx, int(ij[0]), int(ij[1]), int(ij[2])].item())
                        for ij in mask.nonzero(as_tuple=False)]

            cy_voxels = voxels_for_class(1)
            ve_voxels = voxels_for_class(2)

            if vcy:
                n_gt_cy += 1
                tch = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                tru_d, tru_r = int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])
                tru = dict(valid=True, true_ch=tch, true_d=tru_d, true_r=tru_r,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
                cy_sample_info.append(dict(
                    sample_idx=idx, r_diff=r_diff, n_det=len(cy_voxels),
                    gt_pred_label=int(pred[tch, tru_d, tru_r].item()),
                ))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

            if vve:
                n_gt_ve += 1
                tch = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                tru_d, tru_r = int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])
                tru = dict(valid=True, true_ch=tch, true_d=tru_d, true_r=tru_r,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
                ve_sample_info.append(dict(
                    sample_idx=idx, r_diff=r_diff, n_det=len(ve_voxels),
                    gt_pred_label=int(pred[tch, tru_d, tru_r].item()),
                ))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

    return cy_raw, ve_raw, cy_sample_info, ve_sample_info, n_gt_cy, n_gt_ve


# ===== AP 計算（PASCAL VOC方式 + greedy matching）=====
def compute_ap(raw, n_gt, a_tol, d_tol, r_tol):
    """
    キャッシュ済みの raw に対して許容幅 (a_tol, d_tol, r_tol) でヒット判定し AP を計算する。
    スコア降順でソート後、各 GT に対して最初にヒットした検出のみ TP とする（greedy matching）。
    Returns: AP (float), precisions (ndarray), recalls (ndarray)
    """
    dets = []
    for rec in raw:
        if rec["valid"]:
            is_hit = (abs(rec["det_ch"] - rec["true_ch"]) <= a_tol and
                      abs(rec["det_d"]  - rec["true_d"])  <= d_tol and
                      abs(rec["det_r"]  - rec["true_r"])  <= r_tol)
        else:
            is_hit = False
        dets.append({"score": rec["score"], "is_hit": is_hit, "sample_idx": rec["sample_idx"]})

    sorted_dets = sorted(dets, key=lambda x: -x["score"])
    matched = set()  # マッチ済み GT の sample_idx
    tp = 0
    precs, recs = [], []
    for i, det in enumerate(sorted_dets):
        if det["is_hit"] and det["sample_idx"] not in matched:
            tp += 1
            matched.add(det["sample_idx"])
        precs.append(tp / (i + 1))
        recs.append(tp / n_gt if n_gt > 0 else 0.0)

    precs = np.concatenate([[1.0], precs])
    recs  = np.concatenate([[0.0], recs])
    ap = float(np.sum((recs[1:] - recs[:-1]) * precs[1:]))
    return ap, precs, recs


# ===== ヘルパー: r_diff フィルタ付き AP =====
def compute_ap_rdiff(raw, sample_info, a_tol, d_tol, r_tol, lo=0, hi=9999):
    """r_diff 範囲 [lo, hi] でフィルタして AP を計算する"""
    raw_f  = [r for r in raw        if lo <= r["r_diff"] <= hi]
    n_gt_f = sum(1 for s in sample_info if lo <= s["r_diff"] <= hi)
    if n_gt_f == 0:
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 0.0])
    return compute_ap(raw_f, n_gt_f, a_tol, d_tol, r_tol)


# ===== ヘルパー: greedy matching で TP になった sample_idx を返す =====
def get_matched_samples(raw, a_tol, d_tol, r_tol):
    dets = []
    for rec in raw:
        if rec["valid"]:
            is_hit = (abs(rec["det_ch"] - rec["true_ch"]) <= a_tol and
                      abs(rec["det_d"]  - rec["true_d"])  <= d_tol and
                      abs(rec["det_r"]  - rec["true_r"])  <= r_tol)
        else:
            is_hit = False
        dets.append({"score": rec["score"], "is_hit": is_hit, "sample_idx": rec["sample_idx"]})
    matched = set()
    for det in sorted(dets, key=lambda x: -x["score"]):
        if det["is_hit"] and det["sample_idx"] not in matched:
            matched.add(det["sample_idx"])
    return matched


# ===== 推論実行 =====
model = RadarUNet3DSoftmax().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"model loaded: {os.path.basename(MODEL_PATH)}")

eval_df = build_eval_df()
print(f"eval samples: {len(eval_df)}")

print("running inference (once)...")
cy_raw, ve_raw, cy_sample_info, ve_sample_info, n_gt_cy, n_gt_ve = collect_raw(model, eval_df)
print(f"GT: cy={n_gt_cy}, ve={n_gt_ve}")


# =============================================================
# Stage 1: diagonal sweep  (k, k, k) for k in 0..DIAG_MAX
# =============================================================
print("\n=== Stage 1: diagonal sweep ===")
stage1 = []
for k in range(DIAG_MAX + 1):
    ap_cy, precs_cy, recs_cy = compute_ap(cy_raw, n_gt_cy, k, k, k)
    ap_ve, precs_ve, recs_ve = compute_ap(ve_raw, n_gt_ve, k, k, k)
    mAP = (ap_cy + ap_ve) / 2.0
    stage1.append(dict(tol=k, AP_cy=ap_cy, AP_ve=ap_ve, mAP=mAP,
                        precs_cy=precs_cy, recs_cy=recs_cy,
                        precs_ve=precs_ve, recs_ve=recs_ve))
    print(f"  tol={k}  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  mAP={mAP:.4f}")

# best: mAP 最大（同値なら最小 tol）
best = max(stage1, key=lambda x: (x["mAP"], -x["tol"]))
best_tol = best["tol"]
print(f"\nbest diagonal tol = {best_tol}  (mAP={best['mAP']:.4f})")

# Stage 1 プロット1: AP vs tol (line)
tols    = [r["tol"]   for r in stage1]
ap_cys  = [r["AP_cy"] for r in stage1]
ap_ves  = [r["AP_ve"] for r in stage1]
mAPs    = [r["mAP"]   for r in stage1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tols, ap_cys, "o-", color="tab:blue",   label="AP_cy")
ax.plot(tols, ap_ves, "s-", color="tab:orange", label="AP_ve")
ax.plot(tols, mAPs,   "^-", color="tab:green",  label="mAP",  lw=2)
ax.axvline(best_tol, color="gray", lw=1, linestyle="--", label=f"best tol={best_tol}")
ax.set_xlabel("Tolerance (A=D=R=tol)")
ax.set_ylabel("AP")
ax.set_title("Stage 1: Diagonal Tolerance Sweep")
ax.set_ylim(0, 1.05); ax.set_xticks(tols); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_ap_vs_tol.png"), dpi=150, bbox_inches="tight")
print("saved: stage1_ap_vs_tol.png")

# Stage 1 プロット2: PR curves for all tol values (cy and ve)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(stage1)))
for ax, key_p, key_r, cls_label in [
    (axes[0], "precs_cy", "recs_cy", "Cyclist"),
    (axes[1], "precs_ve", "recs_ve", "Vehicle"),
]:
    for r, c in zip(stage1, colors):
        ax.step(r[key_r], r[key_p], where="post", color=c, lw=1.5,
                label=f"tol={r['tol']}  AP={r['AP_' + key_r[5:7]]:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{cls_label} PR curves (diagonal sweep)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle("Stage 1: PR Curves by Diagonal Tolerance", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_pr_curves.png"), dpi=150, bbox_inches="tight")
print("saved: stage1_pr_curves.png")


# =============================================================
# Stage 2: per-axis sweep  (fix two axes at best_tol, vary one)
# =============================================================
print(f"\n=== Stage 2: per-axis sweep (base tol={best_tol}) ===")

axis_tols = list(range(AXIS_MAX + 1))
stage2 = {"A_TOL": [], "D_TOL": [], "R_TOL": []}

for ax_name, ax_idx in [("A_TOL", 0), ("D_TOL", 1), ("R_TOL", 2)]:
    print(f"  sweeping {ax_name}...")
    for v in axis_tols:
        a_tol = v           if ax_idx == 0 else best_tol
        d_tol = v           if ax_idx == 1 else best_tol
        r_tol = v           if ax_idx == 2 else best_tol
        ap_cy, _, _ = compute_ap(cy_raw, n_gt_cy, a_tol, d_tol, r_tol)
        ap_ve, _, _ = compute_ap(ve_raw, n_gt_ve, a_tol, d_tol, r_tol)
        mAP = (ap_cy + ap_ve) / 2.0
        stage2[ax_name].append(dict(val=v, a_tol=a_tol, d_tol=d_tol, r_tol=r_tol,
                                     AP_cy=ap_cy, AP_ve=ap_ve, mAP=mAP))
        print(f"    {ax_name}={v}  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  mAP={mAP:.4f}")

# Stage 2 プロット: 3 subplots (one per axis)
fig, axes_s2 = plt.subplots(1, 3, figsize=(15, 4))
for ax, (ax_name, rows) in zip(axes_s2, stage2.items()):
    vals   = [r["val"]   for r in rows]
    ap_cys = [r["AP_cy"] for r in rows]
    ap_ves = [r["AP_ve"] for r in rows]
    mAPs   = [r["mAP"]   for r in rows]
    ax.plot(vals, ap_cys, "o-", color="tab:blue",   label="AP_cy")
    ax.plot(vals, ap_ves, "s-", color="tab:orange", label="AP_ve")
    ax.plot(vals, mAPs,   "^-", color="tab:green",  label="mAP", lw=2)
    ax.axvline(best_tol, color="gray", lw=1, linestyle="--", label=f"base={best_tol}")
    other = {k: best_tol for k in ["A_TOL","D_TOL","R_TOL"] if k != ax_name}
    other_str = "  ".join(f"{k}={v}" for k, v in other.items())
    ax.set_xlabel(f"{ax_name}  ({other_str})")
    ax.set_ylabel("AP")
    ax.set_title(f"Stage 2: {ax_name} sweep")
    ax.set_ylim(0, 1.05); ax.set_xticks(vals); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle(f"Stage 2: Per-axis Tolerance Sweep (base tol={best_tol})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage2_per_axis.png"), dpi=150, bbox_inches="tight")
print("saved: stage2_per_axis.png")


# =============================================================
# Stage 3: combinatorial sweep  D_TOL=0 fixed,
#           A_TOL in {0, 1},  R_TOL in {0, 1, 2, 3, 4, 5}
# =============================================================
print("\n=== Stage 3: combinatorial sweep (D_TOL=0, A in {0,1}, R in 0..5) ===")

A_TOLS_S3 = [0, 1]
R_TOLS_S3 = list(range(6))
D_TOL_S3  = 0

stage3 = []
for a in A_TOLS_S3:
    for r in R_TOLS_S3:
        ap_cy, precs_cy, recs_cy = compute_ap(cy_raw, n_gt_cy, a, D_TOL_S3, r)
        ap_ve, precs_ve, recs_ve = compute_ap(ve_raw, n_gt_ve, a, D_TOL_S3, r)
        mAP = (ap_cy + ap_ve) / 2.0
        stage3.append(dict(A_TOL=a, D_TOL=D_TOL_S3, R_TOL=r,
                           AP_cy=ap_cy, AP_ve=ap_ve, mAP=mAP,
                           precs_cy=precs_cy, recs_cy=recs_cy,
                           precs_ve=precs_ve, recs_ve=recs_ve))
        print(f"  A={a} D={D_TOL_S3} R={r}  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  mAP={mAP:.4f}")

# プロット1: AP vs R_TOL, 線 = A_TOL (cy / ve / mAP の3枚)
fig, axes_s3 = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric, label, color_map in [
    (axes_s3[0], "AP_cy", "AP_cy (Cyclist)", ["tab:blue",   "royalblue"]),
    (axes_s3[1], "AP_ve", "AP_ve (Vehicle)", ["tab:orange", "darkorange"]),
    (axes_s3[2], "mAP",   "mAP",             ["tab:green",  "darkgreen"]),
]:
    for a, color in zip(A_TOLS_S3, color_map):
        rows = [r for r in stage3 if r["A_TOL"] == a]
        rs   = [r["R_TOL"]  for r in rows]
        vals = [r[metric]   for r in rows]
        ax.plot(rs, vals, "o-", color=color, label=f"A_TOL={a}", lw=2)
    ax.set_xlabel("R_TOL  (D_TOL=0)")
    ax.set_ylabel(label)
    ax.set_title(f"Stage 3: {label}")
    ax.set_ylim(0, 1.05); ax.set_xticks(R_TOLS_S3); ax.legend(); ax.grid(alpha=0.3)
plt.suptitle("Stage 3: Combinatorial Sweep (D_TOL=0, A_TOL x R_TOL)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage3_combo.png"), dpi=150, bbox_inches="tight")
print("saved: stage3_combo.png")

# プロット2: PR curves grid (A_TOL x R_TOL)
fig, axes_grid = plt.subplots(len(A_TOLS_S3), len(R_TOLS_S3),
                               figsize=(4 * len(R_TOLS_S3), 3.5 * len(A_TOLS_S3)),
                               sharex=True, sharey=True)
colors_pr = {"cy": "tab:blue", "ve": "tab:orange"}
for row_i, a in enumerate(A_TOLS_S3):
    for col_i, r in enumerate(R_TOLS_S3):
        rec = next(x for x in stage3 if x["A_TOL"] == a and x["R_TOL"] == r)
        ax  = axes_grid[row_i][col_i]
        ax.step(rec["recs_cy"], rec["precs_cy"], where="post",
                color=colors_pr["cy"], lw=1.5, label=f"cy {rec['AP_cy']:.3f}")
        ax.step(rec["recs_ve"], rec["precs_ve"], where="post",
                color=colors_pr["ve"], lw=1.5, label=f"ve {rec['AP_ve']:.3f}")
        ax.set_title(f"A={a} R={r}  mAP={rec['mAP']:.3f}", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6); ax.grid(alpha=0.2)
        if col_i == 0:      ax.set_ylabel("Precision")
        if row_i == len(A_TOLS_S3) - 1: ax.set_xlabel("Recall")
plt.suptitle("Stage 3: PR Curves Grid (D_TOL=0)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage3_pr_grid.png"), dpi=150, bbox_inches="tight")
print("saved: stage3_pr_grid.png")

# プロット3: A_TOL=1, D_TOL=0 固定で R_TOL を変化させた PR 曲線
rows_a1 = [r for r in stage3 if r["A_TOL"] == 1]  # R_TOL=0..5
colors_r = plt.cm.plasma(np.linspace(0.1, 0.85, len(rows_a1)))

fig, axes_r = plt.subplots(1, 2, figsize=(11, 4))
for ax, p_key, r_key, cls_label, color_base in [
    (axes_r[0], "precs_cy", "recs_cy", "Cyclist", "tab:blue"),
    (axes_r[1], "precs_ve", "recs_ve", "Vehicle",  "tab:orange"),
]:
    for rec, c in zip(rows_a1, colors_r):
        ap_val = rec["AP_cy"] if cls_label == "Cyclist" else rec["AP_ve"]
        ax.step(rec[r_key], rec[p_key], where="post", color=c, lw=2,
                label=f"R_TOL={rec['R_TOL']}  AP={ap_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{cls_label}  (A_TOL=1, D_TOL=0)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("PR Curves: A_TOL=1, D_TOL=0, R_TOL sweep", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage3_pr_rtol_sweep.png"), dpi=150, bbox_inches="tight")
print("saved: stage3_pr_rtol_sweep.png")

# --- 論文用: coco_AP PR 曲線 2 パネル ---
ap_cy_avg_coco = float(np.mean([r["AP_cy"] for r in rows_a1]))
ap_ve_avg_coco = float(np.mean([r["AP_ve"] for r in rows_a1]))
fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 5))
for ax, p_key, r_key, cls_label in [
    (axes_p[0], "precs_cy", "recs_cy", "Cyclist"),
    (axes_p[1], "precs_ve", "recs_ve", "Vehicle"),
]:
    ap_key = "AP_cy" if cls_label == "Cyclist" else "AP_ve"
    rec1 = next(r for r in rows_a1 if r["R_TOL"] == 1)
    ax.step(rec1[r_key], rec1[p_key], where="post", color=colors_r[1], lw=2,
            label=f"R_TOL=1  AUC={rec1[ap_key]:.4f}")
    ax.set_xlabel("Recall",    fontsize=FP_LABEL)
    ax.set_ylabel("Precision", fontsize=FP_LABEL)
    ax.tick_params(axis="both", labelsize=FP_TICK)
    ax.set_title(cls_label, fontsize=FP_TITLE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FP_LEGEND); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "coco_ap_pr_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved (for_paper_pm5deg): coco_ap_pr_curves.png")


# =============================================================
# Stage 4: nuScenes-style mAP
#   A_TOL in {0, 1}, D_TOL=0 固定, R_TOL=0..5
#   全 12 通り (2 x 6) の AP を平均して mAP を算出
# =============================================================
print("\n=== Stage 4: nuScenes-style mAP (A_TOL x R_TOL, 12 thresholds) ===")

rows_a0 = [r for r in stage3 if r["A_TOL"] == 0]
# rows_a1 は Stage 3 プロット3 で定義済み

ap_cy_vals_a0 = [r["AP_cy"] for r in rows_a0]
ap_ve_vals_a0 = [r["AP_ve"] for r in rows_a0]
ap_cy_vals_a1 = [r["AP_cy"] for r in rows_a1]
ap_ve_vals_a1 = [r["AP_ve"] for r in rows_a1]

s4_ap_cy_avg = float(np.mean(ap_cy_vals_a0 + ap_cy_vals_a1))
s4_ap_ve_avg = float(np.mean(ap_ve_vals_a0 + ap_ve_vals_a1))
s4_mAP_avg   = (s4_ap_cy_avg + s4_ap_ve_avg) / 2.0

print(f"  {'A_TOL':>5} {'R_TOL':>5}  {'AP_cy':>7}  {'AP_ve':>7}  {'mAP':>7}")
for r in stage3:
    print(f"  {r['A_TOL']:>5} {r['R_TOL']:>5}  {r['AP_cy']:>7.4f}  {r['AP_ve']:>7.4f}  {(r['AP_cy']+r['AP_ve'])/2:>7.4f}")
print(f"\n  AP_cy (avg, 12 thresholds) = {s4_ap_cy_avg:.4f}")
print(f"  AP_ve (avg, 12 thresholds) = {s4_ap_ve_avg:.4f}")
print(f"  mAP                        = {s4_mAP_avg:.4f}")

mAP_vals_a0 = [(c + v) / 2 for c, v in zip(ap_cy_vals_a0, ap_ve_vals_a0)]
mAP_vals_a1 = [(c + v) / 2 for c, v in zip(ap_cy_vals_a1, ap_ve_vals_a1)]

fig, axes_s4 = plt.subplots(1, 3, figsize=(15, 4))
for ax, vals_a0, vals_a1, metric_label, c0, c1 in [
    (axes_s4[0], ap_cy_vals_a0, ap_cy_vals_a1, "AP_cy",
     "steelblue",  "tab:blue"),
    (axes_s4[1], ap_ve_vals_a0, ap_ve_vals_a1, "AP_ve",
     "darkorange", "tab:orange"),
    (axes_s4[2], mAP_vals_a0,   mAP_vals_a1,   "mAP",
     "darkgreen",  "tab:green"),
]:
    avg_a0      = float(np.mean(vals_a0))
    avg_a1      = float(np.mean(vals_a1))
    overall_avg = float(np.mean(vals_a0 + vals_a1))
    ax.plot(R_TOLS_S3, vals_a0, "s--", color=c0, lw=2, label=f"A_TOL=0  avg={avg_a0:.4f}")
    ax.plot(R_TOLS_S3, vals_a1, "o-",  color=c1, lw=2, label=f"A_TOL=1  avg={avg_a1:.4f}")
    ax.axhline(overall_avg, color="gray", lw=1.5, linestyle=":",
               label=f"12-thresh avg={overall_avg:.4f}")
    ax.set_xlabel("R_TOL  (D_TOL=0)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs R_TOL")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(R_TOLS_S3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    f"Stage 4: nuScenes-style mAP  (D_TOL=0, avg over A_TOL x R_TOL)\n"
    f"AP_cy={s4_ap_cy_avg:.4f}  AP_ve={s4_ap_ve_avg:.4f}  mAP={s4_mAP_avg:.4f}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage4_nuscenes_map.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved: stage4_nuscenes_map.png")

# --- 論文用: mAP vs R_TOL のみ単体 ---
fig_p, ax_p = plt.subplots(figsize=(6, 5))
ax_p.plot(R_TOLS_S3, mAP_vals_a0, "s--", color="darkgreen",  lw=2,
          label=f"A_TOL=0  avg={float(np.mean(mAP_vals_a0)):.4f}")
ax_p.plot(R_TOLS_S3, mAP_vals_a1, "o-",  color="tab:green",  lw=2,
          label=f"A_TOL=1  avg={float(np.mean(mAP_vals_a1)):.4f}")
ax_p.axhline(s4_mAP_avg, color="gray", lw=1.5, linestyle=":",
             label=f"avg={s4_mAP_avg:.4f}")
ax_p.set_xlabel("R_TOL", fontsize=FP_LABEL)
ax_p.set_ylabel("mAP",              fontsize=FP_LABEL)
ax_p.tick_params(axis="both", labelsize=FP_TICK)
ax_p.set_ylim(0, 1.05)
ax_p.set_xticks(R_TOLS_S3)
ax_p.legend(fontsize=FP_LEGEND, loc="lower right")
ax_p.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "nuscenes_map_vs_rtol.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved (for_paper_pm5deg): nuscenes_map_vs_rtol.png")


# =============================================================
# Stage 6: r_diff 別 AP  (Stage 4 と同設定: A_TOL in {0,1}, D_TOL=0, R_TOL=0..5, 計 12 通り平均)
# =============================================================
print("\n=== Stage 6: r_diff-stratified AP ===")

RDIFF_BINS   = [(0, 2), (3, 5), (6, 10), (11, 9999)]
RDIFF_LABELS = ["r_diff=0-2", "r_diff=3-5", "r_diff=6-10", "r_diff>=11"]

stage6 = []
for (lo, hi), label in zip(RDIFF_BINS, RDIFF_LABELS):
    n_cy_bin = sum(1 for s in cy_sample_info if lo <= s["r_diff"] <= hi)
    n_ve_bin = sum(1 for s in ve_sample_info if lo <= s["r_diff"] <= hi)
    ap_cy_list, ap_ve_list = [], []
    for a_tol in [0, 1]:
        for r_tol in R_TOLS_S3:
            ap_cy_r, _, _ = compute_ap_rdiff(cy_raw, cy_sample_info, a_tol, 0, r_tol, lo, hi)
            ap_ve_r, _, _ = compute_ap_rdiff(ve_raw, ve_sample_info, a_tol, 0, r_tol, lo, hi)
            ap_cy_list.append(ap_cy_r)
            ap_ve_list.append(ap_ve_r)
    ap_cy_avg = float(np.mean(ap_cy_list))
    ap_ve_avg = float(np.mean(ap_ve_list))
    mAP_avg   = (ap_cy_avg + ap_ve_avg) / 2.0
    stage6.append(dict(label=label, lo=lo, hi=hi, n_cy=n_cy_bin, n_ve=n_ve_bin,
                       ap_cy_per_rtol=ap_cy_list, ap_ve_per_rtol=ap_ve_list,
                       AP_cy=ap_cy_avg, AP_ve=ap_ve_avg, mAP=mAP_avg))
    print(f"  {label:<16}  n={n_cy_bin:>3}  AP_cy={ap_cy_avg:.4f}  AP_ve={ap_ve_avg:.4f}  mAP={mAP_avg:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(stage6))
w = 0.35
ax.bar(x - w/2, [r["AP_cy"] for r in stage6], w, color="tab:blue",   label="AP_cy", alpha=0.8)
ax.bar(x + w/2, [r["AP_ve"] for r in stage6], w, color="tab:orange", label="AP_ve", alpha=0.8)
for i, r in enumerate(stage6):
    ax.text(i - w/2, r["AP_cy"] + 0.01, f'n={r["n_cy"]}', ha="center", fontsize=8)
    ax.text(i + w/2, r["AP_ve"] + 0.01, f'n={r["n_ve"]}', ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in stage6])
ax.set_xlabel("r_diff (range bin gap between cy and ve GT)")
ax.set_ylabel("AP (nuScenes-style, avg A_TOL x R_TOL, D_TOL=0)")
ax.set_title("Stage 6: AP by r_diff bin")
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage6_rdiff_ap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved: stage6_rdiff_ap.png")

# --- 論文用: stage6 ---
fig_p, ax_p = plt.subplots(figsize=(8, 5))
x_p = np.arange(len(stage6))
ax_p.bar(x_p - w/2, [r["AP_cy"] for r in stage6], w, color="tab:blue",   label="AP_cy", alpha=0.8)
ax_p.bar(x_p + w/2, [r["AP_ve"] for r in stage6], w, color="tab:orange", label="AP_ve", alpha=0.8)
for i, r in enumerate(stage6):
    ax_p.text(i - w/2, r["AP_cy"] + 0.01, f'n={r["n_cy"]}', ha="center", fontsize=FP_TICK)
    ax_p.text(i + w/2, r["AP_ve"] + 0.01, f'n={r["n_ve"]}', ha="center", fontsize=FP_TICK)
ax_p.set_xticks(x_p)
ax_p.set_xticklabels([r["label"] for r in stage6], fontsize=FP_TICK)
ax_p.tick_params(axis="y", labelsize=FP_TICK)
ax_p.set_xlabel("r_diff (range bin gap between cy and ve GT)", fontsize=FP_LABEL)
ax_p.set_ylabel("AP", fontsize=FP_LABEL)
ax_p.set_ylim(0, 1.15)
ax_p.legend(fontsize=FP_LEGEND)
ax_p.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "stage6_rdiff_ap.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved (for_paper_pm5deg): stage6_rdiff_ap.png")


# =============================================================
# Stage 7: 失敗ケース分類  (A_TOL=1, D_TOL=0, R_TOL=3 固定)
# =============================================================
print("\n=== Stage 7: Failure analysis (A_TOL=1, D_TOL=0, R_TOL=3) ===")

S7_A, S7_D, S7_R = 1, 0, 3
PRED_LABEL_NAMES = {0: "bg", 1: "cy", 2: "ve"}

cy_matched = get_matched_samples(cy_raw, S7_A, S7_D, S7_R)
ve_matched = get_matched_samples(ve_raw, S7_A, S7_D, S7_R)

def failure_category(info, matched_set, cls_idx):
    if info["sample_idx"] in matched_set:
        return "TP"
    gl = info["gt_pred_label"]
    if gl == cls_idx:
        return "pos_miss"
    if gl == 0:
        return "no_det"
    return "gt_confused"

stage7 = {}
for cls_name, sample_info, matched_set, cls_idx in [
    ("cy", cy_sample_info, cy_matched, 1),
    ("ve", ve_sample_info, ve_matched, 2),
]:
    rows = []
    for info in sample_info:
        cat = failure_category(info, matched_set, cls_idx)
        rows.append(dict(**info, category=cat))
    df = pd.DataFrame(rows)
    stage7[cls_name] = df

    n_tp       = (df.category == "TP").sum()
    n_nodet    = (df.category == "no_det").sum()
    n_confused = (df.category == "gt_confused").sum()
    n_pos      = (df.category == "pos_miss").sum()
    print(f"\n  {cls_name.upper()}  (n_gt={len(df)})")
    print(f"    TP            : {n_tp:>3}")
    print(f"    no_det        : {n_nodet:>3}")
    print(f"    gt_confused   : {n_confused:>3}")
    print(f"    pos_miss      : {n_pos:>3}")

cats   = ["TP", "no_det", "gt_confused", "pos_miss"]
colors = ["tab:green", "tab:gray", "tab:purple", "tab:brown"]

fig, axes_s7 = plt.subplots(1, 2, figsize=(12, 4))
for ax, (cls_name, df) in zip(axes_s7, stage7.items()):
    bottom = np.zeros(len(RDIFF_BINS))
    bin_ns = [sum(1 for s in (cy_sample_info if cls_name == "cy" else ve_sample_info)
                  if lo <= s["r_diff"] <= hi)
              for lo, hi in RDIFF_BINS]
    for cat, color in zip(cats, colors):
        vals = [(df[(df.r_diff >= lo) & (df.r_diff <= hi)].category == cat).sum()
                for lo, hi in RDIFF_BINS]
        ax.bar(range(len(RDIFF_BINS)), vals, bottom=bottom, label=cat, color=color, alpha=0.85)
        bottom += np.array(vals, dtype=float)
    tick_labels = [f"{l}\n(n={n})" for l, n in zip(["0-2","3-5","6-10",">=11"], bin_ns)]
    ax.set_xticks(range(len(RDIFF_BINS)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("r_diff bin")
    ax.set_ylabel("sample count")
    ax.set_title(f"{cls_name.upper()}: TP/Failure by r_diff")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"Stage 7: Failure Analysis  (A_TOL={S7_A}, D_TOL={S7_D}, R_TOL={S7_R})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage7_failure.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved: stage7_failure.png")

# --- 論文用: stage7 ---
fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 5))
for ax, (cls_name, df) in zip(axes_p, stage7.items()):
    bottom = np.zeros(len(RDIFF_BINS))
    bin_ns = [sum(1 for s in (cy_sample_info if cls_name == "cy" else ve_sample_info)
                  if lo <= s["r_diff"] <= hi)
              for lo, hi in RDIFF_BINS]
    for cat, color in zip(cats, colors):
        vals = [(df[(df.r_diff >= lo) & (df.r_diff <= hi)].category == cat).sum()
                for lo, hi in RDIFF_BINS]
        ax.bar(range(len(RDIFF_BINS)), vals, bottom=bottom, label=cat, color=color, alpha=0.85)
        bottom += np.array(vals, dtype=float)
    tick_labels = [f"{l}\n(n={n})" for l, n in zip(["0-2","3-5","6-10",">=11"], bin_ns)]
    ax.set_xticks(range(len(RDIFF_BINS)))
    ax.set_xticklabels(tick_labels, fontsize=FP_TICK)
    ax.tick_params(axis="y", labelsize=FP_TICK)
    ax.set_xlabel("r_diff bin",    fontsize=FP_LABEL)
    ax.set_ylabel("sample count",  fontsize=FP_LABEL)
    ax.set_title(cls_name.upper(), fontsize=FP_TITLE)
    ax.legend(fontsize=FP_LEGEND)
    ax.grid(axis="y", alpha=0.3)
fig_p.suptitle(
    f"A_TOL={S7_A}, D_TOL={S7_D}, R_TOL={S7_R}",
    fontsize=FP_TITLE,
)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "stage7_failure.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved (for_paper_pm5deg): stage7_failure.png")


# ===== 結果サマリ保存 =====
summary = {
    "model": os.path.basename(MODEL_PATH),
    "n_eval": len(eval_df),
    "n_gt_cy": n_gt_cy,
    "n_gt_ve": n_gt_ve,
    "stage1": [
        {k: v for k, v in r.items() if not k.startswith("precs") and not k.startswith("recs")}
        for r in stage1
    ],
    "best_diagonal_tol": best_tol,
    "best_mAP": round(best["mAP"], 4),
    "stage2": {
        ax_name: [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            for r in rows
        ]
        for ax_name, rows in stage2.items()
    },
    "stage3": [
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in r.items() if not k.startswith("precs") and not k.startswith("recs")}
        for r in stage3
    ],
    "stage4": {
        "D_TOL": 0,
        "A_TOL_range": [0, 1],
        "R_TOL_range": R_TOLS_S3,
        "AP_cy_per_rtol_a0": [round(v, 4) for v in ap_cy_vals_a0],
        "AP_ve_per_rtol_a0": [round(v, 4) for v in ap_ve_vals_a0],
        "AP_cy_per_rtol_a1": [round(v, 4) for v in ap_cy_vals_a1],
        "AP_ve_per_rtol_a1": [round(v, 4) for v in ap_ve_vals_a1],
        "AP_cy_avg": round(s4_ap_cy_avg, 4),
        "AP_ve_avg": round(s4_ap_ve_avg, 4),
        "mAP_avg":   round(s4_mAP_avg,   4),
    },
    "stage6": [
        {k: (round(v, 4) if isinstance(v, float) else
             [round(x, 4) for x in v] if isinstance(v, list) else v)
         for k, v in r.items()}
        for r in stage6
    ],
    "stage7": {
        cls_name: df[["sample_idx", "r_diff", "n_det", "gt_pred_label", "category"]]
                    .to_dict(orient="records")
        for cls_name, df in stage7.items()
    },
}
out_json = os.path.join(OUTPUT_DIR, "tolerance_sweep_results_wide_best.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"saved: {out_json}")

print(f"\n=== 最終結果 ===")
print(f"  Stage 4 mAP = {s4_mAP_avg:.4f}  (AP_cy={s4_ap_cy_avg:.4f}, AP_ve={s4_ap_ve_avg:.4f})")
print(f"\nAll done.")
