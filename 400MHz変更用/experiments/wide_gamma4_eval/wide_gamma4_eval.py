"""
広角度（±5°）データセット × GAMMA=4.0 / ALPHA=500 の学習＋tolerance_sweep 評価。

narrow との比較用。
  FIXED_ANGLES : np.linspace(-5, 5, 10)
  train/val    : learn_dataset_single_object
  eval holdout : learn_dataset_fixed_angle (200件以降 100件)
  モデル保存   : ./wide_gamma4_results/wide_gamma4.pt
  結果出力     : ./wide_gamma4_results/
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
from torch.utils.data import Dataset, DataLoader

# ===== パス =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

SINGLE_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_single_object", "metadata.csv")
FIXED_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_fixed_angle",   "metadata.csv")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "wide_gamma4_results")
FOR_PAPER_DIR   = os.path.join(OUTPUT_DIR, "for_paper")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "wide_gamma4.pt")
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(FOR_PAPER_DIR, exist_ok=True)

# ===== 固定設定 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)   # 広角度グリッド [度]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4

# ===== 損失ハイパーパラメータ（sweep best） =====
FOCAL_GAMMA   = 4.0
FOCAL_ALPHA_POS = 500.0

# ===== Stage 4 評価設定（tolerance_sweep_narrow と同一） =====
DIAG_MAX = 5
AXIS_MAX = 7
A_TOLS   = [0, 1]
R_TOLS   = list(range(6))
D_TOL    = 0

# 論文用フォントサイズ
FP_LABEL  = 16
FP_TICK   = 14
FP_LEGEND = 12
FP_TITLE  = 14

print(f"DEVICE: {DEVICE}")
print(f"FOCAL_GAMMA={FOCAL_GAMMA}, FOCAL_ALPHA_pos={FOCAL_ALPHA_POS}")
print(f"FIXED_ANGLES: {FIXED_ANGLES}")


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
        # x: (B, N_FIXED, H, W) → (B, 1, N_FIXED, H, W)
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


# ===== データセット =====

class MultiTargetDatasetSeg(Dataset):
    def __init__(self, meta_df):
        self.samples = []
        print(f"  データ読み込み中... ({len(meta_df)} サンプル)", flush=True)
        for idx in range(len(meta_df)):
            row  = meta_df.iloc[idx]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))
            data     = np.load(path)
            x        = np.stack([
                20 * np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                for i in range(data["rd_maps"].shape[0])
            ], axis=0)
            fa       = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES
            valid_cy = 1 if str(row["valid_cyclist"]).strip() in ("1", "True") else 0
            valid_ve = 1 if str(row["valid_vehicle"]).strip() in ("1", "True") else 0
            cy_angle = float(data["cyclist_true_angle_deg"]) if (valid_cy and "cyclist_true_angle_deg" in data) else 0.0
            ve_angle = float(data["vehicle_true_angle_deg"]) if (valid_ve and "vehicle_true_angle_deg" in data) else 0.0

            H, W  = x.shape[1], x.shape[2]
            y_seg = np.zeros((N_FIXED, H, W), dtype=np.int64)
            if valid_cy:
                cy_ch = int(np.argmin(np.abs(fa - cy_angle)))
                y_seg[cy_ch, int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])] = 1
            if valid_ve:
                ve_ch = int(np.argmin(np.abs(fa - ve_angle)))
                y_seg[ve_ch, int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])] = 2

            self.samples.append({
                "x":        torch.from_numpy(x).float(),
                "y_seg":    torch.from_numpy(y_seg).long(),
                "cy_true_d": torch.tensor(int(row["cyclist_true_d_idx"]), dtype=torch.long),
                "cy_true_r": torch.tensor(int(row["cyclist_true_r_idx"]), dtype=torch.long),
                "ve_true_d": torch.tensor(int(row["vehicle_true_d_idx"]), dtype=torch.long),
                "ve_true_r": torch.tensor(int(row["vehicle_true_r_idx"]), dtype=torch.long),
                "valid_cy":  torch.tensor(valid_cy, dtype=torch.long),
                "valid_ve":  torch.tensor(valid_ve, dtype=torch.long),
            })

    def __len__(self):          return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ===== 損失関数 =====

ALPHA_TENSOR = torch.tensor([1.0, FOCAL_ALPHA_POS, FOCAL_ALPHA_POS])

def compute_loss(logits, y_seg):
    a   = ALPHA_TENSOR.to(logits.device)
    ce  = F.cross_entropy(logits, y_seg, weight=a, reduction="none")
    p_t = torch.exp(-ce)
    return ((1 - p_t) ** FOCAL_GAMMA * ce).mean()


# ===== 学習ループ =====

def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            x     = batch["x"].to(DEVICE)
            y_seg = batch["y_seg"].to(DEVICE)
            if train_mode:
                optimizer.zero_grad()
            loss = compute_loss(model(x), y_seg)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


# ===== データ準備 =====

print("\n=== データ読み込み ===")
single_df = pd.read_csv(SINGLE_META_CSV)
single_df = single_df[single_df["valid_all"] == 1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
train_df  = single_df.iloc[:280].reset_index(drop=True)
val_df    = single_df.iloc[280:340].reset_index(drop=True)

fixed_df  = pd.read_csv(FIXED_META_CSV)
fixed_df  = fixed_df[fixed_df["valid_all"] == 1].reset_index(drop=True)
fixed_df  = fixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
eval_df   = fixed_df.iloc[200:].reset_index(drop=True)

print(f"train: {len(train_df)}, val: {len(val_df)}, eval holdout: {len(eval_df)}")

print("train dataset 読み込み中...")
train_ds = MultiTargetDatasetSeg(train_df)
print("val dataset 読み込み中...")
val_ds   = MultiTargetDatasetSeg(val_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


# ===== 学習 =====

print("\n=== 学習開始 ===")
model     = RadarUNet3DSoftmax().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
best_state    = None

for epoch in range(EPOCHS):
    train_loss = run_epoch(model, train_loader, optimizer)
    val_loss   = run_epoch(model, val_loader,   None)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"epoch={epoch+1:02d}  train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

model.load_state_dict(best_state)
torch.save(best_state, MODEL_SAVE_PATH)
print(f"モデル保存: {MODEL_SAVE_PATH}  (best_val={best_val_loss:.4f})")


# ===== 推論キャッシュ =====

def load_sample(path):
    data = np.load(path)
    x = np.stack([
        20 * np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
        for i in range(data["rd_maps"].shape[0])
    ], axis=0)
    return x, data


def collect_raw(model, eval_df):
    cy_raw, ve_raw = [], []
    cy_sample_info, ve_sample_info = [], []
    n_gt_cy = n_gt_ve = 0

    model.eval()
    with torch.no_grad():
        for idx in range(len(eval_df)):
            if idx % 50 == 0:
                print(f"  {idx}/{len(eval_df)}", flush=True)
            row  = eval_df.iloc[idx]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))

            x, data = load_sample(path)
            fa  = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES
            vcy = int(str(row["valid_cyclist"]).strip() in ("1", "True"))
            vve = int(str(row["valid_vehicle"]).strip() in ("1", "True"))

            probs = F.softmax(
                model(torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)), dim=1
            )[0]  # (3, N_FIXED, H, W)
            pred  = probs.argmax(dim=0)

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
                tch   = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                tru_d = int(row["cyclist_true_d_idx"])
                tru_r = int(row["cyclist_true_r_idx"])
                tru   = dict(valid=True, true_ch=tch, true_d=tru_d, true_r=tru_r,
                             sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
                cy_sample_info.append(dict(sample_idx=idx, r_diff=r_diff, n_det=len(cy_voxels),
                                           gt_pred_label=int(pred[tch, tru_d, tru_r].item())))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

            if vve:
                n_gt_ve += 1
                tch   = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                tru_d = int(row["vehicle_true_d_idx"])
                tru_r = int(row["vehicle_true_r_idx"])
                tru   = dict(valid=True, true_ch=tch, true_d=tru_d, true_r=tru_r,
                             sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
                ve_sample_info.append(dict(sample_idx=idx, r_diff=r_diff, n_det=len(ve_voxels),
                                           gt_pred_label=int(pred[tch, tru_d, tru_r].item())))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

    return cy_raw, ve_raw, cy_sample_info, ve_sample_info, n_gt_cy, n_gt_ve


def compute_ap(raw, n_gt, a_tol, d_tol, r_tol):
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
    tp = 0
    precs, recs = [], []
    for i, det in enumerate(sorted(dets, key=lambda x: -x["score"])):
        if det["is_hit"] and det["sample_idx"] not in matched:
            tp += 1
            matched.add(det["sample_idx"])
        precs.append(tp / (i + 1))
        recs.append(tp / n_gt if n_gt > 0 else 0.0)

    precs = np.concatenate([[1.0], precs])
    recs  = np.concatenate([[0.0], recs])
    ap    = float(np.sum((recs[1:] - recs[:-1]) * precs[1:]))
    return ap, precs, recs


def compute_ap_rdiff(raw, sample_info, a_tol, d_tol, r_tol, lo=0, hi=9999):
    raw_f  = [r for r in raw        if lo <= r["r_diff"] <= hi]
    n_gt_f = sum(1 for s in sample_info if lo <= s["r_diff"] <= hi)
    if n_gt_f == 0:
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 0.0])
    return compute_ap(raw_f, n_gt_f, a_tol, d_tol, r_tol)


# ===== 推論実行 =====

print("\n=== 推論（1回） ===")
cy_raw, ve_raw, cy_sample_info, ve_sample_info, n_gt_cy, n_gt_ve = collect_raw(model, eval_df)
print(f"GT: cy={n_gt_cy}, ve={n_gt_ve}")


# ===== Stage 1: diagonal sweep =====
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

best    = max(stage1, key=lambda x: (x["mAP"], -x["tol"]))
best_tol = best["tol"]
print(f"\nbest diagonal tol = {best_tol}  (mAP={best['mAP']:.4f})")

tols   = [r["tol"]   for r in stage1]
ap_cys = [r["AP_cy"] for r in stage1]
ap_ves = [r["AP_ve"] for r in stage1]
mAPs   = [r["mAP"]   for r in stage1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tols, ap_cys, "o-", color="tab:blue",   label="AP_cy")
ax.plot(tols, ap_ves, "s-", color="tab:orange", label="AP_ve")
ax.plot(tols, mAPs,   "^-", color="tab:green",  label="mAP", lw=2)
ax.axvline(best_tol, color="gray", lw=1, linestyle="--", label=f"best tol={best_tol}")
ax.set_xlabel("Tolerance (A=D=R=tol)"); ax.set_ylabel("AP")
ax.set_title("Stage 1: Diagonal Tolerance Sweep (wide, gamma4)")
ax.set_ylim(0, 1.05); ax.set_xticks(tols); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_ap_vs_tol.png"), dpi=150, bbox_inches="tight")
plt.close()

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
    ax.set_title(f"{cls_label} PR curves")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle("Stage 1: PR Curves (wide, gamma4)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage1_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved: stage1_*.png")


# ===== Stage 3: combinatorial (D_TOL=0, A in {0,1}, R in 0..5) =====
print("\n=== Stage 3: combinatorial sweep ===")
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


# ===== Stage 4: nuScenes-style mAP =====
print("\n=== Stage 4: nuScenes-style mAP ===")
rows_a0 = [r for r in stage3 if r["A_TOL"] == 0]
rows_a1 = [r for r in stage3 if r["A_TOL"] == 1]

ap_cy_vals_a0 = [r["AP_cy"] for r in rows_a0]
ap_ve_vals_a0 = [r["AP_ve"] for r in rows_a0]
ap_cy_vals_a1 = [r["AP_cy"] for r in rows_a1]
ap_ve_vals_a1 = [r["AP_ve"] for r in rows_a1]

s4_ap_cy = float(np.mean(ap_cy_vals_a0 + ap_cy_vals_a1))
s4_ap_ve = float(np.mean(ap_ve_vals_a0 + ap_ve_vals_a1))
s4_mAP   = (s4_ap_cy + s4_ap_ve) / 2.0

print(f"  AP_cy (12-thresh avg) = {s4_ap_cy:.4f}")
print(f"  AP_ve (12-thresh avg) = {s4_ap_ve:.4f}")
print(f"  mAP                   = {s4_mAP:.4f}")

mAP_vals_a0 = [(c + v) / 2 for c, v in zip(ap_cy_vals_a0, ap_ve_vals_a0)]
mAP_vals_a1 = [(c + v) / 2 for c, v in zip(ap_cy_vals_a1, ap_ve_vals_a1)]

fig, axes_s4 = plt.subplots(1, 3, figsize=(15, 4))
for ax, v0, v1, label, c0, c1 in [
    (axes_s4[0], ap_cy_vals_a0, ap_cy_vals_a1, "AP_cy", "steelblue",  "tab:blue"),
    (axes_s4[1], ap_ve_vals_a0, ap_ve_vals_a1, "AP_ve", "darkorange", "tab:orange"),
    (axes_s4[2], mAP_vals_a0,   mAP_vals_a1,   "mAP",   "darkgreen",  "tab:green"),
]:
    ax.plot(R_TOLS_S3, v0, "s--", color=c0, lw=2, label=f"A_TOL=0  avg={np.mean(v0):.4f}")
    ax.plot(R_TOLS_S3, v1, "o-",  color=c1, lw=2, label=f"A_TOL=1  avg={np.mean(v1):.4f}")
    ax.axhline(float(np.mean(v0 + v1)), color="gray", lw=1.5, linestyle=":",
               label=f"12-avg={float(np.mean(v0+v1)):.4f}")
    ax.set_xlabel("R_TOL  (D_TOL=0)"); ax.set_ylabel(label)
    ax.set_title(f"{label} vs R_TOL (wide, gamma4)")
    ax.set_ylim(0, 1.05); ax.set_xticks(R_TOLS_S3); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle(
    f"Stage 4: nuScenes-style mAP (wide, GAMMA={FOCAL_GAMMA}, ALPHA={FOCAL_ALPHA_POS})\n"
    f"AP_cy={s4_ap_cy:.4f}  AP_ve={s4_ap_ve:.4f}  mAP={s4_mAP:.4f}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage4_nuscenes_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# 論文用 mAP vs R_TOL
fig_p, ax_p = plt.subplots(figsize=(6, 5))
ax_p.plot(R_TOLS_S3, mAP_vals_a0, "s--", color="darkgreen", lw=2,
          label=f"A_TOL=0  avg={float(np.mean(mAP_vals_a0)):.4f}")
ax_p.plot(R_TOLS_S3, mAP_vals_a1, "o-",  color="tab:green", lw=2,
          label=f"A_TOL=1  avg={float(np.mean(mAP_vals_a1)):.4f}")
ax_p.axhline(s4_mAP, color="gray", lw=1.5, linestyle=":", label=f"avg={s4_mAP:.4f}")
ax_p.set_xlabel("R_TOL  (D_TOL=0)", fontsize=FP_LABEL)
ax_p.set_ylabel("mAP",              fontsize=FP_LABEL)
ax_p.tick_params(axis="both", labelsize=FP_TICK)
ax_p.set_ylim(0, 1.05); ax_p.set_xticks(R_TOLS_S3)
ax_p.legend(fontsize=FP_LEGEND); ax_p.grid(alpha=0.3)
ax_p.set_title(f"D_TOL=0, A_TOL∈{{0,1}}, R_TOL=0-5  mAP={s4_mAP:.4f}", fontsize=FP_TITLE)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "nuscenes_map_vs_rtol.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved: stage4_*.png")


# ===== Stage 6: r_diff 別 AP =====
print("\n=== Stage 6: r_diff-stratified AP ===")
RDIFF_BINS   = [(0, 2), (3, 5), (6, 10), (11, 9999)]
RDIFF_LABELS = ["r_diff=0-2", "r_diff=3-5", "r_diff=6-10", "r_diff>=11"]

stage6 = []
for (lo, hi), label in zip(RDIFF_BINS, RDIFF_LABELS):
    n_cy = sum(1 for s in cy_sample_info if lo <= s["r_diff"] <= hi)
    n_ve = sum(1 for s in ve_sample_info if lo <= s["r_diff"] <= hi)
    ap_cy_list, ap_ve_list = [], []
    for a_tol in [0, 1]:
        for r_tol in R_TOLS_S3:
            ap_cy_r, _, _ = compute_ap_rdiff(cy_raw, cy_sample_info, a_tol, 0, r_tol, lo, hi)
            ap_ve_r, _, _ = compute_ap_rdiff(ve_raw, ve_sample_info, a_tol, 0, r_tol, lo, hi)
            ap_cy_list.append(ap_cy_r); ap_ve_list.append(ap_ve_r)
    ap_cy_avg = float(np.mean(ap_cy_list))
    ap_ve_avg = float(np.mean(ap_ve_list))
    stage6.append(dict(label=label, lo=lo, hi=hi, n_cy=n_cy, n_ve=n_ve,
                       AP_cy=ap_cy_avg, AP_ve=ap_ve_avg, mAP=(ap_cy_avg+ap_ve_avg)/2))
    print(f"  {label:<16}  n={n_cy:>3}  AP_cy={ap_cy_avg:.4f}  AP_ve={ap_ve_avg:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(stage6)); w = 0.35
ax.bar(x - w/2, [r["AP_cy"] for r in stage6], w, color="tab:blue",   label="AP_cy", alpha=0.8)
ax.bar(x + w/2, [r["AP_ve"] for r in stage6], w, color="tab:orange", label="AP_ve", alpha=0.8)
for i, r in enumerate(stage6):
    ax.text(i - w/2, r["AP_cy"] + 0.01, f'n={r["n_cy"]}', ha="center", fontsize=8)
    ax.text(i + w/2, r["AP_ve"] + 0.01, f'n={r["n_ve"]}', ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in stage6])
ax.set_xlabel("r_diff bin"); ax.set_ylabel("AP (nuScenes-style)")
ax.set_title(f"Stage 6: AP by r_diff (wide, GAMMA={FOCAL_GAMMA}, ALPHA={FOCAL_ALPHA_POS})")
ax.set_ylim(0, 1.15); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage6_rdiff_ap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved: stage6_rdiff_ap.png")


# ===== PR曲線（論文用: COCO AP スタイル） =====
colors_r = plt.cm.plasma(np.linspace(0.1, 0.85, len(rows_a1)))
ap_cy_avg_coco = float(np.mean([r["AP_cy"] for r in rows_a1]))
ap_ve_avg_coco = float(np.mean([r["AP_ve"] for r in rows_a1]))

fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 5))
for ax, p_key, r_key, cls_label in [
    (axes_p[0], "precs_cy", "recs_cy", "Cyclist"),
    (axes_p[1], "precs_ve", "recs_ve", "Vehicle"),
]:
    ap_key = "AP_cy" if cls_label == "Cyclist" else "AP_ve"
    for rec, c in zip(rows_a1, colors_r):
        ax.step(rec[r_key], rec[p_key], where="post", color=c, lw=2,
                label=f"R_TOL={rec['R_TOL']}  AP={rec[ap_key]:.4f}")
    ax.set_xlabel("Recall",    fontsize=FP_LABEL)
    ax.set_ylabel("Precision", fontsize=FP_LABEL)
    ax.tick_params(axis="both", labelsize=FP_TICK)
    ax.set_title(cls_label, fontsize=FP_TITLE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FP_LEGEND); ax.grid(alpha=0.3)
fig_p.suptitle(
    f"A_TOL=1, D_TOL=0  AP_cy={ap_cy_avg_coco:.4f}  AP_ve={ap_ve_avg_coco:.4f}  (wide, gamma4)",
    fontsize=FP_TITLE,
)
plt.tight_layout()
plt.savefig(os.path.join(FOR_PAPER_DIR, "coco_ap_pr_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print("saved (for_paper): coco_ap_pr_curves.png")


# ===== 結果サマリ保存 =====
summary = {
    "model":         "wide_gamma4.pt",
    "focal_gamma":   FOCAL_GAMMA,
    "focal_alpha":   FOCAL_ALPHA_POS,
    "fixed_angles":  FIXED_ANGLES.tolist(),
    "n_eval":        len(eval_df),
    "n_gt_cy":       n_gt_cy,
    "n_gt_ve":       n_gt_ve,
    "best_val_loss": round(best_val_loss, 4),
    "stage4": {
        "AP_cy_avg": round(s4_ap_cy, 4),
        "AP_ve_avg": round(s4_ap_ve, 4),
        "mAP_avg":   round(s4_mAP,   4),
    },
    "stage6": [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
         if k not in ("ap_cy_per_rtol", "ap_ve_per_rtol")}
        for r in stage6
    ],
}
json_path = os.path.join(OUTPUT_DIR, "wide_gamma4_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"saved: {json_path}")

print(f"\n=== 最終結果 ===")
print(f"  Stage 4 mAP = {s4_mAP:.4f}  (AP_cy={s4_ap_cy:.4f}, AP_ve={s4_ap_ve:.4f})")
print(f"\nAll done.")
