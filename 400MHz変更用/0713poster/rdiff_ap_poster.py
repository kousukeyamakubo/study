# ポスター Experiments 帯用: r_diff 別 AP 棒グラフ
# 許容誤差はポスターの PR 曲線 (coco_ap_pr_curves.png) と同一の
# A_TOL=1, D_TOL=0, R_TOL=1 に統一する（12通り平均だった stage6 と数値を揃えるため）。
# 検出収集・AP 計算は experiments/eval_nn/tolerance_sweep.py と同一ロジック。
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== パス（スクリプト位置基準） =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH     = os.path.join(ROOT_DIR, "models", "best_detector_narrow_angle.pt")
FIXED_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed", "metadata.csv")
CACHE_PATH     = os.path.join(SCRIPT_DIR, "rdiff_raw_cache.npz")
OUT_PNG        = os.path.join(SCRIPT_DIR, "rdiff_ap_poster.png")

# ===== 統一許容誤差（PR 曲線と同一） =====
A_TOL, D_TOL, R_TOL = 1, 0, 1

N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
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


def load_sample(path):
    data = np.load(path)
    x = np.stack([20*np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                  for i in range(data["rd_maps"].shape[0])], axis=0)
    return x, data


def collect_raw(model, eval_df):
    # tolerance_sweep.py の collect_raw と同一（Stage 6 に必要な項目のみ）
    cy_raw, ve_raw = [], []
    cy_info, ve_info = [], []
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
            pred = probs.argmax(dim=0)  # (N_FIXED, H, W)

            r_diff = (abs(int(row["cyclist_true_r_idx"]) - int(row["vehicle_true_r_idx"]))
                      if vcy and vve else -1)

            def voxels_for_class(cls_idx):
                mask = (pred == cls_idx)
                if not mask.any():
                    return []
                return [(int(ij[0]), int(ij[1]), int(ij[2]),
                         probs[cls_idx, int(ij[0]), int(ij[1]), int(ij[2])].item())
                        for ij in mask.nonzero(as_tuple=False)]

            for cls_idx, valid, angle_key, d_key, r_key, raw, info, is_cy in (
                (1, vcy, "cyclist_true_angle_deg", "cyclist_true_d_idx", "cyclist_true_r_idx", cy_raw, cy_info, True),
                (2, vve, "vehicle_true_angle_deg", "vehicle_true_d_idx", "vehicle_true_r_idx", ve_raw, ve_info, False),
            ):
                voxels = voxels_for_class(cls_idx)
                if valid:
                    if is_cy: n_gt_cy += 1
                    else:     n_gt_ve += 1
                    tch = int(np.argmin(np.abs(fa - float(data[angle_key]))))
                    tru = dict(valid=True, true_ch=tch, true_d=int(row[d_key]), true_r=int(row[r_key]),
                               sample_idx=idx, r_diff=r_diff)
                    info.append(dict(sample_idx=idx, r_diff=r_diff))
                else:
                    tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                               sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in voxels:
                    raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

    return cy_raw, ve_raw, cy_info, ve_info, n_gt_cy, n_gt_ve


def compute_ap(raw, n_gt, a_tol, d_tol, r_tol):
    # PASCAL VOC 方式 + greedy matching（tolerance_sweep.py と同一）
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
    matched, tp = set(), 0
    precs, recs = [], []
    for i, det in enumerate(sorted_dets):
        if det["is_hit"] and det["sample_idx"] not in matched:
            tp += 1
            matched.add(det["sample_idx"])
        precs.append(tp / (i + 1))
        recs.append(tp / n_gt if n_gt > 0 else 0.0)
    precs = np.concatenate([[1.0], precs])
    recs  = np.concatenate([[0.0], recs])
    return float(np.sum((recs[1:] - recs[:-1]) * precs[1:]))


def compute_ap_rdiff(raw, info, a_tol, d_tol, r_tol, lo, hi):
    raw_f  = [r for r in raw  if lo <= r["r_diff"] <= hi]
    n_gt_f = sum(1 for s in info if lo <= s["r_diff"] <= hi)
    return compute_ap(raw_f, n_gt_f, a_tol, d_tol, r_tol), n_gt_f


# ===== メイン =====
if os.path.exists(CACHE_PATH):
    print(f"Loading cache: {CACHE_PATH}")
    _c = np.load(CACHE_PATH, allow_pickle=True)
    cy_raw, ve_raw = list(_c["cy_raw"]), list(_c["ve_raw"])
    cy_info, ve_info = list(_c["cy_info"]), list(_c["ve_info"])
    n_gt_cy, n_gt_ve = int(_c["n_gt_cy"]), int(_c["n_gt_ve"])
else:
    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    df = pd.read_csv(FIXED_META_CSV)
    df = df[df["valid_all"] == 1].reset_index(drop=True)
    print(f"Samples: {len(df)}")
    cy_raw, ve_raw, cy_info, ve_info, n_gt_cy, n_gt_ve = collect_raw(model, df)
    np.savez(CACHE_PATH,
             cy_raw=np.array(cy_raw, dtype=object), ve_raw=np.array(ve_raw, dtype=object),
             cy_info=np.array(cy_info, dtype=object), ve_info=np.array(ve_info, dtype=object),
             n_gt_cy=n_gt_cy, n_gt_ve=n_gt_ve)
    print(f"Cache saved: {CACHE_PATH}")

# 全体 AP（PR 曲線の 0.8842 / 0.7752 と一致するかの確認用）
ap_cy_all = compute_ap(cy_raw, n_gt_cy, A_TOL, D_TOL, R_TOL)
ap_ve_all = compute_ap(ve_raw, n_gt_ve, A_TOL, D_TOL, R_TOL)
print(f"Overall (A={A_TOL}, D={D_TOL}, R={R_TOL}): AP_cy={ap_cy_all:.4f}  AP_ve={ap_ve_all:.4f}")

RDIFF_BINS   = [(0, 2), (3, 5), (6, 10), (11, 9999)]
RDIFF_LABELS = ["0-2", "3-5", "6-10", ">=11"]  # cp932 コンソールで print できるよう ASCII に限定

rows = []
for (lo, hi), label in zip(RDIFF_BINS, RDIFF_LABELS):
    (ap_cy, n_cy) = compute_ap_rdiff(cy_raw, cy_info, A_TOL, D_TOL, R_TOL, lo, hi)
    (ap_ve, n_ve) = compute_ap_rdiff(ve_raw, ve_info, A_TOL, D_TOL, R_TOL, lo, hi)
    rows.append(dict(label=label, AP_cy=ap_cy, AP_ve=ap_ve, n_cy=n_cy, n_ve=n_ve))
    print(f"  r_diff={label:<5} n={n_cy:>3}  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}")

# ===== ポスター用プロット =====
plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(rows))
w = 0.35
ax.bar(x - w/2, [r["AP_cy"] for r in rows], w, color="tab:blue",   label="Cyclist", alpha=0.85)
ax.bar(x + w/2, [r["AP_ve"] for r in rows], w, color="tab:orange", label="Vehicle", alpha=0.85)
for i, r in enumerate(rows):
    ax.text(i - w/2, r["AP_cy"] + 0.015, f'n={r["n_cy"]}', ha="center", fontsize=12)
    ax.text(i + w/2, r["AP_ve"] + 0.015, f'n={r["n_ve"]}', ha="center", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows], fontsize=14)
ax.set_xlabel("Range-bin gap between the two targets", fontsize=15)
ax.set_ylabel("AP", fontsize=15)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=13)
ax.grid(axis="y", alpha=0.3)
# 許容誤差設定は小さく図中に明記（PR 曲線との整合を示す）
ax.text(0.98, 0.02, f"A_TOL={A_TOL}, D_TOL={D_TOL}, R_TOL={R_TOL}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"saved: {OUT_PNG}")
