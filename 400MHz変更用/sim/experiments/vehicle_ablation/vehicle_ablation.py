"""
Vehicle 検出性能アブレーション分析

PDCA Cycle 1: スコア分布分析
  TP/FP 別のsoftmax最大確率分布を cy vs ve で比較し、
  モデルがどの程度 TP/FP を分離できているかを確認する。

PDCA Cycle 2: シーンタイプ別 AP
  単一物体シーン / 2物体シーンに分けて AP を計算し、
  2物体シーンで ve AP が低下するかを確認する。

PDCA Cycle 3: FP 検出位置の混同分析
  2物体シーンで ve が FP になったとき、検出位置が
  cy GT 近傍に着地しているかどうかを確認する。
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

MODEL_PATH      = os.path.join(ROOT_DIR, "models", "best_detector_softmax_heatmap_single_train.pt")
SINGLE_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_single_object", "metadata.csv")
FIXED_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_fixed_angle",   "metadata.csv")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "vehicle_ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42
A_TOL, D_TOL, R_TOL = 1, 2, 3

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
        self.encoders  = nn.ModuleList([ConvBlock3D(1 if i == 0 else ch, ch, dropout) for i in range(4)])
        self.pools     = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck= ConvBlock3D(ch, ch, dropout)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders  = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head  = nn.Conv3d(ch, 3, 1)

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
    single = pd.read_csv(SINGLE_META_CSV)
    single = single[single["valid_all"]==1].reset_index(drop=True)
    single = single.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    holdout = single.iloc[340:].reset_index(drop=True)
    holdout["scene_type"] = "single"

    fixed = pd.read_csv(FIXED_META_CSV)
    fixed = fixed[fixed["valid_all"]==1].reset_index(drop=True)
    fixed = fixed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test  = fixed.iloc[200:].reset_index(drop=True)
    test["scene_type"] = "two"

    return pd.concat([holdout, test], ignore_index=True)


# ===== 推論・全情報収集 =====
def collect(model, df):
    records = []
    model.eval()
    with torch.no_grad():
        for i in range(len(df)):
            if i % 60 == 0: print(f"  {i}/{len(df)}", flush=True)
            row = df.iloc[i]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))

            x, data = load_sample(path)
            fa = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES

            vcy = int(str(row["valid_cyclist"]).strip() in ("1","True"))
            vve = int(str(row["valid_vehicle"]).strip() in ("1","True"))

            probs = F.softmax(model(torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)), dim=1)[0]
            # (3, N_FIXED, H, W)
            _, H, W = probs.shape[1], probs.shape[2], probs.shape[3]

            def peak(cls_idx):
                m = probs[cls_idx]          # (N, H, W)
                sc = m.max().item()
                fi = m.argmax().item()
                N2 = m.shape[0]
                ch = fi // (H*W); rem = fi % (H*W); d = rem//W; r = rem%W
                return sc, int(ch), int(d), int(r)

            cy_sc, cy_ch, cy_d, cy_r = peak(1)
            ve_sc, ve_ch, ve_d, ve_r = peak(2)

            # cy 真値・TP判定
            if vcy:
                cyd = int(row["cyclist_true_d_idx"]); cyr = int(row["cyclist_true_r_idx"])
                cych = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                tp_cy = (abs(cy_ch-cych)<=A_TOL and abs(cy_d-cyd)<=D_TOL and abs(cy_r-cyr)<=R_TOL)
            else:
                cyd = cyr = cych = None; tp_cy = False

            # ve 真値・TP判定
            if vve:
                ved = int(row["vehicle_true_d_idx"]); ver = int(row["vehicle_true_r_idx"])
                vech = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                tp_ve = (abs(ve_ch-vech)<=A_TOL and abs(ve_d-ved)<=D_TOL and abs(ve_r-ver)<=R_TOL)
            else:
                ved = ver = vech = None; tp_ve = False

            r_diff = abs(cyr - ver) if (vcy and vve) else None

            records.append(dict(
                scene=row["scene_type"],
                vcy=vcy, vve=vve,
                cy_sc=cy_sc, cy_ch=cy_ch, cy_d=cy_d, cy_r=cy_r,
                cy_true_ch=cych, cy_true_d=cyd, cy_true_r=cyr, tp_cy=tp_cy,
                ve_sc=ve_sc, ve_ch=ve_ch, ve_d=ve_d, ve_r=ve_r,
                ve_true_ch=vech, ve_true_d=ved, ve_true_r=ver, tp_ve=tp_ve,
                r_diff=r_diff,
            ))
    return pd.DataFrame(records)


# ===== AP計算 =====
def ap_from_list(dets, n_gt):
    """dets: list of (score, is_tp)"""
    dets = sorted(dets, key=lambda x: -x[0])
    tp = 0
    ps, rs = [1.0], [0.0]
    for i, (sc, itp) in enumerate(dets):
        if itp: tp += 1
        ps.append(tp/(i+1)); rs.append(tp/n_gt if n_gt else 0)
    ps, rs = np.array(ps), np.array(rs)
    return float(np.sum((rs[1:]-rs[:-1])*ps[1:])), ps, rs


# =============================================================
# PDCA Cycle 1: スコア分布分析
# =============================================================
print("\n=== Cycle 1: スコア分布分析 ===")
model = RadarUNet3DSoftmax().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
df = build_eval_df()
print(f"eval: {len(df)} samples")
rec = collect(model, df)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
bins = np.linspace(0, 1, 40)

for row_i, (cls, col_sc, col_tp) in enumerate([("Cyclist","cy_sc","tp_cy"), ("Vehicle","ve_sc","tp_ve")]):
    mask_gt  = rec[f"v{'cy' if cls=='Cyclist' else 've'}"] == 1
    tp_scores  = rec.loc[mask_gt  &  rec[col_tp], col_sc].values
    fp_gt_scores = rec.loc[mask_gt & ~rec[col_tp], col_sc].values   # GT あり・外れ
    fp_ng_scores = rec.loc[~mask_gt, col_sc].values                 # GT なし・誤検出

    ax = axes[row_i][0]
    ax.hist(tp_scores,    bins=bins, alpha=0.6, color="tab:green",  label=f"TP  (n={len(tp_scores)})")
    ax.hist(fp_gt_scores, bins=bins, alpha=0.6, color="tab:orange", label=f"FP(wrong pos) (n={len(fp_gt_scores)})")
    ax.hist(fp_ng_scores, bins=bins, alpha=0.4, color="tab:red",    label=f"FP(no GT) (n={len(fp_ng_scores)})")
    ax.set_title(f"{cls}: Score Distribution (TP/FP)")
    ax.set_xlabel("max softmax prob"); ax.set_ylabel("count"); ax.legend(fontsize=8)

    # TP スコアの CDF（累積）
    ax2 = axes[row_i][1]
    for scores, label, color in [
        (tp_scores,    "TP",            "tab:green"),
        (fp_gt_scores, "FP(wrong pos)", "tab:orange"),
        (fp_ng_scores, "FP(no GT)",     "tab:red"),
    ]:
        if len(scores) == 0: continue
        sorted_s = np.sort(scores)
        cdf = np.arange(1, len(sorted_s)+1) / len(sorted_s)
        ax2.plot(sorted_s, cdf, color=color, label=label, lw=2)
    ax2.set_title(f"{cls}: Score CDF")
    ax2.set_xlabel("max softmax prob"); ax2.set_ylabel("cumulative fraction")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.suptitle("Cycle 1: Score Distribution by TP/FP", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cycle1_score_dist.png"), dpi=150, bbox_inches="tight")
print("saved: cycle1_score_dist.png")

# 数値サマリ
c1 = {}
for cls, col_sc, col_tp, vcol in [
    ("cy","cy_sc","tp_cy","vcy"), ("ve","ve_sc","tp_ve","vve")
]:
    tp_s  = rec.loc[rec[vcol]==1 &  rec[col_tp], col_sc]
    fp_wp = rec.loc[rec[vcol]==1 & ~rec[col_tp], col_sc]
    fp_ng = rec.loc[rec[vcol]==0, col_sc]
    c1[cls] = dict(
        tp_median  = float(tp_s.median())  if len(tp_s)  else 0,
        fp_wp_median= float(fp_wp.median()) if len(fp_wp) else 0,
        fp_ng_median= float(fp_ng.median()) if len(fp_ng) else 0,
        n_tp=len(tp_s), n_fp_wp=len(fp_wp), n_fp_ng=len(fp_ng),
    )
    print(f"  {cls}: TP中央値={c1[cls]['tp_median']:.3f}  "
          f"FP(wrong)中央値={c1[cls]['fp_wp_median']:.3f}  "
          f"FP(noGT)中央値={c1[cls]['fp_ng_median']:.3f}")


# =============================================================
# PDCA Cycle 2: シーンタイプ別 AP
# =============================================================
print("\n=== Cycle 2: シーンタイプ別 AP ===")

results_c2 = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for cls_i, (cls, col_sc, col_tp, vcol) in enumerate([
    ("Cyclist","cy_sc","tp_cy","vcy"), ("Vehicle","ve_sc","tp_ve","vve")
]):
    ap_all, _, _ = ap_from_list(
        list(zip(rec[col_sc], rec[col_tp])),
        int((rec[vcol]==1).sum())
    )

    # single-only: 当該クラスのGTが存在するサンプル（＋そのシーンのみ）
    single_mask = rec["scene"] == "single"
    two_mask    = rec["scene"] == "two"

    ap_single, _, _ = ap_from_list(
        list(zip(rec.loc[single_mask, col_sc], rec.loc[single_mask, col_tp])),
        int((rec.loc[single_mask, vcol]==1).sum())
    )
    ap_two, _, _ = ap_from_list(
        list(zip(rec.loc[two_mask, col_sc], rec.loc[two_mask, col_tp])),
        int((rec.loc[two_mask, vcol]==1).sum())
    )
    results_c2[cls] = dict(ap_all=ap_all, ap_single=ap_single, ap_two=ap_two)
    print(f"  {cls}: AP_all={ap_all:.4f}  AP_single={ap_single:.4f}  AP_two={ap_two:.4f}")

    ax = axes[cls_i]
    bars = ax.bar(["All", "Single", "Two-object"], [ap_all, ap_single, ap_two],
                  color=["tab:gray","tab:blue","tab:orange"])
    ax.set_ylim(0, 1.05); ax.set_title(f"{cls} AP by Scene Type")
    ax.set_ylabel("AP"); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, [ap_all, ap_single, ap_two]):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}", ha="center", fontsize=9)

plt.suptitle("Cycle 2: AP by Scene Type", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cycle2_scene_ap.png"), dpi=150, bbox_inches="tight")
print("saved: cycle2_scene_ap.png")


# =============================================================
# PDCA Cycle 3: FP ve 検出位置の混同分析（2物体シーン限定）
# =============================================================
print("\n=== Cycle 3: FP ve 位置混同分析（2物体シーン） ===")

two = rec[(rec["scene"]=="two") & (rec["vcy"]==1) & (rec["vve"]==1)].copy()
tp_ve_two  = two[two["tp_ve"]==True]
fp_ve_two  = two[two["tp_ve"]==False]  # FP: ve 検出が GT を外れた

print(f"  2物体シーン: TP_ve={len(tp_ve_two)}, FP_ve={len(fp_ve_two)}")

# FP ve 検出の位置ずれ（ve GT 基準 / cy GT 基準）
if len(fp_ve_two) > 0:
    fp = fp_ve_two.copy()
    fp["dd_to_ve"] = fp["ve_d"] - fp["ve_true_d"]
    fp["dr_to_ve"] = fp["ve_r"] - fp["ve_true_r"]
    fp["dd_to_cy"] = fp["ve_d"] - fp["cy_true_d"]
    fp["dr_to_cy"] = fp["ve_r"] - fp["cy_true_r"]
    fp["dist_to_ve"] = np.sqrt(fp["dd_to_ve"]**2 + fp["dr_to_ve"]**2)
    fp["dist_to_cy"] = np.sqrt(fp["dd_to_cy"]**2 + fp["dr_to_cy"]**2)

    print(f"  FP ve 検出の ve_GT 距離（平均）: {fp['dist_to_ve'].mean():.2f} bins")
    print(f"  FP ve 検出の cy_GT 距離（平均）: {fp['dist_to_cy'].mean():.2f} bins")
    print(f"  FP ve が cy GT より ve GT に近い件数: "
          f"{(fp['dist_to_ve'] < fp['dist_to_cy']).sum()}/{len(fp)}")

    # TP ve の位置ずれも参照用に計算
    tp = tp_ve_two.copy()
    tp["dd_to_ve"] = tp["ve_d"] - tp["ve_true_d"]
    tp["dr_to_ve"] = tp["ve_r"] - tp["ve_true_r"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 散布図: (Δr_to_ve, Δd_to_ve) — TP vs FP
    ax = axes[0]
    ax.scatter(tp["dr_to_ve"], tp["dd_to_ve"], alpha=0.5, s=30,
               color="tab:green", label=f"TP (n={len(tp)})")
    ax.scatter(fp["dr_to_ve"], fp["dd_to_ve"], alpha=0.7, s=50,
               color="tab:red",   label=f"FP (n={len(fp)})", marker="x")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    rect_r = plt.Rectangle((-R_TOL-0.5, -D_TOL-0.5), 2*R_TOL+1, 2*D_TOL+1,
                            fill=False, edgecolor="blue", lw=1.5, linestyle="--", label="hit region")
    ax.add_patch(rect_r)
    ax.set_xlabel("Δr (det − ve_GT)"); ax.set_ylabel("Δd (det − ve_GT)")
    ax.set_title("ve detection offset (ref: ve GT)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 散布図: FP ve の位置を cy GT 基準で表示
    ax = axes[1]
    ax.scatter(fp["dr_to_cy"], fp["dd_to_cy"], alpha=0.8, s=50,
               color="tab:orange", label=f"FP ve (ref: cy GT, n={len(fp)})", marker="x")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    rect_c = plt.Rectangle((-R_TOL-0.5, -D_TOL-0.5), 2*R_TOL+1, 2*D_TOL+1,
                            fill=False, edgecolor="tab:blue", lw=1.5, linestyle="--", label="hit region")
    ax.add_patch(rect_c)
    ax.set_xlabel("Δr (det − cy_GT)"); ax.set_ylabel("Δd (det − cy_GT)")
    ax.set_title("FP ve position (ref: cy GT)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # r_diff 別の TP/FP 分布
    ax = axes[2]
    bins_rd = [0, 2, 5, 10, 999]
    labels_rd = ["0-2", "3-5", "6-10", ">=11"]
    tp_counts, fp_counts = [], []
    for lo, hi in zip(bins_rd[:-1], bins_rd[1:]):
        t = two[(two["r_diff"] >= lo) & (two["r_diff"] < hi)]
        tp_counts.append((t["tp_ve"]==True).sum())
        fp_counts.append((t["tp_ve"]==False).sum())
    x_pos = np.arange(len(labels_rd))
    ax.bar(x_pos-0.2, tp_counts, 0.4, color="tab:green", label="TP")
    ax.bar(x_pos+0.2, fp_counts, 0.4, color="tab:red",   label="FP/miss")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_rd)
    ax.set_xlabel("r_diff (range bin gap)"); ax.set_ylabel("count")
    ax.set_title("ve TP/FP by r_diff (two-object)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cycle 3: FP ve Position Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cycle3_fp_position.png"), dpi=150, bbox_inches="tight")
    print("saved: cycle3_fp_position.png")

    # FP ve の詳細リスト
    fp_detail = fp[["r_diff","ve_d","ve_r","ve_true_d","ve_true_r",
                     "cy_true_d","cy_true_r","dd_to_ve","dr_to_ve",
                     "dd_to_cy","dr_to_cy","dist_to_ve","dist_to_cy"]].reset_index(drop=True)
    fp_detail.to_csv(os.path.join(OUTPUT_DIR, "cycle3_fp_ve_detail.csv"), index=False)
    print("saved: cycle3_fp_ve_detail.csv")
    print(fp_detail.to_string())
else:
    print("  FP ve なし（2物体シーン）")


# =============================================================
# 統合レポート
# =============================================================
report_lines = [
    "=" * 60,
    "Vehicle 検出アブレーション分析レポート",
    "=" * 60,
    "",
    "【Cycle 1: スコア分布】",
]
for cls in ["cy", "ve"]:
    d = c1[cls]
    report_lines += [
        f"  {cls}: TP中央値={d['tp_median']:.3f}  "
        f"FP(wrong)中央値={d['fp_wp_median']:.3f}  "
        f"FP(noGT)中央値={d['fp_ng_median']:.3f}",
        f"       件数: TP={d['n_tp']}, FP(wrong pos)={d['n_fp_wp']}, FP(no GT)={d['n_fp_ng']}",
    ]

report_lines += ["", "【Cycle 2: シーンタイプ別 AP】"]
for cls, d in results_c2.items():
    report_lines.append(
        f"  {cls}: AP_all={d['ap_all']:.4f}  "
        f"AP_single={d['ap_single']:.4f}  "
        f"AP_two={d['ap_two']:.4f}"
    )

if len(fp_ve_two) > 0:
    report_lines += [
        "", "【Cycle 3: FP ve 位置混同（2物体）】",
        f"  FP ve 件数 = {len(fp_ve_two)}",
        f"  ve GT 距離（平均）= {fp['dist_to_ve'].mean():.2f} bins",
        f"  cy GT 距離（平均）= {fp['dist_to_cy'].mean():.2f} bins",
        f"  cy GT より ve GT に近い件数 = "
        f"{(fp['dist_to_ve']<fp['dist_to_cy']).sum()}/{len(fp)}",
    ]

report_lines += ["", "【出力ファイル】",
    "  cycle1_score_dist.png", "  cycle2_scene_ap.png",
    "  cycle3_fp_position.png", "  cycle3_fp_ve_detail.csv",
    "  ablation_report.txt",
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
with open(os.path.join(OUTPUT_DIR, "ablation_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
