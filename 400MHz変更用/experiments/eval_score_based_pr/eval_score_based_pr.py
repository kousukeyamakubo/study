"""
スコアベース PR 曲線評価スクリプト

NNとCFAR両方に対してスコアベースのPR曲線を計算する。
- NN:   NMS閾値=0.5固定, softmax確率値をスコアとして使用
- CFAR: pfa固定, rd_db値をスコアとして使用
  - バリアント1: tar_thresh無し（全CFAR通過検出）
  - バリアント2: tar_thresh=-28dB固定

スコアベースの手順：
  1. パラメータ固定で全サンプルの全検出を収集
  2. スコア降順にソート
  3. 1件ずつTP/FP確定（1GTに対して最初のマッチのみTP）
  4. GT総数で正規化してRecallを計算
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
from datetime import datetime

# ===== 定数 =====
MIXED_META_CSV  = "../../learn_dataset_fixed_angle/metadata.csv"
V3_RESULTS_DIR  = "../sweep_v3/sweep_experiment_results_v3"
OUTPUT_DIR      = "./eval_score_based_pr_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

NMS_THRESHOLD   = 0.5
CFAR_PFA        = 1e-3
CFAR_N_TRAIN    = (1, 1, 10)
CFAR_N_GUARD    = (1, 1, 5)
CFAR_TAR_THRESH = -28  # 固定版の強度閾値 [dB]

TARGET_RUNS = [
    "D_tk3_fp0.001_mixed",
    "D_dw_fp0.001_single",
    "D_dw_fp0.001_mixed",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)


# ===== モデル定義 (eval_pr_roc.py と同一) =====
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
        self.upsamples  = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2),
                          mode="trilinear", align_corners=False) for _ in range(4)])
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


# ===== 3D CFAR (eval_pr_roc.py と同一) =====
def ca_cfar_3d(x, n_train=(2, 1, 7), n_guard=(1, 1, 5), pfa=1e-4):
    B, C, A, H, W = x.shape
    ta, th, tw = n_train
    ga, gh, gw = n_guard
    ka, kh, kw = 2*(ta+ga)+1, 2*(th+gh)+1, 2*(tw+gw)+1

    kernel = torch.ones((1, 1, ka, kh, kw), dtype=x.dtype)
    kernel[:, :, ta:ta+2*ga+1, th:th+2*gh+1, tw:tw+2*gw+1] = 0
    N_train = kernel.sum().item()
    alpha   = N_train * (pfa ** (-1.0 / N_train) - 1.0)

    pad_a, pad_h, pad_w = ka//2, kh//2, kw//2
    x_pad = F.pad(x, (pad_w, pad_w, 0, 0, 0, 0), mode="replicate")
    x_pad = F.pad(x_pad, (0, 0, pad_h, pad_h, 0, 0), mode="circular")
    x_pad = F.pad(x_pad, (0, 0, 0, 0, pad_a, pad_a), mode="replicate")

    noise_est = F.conv3d(x_pad, kernel) / N_train
    return x > alpha * noise_est


# ===== データ読み込み =====
def load_rd_maps_db(npz_path):
    d = np.load(npz_path)
    return np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                     for i in range(d["rd_maps"].shape[0])], axis=0)

def load_rd_maps_linear(npz_path):
    d = np.load(npz_path)
    return np.abs(d["rd_maps"]).astype(np.float32)

def load_angle_info(npz_path):
    d = np.load(npz_path)
    fa     = d["fixed_angles"].astype(float) if "fixed_angles" in d else FIXED_ANGLES
    cy_ach = int(np.argmin(np.abs(fa - float(d["cyclist_true_angle_deg"])))) \
             if "cyclist_true_angle_deg" in d else None
    ve_ach = int(np.argmin(np.abs(fa - float(d["vehicle_true_angle_deg"])))) \
             if "vehicle_true_angle_deg" in d else None
    return cy_ach, ve_ach


# ===== スコア付き NMS デコード =====
def decode_detections_with_score(logits, threshold=0.5):
    """
    NMS付きデコード。ピーク位置と確率値をスコアとして返す。
    戻り値: (cy_dets, ve_dets)  各要素は (ch, d, r, score) のリスト
    """
    probs = F.softmax(logits, dim=1)
    def _nms(pmap, thr):
        # pmap: (N_FIXED, H, W)
        p = pmap.clone(); N, H, W = p.shape; dets = []
        while p.max().item() >= thr:
            fi  = torch.argmax(p).item()
            ch, rem = fi // (H*W), fi % (H*W)
            d, r    = rem // W, rem % W
            score   = p[ch, d, r].item()
            dets.append((ch, d, r, score))
            p[max(ch-1,0):ch+2, max(d-3,0):d+4, max(r-1,0):r+2] = 0.
        return dets
    return _nms(probs[0, 1], threshold), _nms(probs[0, 2], threshold)


# ===== NN 全検出収集 =====
def collect_nn_detections(model, df):
    """
    全サンプルの NN 検出をスコア付きで収集する。
    戻り値: list of {"sample_idx", "class", "ch", "d", "r", "score"}
    """
    all_dets = []
    for i in range(len(df)):
        row = df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        x_t = torch.from_numpy(load_rd_maps_db(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            logits = model(x_t)
        cy_dets, ve_dets = decode_detections_with_score(logits.cpu(), NMS_THRESHOLD)
        for ch, d, r, score in cy_dets:
            all_dets.append({"sample_idx": i, "class": "cy",
                             "ch": ch, "d": d, "r": r, "score": score})
        for ch, d, r, score in ve_dets:
            all_dets.append({"sample_idx": i, "class": "ve",
                             "ch": ch, "d": d, "r": r, "score": score})
    return all_dets


# ===== CFAR キャッシュ構築 =====
def build_cfar_cache(df):
    """CFAR マスクと真値をキャッシュする（tar_thresh ループの高速化）"""
    cache = []
    for i in range(len(df)):
        row    = df.iloc[i]
        npz    = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        rd_lin = load_rd_maps_linear(npz)  # (A, H, W)
        x      = torch.from_numpy(rd_lin).unsqueeze(0).unsqueeze(0)  # (1, 1, A, H, W)
        mask   = ca_cfar_3d(x, n_train=CFAR_N_TRAIN, n_guard=CFAR_N_GUARD,
                             pfa=CFAR_PFA)[0, 0].numpy()  # (A, H, W) bool
        rd_db  = 20. * np.log10(np.maximum(rd_lin, 1e-12))
        cy_ach = int(np.argmin(np.abs(FIXED_ANGLES - float(row["cyclist_true_angle_deg"])))) \
                 if "cyclist_true_angle_deg" in row else None
        ve_ach = int(np.argmin(np.abs(FIXED_ANGLES - float(row["vehicle_true_angle_deg"])))) \
                 if "vehicle_true_angle_deg" in row else None
        cache.append({
            "mask": mask, "rd_db": rd_db,
            "cy_d": int(row["cyclist_true_d_idx"]) if "cyclist_true_d_idx" in row else None,
            "cy_r": int(row["cyclist_true_r_idx"]) if "cyclist_true_r_idx" in row else None,
            "cy_a": cy_ach,
            "ve_d": int(row["vehicle_true_d_idx"])  if "vehicle_true_d_idx" in row else None,
            "ve_r": int(row["vehicle_true_r_idx"])  if "vehicle_true_r_idx" in row else None,
            "ve_a": ve_ach,
        })
    return cache


# ===== CFAR 全検出収集 =====
def collect_cfar_detections(cache, tar_thresh=None):
    """
    全サンプルの CFAR 検出をスコア付きで収集する。
    tar_thresh=None: 全 CFAR 通過検出を使用
    tar_thresh=X:    CFAR 通過かつ rd_db > X の検出のみ使用
    戻り値: list of {"sample_idx", "ch", "d", "r", "score"}
    """
    all_dets = []
    for i, c in enumerate(cache):
        mask = c["mask"].copy()
        if tar_thresh is not None:
            # tar_thresh は「スコア下限」なので、これ以下の検出をリストから除外する
            mask = mask & (c["rd_db"] > tar_thresh)
        locs = np.argwhere(mask)  # (K, 3): [a, d, r]
        for a, d, r in locs:
            score = float(c["rd_db"][a, d, r])
            all_dets.append({"sample_idx": i, "ch": int(a),
                             "d": int(d), "r": int(r), "score": score})
    return all_dets


# ===== GT リスト構築 =====
def build_gts(cache):
    """
    サンプルインデックスをキーとする GT リストを構築する。
    gts[i] = [{"class": "cy"/"ve", "ch": ..., "d": ..., "r": ...}, ...]
    """
    gts = {}
    for i, c in enumerate(cache):
        entries = []
        if c["cy_d"] is not None:
            entries.append({"class": "cy", "ch": c["cy_a"], "d": c["cy_d"], "r": c["cy_r"]})
        if c["ve_d"] is not None:
            entries.append({"class": "ve", "ch": c["ve_a"], "d": c["ve_d"], "r": c["ve_r"]})
        gts[i] = entries
    return gts


# ===== スコアベース PR 曲線計算 =====
def compute_score_based_pr(dets, gts, n_gt, class_filter=None, cfar_agnostic=False):
    """
    スコア降順で全検出を評価し PR 曲線を生成する。

    class_filter:   "cy" / "ve" / None (aggregate)
    cfar_agnostic:  True の場合 CFAR 検出として扱い cy/ve 両 GT とマッチ可能にする
                    False の場合 det["class"] と gt["class"] を一致させる

    一対一対応: 1GT に対して最初にマッチした検出のみ TP
    """
    if class_filter is not None:
        dets = [d for d in dets if d.get("class") == class_filter]

    dets_sorted = sorted(dets, key=lambda x: -x["score"])
    matched_gts = set()  # (sample_idx, gt_class, gt_idx)
    precisions, recalls = [], []
    tp = fp = 0

    for det in dets_sorted:
        si    = det["sample_idx"]
        is_tp = False

        for gi, gt in enumerate(gts.get(si, [])):
            if not cfar_agnostic:
                # NN: 検出クラスと GT クラスを一致させる
                if det.get("class") != gt["class"]:
                    continue

            gt_key = (si, gt["class"], gi)
            if gt_key in matched_gts:
                continue

            # 位置判定（A_TOL / D_TOL / R_TOL）
            if gt["ch"] is not None and abs(det["ch"] - gt["ch"]) > A_TOL:
                continue
            if abs(det["d"] - gt["d"]) <= D_TOL and abs(det["r"] - gt["r"]) <= R_TOL:
                matched_gts.add(gt_key)
                is_tp = True
                break

        tp += int(is_tp)
        fp += int(not is_tp)
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)

    return precisions, recalls


# ===== AP 計算 =====
def compute_ap(precisions, recalls):
    """PR 曲線から AP を台形積分で計算（単調化後）"""
    if len(precisions) == 0:
        return 0.0
    order = np.argsort(recalls)
    r = np.array(recalls)[order]
    p = np.array(precisions)[order]
    p_env = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.trapz(p_env, r))


# ===== データ準備 =====
mixed_df    = pd.read_csv(MIXED_META_CSV)
mixed_df    = mixed_df[mixed_df["valid_all"]==1].reset_index(drop=True)
mixed_df    = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
holdout2obj = mixed_df.iloc[200:].reset_index(drop=True)
N           = len(holdout2obj)
print(f"holdout2obj: {N} 件", flush=True)


# ===== CFAR キャッシュ・GT 構築 =====
print("\n[CFAR] キャッシュ構築中...", flush=True)
cfar_cache = build_cfar_cache(holdout2obj)
gts        = build_gts(cfar_cache)
n_gt_cy    = sum(1 for v in gts.values() for g in v if g["class"] == "cy")
n_gt_ve    = sum(1 for v in gts.values() for g in v if g["class"] == "ve")
n_gt_agg   = n_gt_cy + n_gt_ve
print(f"[CFAR] 完了  GT: cy={n_gt_cy}, ve={n_gt_ve}, total={n_gt_agg}", flush=True)


# ===== CFAR 検出収集・PR 計算 =====
print("\n[CFAR] 検出収集中...", flush=True)
cfar_dets_all = collect_cfar_detections(cfar_cache, tar_thresh=None)
cfar_dets_t28 = collect_cfar_detections(cfar_cache, tar_thresh=CFAR_TAR_THRESH)
print(f"  全通過: {len(cfar_dets_all)} 件,  {CFAR_TAR_THRESH}dB 固定: {len(cfar_dets_t28)} 件", flush=True)

cfar_p_all, cfar_r_all = compute_score_based_pr(cfar_dets_all, gts, n_gt_agg, cfar_agnostic=True)
cfar_p_t28, cfar_r_t28 = compute_score_based_pr(cfar_dets_t28, gts, n_gt_agg, cfar_agnostic=True)
cfar_ap_all = compute_ap(cfar_p_all, cfar_r_all)
cfar_ap_t28 = compute_ap(cfar_p_t28, cfar_r_t28)
print(f"  AP (全通過): {cfar_ap_all:.4f},  AP ({CFAR_TAR_THRESH}dB): {cfar_ap_t28:.4f}", flush=True)


# ===== NN 評価ループ =====
all_nn_results = {}

for name in TARGET_RUNS:
    pt_path = os.path.join(V3_RESULTS_DIR, f"{name}.pt")
    if not os.path.exists(pt_path):
        print(f"[{name}] モデルファイル未検出: {pt_path}", flush=True)
        continue

    print(f"\n{'='*50}", flush=True)
    print(f"[{name}] 検出収集開始", flush=True)

    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()

    nn_dets   = collect_nn_detections(model, holdout2obj)
    n_cy_dets = sum(1 for d in nn_dets if d["class"] == "cy")
    n_ve_dets = sum(1 for d in nn_dets if d["class"] == "ve")
    print(f"  検出数: cy={n_cy_dets}, ve={n_ve_dets}, total={len(nn_dets)}", flush=True)

    # クラス別 PR
    p_cy,  r_cy  = compute_score_based_pr(nn_dets, gts, n_gt_cy,  class_filter="cy")
    p_ve,  r_ve  = compute_score_based_pr(nn_dets, gts, n_gt_ve,  class_filter="ve")
    # aggregate PR（cy/ve 混合でスコア降順、クラス一致を強制）
    p_agg, r_agg = compute_score_based_pr(nn_dets, gts, n_gt_agg, class_filter=None)

    ap_cy  = compute_ap(p_cy,  r_cy)
    ap_ve  = compute_ap(p_ve,  r_ve)
    ap_agg = compute_ap(p_agg, r_agg)
    map_   = (ap_cy + ap_ve) / 2

    print(f"  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  AP_agg={ap_agg:.4f}  mAP={map_:.4f}", flush=True)

    all_nn_results[name] = {
        "p_cy":  p_cy,  "r_cy":  r_cy,
        "p_ve":  p_ve,  "r_ve":  r_ve,
        "p_agg": p_agg, "r_agg": r_agg,
        "ap_cy": ap_cy, "ap_ve": ap_ve, "ap_agg": ap_agg, "map": map_,
    }


# ===== JSON 保存（PR 曲線は間引いて保存）=====
def downsample_curve(p, r, max_points=300):
    """PR 曲線を等間隔で max_points 点にダウンサンプリングする"""
    if len(p) <= max_points:
        return p, r
    idx = np.round(np.linspace(0, len(p) - 1, max_points)).astype(int)
    return [p[i] for i in idx], [r[i] for i in idx]

results = {
    "nn": {},
    "cfar": {
        "all": {"ap": cfar_ap_all, **dict(zip(["p", "r"], downsample_curve(cfar_p_all, cfar_r_all)))},
        "t28": {"ap": cfar_ap_t28, **dict(zip(["p", "r"], downsample_curve(cfar_p_t28, cfar_r_t28)))},
    },
    "meta": {
        "generated":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nms_threshold":   NMS_THRESHOLD,
        "cfar_pfa":        CFAR_PFA,
        "cfar_n_train":    list(CFAR_N_TRAIN),
        "cfar_n_guard":    list(CFAR_N_GUARD),
        "cfar_tar_thresh": CFAR_TAR_THRESH,
        "d_tol": D_TOL, "r_tol": R_TOL, "a_tol": A_TOL,
        "n_holdout": N, "n_gt_cy": n_gt_cy, "n_gt_ve": n_gt_ve,
    }
}

for name, res in all_nn_results.items():
    p_cy_ds,  r_cy_ds  = downsample_curve(res["p_cy"],  res["r_cy"])
    p_ve_ds,  r_ve_ds  = downsample_curve(res["p_ve"],  res["r_ve"])
    p_agg_ds, r_agg_ds = downsample_curve(res["p_agg"], res["r_agg"])
    results["nn"][name] = {
        "ap_cy": res["ap_cy"], "ap_ve": res["ap_ve"],
        "ap_agg": res["ap_agg"], "map": res["map"],
        "p_cy":  p_cy_ds,  "r_cy":  r_cy_ds,
        "p_ve":  p_ve_ds,  "r_ve":  r_ve_ds,
        "p_agg": p_agg_ds, "r_agg": r_agg_ds,
    }

json_path = os.path.join(OUTPUT_DIR, "score_based_pr_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nJSON保存: {json_path}", flush=True)


# ===== プロット =====
COLORS = {
    "D_tk3_fp0.001_mixed":  ("tab:blue",   "topK (K=3, mixed)"),
    "D_dw_fp0.001_single":  ("tab:orange", "daware (single)"),
    "D_dw_fp0.001_mixed":   ("tab:green",  "daware (mixed)"),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Score-based PR Curves (holdout 2-object, {N} samples)\n"
    f"NN: NMS thr={NMS_THRESHOLD},  CFAR: pfa={CFAR_PFA}"
)

panels = [
    ("p_cy",  "r_cy",  "ap_cy", n_gt_cy,  "Cyclist PR"),
    ("p_ve",  "r_ve",  "ap_ve", n_gt_ve,  "Vehicle PR"),
    ("p_agg", "r_agg", "ap_agg", n_gt_agg, "Aggregate PR"),
]

for ax, (pk, rk, apk, ngt, title) in zip(axes, panels):
    ax.set_title(f"{title}  (GT={ngt})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    for name in TARGET_RUNS:
        if name not in all_nn_results:
            continue
        col, lbl = COLORS[name]
        res = all_nn_results[name]
        ax.plot(res[rk], res[pk], "-", color=col,
                label=f"{lbl} (AP={res[apk]:.3f})", lw=1.5)

    ax.legend(fontsize=7, loc="center")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "score_based_pr_curves.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"PR曲線保存: {plot_path}", flush=True)


# ===== AP サマリー =====
print("\n" + "="*65, flush=True)
print("Score-based AP / mAP サマリー")
print(f"{'run':30s}  {'AP_cy':>6}  {'AP_ve':>6}  {'mAP':>6}  {'AP_agg':>7}")
print("-"*65)
for name in TARGET_RUNS:
    if name not in all_nn_results:
        continue
    r = all_nn_results[name]
    print(f"{name:30s}  {r['ap_cy']:>6.4f}  {r['ap_ve']:>6.4f}"
          f"  {r['map']:>6.4f}  {r['ap_agg']:>7.4f}", flush=True)
print(f"{'CFAR (全通過)':30s}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {cfar_ap_all:>7.4f}", flush=True)
print(f"{'CFAR ('+str(CFAR_TAR_THRESH)+'dB)':30s}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {cfar_ap_t28:>7.4f}", flush=True)
print("="*65, flush=True)
print(f"\n完了。結果は {OUTPUT_DIR}/ に保存されました。", flush=True)
