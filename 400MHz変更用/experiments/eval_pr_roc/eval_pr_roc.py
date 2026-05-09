"""
PR曲線・ROC曲線・AP/mAP 比較評価スクリプト (Phase 1: NMS込み閾値スイープ)

評価対象:
  - D_tk3_fp0.001_mixed  (topK, K=3)
  - D_dw_fp0.001_single  (daware, single学習)
  - D_dw_fp0.001_mixed   (daware, mixed学習)
  - 3D CA-CFAR           (pfa固定, tar_thresh スイープ)

指標:
  - PR曲線 per class (cy/ve)  on holdout2obj (100件)
  - ROC的曲線: TPR vs cross-class FPR  on single-target テスト
  - AP per class, mAP
  - 3D CFARは class-agnostic なので aggregate PR + recall vs FP/sample
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
from scipy.ndimage import maximum_filter
from datetime import datetime

# ===== 定数 =====
SINGLE_META_CSV = "../../learn_dataset_single_object/metadata.csv"
MIXED_META_CSV  = "../../learn_dataset_fixed_angle/metadata.csv"
V3_RESULTS_DIR  = "../sweep_v3/sweep_experiment_results_v3"
OUTPUT_DIR      = "./eval_pr_roc_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
D_TOL, R_TOL = 2, 3
A_TOL        = 1
RANDOM_SEED  = 42

TARGET_RUNS = [
    "D_tk3_fp0.001_mixed",
    "D_dw_fp0.001_single",
    "D_dw_fp0.001_mixed",
]

# NN: 0.05〜0.95 の 25 点（中央域を密に）
NN_THRESHOLDS = np.round(np.concatenate([
    np.arange(0.05, 0.20, 0.05),
    np.arange(0.20, 0.70, 0.025),
    np.arange(0.70, 0.96, 0.05),
]), 4)

# 3D CFAR: pfa 固定, tar_thresh をスイープ
CFAR_PFA     = 1e-3
CFAR_N_TRAIN = (1, 1, 10)
CFAR_N_GUARD = (1, 1, 5)
TAR_THRESH_LIST = list(range(-10, -38, -2))  # dB: -10 〜 -36

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)
print(f"NN thresholds ({len(NN_THRESHOLDS)}): {NN_THRESHOLDS}", flush=True)


# ===== モデル定義 (v3 と同一) =====
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


# ===== 3D CFAR 関数 (feature/3d-cfar から移植) =====
def ca_cfar_3d(x, n_train=(2, 1, 7), n_guard=(1, 1, 5), pfa=1e-4):
    """
    3次元 CA-CFAR
    x: (B, 1, A, H, W) 線形振幅マップ
    戻り値: (B, 1, A, H, W) Bool 検出マスク
    角度軸(A): replicate, ドップラー軸(H): circular, レンジ軸(W): replicate
    """
    B, C, A, H, W = x.shape
    ta, th, tw = n_train
    ga, gh, gw = n_guard
    ka, kh, kw = 2*(ta+ga)+1, 2*(th+gh)+1, 2*(tw+gw)+1

    kernel = torch.ones((1, 1, ka, kh, kw), dtype=x.dtype)
    kernel[:, :, ta:ta+2*ga+1, th:th+2*gh+1, tw:tw+2*gw+1] = 0
    N_train = kernel.sum().item()
    alpha   = N_train * (pfa ** (-1.0 / N_train) - 1.0)

    pad_a, pad_h, pad_w = ka//2, kh//2, kw//2
    # F.pad の順序: (W_left, W_right, H_left, H_right, A_left, A_right)
    x_pad = F.pad(x, (pad_w, pad_w, 0, 0, 0, 0), mode="replicate")
    x_pad = F.pad(x_pad, (0, 0, pad_h, pad_h, 0, 0), mode="circular")
    x_pad = F.pad(x_pad, (0, 0, 0, 0, pad_a, pad_a), mode="replicate")

    noise_est = F.conv3d(x_pad, kernel) / N_train
    return x > alpha * noise_est


# ===== データ読み込み =====
def load_rd_maps_db(npz_path):
    """RDマップを dB 変換して (N_FIXED, H, W) で返す"""
    d = np.load(npz_path)
    return np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                     for i in range(d["rd_maps"].shape[0])], axis=0)

def load_rd_maps_linear(npz_path):
    """線形振幅マップ (N_FIXED, H, W) を返す"""
    d = np.load(npz_path)
    return np.abs(d["rd_maps"]).astype(np.float32)

def load_angle_info(npz_path):
    d = np.load(npz_path)
    fa  = d["fixed_angles"].astype(float) if "fixed_angles" in d else FIXED_ANGLES
    cy_ach = int(np.argmin(np.abs(fa - float(d["cyclist_true_angle_deg"])))) \
             if "cyclist_true_angle_deg" in d else None
    ve_ach = int(np.argmin(np.abs(fa - float(d["vehicle_true_angle_deg"])))) \
             if "vehicle_true_angle_deg" in d else None
    return cy_ach, ve_ach


# ===== NN デコード =====
def decode_detections(logits, threshold=0.5):
    """NMS 付き検出デコード。(cy_dets, ve_dets) を返す。各要素は (ch, d, r)"""
    probs = F.softmax(logits, dim=1)
    def _nms(pmap, thr):
        p = pmap.clone(); N, H, W = p.shape; dets = []
        while p.max().item() >= thr:
            fi = torch.argmax(p).item()
            ch, rem = fi//(H*W), fi%(H*W); d, r = rem//W, rem%W
            dets.append((ch, d, r))
            p[max(ch-1,0):ch+2, max(d-3,0):d+4, max(r-1,0):r+2] = 0.
        return dets
    return _nms(probs[0,1], threshold), _nms(probs[0,2], threshold)


def _hit(dets, true_ach, true_d, true_r):
    for ch, d, r in dets:
        if true_ach is not None and abs(ch - true_ach) > A_TOL:
            continue
        if abs(d - true_d) <= D_TOL and abs(r - true_r) <= R_TOL:
            return True
    return False


# ===== NN 評価: holdout2obj (PR 用) =====
def eval_nn_holdout2obj(model, df, thr):
    """
    2物体 holdout セットで cy/ve それぞれの hit 数・検出数を集計。
    precision = hit_sum / det_sum, recall = hit_sum / N として PR 曲線を構成する。
    """
    cy_hits, ve_hits = 0, 0
    n_cy_dets, n_ve_dets = 0, 0
    N = len(df)
    for i in range(N):
        row   = df.iloc[i]
        npz   = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        cy_ach, ve_ach = load_angle_info(npz)
        x_t = torch.from_numpy(load_rd_maps_db(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        n_cy_dets += len(cy_dets)
        n_ve_dets += len(ve_dets)
        if "cyclist_true_d_idx" in row:
            cy_hits += int(_hit(cy_dets, cy_ach, int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])))
        if "vehicle_true_d_idx" in row:
            ve_hits += int(_hit(ve_dets, ve_ach, int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])))
    return cy_hits, ve_hits, n_cy_dets, n_ve_dets, N


# ===== NN 評価: single-target (ROC 用) =====
def eval_nn_single(model, cy_df, ve_df, thr):
    """
    単一物体テストで TPR・cross-class FPR を計算。
    cy_only: cy_hit = TPR_cy, ve_fp = FPR_cy (ve が出現)
    ve_only: ve_hit = TPR_ve, cy_fp = FPR_ve (cy が出現)
    """
    def _eval_class(df, target, other_ach_fn, tgt_d_col, tgt_r_col, model, thr):
        hits = fp = 0
        for i in range(len(df)):
            row = df.iloc[i]
            npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
            cy_ach, ve_ach = load_angle_info(npz)
            x_t = torch.from_numpy(load_rd_maps_db(npz)).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad(): logits = model(x_t)
            cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
            tgt_ach = cy_ach if target == "cy" else ve_ach
            tgt_dets, opp_dets = (cy_dets, ve_dets) if target == "cy" else (ve_dets, cy_dets)
            if _hit(tgt_dets, tgt_ach, int(row[tgt_d_col]), int(row[tgt_r_col])): hits += 1
            if len(opp_dets) > 0: fp += 1
        return hits, fp, len(df)

    cy_h, cy_fp, cy_n = _eval_class(cy_df, "cy", None, "cyclist_true_d_idx", "cyclist_true_r_idx", model, thr)
    ve_h, ve_fp, ve_n = _eval_class(ve_df, "ve", None, "vehicle_true_d_idx",  "vehicle_true_r_idx",  model, thr)
    return cy_h, cy_fp, cy_n, ve_h, ve_fp, ve_n


# ===== CFAR 評価: holdout2obj (PR 用, tar_thresh スイープ) =====
def build_cfar_cache(df):
    """CFAR マスクと真値をキャッシュ（tar_thresh ループ高速化）"""
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


def eval_cfar_holdout2obj(cache, tar_thresh):
    """
    tar_thresh でフィルタした CFAR マスクから cy/ve hit 数・総検出セル数を集計。
    CFAR は class-agnostic なので aggregate PR のみ計算可能。
    cy/ve の hit 数は "どちらの真値にも対応するセルがあるか" で判定。
    """
    cy_hits = ve_hits = 0
    total_dets = 0
    N = len(cache)
    for c in cache:
        m       = c["mask"] & (c["rd_db"] > tar_thresh)  # (A, H, W) bool
        locs    = np.argwhere(m)          # (K, 3): [a, d, r]
        total_dets += len(locs)
        if len(locs) == 0:
            continue
        al, dl, rl = locs[:, 0], locs[:, 1], locs[:, 2]
        if c["cy_d"] is not None:
            cy_hits += int(np.any(
                (np.abs(al - c["cy_a"]) <= A_TOL) &
                (np.abs(dl - c["cy_d"]) <= D_TOL) &
                (np.abs(rl - c["cy_r"]) <= R_TOL)
            ))
        if c["ve_d"] is not None:
            ve_hits += int(np.any(
                (np.abs(al - c["ve_a"]) <= A_TOL) &
                (np.abs(dl - c["ve_d"]) <= D_TOL) &
                (np.abs(rl - c["ve_r"]) <= R_TOL)
            ))
    return cy_hits, ve_hits, total_dets, N


# ===== AP 計算 (11点補間 / trapezoidal) =====
def compute_ap(precisions, recalls):
    """PR 曲線から AP を台形積分で計算（単調化後）"""
    if len(precisions) == 0:
        return 0.0
    # recall 昇順に並べ替え
    order = np.argsort(recalls)
    r = np.array(recalls)[order]
    p = np.array(precisions)[order]
    # precision を右から単調増加（エンベロープ）
    p_env = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.trapz(p_env, r))


# ===== データ準備 =====
single_df = pd.read_csv(SINGLE_META_CSV)
single_df = single_df[single_df["valid_all"]==1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
single_holdout = single_df.iloc[340:].reset_index(drop=True)

mixed_df = pd.read_csv(MIXED_META_CSV)
mixed_df = mixed_df[mixed_df["valid_all"]==1].reset_index(drop=True)
mixed_df = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
holdout2obj = mixed_df.iloc[200:].reset_index(drop=True)  # 100件

cy_only = single_holdout[
    single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"]) &
    ~single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)
ve_only = single_holdout[
    single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"]) &
    ~single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)

print(f"holdout2obj={len(holdout2obj)}, cy_only={len(cy_only)}, ve_only={len(ve_only)}", flush=True)


# ===== CFAR キャッシュ構築 =====
print("\n[CFAR] CFARマスクのキャッシュ構築中...", flush=True)
cfar_cache = build_cfar_cache(holdout2obj)
print(f"[CFAR] キャッシュ完了 ({len(cfar_cache)} 件)", flush=True)


# ===== CFAR 評価ループ =====
cfar_points = []
for tar_thresh in TAR_THRESH_LIST:
    cy_h, ve_h, tot_dets, N = eval_cfar_holdout2obj(cfar_cache, tar_thresh)
    # aggregate PR (class-agnostic)
    total_gt = 2 * N  # cy + ve
    tp = cy_h + ve_h
    p_agg = tp / tot_dets if tot_dets > 0 else 0.0
    r_agg = tp / total_gt if total_gt > 0 else 0.0
    # per-class recall (precision は class-agnostic なので aggregate のみ)
    r_cy = cy_h / N
    r_ve = ve_h / N
    fp_per_sample = (tot_dets - tp) / N
    cfar_points.append({
        "tar_thresh": tar_thresh,
        "cy_hit": cy_h, "ve_hit": ve_h,
        "total_dets": tot_dets,
        "p_agg": p_agg, "r_agg": r_agg,
        "r_cy": r_cy, "r_ve": r_ve,
        "fp_per_sample": fp_per_sample,
    })
    print(f"  CFAR tar_thresh={tar_thresh:>4}dB  "
          f"cy={cy_h}/{N}  ve={ve_h}/{N}  "
          f"dets={tot_dets}  P={p_agg:.3f}  R={r_agg:.3f}  FP/sample={fp_per_sample:.2f}", flush=True)


# ===== NN 評価ループ =====
all_nn_results = {}

for name in TARGET_RUNS:
    pt_path = os.path.join(V3_RESULTS_DIR, f"{name}.pt")
    if not os.path.exists(pt_path):
        print(f"[{name}] モデルファイルが見つかりません: {pt_path}", flush=True)
        continue

    print(f"\n{'='*50}", flush=True)
    print(f"[{name}] PR/ROC 評価開始", flush=True)

    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()

    pr_points   = []  # holdout2obj での PR 動作点
    roc_points  = []  # single-target での TPR/FPR 動作点

    for thr in NN_THRESHOLDS:
        # PR 評価 (holdout2obj)
        cy_h, ve_h, n_cy, n_ve, N = eval_nn_holdout2obj(model, holdout2obj, float(thr))
        p_cy = cy_h / n_cy if n_cy > 0 else 1.0
        r_cy = cy_h / N
        p_ve = ve_h / n_ve if n_ve > 0 else 1.0
        r_ve = ve_h / N
        # aggregate
        tp   = cy_h + ve_h
        dets = n_cy + n_ve
        p_agg = tp / dets if dets > 0 else 1.0
        r_agg = tp / (2 * N)

        pr_points.append({
            "thr": float(thr),
            "cy_hit": cy_h, "ve_hit": ve_h,
            "n_cy_dets": n_cy, "n_ve_dets": n_ve,
            "p_cy": p_cy, "r_cy": r_cy,
            "p_ve": p_ve, "r_ve": r_ve,
            "p_agg": p_agg, "r_agg": r_agg,
        })

        # ROC 評価 (single-target)
        cy_hit_s, cy_fp_s, cy_n_s, ve_hit_s, ve_fp_s, ve_n_s = \
            eval_nn_single(model, cy_only, ve_only, float(thr))
        roc_points.append({
            "thr": float(thr),
            "tpr_cy": cy_hit_s / cy_n_s, "fpr_cy": cy_fp_s / cy_n_s,
            "tpr_ve": ve_hit_s / ve_n_s, "fpr_ve": ve_fp_s / ve_n_s,
        })

        print(f"  thr={thr:.3f}  "
              f"P_cy={p_cy:.3f} R_cy={r_cy:.3f}  "
              f"P_ve={p_ve:.3f} R_ve={r_ve:.3f}  "
              f"TPR_cy={cy_hit_s/cy_n_s:.3f} FPR_cy={cy_fp_s/cy_n_s:.3f}  "
              f"TPR_ve={ve_hit_s/ve_n_s:.3f} FPR_ve={ve_fp_s/ve_n_s:.3f}", flush=True)

    # AP 計算
    ap_cy  = compute_ap([p["p_cy"] for p in pr_points], [p["r_cy"] for p in pr_points])
    ap_ve  = compute_ap([p["p_ve"] for p in pr_points], [p["r_ve"] for p in pr_points])
    map_   = (ap_cy + ap_ve) / 2
    print(f"  AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  mAP={map_:.4f}", flush=True)

    all_nn_results[name] = {
        "pr_points": pr_points,
        "roc_points": roc_points,
        "ap_cy": ap_cy,
        "ap_ve": ap_ve,
        "map": map_,
    }

# ===== 結果保存 =====
results = {
    "nn": all_nn_results,
    "cfar": {
        "config": {"pfa": CFAR_PFA, "n_train": list(CFAR_N_TRAIN), "n_guard": list(CFAR_N_GUARD)},
        "points": cfar_points,
    },
    "meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "d_tol": D_TOL, "r_tol": R_TOL, "a_tol": A_TOL,
        "nn_thresholds": NN_THRESHOLDS.tolist(),
        "tar_thresh_list": TAR_THRESH_LIST,
    }
}
json_path = os.path.join(OUTPUT_DIR, "pr_roc_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nJSON保存: {json_path}", flush=True)


# ===== プロット =====
COLORS = {
    "D_tk3_fp0.001_mixed":  ("tab:blue",   "topK (K=3, mixed)"),
    "D_dw_fp0.001_single":  ("tab:orange", "daware (single)"),
    "D_dw_fp0.001_mixed":   ("tab:green",  "daware (mixed)"),
    "cfar":                 ("tab:red",    "3D CFAR"),
}

# ---- 図1: PR 曲線 cy/ve / aggregate ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PR Curves (holdout 2-object, 100 samples)")

for ax, key, title in zip(
    axes,
    [("p_cy", "r_cy"), ("p_ve", "r_ve"), ("p_agg", "r_agg")],
    ["Cyclist PR", "Vehicle PR", "Aggregate PR"],
):
    pk, rk = key
    ax.set_title(title)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    for name in TARGET_RUNS:
        if name not in all_nn_results: continue
        col, lbl = COLORS[name]
        pts = all_nn_results[name]["pr_points"]
        ps  = [p[pk] for p in pts]
        rs  = [p[rk] for p in pts]
        ap  = compute_ap(ps, rs)
        ax.plot(rs, ps, "o-", color=col, label=f"{lbl} (AP={ap:.3f})", markersize=3)

    # CFAR は aggregate のみ
    if pk == "p_agg":
        col, lbl = COLORS["cfar"]
        ps = [p["p_agg"] for p in cfar_points]
        rs = [p["r_agg"] for p in cfar_points]
        ap = compute_ap(ps, rs)
        axes[2].plot(rs, ps, "s--", color=col, label=f"{lbl} (AP={ap:.3f})", markersize=5)
    elif pk == "p_cy":
        col, lbl = COLORS["cfar"]
        # cy recall のみ (precision はaggregate しか定義できない)
        rs = [p["r_cy"] for p in cfar_points]
        ps = [p["p_agg"] for p in cfar_points]  # 近似: aggregate precision を使用
        axes[0].plot(rs, ps, "s--", color=col, label=f"{lbl} (agg P, approx)", markersize=5)
    elif pk == "p_ve":
        col, lbl = COLORS["cfar"]
        rs = [p["r_ve"] for p in cfar_points]
        ps = [p["p_agg"] for p in cfar_points]
        axes[1].plot(rs, ps, "s--", color=col, label=f"{lbl} (agg P, approx)", markersize=5)

    ax.legend(fontsize=7, loc="lower left")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("PR 曲線保存完了", flush=True)


# ---- 図2: ROC 的曲線 (TPR vs cross-class FPR) cy/ve ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("ROC-like Curves (single-target test)\nTPR = hit rate, FPR = cross-class detection rate")

for ax, (tpr_k, fpr_k), title in zip(
    axes,
    [("tpr_cy", "fpr_cy"), ("tpr_ve", "fpr_ve")],
    ["Cyclist (cy_only set)", "Vehicle (ve_only set)"],
):
    ax.set_title(title)
    ax.set_xlabel("FPR (cross-class)"); ax.set_ylabel("TPR")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.plot([0,1],[0,1],"k--",alpha=0.3,lw=1)
    ax.grid(True, alpha=0.3)

    for name in TARGET_RUNS:
        if name not in all_nn_results: continue
        col, lbl = COLORS[name]
        pts  = all_nn_results[name]["roc_points"]
        tprs = [p[tpr_k] for p in pts]
        fprs = [p[fpr_k] for p in pts]
        auc  = float(np.trapz(sorted(tprs), sorted(fprs)))
        ax.plot(fprs, tprs, "o-", color=col, label=f"{lbl} (AUC≈{auc:.3f})", markersize=3)

    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("ROC 曲線保存完了", flush=True)


# ---- 図3: AP/mAP サマリー ----
names  = [n for n in TARGET_RUNS if n in all_nn_results]
ap_cys = [all_nn_results[n]["ap_cy"]  for n in names]
ap_ves = [all_nn_results[n]["ap_ve"]  for n in names]
maps   = [all_nn_results[n]["map"]     for n in names]
labels = [COLORS[n][1] for n in names]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - 0.25, ap_cys, 0.25, label="AP_cy", color="steelblue")
ax.bar(x,        ap_ves, 0.25, label="AP_ve", color="darkorange")
ax.bar(x + 0.25, maps,   0.25, label="mAP",   color="green")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("AP"); ax.set_ylim(0, 1.05)
ax.set_title("AP / mAP Summary (NN only, holdout 2-object)")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ap_summary.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("AP サマリー保存完了", flush=True)


# ---- 図4: CFAR Recall vs FP/sample ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("3D CFAR: Recall vs FP per sample (holdout 2-object)\npfa=1e-3, tar_thresh sweep")
ax.set_xlabel("FP detections / sample"); ax.set_ylabel("Recall")
ax.grid(True, alpha=0.3)
fps_cfar = [p["fp_per_sample"] for p in cfar_points]
r_cy_cfar = [p["r_cy"] for p in cfar_points]
r_ve_cfar = [p["r_ve"] for p in cfar_points]
r_agg_cfar = [p["r_agg"] for p in cfar_points]
ax.plot(fps_cfar, r_cy_cfar,  "s-", color="tab:blue",  label="Recall_cy",  markersize=6)
ax.plot(fps_cfar, r_ve_cfar,  "^-", color="tab:orange", label="Recall_ve", markersize=6)
ax.plot(fps_cfar, r_agg_cfar, "o-", color="tab:red",   label="Recall_agg", markersize=6)
for p in cfar_points:
    ax.annotate(f"{p['tar_thresh']}dB",
                (p["fp_per_sample"], p["r_agg"]),
                fontsize=6, ha="left", xytext=(3, 2), textcoords="offset points")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cfar_recall_vs_fp.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("CFAR recall vs FP 保存完了", flush=True)


# ===== AP/mAP テキストサマリー =====
print("\n" + "="*60, flush=True)
print("AP / mAP サマリー")
print(f"{'run':30s}  {'AP_cy':>6}  {'AP_ve':>6}  {'mAP':>6}")
print("-" * 60)
for name in TARGET_RUNS:
    if name not in all_nn_results: continue
    r = all_nn_results[name]
    print(f"{name:30s}  {r['ap_cy']:>6.4f}  {r['ap_ve']:>6.4f}  {r['map']:>6.4f}", flush=True)
print("="*60, flush=True)


# ===== Markdown レポート生成 =====
now = datetime.now().strftime("%Y-%m-%d %H:%M")
lines = [
    "# PR / ROC 評価レポート",
    "",
    f"生成日時: {now}",
    "",
    "## 評価設定",
    "",
    f"- データセット: holdout 2-object ({len(holdout2obj)} 件) / single-target cy ({len(cy_only)} 件) / ve ({len(ve_only)} 件)",
    f"- 許容幅: D_TOL={D_TOL}, R_TOL={R_TOL}, A_TOL={A_TOL}",
    f"- NN 閾値スイープ: {len(NN_THRESHOLDS)} 点 ({NN_THRESHOLDS[0]:.2f} 〜 {NN_THRESHOLDS[-1]:.2f})",
    "",
    "### 3D CFAR 設定",
    "",
    f"- pfa: {CFAR_PFA}  (固定)",
    f"- n_train: {CFAR_N_TRAIN}  (角度, ドップラー, レンジ)",
    f"- n_guard: {CFAR_N_GUARD}",
    f"- tar_thresh スイープ: {TAR_THRESH_LIST[0]} 〜 {TAR_THRESH_LIST[-1]} dB",
    "",
    "## AP / mAP サマリー (NN, holdout 2-object)",
    "",
    "| run | type | AP_cy | AP_ve | mAP |",
    "|---|---|---|---|---|",
]
for name in TARGET_RUNS:
    if name not in all_nn_results:
        continue
    r   = all_nn_results[name]
    cfg_path = os.path.join(V3_RESULTS_DIR, f"{name}.json")
    typ = "-"
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            typ = json.load(f).get("config", {}).get("type", "-")
    lines.append(f"| {name} | {typ} | {r['ap_cy']:.4f} | {r['ap_ve']:.4f} | {r['map']:.4f} |")

lines += [
    "",
    "## PR 動作点詳細 (holdout 2-object, threshold=0.5)",
    "",
    "| run | P_cy | R_cy | P_ve | R_ve | P_agg | R_agg |",
    "|---|---|---|---|---|---|---|",
]
for name in TARGET_RUNS:
    if name not in all_nn_results:
        continue
    pts = all_nn_results[name]["pr_points"]
    # threshold=0.5 に最も近い点
    pt  = min(pts, key=lambda p: abs(p["thr"] - 0.5))
    lines.append(
        f"| {name} | {pt['p_cy']:.3f} | {pt['r_cy']:.3f}"
        f" | {pt['p_ve']:.3f} | {pt['r_ve']:.3f}"
        f" | {pt['p_agg']:.3f} | {pt['r_agg']:.3f} |"
    )

lines += [
    "",
    "## ROC 動作点詳細 (single-target, threshold=0.5)",
    "",
    "| run | TPR_cy | FPR_cy | TPR_ve | FPR_ve |",
    "|---|---|---|---|---|",
]
for name in TARGET_RUNS:
    if name not in all_nn_results:
        continue
    pts = all_nn_results[name]["roc_points"]
    pt  = min(pts, key=lambda p: abs(p["thr"] - 0.5))
    lines.append(
        f"| {name} | {pt['tpr_cy']:.3f} | {pt['fpr_cy']:.3f}"
        f" | {pt['tpr_ve']:.3f} | {pt['fpr_ve']:.3f} |"
    )

lines += [
    "",
    "## 3D CFAR 動作点一覧 (holdout 2-object)",
    "",
    f"pfa={CFAR_PFA}, n_train={CFAR_N_TRAIN}, n_guard={CFAR_N_GUARD}",
    "",
    "| tar_thresh (dB) | cy_hit | ve_hit | total_dets | P_agg | R_agg | R_cy | R_ve | FP/sample |",
    "|---|---|---|---|---|---|---|---|---|",
]
for p in cfar_points:
    N = len(cfar_cache)
    lines.append(
        f"| {p['tar_thresh']} | {p['cy_hit']}/{N} | {p['ve_hit']}/{N}"
        f" | {p['total_dets']} | {p['p_agg']:.3f} | {p['r_agg']:.3f}"
        f" | {p['r_cy']:.3f} | {p['r_ve']:.3f} | {p['fp_per_sample']:.2f} |"
    )

lines += [
    "",
    "## 図",
    "",
    "| ファイル | 内容 |",
    "|---|---|",
    "| pr_curves.png | cy / ve / aggregate の PR 曲線 |",
    "| roc_curves.png | cross-class FPR vs TPR (ROC的曲線) |",
    "| ap_summary.png | AP / mAP 棒グラフ |",
    "| cfar_recall_vs_fp.png | CFAR recall vs FP/sample |",
    "",
    "## 注記",
    "",
    "- NN の PR 曲線は NMS 閾値スイープ込み（閾値を下げると NMS 挙動も変化）。",
    "  良好な動作点が見つかった場合は、その閾値を学習時に固定して再評価することを推奨。",
    "- CFAR の Precision は class-agnostic (cy/ve 区別なし) な aggregate 値。",
    "  cy/ve の Recall は個別に計算可能だが Precision は計算不可のため、",
    "  cy/ve PR 図への CFAR プロットは aggregate Precision を近似として使用している。",
    "- ROC 的曲線の FPR は「対象クラス不在サンプルにおける他クラス出現率」であり、",
    "  標準的な二値分類 ROC の FPR とは定義が異なる。",
]

report_path = os.path.join(OUTPUT_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"レポート保存: {report_path}", flush=True)
print(f"\n完了。結果は {OUTPUT_DIR}/ に保存されました。", flush=True)
