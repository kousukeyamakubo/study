"""
閾値スイープ + FP位置分析スクリプト
- フェーズ1: 閾値を変えながら cy-only / ve-only / test_df(2物体) で評価
- フェーズ2: ve-only で cyclist FP が出た件について r_idx ずれ分布を分析
"""

import os, sys, collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 定数 ──────────────────────────────────────────────────────────────
N_FIXED      = 10
RANDOM_SEED  = 42
D_TOL        = 2
R_TOL        = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_META_CSV = "../../learn_dataset_single_object/metadata.csv"
FIXED_META_CSV = "../../learn_dataset_fixed_angle/metadata.csv"
MODEL_PATH_SEG = "../../best_detector_softmax_heatmap.pt"

RESULT_FILE    = "./threshold_analysis_result/threshold_analysis_result.txt"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ─── モデル定義 ────────────────────────────────────────────────────────
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
            ConvBlock3D(1, fixed_channels, dropout),
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
        # x: (B, N_FIXED, H, W) → unsqueeze → (B, 1, N_FIXED, H, W)
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
        return self.seg_head(feat)


# ─── データ読み込み ────────────────────────────────────────────────────
def load_all_rd_maps(npz_path):
    data    = np.load(npz_path)
    rd_maps = data["rd_maps"]           # (N, H, W) complex
    result  = []
    for i in range(rd_maps.shape[0]):
        rd_mag = np.abs(rd_maps[i]).astype(np.float32)
        rd_db  = 20.0 * np.log10(np.maximum(rd_mag, 1e-12))
        result.append(rd_db)
    return np.stack(result, axis=0)     # (N, H, W)


def decode_softmax_detections(logits, threshold=0.5):
    """logits: (1, 3, N, H, W) → (cy_det, ve_det) 各=(ch, d, r) or (-1,-1,-1)"""
    probs   = F.softmax(logits, dim=1)
    cy_prob = probs[0, 1]   # (N, H, W)
    ve_prob = probs[0, 2]   # (N, H, W)

    def _detect(prob_map, thr):
        if prob_map.max().item() < thr:
            return (-1, -1, -1)
        N, H, W  = prob_map.shape
        flat_idx = torch.argmax(prob_map).item()
        ch  = flat_idx // (H * W)
        rem = flat_idx %  (H * W)
        d   = rem // W
        r   = rem %  W
        return (int(ch), int(d), int(r))

    return _detect(cy_prob, threshold), _detect(ve_prob, threshold)


# ─── モデルロード ──────────────────────────────────────────────────────
def load_model():
    model = RadarUNet3DSoftmax(n_angles=N_FIXED)
    state = torch.load(MODEL_PATH_SEG, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ─── 1サンプル推論（npzファイルのキャッシュ付き） ─────────────────────
_npz_cache: dict = {}

def infer_sample(model, npz_path, threshold):
    """指定 npz を推論して (cy_det, ve_det) を返す"""
    if npz_path not in _npz_cache:
        _npz_cache[npz_path] = load_all_rd_maps(npz_path)
    rd_db = _npz_cache[npz_path]          # (N, H, W)
    x = torch.tensor(rd_db, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, N, H, W)
    with torch.no_grad():
        logits = model(x)                  # (1, 3, N, H, W)
    return decode_softmax_detections(logits, threshold)


# ─── データセット構築 ──────────────────────────────────────────────────
def build_datasets():
    # single_object holdout（列名は cyclist_true_d_idx 等）
    single_df = pd.read_csv(TRAIN_META_CSV)
    single_df = single_df[single_df["valid_all"] == 1].reset_index(drop=True)
    single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    holdout_df = single_df.iloc[340:].reset_index(drop=True)

    cy_only_df = holdout_df[
        holdout_df["valid_cyclist"].astype(str).str.strip().isin(["1", "True"]) &
        ~holdout_df["valid_vehicle"].astype(str).str.strip().isin(["1", "True"])
    ].reset_index(drop=True)

    ve_only_df = holdout_df[
        holdout_df["valid_vehicle"].astype(str).str.strip().isin(["1", "True"]) &
        ~holdout_df["valid_cyclist"].astype(str).str.strip().isin(["1", "True"])
    ].reset_index(drop=True)

    # fixed_angle test_df（2物体）
    two_df = pd.read_csv(FIXED_META_CSV)
    two_df = two_df[two_df["valid_all"] == 1].reset_index(drop=True)
    two_df = two_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = two_df.iloc[200:].reset_index(drop=True)

    return cy_only_df, ve_only_df, test_df


# ─── cy-only 評価（正解: cyclist あり、vehicle なし） ──────────────────
def eval_cy_only(model, df, threshold):
    """cy_hit率と vehicle FP率を返す"""
    cy_hit = 0
    ve_fp  = 0
    n = len(df)
    for _, row in df.iterrows():
        npz = row["file"]
        cy_det, ve_det = infer_sample(model, npz, threshold)
        true_d = int(row["cyclist_true_d_idx"])
        true_r = int(row["cyclist_true_r_idx"])
        # cyclist が正しい位置に検出されたか
        if cy_det != (-1, -1, -1):
            if abs(cy_det[1] - true_d) <= D_TOL and abs(cy_det[2] - true_r) <= R_TOL:
                cy_hit += 1
        # vehicle は存在しないのに検出されたら FP
        if ve_det != (-1, -1, -1):
            ve_fp += 1
    return cy_hit / n, ve_fp / n


# ─── ve-only 評価（正解: vehicle あり、cyclist なし） ──────────────────
def eval_ve_only(model, df, threshold):
    """ve_hit率と cyclist FP率を返す"""
    ve_hit = 0
    cy_fp  = 0
    n = len(df)
    for _, row in df.iterrows():
        npz = row["file"]
        cy_det, ve_det = infer_sample(model, npz, threshold)
        true_d = int(row["vehicle_true_d_idx"])
        true_r = int(row["vehicle_true_r_idx"])
        # vehicle が正しい位置に検出されたか
        if ve_det != (-1, -1, -1):
            if abs(ve_det[1] - true_d) <= D_TOL and abs(ve_det[2] - true_r) <= R_TOL:
                ve_hit += 1
        # cyclist は存在しないのに検出されたら FP
        if cy_det != (-1, -1, -1):
            cy_fp += 1
    return ve_hit / n, cy_fp / n


# ─── 2物体 test_df 評価 ────────────────────────────────────────────────
def eval_two_obj(model, df, threshold):
    """cy_hit率と ve_hit率を返す"""
    cy_hit = 0
    ve_hit = 0
    n = len(df)
    for _, row in df.iterrows():
        npz = row["file"]
        cy_det, ve_det = infer_sample(model, npz, threshold)
        true_cy_d = int(row["cyclist_true_d_idx"])
        true_cy_r = int(row["cyclist_true_r_idx"])
        true_ve_d = int(row["vehicle_true_d_idx"])
        true_ve_r = int(row["vehicle_true_r_idx"])
        if cy_det != (-1, -1, -1):
            if abs(cy_det[1] - true_cy_d) <= D_TOL and abs(cy_det[2] - true_cy_r) <= R_TOL:
                cy_hit += 1
        if ve_det != (-1, -1, -1):
            if abs(ve_det[1] - true_ve_d) <= D_TOL and abs(ve_det[2] - true_ve_r) <= R_TOL:
                ve_hit += 1
    return cy_hit / n, ve_hit / n


# ─── フェーズ2: ve-only FP 位置分析（threshold=0.5固定） ─────────────
def analyze_fp_position(model, ve_only_df, threshold=0.5):
    """cyclist が誤検出された件の r_idx 差リストを返す"""
    fp_r_diffs = []
    for _, row in ve_only_df.iterrows():
        npz = row["file"]
        cy_det, _ = infer_sample(model, npz, threshold)
        if cy_det == (-1, -1, -1):
            continue   # FPなし
        fp_r   = cy_det[2]
        true_r = int(row["vehicle_true_r_idx"])
        fp_r_diffs.append(fp_r - true_r)
    return fp_r_diffs


# ─── メイン ────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] DEVICE={DEVICE}", flush=True)
    print("[INFO] モデルをロード中...", flush=True)
    model = load_model()
    print("[INFO] モデルロード完了", flush=True)

    print("[INFO] データセットを構築中...", flush=True)
    cy_only_df, ve_only_df, test_df = build_datasets()
    print(f"[INFO] cy_only={len(cy_only_df)}件, ve_only={len(ve_only_df)}件, test_df={len(test_df)}件", flush=True)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:

        # ── メタ情報 ──────────────────────────────────────────────────
        print("=" * 65, file=f)
        print("閾値スイープ + FP位置分析 結果", file=f)
        print("=" * 65, file=f)
        print(f"DEVICE        : {DEVICE}", file=f)
        print(f"MODEL_PATH    : {MODEL_PATH_SEG}", file=f)
        print(f"cy_only       : {len(cy_only_df)} 件", file=f)
        print(f"ve_only       : {len(ve_only_df)} 件", file=f)
        print(f"test_df(2物体): {len(test_df)} 件", file=f)
        print(f"D_TOL={D_TOL}, R_TOL={R_TOL}", file=f)
        print("", file=f)

        # ── フェーズ1: 閾値スイープ ────────────────────────────────
        print("=" * 65, file=f)
        print("フェーズ1: 閾値スイープ", file=f)
        print("=" * 65, file=f)
        header = (
            f"{'thr':>5} | "
            f"{'cy_only':>14} | "
            f"{'cy_only':>13} | "
            f"{'ve_only':>14} | "
            f"{'ve_only':>13} | "
            f"{'2obj':>11} | "
            f"{'2obj':>11}"
        )
        sub_hdr = (
            f"{'':>5} | "
            f"{'cy_hit':>14} | "
            f"{'ve_FP':>13} | "
            f"{'ve_hit':>14} | "
            f"{'cy_FP':>13} | "
            f"{'cy_hit':>11} | "
            f"{'ve_hit':>11}"
        )
        sep = "-" * len(sub_hdr)
        print(header, file=f)
        print(sub_hdr, file=f)
        print(sep, file=f)

        for thr in THRESHOLDS:
            print(f"[INFO] 閾値={thr} 評価中...", flush=True)
            cy_hit_cy, ve_fp_cy = eval_cy_only(model, cy_only_df, thr)
            ve_hit_ve, cy_fp_ve = eval_ve_only(model, ve_only_df, thr)
            cy_hit_2,  ve_hit_2 = eval_two_obj(model, test_df, thr)

            row_str = (
                f"{thr:>5.1f} | "
                f"{cy_hit_cy:>14.3f} | "
                f"{ve_fp_cy:>13.3f} | "
                f"{ve_hit_ve:>14.3f} | "
                f"{cy_fp_ve:>13.3f} | "
                f"{cy_hit_2:>11.3f} | "
                f"{ve_hit_2:>11.3f}"
            )
            print(row_str, file=f)
            print(f"  thr={thr}: cy_only cy_hit={cy_hit_cy:.3f} ve_FP={ve_fp_cy:.3f} | "
                  f"ve_only ve_hit={ve_hit_ve:.3f} cy_FP={cy_fp_ve:.3f} | "
                  f"2obj cy={cy_hit_2:.3f} ve={ve_hit_2:.3f}", flush=True)

        print("", file=f)

        # ── フェーズ2: FP位置分析 ─────────────────────────────────
        print("=" * 65, file=f)
        print("フェーズ2: ve-only FP 位置分析 (threshold=0.5)", file=f)
        print("=" * 65, file=f)

        print("[INFO] FP位置分析中 (threshold=0.5)...", flush=True)
        fp_r_diffs = analyze_fp_position(model, ve_only_df, threshold=0.5)

        n_fp = len(fp_r_diffs)
        print(f"cyclist FP 件数 (threshold=0.5): {n_fp} / {len(ve_only_df)}", file=f)

        if n_fp > 0:
            n_zero    = sum(1 for d in fp_r_diffs if d == 0)
            n_nonzero = n_fp - n_zero
            print(f"  fp_r_diff=0  (vehicle と同じ距離セルに幽霊): {n_zero:4d} 件 ({100*n_zero/n_fp:.1f}%)", file=f)
            print(f"  fp_r_diff≠0  (別の距離セルに幽霊): {n_nonzero:4d} 件 ({100*n_nonzero/n_fp:.1f}%)", file=f)
            print("", file=f)

            counter = collections.Counter(fp_r_diffs)
            print("fp_r_diff 分布 (FP r_idx − true vehicle r_idx):", file=f)
            print(f"  {'diff':>6} : {'件数':>6} | バー", file=f)
            print(f"  {'------':>6}   {'------':>6}", file=f)
            for diff_val in sorted(counter.keys()):
                cnt = counter[diff_val]
                bar = "#" * cnt
                print(f"  {diff_val:>6} : {cnt:>6} | {bar}", file=f)

            print("", file=f)
            print(f"  平均差    : {np.mean(fp_r_diffs):.2f}", file=f)
            print(f"  標準偏差  : {np.std(fp_r_diffs):.2f}", file=f)
            print(f"  最小 / 最大: {min(fp_r_diffs)} / {max(fp_r_diffs)}", file=f)
        else:
            print("  FPなし（cyclist 誤検出は0件）", file=f)

        print("", file=f)
        print("=" * 65, file=f)
        print("分析完了", file=f)
        print("=" * 65, file=f)

    print(f"[INFO] 結果を {RESULT_FILE} に保存しました", flush=True)


if __name__ == "__main__":
    main()