"""
実験A: softmax-baseline
現状アーキテクチャ（NMS複数検出対応版）で学習・評価を行い結果を記録する
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 定数 =====
TRAIN_DATASET_DIR  = "../../learn_dataset_single_object"
TEST_DATASET_DIR   = "../../learn_dataset_fixed_angle"
SCENARIO_META_CSV  = "../../learn_dataset_scenario_test/metadata.csv"
TRAIN_META_CSV     = os.path.join(TRAIN_DATASET_DIR, "metadata.csv")
TEST_META_CSV      = os.path.join(TEST_DATASET_DIR,  "metadata.csv")
MODEL_PATH_OUT     = "../../best_detector_baseline.pt"
RESULT_PATH        = "./train_baseline_results/experiment_results_baseline.txt"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4
D_TOL        = 2
R_TOL        = 3
RANDOM_SEED  = 42

SEG_BG_WEIGHT = 1.0
SEG_CY_WEIGHT = 500.0
SEG_VE_WEIGHT = 500.0

print(f"DEVICE: {DEVICE}", flush=True)

# ===== モデル =====
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

# ===== データセット =====
def load_all_rd_maps(npz_path):
    data = np.load(npz_path)
    rd_maps = data["rd_maps"]
    result = []
    for i in range(rd_maps.shape[0]):
        rd_mag = np.abs(rd_maps[i]).astype(np.float32)
        rd_db  = 20.0 * np.log10(np.maximum(rd_mag, 1e-12))
        result.append(rd_db)
    return np.stack(result, axis=0)

class MultiTargetDatasetSeg(Dataset):
    def __init__(self, meta_df):
        self.samples = []
        print(f"データ読み込み中... ({len(meta_df)} サンプル)", flush=True)
        for idx in range(len(meta_df)):
            row = meta_df.iloc[idx]
            npz_path = row["file"]
            if not os.path.isabs(npz_path):
                npz_path = os.path.normpath(os.path.join(".", npz_path))
            data    = np.load(npz_path)
            rd_maps = data["rd_maps"]
            x = np.stack(
                [20.0 * np.log10(np.maximum(np.abs(rd_maps[i]).astype(np.float32), 1e-12))
                 for i in range(rd_maps.shape[0])],
                axis=0
            )  # (N_FIXED, H, W)
            H, W = x.shape[1], x.shape[2]
            fixed_angles = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES
            valid_cy = 1 if str(row["valid_cyclist"]).strip() in ("1", "True") else 0
            valid_ve = 1 if str(row["valid_vehicle"]).strip() in ("1", "True") else 0
            cy_angle = float(data["cyclist_true_angle_deg"]) if (valid_cy and "cyclist_true_angle_deg" in data) else 0.0
            ve_angle = float(data["vehicle_true_angle_deg"]) if (valid_ve and "vehicle_true_angle_deg" in data) else 0.0
            y_seg = np.zeros((N_FIXED, H, W), dtype=np.int64)
            if valid_cy:
                cy_ch = int(np.argmin(np.abs(fixed_angles - cy_angle)))
                y_seg[cy_ch, int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])] = 1
            if valid_ve:
                ve_ch = int(np.argmin(np.abs(fixed_angles - ve_angle)))
                y_seg[ve_ch, int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])] = 2
            self.samples.append({
                "x":         torch.from_numpy(x).float(),
                "y_seg":     torch.from_numpy(y_seg).long(),
                "cy_true_d": torch.tensor(int(row["cyclist_true_d_idx"]), dtype=torch.long),
                "cy_true_r": torch.tensor(int(row["cyclist_true_r_idx"]), dtype=torch.long),
                "ve_true_d": torch.tensor(int(row["vehicle_true_d_idx"]), dtype=torch.long),
                "ve_true_r": torch.tensor(int(row["vehicle_true_r_idx"]), dtype=torch.long),
                "valid_cy":  torch.tensor(valid_cy, dtype=torch.long),
                "valid_ve":  torch.tensor(valid_ve, dtype=torch.long),
            })
        print("読み込み完了", flush=True)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ===== 損失・デコード =====
def compute_seg_loss(logits, y_seg):
    # logits: (B, 3, N_FIXED, H, W)
    # y_seg:  (B, N_FIXED, H, W) long
    weight = torch.tensor([SEG_BG_WEIGHT, SEG_CY_WEIGHT, SEG_VE_WEIGHT],
                          dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits, y_seg, weight=weight)

def decode_softmax_detections(logits, threshold=0.5):
    """NMSで閾値超えピークを全列挙。戻り値は [(ch,d,r),...] のリスト×2。"""
    probs   = F.softmax(logits, dim=1)  # (1, 3, N_FIXED, H, W)
    cy_prob = probs[0, 1]               # (N_FIXED, H, W)
    ve_prob = probs[0, 2]

    def _detect_all(prob_map, thr):
        # ch±1, d±3, r±1 の範囲を抑制して繰り返す
        prob = prob_map.clone()
        N, H, W = prob.shape
        detections = []
        while prob.max().item() >= thr:
            flat_idx = torch.argmax(prob).item()
            ch  = flat_idx // (H * W)
            rem = flat_idx %  (H * W)
            d   = rem // W
            r   = rem %  W
            detections.append((ch, d, r))
            prob[max(ch-1,0):min(ch+1,N-1)+1,
                 max(d-3, 0):min(d+3, H-1)+1,
                 max(r-1, 0):min(r+1, W-1)+1] = 0.0
        return detections

    return _detect_all(cy_prob, threshold), _detect_all(ve_prob, threshold)

# ===== 学習ループ =====
def run_epoch_seg(model, loader, optimizer=None, device='cpu', threshold=0.5):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    cy_hits, cy_total = 0, 0
    ve_hits, ve_total = 0, 0

    for batch in loader:
        x         = batch['x'].to(device)
        y_seg     = batch['y_seg'].to(device)
        cy_true_d = batch['cy_true_d'].cpu().numpy()
        cy_true_r = batch['cy_true_r'].cpu().numpy()
        ve_true_d = batch['ve_true_d'].cpu().numpy()
        ve_true_r = batch['ve_true_r'].cpu().numpy()
        valid_cy  = batch['valid_cy'].cpu().numpy()
        valid_ve  = batch['valid_ve'].cpu().numpy()

        if train_mode:
            optimizer.zero_grad()
        logits = model(x)
        loss   = compute_seg_loss(logits, y_seg)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        for i in range(x.size(0)):
            cy_dets, ve_dets = decode_softmax_detections(logits[i:i+1].detach().cpu(), threshold)
            if valid_cy[i]:
                cy_total += 1
                if any(abs(det[1]-cy_true_d[i]) <= D_TOL and abs(det[2]-cy_true_r[i]) <= R_TOL for det in cy_dets):
                    cy_hits += 1
            if valid_ve[i]:
                ve_total += 1
                if any(abs(det[1]-ve_true_d[i]) <= D_TOL and abs(det[2]-ve_true_r[i]) <= R_TOL for det in ve_dets):
                    ve_hits += 1

    return (total_loss / max(len(loader.dataset), 1),
            cy_hits / max(cy_total, 1),
            ve_hits / max(ve_total, 1))

# ===== データ分割 =====
single_df = pd.read_csv(TRAIN_META_CSV)
single_df = single_df[single_df["valid_all"] == 1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
train_df  = single_df.iloc[:280].reset_index(drop=True)
val_df    = single_df.iloc[280:340].reset_index(drop=True)

two_df   = pd.read_csv(TEST_META_CSV)
two_df   = two_df[two_df["valid_all"] == 1].reset_index(drop=True)
two_df   = two_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df  = two_df.iloc[200:].reset_index(drop=True)

print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", flush=True)

# ===== 学習 =====
train_ds = MultiTargetDatasetSeg(train_df)
val_ds   = MultiTargetDatasetSeg(val_df)
test_ds  = MultiTargetDatasetSeg(test_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

model     = RadarUNet3DSoftmax(n_angles=N_FIXED).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val  = float("inf")
history   = []

for epoch in range(EPOCHS):
    tr_loss, tr_cy, tr_ve = run_epoch_seg(model, train_loader, optimizer, DEVICE)
    va_loss, va_cy, va_ve = run_epoch_seg(model, val_loader,   None,      DEVICE)
    history.append({"epoch": epoch+1,
                    "train_loss": tr_loss, "train_cy": tr_cy, "train_ve": tr_ve,
                    "val_loss":   va_loss, "val_cy":   va_cy, "val_ve":   va_ve})
    if va_loss < best_val:
        best_val = va_loss
        torch.save(model.state_dict(), MODEL_PATH_OUT)
    print(f"epoch={epoch+1:02d}  train={tr_loss:.4f} cy={tr_cy:.3f} ve={tr_ve:.3f}"
          f"  |  val={va_loss:.4f} cy={va_cy:.3f} ve={va_ve:.3f}", flush=True)

print(f"best model saved to: {MODEL_PATH_OUT}")

# ===== 評価 =====
model.load_state_dict(torch.load(MODEL_PATH_OUT, map_location=DEVICE))
model.eval()

def evaluate_two_object(sub_df, model, label):
    rows = []
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz_path = row["file"]
        if not os.path.isabs(npz_path):
            npz_path = os.path.normpath(os.path.join(".", npz_path))
        x = load_all_rd_maps(npz_path)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            logits = model(x_t)
        cy_dets, ve_dets = decode_softmax_detections(logits.cpu())
        cy_true_d = int(row["cyclist_true_d_idx"]); cy_true_r = int(row["cyclist_true_r_idx"])
        ve_true_d = int(row["vehicle_true_d_idx"]);  ve_true_r = int(row["vehicle_true_r_idx"])
        cy_hit = any(abs(det[1]-cy_true_d) <= D_TOL and abs(det[2]-cy_true_r) <= R_TOL for det in cy_dets)
        ve_hit = any(abs(det[1]-ve_true_d) <= D_TOL and abs(det[2]-ve_true_r) <= R_TOL for det in ve_dets)
        rows.append({"cy_hit": cy_hit, "ve_hit": ve_hit,
                     "n_cy": len(cy_dets), "n_ve": len(ve_dets),
                     "r_diff": abs(cy_true_r - ve_true_r)})
    df = pd.DataFrame(rows)
    total = len(df)
    both = (df["cy_hit"] & df["ve_hit"]).sum()
    lines = [
        f"--- {label} ({total}件) ---",
        f"cy_hit   : {df.cy_hit.sum()}/{total} ({df.cy_hit.mean()*100:.1f}%)",
        f"ve_hit   : {df.ve_hit.sum()}/{total} ({df.ve_hit.mean()*100:.1f}%)",
        f"両方正確 : {both}/{total} ({both/total*100:.1f}%)",
        f"cy 平均検出数: {df.n_cy.mean():.2f}  ve 平均検出数: {df.n_ve.mean():.2f}",
    ]
    bins   = [(0,0),(1,2),(3,5),(6,999)]
    blabels = ["r_diff=0","r_diff=1-2","r_diff=3-5","r_diff≥6"]
    for (lo, hi), bl in zip(bins, blabels):
        sub = df[(df["r_diff"] >= lo) & (df["r_diff"] <= hi)]
        if len(sub) == 0: continue
        lines.append(f"  {bl:>12}: ve_hit {sub.ve_hit.sum()}/{len(sub)} ({sub.ve_hit.mean()*100:.1f}%)"
                     f"  cy_hit {sub.cy_hit.sum()}/{len(sub)} ({sub.cy_hit.mean()*100:.1f}%)")
    return "\n".join(lines)

def evaluate_single_target(sub_df, target_class, model):
    true_d_col = "cyclist_true_d_idx" if target_class == "cy" else "vehicle_true_d_idx"
    true_r_col = "cyclist_true_r_idx" if target_class == "cy" else "vehicle_true_r_idx"
    hits, fp_count = 0, 0
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz_path = row["file"]
        if not os.path.isabs(npz_path):
            npz_path = os.path.normpath(os.path.join(".", npz_path))
        x = load_all_rd_maps(npz_path)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            logits = model(x_t)
        cy_dets, ve_dets = decode_softmax_detections(logits.cpu())
        true_d = int(row[true_d_col]); true_r = int(row[true_r_col])
        target_dets   = cy_dets if target_class == "cy" else ve_dets
        opponent_dets = ve_dets if target_class == "cy" else cy_dets
        if any(abs(det[1]-true_d) <= D_TOL and abs(det[2]-true_r) <= R_TOL for det in target_dets):
            hits += 1
        if len(opponent_dets) > 0:
            fp_count += 1
    total = len(sub_df)
    return hits, fp_count, total

# 単一物体holdout
single_all_df     = pd.read_csv(TRAIN_META_CSV)
single_all_df     = single_all_df[single_all_df["valid_all"] == 1].reset_index(drop=True)
single_all_df     = single_all_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
single_holdout_df = single_all_df.iloc[340:].reset_index(drop=True)
cy_only = single_holdout_df[
    single_holdout_df["valid_cyclist"].astype(str).str.strip().isin(["1","True"])
    & ~single_holdout_df["valid_vehicle"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)
ve_only = single_holdout_df[
    single_holdout_df["valid_vehicle"].astype(str).str.strip().isin(["1","True"])
    & ~single_holdout_df["valid_cyclist"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)

cy_hits, cy_fp, cy_total = evaluate_single_target(cy_only, "cy", model)
ve_hits, ve_fp, ve_total = evaluate_single_target(ve_only, "ve", model)

# シナリオテスト
scenario_df = pd.read_csv(SCENARIO_META_CSV)
scenario_df = scenario_df[scenario_df["valid_all"] == 1].reset_index(drop=True)
sc_result   = evaluate_two_object(scenario_df, model, "シナリオテスト")

# 2物体holdout
test_result = evaluate_two_object(test_df, model, "2物体holdout")

# ===== 結果出力 =====
result_lines = [
    "=" * 50,
    "実験A: softmax-baseline",
    "=" * 50,
    "",
    sc_result,
    "",
    test_result,
    "",
    "--- 単一物体テスト ---",
    f"cy-only: hit={cy_hits}/{cy_total} ({cy_hits/max(cy_total,1)*100:.1f}%)  FP={cy_fp}/{cy_total} ({cy_fp/max(cy_total,1)*100:.1f}%)",
    f"ve-only: hit={ve_hits}/{ve_total} ({ve_hits/max(ve_total,1)*100:.1f}%)  FP={ve_fp}/{ve_total} ({ve_fp/max(ve_total,1)*100:.1f}%)",
    "",
    "--- 学習履歴（全エポック）---",
]
for h in history:
    result_lines.append(
        f"epoch={h['epoch']:02d}  train={h['train_loss']:.4f} cy={h['train_cy']:.3f} ve={h['train_ve']:.3f}"
        f"  |  val={h['val_loss']:.4f} cy={h['val_cy']:.3f} ve={h['val_ve']:.3f}"
    )

result_text = "\n".join(result_lines)
print(result_text)
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text + "\n")
print(f"\n結果を {RESULT_PATH} に保存しました")
