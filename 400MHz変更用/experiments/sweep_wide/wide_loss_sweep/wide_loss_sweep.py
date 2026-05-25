"""
広角度（±5°）データセット 損失関数ハイパーパラメータ sweep

FOCAL_GAMMA x FOCAL_ALPHA_pos の 5x3 = 15 通りを探索。
gamma=4.0, alpha=500 は wide_gamma4_eval の結果を流用してスキップ。

出力: ./wide_loss_sweep_results/
"""

import os
import json
import itertools
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

SINGLE_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_single_object", "metadata.csv")
FIXED_META_CSV   = os.path.join(ROOT_DIR, "learn_dataset_fixed_angle",   "metadata.csv")
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "wide_loss_sweep_results")
MODELS_DIR       = os.path.join(SCRIPT_DIR, "sweep_models")
WIDE_GAMMA4_JSON = os.path.join(ROOT_DIR, "experiments", "wide_gamma4_eval",
                                "wide_gamma4_results", "wide_gamma4_results.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 固定設定 =====
N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4

# ===== sweep グリッド =====
GAMMA_VALUES     = [1.0, 2.0, 3.0, 4.0, 5.0]
ALPHA_POS_VALUES = [200.0, 500.0, 1000.0]

# wide_gamma4_eval で既に計算済み
DONE = {(4.0, 500.0)}

# ===== Stage 4 評価設定 =====
A_TOLS = [0, 1]
R_TOLS = list(range(6))
D_TOL  = 0

new_runs = [(g, a) for g, a in itertools.product(GAMMA_VALUES, ALPHA_POS_VALUES)
            if (g, a) not in DONE]
print(f"DEVICE: {DEVICE}")
print(f"新規実行: {len(new_runs)} 通り  スキップ(既存): {len(DONE)} 通り")


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

def make_loss_fn(gamma, alpha_pos):
    alpha = torch.tensor([1.0, alpha_pos, alpha_pos])
    def compute_loss(logits, y_seg):
        a   = alpha.to(logits.device)
        ce  = F.cross_entropy(logits, y_seg, weight=a, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** gamma * ce).mean()
    return compute_loss


# ===== 学習ループ =====

def run_epoch(model, loader, loss_fn, optimizer=None):
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
            loss = loss_fn(model(x), y_seg)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


def train_model(train_loader, val_loader, gamma, alpha_pos, run_name):
    model     = RadarUNet3DSoftmax().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = make_loss_fn(gamma, alpha_pos)

    best_val_loss = float("inf")
    best_state    = None
    save_path     = os.path.join(MODELS_DIR, f"wide_{run_name}.pt")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss   = run_epoch(model, val_loader,   loss_fn, None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{run_name}] epoch={epoch+1:02d}  train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f"  saved: {save_path}  (best_val={best_val_loss:.4f})")
    return model, best_val_loss


# ===== mAP 評価 =====

def collect_raw(model, eval_df):
    cy_raw, ve_raw = [], []
    n_gt_cy = n_gt_ve = 0
    model.eval()
    with torch.no_grad():
        for idx in range(len(eval_df)):
            row  = eval_df.iloc[idx]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))
            data = np.load(path)
            x    = np.stack([
                20 * np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                for i in range(data["rd_maps"].shape[0])
            ], axis=0)
            fa   = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES
            vcy  = int(str(row["valid_cyclist"]).strip() in ("1", "True"))
            vve  = int(str(row["valid_vehicle"]).strip() in ("1", "True"))

            probs = F.softmax(
                model(torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)), dim=1
            )[0]
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
                tch = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                tru = dict(valid=True, true_ch=tch, true_d=int(row["cyclist_true_d_idx"]),
                           true_r=int(row["cyclist_true_r_idx"]), sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in cy_voxels:
                    cy_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

            if vve:
                n_gt_ve += 1
                tch = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                tru = dict(valid=True, true_ch=tch, true_d=int(row["vehicle_true_d_idx"]),
                           true_r=int(row["vehicle_true_r_idx"]), sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))
            else:
                tru = dict(valid=False, true_ch=0, true_d=0, true_r=0,
                           sample_idx=idx, r_diff=r_diff)
                for ch, d, r, sc in ve_voxels:
                    ve_raw.append(dict(score=sc, det_ch=ch, det_d=d, det_r=r, **tru))

    return cy_raw, ve_raw, n_gt_cy, n_gt_ve


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
    return float(np.sum((recs[1:] - recs[:-1]) * precs[1:]))


def compute_stage4_map(model, eval_df):
    cy_raw, ve_raw, n_gt_cy, n_gt_ve = collect_raw(model, eval_df)
    ap_cy_list, ap_ve_list = [], []
    for a_tol in A_TOLS:
        for r_tol in R_TOLS:
            ap_cy_list.append(compute_ap(cy_raw, n_gt_cy, a_tol, D_TOL, r_tol))
            ap_ve_list.append(compute_ap(ve_raw, n_gt_ve, a_tol, D_TOL, r_tol))
    ap_cy = float(np.mean(ap_cy_list))
    ap_ve = float(np.mean(ap_ve_list))
    return ap_cy, ap_ve, (ap_cy + ap_ve) / 2.0


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


# ===== sweep 実行 =====

print("\n=== wide loss sweep 開始 ===")
new_results = []

for gamma, alpha_pos in itertools.product(GAMMA_VALUES, ALPHA_POS_VALUES):
    if (gamma, alpha_pos) in DONE:
        print(f"\n--- gamma{gamma}_alpha{int(alpha_pos)}: 既存のためスキップ ---")
        continue

    run_name = f"gamma{gamma}_alpha{int(alpha_pos)}"
    print(f"\n--- {run_name} ---")

    model, best_val_loss = train_model(train_loader, val_loader, gamma, alpha_pos, run_name)

    print("  評価中...", flush=True)
    ap_cy, ap_ve, mAP = compute_stage4_map(model, eval_df)

    new_results.append({
        "gamma":         gamma,
        "alpha_pos":     alpha_pos,
        "best_val_loss": round(best_val_loss, 4),
        "AP_cy":         round(ap_cy, 4),
        "AP_ve":         round(ap_ve, 4),
        "mAP":           round(mAP,   4),
    })
    print(f"  => AP_cy={ap_cy:.4f}  AP_ve={ap_ve:.4f}  mAP={mAP:.4f}")
    del model


# ===== 既存結果（gamma=4.0, alpha=500）を読み込んで結合 =====

existing_entry = {"gamma": 4.0, "alpha_pos": 500.0, "best_val_loss": None,
                  "AP_cy": None, "AP_ve": None, "mAP": None}
if os.path.exists(WIDE_GAMMA4_JSON):
    with open(WIDE_GAMMA4_JSON, encoding="utf-8") as f:
        j = json.load(f)
    existing_entry.update({
        "best_val_loss": j.get("best_val_loss"),
        "AP_cy":         round(j["stage4"]["AP_cy_avg"], 4),
        "AP_ve":         round(j["stage4"]["AP_ve_avg"], 4),
        "mAP":           round(j["stage4"]["mAP_avg"],   4),
    })
    print(f"\n既存結果読み込み: gamma=4.0, alpha=500  mAP={existing_entry['mAP']}")
else:
    print(f"\n警告: {WIDE_GAMMA4_JSON} が見つかりません。既存結果なしで続行。")

combined_df = pd.DataFrame(new_results + [existing_entry]).sort_values(
    ["gamma", "alpha_pos"]
).reset_index(drop=True)

csv_path = os.path.join(OUTPUT_DIR, "wide_sweep_results.csv")
combined_df.to_csv(csv_path, index=False)
print(f"\nsaved: {csv_path}")
print(combined_df.to_string(index=False))


# ===== ヒートマップ =====

gammas     = sorted(combined_df["gamma"].unique())
alpha_vals = sorted(combined_df["alpha_pos"].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, title in [
    (axes[0], "mAP",   "mAP"),
    (axes[1], "AP_cy", "AP_cy (Cyclist)"),
    (axes[2], "AP_ve", "AP_ve (Vehicle)"),
]:
    grid = np.full((len(gammas), len(alpha_vals)), np.nan)
    for i, g in enumerate(gammas):
        for j, a in enumerate(alpha_vals):
            row = combined_df[(combined_df["gamma"] == g) & (combined_df["alpha_pos"] == a)]
            if len(row) and row[metric].notna().any():
                grid[i, j] = float(row[metric].values[0])

    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([str(int(a)) for a in alpha_vals])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([str(g) for g in gammas])
    ax.set_xlabel("FOCAL_ALPHA (positive weight)")
    ax.set_ylabel("FOCAL_GAMMA")
    ax.set_title(title)
    for i in range(len(gammas)):
        for j in range(len(alpha_vals)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=11)
    plt.colorbar(im, ax=ax)

fig.suptitle(
    "Wide Loss Sweep (±5°) — Stage 4 nuScenes-style mAP\n"
    "(D_TOL=0, A_TOL∈{0,1}, R_TOL=0..5)",
    fontsize=12,
)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "wide_sweep_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved: {heatmap_path}")


# ===== best 設定 =====
valid_rows = combined_df.dropna(subset=["mAP"])
best_row   = valid_rows.loc[valid_rows["mAP"].idxmax()]
print(f"\n=== Best 設定 ===")
print(f"  FOCAL_GAMMA     = {best_row['gamma']}")
print(f"  FOCAL_ALPHA_pos = {best_row['alpha_pos']}")
print(f"  mAP = {best_row['mAP']:.4f}  (AP_cy={best_row['AP_cy']:.4f}, AP_ve={best_row['AP_ve']:.4f})")

json_path = os.path.join(OUTPUT_DIR, "wide_sweep_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "GAMMA_VALUES":     GAMMA_VALUES,
            "ALPHA_POS_VALUES": ALPHA_POS_VALUES,
            "EPOCHS": EPOCHS, "LR": LR, "BATCH_SIZE": BATCH_SIZE,
        },
        "results": combined_df.where(combined_df.notna(), other=None).to_dict(orient="records"),
        "best": {
            "gamma":     float(best_row["gamma"]),
            "alpha_pos": float(best_row["alpha_pos"]),
            "mAP":       float(best_row["mAP"]),
            "AP_cy":     float(best_row["AP_cy"]),
            "AP_ve":     float(best_row["AP_ve"]),
        },
    }, f, indent=2, ensure_ascii=False)
print(f"saved: {json_path}")
print("\nAll done.")
