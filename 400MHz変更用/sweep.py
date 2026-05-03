import os
import csv
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 定数 =====
TRAIN_DATASET_DIR = "./learn_dataset_single_object"
TEST_DATASET_DIR  = "./learn_dataset_fixed_angle"
TRAIN_META_CSV = os.path.join(TRAIN_DATASET_DIR, "metadata.csv")
TEST_META_CSV  = os.path.join(TEST_DATASET_DIR,  "metadata.csv")

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
SIGMA_ANGLE  = 0.3

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS     = 30
LR         = 1e-4
RANDOM_SEED = 42

D_TOL = 2
R_TOL = 3

SWEEP_MODEL_DIR = "./sweep_models"
SWEEP_RESULT_CSV = "./sweep_results.csv"

os.makedirs(SWEEP_MODEL_DIR, exist_ok=True)

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


class RadarUNet3DIntensity(nn.Module):
    def __init__(self, n_angles=N_FIXED, fixed_channels=32, dropout=0.1):
        super().__init__()
        self.n_angles = n_angles

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
        self.cy_head = nn.Conv3d(fixed_channels, 1, kernel_size=1)
        self.ve_head = nn.Conv3d(fixed_channels, 1, kernel_size=1)
        self.cy_exist_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(fixed_channels, 1),
        )
        self.ve_exist_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(fixed_channels, 1),
        )

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

        cy_logits = self.cy_head(feat).squeeze(1)   # (B, N_FIXED, H, W)
        ve_logits = self.ve_head(feat).squeeze(1)   # (B, N_FIXED, H, W)
        cy_exist  = self.cy_exist_head(feat).squeeze(1)  # (B,)
        ve_exist  = self.ve_exist_head(feat).squeeze(1)  # (B,)
        return cy_logits, ve_logits, cy_exist, ve_exist


# ===== データセット =====
def make_gaussian_heatmap(H, W, center_d, center_r, sigma_d=3.0, sigma_r=5.0):
    d = np.arange(H, dtype=np.float32)[:, None]
    r = np.arange(W, dtype=np.float32)[None, :]
    heatmap = np.exp(
        -0.5 * (((d - center_d) / sigma_d) ** 2 + ((r - center_r) / sigma_r) ** 2)
    )
    return heatmap.astype(np.float32)


class MultiTargetDataset(Dataset):
    def __init__(self, meta_df, sigma_r=5.0):
        self.meta_df = meta_df.reset_index(drop=True)
        self.samples = []
        for idx in range(len(self.meta_df)):
            row = self.meta_df.iloc[idx]
            npz_path = row["file"]
            if not os.path.isabs(npz_path):
                npz_path = os.path.normpath(os.path.join(".", npz_path))

            data = np.load(npz_path)
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

            cy_angle = float(data["cyclist_true_angle_deg"])  if (valid_cy and "cyclist_true_angle_deg" in data) else 0.0
            ve_angle = float(data["vehicle_true_angle_deg"])  if (valid_ve and "vehicle_true_angle_deg" in data) else 0.0

            cy_heatmap = make_gaussian_heatmap(H, W, int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"]), sigma_r=sigma_r) if valid_cy else np.zeros((H, W), dtype=np.float32)
            ve_heatmap = make_gaussian_heatmap(H, W, int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"]), sigma_r=sigma_r) if valid_ve else np.zeros((H, W), dtype=np.float32)

            y_cy = np.zeros((N_FIXED, H, W), dtype=np.float32)
            y_ve = np.zeros((N_FIXED, H, W), dtype=np.float32)
            if valid_cy:
                cy_ch = int(np.argmin(np.abs(fixed_angles - cy_angle)))
                y_cy[cy_ch] = np.clip(y_cy[cy_ch] + cy_heatmap, 0.0, 1.0)
            if valid_ve:
                ve_ch = int(np.argmin(np.abs(fixed_angles - ve_angle)))
                y_ve[ve_ch] = np.clip(y_ve[ve_ch] + ve_heatmap, 0.0, 1.0)

            self.samples.append({
                "x":         torch.from_numpy(x).float(),
                "y_cy":      torch.from_numpy(y_cy).float(),
                "y_ve":      torch.from_numpy(y_ve).float(),
                "cy_true_d": torch.tensor(int(row["cyclist_true_d_idx"]),  dtype=torch.long),
                "cy_true_r": torch.tensor(int(row["cyclist_true_r_idx"]),  dtype=torch.long),
                "ve_true_d": torch.tensor(int(row["vehicle_true_d_idx"]),  dtype=torch.long),
                "ve_true_r": torch.tensor(int(row["vehicle_true_r_idx"]),  dtype=torch.long),
                "valid_cy":  torch.tensor(valid_cy, dtype=torch.long),
                "valid_ve":  torch.tensor(valid_ve, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ===== 損失・学習 =====
def centernet_focal_loss(logits, targets, alpha, beta):
    p = torch.sigmoid(logits)
    pos_mask = (targets == 1).float()
    neg_mask = (targets < 1).float()
    pos_loss = -(1 - p).pow(alpha) * torch.log(p.clamp(min=1e-6)) * pos_mask
    neg_loss = -(1 - targets).pow(beta) * p.pow(alpha) * torch.log((1 - p).clamp(min=1e-6)) * neg_mask
    n_pos = pos_mask.sum().clamp(min=1)
    n_neg = neg_mask.sum().clamp(min=1)
    return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg


def compute_loss(cy_logits, ve_logits, y_cy, y_ve, cy_exist, ve_exist, valid_cy, valid_ve, alpha, beta):
    heatmap_loss = (
        centernet_focal_loss(cy_logits, y_cy, alpha, beta) +
        centernet_focal_loss(ve_logits, y_ve, alpha, beta)
    )
    exist_loss = (
        F.binary_cross_entropy_with_logits(cy_exist, valid_cy.float()) +
        F.binary_cross_entropy_with_logits(ve_exist, valid_ve.float())
    )
    return heatmap_loss + exist_loss


def decode_class_detections(cy_logits, ve_logits, cy_exist, ve_exist):
    def _detect(logits, exist_logit):
        if torch.sigmoid(exist_logit[0]).item() < 0.5:
            return (-1, -1, -1)
        prob = torch.sigmoid(logits[0])  # (N_FIXED, H, W)
        C, H, W = prob.shape
        flat_idx = torch.argmax(prob).item()
        ch  = flat_idx // (H * W)
        rem = flat_idx %  (H * W)
        d   = rem // W
        r   = rem %  W
        return (ch, d, r)
    return _detect(cy_logits, cy_exist), _detect(ve_logits, ve_exist)


def run_epoch(model, loader, optimizer, device, alpha, beta):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    cy_hits, cy_total = 0, 0
    ve_hits, ve_total = 0, 0

    for batch in loader:
        x          = batch['x'].to(device)
        y_cy       = batch['y_cy'].to(device)
        y_ve       = batch['y_ve'].to(device)
        valid_cy_t = batch['valid_cy'].to(device)
        valid_ve_t = batch['valid_ve'].to(device)
        cy_true_d  = batch['cy_true_d'].cpu().numpy()
        cy_true_r  = batch['cy_true_r'].cpu().numpy()
        ve_true_d  = batch['ve_true_d'].cpu().numpy()
        ve_true_r  = batch['ve_true_r'].cpu().numpy()
        valid_cy   = batch['valid_cy'].cpu().numpy()
        valid_ve   = batch['valid_ve'].cpu().numpy()

        if train_mode:
            optimizer.zero_grad()

        cy_logits, ve_logits, cy_exist, ve_exist = model(x)
        loss = compute_loss(cy_logits, ve_logits, y_cy, y_ve,
                            cy_exist, ve_exist, valid_cy_t, valid_ve_t, alpha, beta)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

        for i in range(x.size(0)):
            cy_det, ve_det = decode_class_detections(
                cy_logits[i:i+1].detach().cpu(),
                ve_logits[i:i+1].detach().cpu(),
                cy_exist[i:i+1].detach().cpu(),
                ve_exist[i:i+1].detach().cpu(),
            )
            if valid_cy[i]:
                cy_total += 1
                if cy_det[0] != -1 and abs(cy_det[1] - cy_true_d[i]) <= D_TOL and abs(cy_det[2] - cy_true_r[i]) <= R_TOL:
                    cy_hits += 1
            if valid_ve[i]:
                ve_total += 1
                if ve_det[0] != -1 and abs(ve_det[1] - ve_true_d[i]) <= D_TOL and abs(ve_det[2] - ve_true_r[i]) <= R_TOL:
                    ve_hits += 1

    avg_loss   = total_loss / len(loader.dataset)
    cy_hit_rate = cy_hits / max(cy_total, 1)
    ve_hit_rate = ve_hits / max(ve_total, 1)
    return avg_loss, cy_hit_rate, ve_hit_rate


# ===== データ読み込み =====
def load_dataframes():
    single_df = pd.read_csv(TRAIN_META_CSV)
    single_df = single_df[single_df["valid_all"] == 1].reset_index(drop=True)
    single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    single_train_df = single_df.iloc[:280].reset_index(drop=True)
    single_val_df   = single_df.iloc[280:340].reset_index(drop=True)

    two_df = pd.read_csv(TEST_META_CSV)
    two_df = two_df[two_df["valid_all"] == 1].reset_index(drop=True)
    two_df = two_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    two_train_df = two_df.iloc[:170].reset_index(drop=True)
    two_val_df   = two_df.iloc[170:200].reset_index(drop=True)

    train_df = pd.concat([single_train_df, two_train_df], ignore_index=True)
    val_df   = pd.concat([single_val_df,   two_val_df],   ignore_index=True)
    return train_df, val_df


# ===== スイープ =====
SIGMA_R_LIST = [5.0]
ALPHA_LIST   = [2, 3, 4]
BETA_LIST    = [8]

def run_sweep():
    train_df, val_df = load_dataframes()
    configs = list(itertools.product(SIGMA_R_LIST, ALPHA_LIST, BETA_LIST))
    print(f"スイープ開始: {len(configs)} 通り  デバイス: {DEVICE}")

    # 結果CSVの準備
    fieldnames = ["sigma_r", "alpha", "beta", "best_val_loss",
                  "best_epoch", "final_val_cy_hit", "final_val_ve_hit", "model_path"]
    with open(SWEEP_RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for run_idx, (sigma_r, alpha, beta) in enumerate(configs):
        tag = f"sr{sigma_r}_a{alpha}_b{beta}"
        model_path = os.path.join(SWEEP_MODEL_DIR, f"best_{tag}.pt")
        print(f"\n[{run_idx+1}/{len(configs)}] sigma_r={sigma_r}  alpha={alpha}  beta={beta}")

        # データセット構築（sigma_rごとに再構築）
        train_ds = MultiTargetDataset(train_df, sigma_r=sigma_r)
        val_ds   = MultiTargetDataset(val_df,   sigma_r=sigma_r)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        model     = RadarUNet3DIntensity(n_angles=N_FIXED).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val  = float("inf")
        best_epoch = 0
        final_cy_hit = 0.0
        final_ve_hit = 0.0

        for epoch in range(EPOCHS):
            train_loss, _, _ = run_epoch(model, train_loader, optimizer, DEVICE, alpha, beta)
            val_loss, val_cy, val_ve = run_epoch(model, val_loader, None, DEVICE, alpha, beta)

            if val_loss < best_val:
                best_val   = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)

            final_cy_hit = val_cy
            final_ve_hit = val_ve
            print(f"  epoch={epoch+1:02d}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  cy={val_cy:.3f}  ve={val_ve:.3f}")

        # 結果を追記
        row = {
            "sigma_r":          sigma_r,
            "alpha":            alpha,
            "beta":             beta,
            "best_val_loss":    round(best_val, 5),
            "best_epoch":       best_epoch,
            "final_val_cy_hit": round(final_cy_hit, 4),
            "final_val_ve_hit": round(final_ve_hit, 4),
            "model_path":       model_path,
        }
        with open(SWEEP_RESULT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(f"  → best_val={best_val:.5f} (epoch {best_epoch})  saved: {model_path}")

    print(f"\nスイープ完了。結果: {SWEEP_RESULT_CSV}")


if __name__ == "__main__":
    run_sweep()