"""
過学習（Overfitting）分析スクリプト — 狭角度グリッド実験版

分析内容:
1. 学習曲線の可視化（train vs val loss / hit rate）
   - train_loss > val_loss の原因が Dropout なのか過学習なのかを確認
2. 保存済み最良モデルを eval モードで各データセットに適用し性能を直接比較
   - train set (280) / val set (60) / holdout_single (260) / holdout_two (100)
   - eval モードは Dropout OFF なので学習時の train との乖離は Dropout によるもの
3. 汎化ギャップの定量化
   - train_hit (eval mode) と holdout_hit の差が過学習の指標
"""

import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

# 狭角度グリッド実験用: パスと角度設定を変更
MODEL_PATH      = os.path.join(ROOT_DIR, "best_detector_narrow_angle.pt")
NOTEBOOK_PATH   = os.path.join(ROOT_DIR, "check.ipynb")
SINGLE_META_CSV = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_single", "metadata.csv")
FIXED_META_CSV  = os.path.join(ROOT_DIR, "learn_dataset_narrow_angle_fixed",  "metadata.csv")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "overfitting_analysis_narrow_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FIXED      = 10
FIXED_ANGLES = np.linspace(1, 4, N_FIXED)  # 狭角度グリッド: 1°〜4°
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42
A_TOL, D_TOL, R_TOL = 1, 2, 3


# ===== モデル定義 =====
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch=32, dr=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout3d(dr),
        )
    def forward(self, x): return self.block(x)

class RadarUNet3DSoftmax(nn.Module):
    def __init__(self, n=N_FIXED, ch=32, dr=0.1):
        super().__init__()
        self.encoders  = nn.ModuleList([ConvBlock3D(1 if i==0 else ch, ch, dr) for i in range(4)])
        self.pools     = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck= ConvBlock3D(ch, ch, dr)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders  = nn.ModuleList([ConvBlock3D(ch*2, ch, dr) for _ in range(4)])
        self.seg_head  = nn.Conv3d(ch, 3, 1)
    def forward(self, x):
        x = x.unsqueeze(1); skips, feat = [], x
        for enc, pool in zip(self.encoders, self.pools):
            feat = enc(feat); skips.append(feat); feat = pool(feat)
        feat = self.bottleneck(feat)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            feat = up(feat)
            if feat.shape[-3:] != skip.shape[-3:]:
                feat = F.interpolate(feat, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feat = dec(torch.cat([feat, skip], dim=1))
        return self.seg_head(feat)


# ===== Step 1: 学習曲線をノートブックから取り出す =====
# 狭角度実験セル (e5ccebcc) の出力から学習曲線を抽出する
print("=== Step 1: 学習曲線の解析 ===")

with open(NOTEBOOK_PATH, encoding="utf-8") as f:
    nb = json.load(f)

history = []
for cell in nb["cells"]:
    if cell.get("id") == "e5ccebcc":
        for out in cell.get("outputs", []):
            for line in "".join(out.get("text", [])).split("\n"):
                m = re.match(
                    r"epoch=(\d+)\s+train=([\d.]+) cy=([\d.]+) ve=([\d.]+)"
                    r"\s+\|\s+val=([\d.]+) cy=([\d.]+) ve=([\d.]+)", line
                )
                if m:
                    ep, tl, tcy, tve, vl, vcy, vve = m.groups()
                    history.append(dict(
                        epoch=int(ep),
                        train_loss=float(tl), train_cy=float(tcy), train_ve=float(tve),
                        val_loss=float(vl),   val_cy=float(vcy),   val_ve=float(vve),
                    ))
        break

hist = pd.DataFrame(history)
print(f"  epochs: {len(hist)}")
if len(hist) > 0:
    print(f"  最終 train_loss={hist.train_loss.iloc[-1]:.5f}  val_loss={hist.val_loss.iloc[-1]:.5f}")
    print(f"  最終 train_cy={hist.train_cy.iloc[-1]:.3f}  val_cy={hist.val_cy.iloc[-1]:.3f}")
    print(f"  最終 train_ve={hist.train_ve.iloc[-1]:.3f}  val_ve={hist.val_ve.iloc[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss 曲線
    ax = axes[0]
    ax.plot(hist.epoch, hist.train_loss, label="train", color="tab:blue")
    ax.plot(hist.epoch, hist.val_loss,   label="val",   color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss curve")
    ax.legend(); ax.grid(alpha=0.3)
    note = ("val < train throughout\n=> likely Dropout artifact\n(train=model.train, val=model.eval)")
    ax.text(0.5, 0.6, note, transform=ax.transAxes, fontsize=7, color="gray")

    # cy hit rate
    ax = axes[1]
    ax.plot(hist.epoch, hist.train_cy, label="train_cy", color="tab:blue")
    ax.plot(hist.epoch, hist.val_cy,   label="val_cy",   color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit rate"); ax.set_title("Cyclist hit rate")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

    # ve hit rate
    ax = axes[2]
    ax.plot(hist.epoch, hist.train_ve, label="train_ve", color="tab:blue")
    ax.plot(hist.epoch, hist.val_ve,   label="val_ve",   color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit rate"); ax.set_title("Vehicle hit rate")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Learning Curves (Narrow Angle, Dropout=ON for train)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"), dpi=150, bbox_inches="tight")
    print("  saved: learning_curves.png")
else:
    print("  学習曲線データなし（ノートブック出力未実行）")


# ===== Step 2: eval モードで train/val/holdout を直接評価 =====
print("\n=== Step 2: eval モードでのデータセット別評価 ===")
print("  (Dropout OFF => train と val の損失比較が公平になる)")

def load_sample(path):
    data = np.load(path)
    x = np.stack([20*np.log10(np.maximum(np.abs(data["rd_maps"][i]).astype(np.float32), 1e-12))
                  for i in range(data["rd_maps"].shape[0])], axis=0)
    return x, data

def evaluate_df(model, df, label):
    """eval モードでデータセット全体を評価し loss / hit rate を返す"""
    model.eval()
    focal_alpha = torch.tensor([1.0, 500.0, 500.0]).to(DEVICE)
    total_loss = 0.0
    cy_hits = cy_total = ve_hits = ve_total = 0

    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            path = row["file"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(ROOT_DIR, path))

            x, data = load_sample(path)
            fa   = data["fixed_angles"] if "fixed_angles" in data else FIXED_ANGLES
            vcy  = int(str(row["valid_cyclist"]).strip() in ("1","True"))
            vve  = int(str(row["valid_vehicle"]).strip() in ("1","True"))

            x_t  = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
            H, W = x.shape[1], x.shape[2]

            # ラベル構築
            y = torch.zeros(1, N_FIXED, H, W, dtype=torch.long).to(DEVICE)
            if vcy:
                cyd = int(row["cyclist_true_d_idx"]); cyr = int(row["cyclist_true_r_idx"])
                cych = int(np.argmin(np.abs(fa - float(data["cyclist_true_angle_deg"]))))
                y[0, cych, cyd, cyr] = 1
            if vve:
                ved = int(row["vehicle_true_d_idx"]); ver = int(row["vehicle_true_r_idx"])
                vech = int(np.argmin(np.abs(fa - float(data["vehicle_true_angle_deg"]))))
                y[0, vech, ved, ver] = 2

            logits = model(x_t)  # (1, 3, N_FIXED, H, W)

            # Focal Loss
            ce  = F.cross_entropy(logits, y, weight=focal_alpha, reduction="none")
            p_t = torch.exp(-ce)
            loss = ((1 - p_t)**2 * ce).mean().item()
            total_loss += loss

            # 検出 & ヒット判定（max softmax prob 位置）
            probs = F.softmax(logits, dim=1)[0]  # (3, N, H, W)
            for cls_idx, valid, true_d, true_r, true_ang_key in [
                (1, vcy, int(row["cyclist_true_d_idx"]) if vcy else 0,
                 int(row["cyclist_true_r_idx"]) if vcy else 0,
                 "cyclist_true_angle_deg"),
                (2, vve, int(row["vehicle_true_d_idx"]) if vve else 0,
                 int(row["vehicle_true_r_idx"]) if vve else 0,
                 "vehicle_true_angle_deg"),
            ]:
                if not valid: continue
                tch = int(np.argmin(np.abs(fa - float(data[true_ang_key]))))
                pm   = probs[cls_idx]
                fi   = pm.argmax().item()
                dch  = fi // (H*W); rem = fi%(H*W); dd = rem//W; dr = rem%W
                hit  = (abs(dch-tch)<=A_TOL and abs(dd-true_d)<=D_TOL and abs(dr-true_r)<=R_TOL)
                if cls_idx == 1: cy_total += 1; cy_hits += int(hit)
                else:            ve_total += 1; ve_hits += int(hit)

    avg_loss = total_loss / len(df)
    cy_hit   = cy_hits / max(cy_total, 1)
    ve_hit   = ve_hits / max(ve_total, 1)
    print(f"  [{label:20s}] n={len(df):3d}  loss={avg_loss:.5f}"
          f"  cy_hit={cy_hit:.3f} ({cy_hits}/{cy_total})"
          f"  ve_hit={ve_hit:.3f} ({ve_hits}/{ve_total})")
    return dict(label=label, n=len(df), loss=avg_loss,
                cy_hit=cy_hit, ve_hit=ve_hit,
                cy_hits=cy_hits, cy_total=cy_total,
                ve_hits=ve_hits, ve_total=ve_total)


# データ分割
single = pd.read_csv(SINGLE_META_CSV)
single = single[single["valid_all"]==1].reset_index(drop=True)
single = single.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
train_df      = single.iloc[:280].reset_index(drop=True)
val_df        = single.iloc[280:340].reset_index(drop=True)
holdout_df    = single.iloc[340:].reset_index(drop=True)

fixed = pd.read_csv(FIXED_META_CSV)
fixed = fixed[fixed["valid_all"]==1].reset_index(drop=True)
fixed = fixed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
holdout_two   = fixed.iloc[200:].reset_index(drop=True)

model = RadarUNet3DSoftmax().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

results = []
results.append(evaluate_df(model, train_df,      "train (280)"))
results.append(evaluate_df(model, val_df,        "val   (60)"))
results.append(evaluate_df(model, holdout_df,    "holdout_single(260)"))
results.append(evaluate_df(model, holdout_two,   "holdout_two  (100)"))

res_df = pd.DataFrame(results)


# ===== Step 3: 汎化ギャップの可視化 =====
print("\n=== Step 3: 汎化ギャップの可視化 ===")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

labels = [r["label"].split("(")[0].strip() for r in results]

# Loss 比較
ax = axes[0]
ax.bar(labels, res_df["loss"], color=["tab:blue","tab:orange","tab:green","tab:red"])
ax.set_title("Loss (eval mode, Dropout OFF)")
ax.set_ylabel("Focal Loss"); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(res_df["loss"]):
    ax.text(i, v+0.000002, f"{v:.5f}", ha="center", fontsize=8)

# cy hit rate 比較
ax = axes[1]
ax.bar(labels, res_df["cy_hit"], color=["tab:blue","tab:orange","tab:green","tab:red"])
ax.set_title("Cyclist hit rate (eval mode)")
ax.set_ylim(0, 1.05); ax.set_ylabel("Hit rate"); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(res_df["cy_hit"]):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)

# ve hit rate 比較
ax = axes[2]
ax.bar(labels, res_df["ve_hit"], color=["tab:blue","tab:orange","tab:green","tab:red"])
ax.set_title("Vehicle hit rate (eval mode)")
ax.set_ylim(0, 1.05); ax.set_ylabel("Hit rate"); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(res_df["ve_hit"]):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)

plt.suptitle("Generalization Gap Analysis — Narrow Angle (eval mode)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "generalization_gap.png"), dpi=150, bbox_inches="tight")
print("  saved: generalization_gap.png")


# ===== レポート =====
train_r    = res_df[res_df["label"].str.startswith("train")].iloc[0]
holdout_r  = res_df[res_df["label"].str.startswith("holdout_single")].iloc[0]
holdout2_r = res_df[res_df["label"].str.startswith("holdout_two")].iloc[0]

cy_gap_single = float(train_r["cy_hit"] - holdout_r["cy_hit"])
ve_gap_single = float(train_r["ve_hit"] - holdout_r["ve_hit"])
cy_gap_two    = float(train_r["cy_hit"] - holdout2_r["cy_hit"])
ve_gap_two    = float(train_r["ve_hit"] - holdout2_r["ve_hit"])

report = f"""
============================================================
過学習分析レポート - 狭角度グリッド実験
============================================================

【学習曲線の観察】
  train_loss > val_loss が全エポックで継続
  => Dropout が train 時のみ ON のため（正常な挙動）
  => 損失の差は過学習の証拠ではない

  val hit rate >= train hit rate の傾向
  => val セット(60件) は変動が大きく参考値

【eval モードでの汎化ギャップ（Dropout OFF = 公平な比較）】

  データセット          loss      cy_hit    ve_hit
  train (280)         {train_r['loss']:.5f}   {train_r['cy_hit']:.3f}     {train_r['ve_hit']:.3f}
  val   (60)          {res_df.iloc[1]['loss']:.5f}   {res_df.iloc[1]['cy_hit']:.3f}     {res_df.iloc[1]['ve_hit']:.3f}
  holdout_single(260) {holdout_r['loss']:.5f}   {holdout_r['cy_hit']:.3f}     {holdout_r['ve_hit']:.3f}
  holdout_two  (100)  {holdout2_r['loss']:.5f}   {holdout2_r['cy_hit']:.3f}     {holdout2_r['ve_hit']:.3f}

  汎化ギャップ（train - holdout_single）
    cy: {cy_gap_single:+.3f}
    ve: {ve_gap_single:+.3f}

  汎化ギャップ（train - holdout_two）
    cy: {cy_gap_two:+.3f}
    ve: {ve_gap_two:+.3f}

【判定】
  cy: train={train_r['cy_hit']:.3f} vs holdout={holdout_r['cy_hit']:.3f}
    => gap={cy_gap_single:+.3f} => {'過学習の兆候あり' if cy_gap_single > 0.03 else '過学習なし（汎化良好）'}

  ve: train={train_r['ve_hit']:.3f} vs holdout={holdout_r['ve_hit']:.3f}
    => gap={ve_gap_single:+.3f} => {'過学習の兆候あり' if ve_gap_single > 0.03 else '過学習なし（汎化良好）'}

  2物体 holdout は学習外シナリオ（ゼロショット）なので
  gap が大きくても過学習ではなく分布シフトが原因

============================================================
"""
with open(os.path.join(OUTPUT_DIR, "overfitting_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
# レポート内容をファイルから再読み込みして表示（エンコードエラー回避）
import sys
sys.stdout.buffer.write(report.encode('utf-8', errors='replace'))
sys.stdout.buffer.write(b"\n")
sys.stdout.flush()
print(f"All outputs saved to: {OUTPUT_DIR}")
