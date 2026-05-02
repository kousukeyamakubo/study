"""
実験スイープ実行スクリプト
A1: baseline CY=500 (既存結果があればスキップ)
A2: baseline CY=1000
B1-B4: detection-aware (FP_penalty/detect_thr を変えて4パターン)
各runを threshold=0.3/0.5/0.7 で評価し、最後にMarkdownレポートを生成する
"""
import os, sys, json, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 定数 =====
TRAIN_DATASET_DIR = "./learn_dataset_single_object"
TEST_DATASET_DIR  = "./learn_dataset_fixed_angle"
SCENARIO_CSV      = "./learn_dataset_scenario_test/metadata.csv"
TRAIN_META_CSV    = os.path.join(TRAIN_DATASET_DIR, "metadata.csv")
TEST_META_CSV     = os.path.join(TEST_DATASET_DIR,  "metadata.csv")
RESULTS_DIR       = "./sweep_experiment_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4
D_TOL, R_TOL = 2, 3
RANDOM_SEED  = 42
EVAL_THRESHOLDS = [0.3, 0.5, 0.7]

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)

# ===== 実験設定 =====
CONFIGS = [
    {"name": "A1_baseline_cy500",    "type": "baseline",  "cy_w": 500,  "ve_w": 500, "fp_w": None, "dt": None},
    {"name": "A2_baseline_cy1000",   "type": "baseline",  "cy_w": 1000, "ve_w": 500, "fp_w": None, "dt": None},
    {"name": "B1_daware_fp0.5",      "type": "daware",    "cy_w": 500,  "ve_w": 500, "fp_w": 0.5,  "dt": 0.5},
    {"name": "B2_daware_fp1.0",      "type": "daware",    "cy_w": 500,  "ve_w": 500, "fp_w": 1.0,  "dt": 0.5},
    {"name": "B3_daware_fp2.0",      "type": "daware",    "cy_w": 500,  "ve_w": 500, "fp_w": 2.0,  "dt": 0.5},
    {"name": "B4_daware_fp1.0_dt0.3","type": "daware",    "cy_w": 500,  "ve_w": 500, "fp_w": 1.0,  "dt": 0.3},
]

# ===== モデル =====
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
        self.encoders = nn.ModuleList([ConvBlock3D(1,ch,dropout), ConvBlock3D(ch,ch,dropout),
                                       ConvBlock3D(ch,ch,dropout), ConvBlock3D(ch,ch,dropout)])
        self.pools     = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck= ConvBlock3D(ch, ch, dropout)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders  = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head  = nn.Conv3d(ch, 3, 1)

    def forward(self, x):
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

# ===== データセット =====
def load_rd_maps(npz_path):
    d = np.load(npz_path)
    return np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                     for i in range(d["rd_maps"].shape[0])], axis=0)

class SegDataset(Dataset):
    def __init__(self, meta_df):
        self.samples = []
        for idx in range(len(meta_df)):
            row = meta_df.iloc[idx]
            npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
            d   = np.load(npz)
            x   = np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                             for i in range(d["rd_maps"].shape[0])], axis=0)
            H, W = x.shape[1], x.shape[2]
            fa   = d["fixed_angles"] if "fixed_angles" in d else FIXED_ANGLES
            vcy  = 1 if str(row["valid_cyclist"]).strip() in ("1","True") else 0
            vve  = 1 if str(row["valid_vehicle"]).strip() in ("1","True") else 0
            cya  = float(d["cyclist_true_angle_deg"]) if (vcy and "cyclist_true_angle_deg" in d) else 0.
            vea  = float(d["vehicle_true_angle_deg"]) if (vve and "vehicle_true_angle_deg" in d) else 0.
            y    = np.zeros((N_FIXED, H, W), dtype=np.int64)
            if vcy: y[int(np.argmin(np.abs(fa-cya))), int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])] = 1
            if vve: y[int(np.argmin(np.abs(fa-vea))), int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])] = 2
            self.samples.append({
                "x": torch.from_numpy(x).float(), "y_seg": torch.from_numpy(y).long(),
                "cy_true_d": torch.tensor(int(row["cyclist_true_d_idx"]), dtype=torch.long),
                "cy_true_r": torch.tensor(int(row["cyclist_true_r_idx"]), dtype=torch.long),
                "ve_true_d": torch.tensor(int(row["vehicle_true_d_idx"]), dtype=torch.long),
                "ve_true_r": torch.tensor(int(row["vehicle_true_r_idx"]), dtype=torch.long),
                "valid_cy":  torch.tensor(vcy, dtype=torch.long),
                "valid_ve":  torch.tensor(vve, dtype=torch.long),
            })
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ===== デコード =====
def decode_detections(logits, threshold=0.5):
    probs = F.softmax(logits, dim=1)
    def _nms(pmap, thr):
        p = pmap.clone(); N,H,W = p.shape; dets = []
        while p.max().item() >= thr:
            fi = torch.argmax(p).item()
            ch, rem = fi//(H*W), fi%(H*W); d, r = rem//W, rem%W
            dets.append((ch,d,r))
            p[max(ch-1,0):min(ch+1,N-1)+1, max(d-3,0):min(d+3,H-1)+1, max(r-1,0):min(r+1,W-1)+1] = 0.
        return dets
    return _nms(probs[0,1], threshold), _nms(probs[0,2], threshold)

# ===== 損失 =====
def seg_loss(logits, y_seg, cy_w, ve_w):
    w = torch.tensor([1.0, cy_w, ve_w], dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits, y_seg, weight=w)

def seg_loss_daware(logits, y_seg, cy_w, ve_w, fp_w, dt):
    base = seg_loss(logits, y_seg, cy_w, ve_w)
    probs   = F.softmax(logits, dim=1)
    fp_loss = torch.tensor(0., device=logits.device)
    for i in range(logits.shape[0]):
        cy_dets, ve_dets = decode_detections(logits[i:i+1].detach().cpu(), threshold=dt)
        cy_true = (y_seg[i] == 1); ve_true = (y_seg[i] == 2)
        for ch,d,r in cy_dets:
            if not cy_true[ch,d,r].item(): fp_loss = fp_loss + probs[i,1,ch,d,r]
        for ch,d,r in ve_dets:
            if not ve_true[ch,d,r].item(): fp_loss = fp_loss + probs[i,2,ch,d,r]
    return base + fp_w * fp_loss / logits.shape[0]

# ===== 学習ループ =====
def run_epoch(model, loader, cfg, optimizer=None, eval_thr=0.5):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.; cy_hits = cy_total = ve_hits = ve_total = 0
    for batch in loader:
        x = batch['x'].to(DEVICE); y_seg = batch['y_seg'].to(DEVICE)
        cy_td = batch['cy_true_d'].numpy(); cy_tr = batch['cy_true_r'].numpy()
        ve_td = batch['ve_true_d'].numpy(); ve_tr = batch['ve_true_r'].numpy()
        vcy = batch['valid_cy'].numpy(); vve = batch['valid_ve'].numpy()
        if train_mode: optimizer.zero_grad()
        logits = model(x)
        if cfg["type"] == "baseline":
            loss = seg_loss(logits, y_seg, cfg["cy_w"], cfg["ve_w"])
        else:
            loss = seg_loss_daware(logits, y_seg, cfg["cy_w"], cfg["ve_w"], cfg["fp_w"], cfg["dt"])
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        for i in range(x.size(0)):
            cy_dets, ve_dets = decode_detections(logits[i:i+1].detach().cpu(), eval_thr)
            if vcy[i]:
                cy_total += 1
                if any(abs(det[1]-cy_td[i])<=D_TOL and abs(det[2]-cy_tr[i])<=R_TOL for det in cy_dets): cy_hits += 1
            if vve[i]:
                ve_total += 1
                if any(abs(det[1]-ve_td[i])<=D_TOL and abs(det[2]-ve_tr[i])<=R_TOL for det in ve_dets): ve_hits += 1
    return (total_loss/max(len(loader.dataset),1),
            cy_hits/max(cy_total,1), ve_hits/max(ve_total,1))

# ===== 評価 =====
def evaluate(model, sub_df, thr):
    rows = []
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        row_d = {"cy_hit": False, "ve_hit": False, "n_cy": len(cy_dets), "n_ve": len(ve_dets)}
        if "cyclist_true_d_idx" in row and "cyclist_true_r_idx" in row:
            ctd, ctr = int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"])
            row_d["cy_hit"] = any(abs(d[1]-ctd)<=D_TOL and abs(d[2]-ctr)<=R_TOL for d in cy_dets)
        if "vehicle_true_d_idx" in row and "vehicle_true_r_idx" in row:
            vtd, vtr = int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"])
            row_d["ve_hit"] = any(abs(d[1]-vtd)<=D_TOL and abs(d[2]-vtr)<=R_TOL for d in ve_dets)
            row_d["r_diff"] = abs(ctr - vtr) if "cyclist_true_r_idx" in row else 0
        rows.append(row_d)
    return pd.DataFrame(rows)

def eval_single_target(sub_df, target_class, model, thr):
    td_col = "cyclist_true_d_idx" if target_class=="cy" else "vehicle_true_d_idx"
    tr_col = "cyclist_true_r_idx" if target_class=="cy" else "vehicle_true_r_idx"
    hits = fp = 0
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        td, tr = int(row[td_col]), int(row[tr_col])
        tgt  = cy_dets if target_class=="cy" else ve_dets
        opp  = ve_dets if target_class=="cy" else cy_dets
        if any(abs(d[1]-td)<=D_TOL and abs(d[2]-tr)<=R_TOL for d in tgt): hits += 1
        if len(opp) > 0: fp += 1
    return hits, fp, len(sub_df)

# ===== データ準備 =====
single_df = pd.read_csv(TRAIN_META_CSV)
single_df = single_df[single_df["valid_all"]==1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
train_df  = single_df.iloc[:280].reset_index(drop=True)
val_df    = single_df.iloc[280:340].reset_index(drop=True)

two_df  = pd.read_csv(TEST_META_CSV)
two_df  = two_df[two_df["valid_all"]==1].reset_index(drop=True)
two_df  = two_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df = two_df.iloc[200:].reset_index(drop=True)

single_holdout = single_df.iloc[340:].reset_index(drop=True)
cy_only = single_holdout[single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"])
                          & ~single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"])].reset_index(drop=True)
ve_only = single_holdout[single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"])
                          & ~single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"])].reset_index(drop=True)
scenario_df = pd.read_csv(SCENARIO_CSV)
scenario_df = scenario_df[scenario_df["valid_all"]==1].reset_index(drop=True)

print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
      f"scenario={len(scenario_df)}, cy_only={len(cy_only)}, ve_only={len(ve_only)}", flush=True)

train_ds = SegDataset(train_df)
val_ds   = SegDataset(val_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ===== 全run実行 =====
all_results = {}

for cfg in CONFIGS:
    name     = cfg["name"]
    out_pt   = os.path.join(RESULTS_DIR, f"{name}.pt")
    out_json = os.path.join(RESULTS_DIR, f"{name}.json")

    # 既存結果があればスキップ（A1はbaseline.ptが既にある可能性）
    if name == "A1_baseline_cy500" and os.path.exists("./best_detector_baseline.pt"):
        print(f"\n[{name}] 既存モデルを使用します: ./best_detector_baseline.pt", flush=True)
        out_pt = "./best_detector_baseline.pt"
    elif os.path.exists(out_json):
        print(f"\n[{name}] 既存結果をスキップ", flush=True)
        with open(out_json, "r", encoding="utf-8") as f:
            all_results[name] = json.load(f)
        continue

    print(f"\n{'='*50}\n[{name}] 学習開始\n{'='*50}", flush=True)
    t0 = time.time()

    model     = RadarUNet3DSoftmax().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val  = float("inf"); history = []

    for epoch in range(EPOCHS):
        tr_loss, tr_cy, tr_ve = run_epoch(model, train_loader, cfg, optimizer)
        va_loss, va_cy, va_ve = run_epoch(model, val_loader,   cfg, None)
        history.append({"epoch": epoch+1,
                         "train_loss": tr_loss, "train_cy": tr_cy, "train_ve": tr_ve,
                         "val_loss":   va_loss, "val_cy":   va_cy, "val_ve":   va_ve})
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), out_pt)
        print(f"  ep{epoch+1:02d}  tr={tr_loss:.4f} cy={tr_cy:.3f} ve={tr_ve:.3f}"
              f"  | val={va_loss:.4f} cy={va_cy:.3f} ve={va_ve:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"[{name}] 学習完了 ({elapsed/60:.1f}分)", flush=True)

    # 評価
    model.load_state_dict(torch.load(out_pt, map_location=DEVICE))
    model.eval()
    result = {"name": name, "config": cfg, "history": history, "elapsed_min": elapsed/60, "eval": {}}

    for thr in EVAL_THRESHOLDS:
        sc_df   = evaluate(model, scenario_df, thr)
        test_df_r = evaluate(model, test_df, thr)
        cy_h, cy_fp, cy_n = eval_single_target(cy_only, "cy", model, thr)
        ve_h, ve_fp, ve_n = eval_single_target(ve_only, "ve", model, thr)

        sc_both   = (sc_df["cy_hit"] & sc_df["ve_hit"]).sum()
        test_both = (test_df_r["cy_hit"] & test_df_r["ve_hit"]).sum()

        result["eval"][str(thr)] = {
            "scenario": {
                "cy_hit": int(sc_df["cy_hit"].sum()), "ve_hit": int(sc_df["ve_hit"].sum()),
                "both": int(sc_both), "total": len(sc_df),
                "n_cy_mean": float(sc_df["n_cy"].mean()), "n_ve_mean": float(sc_df["n_ve"].mean()),
            },
            "holdout2obj": {
                "cy_hit": int(test_df_r["cy_hit"].sum()), "ve_hit": int(test_df_r["ve_hit"].sum()),
                "both": int(test_both), "total": len(test_df_r),
                "n_cy_mean": float(test_df_r["n_cy"].mean()), "n_ve_mean": float(test_df_r["n_ve"].mean()),
            },
            "single_cy": {"hit": cy_h, "fp": cy_fp, "total": cy_n},
            "single_ve": {"hit": ve_h, "fp": ve_fp, "total": ve_n},
        }
        print(f"  thr={thr}  scenario both={sc_both}/{len(sc_df)}"
              f"  holdout both={test_both}/{len(test_df_r)}"
              f"  FP cy={cy_fp}/{cy_n} ve={ve_fp}/{ve_n}", flush=True)

    all_results[name] = result
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ===== レポート生成 =====
print("\n\nレポートを生成中...", flush=True)

now = datetime.now().strftime("%Y-%m-%d %H:%M")
lines = [
    f"# 実験レポート: softmax-baseline vs detection-aware-loss",
    f"",
    f"生成日時: {now}",
    f"",
    f"## 実験設定",
    f"",
    f"| run | type | CY_weight | VE_weight | FP_penalty | detect_thr |",
    f"|---|---|---|---|---|---|",
]
for cfg in CONFIGS:
    c = cfg
    lines.append(f"| {c['name']} | {c['type']} | {c['cy_w']} | {c['ve_w']} "
                 f"| {c['fp_w'] if c['fp_w'] else '—'} | {c['dt'] if c['dt'] else '—'} |")

lines += ["", "共通設定: EPOCHS=30, LR=1e-4, BATCH_SIZE=8, D_TOL=2, R_TOL=3", "",
          "学習: 単一物体データ280件 / val: 60件 / test: 2物体holdout100件 + シナリオ31件", ""]

# 主要指標テーブル（threshold=0.5）
thr_key = "0.5"
lines += [
    f"## 主要結果（threshold={thr_key}）",
    f"",
    f"### 2物体 holdout (100件)",
    f"",
    f"| run | cy_hit | ve_hit | 両方正確 | cy平均検出数 | ve平均検出数 |",
    f"|---|---|---|---|---|---|",
]
for cfg in CONFIGS:
    name = cfg["name"]
    if name not in all_results: continue
    e = all_results[name]["eval"].get(thr_key, {}).get("holdout2obj", {})
    n = e.get("total", 100)
    lines.append(f"| {name} | {e.get('cy_hit','?')}/{n} ({e.get('cy_hit',0)/n*100:.1f}%)"
                 f" | {e.get('ve_hit','?')}/{n} ({e.get('ve_hit',0)/n*100:.1f}%)"
                 f" | {e.get('both','?')}/{n} ({e.get('both',0)/n*100:.1f}%)"
                 f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")

lines += [
    f"",
    f"### シナリオテスト (31件)",
    f"",
    f"| run | cy_hit | ve_hit | 両方正確 | cy平均検出数 | ve平均検出数 |",
    f"|---|---|---|---|---|---|",
]
for cfg in CONFIGS:
    name = cfg["name"]
    if name not in all_results: continue
    e = all_results[name]["eval"].get(thr_key, {}).get("scenario", {})
    n = e.get("total", 31)
    lines.append(f"| {name} | {e.get('cy_hit','?')}/{n} ({e.get('cy_hit',0)/n*100:.1f}%)"
                 f" | {e.get('ve_hit','?')}/{n} ({e.get('ve_hit',0)/n*100:.1f}%)"
                 f" | {e.get('both','?')}/{n} ({e.get('both',0)/n*100:.1f}%)"
                 f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")

lines += [
    f"",
    f"### 単一物体テスト（誤検出率）",
    f"",
    f"| run | cy hit率 | cy FP率 | ve hit率 | ve FP率 |",
    f"|---|---|---|---|---|",
]
for cfg in CONFIGS:
    name = cfg["name"]
    if name not in all_results: continue
    sc = all_results[name]["eval"].get(thr_key, {}).get("single_cy", {})
    sv = all_results[name]["eval"].get(thr_key, {}).get("single_ve", {})
    cn, vn = sc.get("total",1), sv.get("total",1)
    lines.append(f"| {name} | {sc.get('hit',0)}/{cn} ({sc.get('hit',0)/cn*100:.1f}%)"
                 f" | {sc.get('fp',0)}/{cn} ({sc.get('fp',0)/cn*100:.1f}%)"
                 f" | {sv.get('hit',0)}/{vn} ({sv.get('hit',0)/vn*100:.1f}%)"
                 f" | {sv.get('fp',0)}/{vn} ({sv.get('fp',0)/vn*100:.1f}%) |")

# threshold 感度テーブル
lines += ["", "## 閾値感度 (threshold=0.3/0.5/0.7)", ""]
for cfg in CONFIGS:
    name = cfg["name"]
    if name not in all_results: continue
    lines += [f"### {name}", "", f"| threshold | holdout both | scenario both | cy FP率 | ve FP率 |",
              f"|---|---|---|---|---|"]
    for thr in [str(t) for t in EVAL_THRESHOLDS]:
        ev = all_results[name]["eval"].get(thr, {})
        h2 = ev.get("holdout2obj", {}); sc = ev.get("scenario", {})
        scy= ev.get("single_cy", {}); sve= ev.get("single_ve", {})
        h2n, scn = h2.get("total",100), sc.get("total",31)
        cn, vn = scy.get("total",1), sve.get("total",1)
        lines.append(f"| {thr} | {h2.get('both',0)}/{h2n} ({h2.get('both',0)/h2n*100:.1f}%)"
                     f" | {sc.get('both',0)}/{scn} ({sc.get('both',0)/scn*100:.1f}%)"
                     f" | {scy.get('fp',0)}/{cn} ({scy.get('fp',0)/cn*100:.1f}%)"
                     f" | {sve.get('fp',0)}/{vn} ({sve.get('fp',0)/vn*100:.1f}%) |")
    lines.append("")

# 学習履歴（最終5エポック）
lines += ["## 学習履歴（最終5エポック）", ""]
for cfg in CONFIGS:
    name = cfg["name"]
    if name not in all_results: continue
    hist = all_results[name].get("history", [])
    if not hist: continue
    lines += [f"### {name}", ""]
    lines += [f"| epoch | train_loss | train_cy | train_ve | val_loss | val_cy | val_ve |",
              f"|---|---|---|---|---|---|---|"]
    for h in hist[-5:]:
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_cy']:.3f} | {h['train_ve']:.3f}"
                     f" | {h['val_loss']:.4f} | {h['val_cy']:.3f} | {h['val_ve']:.3f} |")
    lines.append("")

# 考察欄
lines += [
    "## 考察",
    "",
    "*(実験後に記入)*",
    "",
    "- baseline vs detection-aware の検出性能の差:",
    "- FP_PENALTY_WEIGHT の影響 (B1/B2/B3 比較):",
    "- DETECT_THRESHOLD の影響 (B2/B4 比較):",
    "- 閾値感度（threshold変化による変動）:",
    "- 誤検出率への影響:",
    "",
]

report_path = os.path.join(RESULTS_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"レポートを保存しました: {report_path}")
print("スイープ完了")
