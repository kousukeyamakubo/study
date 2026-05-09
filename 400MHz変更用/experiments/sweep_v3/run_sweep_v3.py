"""
実験スイープ v3: top-K soft selection の網羅的探索
- K (top-K数): 1, 3, 5
- fp_w: 0.001, 0.01, 0.1
- 学習データ: 単一物体(single) / 混合2物体(mixed)
  -> 3 x 3 x 2 = 18 run + 混合baseline 1 run = 計19 run
4時間以内完了を目標（各run ~7分 x 19 = 約2.2時間 + 評価）

前回実験との対応:
  v1: A1 baseline(best), B1-B4 detection-aware(全崩壊)
  v2: C1-C3 small FP_penalty(thr=0.5で崩壊), C4 warmup(実質CE-only), C5 top-K K=3 fp_w=1.0(崩壊)
  v3: top-K の K と fp_w を網羅的に探索 + 混合学習の効果を検証

結果は sweep_experiment_results_v3/ に保存（既存結果は変更しない）
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 定数 =====
SINGLE_META_CSV   = "../../learn_dataset_single_object/metadata.csv"
MIXED_META_CSV    = "../../learn_dataset_fixed_angle/metadata.csv"
SCENARIO_CSV      = "../../learn_dataset_scenario_test/metadata.csv"
PREV_RESULTS_DIR  = "../sweep_v1/sweep_experiment_results"
RESULTS_DIR       = "./sweep_experiment_results_v3"

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
# type: "baseline" / "topk" / "daware"
# data: "single" / "mixed"
# k: top-K数 (topkのみ使用)
# dt: NMS閾値 (dawareのみ使用)
CONFIGS = []

# top-K: K x fp_w x data を網羅
for k in [1, 3, 5]:
    for fp_w in [0.001, 0.01, 0.1]:
        CONFIGS.append({
            "name": f"D_tk{k}_fp{fp_w}_single",
            "type": "topk", "k": k, "fp_w": fp_w, "dt": None,
            "data": "single", "cy_w": 500, "ve_w": 500,
        })
        CONFIGS.append({
            "name": f"D_tk{k}_fp{fp_w}_mixed",
            "type": "topk", "k": k, "fp_w": fp_w, "dt": None,
            "data": "mixed", "cy_w": 500, "ve_w": 500,
        })

# daware: 小さいfp_w で単一/混合（B1-B4 0.5-2.0は全崩壊、C1 0.01も崩壊 -> さらに小さく試す）
for fp_w in [0.001, 0.005, 0.01]:
    CONFIGS.append({
        "name": f"D_dw_fp{fp_w}_single",
        "type": "daware", "k": None, "fp_w": fp_w, "dt": 0.5,
        "data": "single", "cy_w": 500, "ve_w": 500,
    })
    CONFIGS.append({
        "name": f"D_dw_fp{fp_w}_mixed",
        "type": "daware", "k": None, "fp_w": fp_w, "dt": 0.5,
        "data": "mixed", "cy_w": 500, "ve_w": 500,
    })

# 混合学習の baseline（比較基準）
CONFIGS.append({
    "name": "D_baseline_mixed",
    "type": "baseline", "k": None, "fp_w": None, "dt": None,
    "data": "mixed", "cy_w": 500, "ve_w": 500,
})

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
        self.encoders  = nn.ModuleList([ConvBlock3D(1,ch,dropout), ConvBlock3D(ch,ch,dropout),
                                        ConvBlock3D(ch,ch,dropout), ConvBlock3D(ch,ch,dropout)])
        self.pools     = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck= ConvBlock3D(ch, ch, dropout)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders  = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head  = nn.Conv3d(ch, 3, 1)

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
        p = pmap.clone(); N, H, W = p.shape; dets = []
        while p.max().item() >= thr:
            fi = torch.argmax(p).item()
            ch, rem = fi//(H*W), fi%(H*W); d, r = rem//W, rem%W
            dets.append((ch, d, r))
            p[max(ch-1,0):min(ch+1,N-1)+1, max(d-3,0):min(d+3,H-1)+1,
              max(r-1,0):min(r+1,W-1)+1] = 0.
        return dets
    return _nms(probs[0,1], threshold), _nms(probs[0,2], threshold)

# ===== 損失 =====
def seg_loss_base(logits, y_seg, cy_w, ve_w):
    w = torch.tensor([1.0, cy_w, ve_w], dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits, y_seg, weight=w)

def seg_loss_daware(logits, y_seg, cy_w, ve_w, fp_w, dt):
    """NMSで検出されたFPピクセルの確率をペナルティとして加算。ハード閾値dt未満では逃げられる問題あり。"""
    base  = seg_loss_base(logits, y_seg, cy_w, ve_w)
    probs = F.softmax(logits, dim=1)
    fp_loss = torch.tensor(0., device=logits.device)
    for i in range(logits.shape[0]):
        cy_dets, ve_dets = decode_detections(logits[i:i+1].detach().cpu(), threshold=dt)
        cy_true = (y_seg[i] == 1); ve_true = (y_seg[i] == 2)
        for ch, d, r in cy_dets:
            if not cy_true[ch, d, r].item(): fp_loss = fp_loss + probs[i, 1, ch, d, r]
        for ch, d, r in ve_dets:
            if not ve_true[ch, d, r].item(): fp_loss = fp_loss + probs[i, 2, ch, d, r]
    return base + fp_w * fp_loss / logits.shape[0]

def seg_loss_topk(logits, y_seg, cy_w, ve_w, fp_w, k):
    """
    top-K soft selection:
    ハード閾値なしで常に上位K個のピクセルをFP候補とする。
    GTと一致しないtop-Kピクセルの確率をペナルティとして加算。
    fp_wが小さすぎるとCE損失に埋もれてFP抑制効果なし、
    大きすぎるとCE損失を圧倒して検出崩壊する。
    """
    base  = seg_loss_base(logits, y_seg, cy_w, ve_w)
    probs = F.softmax(logits, dim=1)  # (B, 3, N, H, W)
    fp_loss = torch.tensor(0., device=logits.device)
    B = logits.shape[0]
    for i in range(B):
        for cls_idx, cls_val in [(1, 1), (2, 2)]:
            p_flat = probs[i, cls_idx].reshape(-1)           # (N*H*W,)
            topk_vals, topk_idx = torch.topk(p_flat, k=min(k, p_flat.numel()))
            y_flat = y_seg[i].reshape(-1)
            for idx_val, prob_val in zip(topk_idx, topk_vals):
                if y_flat[idx_val].item() != cls_val:
                    fp_loss = fp_loss + prob_val
    return base + fp_w * fp_loss / B

# ===== 学習ループ =====
def run_epoch(model, loader, cfg, optimizer=None, eval_thr=0.5):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.; cy_hits = cy_total = ve_hits = ve_total = 0
    for batch in loader:
        x     = batch['x'].to(DEVICE)
        y_seg = batch['y_seg'].to(DEVICE)
        cy_td = batch['cy_true_d'].numpy(); cy_tr = batch['cy_true_r'].numpy()
        ve_td = batch['ve_true_d'].numpy(); ve_tr = batch['ve_true_r'].numpy()
        vcy   = batch['valid_cy'].numpy(); vve   = batch['valid_ve'].numpy()
        if train_mode: optimizer.zero_grad()
        logits = model(x)
        if cfg["type"] == "baseline":
            loss = seg_loss_base(logits, y_seg, cfg["cy_w"], cfg["ve_w"])
        elif cfg["type"] == "topk":
            loss = seg_loss_topk(logits, y_seg, cfg["cy_w"], cfg["ve_w"], cfg["fp_w"], cfg["k"])
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
                if any(abs(d[1]-cy_td[i])<=D_TOL and abs(d[2]-cy_tr[i])<=R_TOL for d in cy_dets): cy_hits += 1
            if vve[i]:
                ve_total += 1
                if any(abs(d[1]-ve_td[i])<=D_TOL and abs(d[2]-ve_tr[i])<=R_TOL for d in ve_dets): ve_hits += 1
    return (total_loss / max(len(loader.dataset), 1),
            cy_hits / max(cy_total, 1),
            ve_hits / max(ve_total, 1))

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
        rows.append(row_d)
    return pd.DataFrame(rows)

def eval_single_target(sub_df, target_class, model, thr):
    td_col = "cyclist_true_d_idx" if target_class == "cy" else "vehicle_true_d_idx"
    tr_col = "cyclist_true_r_idx" if target_class == "cy" else "vehicle_true_r_idx"
    hits = fp = 0
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        td, tr = int(row[td_col]), int(row[tr_col])
        tgt = cy_dets if target_class == "cy" else ve_dets
        opp = ve_dets if target_class == "cy" else cy_dets
        if any(abs(d[1]-td)<=D_TOL and abs(d[2]-tr)<=R_TOL for d in tgt): hits += 1
        if len(opp) > 0: fp += 1
    return hits, fp, len(sub_df)

# ===== データ準備 =====
# 単一物体データ
single_df = pd.read_csv(SINGLE_META_CSV)
single_df = single_df[single_df["valid_all"]==1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
single_train_df = single_df.iloc[:280].reset_index(drop=True)
single_val_df   = single_df.iloc[280:340].reset_index(drop=True)
single_holdout  = single_df.iloc[340:].reset_index(drop=True)

# 2物体混合データ
mixed_df = pd.read_csv(MIXED_META_CSV)
mixed_df = mixed_df[mixed_df["valid_all"]==1].reset_index(drop=True)
mixed_df = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
mixed_train_df = mixed_df.iloc[:180].reset_index(drop=True)
mixed_val_df   = mixed_df.iloc[180:200].reset_index(drop=True)
test_df        = mixed_df.iloc[200:].reset_index(drop=True)  # 両実験共通テスト

# FP評価用（単一物体holdout）
cy_only = single_holdout[
    single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"]) &
    ~single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)
ve_only = single_holdout[
    single_holdout["valid_vehicle"].astype(str).str.strip().isin(["1","True"]) &
    ~single_holdout["valid_cyclist"].astype(str).str.strip().isin(["1","True"])
].reset_index(drop=True)
scenario_df = pd.read_csv(SCENARIO_CSV)
scenario_df = scenario_df[scenario_df["valid_all"]==1].reset_index(drop=True)

print(f"single train={len(single_train_df)}, val={len(single_val_df)}", flush=True)
print(f"mixed  train={len(mixed_train_df)},  val={len(mixed_val_df)}", flush=True)
print(f"test={len(test_df)}, scenario={len(scenario_df)}, cy_only={len(cy_only)}, ve_only={len(ve_only)}", flush=True)

# DataLoader（single/mixed それぞれ作成）
single_train_loader = DataLoader(SegDataset(single_train_df), batch_size=BATCH_SIZE, shuffle=True)
single_val_loader   = DataLoader(SegDataset(single_val_df),   batch_size=BATCH_SIZE, shuffle=False)
mixed_train_loader  = DataLoader(SegDataset(mixed_train_df),  batch_size=BATCH_SIZE, shuffle=True)
mixed_val_loader    = DataLoader(SegDataset(mixed_val_df),    batch_size=BATCH_SIZE, shuffle=False)

# ===== A1 baseline 比較用に読み込み =====
all_results = {}
a1_json = os.path.join(PREV_RESULTS_DIR, "A1_baseline_cy500.json")
if os.path.exists(a1_json):
    with open(a1_json, "r", encoding="utf-8") as f:
        all_results["A1_baseline_cy500"] = json.load(f)
    print("A1 baseline: 既存結果を読み込みました", flush=True)

# ===== 全run実行 =====
for cfg in CONFIGS:
    name     = cfg["name"]
    out_pt   = os.path.join(RESULTS_DIR, f"{name}.pt")
    out_json = os.path.join(RESULTS_DIR, f"{name}.json")

    if os.path.exists(out_json):
        print(f"\n[{name}] 既存結果をスキップ", flush=True)
        with open(out_json, "r", encoding="utf-8") as f:
            all_results[name] = json.load(f)
        continue

    data_type = cfg["data"]
    train_loader = single_train_loader if data_type == "single" else mixed_train_loader
    val_loader   = single_val_loader   if data_type == "single" else mixed_val_loader

    print(f"\n{'='*50}", flush=True)
    if cfg["type"] == "topk":
        print(f"[{name}] K={cfg['k']}, fp_w={cfg['fp_w']}, data={data_type}", flush=True)
    else:
        print(f"[{name}] baseline CE, data={data_type}", flush=True)
    print(f"{'='*50}", flush=True)

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
    print(f"[{name}] 完了 ({elapsed/60:.1f}分)", flush=True)

    model.load_state_dict(torch.load(out_pt, map_location=DEVICE))
    model.eval()
    result = {"name": name, "config": cfg, "history": history, "elapsed_min": elapsed/60, "eval": {}}

    for thr in EVAL_THRESHOLDS:
        sc_df     = evaluate(model, scenario_df, thr)
        test_df_r = evaluate(model, test_df, thr)
        cy_h, cy_fp, cy_n = eval_single_target(cy_only, "cy", model, thr)
        ve_h, ve_fp, ve_n = eval_single_target(ve_only, "ve", model, thr)
        sc_both   = int((sc_df["cy_hit"] & sc_df["ve_hit"]).sum())
        test_both = int((test_df_r["cy_hit"] & test_df_r["ve_hit"]).sum())
        result["eval"][str(thr)] = {
            "scenario":    {"cy_hit": int(sc_df["cy_hit"].sum()), "ve_hit": int(sc_df["ve_hit"].sum()),
                            "both": sc_both, "total": len(sc_df),
                            "n_cy_mean": float(sc_df["n_cy"].mean()), "n_ve_mean": float(sc_df["n_ve"].mean())},
            "holdout2obj": {"cy_hit": int(test_df_r["cy_hit"].sum()), "ve_hit": int(test_df_r["ve_hit"].sum()),
                            "both": test_both, "total": len(test_df_r),
                            "n_cy_mean": float(test_df_r["n_cy"].mean()), "n_ve_mean": float(test_df_r["n_ve"].mean())},
            "single_cy":   {"hit": cy_h, "fp": cy_fp, "total": cy_n},
            "single_ve":   {"hit": ve_h, "fp": ve_fp, "total": ve_n},
        }
        print(f"  thr={thr}  scenario both={sc_both}/{len(sc_df)}"
              f"  holdout both={test_both}/{len(test_df_r)}"
              f"  FP cy={cy_fp}/{cy_n} ve={ve_fp}/{ve_n}", flush=True)

    all_results[name] = result
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ===== レポート生成 =====
print("\nレポートを生成中...", flush=True)
now = datetime.now().strftime("%Y-%m-%d %H:%M")

# A1をレポート先頭に配置
report_order = ["A1_baseline_cy500"] + [c["name"] for c in CONFIGS]

lines = [
    "# 実験レポート v3: top-K soft selection 網羅的探索",
    "",
    f"生成日時: {now}",
    "",
    "## モデルの概要",
    "",
    "**RadarUNet3DSoftmax**",
    "",
    "- 入力: RDマップ `(B, N_FIXED=10, H, W)` をunsqueeze(1)で `(B, 1, 10, H, W)` として3D UNetに入力",
    "- 出力: `(B, 3, 10, H, W)` の3クラスlogit (0=背景 / 1=サイクリスト / 2=車両)",
    "- softmax + NMSで検出: 閾値を超えるピーク全てを検出候補とする（1枚に複数検出可）",
    "- GT: 各ボクセルに0/1/2の離散値（点ラベル）。cy/ve重複時はveが優先",
    "- 損失: weighted cross-entropy (BG=1.0, cy=500, ve=500) + top-K FPペナルティ（本実験）",
    "",
    "## 実験の背景と経緯",
    "",
    "**v1 (A/B series)**: sigmoid出力の崩壊問題を解消するためsoftmax3クラスに移行。",
    "A1 baseline（CE損失のみ）がholdout 99%・scenario 97%と良好な検出性能を達成。",
    "ただし単一物体学習のためcy-only画面でvehicle誤検出76%という問題が残った。",
    "detection-aware loss（B1-B4）はFPペナルティが強すぎて全崩壊した。",
    "",
    "**v2 (C series)**: FP_penaltyを0.01-0.1に小さくしても閾値崩壊（thr=0.5で0%）は解消されず。",
    "C4（warmup=15エポック）はwarmup中のCE-onlyモデルがbest_valとして保存されたため",
    "実質的にCE-onlyモデルと等価。cy FP=0%という副次的改善はあったが設計意図と異なる。",
    "C5（top-K K=3, fp_w=1.0）はペナルティがCE損失を圧倒して完全崩壊（全ep cy=ve=0）。",
    "",
    "**v3（本実験）**: top-Kのfp_wをさらに小さく(0.001-0.1)、Kも1/3/5と変えて網羅探索。",
    "dawareも同様にfp_w=0.001-0.01で再試行（C1=0.01は崩壊したため下限を探る）。",
    "また混合学習（2物体データ）でve誤検出問題が改善するか同時に検証する。",
    "top-Kのメリット: ハード閾値なしで常にtop-K個を評価するため「閾値以下に隠れる崩壊」が構造的に不可能。",
    "dawareの問題: dt=0.5未満に確率を抑えることでFPペナルティを回避できる（閾値崩壊）。",
    "",
    "## 実験設定",
    "",
    "| run | type | K | fp_w | detect_thr | 学習データ |",
    "|---|---|---|---|---|---|",
    "| A1_baseline_cy500 | baseline | - | - | - | single |",
]
for cfg in CONFIGS:
    k_str  = str(cfg["k"])   if cfg["k"]   else "-"
    fp_str = str(cfg["fp_w"]) if cfg["fp_w"] else "-"
    dt_str = str(cfg["dt"])  if cfg["dt"]  else "-"
    lines.append(f"| {cfg['name']} | {cfg['type']} | {k_str} | {fp_str} | {dt_str} | {cfg['data']} |")

lines += [
    "",
    "共通設定: EPOCHS=30, LR=1e-4, BATCH_SIZE=8, D_TOL=2, R_TOL=3, cy_w=ve_w=500",
    "single学習: 280件train / 60件val",
    "mixed学習: 180件train / 20件val (learn_dataset_fixed_angle 2物体データ)",
    "共通テスト: 2物体holdout, シナリオ31件, cy-only holdout, ve-only holdout",
    "",
]

# 主要結果テーブル（thr=0.5）
thr_key = "0.5"
for section, metric_key, total in [
    ("2物体 holdout", "holdout2obj", 100),
    ("シナリオテスト (31件)", "scenario", 31),
]:
    lines += [
        f"## {section}（threshold=0.5）",
        "",
        "| run | data | cy_hit | ve_hit | 両方正確 | cy平均検出数 | ve平均検出数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in report_order:
        if name not in all_results: continue
        data_label = all_results[name]["config"].get("data", "single") if name != "A1_baseline_cy500" else "single"
        e = all_results[name]["eval"].get(thr_key, {}).get(metric_key, {})
        n = e.get("total", total)
        lines.append(f"| {name} | {data_label}"
                     f" | {e.get('cy_hit',0)}/{n} ({e.get('cy_hit',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('ve_hit',0)}/{n} ({e.get('ve_hit',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('both',0)}/{n} ({e.get('both',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")
    lines.append("")

# 単一物体FP率
lines += [
    "## 単一物体テスト FP率（threshold=0.5）",
    "",
    "| run | data | cy hit率 | cy FP率（ve誤検出） | ve hit率 | ve FP率（cy誤検出） |",
    "|---|---|---|---|---|---|",
]
for name in report_order:
    if name not in all_results: continue
    data_label = all_results[name]["config"].get("data", "single") if name != "A1_baseline_cy500" else "single"
    sc  = all_results[name]["eval"].get(thr_key, {}).get("single_cy", {})
    sv  = all_results[name]["eval"].get(thr_key, {}).get("single_ve", {})
    cn, vn = sc.get("total", 1), sv.get("total", 1)
    lines.append(f"| {name} | {data_label}"
                 f" | {sc.get('hit',0)}/{cn} ({sc.get('hit',0)/cn*100:.1f}%)"
                 f" | {sc.get('fp',0)}/{cn} ({sc.get('fp',0)/cn*100:.1f}%)"
                 f" | {sv.get('hit',0)}/{vn} ({sv.get('hit',0)/vn*100:.1f}%)"
                 f" | {sv.get('fp',0)}/{vn} ({sv.get('fp',0)/vn*100:.1f}%) |")
lines.append("")

# K別・fp_w別比較（single/mixedを並べる）
lines += [
    "## K別・fp_w別比較（threshold=0.5, holdout both）",
    "",
    "| K | fp_w | single holdout | mixed holdout | single cy FP | mixed cy FP |",
    "|---|---|---|---|---|---|",
]
for k in [1, 3, 5]:
    for fp_w in [0.001, 0.01, 0.1]:
        n_s = f"D_tk{k}_fp{fp_w}_single"
        n_m = f"D_tk{k}_fp{fp_w}_mixed"
        def _get(name, key1, key2, default=0):
            return all_results.get(name, {}).get("eval", {}).get(thr_key, {}).get(key1, {}).get(key2, default)
        sb = _get(n_s, "holdout2obj", "both"); sn = _get(n_s, "holdout2obj", "total") or 100
        mb = _get(n_m, "holdout2obj", "both"); mn = _get(n_m, "holdout2obj", "total") or 100
        sfp = _get(n_s, "single_cy", "fp"); sct = _get(n_s, "single_cy", "total") or 1
        mfp = _get(n_m, "single_cy", "fp"); mct = _get(n_m, "single_cy", "total") or 1
        lines.append(f"| {k} | {fp_w}"
                     f" | {sb}/{sn} ({sb/sn*100:.1f}%)"
                     f" | {mb}/{mn} ({mb/mn*100:.1f}%)"
                     f" | {sfp}/{sct} ({sfp/sct*100:.1f}%)"
                     f" | {mfp}/{mct} ({mfp/mct*100:.1f}%) |")
lines.append("")

# 閾値感度
lines += ["## 閾値感度（holdout both）", ""]
for name in report_order:
    if name not in all_results: continue
    lines += [f"### {name}", "",
              "| threshold | holdout both | scenario both | cy FP | ve FP |",
              "|---|---|---|---|---|"]
    for thr in [str(t) for t in EVAL_THRESHOLDS]:
        ev  = all_results[name]["eval"].get(thr, {})
        h2  = ev.get("holdout2obj", {}); sc = ev.get("scenario", {})
        scy = ev.get("single_cy", {}); sve = ev.get("single_ve", {})
        h2n = h2.get("total", 100); scn = sc.get("total", 31)
        cn  = scy.get("total", 1); vn = sve.get("total", 1)
        lines.append(f"| {thr} | {h2.get('both',0)}/{h2n} ({h2.get('both',0)/h2n*100:.1f}%)"
                     f" | {sc.get('both',0)}/{scn} ({sc.get('both',0)/scn*100:.1f}%)"
                     f" | {scy.get('fp',0)}/{cn} ({scy.get('fp',0)/cn*100:.1f}%)"
                     f" | {sve.get('fp',0)}/{vn} ({sve.get('fp',0)/vn*100:.1f}%) |")
    lines.append("")

lines += [
    "## 考察",
    "",
    "*(実験後に記入)*",
    "",
    "- top-K: fp_wが小さすぎる場合（CE損失に埋もれる）と大きすぎる場合（崩壊）の境界:",
    "- K=1 vs K=3 vs K=5 の違い:",
    "- daware小fp_w（0.001-0.01）での閾値崩壊回避の可否:",
    "- 混合学習のve FP改善効果:",
    "- single+top-K vs mixed+baseline の比較:",
    "",
]

report_path = os.path.join(RESULTS_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"レポートを保存しました: {report_path}")
print("スイープv3完了")
