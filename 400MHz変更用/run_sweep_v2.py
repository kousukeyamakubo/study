"""
実験スイープ v2: detection-aware loss の改善バリアント
A1: baseline CY=500 (sweep_experiment_results/ から結果を読み込み、再学習しない)
C1: FP_penalty=0.01, detect_thr=0.5  (小さいペナルティ)
C2: FP_penalty=0.05, detect_thr=0.5
C3: FP_penalty=0.1,  detect_thr=0.5
C4: warmup=15エポックCEのみ -> FP_penalty=1.0  (段階的導入)
C5: top-K soft selection (K=3), ハード閾値なし  (閾値崩壊を回避)
結果は sweep_experiment_results_v2/ に保存（元の sweep_experiment_results/ は変更しない）
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
TRAIN_DATASET_DIR = "./learn_dataset_single_object"
TEST_DATASET_DIR  = "./learn_dataset_fixed_angle"
SCENARIO_CSV      = "./learn_dataset_scenario_test/metadata.csv"
TRAIN_META_CSV    = os.path.join(TRAIN_DATASET_DIR, "metadata.csv")
TEST_META_CSV     = os.path.join(TEST_DATASET_DIR,  "metadata.csv")
PREV_RESULTS_DIR  = "./sweep_experiment_results"    # 読み取り専用（上書きしない）
RESULTS_DIR       = "./sweep_experiment_results_v2"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4
D_TOL, R_TOL = 2, 3
RANDOM_SEED  = 42
EVAL_THRESHOLDS = [0.3, 0.5, 0.7]
TOPK_K = 3  # C5: 常にtop-K pixelをFP候補とみなす

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)

# ===== 実験設定 =====
# type: "baseline" / "daware" / "topk"
# warmup: CE-onlyのエポック数 (0=ウォームアップなし)
CONFIGS = [
    {"name": "C1_daware_fp0.01", "type": "daware", "cy_w": 500, "ve_w": 500, "fp_w": 0.01, "dt": 0.5, "warmup": 0},
    {"name": "C2_daware_fp0.05", "type": "daware", "cy_w": 500, "ve_w": 500, "fp_w": 0.05, "dt": 0.5, "warmup": 0},
    {"name": "C3_daware_fp0.1",  "type": "daware", "cy_w": 500, "ve_w": 500, "fp_w": 0.1,  "dt": 0.5, "warmup": 0},
    {"name": "C4_warmup15",      "type": "daware", "cy_w": 500, "ve_w": 500, "fp_w": 1.0,  "dt": 0.5, "warmup": 15},
    {"name": "C5_topk3",         "type": "topk",   "cy_w": 500, "ve_w": 500, "fp_w": 1.0,  "dt": None, "warmup": 0},
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
        self.encoders  = nn.ModuleList([ConvBlock3D(1,ch,dropout), ConvBlock3D(ch,ch,dropout),
                                        ConvBlock3D(ch,ch,dropout), ConvBlock3D(ch,ch,dropout)])
        self.pools     = nn.ModuleList([nn.MaxPool3d((1,2,2),(1,2,2)) for _ in range(4)])
        self.bottleneck= ConvBlock3D(ch, ch, dropout)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
        self.decoders  = nn.ModuleList([ConvBlock3D(ch*2, ch, dropout) for _ in range(4)])
        self.seg_head  = nn.Conv3d(ch, 3, 1)

    def forward(self, x):
        # (B, N_FIXED, H, W) -> unsqueeze(1) -> (B, 1, N_FIXED, H, W)
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
    """NMSベースの検出（閾値を超えるピーク全て返す）"""
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

# ===== 損失関数 =====
def seg_loss_base(logits, y_seg, cy_w, ve_w):
    w = torch.tensor([1.0, cy_w, ve_w], dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits, y_seg, weight=w)

def seg_loss_daware(logits, y_seg, cy_w, ve_w, fp_w, dt):
    """ハード閾値NMSで検出されたFPピクセルの確率値をペナルティとして加算"""
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

def seg_loss_topk(logits, y_seg, cy_w, ve_w, fp_w, k=TOPK_K):
    """
    top-K soft selection: 常に上位K個のピクセルをFP候補とみなす。
    ハード閾値を使わないため「閾値以下に隠れる」崩壊が起きない。
    """
    base  = seg_loss_base(logits, y_seg, cy_w, ve_w)
    probs = F.softmax(logits, dim=1)  # (B, 3, N, H, W)
    fp_loss = torch.tensor(0., device=logits.device)
    B = logits.shape[0]
    for i in range(B):
        for cls_idx, cls_mask_val in [(1, 1), (2, 2)]:
            # クラスclsの確率マップをフラット化してtop-Kを取得
            p_flat = probs[i, cls_idx].reshape(-1)       # (N*H*W,)
            topk_vals, topk_idx = torch.topk(p_flat, k=min(k, p_flat.numel()))
            y_flat = y_seg[i].reshape(-1)                # (N*H*W,)
            for idx_val, prob_val in zip(topk_idx, topk_vals):
                # GTと一致しないtop-Kピクセルの確率をペナルティとして加算
                if y_flat[idx_val].item() != cls_mask_val:
                    fp_loss = fp_loss + prob_val
    return base + fp_w * fp_loss / B

# ===== 学習ループ =====
def run_epoch(model, loader, cfg, optimizer=None, eval_thr=0.5, current_epoch=0):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.; cy_hits = cy_total = ve_hits = ve_total = 0

    # warmup中はCEのみ（FPペナルティを無効化）
    warmup_active = (cfg.get("warmup", 0) > 0) and (current_epoch < cfg["warmup"])

    for batch in loader:
        x     = batch['x'].to(DEVICE)
        y_seg = batch['y_seg'].to(DEVICE)
        cy_td = batch['cy_true_d'].numpy(); cy_tr = batch['cy_true_r'].numpy()
        ve_td = batch['ve_true_d'].numpy(); ve_tr = batch['ve_true_r'].numpy()
        vcy   = batch['valid_cy'].numpy(); vve   = batch['valid_ve'].numpy()
        if train_mode: optimizer.zero_grad()

        logits = model(x)

        if cfg["type"] == "baseline" or warmup_active:
            loss = seg_loss_base(logits, y_seg, cfg["cy_w"], cfg["ve_w"])
        elif cfg["type"] == "daware":
            loss = seg_loss_daware(logits, y_seg, cfg["cy_w"], cfg["ve_w"], cfg["fp_w"], cfg["dt"])
        elif cfg["type"] == "topk":
            loss = seg_loss_topk(logits, y_seg, cfg["cy_w"], cfg["ve_w"], cfg["fp_w"])
        else:
            loss = seg_loss_base(logits, y_seg, cfg["cy_w"], cfg["ve_w"])

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

        for i in range(x.size(0)):
            cy_dets, ve_dets = decode_detections(logits[i:i+1].detach().cpu(), eval_thr)
            if vcy[i]:
                cy_total += 1
                if any(abs(det[1]-cy_td[i])<=D_TOL and abs(det[2]-cy_tr[i])<=R_TOL for det in cy_dets):
                    cy_hits += 1
            if vve[i]:
                ve_total += 1
                if any(abs(det[1]-ve_td[i])<=D_TOL and abs(det[2]-ve_tr[i])<=R_TOL for det in ve_dets):
                    ve_hits += 1

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

print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
      f"scenario={len(scenario_df)}, cy_only={len(cy_only)}, ve_only={len(ve_only)}", flush=True)

train_ds     = SegDataset(train_df)
val_ds       = SegDataset(val_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ===== A1ベースライン結果を既存JSONから読み込み（再学習しない） =====
all_results = {}
a1_json = os.path.join(PREV_RESULTS_DIR, "A1_baseline_cy500.json")
if os.path.exists(a1_json):
    with open(a1_json, "r", encoding="utf-8") as f:
        all_results["A1_baseline_cy500"] = json.load(f)
    print("A1 baseline: 既存結果を読み込みました（再学習スキップ）", flush=True)
else:
    print("A1 baseline: 既存JSONが見つかりません。比較表にA1は含まれません", flush=True)

# ===== C1-C5 実行 =====
for cfg in CONFIGS:
    name     = cfg["name"]
    out_pt   = os.path.join(RESULTS_DIR, f"{name}.pt")
    out_json = os.path.join(RESULTS_DIR, f"{name}.json")

    if os.path.exists(out_json):
        print(f"\n[{name}] 既存結果をスキップ", flush=True)
        with open(out_json, "r", encoding="utf-8") as f:
            all_results[name] = json.load(f)
        continue

    print(f"\n{'='*50}\n[{name}] 学習開始\n{'='*50}", flush=True)
    if cfg.get("warmup", 0) > 0:
        print(f"  warmup={cfg['warmup']}エポック (CE-only) -> その後FP_penalty={cfg['fp_w']}", flush=True)
    t0 = time.time()

    model     = RadarUNet3DSoftmax().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val  = float("inf"); history = []

    for epoch in range(EPOCHS):
        tr_loss, tr_cy, tr_ve = run_epoch(model, train_loader, cfg, optimizer, current_epoch=epoch)
        va_loss, va_cy, va_ve = run_epoch(model, val_loader,   cfg, None,      current_epoch=epoch)
        history.append({"epoch": epoch+1,
                         "train_loss": tr_loss, "train_cy": tr_cy, "train_ve": tr_ve,
                         "val_loss":   va_loss, "val_cy":   va_cy, "val_ve":   va_ve})
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), out_pt)
        warmup_str = " [warmup]" if (cfg.get("warmup",0)>0 and epoch<cfg["warmup"]) else ""
        print(f"  ep{epoch+1:02d}{warmup_str}  tr={tr_loss:.4f} cy={tr_cy:.3f} ve={tr_ve:.3f}"
              f"  | val={va_loss:.4f} cy={va_cy:.3f} ve={va_ve:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"[{name}] 学習完了 ({elapsed/60:.1f}分)", flush=True)

    # 評価
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
            "scenario": {
                "cy_hit": int(sc_df["cy_hit"].sum()), "ve_hit": int(sc_df["ve_hit"].sum()),
                "both": sc_both, "total": len(sc_df),
                "n_cy_mean": float(sc_df["n_cy"].mean()), "n_ve_mean": float(sc_df["n_ve"].mean()),
            },
            "holdout2obj": {
                "cy_hit": int(test_df_r["cy_hit"].sum()), "ve_hit": int(test_df_r["ve_hit"].sum()),
                "both": test_both, "total": len(test_df_r),
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
print("\nレポートを生成中...", flush=True)

# A1をレポートの先頭に配置
report_configs_order = ["A1_baseline_cy500"] + [c["name"] for c in CONFIGS]

now = datetime.now().strftime("%Y-%m-%d %H:%M")
lines = [
    "# 実験レポート v2: detection-aware loss 改善バリアント",
    "",
    f"生成日時: {now}",
    "",
    "## 概要",
    "",
    "前回実験 (A1/A2 baseline, B1-B4 detection-aware) の結果を受け、",
    "detection-aware loss の「閾値以下崩壊」問題を解決するための5つのバリアントを試した。",
    "A1 baseline は前回結果 (sweep_experiment_results/A1_baseline_cy500.json) を再利用。",
    "",
    "## 実験設定",
    "",
    "| run | type | CY_weight | FP_penalty | detect_thr | warmup |",
    "|---|---|---|---|---|---|",
    "| A1_baseline_cy500 | baseline | 500 | - | - | - |",
]
for cfg in CONFIGS:
    dt_str  = str(cfg["dt"]) if cfg["dt"] is not None else "-"
    fp_str  = str(cfg["fp_w"]) if cfg["fp_w"] is not None else "-"
    wup_str = str(cfg.get("warmup", 0)) if cfg.get("warmup", 0) > 0 else "-"
    lines.append(f"| {cfg['name']} | {cfg['type']} | {cfg['cy_w']} | {fp_str} | {dt_str} | {wup_str} |")

lines += [
    "",
    "共通設定: EPOCHS=30, LR=1e-4, BATCH_SIZE=8, D_TOL=2, R_TOL=3",
    "学習: 単一物体データ280件 / val: 60件 / test: 2物体holdout100件 + シナリオ31件",
    "C5 top-K: K=3 (ハード閾値なし、常にtop-3ピクセルをFP候補とみなす)",
    "",
]

# 主要指標テーブル（threshold=0.5）
thr_key = "0.5"
lines += [
    f"## 主要結果（threshold={thr_key}）",
    "",
    "### 2物体 holdout (100件)",
    "",
    "| run | cy_hit | ve_hit | 両方正確 | cy平均検出数 | ve平均検出数 |",
    "|---|---|---|---|---|---|",
]
for name in report_configs_order:
    if name not in all_results: continue
    e = all_results[name]["eval"].get(thr_key, {}).get("holdout2obj", {})
    n = e.get("total", 100)
    lines.append(f"| {name} | {e.get('cy_hit','?')}/{n} ({e.get('cy_hit',0)/n*100:.1f}%)"
                 f" | {e.get('ve_hit','?')}/{n} ({e.get('ve_hit',0)/n*100:.1f}%)"
                 f" | {e.get('both','?')}/{n} ({e.get('both',0)/n*100:.1f}%)"
                 f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")

lines += [
    "",
    "### シナリオテスト (31件)",
    "",
    "| run | cy_hit | ve_hit | 両方正確 | cy平均検出数 | ve平均検出数 |",
    "|---|---|---|---|---|---|",
]
for name in report_configs_order:
    if name not in all_results: continue
    e = all_results[name]["eval"].get(thr_key, {}).get("scenario", {})
    n = e.get("total", 31)
    lines.append(f"| {name} | {e.get('cy_hit','?')}/{n} ({e.get('cy_hit',0)/n*100:.1f}%)"
                 f" | {e.get('ve_hit','?')}/{n} ({e.get('ve_hit',0)/n*100:.1f}%)"
                 f" | {e.get('both','?')}/{n} ({e.get('both',0)/n*100:.1f}%)"
                 f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")

lines += [
    "",
    "### 単一物体テスト（誤検出率）",
    "",
    "| run | cy hit率 | cy FP率 (ve誤検出) | ve hit率 | ve FP率 (cy誤検出) |",
    "|---|---|---|---|---|",
]
for name in report_configs_order:
    if name not in all_results: continue
    sc  = all_results[name]["eval"].get(thr_key, {}).get("single_cy", {})
    sv  = all_results[name]["eval"].get(thr_key, {}).get("single_ve", {})
    cn, vn = sc.get("total",1), sv.get("total",1)
    lines.append(f"| {name} | {sc.get('hit',0)}/{cn} ({sc.get('hit',0)/cn*100:.1f}%)"
                 f" | {sc.get('fp',0)}/{cn} ({sc.get('fp',0)/cn*100:.1f}%)"
                 f" | {sv.get('hit',0)}/{vn} ({sv.get('hit',0)/vn*100:.1f}%)"
                 f" | {sv.get('fp',0)}/{vn} ({sv.get('fp',0)/vn*100:.1f}%) |")

# 閾値感度テーブル
lines += ["", "## 閾値感度 (threshold=0.3/0.5/0.7)", ""]
for name in report_configs_order:
    if name not in all_results: continue
    lines += [f"### {name}", "",
              "| threshold | holdout both | scenario both | cy FP率 | ve FP率 |",
              "|---|---|---|---|---|"]
    for thr in [str(t) for t in EVAL_THRESHOLDS]:
        ev  = all_results[name]["eval"].get(thr, {})
        h2  = ev.get("holdout2obj", {}); sc = ev.get("scenario", {})
        scy = ev.get("single_cy", {}); sve = ev.get("single_ve", {})
        h2n, scn = h2.get("total", 100), sc.get("total", 31)
        cn,  vn  = scy.get("total", 1), sve.get("total", 1)
        lines.append(f"| {thr} | {h2.get('both',0)}/{h2n} ({h2.get('both',0)/h2n*100:.1f}%)"
                     f" | {sc.get('both',0)}/{scn} ({sc.get('both',0)/scn*100:.1f}%)"
                     f" | {scy.get('fp',0)}/{cn} ({scy.get('fp',0)/cn*100:.1f}%)"
                     f" | {sve.get('fp',0)}/{vn} ({sve.get('fp',0)/vn*100:.1f}%) |")
    lines.append("")

# 学習履歴（最終5エポック）
lines += ["## 学習履歴（最終5エポック）", ""]
for name in [c["name"] for c in CONFIGS]:
    if name not in all_results: continue
    hist = all_results[name].get("history", [])
    if not hist: continue
    lines += [f"### {name}", "",
              "| epoch | train_loss | train_cy | train_ve | val_loss | val_cy | val_ve |",
              "|---|---|---|---|---|---|---|"]
    for h in hist[-5:]:
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_cy']:.3f} | {h['train_ve']:.3f}"
                     f" | {h['val_loss']:.4f} | {h['val_cy']:.3f} | {h['val_ve']:.3f} |")
    lines.append("")

lines += [
    "## 考察",
    "",
    "*(実験後に記入)*",
    "",
    "- C1-C3（小さいFP_penalty）の効果:",
    "- C4（warmup）の効果: warmupなしとの比較:",
    "- C5（top-K soft selection）の効果: 閾値崩壊は解消されたか:",
    "- A1 baseline vs 最良Cバリアント:",
    "- FP率の改善度合い:",
    "",
]

report_path = os.path.join(RESULTS_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"レポートを保存しました: {report_path}")
print("スイープv2完了")
