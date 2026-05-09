"""
v3 選抜モデルの角度チャネル込み再評価スクリプト

v3 の hit 定義は Doppler/Range のみで角度チャネルを無視していた。
本スクリプトでは角度チャネルの許容幅 A_TOL=1 を追加した正しい hit 定義で再評価する。

対象 run:
  - D_tk3_fp0.001_mixed
  - D_dw_fp0.001_single
  - D_dw_fp0.001_mixed

結果は eval_angle_aware_v3_results/ に保存。
"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# ===== 定数 =====
SINGLE_META_CSV  = "../../learn_dataset_single_object/metadata.csv"
MIXED_META_CSV   = "../../learn_dataset_fixed_angle/metadata.csv"
SCENARIO_CSV     = "../../learn_dataset_scenario_test/metadata.csv"
V3_RESULTS_DIR   = "../sweep_v3/sweep_experiment_results_v3"
OUTPUT_DIR       = "./eval_angle_aware_v3_results"

N_FIXED      = 10
FIXED_ANGLES = np.linspace(-5, 5, N_FIXED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
D_TOL, R_TOL = 2, 3
A_TOL        = 1   # 角度チャネル許容幅（±1チャネル）
RANDOM_SEED  = 42
EVAL_THRESHOLDS = [0.3, 0.5, 0.7]

TARGET_RUNS = [
    "D_tk3_fp0.001_mixed",
    "D_dw_fp0.001_single",
    "D_dw_fp0.001_mixed",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DEVICE: {DEVICE}", flush=True)

# ===== モデル（v3 と同一アーキテクチャ）=====
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
        self.upsamples  = nn.ModuleList([nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False) for _ in range(4)])
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

# ===== データ読み込み =====
def load_rd_maps(npz_path):
    d = np.load(npz_path)
    return np.stack([20.*np.log10(np.maximum(np.abs(d["rd_maps"][i].astype(np.float32)), 1e-12))
                     for i in range(d["rd_maps"].shape[0])], axis=0)

def load_angle_info(npz_path):
    """npz から fixed_angles・各物体の真の角度チャネルを返す。"""
    d = np.load(npz_path)
    fa = d["fixed_angles"].astype(float) if "fixed_angles" in d else FIXED_ANGLES
    cy_ach = int(np.argmin(np.abs(fa - float(d["cyclist_true_angle_deg"])))) \
             if "cyclist_true_angle_deg" in d else None
    ve_ach = int(np.argmin(np.abs(fa - float(d["vehicle_true_angle_deg"])))) \
             if "vehicle_true_angle_deg" in d else None
    return cy_ach, ve_ach

# ===== デコード（v3 と同一）=====
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

# ===== 角度込み hit 判定ヘルパー =====
def _hit(dets, true_ach, true_d, true_r):
    """D/R に加えて角度チャネルも ±A_TOL で判定。true_ach が None なら角度条件をスキップ。"""
    for ch, d, r in dets:
        if true_ach is not None and abs(ch - true_ach) > A_TOL:
            continue
        if abs(d - true_d) <= D_TOL and abs(r - true_r) <= R_TOL:
            return True
    return False

# ===== 角度込み evaluate =====
def evaluate(model, sub_df, thr):
    rows = []
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        cy_ach, ve_ach = load_angle_info(npz)
        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        row_d = {"cy_hit": False, "ve_hit": False, "n_cy": len(cy_dets), "n_ve": len(ve_dets)}
        if "cyclist_true_d_idx" in row and "cyclist_true_r_idx" in row:
            row_d["cy_hit"] = _hit(cy_dets, cy_ach, int(row["cyclist_true_d_idx"]), int(row["cyclist_true_r_idx"]))
        if "vehicle_true_d_idx" in row and "vehicle_true_r_idx" in row:
            row_d["ve_hit"] = _hit(ve_dets, ve_ach, int(row["vehicle_true_d_idx"]), int(row["vehicle_true_r_idx"]))
        rows.append(row_d)
    return pd.DataFrame(rows)

# ===== 角度込み eval_single_target =====
def eval_single_target(sub_df, target_class, model, thr):
    td_col  = "cyclist_true_d_idx" if target_class == "cy" else "vehicle_true_d_idx"
    tr_col  = "cyclist_true_r_idx" if target_class == "cy" else "vehicle_true_r_idx"
    hits = fp = 0
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        npz = row["file"] if os.path.isabs(row["file"]) else os.path.normpath(os.path.join(".", row["file"]))
        cy_ach, ve_ach = load_angle_info(npz)
        x_t = torch.from_numpy(load_rd_maps(npz)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): logits = model(x_t)
        cy_dets, ve_dets = decode_detections(logits.cpu(), thr)
        tgt_ach = cy_ach if target_class == "cy" else ve_ach
        tgt     = cy_dets if target_class == "cy" else ve_dets
        opp     = ve_dets if target_class == "cy" else cy_dets
        if _hit(tgt, tgt_ach, int(row[td_col]), int(row[tr_col])): hits += 1
        if len(opp) > 0: fp += 1
    return hits, fp, len(sub_df)

# ===== データ準備 =====
single_df = pd.read_csv(SINGLE_META_CSV)
single_df = single_df[single_df["valid_all"]==1].reset_index(drop=True)
single_df = single_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
single_holdout = single_df.iloc[340:].reset_index(drop=True)

mixed_df = pd.read_csv(MIXED_META_CSV)
mixed_df = mixed_df[mixed_df["valid_all"]==1].reset_index(drop=True)
mixed_df = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df  = mixed_df.iloc[200:].reset_index(drop=True)

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

print(f"test={len(test_df)}, scenario={len(scenario_df)}, cy_only={len(cy_only)}, ve_only={len(ve_only)}", flush=True)

# ===== 再評価ループ =====
all_results = {}
for name in TARGET_RUNS:
    pt_path   = os.path.join(V3_RESULTS_DIR, f"{name}.pt")
    json_path = os.path.join(V3_RESULTS_DIR, f"{name}.json")
    out_json  = os.path.join(OUTPUT_DIR,      f"{name}.json")

    if not os.path.exists(pt_path):
        print(f"[{name}] モデルファイルが見つかりません: {pt_path}", flush=True)
        continue

    # 元の config を v3 の json から読む
    with open(json_path, "r", encoding="utf-8") as f:
        v3_result = json.load(f)
    cfg = v3_result["config"]

    print(f"\n{'='*50}", flush=True)
    print(f"[{name}] 角度込み再評価", flush=True)
    print(f"{'='*50}", flush=True)

    model = RadarUNet3DSoftmax().to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()

    result = {"name": name, "config": cfg, "eval": {}}

    for thr in EVAL_THRESHOLDS:
        sc_df     = evaluate(model, scenario_df, thr)
        test_df_r = evaluate(model, test_df,     thr)
        cy_h, cy_fp, cy_n = eval_single_target(cy_only, "cy", model, thr)
        ve_h, ve_fp, ve_n = eval_single_target(ve_only, "ve", model, thr)
        sc_both   = int((sc_df["cy_hit"]     & sc_df["ve_hit"]).sum())
        test_both = int((test_df_r["cy_hit"] & test_df_r["ve_hit"]).sum())
        result["eval"][str(thr)] = {
            "scenario":    {"cy_hit": int(sc_df["cy_hit"].sum()),     "ve_hit": int(sc_df["ve_hit"].sum()),
                            "both": sc_both,  "total": len(sc_df),
                            "n_cy_mean": float(sc_df["n_cy"].mean()), "n_ve_mean": float(sc_df["n_ve"].mean())},
            "holdout2obj": {"cy_hit": int(test_df_r["cy_hit"].sum()), "ve_hit": int(test_df_r["ve_hit"].sum()),
                            "both": test_both, "total": len(test_df_r),
                            "n_cy_mean": float(test_df_r["n_cy"].mean()), "n_ve_mean": float(test_df_r["n_ve"].mean())},
            "single_cy":   {"hit": cy_h, "fp": cy_fp, "total": cy_n},
            "single_ve":   {"hit": ve_h, "fp": ve_fp, "total": ve_n},
        }
        print(f"  thr={thr}  scenario both={sc_both}/{len(sc_df)}"
              f"  holdout both={test_both}/{len(test_df_r)}"
              f"  cy_hit={int(test_df_r['cy_hit'].sum())}  ve_hit={int(test_df_r['ve_hit'].sum())}"
              f"  FP cy={cy_fp}/{cy_n} ve={ve_fp}/{ve_n}", flush=True)

    all_results[name] = result
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ===== レポート生成 =====
print("\nレポート生成中...", flush=True)
now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = [
    "# v3 選抜モデル 角度チャネル込み再評価",
    "",
    f"生成日時: {now}",
    "",
    "## 変更点",
    "",
    "v3 の hit 定義では角度チャネルを無視していた（Doppler/Range のみ）。",
    f"本評価では A_TOL={A_TOL} を追加し、正解角度チャネルから ±{A_TOL} チャネル以内かつ",
    f"Doppler ±{D_TOL}、Range ±{R_TOL} ビン以内を hit とする。",
    "",
    "## 対象モデル",
    "",
    "| run | type | K | fp_w | 学習データ |",
    "|---|---|---|---|---|",
]
for name in TARGET_RUNS:
    if name not in all_results: continue
    cfg = all_results[name]["config"]
    k_str  = str(cfg.get("k"))    if cfg.get("k")    else "-"
    fp_str = str(cfg.get("fp_w")) if cfg.get("fp_w") else "-"
    lines.append(f"| {name} | {cfg['type']} | {k_str} | {fp_str} | {cfg['data']} |")

lines += ["", f"共通設定: D_TOL={D_TOL}, R_TOL={R_TOL}, A_TOL={A_TOL}", ""]

# v3 元の値との比較表を thr=0.5 で出力
for section, metric_key, total in [
    ("2物体 holdout", "holdout2obj", 100),
    ("シナリオテスト (31件)", "scenario", 31),
]:
    lines += [
        f"## {section}（threshold=0.5）",
        "",
        "| run | cy_hit (角度込み) | ve_hit (角度込み) | 両方正確 | cy平均検出数 | ve平均検出数 |",
        "|---|---|---|---|---|---|",
    ]
    for name in TARGET_RUNS:
        if name not in all_results: continue
        e = all_results[name]["eval"].get("0.5", {}).get(metric_key, {})
        n = e.get("total", total)
        lines.append(f"| {name}"
                     f" | {e.get('cy_hit',0)}/{n} ({e.get('cy_hit',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('ve_hit',0)}/{n} ({e.get('ve_hit',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('both',0)}/{n} ({e.get('both',0)/max(n,1)*100:.1f}%)"
                     f" | {e.get('n_cy_mean',0):.2f} | {e.get('n_ve_mean',0):.2f} |")
    lines.append("")

lines += [
    "## 単一物体テスト FP率（threshold=0.5）",
    "",
    "| run | cy hit率 (角度込み) | cy FP率（ve誤検出） | ve hit率 (角度込み) | ve FP率（cy誤検出） |",
    "|---|---|---|---|---|",
]
for name in TARGET_RUNS:
    if name not in all_results: continue
    e   = all_results[name]["eval"].get("0.5", {})
    sc  = e.get("single_cy", {})
    sv  = e.get("single_ve", {})
    cn, vn = sc.get("total", 1), sv.get("total", 1)
    lines.append(f"| {name}"
                 f" | {sc.get('hit',0)}/{cn} ({sc.get('hit',0)/cn*100:.1f}%)"
                 f" | {sc.get('fp',0)}/{cn} ({sc.get('fp',0)/cn*100:.1f}%)"
                 f" | {sv.get('hit',0)}/{vn} ({sv.get('hit',0)/vn*100:.1f}%)"
                 f" | {sv.get('fp',0)}/{vn} ({sv.get('fp',0)/vn*100:.1f}%) |")
lines.append("")

# 閾値感度
for name in TARGET_RUNS:
    if name not in all_results: continue
    lines += [
        f"### {name} 閾値感度",
        "",
        "| threshold | holdout both | scenario both | cy FP | ve FP |",
        "|---|---|---|---|---|",
    ]
    for thr_str in ["0.3", "0.5", "0.7"]:
        e   = all_results[name]["eval"].get(thr_str, {})
        h   = e.get("holdout2obj", {})
        s   = e.get("scenario",    {})
        sc  = e.get("single_cy",   {})
        sv  = e.get("single_ve",   {})
        hn, sn = h.get("total",1), s.get("total",1)
        cn, vn = sc.get("total",1), sv.get("total",1)
        lines.append(f"| {thr_str}"
                     f" | {h.get('both',0)}/{hn} ({h.get('both',0)/hn*100:.1f}%)"
                     f" | {s.get('both',0)}/{sn} ({s.get('both',0)/sn*100:.1f}%)"
                     f" | {sc.get('fp',0)}/{cn} ({sc.get('fp',0)/cn*100:.1f}%)"
                     f" | {sv.get('fp',0)}/{vn} ({sv.get('fp',0)/vn*100:.1f}%) |")
    lines.append("")

report_path = os.path.join(OUTPUT_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nレポート保存: {report_path}", flush=True)
