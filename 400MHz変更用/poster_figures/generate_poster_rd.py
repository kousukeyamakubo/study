"""
ポスター用RDマップ生成スクリプト（最終版）

- cyclist と vehicle が同時存在するサンプルを使用
- マーカー・注釈なし（パワポで後から追記）
- 全て poster_figures/ に保存
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_ANGLES = np.linspace(-5, 5, 10)

# ---- データ読み込み ----
meta = pd.read_csv('../learn_dataset_fixed_angle/metadata.csv')
valid = meta[meta['valid_cyclist'].astype(bool) & meta['valid_vehicle'].astype(bool)].reset_index(drop=True)

def neighborhood_peak(rd_ch, d_center, r_center, pad=8):
    H, W = rd_ch.shape
    d0, d1 = max(0, d_center-pad), min(H, d_center+pad+1)
    r0, r1 = max(0, r_center-pad), min(W, r_center+pad+1)
    patch = rd_ch[d0:d1, r0:r1]
    ld, lr = np.unravel_index(patch.argmax(), patch.shape)
    return patch.max(), d0+ld, r0+lr

# 強度差が最大のサンプルを選ぶ
best, best_diff = None, -np.inf
for _, row in valid.iterrows():
    rel = row['file'].replace('\\', os.sep)
    # "./learn_dataset_..." のような相対パスを "../" 基準に直す
    if rel.startswith('.' + os.sep):
        rel = rel[2:]
    path = os.path.join('..', rel)
    data = np.load(path, allow_pickle=True)
    rd = np.abs(data['rd_maps'])

    cy_ch = int(np.argmin(np.abs(FIXED_ANGLES - float(row['cyclist_true_angle_deg']))))
    ve_ch = int(np.argmin(np.abs(FIXED_ANGLES - float(row['vehicle_true_angle_deg']))))
    cy_peak, cy_d, cy_r = neighborhood_peak(rd[cy_ch], int(row['cyclist_true_d_idx']), int(row['cyclist_true_r_idx']))
    ve_peak, ve_d, ve_r = neighborhood_peak(rd[ve_ch], int(row['vehicle_true_d_idx']), int(row['vehicle_true_r_idx']))

    if cy_peak > 0 and ve_peak > 0:
        diff = abs(20 * np.log10(ve_peak / cy_peak))
        if diff > best_diff:
            best_diff = diff
            best = dict(rd=rd, cy_ch=cy_ch, ve_ch=ve_ch,
                        cy_d=cy_d, cy_r=cy_r, cy_peak=cy_peak,
                        ve_d=ve_d, ve_r=ve_r, ve_peak=ve_peak,
                        diff_dB=20*np.log10(ve_peak/cy_peak))

rd_mip = best['rd'].max(axis=0)  # (H, W) 最大値投影
vmax = rd_mip.max()

print(f"cyclist  d={best['cy_d']} r={best['cy_r']} peak={best['cy_peak']:.4e}")
print(f"vehicle  d={best['ve_d']} r={best['ve_r']} peak={best['ve_peak']:.4e}")
print(f"diff: {best['diff_dB']:+.1f} dB (vehicle - cyclist)")

def to_db(rd, vmax, floor=-40):
    eps = vmax * 1e-9
    return np.clip(20 * np.log10((rd + eps) / vmax), floor, 0)

# ====================================================
# 図1: ズームイン版 — 両ターゲットが見える範囲に絞る
# ====================================================
# 両ターゲットを含む表示範囲を決める
r_min = min(best['cy_r'], best['ve_r']) - 20
r_max = max(best['cy_r'], best['ve_r']) + 20
d_min = max(0, min(best['cy_d'], best['ve_d']) - 10)
d_max = min(rd_mip.shape[0], max(best['cy_d'], best['ve_d']) + 10)
r_min, r_max = max(0, r_min), min(rd_mip.shape[1], r_max)

rd_zoom = rd_mip[d_min:d_max, r_min:r_max]
DB_FLOOR = -40
rd_zoom_db = to_db(rd_zoom, vmax, floor=DB_FLOOR)

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(rd_zoom_db, aspect='auto', origin='lower',
               cmap='jet', vmin=DB_FLOOR, vmax=0,
               extent=[r_min, r_max, d_min, d_max])
ax.set_xlabel('Range bin', fontsize=15)
ax.set_ylabel('Doppler bin', fontsize=15)
ax.tick_params(labelsize=13)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=13)
cb.ax.tick_params(labelsize=12)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'rd_map_zoom.png')
fig.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
print(f"saved: {out1}")
plt.close()

# ====================================================
# 図2: 全体マップ（-40 dB フロア）
# ====================================================
rd_mip_db = to_db(rd_mip, vmax, floor=DB_FLOOR)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(rd_mip_db, aspect='auto', origin='lower',
               cmap='jet', vmin=DB_FLOOR, vmax=0,
               extent=[0, rd_mip.shape[1], 0, rd_mip.shape[0]])
ax.set_xlabel('Range bin', fontsize=15)
ax.set_ylabel('Doppler bin', fontsize=15)
ax.tick_params(labelsize=13)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=13)
cb.ax.tick_params(labelsize=12)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'rd_map_full.png')
fig.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
print(f"saved: {out2}")
plt.close()

# ====================================================
# 図3: 全体 + ズームのセット（ポスター用パネル）
# ====================================================
fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), constrained_layout=True,
                          gridspec_kw={'width_ratios': [2, 1]})

# 左: 全体マップ
ax = axes[0]
im = ax.imshow(rd_mip_db, aspect='auto', origin='lower',
               cmap='jet', vmin=DB_FLOOR, vmax=0,
               extent=[0, rd_mip.shape[1], 0, rd_mip.shape[0]])
# ズーム範囲をボックスで示す
from matplotlib.patches import Rectangle
rect = Rectangle((r_min, d_min), r_max-r_min, d_max-d_min,
                 linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.set_xlabel('Range bin', fontsize=14)
ax.set_ylabel('Doppler bin', fontsize=14)
ax.set_title('Full Range-Doppler Map', fontsize=15, fontweight='bold')
ax.tick_params(labelsize=12)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=12)

# 右: ズームイン
ax2 = axes[1]
im2 = ax2.imshow(rd_zoom_db, aspect='auto', origin='lower',
                 cmap='jet', vmin=DB_FLOOR, vmax=0,
                 extent=[r_min, r_max, d_min, d_max])
ax2.set_xlabel('Range bin', fontsize=14)
ax2.set_ylabel('Doppler bin', fontsize=14)
ax2.set_title('Zoomed view', fontsize=15, fontweight='bold')
ax2.tick_params(labelsize=12)
cb2 = plt.colorbar(im2, ax=ax2)
cb2.set_label('Normalized power [dB]', fontsize=12)

out3 = os.path.join(OUTPUT_DIR, 'rd_map_fullzoom.png')
fig.savefig(out3, dpi=200, bbox_inches='tight', facecolor='white')
print(f"saved: {out3}")
plt.close()

print("\n=== 生成完了 ===")
print(f"  rd_map_zoom.png     - ターゲット近傍ズーム（推奨・ポスターメイン）")
print(f"  rd_map_full.png     - 全体マップ（-40 dB フロア）")
print(f"  rd_map_fullzoom.png - 全体 + ズームのセットパネル")
print(f"\n強度差: {best['diff_dB']:+.1f} dB (vehicle - cyclist)")
