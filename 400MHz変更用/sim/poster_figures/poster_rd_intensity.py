"""
ポスター用: RDマップ上でサイクリストと車両の強度差を可視化するスクリプト

各ターゲットの真の角度に最も近い角度チャネルのRDマップを使用し、
サイクリストと車両でピーク強度が大きく異なることを示す。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

FIXED_ANGLES = np.linspace(-5, 5, 10)  # 角度グリッド [度]

def load_best_channel(row, scenario):
    """ターゲット真値角度に最も近い角度チャネルを返す"""
    path = row['file'].replace('\\', '/').lstrip('./')
    data = np.load(path, allow_pickle=True)
    if scenario == 'cyclist':
        true_angle = float(row['cyclist_true_angle_deg'])
        d_idx = int(row['cyclist_true_d_idx'])
        r_idx = int(row['cyclist_true_r_idx'])
    else:
        true_angle = float(row['vehicle_true_angle_deg'])
        d_idx = int(row['vehicle_true_d_idx'])
        r_idx = int(row['vehicle_true_r_idx'])
    ch = int(np.argmin(np.abs(FIXED_ANGLES - true_angle)))
    rd_complex = data['rd_maps'][ch]  # (H=89, W=190) complex
    rd_amp = np.abs(rd_complex)
    return rd_amp, ch, d_idx, r_idx, true_angle

# ---- データ読み込み ----
meta = pd.read_csv('./learn_dataset_single_object/metadata.csv')
cy_rows = meta[(meta['scenario'] == 'cyclist_only') & (meta['valid_cyclist'].astype(str) == 'True')]
ve_rows = meta[(meta['scenario'] == 'vehicle_only') & (meta['valid_vehicle'].astype(str) == 'True')]

# peak@true位置が大きいサンプルを選ぶ
best_cy_peak, best_cy_row = 0, None
for _, row in cy_rows.iterrows():
    rd, ch, d, r, ang = load_best_channel(row, 'cyclist')
    if rd[d, r] > best_cy_peak:
        best_cy_peak = rd[d, r]
        best_cy_row = (row, rd, ch, d, r, ang)

best_ve_peak, best_ve_row = 0, None
for _, row in ve_rows.iterrows():
    rd, ch, d, r, ang = load_best_channel(row, 'vehicle')
    if rd[d, r] > best_ve_peak:
        best_ve_peak = rd[d, r]
        best_ve_row = (row, rd, ch, d, r, ang)

cy_row, cy_rd, cy_ch, cy_d, cy_r, cy_ang = best_cy_row
ve_row, ve_rd, ve_ch, ve_d, ve_r, ve_ang = best_ve_row

db_diff = 20 * np.log10(ve_rd[ve_d, ve_r] / cy_rd[cy_d, cy_r])
print(f"Cyclist  peak={cy_rd[cy_d,cy_r]:.4e}  ch={cy_ch}({FIXED_ANGLES[cy_ch]:.1f}°) d={cy_d} r={cy_r}")
print(f"Vehicle  peak={ve_rd[ve_d,ve_r]:.4e}  ch={ve_ch}({FIXED_ANGLES[ve_ch]:.1f}°) d={ve_d} r={ve_r}")
print(f"Peak intensity diff: {db_diff:.1f} dB  (vehicle is stronger)")

# ---- dB変換（両マップの最大値でそれぞれ正規化して見やすくする） ----
def normalize_db(rd, vmax=None, db_floor=-30):
    if vmax is None:
        vmax = rd.max()
    eps = vmax * 1e-7
    db = 20 * np.log10((rd + eps) / (vmax + eps))
    return np.clip(db, db_floor, 0)

# 同一スケールで比較するために共通 vmax を使う
common_vmax = max(cy_rd.max(), ve_rd.max())
cy_db = normalize_db(cy_rd, vmax=common_vmax, db_floor=-30)
ve_db = normalize_db(ve_rd, vmax=common_vmax, db_floor=-30)

# =========================================================
# 図1: シンプルな2パネル並列表示（ポスターメイン図）
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
fig.patch.set_facecolor('white')

cmap = 'jet'
vmin_db, vmax_db = -30, 0

panel_cfg = [
    (axes[0], cy_db, cy_d, cy_r, cy_ang, 'Cyclist',  'dodgerblue'),
    (axes[1], ve_db, ve_d, ve_r, ve_ang, 'Vehicle',   'tomato'),
]

for ax, rd_db, d_idx, r_idx, ang, title, color in panel_cfg:
    im = ax.imshow(rd_db, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin_db, vmax=vmax_db,
                   extent=[0, rd_db.shape[1], 0, rd_db.shape[0]])

    # ターゲット位置マーカー
    ax.plot(r_idx + 0.5, d_idx + 0.5, 'w^',
            markersize=13, markeredgecolor='black', markeredgewidth=1.5, zorder=5,
            label=f'{title} ({ang:.1f}°)')

    # ピーク強度アノテーション
    peak_db = rd_db[d_idx, r_idx]
    ax.annotate(f'{peak_db:.1f} dB', xy=(r_idx + 0.5, d_idx + 0.5),
                xytext=(r_idx + 12, d_idx + 8),
                color='white', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax.set_title(title, fontsize=20, fontweight='bold', color=color, pad=8)
    ax.set_xlabel('Range bin', fontsize=14)
    ax.set_ylabel('Doppler bin', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.7)
    cb = plt.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('Normalized power [dB]', fontsize=12)

# 強度差アノテーションを右パネルに追加
axes[1].annotate(
    f'Peak diff: {db_diff:+.1f} dB\nvs. cyclist',
    xy=(0.03, 0.97), xycoords='axes fraction',
    ha='left', va='top', fontsize=13, fontweight='bold',
    color='white',
    bbox=dict(boxstyle='round,pad=0.4', fc='black', alpha=0.5, ec='white')
)

fig.suptitle('Range-Doppler Maps: Cyclist vs. Vehicle\n(same power normalization)',
             fontsize=16, fontweight='bold')

out1 = 'poster_rd_side_by_side.png'
fig.savefig(out1, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out1}")
plt.close()

# =========================================================
# 図2: 3パネル（RDマップ2枚 + レンジプロファイル比較）
# =========================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
fig2.patch.set_facecolor('white')

# --- パネル1,2: RDマップ ---
for ax, rd_db, d_idx, r_idx, ang, title, color in [
    (axes2[0], cy_db, cy_d, cy_r, cy_ang, 'Cyclist', 'dodgerblue'),
    (axes2[1], ve_db, ve_d, ve_r, ve_ang, 'Vehicle',  'tomato'),
]:
    im = ax.imshow(rd_db, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin_db, vmax=vmax_db,
                   extent=[0, rd_db.shape[1], 0, rd_db.shape[0]])
    ax.plot(r_idx + 0.5, d_idx + 0.5, 'w^',
            markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_title(title, fontsize=18, fontweight='bold', color=color)
    ax.set_xlabel('Range bin', fontsize=13)
    ax.set_ylabel('Doppler bin', fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.88)
    cb.set_label('Normalized power [dB]', fontsize=11)

# --- パネル3: ドップラービン行のレンジプロファイル ---
ax3 = axes2[2]
r_axis = np.arange(cy_rd.shape[1])
ax3.plot(r_axis, cy_db[cy_d, :], color='dodgerblue', linewidth=2.5, label='Cyclist')
ax3.plot(r_axis, ve_db[ve_d, :], color='tomato',     linewidth=2.5, label='Vehicle')
ax3.axvline(cy_r, color='dodgerblue', linestyle='--', alpha=0.8, linewidth=1.5)
ax3.axvline(ve_r, color='tomato',     linestyle='--', alpha=0.8, linewidth=1.5)

# ピーク位置にアノテーション
cy_peak_db = cy_db[cy_d, cy_r]
ve_peak_db = ve_db[ve_d, ve_r]
ax3.annotate(f'{cy_peak_db:.1f} dB', xy=(cy_r, cy_peak_db),
             xytext=(cy_r + 10, cy_peak_db + 2), color='dodgerblue', fontsize=12, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='dodgerblue'))
ax3.annotate(f'{ve_peak_db:.1f} dB', xy=(ve_r, ve_peak_db),
             xytext=(ve_r + 10, ve_peak_db + 2), color='tomato', fontsize=12, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='tomato'))

ax3.set_xlim(0, cy_rd.shape[1])
ax3.set_ylim(vmin_db - 2, 5)
ax3.set_xlabel('Range bin', fontsize=13)
ax3.set_ylabel('Normalized power [dB]', fontsize=13)
ax3.set_title('Range profile at true Doppler bin', fontsize=14, fontweight='bold')
ax3.legend(fontsize=13)
ax3.grid(True, alpha=0.35)
ax3.annotate(f'Δ = {db_diff:+.1f} dB',
             xy=(0.97, 0.05), xycoords='axes fraction',
             ha='right', va='bottom', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray'))

fig2.suptitle('Range-Doppler Map: Intensity Difference between Cyclist and Vehicle',
              fontsize=16, fontweight='bold')

out2 = 'poster_rd_intensity_comparison.png'
fig2.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out2}")
plt.close()

# =========================================================
# 図3: ズームイン比較（ターゲット近傍のみ）
# =========================================================
PAD_D, PAD_R = 20, 30  # ズーム範囲

def get_zoom(rd, d, r, pad_d=PAD_D, pad_r=PAD_R):
    d0 = max(0, d - pad_d); d1 = min(rd.shape[0], d + pad_d)
    r0 = max(0, r - pad_r); r1 = min(rd.shape[1], r + pad_r)
    return rd[d0:d1, r0:r1], d0, r0

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig3.patch.set_facecolor('white')

for ax, rd_db, d_idx, r_idx, ang, title, color in [
    (axes3[0], cy_db, cy_d, cy_r, cy_ang, 'Cyclist', 'dodgerblue'),
    (axes3[1], ve_db, ve_d, ve_r, ve_ang, 'Vehicle',  'tomato'),
]:
    zoomed, d0, r0 = get_zoom(rd_db, d_idx, r_idx)
    ext = [r0, r0 + zoomed.shape[1], d0, d0 + zoomed.shape[0]]
    im = ax.imshow(zoomed, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin_db, vmax=vmax_db, extent=ext)
    ax.plot(r_idx + 0.5, d_idx + 0.5, 'w^',
            markersize=14, markeredgecolor='black', markeredgewidth=2.0, zorder=5)
    peak_db = rd_db[d_idx, r_idx]
    ax.set_title(f'{title}  (peak = {peak_db:.1f} dB)', fontsize=16, fontweight='bold', color=color)
    ax.set_xlabel('Range bin', fontsize=13)
    ax.set_ylabel('Doppler bin', fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.88)
    cb.set_label('Normalized power [dB]', fontsize=12)

fig3.suptitle(f'Range-Doppler Map (zoomed): Cyclist vs. Vehicle\nPeak intensity difference: {db_diff:+.1f} dB',
              fontsize=15, fontweight='bold')

out3 = 'poster_rd_zoom_comparison.png'
fig3.savefig(out3, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out3}")
plt.close()

print("\nAll figures saved successfully.")
print(f"  poster_rd_side_by_side.png      — シンプル2パネル（メイン推奨）")
print(f"  poster_rd_intensity_comparison.png — 3パネル（プロファイル付き）")
print(f"  poster_rd_zoom_comparison.png   — ズームイン2パネル")
