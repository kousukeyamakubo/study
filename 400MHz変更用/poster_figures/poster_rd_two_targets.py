"""
ポスター用: サイクリストと車両が同時に存在するRDマップを生成する

learn_dataset_fixed_angle から両ターゲットが有効なサンプルを選び、
単一のRDマップ上でサイクリスト・車両の強度差を可視化する。
マーカーや注釈は一切入れない（パワポで後から追記）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUTPUT_DIR = 'poster_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_ANGLES = np.linspace(-5, 5, 10)

# ---- データ読み込み ----
meta = pd.read_csv('./learn_dataset_fixed_angle/metadata.csv')
valid = meta[
    (meta['valid_cyclist'].astype(bool)) &
    (meta['valid_vehicle'].astype(bool))
].reset_index(drop=True)
print(f"両方有効なサンプル数: {len(valid)}")

# サイクリストと車両のピーク強度差が大きいサンプルを探す
best_sample = None
best_diff = -np.inf

def neighborhood_peak(rd_ch, d_center, r_center, pad_d=8, pad_r=8):
    """真値位置近傍でのピーク値と位置を返す"""
    H, W = rd_ch.shape
    d0 = max(0, d_center - pad_d); d1 = min(H, d_center + pad_d + 1)
    r0 = max(0, r_center - pad_r); r1 = min(W, r_center + pad_r + 1)
    patch = rd_ch[d0:d1, r0:r1]
    local_d, local_r = np.unravel_index(patch.argmax(), patch.shape)
    return patch.max(), d0 + local_d, r0 + local_r

for _, row in valid.iterrows():
    path = row['file'].replace('\\', '/').lstrip('./')
    data = np.load(path, allow_pickle=True)
    rd_stack = np.abs(data['rd_maps'])  # (10, H, W)

    cy_ang = float(row['cyclist_true_angle_deg'])
    ve_ang = float(row['vehicle_true_angle_deg'])
    cy_ch = int(np.argmin(np.abs(FIXED_ANGLES - cy_ang)))
    ve_ch = int(np.argmin(np.abs(FIXED_ANGLES - ve_ang)))

    cy_d_nom = int(row['cyclist_true_d_idx'])
    cy_r_nom = int(row['cyclist_true_r_idx'])
    ve_d_nom = int(row['vehicle_true_d_idx'])
    ve_r_nom = int(row['vehicle_true_r_idx'])

    # 真値近傍の実際のピークを探す
    cy_peak, cy_d, cy_r = neighborhood_peak(rd_stack[cy_ch], cy_d_nom, cy_r_nom)
    ve_peak, ve_d, ve_r = neighborhood_peak(rd_stack[ve_ch], ve_d_nom, ve_r_nom)

    if cy_peak > 0 and ve_peak > 0:
        diff = abs(20 * np.log10(ve_peak / cy_peak))
        if diff > best_diff:
            best_diff = diff
            best_sample = {
                'row': row,
                'rd_stack': rd_stack,
                'cy_ch': cy_ch, 've_ch': ve_ch,
                'cy_d': cy_d, 'cy_r': cy_r,
                've_d': ve_d, 've_r': ve_r,
                'cy_peak': cy_peak, 've_peak': ve_peak,
                'diff_dB': 20 * np.log10(ve_peak / cy_peak),
            }

s = best_sample
print(f"選択サンプル: {s['row']['file']}")
print(f"cyclist  ch={s['cy_ch']}({FIXED_ANGLES[s['cy_ch']]:.1f}°) d={s['cy_d']} r={s['cy_r']} peak={s['cy_peak']:.4e}")
print(f"vehicle  ch={s['ve_ch']}({FIXED_ANGLES[s['ve_ch']]:.1f}°) d={s['ve_d']} r={s['ve_r']} peak={s['ve_peak']:.4e}")
print(f"vehicle - cyclist: {s['diff_dB']:+.1f} dB")

# ---- 可視化設定 ----
# サイクリストと車両の各最適角度チャネルのRDマップを取得
cy_rd = s['rd_stack'][s['cy_ch']]  # (H=89, W=190)
ve_rd = s['rd_stack'][s['ve_ch']]  # (H=89, W=190)

# 同一チャネルで両方が見えるチャネルを探す（両ピークが最大公約的に見えるチャネル）
# → 全チャネル合成（最大値投影）か、中央チャネルか選ぶ
# 今回は両ターゲットが見える「最大値投影マップ」で表示
rd_mip = s['rd_stack'].max(axis=0)  # (H, W): 全角度チャネルの最大値

def to_db(rd, vmax=None, db_floor=-30):
    """dBスケールへ変換（vmax で正規化）"""
    if vmax is None:
        vmax = rd.max()
    eps = vmax * 1e-8
    db = 20 * np.log10((rd + eps) / (vmax + eps))
    return np.clip(db, db_floor, 0), vmax

rd_mip_db, vmax = to_db(rd_mip, db_floor=-30)
cy_rd_db, _ = to_db(cy_rd, vmax=vmax, db_floor=-30)
ve_rd_db, _ = to_db(ve_rd, vmax=vmax, db_floor=-30)

# ====================================================
# 図A: 最大値投影RDマップ（1枚でboth見える・ポスターメイン）
# ====================================================
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(rd_mip_db, aspect='auto', origin='lower',
               cmap='jet', vmin=-30, vmax=0,
               extent=[0, rd_mip.shape[1], 0, rd_mip.shape[0]])
ax.set_xlabel('Range bin', fontsize=14)
ax.set_ylabel('Doppler bin', fontsize=14)
ax.tick_params(labelsize=12)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=13)
cb.ax.tick_params(labelsize=11)
plt.tight_layout()
out_a = os.path.join(OUTPUT_DIR, 'rd_map_both_targets.png')
fig.savefig(out_a, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out_a}")
plt.close()

# ====================================================
# 図B: サイクリスト最適チャネル RDマップ（単体）
# ====================================================
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(cy_rd_db, aspect='auto', origin='lower',
               cmap='jet', vmin=-30, vmax=0,
               extent=[0, cy_rd.shape[1], 0, cy_rd.shape[0]])
ax.set_xlabel('Range bin', fontsize=14)
ax.set_ylabel('Doppler bin', fontsize=14)
ax.tick_params(labelsize=12)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=13)
cb.ax.tick_params(labelsize=11)
plt.tight_layout()
out_b = os.path.join(OUTPUT_DIR, 'rd_map_cyclist_channel.png')
fig.savefig(out_b, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out_b}")
plt.close()

# ====================================================
# 図C: 車両最適チャネル RDマップ（単体）
# ====================================================
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(ve_rd_db, aspect='auto', origin='lower',
               cmap='jet', vmin=-30, vmax=0,
               extent=[0, ve_rd.shape[1], 0, ve_rd.shape[0]])
ax.set_xlabel('Range bin', fontsize=14)
ax.set_ylabel('Doppler bin', fontsize=14)
ax.tick_params(labelsize=12)
cb = plt.colorbar(im, ax=ax)
cb.set_label('Normalized power [dB]', fontsize=13)
cb.ax.tick_params(labelsize=11)
plt.tight_layout()
out_c = os.path.join(OUTPUT_DIR, 'rd_map_vehicle_channel.png')
fig.savefig(out_c, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out_c}")
plt.close()

# ====================================================
# 図D: 各ターゲット角度チャネルのRDマップを横並び
# ====================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

labels = [
    (axes[0], cy_rd_db, s['cy_d'], s['cy_r'], f'Cyclist (beam {FIXED_ANGLES[s["cy_ch"]]:.1f}°)'),
    (axes[1], ve_rd_db, s['ve_d'], s['ve_r'], f'Vehicle (beam {FIXED_ANGLES[s["ve_ch"]]:.1f}°)'),
]
for ax, rd_db, d, r, title in labels:
    im = ax.imshow(rd_db, aspect='auto', origin='lower',
                   cmap='jet', vmin=-30, vmax=0,
                   extent=[0, rd_db.shape[1], 0, rd_db.shape[0]])
    ax.set_xlabel('Range bin', fontsize=14)
    ax.set_ylabel('Doppler bin', fontsize=14)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Normalized power [dB]', fontsize=12)
    cb.ax.tick_params(labelsize=11)

out_d = os.path.join(OUTPUT_DIR, 'rd_map_side_by_side.png')
fig.savefig(out_d, dpi=180, bbox_inches='tight', facecolor='white')
print(f"saved: {out_d}")
plt.close()

print(f"\n=== 生成完了 ===")
print(f"フォルダ: {OUTPUT_DIR}/")
print(f"  rd_map_both_targets.png    - 全角度MIPで両ターゲットが見える1枚")
print(f"  rd_map_cyclist_channel.png - cyclist最適チャネル")
print(f"  rd_map_vehicle_channel.png - vehicle最適チャネル")
print(f"  rd_map_side_by_side.png    - 2枚横並び（チャネル名付きタイトル）")
print(f"\n強度差: {s['diff_dB']:+.1f} dB (vehicle - cyclist)")
