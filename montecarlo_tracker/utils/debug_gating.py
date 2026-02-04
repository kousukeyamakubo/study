# utils/debug_gating.py として作成
import pandas as pd
import numpy as np
from config import Config
from core.models import get_measurement_model

cfg = Config()
est = pd.read_csv('csv_result/estimated_trajectory.csv')
meas = pd.read_csv('csv_result/measurements.csv')
true = pd.read_csv('csv_result/true_trajectory.csv')

# 各時刻でのトラック数とIDを確認
print("=== 時刻ごとのトラック数 ===")
for t in range(min(10, cfg.num_steps)):
    est_t = est[est['time'] == t]
    meas_t = meas[meas['time'] == t]
    true_t = true[true['time'] == t]
    print(f"t={t}: 真値={len(true_t)}個, 観測={len(meas_t)}個, 推定トラック={len(est_t)}個 (ID: {est_t['target_id'].tolist()})")

# ID 2, 3 がいつ生成されたか
print("\n=== 新規トラック生成タイミング ===")
for track_id in [2, 3]:
    first_appear = est[est['target_id'] == track_id]['time'].min()
    print(f"Track ID {track_id} は t={first_appear} で初めて出現")
    
# その時刻の詳細を確認
for track_id in [2, 3]:
    t = est[est['target_id'] == track_id]['time'].min()
    print(f"\n=== t={t} の状況 (Track {track_id} 生成) ===")
    print(f"観測数: {len(meas[meas['time'] == t])}個")
    print(f"既存トラック数: {len(est[est['time'] == t-1]) if t > 0 else 0}個")
    print("観測位置:")
    print(meas[meas['time'] == t][['x', 'y', 'velocity']])
    if t > 0:
        print("\n既存トラックの予測位置 (t-1の推定):")
        print(est[est['time'] == t-1][['target_id', 'x', 'y', 'vx', 'vy']])