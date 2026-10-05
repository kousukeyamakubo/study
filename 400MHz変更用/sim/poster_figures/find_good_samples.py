"""
ポスター用RDマップ: サイクリストと車両の代表サンプルを探す
- 各サンプルの全角度チャネルから最大ピークを探す
- マップ上の最強反射点での強度を比較する
"""
import numpy as np
import pandas as pd
import os

FIXED_ANGLES = np.linspace(-5, 5, 10)

meta = pd.read_csv('./learn_dataset_single_object/metadata.csv')
cy_rows = meta[meta['scenario'] == 'cyclist_only']
ve_rows = meta[meta['scenario'] == 'vehicle_only']

results = []
for label, rows in [('cyclist', cy_rows), ('vehicle', ve_rows)]:
    for _, row in rows.iterrows():
        path = row['file'].replace('\\', '/').lstrip('./')
        data = np.load(path, allow_pickle=True)
        rd_stack = np.abs(data['rd_maps'])  # (10, 89, 190)

        # 全チャネルにわたる最大値
        global_max = rd_stack.max()
        best_ch = int(np.argmax(rd_stack.max(axis=(1,2))))
        d_peak, r_peak = np.unravel_index(rd_stack[best_ch].argmax(), rd_stack[best_ch].shape)

        results.append({
            'label': label,
            'idx': row.name,
            'best_ch': best_ch,
            'best_angle': FIXED_ANGLES[best_ch],
            'd_peak': int(d_peak),
            'r_peak': int(r_peak),
            'global_max': global_max,
        })

df = pd.DataFrame(results)
print("=== 全サンプルの最大ピーク統計 ===")
print(df.groupby('label')['global_max'].describe())
print()
print("Top 5 cyclist:")
print(df[df['label']=='cyclist'].nlargest(5, 'global_max')[['idx','best_ch','best_angle','d_peak','r_peak','global_max']])
print()
print("Top 5 vehicle:")
print(df[df['label']=='vehicle'].nlargest(5, 'global_max')[['idx','best_ch','best_angle','d_peak','r_peak','global_max']])
