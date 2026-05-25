import numpy as np
import pandas as pd
import os

meta_s = pd.read_csv('./learn_dataset_single_object/metadata.csv')
ve_row = meta_s.iloc[319]
path = ve_row['file'].replace('\\', os.sep).replace('./', '')
data = np.load(path, allow_pickle=True)
rd_stack = np.abs(data['rd_maps'])

print("vehicle_only sample idx=319:")
print(f"  true d={ve_row['vehicle_true_d_idx']} r={ve_row['vehicle_true_r_idx']}")
for ch in range(10):
    d_max, r_max = np.unravel_index(rd_stack[ch].argmax(), rd_stack[ch].shape)
    print(f"  ch={ch}: max={rd_stack[ch].max():.4e} at d={d_max} r={r_max}")

d_t = int(ve_row['vehicle_true_d_idx'])
r_t = int(ve_row['vehicle_true_r_idx'])
print(f"\nvalue at true pos (d={d_t}, r={r_t}) per channel:")
for ch in range(10):
    print(f"  ch={ch}: {rd_stack[ch, d_t, r_t]:.4e}")

# 全体の最大値はどこか
g_max = rd_stack.max()
g_ch, g_d, g_r = np.unravel_index(rd_stack.argmax(), rd_stack.shape)
print(f"\nGlobal max: {g_max:.4e} at ch={g_ch} d={g_d} r={g_r}")
