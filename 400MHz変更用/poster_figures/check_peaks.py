import numpy as np
import pandas as pd
import os

meta = pd.read_csv('./learn_dataset_single_object/metadata.csv')
cy_rows = meta[(meta['scenario'] == 'cyclist_only') & (meta['valid_cyclist'].astype(str) == 'True')]
ve_rows = meta[(meta['scenario'] == 'vehicle_only') & (meta['valid_vehicle'].astype(str) == 'True')]

results = []
for label, rows, d_col, r_col in [
    ('cyclist', cy_rows, 'cyclist_true_d_idx', 'cyclist_true_r_idx'),
    ('vehicle', ve_rows, 'vehicle_true_d_idx', 'vehicle_true_r_idx'),
]:
    for _, row in rows.iloc[:30].iterrows():
        path = row['file'].replace('\\', '/').lstrip('./')
        data = np.load(path, allow_pickle=True)
        rd = np.abs(data['rd_maps'][5])
        d = int(row[d_col])
        r = int(row[r_col])
        peak = rd[d, r]
        rd_max = rd.max()
        results.append({'label': label, 'd': d, 'r': r, 'peak': peak, 'rd_max': rd_max, 'idx': row.name, 'file': path})

df = pd.DataFrame(results)
print(df.groupby('label')['peak'].describe())
print('\nTop cyclist by peak:')
print(df[df['label']=='cyclist'].nlargest(5, 'peak')[['idx','d','r','peak','rd_max']])
print('\nTop vehicle by peak:')
print(df[df['label']=='vehicle'].nlargest(5, 'peak')[['idx','d','r','peak','rd_max']])
