import pandas as pd, numpy as np, os
meta = pd.read_csv('./learn_dataset_narrow_angle_fixed/metadata.csv')
print('columns:', meta.columns.tolist())
print('rows:', len(meta))
raw = meta.iloc[0]['file']
print('raw path:', raw)
# ./learn_dataset_... 形式を想定
path = raw.replace('\\', os.sep)
if path.startswith('.' + os.sep):
    path = path[2:]
print('resolved:', path)
data = np.load(path, allow_pickle=True)
print('rd_maps shape:', data['rd_maps'].shape)
if 'fixed_angles' in data:
    print('fixed_angles:', data['fixed_angles'])
print('valid_all unique:', meta['valid_all'].unique() if 'valid_all' in meta.columns else 'N/A')
