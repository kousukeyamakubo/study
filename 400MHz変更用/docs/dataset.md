# データセット仕様

## フォルダ構成

| フォルダ | 内容 | サンプル数 |
|---|---|---|
| `learn_dataset_single_object/` | 単一物体（サイクリストのみ or 車両のみ） | 約600件 |
| `learn_dataset_fixed_angle/` | 2物体同時（固定角度グリッド対応） | 約300件 |
| `learn_dataset_scenario_test/` | シナリオテスト（r_diff を段階的に変化） | 30件 |

## npz ファイル形式

各サンプルは `.npz` ファイル。主要なキー:

| キー | 型 | 内容 |
|---|---|---|
| `rd_maps` | complex, (N_FIXED, H, W) | 各角度の RD マップ |
| `fixed_angles` | float, (N_FIXED,) | 角度グリッド [度] |
| `cyclist_true_angle_deg` | float | サイクリスト真の角度 |
| `vehicle_true_angle_deg` | float | 車両真の角度 |

## metadata.csv のカラム

| カラム | 内容 |
|---|---|
| `file` | npz ファイルパス |
| `cyclist_true_d_idx` | ドップラービンインデックス（真値） |
| `cyclist_true_r_idx` | レンジビンインデックス（真値） |
| `vehicle_true_d_idx` / `vehicle_true_r_idx` | 同上（車両） |
| `valid_cyclist` / `valid_vehicle` | 各物体が有効かどうか |
| `valid_all` | 両方有効なサンプルのフラグ |

## 学習・評価データ分割

```
学習: 単一物体 280件 + 2物体 200件 = 480件
検証: 単一物体 60件
テスト: 2物体 100件（fixed_angle データセットの後半）
```

## ヒートマップラベル生成

各角度チャネル `i` のラベルは:
```
y[i] = clip(w_cy * heatmap_cy + w_ve * heatmap_ve, 0, 1)
w_cy = exp(-0.5 * ((angle_i - cy_true_angle) / σ_angle)^2)
```
- σ_angle = 0.3 度
- heatmap は (d_center, r_center) を中心とした 2D Gaussian (σ_d=3.0, σ_r=1.0)

## データ生成

`learn_data_generator.ipynb` でデータ生成。
シナリオテストデータは r_diff を連続的に変化させた固定シナリオ。
