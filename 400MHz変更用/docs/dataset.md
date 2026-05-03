# データセット仕様

## フォルダ構成

### モデル学習・評価に使えるデータセット（N_FIXED=10, 推奨）

| フォルダ | 内容 | サンプル数 | rd_maps.shape |
|---|---|---|---|
| `learn_dataset_single_object/` | 単一物体（cy-only / ve-only 混在） | 600件 | (10, 89, 190) |
| `learn_dataset_fixed_angle/` | 2物体同時（固定角度グリッド対応） | 約300件 | (10, 89, 190) |
| `learn_dataset_scenario_test/` | シナリオテスト（r_diff を段階的に変化） | 31件 | (10, 89, 190) |

### 旧フォーマット（**モデル評価には使用不可**）

| フォルダ | 内容 | rd_maps.shape | 問題 |
|---|---|---|---|
| `learn_dataset_ped_single/` | cyclist のみ（旧生成） | **(1, 89, 190)** | 角度1本。現行モデル(N_FIXED=10)と非互換 |
| `learn_dataset_vehicle_single/` | vehicle のみ（旧生成） | **(1, 89, 190)** | 同上 |

> **注意**: `ped_single` / `vehicle_single` は CSV カラム構成も異なる（`valid_cyclist` 等がなく `valid` のみ）。
> 現行モデルへの入力として使うと全サンプルが背景判定になる（0% 検出）。
> 単一ターゲットテストには `learn_dataset_single_object` を使うこと。

## npz ファイル形式

### 推奨フォーマット（single_object / fixed_angle / scenario_test）

各サンプルは `.npz` ファイル。主要なキー:

| キー | 型 | 内容 |
|---|---|---|
| `rd_maps` | complex64, **(N_FIXED, H, W)** | 各角度の RD マップ。N_FIXED=10, H=89, W=190 |
| `fixed_angles` | float, (N_FIXED,) | 角度グリッド [度] |
| `cyclist_true_angle_deg` | float | サイクリスト真の角度 |
| `vehicle_true_angle_deg` | float | 車両真の角度 |

### 旧フォーマット（ped_single / vehicle_single）— 参考のみ

| キー | 型 | 内容 |
|---|---|---|
| `rd_maps` | complex64, **(1, H, W)** | 角度 1 本のみ。現行モデルと非互換 |
| `true_d_idx` / `true_r_idx` | int | ターゲット位置（cyclist/vehicle の区別なし） |
| `valid` | bool | 有効フラグ（`valid_cyclist` / `valid_vehicle` はなし） |

## metadata.csv のカラム

### 推奨フォーマット（single_object / fixed_angle / scenario_test）

| カラム | 内容 |
|---|---|
| `file` | npz ファイルパス |
| `scenario` | `cyclist_only` / `vehicle_only` / `both` |
| `cyclist_true_d_idx` / `cyclist_true_r_idx` | サイクリストのドップラー・レンジ bin（真値） |
| `vehicle_true_d_idx` / `vehicle_true_r_idx` | 車両の同上 |
| `valid_cyclist` / `valid_vehicle` | 各物体が有効かどうか（True/1） |
| `valid_all` | 対象物体が全て有効なサンプルのフラグ（学習フィルタ用） |

### 旧フォーマット（ped_single / vehicle_single）— 参考のみ

| カラム | 内容 |
|---|---|
| `file` | npz ファイルパス |
| `target_kind` | `cyclist` / `vehicle` |
| `true_d_idx` / `true_r_idx` | ターゲット位置 |
| `valid` | 有効フラグ（`valid_cyclist` / `valid_vehicle` はなし） |

## 学習・評価データ分割

```
学習: 単一物体 280件 + 2物体 170件 = 450件  (single_object[0:280] + fixed_angle[0:170])
検証:  単一物体  60件 + 2物体  30件 =  90件  (single_object[280:340] + fixed_angle[170:200])
テスト: 2物体 100件                           (fixed_angle[200:])
holdout（単一物体）: 260件                    (single_object[340:], RANDOM_SEED=42 でシャッフル後)
```

`holdout` は学習・検証いずれにも使われていない。
単一ターゲットの誤検出テスト（`run_single_target_test_v2`）はこの260件を使う。

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
