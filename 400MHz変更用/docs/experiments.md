# 実験結果・現状の課題

## 現在の最良結果 (models/best_detector_narrow_angle.pt)

学習設定: 単一物体 280件 + 2物体 170件（train）/ 単一物体 60件 + 2物体 30件（val）/ 2物体 100件（test）
データセット: `learn_dataset_narrow_angle_single` + `learn_dataset_narrow_angle_fixed`

詳細な定量評価は `experiments/eval_nn/` を参照。

---

## 旧最良結果 (models/best_detector_agnostic.pt)

学習設定: 単一物体 280件 + 2物体 200件（train）/ 単一物体 60件（val）/ 2物体 100件（test）

| 指標 | 値 |
|---|---|
| test_loss | 0.8978 |
| cyclist_hit | 0.980 (98%) |
| vehicle_hit | 0.920 (92%) |

## シナリオテスト結果

r_diff（サイクリスト・車両のレンジビン差）を段階的に変化させたテスト（30ステップ）。

### 検出数の内訳

| 検出数 | 件数 | 解釈 |
|---|---|---|
| n=0 (両方見逃し) | — | 完全失敗 |
| n=1 (片方見逃し) | 多い | r_diff 小さい領域で発生 |
| n=2 (正常) | — | r_diff が十分大きい領域 |
| n≥3 (FP あり) | 少数 | 誤検出あり |

### 観察された課題

1. **r_diff が小さい領域（~5ビン以下）での分離失敗**: 2物体が RD マップ上で重なると n=1 になりやすい
2. **角度チャネルの混同**: 同一空間位置に複数の角度チャネルでピークが出る（FP の一因）
3. **val が単一物体のみ**: 2物体シナリオの汎化が検証できていない

### 検出の傾向

- r_diff > 10 ビン: ほぼ全て n=2 で正確に分離できる
- r_diff < 5 ビン: n=1 が多発。どちらかの物体に引き寄せられる
- 角度推定: ch=6 (1.7deg) と ch=7 (2.8deg) に多く検出される（真値の角度分布に対応）

## 過去の実験経緯

| ブランチ / 実験 | 内容 | 結果 |
|---|---|---|
| BCE のみ | 単純な BCE 損失 | 不安定 |
| BCE + Dice | 現在の構成に戻す | 安定 |
| σ_r=1.0 + NMS 縮小 | ラベル設計実験 | — |
| 角度軸追加 (10ch出力・3D NMS) | 現在の主軸 | hit rate 向上 |

## 実験フォルダ運用ルール

新しい実験（スクリプト + 結果）は必ず `experiments/` 以下に専用フォルダを作成して格納する。

### ディレクトリ構造

```
experiments/
├── eval_cfar/        ← CA-CFAR vs NN の PR曲線・AP比較
├── eval_nn/          ← narrow角度グリッドモデル (best_detector_narrow_angle.pt) の定量評価
├── sweep_narrow/     ← narrow角度グリッドモデルのハイパーパラメータ探索
├── sweep_wide/       ← wide角度モデルの探索（narrow移行前）
└── vehicle_ablation/ ← 車両検出 FP 分析・損失設計根拠
```

各フォルダの内部構造:

```
experiments/<カテゴリ>/
├── README.md               ← 目的・手順の説明
├── <実験スクリプト>.py     ← 実行・評価スクリプト
└── <実験名>_results/       ← 出力ファイル（画像・CSV・JSON など）
```

### 命名規則

| 対象 | 規則 | 例 |
|---|---|---|
| 実験フォルダ | snake_case | `cfar_param_sweep/` |
| 結果サブフォルダ | `<実験フォルダ名>_results/` | `cfar_param_sweep_results/` |
| スクリプト | snake_case、フォルダ名と揃える | `cfar_param_sweep.py` |

### パス記述

スクリプト内のパスはスクリプト自身の位置を基準にした相対パスで記述する。

```python
# experiments/cfar_param_sweep/cfar_param_sweep.py の場合
MIXED_META_CSV = "../../learn_dataset_fixed_angle/metadata.csv"
OUTPUT_DIR     = "./cfar_param_sweep_results"
```

### 注意事項

- ルートや他の実験フォルダに直接スクリプト・結果を置かない
- 複数の実験で共有するデータセット（`learn_dataset_*/`）はルートに残す
- モデル重み（`best_detector_*.pt`）もルートに残す

## 今後の検討事項

- r_diff が小さい場合の分離精度向上（データ拡張？ラベル設計？）
- val セットに 2物体データを追加
- モンテカルロ評価での CFAR との定量比較（`monte-carlo-simulation.ipynb`）
- 角度推定精度の定量評価（現状は d/r の hit rate のみ）
