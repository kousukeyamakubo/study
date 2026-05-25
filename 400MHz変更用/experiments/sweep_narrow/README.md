# sweep_narrow — narrow角度モデルのパラメータ探索

## 目的

narrow角度グリッド（±5°）対応モデルの学習ハイパーパラメータを探索した実験群。

## サブフォルダ一覧

| フォルダ | 内容 |
|---|---|
| `loss_sweep_narrow/` | Focal Loss のγ・α を格子探索。`loss_sweep_narrow_v2.py` が最新版。 |
| `sweep_v4_soft_daware/` | soft distance-aware 損失のスイープ（fp_weight パラメータ探索）。最新スイープ版。 |

## 結果の見方

各サブフォルダの `*_results/` 以下にヒートマップ・CSVが保存されている。
最良パラメータは `best_detector_narrow_angle.pt` として別途保存済み。
