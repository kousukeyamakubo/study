# eval_nn — NN検出器の評価

## 目的

`best_detector_narrow_angle.pt`（narrow角度グリッド対応モデル）の検出性能を定量評価する。
CFAR比較は `../eval_cfar/` を参照。

## スクリプト一覧

| スクリプト | 内容 |
|---|---|
| `pr_curve.py` | PR曲線・AP計算（narrow角度データセット） |
| `tolerance_sweep.py` | tolerance (d_tol, r_tol, a_tol) を段階的に変化させてAPを評価 |
| `tolerance_sweep_wide_best.py` | wide角度ベストモデルとのtolerance比較（参照用） |
| `overfitting.py` | train/val学習曲線・generalization gap分析 |

## 実行方法

```bash
cd experiments/eval_nn
python pr_curve.py
python tolerance_sweep.py
python overfitting.py
```

## 出力

| フォルダ | 内容 |
|---|---|
| `pr_curve_results/` | PR曲線PNG・メトリクスJSON |
| `tolerance_sweep_results/` | tolerance別AP・論文用図版（`for_paper_pm5deg/`） |
| `overfitting_results/` | 学習曲線・generalization gap PNG |

## 依存データセット

- `learn_dataset_narrow_angle_fixed/` — 2物体固定角度（テスト用）
- `learn_dataset_narrow_angle_single/` — 単一物体（train/val）
- `best_detector_narrow_angle.pt` — 評価対象モデル
