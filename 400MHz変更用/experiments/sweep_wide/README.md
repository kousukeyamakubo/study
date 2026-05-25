# sweep_wide — wide角度モデルのパラメータ探索

## 目的

narrow角度グリッド移行前に実施した wide角度（±20°程度）モデルの学習・評価実験群。
現行モデル（narrow）との比較参照用として保持している。

## サブフォルダ一覧

| フォルダ | 内容 |
|---|---|
| `wide_loss_sweep/` | wide角度モデルのFocal Loss パラメータ（γ・α）格子探索。 |
| `wide_gamma4_eval/` | `wide_loss_sweep` の最良設定（γ=4）で学習したモデルの tolerance sweep 評価。 |

## モデル重み

- `wide_gamma4_eval/wide_gamma4_results/wide_gamma4.pt` — wide角度ベストモデル
- `wide_loss_sweep/sweep_models/wide_gamma5.0_alpha500.pt` — tolerance比較用モデル（`eval_nn/tolerance_sweep_wide_best.py` が参照）
