# eval_cfar — CFAR比較評価

## 目的

CA-CFAR検出器とNN検出器のPR曲線・APを比較する。
narrow角度グリッド（±5°）のデータセットに対して評価する。

## スクリプト一覧

| スクリプト | 内容 |
|---|---|
| `cfar_pr_ap.py` | CFARパラメータ (n_train, n_guard, pfa) を総当たりしてAPが最良の設定を選びPR曲線を描画 |
| `plot_ap_bar.py` | `cfar_pr_ap.py` の結果をAP棒グラフとして整形（論文用） |

## 実行方法

```bash
cd experiments/eval_cfar
python cfar_pr_ap.py     # まず実行（結果CSVを生成）
python plot_ap_bar.py    # その後実行（棒グラフを生成）
```

## 出力

| フォルダ | 内容 |
|---|---|
| `for_paper/` | 論文用PR曲線・AP棒グラフ（広め設定） |
| `for_paper_pm5deg/` | 論文用PR曲線・AP棒グラフ（±5°設定） |

## 依存データセット

- `learn_dataset_narrow_angle_fixed/` — 2物体固定角度（テスト用）
