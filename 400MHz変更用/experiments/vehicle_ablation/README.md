# vehicle_ablation — 車両検出 ablation study

## 目的

車両（vehicle）検出のFP発生パターンを分析し、損失設計の改善根拠を示す。

## スクリプト

| スクリプト | 内容 |
|---|---|
| `vehicle_ablation.py` | スコア分布・シーン別AP・FP位置の可視化を3サイクルで段階的に分析 |

## 実行方法

```bash
cd experiments/vehicle_ablation
python vehicle_ablation.py
```

## 出力

| ファイル | 内容 |
|---|---|
| `vehicle_ablation_results/cycle1_score_dist.png` | スコア分布 |
| `vehicle_ablation_results/cycle2_scene_ap.png` | シーン別AP |
| `vehicle_ablation_results/cycle3_fp_position.png` | FP発生位置 |
| `vehicle_ablation_results/cycle3_fp_ve_detail.csv` | FP詳細CSV |
| `vehicle_ablation_results/ablation_report.txt` | 数値サマリ |
