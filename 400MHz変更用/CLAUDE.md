# 研究概要

28GHz帯 ISAC（時分割型）のセンシング部において、チャープ信号から生成した RD マップを入力とし、
サイクリスト・車両の多目標検出と角度推定を行うニューラルネット検出器を開発している。

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `check.ipynb` | メイン: 学習・評価・可視化 |
| `learn_data_generator.ipynb` | データセット生成 |
| `monte-carlo-simulation.ipynb` | モンテカルロ評価・CFAR比較 |
| `0601meeting/rad_from_csv.ipynb` | WaveFarer CSV → RD マップ変換（現フェーズの検証中心） |
| `main.ipynb` | システムパラメータ・SNR計算（2GHz/400MHz比較） |
| `calculate_SNR.ipynb` | SNR計算補助 |
| `models/best_detector_narrow_angle.pt` | 現在の最良モデル重み（narrow角度グリッド対応） |

## docs/ 案内

詳細情報は以下のファイルを参照。作業内容に応じて必要なものを読むこと。

| ファイル | 内容 | 読むべき場面 |
|---|---|---|
| `docs/overview.md` | 研究背景・研究の流れ・問題設定 | 研究の全体像を把握したいとき |
| `docs/simulation.md` | レーダー仕様・WaveFarer・物理的注意事項 | シミュレーションを触るとき |
| `docs/dataset.md` | 角度グリッド・データセット仕様・分割 | データ生成やデータ形式を扱うとき |
| `docs/model.md` | モデル構造・損失関数・学習設定 | モデルの改修や学習条件変更のとき |
| `docs/experiments.md` | 実験結果・現状の課題・分析 | 実験の議論・次の方針を考えるとき |

## ドキュメント鮮度チェック

**セッション開始時に必ず実行すること:**

以下の2コマンドを実行し、docs/ の最終更新日とその後のコード変更を確認する。

```
git log -1 --format="%cr" -- docs/
git log --since="7 days ago" --oneline
```

`docs/` の最終更新が7日以上前、かつ直近コミットに docs/ 以外の変更が含まれている場合、
セッションの最初に「docs/ を更新しますか？」とユーザーに確認する。

更新が必要な可能性が高い項目:
- ブランチ名・最良モデルのパスが変わっていないか（CLAUDE.md）
- 新しいデータセットが追加されていないか（docs/dataset.md）
- 実験結果・課題が変わっていないか（docs/experiments.md）
- モデル構造・学習設定が変わっていないか（docs/model.md）

## ミーティングメモ

週次ミーティングの記録は `meeting/YYYY-MM-DD.md` に蓄積。

**セッション開始時に最新ファイルを確認すること:**

```
ls meeting/
```

最新の `meeting/YYYY-MM-DD.md` を読み、今週の目的・進捗を把握してからコーディング作業に入る。
コーディング作業中は読み返さなくてよい。方針議論や振り返りのときだけ参照する。

## コーディング規則

### 命名規則

| 対象 | 規則 | 例 |
|---|---|---|
| クラス | PascalCase | `RadarLightUNet`, `cyclist_env`（既存は snake_case 混在） |
| 関数・変数 | snake_case | `run_epoch`, `decode_peaks_from_logits` |
| 定数 | UPPER_SNAKE_CASE | `N_FIXED`, `BATCH_SIZE`, `DEVICE` |
| モデル保存ファイル | `best_detector_<説明>.pt` | `best_detector_agnostic.pt` |
| データセットフォルダ | `learn_dataset_<説明>/` | `learn_dataset_fixed_angle/` |

### ノートブック運用

- 実装（クラス・関数定義）は上部セルにまとめ、実験・評価セルと分離する
- セル ID（`id` フィールド）は変更しない
- 出力は残したままでよい（実験記録として機能する）
- 新しい実験セルを追加するときは既存セルを書き換えず、末尾に追加する

### 変更範囲

- 指示された箇所のみ修正する。周辺コードの整理・リファクタは別途指示がある場合のみ行う
- 既存コードの命名規則と異なる変更を加えた場合は、その旨を明記する
- モデルアーキテクチャを変更した場合は保存ファイル名も変更する（既存の `.pt` を上書きしない）

### コメント

- コメントは日本語で書く
- 「何をしているか」ではなく「なぜそうしているか」を書く
- 数式・物理的な意味の説明には積極的にコメントを入れてよい

### 配列・テンソルの次元

コード中で配列を扱うときは次元を明示するコメントを残す（既存のスタイルに合わせる）:
```python
# (B, N_FIXED, H, W)
logits = model(x)
```

### 依存ライブラリ

- 数値計算: `numpy`
- 深層学習: `torch`, `torch.nn`
- データ: `pandas`
- 可視化: `matplotlib`
- 新たなライブラリを追加する場合は事前に確認する

## ブランチ戦略

- `main`: 安定版
- `week/YYYY-MM-DD`: 週次ミーティングに向けた作業ブランチ。日付は対応する `meeting/YYYY-MM-DD.md` に合わせる
- `feature/*`: 複数週にまたがる機能単位の作業（角度グリッド対応など、1回のミーティングで区切れないもの）
- 現在のブランチ: `feature/narrow-angle-grid`（narrow角度グリッド対応）
