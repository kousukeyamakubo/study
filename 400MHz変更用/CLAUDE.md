# 研究概要

28GHz帯 ISAC（時分割型）のセンシング部の研究。
**現在は実測フェーズ（2026-09-08〜）**: 24GHz 評価キット ATLAS（TX2×RX4）を建物 5F に置き、
道路を走る自転車・歩行者を計測して、.dat → RD → 距離−時間 → 軌跡 の後処理と、レーン（横位置）の識別可否を検証している。

シミュレーション（WaveFarer + UNet 検出器）は**保留中**。関連ファイルの一覧は `docs/overview.md` の末尾。
シミュレーション・DICOMO 時代のファイルは一式 `sim/` にある（中の相対パスは `sim/` 基準のまま）。
保留中の学習まわりのファイル（`sim/check.ipynb`、`sim/learn_dataset_*`、`sim/models/` など）は、指示がない限り読まなくてよい。

## 主要ファイル（実測系）

| ファイル | 役割 |
|---|---|
| `weekly/0727/atlas_export.py` | .dat → 時刻付き複素 RD テンソル（npz）。解析の唯一の入口 |
| `weekly/0727/ANALYSIS_OVERVIEW.md` | ATLAS 解析プログラムの全体像 |
| `weekly/0727/atlas_dat_format.md` | .dat フォーマット解読の根拠 |
| `weekly/0727/atlas_track.py` | npz から移動目標の軌跡を抽出 |
| `weekly/0928/geometry.py` | 走行ルートの幾何（地上座標 → 斜距離 R・径方向速度 v_r） |
| `weekly/1001/processing.md` | 10/1 データにかけた処理の流れ（.dat から結論の数字まで） |
| `weekly/1001/report_1002.md` | 10/2 報告の原稿 |

週次フォルダ同士は `sys.path` で互いのコード（特に `weekly/0727/`）を読み込んでいるので、移動・改名しない。

## docs/ 案内

詳細情報は以下のファイルを参照。作業内容に応じて必要なものを読むこと。

| ファイル | 内容 | 読むべき場面 |
|---|---|---|
| `docs/realworld.md` | 実測（ATLAS）の確定仕様・処理の流れ・既知の注意点 | 実測データの解析・測定系を考えるとき（まずここ） |
| `docs/overview.md` | 研究背景・研究の流れ・シミュレーション系ファイル一覧 | 研究の全体像を把握したいとき |
| `docs/sim/simulation.md` | レーダー仕様・WaveFarer・物理的注意事項 | シミュレーションを触るとき |
| `docs/sim/dataset.md` | 角度グリッド・データセット仕様・分割 | 学習データを扱うとき |
| `docs/sim/model.md` | モデル構造・損失関数・学習設定 | モデルの改修や学習条件変更のとき |
| `docs/sim/experiments.md` | sim の実験結果・課題・分析 | sim 実験の議論のとき |

## ドキュメント鮮度チェック

**セッション開始時に必ず実行すること:**

以下の2コマンドを実行し、docs/ の最終更新日とその後のコード変更を確認する。

```
git log -1 --format="%cr" -- docs/
git log --since="7 days ago" --oneline
```

`docs/` の最終更新が7日以上前、かつ直近コミットに docs/ 以外の変更が含まれている場合、
セッションの最初に「docs/ を更新しますか？」とユーザーに確認する。

## ミーティングメモと週次フォルダ

- 週次ミーティングの記録は `meeting/YYYY-MM-DD.md`（git 管理外）
- 週ごとの作業は `weekly/MMDD/`。日付は対応する `meeting/YYYY-MM-DD.md` に合わせる（weekly-slides スキルがこの対応で探す）
- 各週次フォルダの README.md に、その週の目的・結果・判断を書く

**セッション開始時に `ls meeting/` で最新ファイルを確認すること。**
最新の `meeting/YYYY-MM-DD.md` と対応する `weekly/MMDD/README.md` を読み、今週の目的・進捗を把握してから作業に入る。
コーディング作業中は読み返さなくてよい。方針議論や振り返りのときだけ参照する。

## コーディング規則

### 命名規則

| 対象 | 規則 | 例 |
|---|---|---|
| クラス | PascalCase | `RadarLightUNet`, `cyclist_env`（既存は snake_case 混在） |
| 関数・変数 | snake_case | `run_epoch`, `decode_peaks_from_logits` |
| 定数 | UPPER_SNAKE_CASE | `N_FIXED`, `BATCH_SIZE`, `DEVICE` |
| モデル保存ファイル | `best_detector_<説明>.pt` | `best_detector_agnostic.pt` |
| データセットフォルダ | `sim/learn_dataset_<説明>/` | `sim/learn_dataset_fixed_angle/` |

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
- `feature/*`: 複数週にまたがる機能単位の作業
