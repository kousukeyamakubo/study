## 目的
このファイルは、このリポジトリでAIアシスタント（コーディングエージェント）が素早く実用的に作業できるよう、設計上の重要点・実行ワークフロー・プロジェクト固有の慣習をまとめたものです。

## ビッグピクチャ
- **主要コンポーネント**: [core](core/)（フィルタ・状態・モデル） / [interfaces](interfaces/)（抽象インターフェース） / [modules](modules/)（アルゴリズム実装: gating, association, trackers） / [utils](utils/)（入出力・可視化）
- **エントリーポイント**: [main.py](main.py) が典型的な実行フローを構成する。設定は [config.py](config.py) で行う。
- **データフロー**: Config → core/models.py で `F,Q,H,R` を生成 → `factory.create_filter` と `factory.create_tracker` でアルゴリズムを注入 → `TrackingSimulator.run`（[simulator.py](simulator.py)）が `measurements_list` を受け取り `tracker.predict/update` を呼ぶ → 結果は `utils/data_saver.py` を経て `csv_result/` に保存 → `utils/visualizer.py` で可視化。

## 重要な設計・慣習（すぐ役立つポイント）
- **依存注入パターン**: 実装は `interfaces/*.py` に定義された抽象インターフェースを満たすこと。器（`StandardTracker`）へ `IGating`/`IAssociation`/`IFilter` を注入する設計になっている（参照: [factory.py](factory.py)、[modules/trackers/standard_tracker.py](modules/trackers/standard_tracker.py)）。
- **トラッカー実装の期待値**: `ITracker` は `predict(states)` と `update(predicted_states, measurements)` を実装すること。`update` は `(updated_states, info)` を返す（`info` は保存やデバッグ用でアルゴリズム依存）。
- **測定データの形式**: 多くのコードで観測ベクトルの位置成分を `z[2:4]` として扱っている。CSVや生成器が出力する列順を壊さないよう注意（参照: [modules/trackers/standard_tracker.py](modules/trackers/standard_tracker.py)）。
- **モデル行列の生成**: `core/models.py` に `get_motion_model(cfg)` と `get_measurement_model(cfg)` があり、`F,Q,H,R` の作り方はここを変更する。
- **CSVモード**: `config.use_csv_mode` が True のとき、`main.py` は `true_data.csv` と `measurements.csv` を読み込む。可視化は `python utils/visualizer.py` を使う（出力は `csv_result/` を想定）。

## 実行ワークフロー（よく使うコマンド）
- シミュレーション実行（デフォルト）: `python main.py`
- 可視化: `python utils/visualizer.py`
- 設定変更: `config.py` を直接編集して `tracker_type` / `filter_type` / `use_csv_mode` / `gate_threshold` 等を切り替える。

## 拡張・実装時の注意点
- 新しいアルゴリズム（例: 新しい Association や Gating）を追加する場合:
  1. `interfaces/*.py` の該当インターフェースを実装するクラスを `modules/...` に追加する。
  2. `factory.create_tracker` にそのクラスをインポートし、`cfg.tracker_type` による分岐を追加して注入する。
- 大規模ターゲット数での注意: `modules/associations/jpda.py` は全組合せ（直積）を列挙しており組合せ爆発のリスクがある。最適化や近似（門限の厳格化、仮説生成の枝刈り）が必要になる可能性が高い。
- 数値安定性: 楕円ゲーティングやカイ2乗閾値に依存する箇所が多数ある。特異行列対策（小さい正則化項）や閾値の調整を行うと良い。

## 既存ファイルを確認するときの短い探索ガイド（具体例）
- 推移・観測モデル: [core/models.py](core/models.py#L1-L200)
- トラッカーの骨組み: [modules/trackers/standard_tracker.py](modules/trackers/standard_tracker.py#L1-L200)
- JPDA の実装と注意点: [modules/associations/jpda.py](modules/associations/jpda.py#L1-L200)
- ゲーティング実装: [modules/gatings/ellipsoidal.py](modules/gatings/ellipsoidal.py#L1-L200)
- 実行フロー（main）: [main.py](main.py#L1-L200) / シミュレータ: [simulator.py](simulator.py#L1-L200)
- 可視化: [utils/visualizer.py](utils/visualizer.py#L1-L200)（出力CSV列名を参照すること）

## 期待される出力/入出力フォーマット
- 出力ディレクトリ: `csv_result/` に複数のCSV（`true_trajectory.csv`, `estimated_trajectory.csv`, `measurements.csv`, `validated_measurements.csv`（任意））を保存。
- `utils/visualizer.py` はこれらのファイルを前提にプロットを作成するため、CSVのカラム名（`time`, `target_id`, `x`, `y`, `vx`, `velocity` 等）を守ること。

## 物足りない点・確認すべき箇所
- `core/extended_kalman_filter.py` は未完成（READMEとコードに「予定」とある）。EKFを使用する場合は実装の整合性確認が必要。
- `modules/associations/gnn.py` のインターフェース互換性を確認して、`factory.py` での注入が動作するかテストすること。

----
フィードバックください: ここに足りない細部（CSV列の正確なヘッダ、CI/テスト手順、よく使う実験セット）があれば追記して反映します。
