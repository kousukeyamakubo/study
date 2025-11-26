# Tracking Simulator
複数ターゲット追跡シミュレータです。
設定ファイル (`config.py`) を編集するだけで、アルゴリズムの切り替えや実験パラメータの変更が可能です。

## 必要要件 (Requirements)
* Python 3.8+
* numpy
* pandas
* matplotlib (可視化用)

## 使い方
### 実行方法
* "python main.py"でプログラム実行
* "python /utils/visualizer.py"で可視化プログラム実行
### 設定の変更
* config.pyを編集すると動作を変更可能
## ディレクトリ構成
## ディレクトリ構成

```text
.
├── config.py              # 設定ファイル (シミュレーションパラメータ、アルゴリズム選択)
├── factory.py             # アルゴリズム生成ファクトリ (Tracker, Filterの生成)
├── main.py                # 実行エントリーポイント
├── simulator.py           # シミュレーション実行ループ管理
├── core/                  # 共通モジュール・フィルタコア
│   ├── extended_kalman_filter.py # 拡張カルマンフィルタ (EKF) - 実装予定
│   ├── kalman_filter.py          # 線形カルマンフィルタ (KF)
│   ├── models.py                 # 運動モデル・観測モデル定義 (CVモデル等)
│   └── state.py                  # 状態クラス (GaussianState) 定義
├── interfaces/            # インターフェース定義 (抽象基底クラス)
│   ├── association.py     # データ割り当て (IAssociation)
│   ├── filter.py          # フィルタ (IFilter)
│   ├── gating.py          # ゲーティング (IGating)
│   └── tracker.py         # トラッカー (ITracker)
├── modules/               # 各アルゴリズムの実装
│   ├── associations/      # データ割り当てロジック
│   │   ├── gnn.py         # GNN (Global Nearest Neighbor)
│   │   └── jpda.py        # JPDA (Joint Probabilistic Data Association)
│   ├── gatings/           # ゲーティングロジック
│   │   └── ellipsoidal.py # 楕円ゲーティング
│   └── trackers/          # トラッカー本体
│       └── standard_tracker.py # 標準トラッカー (部品を組み合わせて構成)
└── utils/                 # ユーティリティ
    ├── data_generator.py  # シナリオ・観測データ生成
    ├── data_saver.py      # 結果のCSV保存
    └── visualizer.py      # 結果の可視化 (プロット作成)
```
## 
