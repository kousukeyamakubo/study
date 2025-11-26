# JPDAF Tracking Simulator
JPDAF (Joint Probabilistic Data Association Filter) を用いた複数ターゲット追跡シミュレータです。
設定ファイル (`config.py`) を編集するだけで、アルゴリズムの切り替えや実験パラメータの変更が可能です。
## 特徴
* **Config駆動開発**: コードを修正することなく、設定ファイルのみで実験条件を変更可能。
* **拡張性の高い設計**: FactoryパターンとStrategyパターンを採用しており、新しいトラッキングアルゴリズム (GNN等) やゲーティング手法の追加が容易。
* **可視化**: シミュレーション結果をCSV出力し、プロットによる可視化可能。

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
.
├── config.py              # 設定ファイル
├── main.py                # 実行エントリーポイント
├── factory.py             # アルゴリズム生成ファクトリ
├── simulator.py           # シミュレーション実行ループ
├── core/                  # 共通モジュール (State, KF, Models)
├── interfaces/            # インターフェース定義 (Tracker, Gating, Association)
├── modules/               # 各アルゴリズムの実装
│   ├── associations/      # データ割り当て (JPDAなど)
│   ├── gatings/           # ゲーティング (楕円など)
│   └── trackers/          # トラッカー本体
└── utils/                 # ユーティリティ (データ生成、保存、可視化)
## 
