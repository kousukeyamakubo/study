# モデル構造

> **注**: このドキュメントは DICOMO 提出時点のモデル構成を記述している。
> 現在はシミュレーションシナリオ側の検証フェーズであり、モデル自体は変更していない。

## モデル: RadarUNet3DSoftmax

`check.ipynb` に実装。

### 入出力形状

- **入力**: (B, 1, N_FIXED, H, W) — RD マップを depth 方向に並べた 3D テンソル（1ch）
- **出力**: (B, 3, N_FIXED, H, W) — 各ボクセルの 3 クラス logits（背景=0 / cyclist=1 / vehicle=2）

### アーキテクチャ概要

3D U-Net ライク構成。角度軸（N_FIXED）を depth として 3D 畳み込みで処理する。

- エンコーダ: `ConvBlock3D` × 4 段（チャネル数 `fixed_channels=32`、各段は Conv3d × 2 + ReLU + Dropout3d）
- デコーダ: 対称なアップサンプリング + skip connection
- 角度軸方向にはプーリングしない（空間分解能を保持）
- 出力ヘッド: 1×1×1 Conv3d → 3ch logits

### 損失関数

alpha-balanced Focal Loss:

```python
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [1.0, 500.0, 500.0]  # 背景 / cyclist / vehicle
```

背景ピクセルが圧倒的多数のため alpha で重み付け、さらに易しいサンプルを focal term でダウンウェイト。

### 検出デコード

- logits に softmax → `argmax` でクラス判定
- cyclist/vehicle クラスの確率マップにピーク検出（NMS）を適用して最終出力とする

### ラベル形式

`MultiTargetDatasetSeg` が生成する教師ラベルは整数クラスマップ `(B, N_FIXED, H, W)` (long)。
値は `{0=背景, 1=cyclist, 2=vehicle}`（Gaussian heatmap ではなく離散クラス）。
