# モデル構造・学習設定

## モデル: RadarLightUNet

`check.ipynb` に実装。

### アーキテクチャ

- **入力**: (B, N_FIXED=10, H, W) — 各角度の RD マップ
- **出力**: (B, N_FIXED=10, H, W) — 各角度チャネルのヒートマップ（logits）

```
Encoder × 4: ConvBlock(in→32ch) + MaxPool2d
Bottleneck:   ConvBlock(32→32ch)
Decoder × 4:  Upsample + concat(skip) + ConvBlock(64→32ch)
Angle heads:  N_FIXED 個の独立した Conv2d(32+1 → 1)
              （共有特徴マップ + 各角度の入力 RD マップを連結）
```

### ConvBlock

```
Conv2d → ReLU → Conv2d → ReLU → Dropout2d(0.1)
```

### 角度ヘッドの独立性

各角度チャネルに対して独立したヘッドを持つ設計（`feature/angle-head-independence` ブランチで開発中）。
共有 UNet 特徴量に各角度の入力 RD マップ(1ch)を連結してから 1×1 Conv で出力。

## 損失関数

BCE + Dice の組み合わせ:

```python
loss = bce_loss_from_logits(logits, targets) + dice_loss_from_logits(logits, targets)
```

## 学習設定

| パラメータ | 値 |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 16 |
| Epochs | 15 |
| Dropout | 0.1 |

## 推論: ピーク検出 (3D NMS)

`decode_peaks_from_logits(logits, threshold=0.99)` で実装。

1. sigmoid を適用し確信度マップを取得
2. 最大値のピクセルを取得（閾値 0.99 以上のみ）
3. ピーク周辺 (ch±1, d±3, r±1) を 0 にしてから繰り返し

### 真値へのマッチング

検出ピークをサイクリスト・車両に割り当て: 最小コスト割り当て（全組み合わせ比較）。

## モデルファイル

| ファイル | 内容 |
|---|---|
| `best_detector_agnostic.pt` | 現在の最良モデル（角度ヘッド独立版） |
| `best_detector_fixed_angle.pt` | 固定角度版モデル |
| `best_detector_single_object.pt` | 単一物体版 |
| `best_detector_ped_vehicle.pt` | 歩行者・車両版 |
