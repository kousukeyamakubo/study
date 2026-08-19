# 後処理: ピーク抽出 → NMS → 追尾（2026-08-19）

`meeting/2026-09-08.md` S5「後処理（ピーク抽出→NMS→時間方向の追尾）」の実装。
**モデル非依存の部品**として、学習データが揃う前・角度の渡し方が決まる前から着手した。

| ファイル | 役割 | 依存 |
|---|---|---|
| `detect.py` | RD テンソル → パワーマップ → CA-CFAR → NMS | numpy |
| `track.py` | フレームごとのピーク → 時間方向のトラック（greedy 最近傍） | numpy |
| `check_postprocess.py` | 合成データによる動作確認 | numpy |

```
rd (Frame,TX,RX,Doppler,Range) ──[incoherent_power]──> パワーマップ (Frame,Doppler,Range)
                                          │
                                    [cfar_ca + nms_2d]   ← detect_peaks() がまとめて実行
                                          ↓
                              フレームごとの [doppler_idx, range_idx]
                                          │
                                  [greedy_associate]
                                          ↓
                                  時間方向のトラック（Track）
```

## なぜ角度を使わないか

9/8 時点で「角度情報を DBF で潰すか、生の位相のまま渡すか」は未決定（`meeting/2026-09-08.md`）。
この検出器は 7 素子（virt_array）をインコヒーレントに合成したパワーだけで動かし、
位相（角度）を一切使わない。DBF を先取りすると、角度の渡し方を判断するための
実測材料（SNR・素子間位相の安定性）を自分で汚してしまうため。

→ 結果として、この後処理は角度の判断がどちらに転んでも影響を受けない。

## 使い方

```python
import numpy as np
from detect import detect_peaks
from track import greedy_associate

d = np.load("atlas_log_XXXXXXXX_rd.npz")
peaks_by_frame = detect_peaks(d["rd"])           # フレームごとの [doppler_idx, range_idx]
tracks = greedy_associate(peaks_by_frame, d["range_m"], d["vel_ms"],
                          max_range_step_m=d["range_m"][1] * 3, max_vel_step_ms=1.0)
```

## 既知の限界（素朴な検出器としての割り切り）

- **追跡は見失い猶予なし**: 1フレームでも対応が取れないとトラックを打ち切る。
  実測でロストが多ければ、数フレームの猶予を持たせる拡張が要る
- **CFAR のしきい値・窓サイズは初期値**: `experiments/eval_cfar` の sim 向けパラメータを
  実測のビン数（Doppler=16, Range=129）に合わせて縮小しただけで、実測でのチューニングが要る
- **複数目標の分離は NMS 任せ**: `sep_d`/`sep_r` より近い2目標は1つに潰れる。
  9/8 の「近接した複数目標の分離」を検証する段では、この窓サイズの妥当性を見直す必要がある

## 動作確認

`python check_postprocess.py` — 静止ノイズ中に1目標を置いた合成データで、
①SNR を振って検出率を確認、②50フレームが1本のトラックにまとまることを確認、
③`incoherent_power` の出力形状を確認、の3点を検証する。実測データは不要。
