# 実行手順

`0817/` のスクリプトの動かし方。設計と結果は `README.md`、`.cam` の仕様は `camera_capture.md`。

## 構成

```
0817/
  lib/       homography.py  detections.py        共通ライブラリ（直接は実行しない）
  capture/   cam_extract.py                      .cam → 連番JPEG / 動画
  detect/    detect_yolo.py  overlay_detections.py   YOLO と可視化
  calib/     pick_markers.py ground_plot.py      マーカー・俯瞰図
  validate/  check_homography.py check_detections.py  合成データでの検証
  data/      .cam / フレーム / CSV / 図           生成物（git 管理外）
```

各スクリプトは `0817/` から相対パスで直接実行する。`lib/` へのパスはスクリプト側で
通しているので、パッケージとして import する必要はない。

## 依存

```
pip install numpy pandas opencv-python ultralytics
```

`ultralytics` は `detect_yolo.py` だけ、`opencv-python` は `--mp4` と `overlay_detections.py` だけで使う。
モデルの重み（`yolo11m.pt`）は初回実行時に自動でダウンロードされる（約 39 MB）。

---

## 全体の流れ

```
atlas_log_*.cam ──[cam_extract]──> 連番JPEG ──[detect_yolo]──> 検出CSV
                        │                                        │
                        └──> .mp4（目視用）      [overlay_detections]
                                                                 ↓
                                                          bbox付き .mp4
```

---

## 1. `.cam` を展開する

```bash
python capture/cam_extract.py atlas_log_20260816_161458.cam.gz --dat atlas_log_20260816_161458.dat --mp4
```

- `.gz` のままでも読める
- 出力: `<名前>_frames/00000.jpg …`（**無劣化**。中身がもともと JPEG なのでそのまま書き出す）
- `--dat` を付けると `.cam` の枚数と `.dat` のフレーム数を照合する。
  **不一致なら映像が間引かれており index 対応が崩れているので撮り直す**
- `--mp4` で目視用の動画も出る。`fps = 1/dt` なので**再生時間が実時間と一致**（50 枚 → 10 秒）

| オプション | 既定 | 用途 |
|---|---|---|
| `--out DIR` | `<名前>_frames/` | 展開先 |
| `--dat PATH` | なし | フレーム数の照合 |
| `--mp4` | off | 目視用の動画も出す |
| `--dt` | `0.2` | frame periodicity [s]。動画の fps = 1/dt |
| `--scale` | `1.0` | 動画のみ縮小。4K が重ければ `0.5` |

**YOLO には連番 JPEG を渡すこと。** `--mp4` は再エンコードなので画質が落ちる。

---

## 2. 検出する

```bash
python detect/detect_yolo.py atlas_log_20260816_161458_frames/ --dt 0.2 --out det.csv
```

- 入力は**連番画像ディレクトリ**または動画ファイル
- **ディレクトリ入力では `--dt` が必須**。fps が取れないため、指定しないと時刻が 6 倍ずれる
  （`.cam` は 5 fps 相当だが、動画として開くと既定の 30 fps とみなされる）
- 出力 CSV の `frame` 列は**そのままレーダーのフレーム番号**になる

| オプション | 既定 | 備考 |
|---|---|---|
| `--dt` | なし | `.cam` 由来なら `0.2`。ディレクトリ入力では必須 |
| `--imgsz` | `1280` | 5階からの俯角設置で目標が小さいため。`640` だと 40 m 先を落とす |
| `--model` | `yolo11m.pt` | |
| `--conf` | `0.25` | |

CSV の列: `frame, t_s, track_id, cls, conf, x1, y1, x2, y2, foot_u, foot_v`
`cls` は COCO の生 ID（person/bicycle を分けたまま）。cyclist への統合は次段。

---

## 3. 目で確認する

```bash
python detect/overlay_detections.py atlas_log_20260816_161458_frames/ det.csv --scale 0.5
```

bbox・**接地点の赤い十字**・`クラス #track_id conf` を重ねた動画を出す。
見るべきは 3 点:

1. bbox が目標に付いているか（誤検出・取りこぼし）
2. **接地点が実際の接地位置とどれだけずれているか** ← 誤差の支配要因（`README.md` §1）
3. **track_id が維持されているか** — 5 fps では切れうる。
   3 フレーム未満の track_id があれば自動で警告が出る

---

## 4. マーカーを指す

```bash
python calib/pick_markers.py data/<名前>_frames/00025.jpg --out data/markers.csv
```

タイルの角を粗くクリック → 拡大窓（8倍）で確定 → 端末で**格子番号 `i,j`** を入力。
地上座標は `(0.20×i, 0.20×j)` m として自動で入る。`u` 取り消し / `s` 保存 / `q` 保存して終了。
**これらのキーは画像ウィンドウにフォーカスがある間しか効かない**（端末に数値を打った直後は
端末側にフォーカスがある）。端末の入力プロンプトでも `q` で終われる。`3 2 q` のように
数値の後ろに付けると「この点を足して終了」（2026-09-10）。

- **走行範囲を囲むように 6〜8 点**。狭い範囲に固めると外挿が破綻する
  （0.4 m 四方の 5 点では、8 m 先で ±2.8 m まで悪化した）
- 保存時に再投影残差を表示し、**1 点だけ突出していれば ★** を付ける（格子の数え間違い＝0.20 m）
- `--check data/markers.csv` で指し直さず再評価だけ

## 5. 俯瞰図で答え合わせ

```bash
python calib/pick_markers.py data/<名前>_frames/00025.jpg --annotate --out data/annots.csv
python calib/ground_plot.py data/markers.csv data/annots.csv --det data/bike_ground.csv
```

`--annotate` は格子番号でなく**ラベル**を入力するモード（空入力で直前のラベルを引き継ぐ）。
ラベルが `edge` / `line` / `curb` / `側溝` / `縁` で始まると線として結ぶ。

ホモグラフィの推定に**使っていない**情報で検証するのが目的。

- **道幅**が距離によらず一定か（`edge` 2 本で自動計算）
- 縁が**直線**に出るか — 曲がればレンズ歪みか非平面
- 各点の不確かさ。目標 0.28 m を超えた点は図上に赤い×

## 6. 地上座標にする

ここから先はライブラリなので、ノートブックか短いスクリプトで呼ぶ。

```python
import pandas as pd
from detections import merge_cyclist, to_ground, add_slant_range
from homography import estimate_homography

df = merge_cyclist(pd.read_csv("det.csv"))        # person+bicycle → cyclist
H = estimate_homography(uv_markers, xy_markers)   # 画像上のマーカー ↔ 実測した地上座標
df = to_ground(df, H)                             # (foot_u, foot_v) → (X, Y)
df = add_slant_range(df, h=15.0)                  # 学習データ組み立て時のみ
```

**地上座標 (X, Y) がラベルの正本。** 斜距離への変換は h に依存するので、
h の推定値が変わってもラベルを作り直さずに済むよう、保存するのは (X, Y) の側。

---

## 検証スクリプト（合成データ・引数なし）

映像が無くても走る。精度の根拠を再現したいときに。

```bash
python validate/check_homography.py    # ホモグラフィの精度・マーカー配置の指針
python validate/check_detections.py    # 接地点の取り方が地上座標に与える誤差
```

結果は `README.md` にまとめてある。

---

## 実行例（2026-08-16 の環境映像）

```
$ python capture/cam_extract.py atlas_log_20260816_161458.cam.gz --mp4 --scale 0.5
atlas_log_20260816_161458.cam.gz: 50 枚 → ..._frames
  JPEG 1枚 1166〜1289 KB (平均 1258 KB)
  動画 atlas_log_20260816_161458.mp4: 5.0 fps, 10.0 秒（実時間と一致）

$ python detect/detect_yolo.py ..._frames --dt 0.2 --out env.csv
..._frames: 3840x2160, 5.000 fps, 50 フレーム (10.0 秒)
検出 0 件          ← ターゲット不在の映像なので、誤検出ゼロという意味
```
