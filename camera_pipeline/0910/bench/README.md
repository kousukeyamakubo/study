# CPU / CUDA の推論速度比較

`detect_yolo.py` の設定（yolo11m・imgsz=1280・ByteTrack）で、CPU と CUDA の
1 フレームあたり処理時間を測る。目的は撮影データを流すときの所要時間見積り。

## 前提: torch が CPU ビルドだと GPU は使われない

`requirements.txt` は torch をピンせず ultralytics の依存に任せている。PyPI の
Windows ホイールは CPU ビルド（`2.14.0+cpu`）なので、GPU 搭載機で
`pip install -r requirements.txt` しただけでは `torch.cuda.is_available()` は False。
スクリプト冒頭がこれを表示するので、まず確認する。

GPU 機（デスクトップ）では `requirements-cuda.txt` を追加適用して CUDA 版に入れ替える。
CPU 機（ノートPC）では適用せず、`requirements.txt` のままにする。

**入れ替えには `+cu130` までのピンが要る。** `--index-url` だけでは足りない。
CPU 版が既に入っていると pip は `torch==2.14.0` をローカルバージョン識別子を無視して
照合し、`2.14.0+cpu` を「充足済み」と判断して素通りする（2026-09-10 に実際に踏んだ）。

このため**2台で torch のビルドが違う状態が正常**になる。検出 CSV が2台で
バイト一致しなくなる点は下の「既知の限界」を参照。

## 手順

```powershell
.venv\Scripts\python.exe 0910\bench\bench_device.py <frames_dir or video> --devices cpu,cuda:0
.venv\Scripts\python.exe 0910\bench\bench_device.py <入力> --devices cuda:0 --half   # FP16
.venv\Scripts\python.exe 0910\bench\bench_device.py --synthetic 3840x2160            # 実データ無し
```

`predict`（推論のみ）と `track`（本番と同じ経路）を別々に出す。
`--synthetic` は**本番と同じ 3840×2160** を指定する（`docs/calibration.md` の
較正解像度が本番解像度）。

## 結果

計測環境（2026-09-10、デスクトップ機）:
Ryzen 7 5700X（8C/16T） / RTX 3060 Ti 8GiB sm_86 / ドライバ 591.86 /
torch 2.14.0+cu130 / ultralytics 8.4.120 / Python 3.12.2 / Windows 11。
共通設定 yolo11m・imgsz=1280・conf=0.25・warmup=5、値は中央値。

### 合成フレーム 3840×2160（n=40、箱 0）

| モード | デバイス | ms/frame | fps | CPU比 |
|---|---|---|---|---|
| predict | cpu | 573.3 | 1.7 | 1.00x |
| predict | cuda:0 | 30.0 | 33.3 | 19.1x |
| predict | cuda:0+fp16 | 20.0 | 49.9 | 28.7x |
| track | cpu | 577.8 | 1.7 | 1.00x |
| track | cuda:0 | 29.3 | 34.1 | 19.7x |
| track | cuda:0+fp16 | 22.9 | 43.7 | 25.2x |

### 実 JPEG 3840×2160（n=16、箱 9）

入力は `0817/calib/chessboard_images`。**撮影データではない**ので検出負荷は
代表していない（下の「既知の限界」参照）。実ファイル・実内容での前処理コストの確認用。

| モード | デバイス | ms/frame | fps | CPU比 |
|---|---|---|---|---|
| predict | cpu | 555.1 | 1.8 | 1.00x |
| predict | cuda:0 | 29.7 | 33.7 | 18.7x |
| predict | cuda:0+fp16 | 20.6 | 48.6 | 26.9x |
| track | cpu | 557.1 | 1.8 | 1.00x |
| track | cuda:0 | 31.8 | 31.5 | 17.5x |
| track | cuda:0+fp16 | 20.9 | 47.8 | 26.7x |

### 参考: 合成フレーム 1920×1080（n=60、箱 0）

| モード | デバイス | ms/frame | fps | CPU比 |
|---|---|---|---|---|
| predict | cpu | 617.9 | 1.6 | 1.00x |
| predict | cuda:0 | 29.6 | 33.8 | 20.9x |
| track | cpu | 622.4 | 1.6 | 1.00x |
| track | cuda:0 | 30.8 | 32.5 | 20.2x |

**入力解像度は効かない。** 3840×2160 と 1920×1080 で差が出ないのは、どちらも
imgsz=1280 にリサイズしてから推論するため。前処理（リサイズ）は 4K でも
5〜8 ms/frame にとどまり、支配的にならない。

### JPEG デコード（計測ループ外、別途実測）

3840×2160 の JPEG で **17.3 ms/枚**（中央値、n=16、`cv2.imread`、OpenCV 16 スレッド）。

### 所要時間の目安

`track` + デコードで 1 フレームあたり **CPU 約 0.57 s / GPU 約 0.05 s**。
撮影 1 分（30 fps = 1800 フレーム）なら **CPU 約 17 分 → GPU 約 1.5 分**。
デコードは GPU 側で全体の約 35% を占めるので、さらに詰めるならここが次の対象。

## 既知の限界

- **GPU で速くなるのは推論だけ。** NMS 後処理と ByteTrack の対応付け（`lap` の線形割当）は
  CPU に残る。実測でも推論のみは 23〜25x、`track` 全体は 17〜20x で、短縮率は必ず小さくなる。
  所要時間見積りに使うのは `track` 側。
- **JPEG デコードは計測外。** 上記のとおり 4K で 17 ms/枚。CPU 実行では推論 557 ms に対し
  3% で無視できるが、GPU 実行では 32 ms に対して上乗せとなり支配的な部類に入る。
- **実データでの検出負荷は未測定。** `*.mp4` / `*.dat` は gitignore で追跡外のため、
  リポジトリ内で使える実画像は較正用チェスボードしかない。人・車が多数写る本番映像では
  NMS と ByteTrack の負荷が上がり、`track` の短縮率は上表よりさらに小さくなる見込み。
  撮影データが手元に来たら取り直す。
- **`--half`（FP16）は検出結果を変えうる。** 今回は CPU / cuda FP32 / cuda FP16 のいずれも
  箱の数が一致した（合成 0 個、チェスボード 9 個）が、**箱の数の一致は座標の一致ではない**。
  検出が多数出る実データで確かめていないので、本番のラベル生成に使うかは未決。
- **CPU 機と GPU 機で検出 CSV は一致しない。** ultralytics は device 未指定なら CUDA を
  自動選択するため、デスクトップで流した検出結果とノートPCで流した結果は微小にずれる
  （FP32 同士でも畳み込みの実装が違う）。CLAUDE.md にある「torch のバージョン差で
  検出 CSV がバイト一致」という確認は CPU ビルド同士の話で、GPU を挟むと成立しない。
  **2026-09-10 に本番のYOLO推論はデスクトップ固定というルールにしたため、
  2台で分担してどちらで生成したか記録する必要はそもそも生じない**（`CLAUDE.md`・
  `docs/history.md` 参照）。
- 合成フレーム（`--synthetic`）は検出がほぼ出ない条件なので後処理コストが実データより軽い。
  絶対値ではなく相対比較の目安として見る。
