# 研究概要

28GHz帯 ISAC センシング（`../400MHz変更用/`）の教師ラベルを、カメラ映像から生成するパイプライン。

**扱う範囲は「映像 → YOLO検出 → 較正・ホモグラフィ → 地上座標(X,Y)」まで。**
レーダー信号処理・RDマップ生成・検出器モデルそのものは `../400MHz変更用/` 側の領分であり、ここでは扱わない。
このフォルダの成果物（地上座標ラベルのCSV）を渡した先で、レーダーピークとの対応付け（S3）・
教師テンソル組み立て（S4）が行われる。

このフォルダは 2026-09-10 に `../400MHz変更用/` からカメラ関連の一式を切り出して新設した。
**カメラ側の作業はこのフォルダ内で完結する。** 経緯は `docs/history.md` にまとめてある。

## 実装置き場

| ファイル | 役割 |
|---|---|
| `0817/detect/detect_yolo.py` | 映像 → 検出結果CSV（YOLO+ByteTrack、ultralytics依存） |
| `0817/detect/overlay_detections.py` | 検出結果を映像に重ねて可視化 |
| `0817/lib/detections.py` | `merge_cyclist`（cyclist/pedestrian/vehicle統合）・接地点・地上座標化 |
| `0817/lib/homography.py` | (u,v)→(X,Y)変換（DLT）・K既知時のh/俯角分解 |
| `0817/calib/chessboard_calib.py` | カメラ内部パラメータ K の較正 |
| `0817/calib/pick_markers.py` | マーカー地物の画像座標ピッキング |
| `0817/calib/ground_plot.py` | 地上座標の可視化 |
| `0817/validate/check_homography.py` | ホモグラフィ精度の合成データ検証 |
| `0817/validate/check_detections.py` | 接地点誤差の定量化 |
| `0817/capture/cam_extract.py` | `.cam`→連番JPEG抽出、`--dat`で`.dat`フレーム数と枚数照合 |
| `0908/cam_sync/` | カメラ・レーダーの枚数一致検証（S1・完了） |
| `0908/yolo_tracking_notes.md` | 実データでのYOLO/ByteTrackの既知問題（未整理・生ノート） |
| `0910/bench/bench_device.py` | CPU/CUDA の推論速度比較（結果は同フォルダ README） |

日付フォルダ（`0817/`・`0908/`・`0910/`）の運用はレーダー側の慣習を踏襲する。

## docs/ 案内

| ファイル | 内容 | 読むべき場面 |
|---|---|---|
| `docs/overview.md` | パイプライン全体像・境界（何をここで扱い、何をメイン側に渡すか） | 全体像を把握したいとき |
| `docs/calibration.md` | カメラ内部パラメータK・ホモグラフィ・h/俯角の試算と未検証事項 | 較正・座標変換を触るとき |
| `docs/detection.md` | YOLO/ByteTrack・merge_cyclistの実装知見と既知問題 | 検出・トラッキングを触るとき |
| `docs/dataset.md` | 出力CSVの形式・受け渡し契約 | ラベル生成・出力形式を扱うとき |
| `docs/history.md` | カメラ側の決定と経緯（合同ミーティングからの抜粋） | 「なぜこうなっているか」を知りたいとき |

## ドキュメント鮮度チェック

セッション開始時に実行:

```
git log -1 --format="%cr" -- camera_pipeline/docs/
git log --since="7 days ago" --oneline -- camera_pipeline/
```

## ミーティングメモ

**週次ミーティングは `../400MHz変更用/` 側とレーダー・カメラ合同で行われるため、
専用のミーティングメモフォルダはここには作らない。**
議事の正本は `../400MHz変更用/meeting/YYYY-MM-DD.md`。ただし**通常は `docs/history.md` を読めばよい**
（過去の回からカメラ関連の決定・数値・未決事項だけを抜き出して1本に畳んである）。

新しい回でカメラ関連の決定が出たら、正本から拾って `docs/history.md` に追記する。
セッション開始時に、まだ畳んでいない回が無いか確認する:

```
ls ../400MHz変更用/meeting/
```

## コーディング規則

`../400MHz変更用/CLAUDE.md` の規則を踏襲する。特に:

- コメントは日本語・「なぜ」を書く（既存の `detections.py`/`homography.py` のスタイルに合わせる）
- 指示された箇所のみ修正し、周辺の整理は別途指示があるときのみ
- 各フォルダ運用は `0817/`・`0908/` の慣習にならう: README.md に結果・手順・既知の限界、
  次のアクションは meeting メモに集約（フォルダREADMEに「次にやること」を書かない）

### 依存ライブラリ

**venv はこのフォルダ専用**（`camera_pipeline/.venv`。2026-09-10 にレーダー側から分離）。
直接依存とバージョンは `requirements.txt` に固定してある。

```powershell
.venv\Scripts\python.exe 0817/detect/detect_yolo.py <video>
```

- 画像・幾何: `numpy`（`homography.py` はOpenCV非依存）, `opencv-python`
- 検出: `ultralytics`（`lap` は ByteTrack が要求）
- データ: `pandas`, `matplotlib`
- 新たなライブラリを追加する場合は事前に確認する

**torch はレーダー側とバージョンを揃えなくてよい**（カメラ側は YOLO 推論にしか使わない）。
実際カメラ側は 2.14.0、レーダー側は 2.12.1 で、分離時に出力の一致を確認済み
（`detect_yolo.py` の検出CSVがバイト単位で一致、`chessboard_calib.py` の K も一致）。
**ただしこの一致は CPU ビルド同士の話。** GPU 機（デスクトップ）は `requirements-cuda.txt`
を追加適用して CUDA 版（`2.14.0+cu130`）にしてあり、GPU を挟むと検出CSVは一致しない
（FP32 同士でも畳み込み実装が違う）。速度差と注意点は `0910/bench/README.md`。

分離した理由は ArUco（H2、`docs/history.md` の未決事項）。`cv2.aruco` に必要な
`opencv-contrib-python` は `opencv-python` と同居できないため、共有 venv のまま
入れ替えるとレーダー側を巻き込む。着手時は `requirements.txt` の該当行を差し替える。

`yolo11m.pt`（38MB）は ultralytics が初回実行時にカレントディレクトリへ自動取得する。
`.gitignore` の `*.pt` で追跡外。

## ブランチ戦略

`../400MHz変更用/` と同一リポジトリ・同一ブランチ運用（`main` / `week/YYYY-MM-DD` / `feature/*`）。
このフォルダ専用のブランチは切らない。
