# 映像 → 検出結果 CSV。YOLO への薄いラッパ。
#
# 【この層を薄くしている理由】
# 検出器は差し替わりうる（モデル更新・別実装への乗り換え）。接地点の決め方や
# cyclist の統合といった研究側のロジックは detections.py に置き、
# ここは「映像を COLUMNS の CSV に落とす」だけに限定する。
#
# 出力する CSV は COCO の生クラス ID のまま（person/bicycle を分けたまま）。
# cyclist への統合は detections.merge_cyclist() が行う。
#
# 【追跡 ID について】
# 単発検出ではなく track() を使い、フレームをまたいで同一目標に同じ track_id を振る。
# レーダー側との対応付け（同じ目標の軌跡として照合する）に必要なため。
#
# 依存: ultralytics（camera_pipeline/.venv に導入済み。requirements.txt を参照）
#
# 【入力】動画ファイル、または連番画像のディレクトリ。
# ATLAS の `.cam` は cam_extract.py で連番 JPEG に展開して渡す（無劣化のため mp4 化しない）。
# ディレクトリ入力では fps が取れないので、--dt でフレーム間隔を明示すること。
#
# 使い方:
#   python detect_yolo.py video.mp4 --out detections.csv
#   python detect_yolo.py frames/ --dt 0.2 --out detections.csv

import argparse
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path

# lib/ を import 可能にする。各スクリプトは直接実行される前提なので、
# パッケージ化せずパスを通す方式にしている
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from detections import COLUMNS, COCO_BICYCLE, COCO_MOTORCYCLE, COCO_PERSON, VEHICLE_COCO

# 検出対象。person/bicycle は cyclist に統合するため両方拾う
WANTED = {COCO_PERSON, COCO_BICYCLE, COCO_MOTORCYCLE} | VEHICLE_COCO


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_video(video: Path, model_name: str = "yolo11m.pt", conf: float = 0.25,
                 imgsz: int = 1280, tracker: str = "bytetrack.yaml",
                 dt: float | None = None) -> pd.DataFrame:
    """映像または連番画像ディレクトリを検出・追跡し COLUMNS 形式の DataFrame を返す。

    imgsz の既定を 1280 にしてあるのは、5階からの俯角設置では目標が
    画像上で小さくなるため。既定の 640 だと 40 m 先の自転車を取りこぼす見込み

    dt: フレーム間隔 [s]。指定すると t_s = frame * dt で時刻を作る。
        ATLAS の `.cam` は frame periodicity（既定 0.2 s）に律速されるため、
        動画の fps から推定すると時刻が数倍ずれる。ディレクトリ入力では必須"""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics が未導入です。`pip install ultralytics` が必要。\n"
            "新規ライブラリの追加は事前確認の対象（CLAUDE.md）なので、勝手に入れないこと"
        ) from e
    import cv2

    if video.is_dir():
        files = sorted(p for p in video.iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"{video} に画像がありません")
        if dt is None:
            raise ValueError("ディレクトリ入力では --dt が必須です（.cam なら 0.2）")
        n_frame = len(files)
        im = cv2.imread(str(files[0]))
        h, w = im.shape[:2]
        fps = 1.0 / dt
    else:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    print(f"{video.name}: {w}x{h}, {fps:.3f} fps, {n_frame} フレーム "
          f"({n_frame/max(fps,1e-9):.1f} 秒)")
    # 内部パラメータはこの解像度で較正されている必要がある（ズーム・焦点も同条件）
    print(f"  ※ チェスボード較正は {w}x{h} の同じ設定で撮ること")

    model = YOLO(model_name)
    rows = []
    # stream=True でフレームを逐次処理する（長い映像でメモリに載せきらないため）
    for i, res in enumerate(model.track(source=str(video), persist=True, stream=True,
                                        conf=conf, imgsz=imgsz, tracker=tracker,
                                        classes=sorted(WANTED), verbose=False)):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        cf = b.conf.cpu().numpy()
        tid = (b.id.cpu().numpy().astype(int) if b.id is not None
               else [-1] * len(cls))            # 追跡が付かなかったフレーム
        for j in range(len(cls)):
            # dt があればそれを使う。.cam では frame がそのままレーダーのフレーム番号
            rows.append(dict(frame=i, t_s=i * dt if dt else i / fps, track_id=int(tid[j]),
                             cls=int(cls[j]), conf=float(cf[j]),
                             x1=xyxy[j, 0], y1=xyxy[j, 1],
                             x2=xyxy[j, 2], y2=xyxy[j, 3],
                             foot_u=float((xyxy[j, 0] + xyxy[j, 2]) / 2),
                             foot_v=float(xyxy[j, 3])))
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dt", type=float, default=None,
                    help="フレーム間隔[s]。.cam 由来の連番画像なら 0.2")
    args = ap.parse_args()

    df = detect_video(args.video, args.model, args.conf, args.imgsz, dt=args.dt)
    out = args.out or args.video.with_suffix(".detections.csv")
    df.to_csv(out, index=False)

    print(f"\n検出 {len(df)} 件 / 追跡 ID {df['track_id'].nunique()} 個")
    print(df.groupby("cls").size().rename("件数"))
    print(f"保存: {out}")
    print("\n次: detections.merge_cyclist() で cyclist に統合 → "
          "to_ground() で地上座標へ")


if __name__ == "__main__":
    main()
