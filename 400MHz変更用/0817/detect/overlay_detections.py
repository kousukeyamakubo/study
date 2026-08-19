# 検出結果 CSV を映像に重ねて動画にする。目視確認用。
#
# 【何を見るためのものか】
#   1. bbox が目標に付いているか（誤検出・取りこぼし）
#   2. **接地点（bbox 下辺中央）が実際の接地位置とどれだけずれているか** ← 本命。
#      合成データでは cyclist で 0.9 m の系統誤差が出ており、支配的な誤差要因（README §1）
#   3. **track_id がフレームをまたいで維持されているか** ← 5 fps では切れる懸念がある。
#      ByteTrack は 30 fps 前提なので、0.2 s で 1 m 動く自転車では IoU が繋がらない場合がある
#
# 依存: opencv-python, pandas
#
# 使い方:
#   python overlay_detections.py frames/ detections.csv --out overlay.mp4
#   python overlay_detections.py video.mp4 detections.csv --scale 0.5

import argparse
from pathlib import Path

import cv2
import pandas as pd

import sys
from pathlib import Path

# lib/ を import 可能にする。各スクリプトは直接実行される前提なので、
# パッケージ化せずパスを通す方式にしている
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from detections import CLS_CYCLIST, CLS_VEHICLE, COCO_BICYCLE, COCO_MOTORCYCLE, COCO_PERSON, VEHICLE_COCO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FRAME_PERIOD_S = 0.2

# BGR。cyclist を構成する person/bicycle は同系色にして、統合前でも対応が読めるようにする
COLORS = {COCO_PERSON: (80, 200, 80), COCO_BICYCLE: (40, 255, 160),
          COCO_MOTORCYCLE: (40, 255, 160)}
VEHICLE_COLOR = (60, 160, 255)
NAMES = {COCO_PERSON: "person", COCO_BICYCLE: "bicycle", COCO_MOTORCYCLE: "motorcycle",
         2: "car", 5: "bus", 7: "truck"}


def iter_frames(src: Path):
    """連番画像ディレクトリでも動画でも (index, BGR 画像) を返す"""
    if src.is_dir():
        for i, p in enumerate(sorted(q for q in src.iterdir() if q.suffix.lower() in IMG_EXT)):
            yield i, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(src))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield i, im
            i += 1
        cap.release()


def draw(im, rows: pd.DataFrame, thick: int, fscale: float):
    """1 フレーム分の bbox・接地点・ラベルを描く。im を破壊的に更新する"""
    for r in rows.itertuples():
        color = VEHICLE_COLOR if r.cls in VEHICLE_COCO else COLORS.get(r.cls, (200, 200, 200))
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(im, p1, p2, color, thick)
        # 接地点。ラベルの正本になる一点なので、bbox より目立たせる
        cv2.drawMarker(im, (int(r.foot_u), int(r.foot_v)), (0, 0, 255),
                       cv2.MARKER_CROSS, 20 * thick, thick + 1)
        label = f"{NAMES.get(r.cls, r.cls)} #{r.track_id} {r.conf:.2f}"
        # 枠の上に置くと画面外に出ることがあるので、上端に近ければ枠の内側に入れる
        ly = p1[1] - 6 if p1[1] > 24 * fscale else p1[1] + int(28 * fscale)
        cv2.putText(im, label, (p1[0], ly),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="連番画像ディレクトリ または 動画")
    ap.add_argument("csv", type=Path, help="detect_yolo.py が出した検出 CSV")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dt", type=float, default=FRAME_PERIOD_S,
                    help="フレーム間隔[s]。fps = 1/dt。既定 0.2 で実時間と一致する")
    ap.add_argument("--scale", type=float, default=1.0, help="4K が重いとき 0.5 など")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    by_frame = dict(tuple(df.groupby("frame")))
    out = args.out or args.src.with_suffix("").with_name(args.src.stem + "_overlay.mp4")

    vw, n = None, 0
    for i, im in iter_frames(args.src):
        if args.scale != 1.0:
            im = cv2.resize(im, None, fx=args.scale, fy=args.scale)
        rows = by_frame.get(i)
        if rows is not None:
            r = rows.copy()
            for c in ["x1", "y1", "x2", "y2", "foot_u", "foot_v"]:
                r[c] *= args.scale               # CSV は元解像度の座標なので合わせる
            # 線の太さと文字は解像度に比例させる（4K で 1px の枠は見えない）
            t = max(1, int(round(im.shape[1] / 960)))
            draw(im, r, thick=t, fscale=0.6 * t)
        if vw is None:
            h, w = im.shape[:2]
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 1.0 / args.dt, (w, h))
        vw.write(im)
        n += 1
    vw.release()

    print(f"{out}: {n} フレーム, {1/args.dt:.1f} fps, {n*args.dt:.1f} 秒")
    print(f"  検出 {len(df)} 件 / 追跡 ID {df['track_id'].nunique()} 個 "
          f"/ 検出のあったフレーム {df['frame'].nunique()}/{n}")
    # track_id が細切れなら 5 fps で追従できていない。レーダーとの軌跡照合に効く
    span = df.groupby("track_id")["frame"].agg(["min", "max", "count"])
    short = span[span["count"] < 3]
    if len(short):
        print(f"  ★ 3 フレーム未満の track_id が {len(short)} 個。"
              f"5 fps で追従できていない可能性（bytetrack.yaml の緩和を検討）")


if __name__ == "__main__":
    main()
