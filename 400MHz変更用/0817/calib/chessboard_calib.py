# チェスボード画像からカメラの内部パラメータ K・歪み係数を求める。
#
# 【役割】0817/lib/homography.py の pose_from_homography(H, K) が要求する K を作る側。
# ここまでは K が無かったため、地上座標(X,Y)は出せてもh・俯角の分解ができなかった
# （estimate_homography 自体は K 不要で動くため、ラベル生成の本筋は止まっていない）。
#
# 【内側コーナー数について】
# pattern_size はマス目の数ではなく「マス目の交点（内側コーナー）」の数。
# 縦7マス×横10マスのボードなら、内側コーナーは 縦6×横9 になる。
#
# 【square_size は印刷後に定規で実測した値を使うこと】
# プリンタの拡大縮小設定が入っていると指定寸法通りに印刷されないため、
# 較正の物理スケール（ひいては h・俯角の値）が丸ごとずれる。
#
# 依存: opencv-python, numpy
#
# 使い方:
#   python chessboard_calib.py images/*.jpg --cols 9 --rows 6 --square-size 0.024 --out K.npz

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def calibrate(image_paths: list[Path], pattern_size: tuple[int, int], square_size: float):
    """チェスボード画像群から (K, dist, rms, per_image_errors) を求める。

    objp はボード自身の座標系（Z=0平面、単位 m）。画像ごとに検出したコーナーの
    画素座標と対応付け、cv2.calibrateCamera で内部パラメータ・歪み係数を解く"""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points, img_points, used = [], [], []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is None:
            print(f"  [警告] 読み込めない: {p}")
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            print(f"  [検出失敗] {p.name}")
            continue
        # サブピクセル精度化。コーナー検出の粗い格子点を輝度勾配で追い込む
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(p)

    if len(obj_points) < 3:
        raise RuntimeError(f"検出成功が {len(obj_points)} 枚のみ。較正には最低でも数枚必要"
                            "（10枚以上を推奨）")

    h, w = gray.shape
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    # 画像ごとの再投影誤差（点あたりの平均ユークリッド距離[px]）。
    # 特定の画像だけ大きければ、その画像（角度・ピント・検出ミス）を疑う。
    # cv2.norm は shape/type に敏感なので使わず、numpy で直接計算する
    per_image_errors = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        diff = img_points[i].reshape(-1, 2) - proj.reshape(-1, 2)
        err = float(np.linalg.norm(diff, axis=1).mean())
        per_image_errors.append(err)

    return K, dist, rms, used, per_image_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="チェスボード画像（glob可）")
    ap.add_argument("--cols", type=int, required=True, help="内側コーナー数（横）")
    ap.add_argument("--rows", type=int, required=True, help="内側コーナー数（縦）")
    ap.add_argument("--square-size", type=float, required=True,
                    help="1マスの辺の長さ[m]。印刷後に定規で実測した値を使うこと")
    ap.add_argument("--out", type=Path, default=Path("chessboard_calib.npz"))
    args = ap.parse_args()

    paths = sorted(Path(p) for pattern in args.images for p in glob.glob(pattern))
    if not paths:
        raise SystemExit("画像が見つかりません")
    print(f"{len(paths)} 枚を読み込みます")

    K, dist, rms, used, errors = calibrate(paths, (args.cols, args.rows), args.square_size)

    print(f"\n検出成功: {len(used)}/{len(paths)} 枚")
    print(f"RMS再投影誤差: {rms:.4f} px  （目安: 1px未満なら良好、1〜2pxは許容範囲）")
    worst = sorted(zip(used, errors), key=lambda x: -x[1])[:3]
    print("誤差が大きい画像 上位3枚:")
    for p, e in worst:
        print(f"  {p.name}: {e:.4f} px")

    print(f"\nカメラ行列 K =\n{K}")
    print(f"\n歪み係数 (k1,k2,p1,p2,k3) = {dist.ravel()}")

    np.savez(args.out, K=K, dist=dist, rms=rms,
             pattern_size=(args.cols, args.rows), square_size=args.square_size)
    print(f"\n保存: {args.out}")
    print("→ 0817/lib/homography.py の pose_from_homography(H, K) にこの K を渡すと h・俯角が出る")


if __name__ == "__main__":
    main()
