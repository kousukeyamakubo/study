# チェスボード較正の「どこまで信用できるか」を可視化する。
#
# 【背景】RMS再投影誤差は「チェスボードのコーナーが実際に検出できた画素範囲」でしか
# 計算できない。chessboard_calib.pyのRMS値が小さくても、それは較正点が届いた範囲の
# 当てはまりの良さでしかなく、そこより外側（特に画面四隅）は歪みモデルの外挿になる。
# 「較正できている」と言うには、誤差の小ささだけでなく、その低誤差が画面のどこまで
# 実際に検証されているか（＝コーナーがどこまで届いているか）を separately 確認する必要がある。
#
# 出力する3パネル:
#   1) 画面全体でのコーナー分布（誤差で色付け）。網掛け＝コーナーが一度も届いていない領域
#   2) 光学中心からの半径 vs 再投影誤差（ビン平均±標準偏差）。半径とともに誤差が増加し続けた
#      まま観測データが尽きている場合、四隅までの外挿は信用できない
#   3) 周辺まで最もボードが写ったサンプル画像のundistort前後（この範囲限定の目視確認）
#
# 依存: opencv-python, numpy, matplotlib
#
# 使い方:
#   python calib_diagnose.py chessboard_images/*.jpg --cols 9 --rows 6 --square-size 0.024

import argparse
import glob
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def calibrate_with_corners(image_paths, pattern_size, square_size):
    """chessboard_calib.calibrate()と同じ較正処理だが、コーナーごとの誤差を
    可視化するため img_points・rvecs・tvecs もあわせて返す"""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points, img_points, used = [], [], []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(p)

    if len(obj_points) < 3:
        raise RuntimeError(f"検出成功が {len(obj_points)} 枚のみ。較正には最低でも数枚必要")

    h, w = gray.shape
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)
    return K, dist, rms, (w, h), obj_points, img_points, rvecs, tvecs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="チェスボード画像（glob可）")
    ap.add_argument("--cols", type=int, required=True, help="内側コーナー数（横）")
    ap.add_argument("--rows", type=int, required=True, help="内側コーナー数（縦）")
    ap.add_argument("--square-size", type=float, required=True, help="1マスの辺の長さ[m]")
    ap.add_argument("--out", type=Path, default=Path("calib_coverage_diagnosis.png"))
    args = ap.parse_args()

    paths = sorted(Path(p) for pattern in args.images for p in glob.glob(pattern))
    if not paths:
        raise SystemExit("画像が見つかりません")

    (K, dist, rms, (w, h), obj_points, img_points, rvecs, tvecs, used
     ) = calibrate_with_corners(paths, (args.cols, args.rows), args.square_size)
    cx, cy = K[0, 2], K[1, 2]

    # 全コーナーについて、再投影誤差と光学中心からの半径をまとめる
    all_u, all_v, all_err, all_r = [], [], [], []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        pts = img_points[i].reshape(-1, 2)
        prj = proj.reshape(-1, 2)
        err = np.linalg.norm(pts - prj, axis=1)
        r = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        all_u.append(pts[:, 0]); all_v.append(pts[:, 1])
        all_err.append(err); all_r.append(r)

    all_u = np.concatenate(all_u); all_v = np.concatenate(all_v)
    all_err = np.concatenate(all_err); all_r = np.concatenate(all_r)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # Panel 1: 画面全体でのコーナー分布。網掛け＝コーナーが一度も届いていない領域
    ax = axes[0]
    ax.set_facecolor("#f2f2f2")
    u_min, u_max = all_u.min(), all_u.max()
    v_min, v_max = all_v.min(), all_v.max()
    ax.axvspan(0, u_min, color="#ffcccc", alpha=0.5, zorder=0)
    ax.axvspan(u_max, w, color="#ffcccc", alpha=0.5, zorder=0)
    ax.axhspan(0, v_min, color="#ffcccc", alpha=0.5, zorder=0)
    ax.axhspan(v_max, h, color="#ffcccc", alpha=0.5, zorder=0)
    sc = ax.scatter(all_u, all_v, c=all_err, cmap="viridis", s=8,
                     vmin=0, vmax=max(2.0, all_err.max()))
    ax.plot(cx, cy, "r+", markersize=15, markeredgewidth=2, label="principal point (cx,cy)")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Corner coverage & reprojection error\n(red shade = never reached by chessboard = unverified)")
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
    ax.legend(loc="upper right", fontsize=8)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("reprojection error [px]")

    # Panel 2: 半径 vs 誤差。ビン平均±標準偏差（分散のみだと大きさの直感がつかみにくいため
    # 誤差と同じ[px]単位である標準偏差を使う）
    ax = axes[1]
    ax.scatter(all_r, all_err, s=6, alpha=0.4)
    bins = np.linspace(0, all_r.max(), 15)
    idx = np.digitize(all_r, bins)
    valid_bins = [i for i in range(1, len(bins)) if (idx == i).any()]
    means = np.array([all_err[idx == i].mean() for i in valid_bins])
    stds = np.array([all_err[idx == i].std() for i in valid_bins])
    counts = np.array([(idx == i).sum() for i in valid_bins])
    centers = np.array([(bins[i - 1] + bins[i]) / 2 for i in valid_bins])
    ax.fill_between(centers, means - stds, means + stds, color="red", alpha=0.15,
                     label="mean ± 1 std")
    ax.plot(centers, means, "r-o", markersize=4, label="binned mean")
    for xc, yc, n in zip(centers, means + stds, counts):
        ax.annotate(f"n={n}", (xc, yc), textcoords="offset points", xytext=(0, 4),
                    fontsize=6, ha="center", color="gray")
    max_r_observed = all_r.max()
    corner_r = np.linalg.norm([[0, 0], [w, 0], [0, h], [w, h]] - np.array([cx, cy]), axis=1).max()
    ax.axvline(max_r_observed, color="orange", ls="--",
               label=f"max observed radius ({max_r_observed:.0f}px)")
    ax.axvline(corner_r, color="red", ls="--",
               label=f"radius to frame corner ({corner_r:.0f}px)")
    ax.set_xlabel("radius from principal point r [px]"); ax.set_ylabel("reprojection error [px]")
    ax.set_title("error vs radius\n(right of orange line = pure extrapolation, e.g. k3)")
    ax.legend(fontsize=8)

    # Panel 3: 最も周辺までボードが写ったサンプルでのundistort前後（目視確認）
    areas = [(img_points[i].reshape(-1, 2)[:, 0].max() - img_points[i].reshape(-1, 2)[:, 0].min()) *
             (img_points[i].reshape(-1, 2)[:, 1].max() - img_points[i].reshape(-1, 2)[:, 1].min())
             for i in range(len(used))]
    best_idx = int(np.argmax(areas))
    sample_path = used[best_idx]
    im = cv2.imread(str(sample_path))
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0)
    und = cv2.undistort(im, K, dist, None, newK)
    combo = np.hstack([
        cv2.resize(im, (w // 3, h // 3)),
        cv2.resize(und, (w // 3, h // 3)),
    ])
    ax = axes[2]
    ax.imshow(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
    ax.axvline(w // 3, color="white", lw=2)
    ax.set_title(f"before / after undistort\n{sample_path.name} (center region only, board never reached the edges)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)

    print(f"saved: {args.out}")
    print(f"corner coverage: u[{u_min:.0f},{u_max:.0f}]/{w} v[{v_min:.0f},{v_max:.0f}]/{h}")
    print(f"RMS reprojection error: {rms:.3f}px, max per-corner error: {all_err.max():.3f}px")
    print(f"observed max radius: {max_r_observed:.0f}px, frame-corner radius: {corner_r:.0f}px "
          f"({100 * max_r_observed / corner_r:.0f}% covered)")
    print("\nradius bin: mean±std [px] (n)")
    for xc, m, s, n in zip(centers, means, stds, counts):
        print(f"  r~{xc:5.0f}px: {m:.3f}±{s:.3f}  (n={n})")


if __name__ == "__main__":
    main()
