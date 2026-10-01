# 設置幾何を扱う。走行ラインごとの横方向離隔 d(Y) を出すのが目的。
#
# d(Y) = sqrt((Y + y_r)^2 + h^2)      y_r: セットバック[m], h: 設置高[m]
#
# 【測り方】y_r は地上で巻尺（壁面→手前側の路肩）、h はレーザーを窓から真下に向ける。
# どちらも狙いを外しにくく、d への効き方も穏やか（∂d/∂h ≈ 0.94, ∂d/∂y_r ≈ 0.4）。
#
# コーンまでの斜距離から y_r を解く案は捨てた。
#   y_r = (d_far^2 - d_near^2 - 16)/8 は大きさの近い2数の差から小さい量を取り出す形で、
#   誤差の増幅率が d/4 ≈ 4.5 倍ある。5F から 20m 先のコーンを手持ちで狙う誤差 ±0.5m は
#   y_r の ±3m に化け、y_r 自体（6m 程度）と同じ桁になって無意味になる。
#
# 使い方:
#   python solve_site_geometry.py --y-r 6.0 --h 15.0          # 実測値から d(Y) を出す
#   python solve_site_geometry.py --d-hat 16.37 17.76         # 当てはめ結果から y_r を逆算

import argparse

import numpy as np

LINES = (0.5, 3.5)          # パイロットの走行ライン


def d_of_line(y, y_r, h):
    """走行ライン Y の横方向離隔 d [m]"""
    return float(np.hypot(y + y_r, h))


def y_r_from_d_hat(d0, d1, lines=LINES):
    """2ラインの当てはめ結果 d_hat から y_r を逆算する。

    d1^2 - d0^2 = (Y1 + y_r)^2 - (Y0 + y_r)^2 = (Y1^2 - Y0^2) + 2 y_r (Y1 - Y0)

    こちらは差を取るので桁落ちするが、d̂ は約50フレームの当てはめ結果であり
    1点の測距より精度が高いので実用になる。巻尺の実測値との照合に使う。
    """
    y0, y1 = lines
    return ((d1**2 - d0**2) - (y1**2 - y0**2)) / (2 * (y1 - y0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-r", type=float, help="セットバック[m]（地上で巻尺）")
    ap.add_argument("--h", type=float, help="設置高[m]（レーザーを真下へ）")
    ap.add_argument("--d-hat", type=float, nargs=2, metavar=("D_Y05", "D_Y35"),
                    help="当てはめで得た d̂（Y=0.5, 3.5 の順）。y_r を逆算する")
    args = ap.parse_args()

    if args.y_r is not None and args.h is not None:
        print(f"実測 y_r = {args.y_r:.2f} m, h = {args.h:.2f} m")
        ds = [d_of_line(y, args.y_r, args.h) for y in LINES]
        for y, d in zip(LINES, ds):
            print(f"  Y = {y:.1f} m -> d = {d:.3f} m")
        print(f"  ライン間の差 Δd = {abs(ds[1] - ds[0]):.3f} m")
        print("\n  ※ これが fit_trajectory の d_hat の照合先（主張1）")

    if args.d_hat:
        d0, d1 = args.d_hat
        y_r = y_r_from_d_hat(d0, d1)
        print(f"\n当てはめ結果からの逆算: d_hat = {d0:.3f}, {d1:.3f} m")
        print(f"  y_r = {y_r:.2f} m")
        if args.h is not None:
            h_est = np.sqrt(max(d0**2 - (LINES[0] + y_r)**2, 0.0))
            print(f"  h   = {h_est:.2f} m（実測 {args.h:.2f} m との差 {h_est - args.h:+.2f} m）")
        if args.y_r is not None:
            print(f"  巻尺の実測 {args.y_r:.2f} m との差 {y_r - args.y_r:+.2f} m")

    if args.y_r is None and args.d_hat is None:
        ap.error("--y-r と --h、または --d-hat のいずれかを指定してください")


if __name__ == "__main__":
    main()
