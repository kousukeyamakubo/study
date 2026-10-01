# レーザーで実測したコーンまでの斜距離から、設置幾何を解く。
#
# 設置高 h を直接測るのは 5F では面倒なので、コーン6本の斜距離だけから求める。
#
# 道の手前側(Y=y_near)と対面(Y=y_far)に同じ走路位置 s でコーンを置くと、
#   R_far^2 - R_near^2 = (y_far + y_r)^2 - (y_near + y_r)^2
# となり (s - X0)^2 が消える。つまり横断ペアはどの s でも同じ式を与えるので、
# 3ペアそれぞれから y_r が独立に出る（3つ揃えば測定の妥当性チェックになる）。
#
# y_r が決まれば d_near = sqrt(y_r^2 + h^2) から h が出て、
# 走路方向の3点に R^2 = (s - X0)^2 + d^2 を当てはめれば X0 も決まる。
#
# 使い方（s は走路に沿った位置[m]、距離は実測値[m]）:
#   python solve_site_geometry.py --s 3 13 23 --near 19.0 16.2 19.0 --far 20.6 18.0 20.6

import argparse

import numpy as np

Y_NEAR, Y_FAR = 0.0, 4.0        # コーンを置く横断位置[m]（道の両端）
LINES = (0.5, 3.5)              # パイロットの走行ライン


def solve_y_r(r_near, r_far, y_near=Y_NEAR, y_far=Y_FAR):
    """横断ペアから y_r を出す。ペアごとに1個返す (P,)"""
    # R_far^2 - R_near^2 = (y_far^2 - y_near^2) + 2 y_r (y_far - y_near)
    num = (np.asarray(r_far, float)**2 - np.asarray(r_near, float)**2) - (y_far**2 - y_near**2)
    return num / (2 * (y_far - y_near))


def fit_along_path(s, r, ):
    """走路方向の3点に R^2 = (s-X0)^2 + d^2 を当てはめる。戻り値 (X0, d)"""
    s, r = np.asarray(s, float), np.asarray(r, float)
    a = np.stack([-2 * s, np.ones_like(s)], axis=1)          # (P, 2)
    (x0, b), *_ = np.linalg.lstsq(a, r**2 - s**2, rcond=None)
    return float(x0), float(np.sqrt(max(b - x0**2, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, nargs="+", required=True, help="コーンの走路位置[m]")
    ap.add_argument("--near", type=float, nargs="+", required=True, help="手前側コーンまでの斜距離[m]")
    ap.add_argument("--far", type=float, nargs="+", required=True, help="対面コーンまでの斜距離[m]")
    ap.add_argument("--y-near", type=float, default=Y_NEAR)
    ap.add_argument("--y-far", type=float, default=Y_FAR)
    args = ap.parse_args()

    y_r_each = solve_y_r(args.near, args.far, args.y_near, args.y_far)   # (P,)
    print("横断ペアごとの y_r [m]: " + "  ".join(f"{v:.2f}" for v in y_r_each))
    print(f"  平均 {y_r_each.mean():.2f}  ばらつき(sd) {y_r_each.std(ddof=1):.2f}"
          if len(y_r_each) > 1 else "")
    y_r = float(y_r_each.mean())

    x0, d_near = fit_along_path(args.s, args.near)
    h2 = d_near**2 - (args.y_near + y_r)**2
    h = float(np.sqrt(max(h2, 1e-9)))

    print(f"\n設置幾何")
    print(f"  セットバック y_r = {y_r:.2f} m")
    print(f"  設置高       h   = {h:.2f} m")
    print(f"  最接近点     X0  = {x0:.2f} m（走路に沿った位置）")
    print(f"  手前側までの離隔 d_near = {d_near:.2f} m")

    print(f"\n走行ラインごとの横方向離隔 d(Y)（当てはめ結果の照合先）")
    ds = []
    for y in LINES:
        d = float(np.hypot(y + y_r, h))
        ds.append(d)
        print(f"  Y = {y:.1f} m -> d = {d:.3f} m")
    print(f"  ライン間の差 Δd = {abs(ds[1] - ds[0]):.3f} m")

    if not 0.0 <= x0 <= max(args.s) + 3:
        print("\n[注意] X0 が走路の外にあります。最接近点が走路内に入らない配置です"
              "（理論曲線は単調になり、ライン差は近い側ほど大きくなります）")


if __name__ == "__main__":
    main()
