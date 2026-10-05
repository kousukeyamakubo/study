# O2: 走行ライン間の差が、レーダーの分解能で見えるかを事前に見積もる。
#
# 9/29 の下見より前に知りたいこと:
#   (a) 外側2ライン（Y = 0.5 / 3.5 m）の斜距離差 ΔR が距離分解能 0.846 m を超える X の範囲
#   (b) 同じくドップラー差 Δv_r が速度分解能 0.5995 m/s を超えるか
#   (c) セットバック y_r と設置高 h をどう選べば (a) の範囲が広がるか
#
# ここが「全域で見えない」と出るなら、10/2 の主張(2) は撮る前に No 側に倒れることが
# 分かる。その場合はライン4本を引く意味が薄く、角度推定が使えるかの確認（README 未決1）
# を 9/29 の最優先に繰り上げることになる。
#
# 使い方:
#   python line_separability.py                       # 既定のスイープ
#   python line_separability.py --h 15 --y-r 3 --v 2.5

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry import RANGE_RES_M, VEL_RES_MS, radial_velocity, slant_range

Y_INNER, Y_OUTER = 0.5, 3.5          # パイロットで使う外側2ライン
X_MIN, X_MAX = 3.0, 23.0             # 評価区間（近距離端 3 m は暫定）


def separable_span(x, delta, res):
    """delta > res となる X の区間を (下端, 上端) で返す。無ければ None"""
    ok = delta > res
    if not ok.any():
        return None
    return float(x[ok].min()), float(x[ok].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, nargs="+", default=[13.0, 15.0, 17.0],
                    help="設置高 [m]。既定は 0727 の見積もり h≈15 m を中心に振る")
    ap.add_argument("--y-r", type=float, nargs="+", default=[0.0, 3.0, 6.0],
                    help="道路手前側境界からのセットバック [m]")
    ap.add_argument("--v", type=float, default=2.5, help="走行速度 [m/s]（中速）")
    ap.add_argument("--out", default="line_separability.png")
    args = ap.parse_args()

    x = np.linspace(X_MIN, X_MAX, 401)                              # (X,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    print(f"outer lines Y={Y_INNER} / {Y_OUTER} m, v={args.v} m/s")
    print(f"{'h[m]':>5} {'y_r[m]':>7} {'dR@3m':>8} {'dR@23m':>8} "
          f"{'X where dR>0.846':>18} {'dv@3m':>8} {'dv@23m':>8}")

    for h in args.h:
        for y_r in args.y_r:
            d_r = slant_range(x, Y_OUTER, y_r, h) - slant_range(x, Y_INNER, y_r, h)
            d_v = np.abs(radial_velocity(args.v, x, Y_OUTER, y_r, h)
                         - radial_velocity(args.v, x, Y_INNER, y_r, h))

            span = separable_span(x, d_r, RANGE_RES_M)
            span_s = "none" if span is None else f"{span[0]:.1f}-{span[1]:.1f} m"
            print(f"{h:5.0f} {y_r:7.0f} {d_r[0]:8.3f} {d_r[-1]:8.3f} "
                  f"{span_s:>18} {d_v[0]:8.4f} {d_v[-1]:8.4f}")

            label = f"h={h:.0f}, y_r={y_r:.0f}"
            axes[0].plot(x, d_r, label=label)
            axes[1].plot(x, d_v, label=label)

    axes[0].axhline(RANGE_RES_M, color="k", ls="--", lw=1.5)
    axes[0].text(X_MAX, RANGE_RES_M, " range res 0.846 m", va="bottom", ha="right", fontsize=9)
    axes[0].set_ylabel(r"$\Delta R$ between lines [m]")

    axes[1].axhline(VEL_RES_MS, color="k", ls="--", lw=1.5)
    axes[1].text(X_MAX, VEL_RES_MS, " doppler res 0.5995 m/s", va="top", ha="right", fontsize=9)
    axes[1].set_ylabel(r"$\Delta v_r$ between lines [m/s]")
    axes[1].set_ylim(0, VEL_RES_MS * 1.2)

    for ax in axes:
        ax.set_xlabel("X [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Line separability (Y={Y_INNER} vs {Y_OUTER} m, v={args.v} m/s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
