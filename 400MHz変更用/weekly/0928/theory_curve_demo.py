# 理論曲線の形が、走路に対するレーダーの位置でどう変わるかを示す説明用の図。実データは使わない。
#
#   R(s) = sqrt( (s - X0)^2 + d^2 )     s: 走路に沿った位置[m], X0: 最接近点, d: 横方向離隔
#
# X0 が走路の外（端）にあれば R は単調、走路の中にあれば U 字になる。
# どちらになるかは現地でレーダーの投影点がどこに来るかで決まるので、走路の取り方で選べる。
#
# ライン識別に効くのは R の絶対値ではなく、ライン間の差 ΔR。これは最接近点の近くで最大になり、
# 離れるほど縮む。したがって走路のどこに最接近点を置くかが設計上の選択になる。

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry import RANGE_RES_M, slant_range

H, Y_R = 15.0, 6.0              # 設置高[m] / セットバック[m]（いずれも仮。今夜実測）
L = 26.0                        # 総走行距離[m]
EVAL = (3.0, 23.0)              # 評価区間[m]

CONFIGS = [("A: closest approach at path END  (X0 = 0 m)", 0.0),
           ("B: closest approach at path CENTER  (X0 = 13 m)", 13.0)]

s = np.linspace(0, L, 400)                                   # (S,) 走路に沿った位置

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, (title, x0) in enumerate(CONFIGS):
    ax_r, ax_d = axes[0, col], axes[1, col]
    for y, c in ((0.5, "C0"), (3.5, "C1")):
        r = slant_range(s - x0, y, Y_R, H)                   # (S,)
        ax_r.plot(s, r, c, lw=2, label=f"line Y={y} m")

    d_r = (slant_range(s - x0, 3.5, Y_R, H)
           - slant_range(s - x0, 0.5, Y_R, H))               # (S,) ライン間の差
    ax_d.plot(s, d_r, "C2", lw=2)
    ax_d.axhline(RANGE_RES_M, color="k", ls="--", lw=1.2)
    ax_d.text(L, RANGE_RES_M, " range res 0.846 m", fontsize=8.5, ha="right", va="bottom")

    for ax in (ax_r, ax_d):
        ax.axvspan(*EVAL, color="gray", alpha=0.10)
        ax.axvline(x0, color="darkred", ls=":", lw=1.2)
        ax.set_xlabel("position along path  s [m]")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, L)
    ax_r.set_title(title)
    ax_r.set_ylabel("slant range R [m]")
    ax_r.legend(fontsize=9)
    ax_d.set_ylabel(r"$\Delta R$ between lines [m]")
    ax_d.set_ylim(0, 1.6)

axes[0, 0].text(1.0, slant_range(20, 0.5, Y_R, H), "evaluation section (gray)",
                fontsize=8.5, color="gray")
fig.suptitle(f"Theory curve: shape depends on where the closest approach falls  "
             f"(h={H:.0f} m, $y_r$={Y_R:.0f} m)")
fig.tight_layout()
fig.savefig("theory_curve_demo.png", dpi=130)
print("-> theory_curve_demo.png")
