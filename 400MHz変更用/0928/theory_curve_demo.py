# 理論曲線とは何かを示す説明用の図。実データは使わない。
#
# 走者が走路（直線）を等速で動くとき、レーダーまでの斜距離は
#   R(t) = sqrt( (s(t) - X0)^2 + d^2 )
# で決まる。s は走路に沿った位置、X0 は最接近点、d は横方向離隔。
# 近づく→最接近→遠ざかる、の V 字を底で丸めた形になる。
#
# 10/2 の図(1) は、この曲線を観測ピーク列に重ねたもの。
# 図(2) は、ライン違い（d 違い）で曲線がどれだけ離れるかを見るもの。

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry import slant_range

H, Y_R, V = 15.0, 6.0, 2.5      # 設置高[m] / セットバック[m] / 走行速度[m/s]（いずれも仮）
X0 = 13.0                        # 最接近点（走路の中央と仮定）

t = np.linspace(0, 26 / V, 400)  # 26 m を走り切るまで
s = V * t                        # 走路に沿った位置[m]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for y, c in ((0.5, "C0"), (3.5, "C1")):
    r = slant_range(s - X0, y, Y_R, H)                       # (T,)
    axes[0].plot(t, r, c, lw=2, label=f"line Y={y} m")
    axes[1].plot(t, np.gradient(r, t), c, lw=2, label=f"line Y={y} m")

# 観測はこういう形で乗る、という雰囲気（説明用のダミー点）
rng = np.random.default_rng(0)
r_obs = slant_range(s - X0, 0.5, Y_R, H)
pick = np.arange(0, len(t), 18)
axes[0].plot(t[pick], np.round(r_obs[pick] / 0.846) * 0.846 + rng.normal(0, 0.15, pick.size),
             "k.", ms=9, label="observed peak (illustrative)")

axes[0].axvline(X0 / V, color="gray", ls=":", lw=1)
axes[0].text(X0 / V, slant_range(0, 0.5, Y_R, H), "  closest approach", fontsize=9, color="gray")
axes[0].set_ylabel("slant range R [m]")
axes[0].set_title("(1) R(t): theory curve vs observation")

axes[1].axhline(0, color="k", lw=1)
axes[1].set_ylabel("radial velocity $v_r = dR/dt$ [m/s]")
axes[1].set_title("(2) $v_r(t)$: same information, differentiated")

for ax in axes:
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(f"Theory curve  (h={H:.0f} m, $y_r$={Y_R:.0f} m, v={V} m/s, closest approach at s={X0:.0f} m)")
fig.tight_layout()
fig.savefig("theory_curve_demo.png", dpi=130)
print("-> theory_curve_demo.png")
