# レーンによって目標の見え方が違う理由が、幾何（距離・角度）で説明できるかを定量化する。
#
# 夜の距離−時間図では、レーン1（建物側）が最も濃く、レーン3（木の並ぶ側・レーダーに最も近い）は
# 徒歩でも自転車でもほぼ見えない。幾何で説明できるかは、奥の帯（R 40〜49 m）で判定する:
#   - 奥ではレーン間の R の差は 0.5 m 程度で、R^4 則の差は 1 dB 未満（しかもレーン3の方が近い＝強いはず）
#   - 見下ろし角の差は 0.3° 程度で、縦のビームの形では差を作れない
# したがって奥で 10 dB 級の差があれば、幾何以外（遮蔽）による。
#
# 目標の強度は、追跡（pick_ridge）を使わずに測る。追跡は別の筋や静止物の帯を拾うことがあるため。
# 代わりに、申告レーンの d を固定した等速の理論軌跡 R(t) = sqrt(v^2 (t - t0)^2 + d^2) を
# (v, t0) について総当たりし、軌跡上の強度（床からの超過）の平均が最大になるものを採る（整合フィルタ）。
# 総当たりの最大は雑音だけでも正に偏るので、時刻を反転した図に同じ探索を掛けた値を対照にし、
# 「対照からの上乗せ」を目標の強度とする（反転すると本物の筋は向きが合わず拾われない）。
#
# 使い方:
#   python lane_visibility.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_night import L_C1_C5, N_MAIN, ROOT, S0, TAGS, condition, lane_delta_d, npz_path
from atlas_ridge_track import moving_power_db
from plot_lane_truth import LANE_OFFSET, lane_range

BANDS = {"far": (40.0, 49.0), "mid": (30.0, 40.0)}
V_GRID = np.arange(0.8, 5.01, 0.05)       # 徒歩〜自転車の走行速度 [m/s]
T0_GRID = np.arange(-40.0, 70.0, 0.2)     # 最接近時刻 [s]。収録 30 s の外も許す（片側しか通らない走行）


def geometry_far_end():
    """奥の端（C5/C6 の並び）でのレーンごとの R、見下ろし角、方位（走路方向から）"""
    a, h, _ = lane_delta_d(road_w=4.0)
    p = L_C1_C5 - S0                                          # 垂線の足から奥の端までの走路方向の距離
    out = {}
    for ln in (1, 2, 3):
        x = a + LANE_OFFSET[ln]                               # レーダー直下からの横方向の水平距離
        r = float(np.sqrt(p**2 + x**2 + h**2))
        out[ln] = dict(r=r, elev=float(np.degrees(np.arcsin(h / r))), azim=float(np.degrees(np.arctan2(x, p))))
    return out


def matched_score(ex, rng, t, d, sign, band):
    """申告レーンの d に固定した等速軌跡を総当たりし、band 内での軌跡上の床超過の平均の最大を返す。
    ex: (F, R) 床超過 dB。sign: 接近 -1 / 離反 +1（時間とともに R が減る / 増える）"""
    dr = rng[1] - rng[0]
    # ±1 ビンの最大を取っておき、軌跡が量子化で 1 ビンずれても拾えるようにする
    exd = np.max(np.stack([np.roll(ex, k, axis=1) for k in (-1, 0, 1)]), axis=0)   # (F, R)
    best = -np.inf
    for v in V_GRID:
        s = v * (t[None, :] - T0_GRID[:, None])               # (T0, F) 走路上の位置（垂線の足から）
        # 接近は t < t0 で奥から来る（s < 0 の側）、離反は t > t0 で奥へ行く（s > 0 の側）
        valid = (s * sign) >= 0
        r = np.hypot(s, d)                                     # (T0, F)
        inb = valid & (r >= band[0]) & (r <= band[1])
        idx = np.clip(np.round((r - rng[0]) / dr).astype(int), 0, len(rng) - 1)
        val = exd[np.arange(len(t))[None, :], idx]             # (T0, F)
        cnt = inb.sum(axis=1)
        sc = np.where(cnt >= 5, (val * inb).sum(axis=1) / np.maximum(cnt, 1), -np.inf)
        best = max(best, float(sc.max()))
    return best


def main():
    geo = geometry_far_end()
    print("(1) 奥の端（C5/C6 の並び）でのレーンごとの幾何")
    print(f"{'レーン':>4} {'R[m]':>7} {'見下ろし角[°]':>12} {'方位[°]':>8}")
    for ln in (1, 2, 3):
        g = geo[ln]
        print(f"{ln:4d} {g['r']:7.2f} {g['elev']:12.2f} {g['azim']:8.2f}")
    print(f"    レーン1→3 の差: R^4 則 {40 * np.log10(geo[1]['r'] / geo[3]['r']):+.2f} dB"
          f"（レーン3 の方が強いはず）、見下ろし角 {geo[3]['elev'] - geo[1]['elev']:+.2f}°、"
          f"方位 {geo[3]['azim'] - geo[1]['azim']:+.2f}°")

    rows = []
    for n in range(1, N_MAIN + 1):
        mode, lane, dr = condition(n)
        d = lane_range(lane)[0]
        sign = -1.0 if dr == "app" else +1.0
        pw, _, rng, t, *_ = moving_power_db(npz_path(TAGS[n - 1]))   # (F, R)
        ex = pw - np.median(pw, axis=0, keepdims=True)                # (F, R) 床超過
        res = {}
        for name, band in BANDS.items():
            on = matched_score(ex, rng, t, d, sign, band)
            ctrl = matched_score(ex[::-1], rng, t, d, sign, band)     # 時刻反転 = 向きが逆の軌跡しか合わない
            res[name] = on - ctrl
        rows.append(dict(n=n, mode=mode, lane=lane, dr=dr, **res))

    print("\n(2) 軌跡上の目標強度（時刻反転の対照からの上乗せ [dB]）")
    print(f"{'No.':>4} {'手段':>5} {'レーン':>4} {'向き':>4} {'奥 40-49m':>10} {'中 30-40m':>10}")
    for r in rows:
        print(f"{r['n']:4d} {r['mode']:>5} {r['lane']:4d} {r['dr']:>4} {r['far']:10.2f} {r['mid']:10.2f}")

    print("\n    レーン別の中央値 [dB]（各 6 本 = 3回 × 2方向）")
    print(f"{'手段':>5} {'レーン':>4} {'奥 40-49m':>10} {'中 30-40m':>10}")
    summ = {}
    for mode in ("walk", "bike"):
        for ln in (1, 2, 3):
            sel = [r for r in rows if r["mode"] == mode and r["lane"] == ln]
            summ[(mode, ln)] = {b: np.array([r[b] for r in sel]) for b in BANDS}
            print(f"{mode:>5} {ln:4d} {np.median(summ[(mode, ln)]['far']):10.2f} "
                  f"{np.median(summ[(mode, ln)]['mid']):10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    for ax, (b, band) in zip(axes, BANDS.items()):
        for k, mode in enumerate(("walk", "bike")):
            for ln in (1, 2, 3):
                y = summ[(mode, ln)][b]
                x = ln + (k - 0.5) * 0.25
                ax.plot(np.full(len(y), x), y, "o", ms=4, alpha=0.6, color=["tab:gray", "tab:red"][k],
                        label=mode if ln == 1 else None)
                ax.plot([x - 0.08, x + 0.08], [np.median(y)] * 2, color="k", lw=2)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks([1, 2, 3], ["lane1\n(building side)", "lane2", "lane3\n(tree side, nearest)"])
        ax.set_title(f"R {band[0]:.0f}-{band[1]:.0f} m", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("target power on matched path\nminus time-reversed control [dB]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Target visibility by lane (bar: median of 6 runs). "
                 f"Geometry alone predicts lane3 {40 * np.log10(geo[1]['r'] / geo[3]['r']):+.1f} dB vs lane1 at far end",
                 fontsize=10)
    out = ROOT / "figures/lane_visibility.png"
    fig.savefig(out, dpi=120)
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
