# 縦の線の距離を自転車が通るとき、線の電力に自転車の電力が足されているか（加算）を確かめる。開・自転車（15:09:39）の 1 本。
#
# 考え方: 線はチャープごとに位相がばらばらで（../1011/README.md）、目標と相関しないので、電力で足し算になるはず。
#   距離ビンごとに「自転車が通った時刻の値」から「その距離の時間中央値（＝線＋床）」を真数で引けば、自転車ぶんの電力が残る。
#   加算なら、自転車ぶんは距離について滑らかで、線の距離でも隣の距離と同じくらいになる。
#   線に埋もれる・食われる（受信の飽和など）なら、線の距離だけ自転車ぶんが凹む。
#   電力和・MTI あり（チャープ × 8ch の電力和）は本当の電力なので足し算が成り立つ。ドップラー最大は参考として並べる。
#
# 使い方:
#   python -X utf8 bike_on_line.py > bike_on_line_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1011"))
from window_2x2 import run_maps                                  # noqa: E402
from range_profile import setup_font_ja                          # noqa: E402

FIG = HERE / "figures"
STEM = "150939"
LINES = [28.8, 33.9, 39.8]          # line_check_log.txt で線のある距離（33.9 m はこの走行では +0.4 dB と弱い）
R_LO, R_HI = 25.5, 47.5             # 筋がはっきりしている範囲（figures/window_2x2_sum_mti.png）
T_LO, T_HI = 1.5, 13.0             # 筋のある時間。外は雑音や 34.7 m の多重反射（10〜11 s）を拾って傾きがずれる
WIN_S = 1.0                         # 予測の通過時刻の前後 [s]。径方向 約 1.8 m/s で 1 ビン（0.85 m）に約 0.5 s いる
KINDS = {"sum_mti": "電力和・MTI あり", "dmax_mti": "ドップラー最大・MTI あり"}


def fit_track(m, kind):
    """フレームごとに (値 − 時間中央値) が最大の距離を拾い、R = a t + b を外れ値を落としながら当てはめる"""
    rng, t = m["rng"], m["t"]
    ex = m[kind] - np.median(m[kind], axis=0)                       # (F, R)
    keep_r = (rng >= R_LO) & (rng <= R_HI)
    sel = (t >= T_LO) & (t <= T_HI)
    tt = t[sel]
    rr = rng[keep_r][np.argmax(ex[sel][:, keep_r], axis=1)]         # (F',)
    ok = np.ones(len(tt), bool)
    for _ in range(3):
        a, b = np.polyfit(tt[ok], rr[ok], 1)
        ok = np.abs(rr - (a * tt + b)) < 1.5
    return a, b


def bike_excess(m, kind, a, b):
    """距離ビンごとに、通過時刻 ±WIN_S の最大値・時間中央値・自転車ぶん（真数で引いたもの）[dB]"""
    rng, t = m["rng"], m["t"]
    x = m[kind]                                                     # (F, R) dB
    med = np.median(x, axis=0)                                      # (R,) 線 ＋ 床
    rows = []
    for i in np.flatnonzero((rng >= R_LO) & (rng <= R_HI)):
        tc = (rng[i] - b) / a
        w = np.abs(t - tc) <= WIN_S
        peak = x[w, i].max()
        lin = 10 ** (peak / 10) - 10 ** (med[i] / 10)
        bike = 10 * np.log10(lin) if lin > 0 else np.nan
        rows.append((rng[i], tc, med[i], peak, bike))
    return np.array(rows)                                           # (R', 5)


def is_line(r):
    return min(abs(r - l) for l in LINES) < 0.5


def main():
    setup_font_ja()
    FIG.mkdir(exist_ok=True)
    m = run_maps(STEM)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for ax, (kind, ja) in zip(axes, KINDS.items()):
        a, b = fit_track(m, kind)
        res = bike_excess(m, kind, a, b)
        r, med, peak, bike = res[:, 0], res[:, 2], res[:, 3], res[:, 4]
        print(f"[{ja}] 当てはめた軌跡 R = {a:+.2f} t + {b:.1f}（径方向 {a:+.2f} m/s）")
        print(f"  {'R [m]':>6s} {'通過 [s]':>8s} {'中央値':>7s} {'通過時':>7s} {'自転車ぶん':>9s} {'隣との差':>8s}")
        # 隣との差 = 自転車ぶん − 両隣 2 ビンずつ（線のビンを除く）の真数平均
        for k in range(len(r)):
            nb = [j for j in range(k - 2, k + 3) if j != k and 0 <= j < len(r) and not is_line(r[j])]
            ref = 10 * np.log10(np.nanmean(10 ** (bike[nb] / 10)))
            mark = " ← 線" if is_line(r[k]) else ""
            print(f"  {r[k]:6.1f} {res[k, 1]:8.1f} {med[k]:7.1f} {peak[k]:7.1f} {bike[k]:9.1f} {bike[k] - ref:+8.1f}{mark}")
        off = np.array([not is_line(v) for v in r])
        d_all = []
        for k in range(len(r)):
            nb = [j for j in range(k - 2, k + 3) if j != k and 0 <= j < len(r) and not is_line(r[j])]
            d_all.append(bike[k] - 10 * np.log10(np.nanmean(10 ** (bike[nb] / 10))))
        d_all = np.array(d_all)
        print(f"  隣との差: 線でないビンの中央値 {np.median(d_all[off]):+.1f} dB（25〜75% 点 "
              f"{np.percentile(d_all[off], 25):+.1f}〜{np.percentile(d_all[off], 75):+.1f}）、"
              f"線のビン {', '.join(f'{v:+.1f}' for v in d_all[~off])} dB\n")

        ax.plot(r, med, color="0.5", lw=1.0, marker=".", label="時間中央値（線 ＋ 床）")
        ax.plot(r, peak, color="C0", lw=1.0, marker=".", label="通過時の最大")
        ax.plot(r, bike, color="C3", lw=1.4, marker="o", ms=3, label="自転車ぶん（真数で中央値を引いた）")
        for l in LINES:
            ax.axvline(l, color="k", lw=0.8, ls=":")
        ax.set_ylabel("[dB]")
        ax.set_title(f"{ja}（点線 = 線の距離 {', '.join(f'{l}' for l in LINES)} m）", fontsize=10)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("距離 R [m]")
    fig.suptitle("10/12 15:09:39 窓 開・自転車 接近: 線の距離で自転車の電力が足されているか", fontsize=11)
    out = FIG / "bike_on_line.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
