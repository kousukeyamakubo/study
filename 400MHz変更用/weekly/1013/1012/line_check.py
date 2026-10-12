# README の L: 10/12 の 4 本で、縦の線の位置と高さを ../1011/wall_check.py と同じ量で測る。
#
# 線の高さ = ドップラー最大・MTI ありの時間中央値 − 距離方向の床（±5 m の中央値）。壁（10/11）・10/1 夜と並べる。
# 10/12 の図では、開と閉で線の並びが変わり、線が MTI なしの静止物の明るい距離と重なって見えた。
# 線が「IF に入った決まった周波数の妨害」なら窓で変わらないはずで、「強い静止物の MTI の引き残し」なら
# 静止物の強さについて動くはずなので、線の高さと静止物の強さ（電力和・MTI なしの時間中央値 − 同じ床）を距離ごとに並べる。
#
# 使い方:
#   python -X utf8 line_check.py > line_check_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1011"))
sys.path.insert(0, str(HERE.parent / "collect"))
from wall_check import C0, SLOPE, line_bin, profiles, setup_font_ja, smooth_floor   # noqa: E402
from check_run_1013 import export                                                   # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
WALL_DIR = HERE.parent / "1011" / "壁に向ける実験"
RUNS = {"145934": "閉・人なし", "150040": "開・人なし", "150757": "閉・自転車", "150939": "開・自転車"}
WALLS = {"172428": "壁・充電器なし", "173256": "壁・充電器あり"}
# 比べる距離: 壁と 10/1 夜の線（../1011/wall_check_log.txt）と、10/12 の図で見えた線
LINES = [17.0, 22.0, 29.0, 34.0, 40.0, 50.0, 67.0]
PEAK_DB = 2.0
R_MIN, R_MAX = 3.0, 100.0


def load(dat):
    return profiles(export(dat, DATA, False), export(dat, DATA, True))


def main():
    FIG.mkdir(exist_ok=True)
    setup_font_ja()
    profs = {name: load(DATA / f"atlas_log_20261012_{s}.dat") for s, name in RUNS.items()}
    # 壁の npz は 1011 の data/ に出す（1011 と同じ場所。書き換えはしない）
    for s, name in WALLS.items():
        profs[name] = load_wall(s)
    rng = profs["開・人なし"]["rng"]
    ex = {n: p["dmax"] - smooth_floor(p["dmax"], rng) for n, p in profs.items()}       # (R,) 線の高さ
    st = {n: p["static"] - smooth_floor(p["static"], rng) for n, p in profs.items()}   # (R,) 静止物の山
    names = list(profs)

    print(f"線の高さ = ドップラー最大・MTI ありの時間中央値 − 距離方向の床（±5 m の中央値）[dB]")
    print("各距離の ±1.5 m で、それぞれの収録の最大の距離と高さ")
    print(f"  {'線':>4s}  " + "".join(f"{n:>17s}" for n in names))
    for r0 in LINES:
        cells = []
        for n in names:
            i = line_bin(rng, ex[n], r0)
            cells.append(f"{rng[i]:5.1f} m {ex[n][i]:+5.1f}")
        print(f"  {r0:4.0f}  " + "".join(f"{c:>17s}" for c in cells))

    for n in names:
        print(f"\n{n}: 線の高さが {PEAK_DB:.0f} dB 以上の距離（R {R_MIN:.0f}〜{R_MAX:.0f} m）と、同じ距離の静止物の山（MTI なし − 床）")
        for i in np.flatnonzero((ex[n] >= PEAK_DB) & (rng >= R_MIN) & (rng <= R_MAX)):
            fb = 2 * SLOPE * rng[i] / C0 / 1e3
            print(f"  R {rng[i]:6.1f} m  ビート {fb:6.1f} kHz  線 {ex[n][i]:+5.1f} dB  静止物 {st[n][i]:+5.1f} dB  "
                  f"静止物の絶対値 {profs[n]['static'][i]:5.1f} dB")

    # 線の高さと静止物の強さの関係（R 13〜55 m、走路の範囲）
    keep = (rng >= 13.0) & (rng <= 55.0)
    print("\n走路の範囲 R 13〜55 m で、線の高さと静止物の山（どちらも床からの高さ）の相関係数")
    for n in names:
        r = np.corrcoef(ex[n][keep], st[n][keep])[0, 1]
        print(f"  {n:14s} {r:+.2f}")
    # 窓の開閉で、線の高さの変化と静止物の変化が同じ向きか（人なしどうし）
    d_ex = ex["開・人なし"] - ex["閉・人なし"]
    d_abs = profs["開・人なし"]["static"] - profs["閉・人なし"]["static"]
    d_line_abs = profs["開・人なし"]["dmax"] - profs["閉・人なし"]["dmax"]
    print("\n開 − 閉（人なし）: 線の距離での、ドップラー最大・MTI ありの変化と、静止物（MTI なし）の変化 [dB]")
    for r0 in LINES[:5]:
        i = line_bin(rng, ex["開・人なし"], r0)
        print(f"  R {rng[i]:5.1f} m  MTI あり {d_line_abs[i]:+5.1f}  静止物 {d_abs[i]:+5.1f}  線の高さ {d_ex[i]:+5.1f}")
    print(f"  R 13〜55 m 全体の相関（MTI ありの変化 vs 静止物の変化）: {np.corrcoef(d_line_abs[keep], d_abs[keep])[0, 1]:+.2f}")

    # 線の山のビンで速度方向に平らか（DC 以外の 15 ビン）
    print("\n線の山のビンでの、ドップラービンごとの時間中央値 − 床 [dB]（DC を除く 15 ビン: 最小 / 中央値 / 最大）")
    for n in ("開・人なし", "閉・人なし"):
        p = profs[n]
        fl = smooth_floor(p["dmax"], rng)
        for r0 in LINES[:5]:
            i = line_bin(rng, ex[n], r0)
            v = np.delete(p["per_d"][:, i], p["dc"]) - fl[i]
            print(f"  {n:10s} R {rng[i]:5.1f} m: {v.min():+5.1f} / {np.median(v):+5.1f} / {v.max():+5.1f}")

    # 図: 上 = 線の高さ、下 = 静止物（MTI なし）の生の値
    colors = {"開・人なし": "C0", "閉・人なし": "C3", "開・自転車": "C0", "閉・自転車": "C3",
              "壁・充電器なし": "0.5", "壁・充電器あり": "0.5"}
    styles = {"開・人なし": "-", "閉・人なし": "-", "開・自転車": ":", "閉・自転車": ":",
              "壁・充電器なし": "-", "壁・充電器あり": "--"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for n in names:
        axes[0].plot(rng, ex[n], color=colors[n], ls=styles[n], lw=1.1, label=n)
        if not n.startswith("壁"):
            axes[1].plot(rng, profs[n]["static"], color=colors[n], ls=styles[n], lw=1.1, label=n)
    for ax in axes:
        for r0 in LINES:
            ax.axvline(r0, color="gray", lw=0.6, ls=":")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("線の高さ [dB]")
    axes[0].set_title("ドップラー最大・MTI ありの時間中央値 − 距離方向の床（±5 m の中央値）", fontsize=10)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_ylabel("受信電力 [dB]")
    axes[1].set_title("電力和・MTI なしの時間中央値（静止物）", fontsize=10)
    axes[1].set_xlabel("距離 R [m]")
    axes[1].set_xlim(0, 70)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(FIG / "line_check.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("\n-> figures/line_check.png")


def load_wall(stem):
    dat = WALL_DIR / f"atlas_log_20261011_{stem}.dat"
    out = HERE.parent / "1011" / "data"
    return profiles(export(dat, out, False), export(dat, out, True))


if __name__ == "__main__":
    main()
