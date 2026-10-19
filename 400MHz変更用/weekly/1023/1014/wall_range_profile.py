# 壁に向けた収録（10/11）の Range Profile を、10/13 の発表と同じ量（電力和、MTI あり/なし）で作る。
#
# 10/13 の発表では「縦の線は壁に密着させても出る → 外の物の反射ではない」と言ったが、根拠の ../../1013/1011/wall_check.py は
# ドップラー最大の量しか見ていなかった。発表の図（../../1013/1011/figures/overview_sum_mti.png）は電力和なので、同じ量で確かめ直す。
#
# 見ること:
#   1. 壁でも、電力和・MTI ありの距離−時間図に縦の線が出るか（図 wall_rt.png）
#   2. 線の高さ（時間中央値 − 距離方向の床）を、壁・10/1 夜・10/12 の無人背景で並べる（図 wall_profile.png）
#   3. 線が MTI で減らない（チャープごとに位相がそろっていない）か: 線の距離で「MTI なし − MTI あり」を見る。
#      反射なら同じ距離の静止物はチャープ間で位相がそろい MTI で大きく減る。IF に入る妨害がチャープと同期していなければ減らない
#   4. 線が 1 つの収録の中で時間的に安定か: 線の距離でのフレームごとの揺れ（標準偏差）を、床の距離と比べる
#
# 使い方:
#   python -X utf8 wall_range_profile.py > wall_range_profile_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
W1013 = WEEKLY / "1013"
sys.path.insert(0, str(WEEKLY / "0727"))
sys.path.insert(0, str(WEEKLY / "1001"))
sys.path.insert(0, str(W1013 / "1001"))
sys.path.insert(0, str(W1013 / "1011"))
sys.path.insert(0, str(W1013 / "collect"))
from analyze_night import TAGS, npz_path                                    # noqa: E402
from range_profile import chirp_profiles                                    # noqa: E402
from wall_check import C0, SLOPE, setup_font_ja, smooth_floor               # noqa: E402
from check_run_1013 import export                                           # noqa: E402

FIG = HERE / "figures"
WALL_DIR = W1013 / "1011" / "壁に向ける実験"
WALL_DATA = W1013 / "1011" / "data"         # 1011 で作った npz をそのまま使う（書き換えない）
WALLS = {"172428": "壁・充電器なし", "173256": "壁・充電器あり"}
D1012 = W1013 / "1012" / "data"
BG1012 = {"150040": "10/12 開・人なし", "145934": "10/12 閉・人なし"}
RUNS_1001 = [1, 3, 5, 9, 19, 20, 21, 22, 23, 24, 27, 28]   # ../../1013/1011/range_profile.py と同じ 12 本
# 比べる距離: 壁・10/1 夜の線（17・34・50・67 m）と、10/12 だけに出た線（22.9・28.8・39.8 m）
LINES = [17.0, 22.9, 28.8, 34.0, 39.8, 50.0, 67.0]
R_MIN, R_MAX = 3.0, 70.0
LIM = {"sum_nomti": (35.0, 70.0), "sum_mti": (27.0, 42.0)}    # range_profile.py と同じ色の範囲（発表の図と比べるため）
KIND_JA = {"sum_nomti": "電力和・MTI なし", "sum_mti": "電力和・MTI あり"}


def sum_maps(npz_mti, npz_nomti):
    """電力和の距離−時間 (F, R) を MTI あり/なしで返す（range_profile.run_maps と同じ計算）"""
    out = {}
    for key, npz in (("sum_mti", npz_mti), ("sum_nomti", npz_nomti)):
        d = np.load(npz)
        x = chirp_profiles(d["rd"])                                  # (F, TX, RX, C, R)
        p = np.sum(np.abs(x) ** 2, axis=(1, 2, 3))                   # (F, R) チャープ × 8ch の電力和
        out[key] = 10 * np.log10(p + 1e-30)
        out["rng"], out["t"] = d["range_m"], d["t_rel_s"]
    return out


def load_all():
    srcs = {}
    for s, name in WALLS.items():
        dat = WALL_DIR / f"atlas_log_20261011_{s}.dat"
        srcs[name] = sum_maps(export(dat, WALL_DATA, False), export(dat, WALL_DATA, True))
    for s, name in BG1012.items():
        dat = D1012 / f"atlas_log_20261012_{s}.dat"
        srcs[name] = sum_maps(export(dat, D1012, False), export(dat, D1012, True))
    # 10/1 夜は走行中のデータなので、12 本それぞれの時間中央値をとってから 12 本の中央値にする（wall_check と同じ）
    night = [sum_maps(npz_path(TAGS[n - 1], nomti=False), npz_path(TAGS[n - 1], nomti=True)) for n in RUNS_1001]
    srcs["10/1 夜（12 本の中央値）"] = dict(
        rng=night[0]["rng"],
        med={k: np.median([np.median(m[k], axis=0) for m in night], axis=0) for k in LIM},
        std_mti=np.median([np.std(m["sum_mti"], axis=0) for m in night], axis=0),
    )
    for name, m in srcs.items():
        if "med" not in m:
            m["med"] = {k: np.median(m[k], axis=0) for k in LIM}     # (R,) 時間中央値
            m["std_mti"] = np.std(m["sum_mti"], axis=0)              # (R,) フレームごとの揺れ [dB]
    return srcs


def peak_near(rng, ex, r0, half=1.5):
    """r0 の ±half m で ex が最大のビン"""
    idx = np.where(np.abs(rng - r0) <= half)[0]
    return idx[np.argmax(ex[idx])]


def rt_figure(srcs, out):
    """壁の 2 本: 電力和 MTI なし / あり の距離−時間図"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, name in zip(axes, WALLS.values()):
        m = srcs[name]
        keep = (m["rng"] >= R_MIN) & (m["rng"] <= R_MAX)
        for ax, kind in zip(row, ("sum_nomti", "sum_mti")):
            im = ax.pcolormesh(m["rng"][keep], m["t"], m[kind][:, keep], shading="nearest", cmap="viridis",
                               vmin=LIM[kind][0], vmax=LIM[kind][1])
            ax.set_title(f"{name}（{KIND_JA[kind]}）", fontsize=10)
            ax.set_ylabel("時刻 [s]")
            fig.colorbar(im, ax=ax, label="受信電力 [dB]", shrink=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("距離 R [m]")
    fig.suptitle("壁に密着させた収録（10/11、各 10 s）の距離−時間図。色の範囲は 10/1 夜の図（発表）と同じ", fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def profile_figure(srcs, out):
    """線の高さ（電力和・MTI ありの時間中央値 − 距離方向の床）を、収録ごとに縦にずらして並べる"""
    names = list(srcs)
    fig, ax = plt.subplots(figsize=(12, 7))
    for k, name in enumerate(names):
        m = srcs[name]
        keep = (m["rng"] >= R_MIN) & (m["rng"] <= R_MAX)
        ex = m["med"]["sum_mti"] - smooth_floor(m["med"]["sum_mti"], m["rng"])
        off = -10.0 * k                     # 重ならないように 10 dB ずつ下げる
        ax.plot(m["rng"][keep], ex[keep] + off, lw=1.2)
        ax.axhline(off, color="0.7", lw=0.6)
        ax.text(R_MAX + 0.5, off, name, va="center", fontsize=9)
    for r0 in LINES:
        ax.axvline(r0, color="r", ls=":", lw=0.8)
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_xlabel("距離 R [m]（赤の点線：比べる線の距離 17・22.9・28.8・34・39.8・50・67 m）")
    ax.set_ylabel("線の高さ [dB]（収録ごとに 10 dB ずつずらした。灰色の線が 0 dB）")
    ax.set_title("電力和・MTI ありの時間中央値 − 距離方向の床（±5 m の中央値）", fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    FIG.mkdir(exist_ok=True)
    setup_font_ja()
    srcs = load_all()
    names = list(srcs)
    rng = srcs[names[0]]["rng"]

    print("線の高さ = 電力和・MTI ありの時間中央値 − 距離方向の床（±5 m の中央値）[dB]")
    print("各距離の ±1.5 m で、それぞれの収録の最大の距離と高さ。括弧はビート周波数")
    print(f"  {'線':>5s}  " + "".join(f"{n:>22s}" for n in names))
    ex = {n: srcs[n]["med"]["sum_mti"] - smooth_floor(srcs[n]["med"]["sum_mti"], rng) for n in names}
    for r0 in LINES:
        cells = []
        for n in names:
            i = peak_near(rng, ex[n], r0)
            cells.append(f"{rng[i]:5.1f} m {ex[n][i]:+5.1f}")
        fb = 2 * SLOPE * r0 / C0 / 1e3
        print(f"  {r0:5.1f}  " + "".join(f"{c:>22s}" for c in cells) + f"   ({fb:.0f} kHz)")

    print()
    print("線の距離での「MTI なし − MTI あり」[dB]（電力和の時間中央値）。小さいほど MTI で減らない＝チャープ間で位相がそろっていない")
    print("比較: 床（線から 3 m 以上離れた R 10〜60 m の中央値）と、MTI なしで最も強い距離（静止物）")
    for n in names:
        d = srcs[n]["med"]["sum_nomti"] - srcs[n]["med"]["sum_mti"]           # (R,)
        far = np.ones_like(rng, bool)
        for r0 in LINES:
            far &= np.abs(rng - r0) > 3.0
        band = (rng >= 10) & (rng <= 60)
        i_static = np.where(band)[0][np.argmax(srcs[n]["med"]["sum_nomti"][band])]
        cells = []
        for r0 in LINES:
            i = peak_near(rng, ex[n], r0)
            cells.append(f"{r0:.0f}m {d[i]:+5.1f}" if ex[n][i] >= 2.0 else f"{r0:.0f}m   ―  ")
        print(f"  {n:22s} 床 {np.median(d[far & band]):+5.1f}  静止物 R {rng[i_static]:4.1f} m {d[i_static]:+5.1f}   線: "
              + "  ".join(cells))
    print("  （線: 高さ 2 dB 未満の距離は ― にした）")

    print()
    print("線の距離での、フレームごとの揺れ（電力和・MTI ありの標準偏差）[dB]。床は R 10〜60 m で線から 3 m 以上離れた距離の中央値")
    for n in names:
        s = srcs[n]["std_mti"]
        far = np.ones_like(rng, bool)
        for r0 in LINES:
            far &= np.abs(rng - r0) > 3.0
        band = (rng >= 10) & (rng <= 60)
        cells = []
        for r0 in LINES:
            i = peak_near(rng, ex[n], r0)
            cells.append(f"{r0:.0f}m {s[i]:4.1f}" if ex[n][i] >= 2.0 else f"{r0:.0f}m  ― ")
        print(f"  {n:22s} 床 {np.median(s[far & band]):4.1f}   線: " + "  ".join(cells))
    print("  （10/1 夜は走行中なので、目標が通る距離では揺れが大きくなる）")

    rt_figure(srcs, FIG / "wall_rt.png")
    profile_figure(srcs, FIG / "wall_profile.png")


if __name__ == "__main__":
    main()
