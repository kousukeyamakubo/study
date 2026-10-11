# 縦の線が回路・電源側の妨害か、外の物かを分ける（屋内で壁に密着させた 10 s の収録、10/11）。
# レーダーの電源は PC の USB。17:24 は PC の充電器を抜いた状態、17:32 はつないだ状態（構成はそれ以外同じ）。
#
# 考え方: 壁に向ければ外の物は視野に入らない。それでも 10/1 夜と同じ距離（17・34・50・67 m 付近、
# ビート周波数で約 78 kHz の整数倍）に線が出れば、反射ではなく IF に入った妨害と言える。消えれば外の物（または外との多重反射）。
#
# 比べる量は range_profile.py の「ドップラー最大・MTI あり」（../1005/ と同じ計算）の、距離ごとの時間中央値。
# 線の高さは「その距離の値 − 距離方向の滑らかな床（±5 m の中央値）」で見る。線の幅は 1〜3 ビンなので、±5 m の中央値には乗らない。
# 速度方向に平らかは、線の距離で DC 以外の 15 ビンの時間中央値を並べて見る。
#
# 使い方:
#   python -X utf8 wall_check.py > wall_check_log.txt

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
sys.path.insert(0, str(WEEKLY / "0727"))
sys.path.insert(0, str(WEEKLY / "1001"))
from analyze_night import TAGS, npz_path                                   # noqa: E402

EXPORT = WEEKLY / "0727" / "atlas_export.py"
WALL_DIR = HERE / "壁に向ける実験"
WALLS = {"172428": "壁・充電器なし", "173256": "壁・充電器あり"}
DATA = HERE / "data"            # 中間生成物（git 管理外）
FIG = HERE / "figures"
RUNS = [1, 3, 5, 9, 19, 20, 21, 22, 23, 24, 27, 28]    # range_profile.py と同じ 12 本
LINES_1001 = [17.0, 34.0, 50.0, 67.0]                   # 10/1 で見えている線（考察メモの観測 7）
FLOOR_HALF_M = 5.0              # 床を取る距離方向の窓の片側 [m]
SLOPE = 0.692e12                # チャープの傾き [Hz/s]（考察メモの前提）
C0 = 299_792_458.0


def export(dat, nomti):
    sub = DATA / ("export_nomti" if nomti else "export") / dat.stem
    npz = sub / f"{dat.stem}_rd.npz"
    if not npz.exists():
        cmd = [sys.executable, str(EXPORT), str(dat), "--outdir", str(sub)] + (["--no-mti"] if nomti else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return npz


def profiles(npz_mti, npz_nomti):
    """距離ごとの時間中央値: ドップラー最大・MTI あり (R,)、ドップラー別・MTI あり (D, R)、MTI なしの電力和 (R,)"""
    d = np.load(npz_mti)
    rd, rng, vel = d["rd"], d["range_m"], d["vel_ms"]                     # rd: (F, TX, RX, D, R)
    dc = int(np.abs(vel).argmin())
    db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)              # (F, D, R)
    dmax = np.median(np.delete(db, dc, axis=1).max(axis=1), axis=0)       # (R,)
    per_d = np.median(db, axis=0)                                         # (D, R)
    rd0 = np.load(npz_nomti)["rd"]                                        # (F, TX, RX, D, R)
    static = np.median(10 * np.log10(np.sum(np.abs(rd0) ** 2, axis=(1, 2, 3)) + 1e-30), axis=0)   # (R,)
    return dict(rng=rng, vel=vel, dc=dc, dmax=dmax, per_d=per_d, static=static, n_frame=rd.shape[0])


def smooth_floor(x, rng):
    """距離方向の滑らかな床（±FLOOR_HALF_M の中央値）"""
    step = rng[1] - rng[0]
    h = int(round(FLOOR_HALF_M / step))
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def setup_font_ja():
    plt.rcParams["font.family"] = ["Meiryo"]
    plt.rcParams["axes.unicode_minus"] = False


def line_bin(rng, ex_ref, r0):
    """r0 の ±1.5 m（線の幅 1〜3 ビン）で、ex_ref が最大のビン"""
    k = np.flatnonzero(np.abs(rng - r0) <= 1.5)
    return k[np.argmax(ex_ref[k])]


def main():
    FIG.mkdir(exist_ok=True)
    setup_font_ja()
    walls = {}
    for tag, name in WALLS.items():
        dat = WALL_DIR / f"atlas_log_20261011_{tag}.dat"
        walls[name] = profiles(export(dat, nomti=False), export(dat, nomti=True))
    nights = [profiles(npz_path(TAGS[n - 1]), npz_path(TAGS[n - 1], nomti=True)) for n in RUNS]
    rng = nights[0]["rng"]
    night_dmax = np.median([p["dmax"] for p in nights], axis=0)          # (R,) 12 本の中央値
    night_static = np.median([p["static"] for p in nights], axis=0)      # (R,)

    ex = {name: p["dmax"] - smooth_floor(p["dmax"], rng) for name, p in walls.items()}
    ex["10/1 夜"] = night_dmax - smooth_floor(night_dmax, rng)
    names = list(ex)

    for name, p in walls.items():
        print(f"{name}: フレーム {p['n_frame']}")
    print(f"10/1 夜: {len(RUNS)} 本（各走行の時間中央値の、12 本の中央値）")
    print(f"\n線の高さ = ドップラー最大・MTI ありの時間中央値 − 距離方向の床（±{FLOOR_HALF_M:.0f} m の中央値）[dB]")
    print("各線の近く（±1.5 m）で、それぞれの収録の最大の距離と高さ")
    print(f"  {'線':>5s}  " + "  ".join(f"{n:>18s}" for n in names))
    for r0 in LINES_1001:
        cells = []
        for n in names:
            i = line_bin(rng, ex[n], r0)
            cells.append(f"{rng[i]:5.1f} m {ex[n][i]:+5.1f} dB")
        print(f"  {r0:5.0f}  " + "  ".join(f"{c:>18s}" for c in cells))

    for name in WALLS.values():
        print(f"\n{name}: 床から 2 dB 以上高い距離（R 3〜100 m）")
        for i in np.flatnonzero((ex[name] >= 2.0) & (rng >= 3.0) & (rng <= 100.0)):
            fb = 2 * SLOPE * rng[i] / C0 / 1e3
            print(f"  R {rng[i]:6.1f} m  ビート {fb:6.1f} kHz  {ex[name][i]:+.1f} dB")

    # 線の山のビンで、速度方向に平らか（DC 以外の 15 ビン）
    print("\n線の山のビンでの、ドップラービンごとの時間中央値 − 床 [dB]（DC を除く 15 ビン: 最小 / 中央値 / 最大）")
    profs = dict(walls)
    profs["10/1 夜 No.19"] = nights[RUNS.index(19)]
    for name, p in profs.items():
        e = p["dmax"] - smooth_floor(p["dmax"], rng)
        fl = smooth_floor(p["dmax"], rng)
        for r0 in LINES_1001:
            i = line_bin(rng, e, r0)
            v = np.delete(p["per_d"][:, i], p["dc"]) - fl[i]
            print(f"  {name:14s} R {rng[i]:5.1f} m: {v.min():+5.1f} / {np.median(v):+5.1f} / {v.max():+5.1f}")

    # 図: 距離ごとの線の高さ（上）と、生の値（下）
    colors = {"10/1 夜": "C0", "壁・充電器なし": "C1", "壁・充電器あり": "C3"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    for n in names:
        ax.plot(rng, ex[n], color=colors[n], lw=1.2, label=n + ("（12 本の中央値）" if n == "10/1 夜" else ""))
    for r0 in LINES_1001:
        ax.axvline(r0, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("床からの高さ [dB]")
    ax.set_title("ドップラー最大・MTI あり の時間中央値 − 距離方向の床（±5 m の中央値）。点線は 10/1 で見えている線（17・34・50・67 m）",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(rng, night_dmax, color=colors["10/1 夜"], lw=1.2, label="10/1 夜: ドップラー最大・MTI あり")
    ax.plot(rng, night_static - 30, color=colors["10/1 夜"], lw=0.8, ls="--", label="10/1 夜: 電力和・MTI なし − 30 dB")
    for name, p in walls.items():
        ax.plot(rng, p["dmax"], color=colors[name], lw=1.2, label=f"{name}: ドップラー最大・MTI あり")
        ax.plot(rng, p["static"] - 30, color=colors[name], lw=0.8, ls="--", label=f"{name}: 電力和・MTI なし − 30 dB")
    for r0 in LINES_1001:
        ax.axvline(r0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("距離 R [m]")
    ax.set_ylabel("受信電力 [dB]")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "wall_check.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("\n-> figures/wall_check.png")


if __name__ == "__main__":
    main()
