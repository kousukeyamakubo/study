# 10/1 夜の 12 本（../1008/ と同じ）に、フレーム差（1 つ前のフレームを引く）を掛けて、距離−時間図で筋が残るかを見る。
#
# 10/13 ミーティングの提案「検出の比較は全 N 共通の『ドップラーなし』の流れに固定する」の確かめ。
# MTI（チャープ平均を引く）は N=1 で結果がゼロになり使えないので、全 N で使えるフレーム差に置き換えたとき、
# ../1011/ で MTI ありなら 12 本とも見えた筋が、フレーム差でも見えるか。10/12 は自転車 1 本だけで確かめていた（../1012/README.md S）。
#
# 並べるもの（どれもチャープ × 8ch の電力和、ドップラーは使わない。../1011/range_profile.py と同じ量）:
#   MTI あり（N=16）/ フレーム差（N=16）/ フレーム差（N=1、先頭の 1 チャープだけ。時分割 ISAC の「先頭から N 本」と同じ間引き方）
# フレーム差の計算は ../1012/bg_subtract.py と同じ（チャープごとに 1 つ前のフレームの同じチャープを引いてから電力を足す）。
#
# 色の範囲は処理ごとに「目標が来ない帯 R 55〜62 m の中央値（12 本まとめて）」−2 dB から 15 dB 幅にする。
# 処理によって値の大きさが違う（N=1 は足すチャープが 1/16）ので、雑音の床からの見え方をそろえて比べる。
# 幅 15 dB は ../1011/ の電力和・MTI あり（27〜42 dB）と同じ。
#
# 数字は真値を使わないもの（引き残しの床 = 帯ごとの中央値 − 目標が来ない帯の中央値）だけを出す。
# 筋の上の強さは、今の真値が同じデータからの当てはめなので出さない（../1011/README.md「見直す理由」）。
#
# 使い方:
#   python -X utf8 frame_diff_1001.py > frame_diff_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
sys.path.insert(0, str(WEEKLY / "1001"))
sys.path.insert(0, str(HERE.parent / "1011"))
from analyze_night import TAGS, condition, npz_path                         # noqa: E402
from range_profile import RUNS, MODE_JA, DR_JA, chirp_profiles, setup_font_ja  # noqa: E402
from axes_maps import R_HI, R_LO, RANGE_RES, res_bar                       # noqa: E402

FIG = HERE / "figures"
NOISE_BAND = (55.0, 62.0)
SPAN_DB = 15.0
BANDS = {"R 13-20 m": (13.0, 20.0), "R 20-28 m": (20.0, 28.0), "R 30-40 m": (30.0, 40.0), "R 40-49 m": (40.0, 49.0)}
KINDS = ["mti16", "diff16", "diff1"]
KIND_JA = {"mti16": "MTI あり（チャープ平均を引く）N=16",
           "diff16": "フレーム差 N=16",
           "diff1": "フレーム差 N=1"}


def to_db(x):
    return 10 * np.log10(np.sum(np.abs(x) ** 2, axis=(1, 2, 3)) + 1e-30)   # (F, R)


def frame_diff(x):
    """1 つ前のフレームを引く。先頭のフレームは 1 つ前がないので、2 番目の値で埋めて行数をそろえる（../1012/bg_subtract.py と同じ）"""
    d = to_db(x[1:] - x[:-1])                                               # (F-1, R)
    return np.vstack([d[:1], d])                                            # (F, R)


def run_maps(n):
    d = np.load(npz_path(TAGS[n - 1], nomti=True))
    x = chirp_profiles(d["rd"])                                             # (F, TX, RX, C, R) MTI なしのチャープごと
    out = {"mti16": to_db(x - x.mean(axis=3, keepdims=True)),
           "diff16": frame_diff(x),
           "diff1": frame_diff(x[:, :, :, :1, :])}
    out.update(rng=d["range_m"], t=d["t_rel_s"])
    return out


def noise_median(m, kind):
    nb = (m["rng"] >= NOISE_BAND[0]) & (m["rng"] <= NOISE_BAND[1])
    return np.median(m[kind][:, nb])


def draw_rt(ax, m, kind, lim, label):
    keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
    im = ax.pcolormesh(m["rng"][keep], m["t"], m[kind][:, keep], shading="nearest", cmap="viridis",
                       vmin=lim[0], vmax=lim[1])
    res_bar(ax, "range", label=label)
    return im


def compare_figure(maps, lims, out):
    """行 = 処理（3）、列 = 走行（12）"""
    fig, axes = plt.subplots(3, 12, figsize=(2.0 * 12, 4.2 * 3), sharey="row",
                             gridspec_kw=dict(hspace=0.35, wspace=0.08))
    for i, kind in enumerate(KINDS):
        for j, n in enumerate(RUNS):
            ax = axes[i, j]
            im = draw_rt(ax, maps[n], kind, lims[kind], label=(i == 0 and j == 0))
            if i == 0:
                mode, lane, dr = condition(n)
                ax.set_title(f"No.{n}\n{MODE_JA[mode]} L{lane} {DR_JA[dr]}", fontsize=9)
            if i == 2:
                ax.set_xlabel("距離 R [m]", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[i, 0].set_ylabel(f"{KIND_JA[kind]}\n時刻 [s]", fontsize=9)
        fig.colorbar(im, ax=axes[i, :], label="床からの高さ [dB]", shrink=0.9, pad=0.01,
                     format=matplotlib.ticker.FuncFormatter(lambda v, _, b=lims[kind][0] + 2: f"{v - b:+.0f}"))
    fig.suptitle("10/1 夜の 12 本：MTI とフレーム差（電力和・ドップラーなし）\n"
                 f"色は目標が来ない帯 R 55〜62 m の中央値を 0 として −2〜+13 dB。上端の白い線 = 距離分解能 {RANGE_RES:.2f} m",
                 fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    FIG.mkdir(exist_ok=True)
    setup_font_ja()
    maps = {n: run_maps(n) for n in RUNS}

    lims = {}
    print("目標が来ない帯 R 55〜62 m の中央値（12 本まとめた値は走行ごとの中央値の中央値）[dB]")
    for kind in KINDS:
        nm = np.median([noise_median(maps[n], kind) for n in RUNS])
        lims[kind] = (nm - 2.0, nm - 2.0 + SPAN_DB)
        print(f"  {KIND_JA[kind]:28s} {nm:6.1f}  → 色の範囲 {lims[kind][0]:.1f}〜{lims[kind][1]:.1f}")

    print("\n引き残しの床（帯ごとの中央値 − 目標が来ない帯の中央値、全フレーム、走行ごとの値の中央値）[dB]")
    print("  " + " " * 30 + "".join(f"{b:>12s}" for b in BANDS))
    for kind in KINDS:
        row = []
        for lo, hi in BANDS.values():
            v = []
            for n in RUNS:
                m = maps[n]
                k = (m["rng"] >= lo) & (m["rng"] < hi)
                v.append(np.median(m[kind][:, k]) - noise_median(m, kind))
            row.append(np.median(v))
        print(f"  {KIND_JA[kind]:28s}" + "".join(f"{x:+12.1f}" for x in row))

    compare_figure(maps, lims, FIG / "frame_diff_overview.png")


if __name__ == "__main__":
    main()
