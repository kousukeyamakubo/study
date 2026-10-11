# README の A: 10/1 夜の 12 本（../1008/ と同じ）で、距離−時間図を 3 通りに作り、どの段階から目標が見えるかを比べる。
#
#   1. 電力和・MTI なし  : チャープ × 8ch の電力を足す（ドップラーを使わない）。何も引かない
#   2. 電力和・MTI あり  : 同じ。チャープ平均を引いたもの
#   3. ドップラー最大・MTI あり : ../1005/axes_maps_1005.py の距離−時間と同じ計算（8ch の振幅和を dB にし、
#                                 DC 以外のドップラーで最大）。ドップラーで電力を集めた利得が入る。../1008/ の T1 とほぼ同じ量
#
# 描き方（R 範囲・色・分解能の白線・日本語ラベル）は ../1005/axes_maps_1005.py の *_ja.png に合わせる。
# 色の範囲は計算の種類ごとに決める（値の大きさが違うので、同じ範囲にすると片方が飽和する）。
#
# 使い方:
#   python -X utf8 range_profile.py > run_log.txt

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
sys.path.insert(0, str(WEEKLY / "1013" / "1001"))
from atlas_export import hanning_matlab                                    # noqa: E402
from analyze_night import TAGS, condition, npz_path                         # noqa: E402
from axes_maps import PW_LIM, R_HI, R_LO, RANGE_RES, res_bar               # noqa: E402

FIG = HERE / "figures"
RUNS = [1, 3, 5, 9, 19, 20, 21, 22, 23, 24, 27, 28]    # ../1008/run_log.txt「評価に使う走行 12 本」
KINDS = ["sum_nomti", "sum_mti", "dmax_mti"]
KIND_JA = {
    "sum_nomti": "電力和（ドップラーなし）・MTI なし",
    "sum_mti": "電力和（ドップラーなし）・MTI あり",
    "dmax_mti": "ドップラー最大・MTI あり",
}
# 色の範囲 [dB]。ドップラー最大は 1005 と同じ。電力和は図を見て決めた（run_log.txt の分位点を参照）
LIM = {"sum_nomti": (35.0, 70.0), "sum_mti": (27.0, 42.0), "dmax_mti": PW_LIM}
MODE_JA = {"walk": "徒歩", "bike": "自転車"}
DR_JA = {"app": "接近", "dep": "離反"}


def chirp_profiles(rd):
    """RD (F, TX, RX, D, R) からチャープごとの距離プロファイル (F, TX, RX, C, R) に戻す。

    atlas_export.rd_tensor はチャープ方向に Hann 窓を掛けてから FFT している。RD の電力をドップラー方向に
    足すと（パーセバル）窓で重みづけしたチャープの電力和になり、端のチャープがほとんど効かない。
    README の「チャープごとの電力を足す」に合わせるため、逆 FFT して窓を割り戻す（MATLAB の hanning は端も 0 でない）
    """
    n = rd.shape[-2]
    win = hanning_matlab(n)
    x = np.fft.ifft(np.fft.ifftshift(rd, axes=-2), axis=-2) * win.sum()       # 窓つきのチャープ
    return x / win[:, None]


def run_maps(n):
    out = {}
    for nomti in (True, False):
        d = np.load(npz_path(TAGS[n - 1], nomti=nomti))
        rd, rng, vel, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]   # rd: (F, TX, RX, D, R)
        x = chirp_profiles(rd)                                                # (F, TX, RX, C, R)
        p = np.sum(np.abs(x) ** 2, axis=(1, 2, 3))                            # (F, R) チャープ × 8ch の電力和
        out["sum_nomti" if nomti else "sum_mti"] = 10 * np.log10(p + 1e-30)
        if not nomti:
            # ../1005/axes_maps_1005.py run_maps の "mti" と同じ
            dc = int(np.abs(vel).argmin())
            db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)          # (F, D, R)
            out["dmax_mti"] = np.delete(db, dc, axis=1).max(axis=1)           # (F, R)
    out.update(rng=rng, t=t)
    return out


def run_label(n):
    mode, lane, dr = condition(n)
    return f"No.{n} {MODE_JA[mode]} レーン{lane} {DR_JA[dr]}"


def draw_rt(ax, m, kind, label=True):
    """距離−時間を 1 枚描く（../1005/axes_maps_1005.py draw_rt_ja と同じ描き方）"""
    keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
    im = ax.pcolormesh(m["rng"][keep], m["t"], m[kind][:, keep], shading="nearest", cmap="viridis",
                       vmin=LIM[kind][0], vmax=LIM[kind][1])
    res_bar(ax, "range", label=label)
    ax.set_xlabel("距離 R [m]")
    return im


def setup_font_ja():
    # ../1005/axes_maps_1005.py と同じ。Meiryo の「−」は字形が崩れるので ASCII にする
    plt.rcParams["font.family"] = ["Meiryo"]
    plt.rcParams["axes.unicode_minus"] = False


def run_figure(m, n, out):
    """1 走行: 3 枚（電力和・MTI なし / 電力和・MTI あり / ドップラー最大・MTI あり）を横に並べる"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        im = draw_rt(ax, m, kind)
        ax.set_title(KIND_JA[kind], fontsize=10)
        fig.colorbar(im, ax=ax, label="受信電力 [dB]", shrink=0.8)
    axes[0].set_ylabel("時刻 [s]")
    fig.suptitle(f"10/1 夜 {run_label(n)}（距離−時間、上端の白い線の長さ = 距離分解能 {RANGE_RES:.2f} m）",
                 fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def overview_figure(maps, kind, out):
    """12 本を 2 行 × 6 列に並べる（1 つの計算の種類）"""
    fig, axes = plt.subplots(2, 6, figsize=(3.2 * 6, 10), sharey=True, gridspec_kw=dict(hspace=0.35))
    for j, (ax, n) in enumerate(zip(axes.flat, RUNS)):
        # 細い格子では数値が重なるので、左上だけに書く（1005 と同じ）
        im = draw_rt(ax, maps[n], kind, label=(j == 0))
        mode, lane, dr = condition(n)
        ax.set_title(f"No.{n}\n{MODE_JA[mode]} レーン{lane} {DR_JA[dr]}", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("時刻 [s]")
    fig.suptitle(f"10/1 夜の 12 本（{KIND_JA[kind]}）\n上端の白い線の長さ = 距離分解能 {RANGE_RES:.2f} m",
                 fontsize=11)
    fig.colorbar(im, ax=axes, label="受信電力 [dB]", shrink=0.6)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    FIG.mkdir(exist_ok=True)
    setup_font_ja()
    maps = {n: run_maps(n) for n in RUNS}

    # 色の範囲を決めるための分位点（R_LO〜R_HI、12 本まとめて）
    print("色の範囲を決めるための分位点 [dB]（R 15〜55 m、12 本まとめて）")
    for kind in KINDS:
        v = np.concatenate([m[kind][:, (m["rng"] >= R_LO) & (m["rng"] <= R_HI)].ravel() for m in maps.values()])
        q = np.percentile(v, [1, 50, 99, 99.9])
        print(f"  {KIND_JA[kind]:24s} 1%: {q[0]:5.1f}  50%: {q[1]:5.1f}  99%: {q[2]:5.1f}  99.9%: {q[3]:5.1f}"
              f"  （色の範囲 {LIM[kind][0]:.0f}〜{LIM[kind][1]:.0f}）")

    for n in RUNS:
        mode, lane, dr = condition(n)
        run_figure(maps[n], n, FIG / f"No{n:02d}_{mode}_lane{lane}_{dr}.png")
    for kind in KINDS:
        overview_figure(maps, kind, FIG / f"overview_{kind}.png")


if __name__ == "__main__":
    main()
