# README の B: A の距離−時間（range_profile.py の 3 通り）に、距離方向だけの CA-CFAR をラフに 1 回掛ける。
#
# 窓: 片側 訓練 5・ガード 2（../1008/ の距離方向と同じ）。端では取れる側のセルだけで平均する。
# しきい値: 公称（Pfa から決める係数）は「セルの雑音が指数分布」の仮定で、電力和（16 チャープ × 8ch = 128 個の和）では
#   ばらつきが小さく成り立たない。そこで、目標が来ない帯 R 55〜62 m（走路の奥の端は R ≈ 49 m）で
#   「セルの比 > しきい値」の割合が 1e-3 になる値を、図の種類ごとにデータから決める。真値は使わない。
#   窓・しきい値は振らない（真値がないので、どれがよいかを決められないため）。
# 出すもの:
#   - 距離−時間図に検出点を重ねたもの（走行ごと・overview）
#   - 距離ビンごとの検出の割合（12 本・全フレーム）。目標の筋は距離に広がり、縦の線（妨害）は決まった距離に集まるので、
#     真値なしでも線による誤検出が山として見える
#   - しきい値を振ったときの検出点の数（README B）。決めたしきい値の前後の変わり方を見るだけ
#
# 使い方:
#   python -X utf8 cfar_range.py > cfar_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from range_profile import (FIG, KIND_JA, KINDS, MODE_JA, DR_JA, R_HI, R_LO, RANGE_RES, RUNS,   # noqa: E402
                           condition, draw_rt, run_label, run_maps, setup_font_ja)

N_TRAIN, N_GUARD = 5, 2         # 片側のセル数（距離方向）
NOISE_R = (55.0, 62.0)          # 目標が来ない帯（../../1001/analyze_night.py の NOISE_R と同じ）
FA_NOISE = 1e-3                 # 目標が来ない帯での、セルの誤検出の割合
ALPHA_SWEEP = np.arange(0.0, 20.01, 0.5)   # しきい値を振る範囲 [dB]
# 縦の線（妨害）の距離。README「縦の線の切り分け」の 10/1 夜の値
LINES = {"17 m": (15.5, 19.5), "34 m": (32.0, 37.5), "50 m": (49.5, 52.0), "67 m": (65.5, 69.0)}


def cfar_ratio_db(db):
    """距離方向の CA-CFAR の「セルの電力 / 訓練セル平均」[dB]。(F, R) → (F, R)"""
    p = 10 ** (db / 10)                                                   # (F, R) 電力（ドップラー最大は 20 log 振幅なので同じ式）
    n_r = p.shape[1]
    out = np.empty_like(p)
    for i in range(n_r):
        idx = [j for j in range(i - N_GUARD - N_TRAIN, i - N_GUARD)] + \
              [j for j in range(i + N_GUARD + 1, i + N_GUARD + N_TRAIN + 1)]
        idx = [j for j in idx if 0 <= j < n_r]
        out[:, i] = 10 * np.log10(p[:, i] / p[:, idx].mean(axis=1))
    return out


def main():
    setup_font_ja()
    maps = {n: run_maps(n) for n in RUNS}
    rng = maps[RUNS[0]]["rng"]
    noise = (rng >= NOISE_R[0]) & (rng <= NOISE_R[1])
    road = (rng >= R_LO) & (rng <= R_HI)
    ratio = {n: {k: cfar_ratio_db(m[k]) for k in KINDS} for n, m in maps.items()}

    # 目標が来ない帯に縦の線が掛かっていないかの確認: 帯の中の時間中央値の、帯の中央値からの外れ
    print(f"目標が来ない帯 R {NOISE_R[0]:.0f}〜{NOISE_R[1]:.0f} m（{noise.sum()} ビン）の確認")
    for k in KINDS:
        med = np.median([np.median(maps[n][k][:, noise], axis=0) for n in RUNS], axis=0)   # (ビン,)
        print(f"  {KIND_JA[k]:24s} 時間中央値の帯の中の幅 {med.max() - med.min():.1f} dB"
              f"（最大のビン R {rng[noise][np.argmax(med)]:.1f} m）")

    # しきい値: 目標が来ない帯でセルの誤検出が FA_NOISE になる値
    alpha = {}
    print(f"\nしきい値（目標が来ない帯でセルの誤検出 {FA_NOISE:g}）")
    for k in KINDS:
        v = np.concatenate([ratio[n][k][:, noise].ravel() for n in RUNS])
        alpha[k] = float(np.quantile(v, 1 - FA_NOISE))
        print(f"  {KIND_JA[k]:24s} {alpha[k]:5.2f} dB（セル {v.size}）")

    # 検出の数
    print(f"\n検出（上のしきい値）: 走路の帯 R {R_LO:.0f}〜{R_HI:.0f} m の、1 フレームあたりの検出セル数（12 本の平均）")
    det = {n: {k: ratio[n][k] > alpha[k] for k in KINDS} for n in RUNS}   # (F, R)
    for k in KINDS:
        per = [det[n][k][:, road].sum(axis=1).mean() for n in RUNS]
        print(f"  {KIND_JA[k]:24s} {np.mean(per):5.2f}（走行ごと {min(per):.2f}〜{max(per):.2f}）")

    # 縦の線の距離に落ちた検出
    print("\n縦の線の距離での、検出の割合（その距離帯のセルのうち検出されたもの、12 本・全フレーム）")
    print(f"  {'':24s} " + "  ".join(f"{name:>6s}" for name in LINES) + "   線の外（走路の帯）")
    in_line = np.zeros(len(rng), bool)
    for lo, hi in LINES.values():
        in_line |= (rng >= lo) & (rng <= hi)
    for k in KINDS:
        cells = []
        for lo, hi in LINES.values():
            sel = (rng >= lo) & (rng <= hi)
            cells.append(np.mean([det[n][k][:, sel].mean() for n in RUNS]))
        out = np.mean([det[n][k][:, road & ~in_line].mean() for n in RUNS])
        print(f"  {KIND_JA[k]:24s} " + "  ".join(f"{c:6.3f}" for c in cells) + f"   {out:6.3f}")

    # 図 1: 距離ビンごとの検出の割合
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, c in zip(KINDS, ("C7", "C1", "C0")):
        rate = np.mean([det[n][k].mean(axis=0) for n in RUNS], axis=0)   # (R,)
        ax.plot(rng, rate, color=c, lw=1.2, label=f"{KIND_JA[k]}（しきい値 {alpha[k]:.1f} dB）")
    for lo, hi in LINES.values():
        ax.axvspan(lo, hi, color="gray", alpha=0.15, lw=0)
    ax.axvspan(*NOISE_R, color="C2", alpha=0.10, lw=0)
    ax.set_xlim(0, 80)
    ax.set_xlabel("距離 R [m]")
    ax.set_ylabel("検出の割合（12 本・全フレーム）")
    ax.set_title(f"距離方向 CA-CFAR（訓練 {N_TRAIN}・ガード {N_GUARD}）の検出の割合。灰色 = 縦の線の距離、"
                 f"緑 = しきい値を決めた帯（誤検出 {FA_NOISE:g}）", fontsize=10)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "cfar_rate_by_range.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("\n-> figures/cfar_rate_by_range.png")

    # 図 2: しきい値を振ったときの検出の数（走路の帯・目標が来ない帯）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for k, c in zip(KINDS, ("C7", "C1", "C0")):
        for ax, sel in zip(axes, (road, noise)):
            cnt = [np.mean([(ratio[n][k][:, sel] > a).sum(axis=1).mean() for n in RUNS]) for a in ALPHA_SWEEP]
            ax.plot(ALPHA_SWEEP, cnt, color=c, lw=1.2, label=KIND_JA[k])
            ax.axvline(alpha[k], color=c, lw=0.8, ls=":")
    axes[0].set_title(f"走路の帯 R {R_LO:.0f}〜{R_HI:.0f} m", fontsize=10)
    axes[1].set_title(f"目標が来ない帯 R {NOISE_R[0]:.0f}〜{NOISE_R[1]:.0f} m", fontsize=10)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("しきい値 [dB]（訓練セル平均からの比）")
    axes[0].set_ylabel("1 フレームあたりの検出セル数")
    axes[0].legend(fontsize=8)
    fig.suptitle("しきい値を振ったときの検出の数（12 本の平均）。点線は図の種類ごとに決めたしきい値", fontsize=10)
    fig.savefig(FIG / "cfar_count_vs_threshold.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("-> figures/cfar_count_vs_threshold.png")

    # 図 3: 距離−時間図に検出点を重ねる（走行ごと）
    for n in RUNS:
        m = maps[n]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
        for ax, k in zip(axes, KINDS):
            im = draw_rt(ax, m, k)
            f_i, r_i = np.nonzero(det[n][k] & road[None, :])
            ax.scatter(rng[r_i], m["t"][f_i], s=4, c="red", lw=0)
            ax.set_title(f"{KIND_JA[k]}\nしきい値 {alpha[k]:.1f} dB", fontsize=10)
            fig.colorbar(im, ax=ax, label="受信電力 [dB]", shrink=0.8)
        axes[0].set_ylabel("時刻 [s]")
        mode, lane, dr = condition(n)
        fig.suptitle(f"10/1 夜 {run_label(n)}（距離−時間 ＋ 距離方向 CA-CFAR の検出 = 赤点、"
                     f"上端の白い線の長さ = 距離分解能 {RANGE_RES:.2f} m）", fontsize=11)
        out = FIG / f"cfar_No{n:02d}_{mode}_lane{lane}_{dr}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out.relative_to(HERE)}")

    # 図 4: overview（図の種類ごとに 12 本）
    for k in KINDS:
        fig, axes = plt.subplots(2, 6, figsize=(3.2 * 6, 10), sharey=True, gridspec_kw=dict(hspace=0.35))
        for j, (ax, n) in enumerate(zip(axes.flat, RUNS)):
            im = draw_rt(ax, maps[n], k, label=(j == 0))
            f_i, r_i = np.nonzero(det[n][k] & road[None, :])
            ax.scatter(rng[r_i], maps[n]["t"][f_i], s=3, c="red", lw=0)
            mode, lane, dr = condition(n)
            ax.set_title(f"No.{n}\n{MODE_JA[mode]} レーン{lane} {DR_JA[dr]}", fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel("時刻 [s]")
        fig.suptitle(f"10/1 夜の 12 本（{KIND_JA[k]}）＋ 距離方向 CA-CFAR の検出（赤点、しきい値 {alpha[k]:.1f} dB）\n"
                     f"上端の白い線の長さ = 距離分解能 {RANGE_RES:.2f} m", fontsize=11)
        fig.colorbar(im, ax=axes, label="受信電力 [dB]", shrink=0.6)
        out = FIG / f"cfar_overview_{k}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
