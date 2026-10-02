# 夜の各走行で、「走路の真値から決まる R の範囲」と、レーダーの距離−時間を重ねる。1走行1枚。
#
# 真値は空間の経路そのもの。コーン4隅で走路の長方形が、その中のレーン（道幅 4 m 実測・1 m 間隔）で
# 通った線が決まっている。ただし「いつ線上のどこに居たか」は映像なしでは決まらないので、
# 時刻つきの理論曲線は描かない（v と t_min を当てはめで埋めると、追跡の誤りに引きずられて真値でなくなる）。
#
# 時刻を使わずに真値から言えるのは、その経路を通ったなら観測される R の範囲:
#   手前の端 = 最接近距離 d（垂線の足は走路の中、C1 から奥へ S0）
#   奥の端   = C5/C6 の並び（垂線の足から L_C1_C5 - S0）
# この範囲のうち、どこまで目標の筋が見えているかを背景の強度で確かめる。
# レーン間の d の差は 0.6 m 程度で図の上では区別できないので、レーンごとの線は描かない。
#
# 使い方:
#   python plot_lane_truth.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_night import (CONES, L_C1_C5, N_MAIN, ROOT, S0, TAGS, condition, lane_delta_d,
                           npz_path, pick_ridge)
from atlas_ridge_track import moving_power_db

LANE_OFFSET = {3: 1.0, 2: 2.0, 1: 3.0}     # 右の縁（レーダーに近い側）からの水平距離 [m]


def lane_range(lane):
    """申告レーンを通ったときに観測される R の範囲 (R_near, R_far) [m]"""
    a, h, _ = lane_delta_d(road_w=4.0)     # a: レーダー直下→右の縁、h: 設置高
    d = float(np.hypot(a + LANE_OFFSET[lane], h))
    return d, float(np.hypot(L_C1_C5 - S0, d))


def plot_start_by_lane(mode, dr, r_lo, r_hi, out):
    """同じ手段・向きの9本（3レーン × 3回）を、開始点付近の距離帯に拡大して並べる。

    接近の開始点は奥の端（C5/C6）、離反の開始点は手前の端（C1/C2）。
    横方向 1 m のずれが R に効く割合は (横方向の水平距離)/R なので、奥では約 0.27 m、
    手前では約 0.6 m。レーン差が R に出るかを、真値の開始 R の線と並べて見る
    """
    runs = [n for n in range(1, N_MAIN + 1) if condition(n)[0] == mode and condition(n)[2] == dr]
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, layout="constrained")
    for n in runs:
        _, lane, _ = condition(n)
        rep = sum(1 for m in runs if condition(m)[1] == lane and m < n)       # 0, 1, 2
        ax = axes[lane - 1, rep]
        pw, _, rng, t, *_ = moving_power_db(npz_path(TAGS[n - 1]))       # (F, R), (R,), (F,)
        keep = (rng >= r_lo - 1) & (rng <= r_hi + 1)
        im = ax.pcolormesh(t, rng[keep], pw[:, keep].T, shading="auto", cmap="gray_r", vmin=12, vmax=35)
        for o in (1, 2, 3):
            r0 = lane_range(o)[1 if dr == "app" else 0]                   # 開始点の R
            ax.axhline(r0, color=colors[o], lw=1.6 if o == lane else 0.8, ls="-" if o == lane else ":")
        ax.set_title(f"No.{n}  lane{lane}  rep{rep + 1}", fontsize=9)
        ax.set_ylim(r_lo, r_hi)
    for i in range(3):
        axes[i, 0].set_ylabel(f"lane{i + 1}\nrange R [m]")
        axes[2, i].set_xlabel("time [s]")
    handles = [plt.Line2D([], [], color=colors[o], label=f"lane{o} start R (truth) "
                          f"{lane_range(o)[1 if dr == 'app' else 0]:.2f} m") for o in (1, 2, 3)]
    fig.legend(handles=handles, loc="outside lower center", fontsize=8, ncol=3)
    fig.suptitle(f"{mode} {'approach (start = far end C5/C6)' if dr == 'app' else 'depart (start = near end C1/C2)'}"
                 f"  -  solid: declared lane, dotted: other lanes. range bin 0.846 m", fontsize=10)
    fig.colorbar(im, ax=axes, label="moving power [dB]", shrink=0.6)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def main():
    for ln in (1, 2, 3):
        rn, rf = lane_range(ln)
        print(f"レーン{ln}: 真値の R 範囲 {rn:.2f}–{rf:.2f} m")

    outdir = ROOT / "figures/lane_truth"
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob("*.png"):          # 旧版（当てはめ曲線つき・31枚）を残さない
        old.unlink()

    for n in range(1, N_MAIN + 1):
        mode, lane, dr = condition(n)
        r_near, r_far = lane_range(lane)
        pw, _, rng, t, *_ = moving_power_db(npz_path(TAGS[n - 1]))   # (F, R), (R,), (F,)
        g = pick_ridge(TAGS[n - 1])

        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        keep = (rng >= 15) & (rng <= 55)
        im = ax.pcolormesh(t, rng[keep], pw[:, keep].T, shading="auto", cmap="gray_r", vmin=12, vmax=35)
        ax.axhspan(r_near, r_far, color="tab:blue", alpha=0.12,
                   label=f"truth: R range of the declared path ({r_near:.1f}-{r_far:.1f} m)")
        ax.axhline(r_near, color="tab:blue", lw=1.2)
        ax.axhline(r_far, color="tab:blue", lw=1.2)
        for name, r in CONES.items():
            ax.axhline(r, color="tab:red", lw=0.5, ls="--")
            ax.text(t[-1], r, f" {name}", color="tab:red", fontsize=7, va="center")
        if g is not None:
            # 追跡は濃い筋ではなく別の筋や静止物の帯を拾うことがある（No.19, 22, 28 で確認）。参考表示に留める
            ax.plot(g["t"], g["rng"], "o", ms=3, mfc="none", mec="tab:orange", alpha=0.8,
                    label="tracker output (reference; may follow a wrong streak)")
        ax.set_ylim(15, 55)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("range R [m]")
        ax.set_title(f"No.{n}  {mode} lane{lane} {'approach' if dr == 'app' else 'depart'}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, label="moving power [dB]")
        fig.savefig(outdir / f"No{n:02d}_{mode}_lane{lane}_{dr}.png", dpi=110)
        plt.close(fig)
    print(f"\n-> figures/lane_truth/ に {N_MAIN} 枚")

    plot_start_by_lane("bike", "app", 42, 54, ROOT / "figures/bike_app_start_by_lane.png")
    plot_start_by_lane("bike", "dep", 17, 35, ROOT / "figures/bike_dep_start_by_lane.png")


if __name__ == "__main__":
    main()
