# 距離−時間図の MTI なし版。figures/range_time.png（日中4本）と figures/night_range_time.png（夜の代表6本）
# と同じ並びで描く。
#
# MTI あり版は静止成分を消した後の図なので、「消した後に何が残ったか」しか見えない。
# 手前（R ≤ 30 m）で目標が見えない理由を議論するには、そもそも静止クラッタがどの距離に
# どれだけ居るか（目標がその下に埋もれていないか）を、処理を挟まずに見ておく必要がある。
#
# 電力の定義は MTI あり版と揃え、DC ビンを除外しないことだけを変える:
#   20 log10( Σ_TX,RX |rd| ) をドップラー方向に最大（DC 含む）
#
# 使い方:
#   python plot_range_time_nomti.py
#   （事前に data/export_nomti/ に atlas_export.py --no-mti の出力が要る）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_1001 import RUNS
from analyze_night import CONES, TAGS, ROOT, condition, npz_path

RMAX = 60.0
VMIN, VMAX = 20.0, 80.0     # MTI なしは静止クラッタが 50〜80 dB に達するので、MTI あり版（10〜35/40）とは別スケール


def power_nomti_db(tag):
    """MTI なしの (Frame, Range) 電力 [dB]。DC ビンも含めたドップラー方向の最大"""
    d = np.load(npz_path(tag, nomti=True))
    db = 20 * np.log10(np.abs(d["rd"]).sum(axis=(1, 2)) + 1e-12)   # (F, D, R)
    return db.max(axis=1), d["range_m"], d["t_rel_s"]               # (F, R), (R,), (F,)


def draw(tags, titles, out, cones):
    fig, axes = plt.subplots(1, len(tags), figsize=(3.3 * len(tags), 4.6), sharey=True)
    for ax, tag, title in zip(axes, tags, titles):
        pw, rng, t = power_nomti_db(tag)
        keep = rng <= RMAX
        im = ax.pcolormesh(rng[keep], t, pw[:, keep], shading="auto", cmap="viridis", vmin=VMIN, vmax=VMAX)
        if cones:
            for r in CONES.values():
                ax.axvline(r, color="r", lw=0.6, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("range R [m]")
    axes[0].set_ylabel("time [s]")
    fig.colorbar(im, ax=axes, label="power, no MTI, DC included [dB]")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"-> {out.relative_to(ROOT)}")


def main():
    # 日中4本（range_time.png と同じ順）
    draw(list(RUNS), [en for _, en in RUNS.values()], ROOT / "figures/range_time_nomti.png", cones=False)

    # 夜の代表6本（night_range_time.png と同じ: 自転車の各レーン、接近と離反）
    pick = [19, 20, 25, 26, 31, 32]
    titles = []
    for n in pick:
        mode, lane, dr = condition(n)
        titles.append(f"#{n} {mode} lane{lane} {'approach' if dr == 'app' else 'depart'}")
    draw([TAGS[n - 1] for n in pick], titles, ROOT / "figures/night_range_time_nomti.png", cones=True)


if __name__ == "__main__":
    main()
