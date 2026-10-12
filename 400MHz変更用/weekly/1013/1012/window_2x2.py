# README の W（定性版）: 窓 開/閉 × 人 あり/なし の 4 本を、同じ色の範囲の距離−時間図で並べる。
#
# 10/2 の指摘（with/without person、電力そのものを見る）に答えるための図なので、判定の数字は出さず、
# 目で比べられることを優先する。計算と描き方は ../1011/range_profile.py と同じ
# （電力和・MTI なし / 電力和・MTI あり / ドップラー最大・MTI あり）。
# 4 本で色の範囲をそろえないと、窓の効果（床の上がり・静止物の減衰）が見えなくなるので、種類ごとに共通にする。
#
# 使い方:
#   python -X utf8 window_2x2.py > run_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1011"))
sys.path.insert(0, str(HERE.parent / "collect"))
from range_profile import KIND_JA, KINDS, LIM, chirp_profiles, draw_rt, setup_font_ja   # noqa: E402
from check_run_1013 import export                                                    # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
# 収録の順（古い順）。人ありはどちらも自転車がレーン 1 を接近（10/12 の聞き取り）
RUNS = [
    ("145934", "閉", "人なし"),
    ("150040", "開", "人なし"),
    ("150757", "閉", "自転車 接近"),
    ("150939", "開", "自転車 接近"),
]
# 2 × 2 の並べ方: 行 = 人なし / 人あり、列 = 開 / 閉
GRID = {("開", "人なし"): (0, 0), ("閉", "人なし"): (0, 1),
        ("開", "自転車 接近"): (1, 0), ("閉", "自転車 接近"): (1, 1)}
BANDS = {"R 13-20 m": (13.0, 20.0), "R 20-28 m": (20.0, 28.0), "R 30-40 m": (30.0, 40.0),
         "R 40-49 m": (40.0, 49.0), "R 55-62 m": (55.0, 62.0)}


def run_maps(stem):
    """1 本の (F, R) マップ 3 種。../1011/range_profile.py run_maps と同じ計算（npz の場所だけ違う）"""
    dat = DATA / f"atlas_log_20261012_{stem}.dat"
    out = {}
    for nomti in (True, False):
        d = np.load(export(dat, DATA, nomti))
        rd, rng, vel, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]   # rd: (F, TX, RX, D, R)
        x = chirp_profiles(rd)                                                # (F, TX, RX, C, R)
        p = np.sum(np.abs(x) ** 2, axis=(1, 2, 3))                            # (F, R)
        out["sum_nomti" if nomti else "sum_mti"] = 10 * np.log10(p + 1e-30)
        if not nomti:
            dc = int(np.abs(vel).argmin())
            db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)          # (F, D, R)
            out["dmax_mti"] = np.delete(db, dc, axis=1).max(axis=1)           # (F, R)
    out.update(rng=rng, t=t)
    return out


def band_stats(m, kind):
    """帯ごとの時間・距離の中央値（床）と 99 パーセンタイル（明るいところ）[dB]。最初の 5 s は捨てる"""
    keep_t = m["t"] >= 5.0
    res = {}
    for name, (lo, hi) in BANDS.items():
        keep_r = (m["rng"] >= lo) & (m["rng"] < hi)
        v = m[kind][keep_t][:, keep_r]                                        # (F', R')
        res[name] = (np.median(v), np.percentile(v, 99))
    return res


def grid_figure(maps, kind, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.25, wspace=0.08))
    for stem, win, who in RUNS:
        i, j = GRID[(win, who)]
        ax = axes[i, j]
        im = draw_rt(ax, maps[stem], kind, label=(i == 0 and j == 0))
        ax.set_title(f"窓 {win}・{who}（{stem[:2]}:{stem[2:4]}:{stem[4:]}）", fontsize=11)
        if i == 0:
            ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("時刻 [s]")
    fig.suptitle(f"10/12 窓 開/閉 × 人 あり/なし（{KIND_JA[kind]}、色の範囲は 4 枚共通）", fontsize=12)
    fig.colorbar(im, ax=axes, label="受信電力 [dB]", shrink=0.6)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    setup_font_ja()
    FIG.mkdir(exist_ok=True)
    maps = {stem: run_maps(stem) for stem, _, _ in RUNS}
    for kind in KINDS:
        grid_figure(maps, kind, FIG / f"window_2x2_{kind}.png")

    print("\n帯ごとの床（中央値）/ 99 パーセンタイル [dB]、t >= 5 s")
    for kind in KINDS:
        print(f"\n[{KIND_JA[kind]}]")
        print("窓・人".ljust(14) + "".join(n.rjust(16) for n in BANDS))
        for stem, win, who in RUNS:
            s = band_stats(maps[stem], kind)
            print(f"{win}・{who}".ljust(14) + "".join(f"{a:7.1f} /{b:6.1f}".rjust(16) for a, b in s.values()))


if __name__ == "__main__":
    main()
