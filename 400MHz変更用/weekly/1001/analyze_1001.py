# 10/1 日中の予備測定を定量化する。
#
# 目的は2つ:
#   (a) 検出・追跡が実データで成立するかの確認（しきい値・ゲートの決定）
#   (b) 当てはめで横方向離隔 d が決まるかの確認
#
# (b) は決まらない。走行区間が最接近点から離れており、R(t) がほぼ直線になるため
# d と v が縮退する。本スクリプトはその度合いを rms の d 依存として定量化する。
#
# 使い方:
#   python analyze_1001.py                 # 表と図を出す
#   python analyze_1001.py --thr 22        # しきい値を変える

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "0727"))
sys.path.insert(0, str(ROOT.parent / "0928"))
from atlas_ridge_track import build_ridges, moving_power_db   # noqa: E402

# 収録順と走行条件。被験者の申告による
# ファイル名末尾6桁は収録開始時刻 HHMMSS。16:55:06 から約4分で4本。
# 徒歩で1往復（遠→近, 近→遠）してから自転車に乗り換え、同じ道をもう1往復した。
# いずれも道の中央を通行しており、走行ラインを分けた収録ではない。図の軸は ASCII にする
RUNS = {
    "165506": ("16:55:06 徒歩・遠→近", "16:55:06 walk, far->near"),
    "165623": ("16:56:23 徒歩・近→遠", "16:56:23 walk, near->far"),
    "165822": ("16:58:22 自転車・遠→近", "16:58:22 bike, far->near"),
    "165922": ("16:59:22 自転車・近→遠", "16:59:22 bike, near->far"),
}
RMIN, RMAX = 25.0, 55.0      # 近傍（窓枠・三脚）と遠方を外す
NOISE_R = 60.0               # ここより遠方を雑音床の参照にする


def load(tag):
    npz = ROOT / "data/export" / f"atlas_log_20261001_{tag}" / f"atlas_log_20261001_{tag}_rd.npz"
    return moving_power_db(npz)                               # pw, vel, rng, t, v_ax, db, k


def pick_ridge(pw, vel, rng, t, v_ax, thr):
    """レンジ変化率とドップラーが一致する尾根のうち最長のものを走行とみなす。

    両者は別々の物理量（フレーム間の位置の傾き / 1フレーム内の位相変化）から出るので、
    一致すれば実在の移動目標と言える。持続クラッタの帯は傾き 0 なので弾ける。
    """
    keep = (rng >= RMIN) & (rng <= RMAX)
    gate = int(np.ceil(abs(v_ax).max() * (t[1] - t[0]) / (rng[1] - rng[0])))
    ridges = build_ridges(pw[:, keep], vel[:, keep], rng[keep], t, thr, gate,
                          max_miss=3, min_run=5, dop_gate=2.0, v_res=abs(v_ax[1] - v_ax[0]))
    real = [g for g in ridges if abs(g["slope"] - g["vel"]) < 1.0 and abs(g["slope"]) > 0.3]
    return max(real, key=lambda g: len(g["f"])) if real else None


def profile_d(r_obs, t, d_grid):
    """d を固定して (v, t_min) を最適化し、残差の d 依存を見る。

    R^2 = v^2 (t - t_min)^2 + d^2 なので、d を決めれば |t - t_min| = sqrt(R^2-d^2)/v。
    d が決まらない（残差が d に鈍感）なら、その軌跡からは横方向離隔を推定できない。
    """
    out = []
    for d in d_grid:
        best = None
        for v in np.linspace(0.5, 6.0, 221):
            q = np.sqrt(np.maximum(r_obs**2 - d**2, 0)) / v
            sign = -1.0 if r_obs[-1] < r_obs[0] else +1.0     # 接近なら t < t_min
            t_min = np.mean(t - sign * q)
            rms = np.sqrt(np.mean((np.hypot(v * (t - t_min), d) - r_obs)**2))
            if best is None or rms < best[1]:
                best = (v, rms)
        out.append((d, *best))
    return out                                                 # [(d, v, rms), ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=22.0)
    args = ap.parse_args()

    print(f"近傍ゲート {RMIN}-{RMAX} m / しきい値 {args.thr} dB\n")
    print(f"{'file':>8} {'条件':<12} {'n':>4} {'時間[s]':>12} {'R[m]':>14} "
          f"{'dR/dt':>7} {'dop':>7} {'強度':>7} {'雑音床':>7}")

    tracks, fig1 = {}, plt.figure(figsize=(16, 4.2))
    for i, (tag, (label, label_en)) in enumerate(RUNS.items()):
        pw, vel, rng, t, v_ax, db, k = load(tag)
        nf = float(np.median(db[:, k >= 1, :][:, :, rng > NOISE_R]))
        g = pick_ridge(pw, vel, rng, t, v_ax, args.thr)
        if g is None:
            print(f"{tag:>8} {label:<12} —  尾根なし")
            continue
        tracks[tag] = g
        print(f"{tag:>8} {label:<12} {len(g['f']):4d} "
              f"{g['t'][0]:5.1f}-{g['t'][-1]:5.1f} "
              f"{g['rng'][0]:6.1f}->{g['rng'][-1]:6.1f} "
              f"{g['slope']:+7.2f} {g['vel']:+7.2f} "
              f"{np.median(g['lev']):7.1f} {nf:7.1f}")

        ax = fig1.add_subplot(1, len(RUNS), i + 1)
        ax.plot(g["t"], g["rng"], "o-", ms=4)
        ax.set_title(f"{label_en}\nslope {g['slope']:+.2f} / dop {g['vel']:+.2f}", fontsize=9)
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("R [m]")
    fig1.tight_layout()
    fig1.savefig(ROOT / "figures/tracks.png", dpi=120)

    # d が決まらないことの定量化。最も点数の多い軌跡で見る
    tag = max(tracks, key=lambda k: len(tracks[k]["f"]))
    g = tracks[tag]
    d_grid = np.arange(5.0, g["rng"].min(), 2.5)
    prof = profile_d(g["rng"], g["t"], d_grid)
    print(f"\n当てはめの d 依存（{tag} {RUNS[tag][0]}、n={len(g['f'])}、観測の最小 R={g['rng'].min():.1f} m）")
    print(f"{'d[m]':>6} {'v[m/s]':>8} {'rms[m]':>8}")
    for d, v, rms in prof:
        print(f"{d:6.1f} {v:8.2f} {rms:8.3f}")
    rmss = np.array([p[2] for p in prof])
    print(f"\n  d を {d_grid[0]:.0f}〜{d_grid[-1]:.0f} m と {d_grid[-1]/d_grid[0]:.0f} 倍動かしても "
          f"rms は {rmss.min():.3f}〜{rmss.max():.3f} m（{rmss.max()/rmss.min():.2f} 倍）"
          f"\n  -> d は決まらない。最接近点が走行区間に入っていないため")

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p[0] for p in prof], rmss, "o-")
    ax.set_xlabel("assumed lateral offset d [m]")
    ax.set_ylabel("fit residual rms [m]")
    ax.set_title(f"d is not determined ({tag}, n={len(g['f'])})")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(ROOT / "figures/d_profile.png", dpi=120)
    print(f"\n-> figures/tracks.png, figures/d_profile.png")


if __name__ == "__main__":
    main()
