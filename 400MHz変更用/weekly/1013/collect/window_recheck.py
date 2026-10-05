# 9/28 の無人背景（窓を開ける / 閉める）を、10/2 の指摘に沿って見直す。
#
# 10/2 の指摘: 「集計値でなく RD マップの電力そのものを見る」「正規化した方がいい」。
# 9/28 の window_compare.py は R 13〜60 m を 1 つの絶対しきい値で数えたので、
# 床が最も上がる手前の帯が誤警報率を決めていた。ここでは
#   (1) RD マップの電力（時間平均と代表フレーム）を窓の開と閉で並べる
#   (2) 距離ビンごとの床（動体ビンの中央値）で正規化してから誤警報率を数え直す（CFAR 相当）
#   (3) 目標側の損失の目安として、屋外の静止物（DC ビン、MTI なし）が窓を閉めると何 dB 落ちるかを出す
# 無人背景しかないので、(3) は「静止物で測ったガラスの往復損失」であって、人の反射で測った値ではない。
#
# ファイル名の対応（9/28 verification_0928.md）: with_window = 窓あり = 閉める、without_window = 開ける
#
# 使い方:
#   python window_recheck.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT.parent.parent / "0928" / "data" / "export"
CASES = {"open": "without_window", "closed": "with_window"}

ROAD_R = (13.0, 60.0)       # 地上目標が現れうる距離帯（window_compare.py と同じ）
STATIC_R = (20.0, 55.0)     # ガラス損失を測る帯。手前はガラス自体の反射と近傍の多重反射が混ざるので外す
FA_TARGET = 1e-3


def load_db(name, mti):
    """(Frame, Doppler, Range) の電力 dB（8ch を振幅で加算。moving_power_db と同じ定義）と軸"""
    sub = name if mti else f"{name}_nomti"
    d = np.load(EXPORT / sub / f"{name}_rd.npz")
    db = 20 * np.log10(np.abs(d["rd"]).sum(axis=(1, 2)) + 1e-12)   # (F, D, R)
    return db, d["range_m"], d["vel_ms"]


def doppler_bin_index(vel):
    """DC からのビン番号 |k|。速度ビンは値でなく番号で切る（docs/realworld.md §5）"""
    dc = int(np.abs(vel).argmin())
    return np.abs(np.arange(len(vel)) - dc), dc


def floor_by_range(db, k, kmin=1):
    """距離ビンごとの雑音の床 = 動体ビン（|k| >= kmin）の全フレームの中央値。(R,)"""
    return np.median(db[:, k >= kmin, :], axis=(0, 1))


def normalized_fa(db, k, rng, thr, kmin=1):
    """距離ごとの床で正規化した超過量で、無人背景の誤警報率をしきい値ごとに数える。(T,)"""
    fl = floor_by_range(db, k, kmin)
    keep = (rng >= ROAD_R[0]) & (rng <= ROAD_R[1])
    ex = (db[:, k >= kmin, :] - fl)[:, :, keep].ravel()             # (N,)
    return (ex[None, :] > thr[:, None]).mean(axis=1)


def main():
    data = {}
    for key, name in CASES.items():
        for mti in (True, False):
            data[(key, mti)] = load_db(name, mti)
    _, rng, vel = data[("open", True)]
    k, dc = doppler_bin_index(vel)

    # ---- 数字 ----
    fl = {key: floor_by_range(data[(key, True)][0], k) for key in CASES}           # (R,)
    st = {key: np.median(data[(key, False)][0][:, dc, :], axis=0) for key in CASES}  # (R,) 静止物
    print("(1) 動体ビンの床（MTI あり、距離ごと中央値）: 閉 − 開 [dB]")
    for lo, hi in [(13, 20), (20, 28), (28, 40), (40, 60)]:
        b = (rng >= lo) & (rng < hi)
        print(f"    R {lo:2d}-{hi:2d} m: 開 {np.median(fl['open'][b]):5.1f} / 閉 {np.median(fl['closed'][b]):5.1f}"
              f"  → {np.median(fl['closed'][b] - fl['open'][b]):+.1f}")

    # 静止物が床から十分出ているビンだけで差を取る（床に埋もれたビンは損失を過小に見せる）
    sb = (rng >= STATIC_R[0]) & (rng <= STATIC_R[1]) & (st["open"] > fl["open"] + 20)
    loss = st["open"][sb] - st["closed"][sb]
    print(f"(2) 屋外の静止物（MTI なし DC ビン）の落ち込み R {STATIC_R[0]:.0f}-{STATIC_R[1]:.0f} m、"
          f"{sb.sum()} ビン: 中央値 {np.median(loss):.1f} dB（四分位 {np.percentile(loss, 25):.1f}〜"
          f"{np.percentile(loss, 75):.1f}）")

    thr = np.arange(0.0, 40.0, 0.25)
    print(f"(3) 距離ごとの床で正規化した誤警報率 {FA_TARGET:.0e} に要る床超過 [dB]（R {ROAD_R[0]:.0f}-{ROAD_R[1]:.0f} m）")
    fa_all = {}
    for key in CASES:
        for mti in (True, False):
            for kmin in (1, 2):
                fa = normalized_fa(data[(key, mti)][0], k, rng, thr, kmin)
                fa_all[(key, mti, kmin)] = fa
                need = thr[np.argmax(fa < FA_TARGET)] if (fa < FA_TARGET).any() else np.nan
                print(f"    {key:6} MTI {'on ' if mti else 'off'} |k|>={kmin}: {need:5.2f}")

    # ---- 図 ----
    fig = plt.figure(figsize=(12, 11), layout="constrained")
    gs = fig.add_gridspec(3, 2)
    rmax = 60.0
    rk = rng <= rmax
    f_rep = len(data[("open", True)][0]) // 2                       # 代表フレーム = 収録の真ん中
    for col, key in enumerate(CASES):
        db_off = data[(key, False)][0]
        db_on = data[(key, True)][0]
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(rng[rk], vel, db_off.mean(axis=0)[:, rk], shading="auto", cmap="viridis",
                           vmin=10, vmax=70)
        ax.set_title(f"window {key}: time-mean RD, MTI off", fontsize=10)
        ax.set_ylabel("velocity [m/s]")
        fig.colorbar(im, ax=ax, label="power [dB, relative]")
        ax = fig.add_subplot(gs[1, col])
        im = ax.pcolormesh(rng[rk], vel, db_on[f_rep][:, rk], shading="auto", cmap="viridis", vmin=10, vmax=35)
        ax.set_title(f"window {key}: frame {f_rep} (t = {f_rep * 0.2:.0f} s), MTI on", fontsize=10)
        ax.set_ylabel("velocity [m/s]")
        ax.set_xlabel("range R [m]")
        fig.colorbar(im, ax=ax, label="power [dB, relative]")

    ax = fig.add_subplot(gs[2, 0])
    for key, c in (("open", "C0"), ("closed", "C1")):
        ax.plot(rng[rk], fl[key][rk], c, lw=1.5, label=f"moving-bin floor, {key}")
        ax.plot(rng[rk], st[key][rk], c, lw=1, ls="--", label=f"static (DC, MTI off), {key}")
    ax.axvspan(22, 28, color="tab:red", alpha=0.08)
    ax.text(25, 72, "near end\n22-28 m", ha="center", va="top", fontsize=8, color="tab:red")
    ax.set_xlabel("range R [m]")
    ax.set_ylabel("power [dB, relative]")
    ax.set_title(f"floor rises at short range; outdoor static drops ~{np.median(loss):.0f} dB through glass",
                 fontsize=10)
    ax.set_ylim(5, 75)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = fig.add_subplot(gs[2, 1])
    # MTI なしの ±1 ビンは静止物の主ローブ（窓関数の裾）がそのまま入るので比較にならない。
    # MTI なしは |k| >= 2 だけで描く
    for (key, mti, kmin), fa in fa_all.items():
        if kmin != (1 if mti else 2):
            continue
        ax.semilogy(thr, np.maximum(fa, 1e-7), "C0" if key == "open" else "C1",
                    ls="-" if mti else "--", label=f"{key}, MTI {'on, |k|>=1' if mti else 'off, |k|>=2'}")
    ax.axhline(FA_TARGET, color="gray", ls=":", lw=1)
    ax.set_xlabel("threshold above per-range floor [dB]")
    ax.set_ylabel("false-alarm rate (unmanned background)")
    ax.set_title(f"after per-range normalization, closed is not worse (R {ROAD_R[0]:.0f}-{ROAD_R[1]:.0f} m)",
                 fontsize=10)
    ax.set_ylim(1e-6, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    out = ROOT / "figures" / "window_recheck.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
