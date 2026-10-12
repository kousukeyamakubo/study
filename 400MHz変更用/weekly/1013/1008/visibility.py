# そもそも目標が見えているかを、真値の位置に頼らずに確かめる（README「見えているかの上限」）。
#
# 等速の真値は人が漕ぐ・歩く以上は崩れうるので、「真値から 2 m 以内」では見えているかを判定しない。
# 代わりに、各フレームで区間（R 30〜49 m）の中の一番強い反射（MTI、DC 以外のドップラー）を取り、
# 動く人として筋が通っているかを 3 つの条件で見る:
#   1. つながる : 前後のフレームの一番強い反射と、人の速さで動ける範囲（STEP_M）でつながり、MIN_LEN フレーム以上続く
#   2. 向き     : そのつながりの中の局所的な距離の変化率 dR/dt が、走行の向き（接近なら負）と合い、MIN_RATE 以上
#   3. ドップラー: 一番強い反射のドップラーの速度が、dR/dt と DOP_TOL 以内で合う（動く本物の反射だけが満たす）
# 真値（等速の当てはめ）は「人が区間の中にいる時間帯」を決めるのにだけ使う。
# 偶然の水準として、人が区間にいないフレーム（区間の端から 3 m 以上外か走路にいない）にも同じ判定を掛ける。
#
# 一番強い反射は 2 通りで取る:
#   raw    : MTI の電力そのもの（8ch の電力和）
#   excess : 各セル（ドップラー, 距離）の時間中央値からの浮き。静止物の消え残りのような、いつも強いセルを下げる
#
# 使い方:
#   python visibility.py > visibility_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cfar_3gpp as c3                                           # noqa: E402

ZONE = c3.ZONE
FRAME_S = 0.2
V_MAX = {"walk": 2.5, "bike": 5.0}       # 1 フレームで動ける距離の上限に使う速さ [m/s]
MIN_LEN = 5                              # つながりとみなす最短のフレーム数（1 s）
SLOPE_HALF = 3                           # dR/dt を当てはめる前後のフレーム数（±0.6 s）
MIN_RATE = 0.5                           # 動いているとみなす |dR/dt| [m/s]。止まった消え残りを落とす
DOP_TOL = 1.0                            # ドップラーの速度と dR/dt の差の許容 [m/s]（ビン幅 0.6 m/s と距離の量子化 0.85 m の分）
MARGINS = [0.0, 3.0]                     # 「区間にいる」を区間の端からこれだけ内側に絞る [m]（真値の時刻のずれに備える）


def strongest(p, use, rng, mode):
    """区間の中の一番強い反射の距離ビン・ドップラービン (F,) ×2。p: (F, D, R)"""
    zr = (rng >= ZONE[0]) & (rng <= ZONE[1])                       # (R,)
    if mode == "excess":
        p = p / np.median(p, axis=0, keepdims=True)                # (F, D, R) 各セルの時間中央値からの浮き
    q = np.where(use[None, :, None] & zr[None, None, :], p, -np.inf)
    flat = q.reshape(len(p), -1).argmax(axis=1)
    kd, kr = np.unravel_index(flat, q.shape[1:])
    return kr, kd


def judge(kr, kd, rng, vel, t, step_m, sign):
    """3 つの条件の判定。(F,) の bool ×3（1、1+2、1+2+3）と、局所の dR/dt (F,)"""
    F = len(kr)
    r = rng[kr]
    link = np.abs(np.diff(r)) <= step_m                            # (F-1,) f と f+1 がつながる
    # つながりの区間に番号をつける
    seg = np.concatenate([[0], np.cumsum(~link)])                  # (F,)
    seg_len = np.bincount(seg)[seg]
    c1 = seg_len >= MIN_LEN
    rate = np.full(F, np.nan)
    for f in np.flatnonzero(c1):
        idx = np.arange(max(f - SLOPE_HALF, 0), min(f + SLOPE_HALF, F - 1) + 1)
        idx = idx[seg[idx] == seg[f]]
        if len(idx) >= 4:
            rate[f] = np.polyfit(t[idx], r[idx], 1)[0]
    c2 = c1 & np.isfinite(rate) & (rate * sign >= MIN_RATE)
    # truth_path と同じく、ドップラーの速度は dR/dt と同じ符号
    c3_ = c2 & (np.abs(vel[kd] - rate) <= DOP_TOL)
    return c1, c2, c3_, rate


def main():
    runs = c3.load_runs()
    print(f"一番強い反射（区間 R {ZONE[0]:g}〜{ZONE[1]:g} m、MTI、DC 以外のドップラー）が動く人として筋が通っているか。12 本")
    print(f"条件: 1 つながる（1 フレームで v_max×0.2 s＋1 ビン以内、{MIN_LEN} フレーム以上）、"
          f"2 向き（|dR/dt| ≥ {MIN_RATE:g} m/s で向きが合う）、3 ドップラー（|v_dop − dR/dt| ≤ {DOP_TOL:g} m/s）")
    print(f"v_max: 徒歩 {V_MAX['walk']:g}・自転車 {V_MAX['bike']:g} m/s。dR/dt は前後 {SLOPE_HALF} フレームの直線の当てはめ")
    results = {}
    for mode in ("raw", "excess"):
        rows = []
        for run in runs:
            dr = run["rng"][1] - run["rng"][0]
            step = V_MAX[run["mode"]] * FRAME_S + dr
            kr, kd = strongest(run["p"], run["use"], run["rng"], mode)
            c1, c2, c3_, rate = judge(kr, kd, run["rng"], run["vel"], run["t"], step, run["sign"])
            r, on = run["r"], run["on"]
            inz = {m: on & (r >= ZONE[0] + m) & (r <= ZONE[1] - m) for m in MARGINS}
            off = ~on | (r < ZONE[0] - c3.T1_MARGIN) | (r > ZONE[1] + c3.T1_MARGIN)
            # 見えているフレームの、一番強い反射と真値の BEV の離れ（参考。真値はずれうる）
            x_pk = np.sqrt(np.maximum(run["rng"][kr] ** 2 - run["d"] ** 2, 0.0))
            dx = np.abs(x_pk - np.abs(run["s"]))
            rows.append(dict(run=run, kr=kr, c=(c1, c2, c3_), inz=inz, off=off, dx=dx))
        results[mode] = rows

        print("\n" + "=" * 80)
        print(f"一番強い反射の取り方: {mode}")
        print("=" * 80)
        for m in MARGINS:
            print(f"\n  人が区間にいるフレーム（区間の端から {m:g} m 内側）で、条件を満たした割合（1 / 1+2 / 1+2+3）")
            groups = {}
            for x in rows:
                run = x["run"]
                g = f"{run['mode']} lane{run['lane']}"
                n = x["inz"][m].sum()
                k = [int((c & x["inz"][m]).sum()) for c in x["c"]]
                groups.setdefault(g, []).append((n, k))
                if m == MARGINS[0]:
                    print(f"    No.{run['n']:2d} {g} {run['dr']}: {n:3d} フレーム → "
                          + " / ".join("-" if n == 0 else f"{v / n:.2f}" for v in k))
            for g, xs in groups.items():
                n = sum(z[0] for z in xs)
                k = np.sum([z[1] for z in xs], axis=0)
                print(f"    {g:12}: {n:3d} フレーム → " + " / ".join(f"{v / max(n, 1):.2f}" for v in k))
            n = sum(x["inz"][m].sum() for x in rows)
            k = [sum(int((x["c"][i] & x["inz"][m]).sum()) for x in rows) for i in range(3)]
            print(f"    {'全体':12}: {n:3d} フレーム → " + " / ".join(f"{v / max(n, 1):.2f}" for v in k))
        n = sum(x["off"].sum() for x in rows)
        k = [sum(int((x["c"][i] & x["off"]).sum()) for x in rows) for i in range(3)]
        print(f"\n  偶然の水準: 人が区間にいないフレーム {n} で、条件を満たした割合 " + " / ".join(f"{v / n:.2f}" for v in k))
        # レーン 2 は等速の当てはめが筋から外れていて（figures/visibility_raw.png）、「いない」フレームに人が入っているので外した値も出す
        l1 = [x for x in rows if x["run"]["lane"] == 1]
        n = sum(x["off"].sum() for x in l1)
        k = [sum(int((x["c"][i] & x["off"]).sum()) for x in l1) for i in range(3)]
        print(f"  同じ、レーン 1 の 9 本だけ（{n} フレーム）: " + " / ".join(f"{v / n:.2f}" for v in k))
        # 位置の参考
        vis = np.concatenate([x["dx"][x["c"][2] & x["inz"][0.0]] for x in rows])
        print(f"  参考: 見えている（1+2+3）フレームの、一番強い反射と等速の真値の BEV の離れ: 中央値 {np.median(vis):.1f} m、"
              f"2 m 超 {(vis > 2).mean():.2f}、5 m 超 {(vis > 5).mean():.2f}（{len(vis)} フレーム）")

    # 図: 走行ごとの一番強い反射の距離−時間（raw）。条件を満たした点と満たさない点、等速の真値
    rows = results["raw"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True, layout="constrained")
    for ax, x in zip(axes.ravel(), rows):
        run = x["run"]
        t, rp = run["t"], run["rng"][x["kr"]]
        ok = x["c"][2]
        ax.axhspan(ZONE[0], ZONE[1], color="tab:blue", alpha=0.05)
        ax.plot(t[~ok], rp[~ok], "x", color="0.6", ms=4, label="strongest, fails 1-3")
        ax.plot(t[ok], rp[ok], "o", color="tab:green", ms=4, mfc="none", label="strongest, passes 1+2+3")
        on = run["on"]
        ax.plot(t[on], run["r"][on], "-", color="tab:red", lw=0.8, label="constant-speed fit (time window only)")
        inz = x["inz"][0.0]
        ax.set_title(f"No.{run['n']} {run['mode']} lane{run['lane']} {run['dr']}: "
                     f"{(ok & inz).sum()}/{inz.sum()} visible", fontsize=9)
        ax.set_ylim(25, 55)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("range R [m]")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Strongest reflection in R 30-49 m (MTI, non-DC Doppler, raw power), night 10/1. "
                 "Visible = continuous, moving in the run direction, Doppler matches dR/dt", fontsize=10)
    o = ROOT / "figures" / "visibility_raw.png"
    fig.savefig(o, dpi=100)
    print(f"\n-> {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
