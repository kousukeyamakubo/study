# npz から移動目標の軌跡を抽出して range-time 図にする
#
# 5F 設置の実測では、歩行者は軌跡上の中央値で 22 dB 程度しかなく、単一フレームの
# しきい値処理では見逃す（0728 の 165256 で実証）。フレーム間の連続性で拾うための道具。
#
# 判定は「レンジ変化率（フレーム間の位置の傾き）と Doppler が独立に一致するか」。
# 両者は別々の物理量から出るので、一致すれば実在の移動目標と言える。
#
# 使い方:
#   python atlas_track.py export_out_ped/atlas_log_20260728_165256_rd.npz

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--rmin", type=float, default=13.0,
                    help="この距離未満を捨てる。既定 13m = 5F(h≈15m)から地上目標が現れうる下限。"
                         "室内の操作者と窓枠を物理的に排除できる")
    ap.add_argument("--rmax", type=float, default=55.0)
    ap.add_argument("--thr", type=float, default=20.0, help="検出しきい値[dB]（雑音床 ≈13dB）")
    ap.add_argument("--min-run", type=int, default=3, help="連続で必要なフレーム数")
    ap.add_argument("--veto-db", type=float, default=30.0,
                    help="近距離(<5m)がこの値を超えるフレームは棄却。操作者が動くと"
                         "レンジサイドローブが遠方まで滲むため（0728 163850 で確認）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    rd, r, v, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]
    db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)     # (Frame, Doppler, Range)

    # 速度ビンは 0.5995 m/s 刻み。|v|>=0.6 のような書き方だと ±1 ビン目（歩行速度）が
    # 浮動小数の比較で漏れるので、DC からのビン番号で切る
    dc = int(np.abs(v).argmin())
    k = np.abs(np.arange(len(v)) - dc)
    mov = np.where((k >= 1)[:, None], db, -np.inf)

    veto = db[:, k >= 1, :][:, :, r < 5].max(axis=(1, 2)) > args.veto_db
    band = (r >= args.rmin) & (r <= args.rmax)
    rb, m = r[band], mov[:, :, band]

    # 各フレームの最大セル
    pk = [np.unravel_index(f.argmax(), f.shape) for f in m]
    lev = np.array([m[i][p] for i, p in enumerate(pk)])
    rng = np.array([rb[p[1]] for p in pk])
    vel = np.array([v[p[0]] for p in pk])
    ok = (lev > args.thr) & ~veto

    # 連続区間のうち Doppler の符号が揃っている最長のものを軌跡とみなす
    best, cur = [], []
    for i in range(len(ok)):
        if ok[i] and (not cur or np.sign(vel[i]) == np.sign(vel[cur[0]])):
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [i] if ok[i] else []
    if len(cur) > len(best):
        best = cur

    nf = np.median(db[:, k >= 1, :][:, :, r > args.rmax])
    print(f"{args.npz.name}")
    print(f"  雑音床 {nf:.1f} dB  しきい値 {args.thr:.0f} dB  棄却フレーム {veto.sum()} / {len(t)}")

    if len(best) < args.min_run:
        print(f"  軌跡なし（最長の連続区間 {len(best)} フレーム < {args.min_run}）")
        return

    i0, i1 = best[0], best[-1]
    slope = np.polyfit(t[best], rng[best], 1)[0]
    v_med = np.median(vel[best])
    dv = abs(slope - v_med)
    print(f"  軌跡: t={t[i0]:.1f}〜{t[i1]:.1f}s（{len(best)}フレーム）  "
          f"{rng[i0]:.2f} → {rng[i1]:.2f} m")
    print(f"    レンジ変化率 {slope:+.2f} m/s  vs  Doppler中央値 {v_med:+.2f} m/s  "
          f"差 {dv:.2f}（分解能の {dv/abs(v[dc+1]-v[dc]):.2f} 倍）")
    print(f"    レベル {lev[best].min():.1f}〜{lev[best].max():.1f} dB（中央値 {np.median(lev[best]):.1f}）")
    print(f"    → {'実在の移動目標と判定' if dv < abs(v[dc+1]-v[dc]) else '一致せず。要確認'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mov.max(axis=1), aspect="auto", origin="lower",
                   extent=[r[0], r[-1], t[0], t[-1]], vmin=nf - 1, vmax=nf + 20)
    fig.colorbar(im, label="Moving-target power [dB]")
    ax.plot(rng[best], t[best], "r.", ms=6, label="detected peak")
    ax.plot(slope * t[best] + np.polyfit(t[best], rng[best], 1)[1], t[best], "w--", lw=1,
            label=f"fit {slope:+.2f} m/s (Doppler {v_med:+.2f})")
    ax.axvline(args.rmin, color="orange", ls=":", lw=1, label=f"gate {args.rmin:.0f} m")
    ax.set_xlim(0, args.rmax); ax.set_xlabel("Slant range [m]"); ax.set_ylabel("Time [s]")
    # 図中は日本語フォント未設定のため英語（他のスクリプトと同じ方針）
    ax.set_title(f"{args.npz.stem}  track")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = args.out or args.npz.with_name("target_track.png")
    fig.savefig(out, dpi=150)
    print(f"  保存: {out}")


if __name__ == "__main__":
    main()
