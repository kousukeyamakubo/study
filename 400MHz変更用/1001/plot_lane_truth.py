# 夜の各走行の距離軌跡を、申告レーンの真値（理論曲線）と重ねて、ずれを定量化する。
#
# 真値はレーンの位置そのもの。道幅 4 m（実測）を 1 m 間隔の3レーンに分けているので、
# 右の縁（レーダーに近い側）から 1, 2, 3 m がレーン3, 2, 1。映像がなくても通った場所は分かる。
# そこから各レーンの最接近距離 d_lane が幾何で決まり、理論軌跡は
#   R(t)^2 = v^2 (t - t_min)^2 + d_lane^2
# となる。映像のコーン通過時刻は取れていないので、v と t_min だけはデータへの当てはめで決め、
# d は申告レーンの値に固定する。残差（観測 R − 理論 R）が「真値からのずれ」。
#
# 注意: 同じ当てはめを隣のレーンの d で行っても残差はほとんど変わらない（README §6 の縮退）。
# したがってこの図は「真値の曲線に沿っているか・どれだけずれているか」は示せるが、
# 「申告レーンの曲線が他レーンより良く合うか」はこのデータでは言えない。その差も併せて出す。
#
# 使い方:
#   python plot_lane_truth.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_night import CONES, N_MAIN, ROOT, S0, TAGS, condition, lane_delta_d, pick_ridge

LANE_OFFSET = {3: 1.0, 2: 2.0, 1: 3.0}     # 右の縁からの水平距離 [m]（1 m 間隔、道幅 4 m）


def lane_d():
    """各レーンの最接近距離 d [m]。a（レーダー直下→右の縁）と h は C1, C2, 左の縁の d から出す"""
    a, h, _ = lane_delta_d(road_w=4.0)
    return {ln: float(np.hypot(a + y, h)) for ln, y in LANE_OFFSET.items()}


def fit_fixed_d(r_obs, t, d):
    """d を固定して (v, t_min) を最小二乗で決める（analyze_1001.profile_d と同じ探索）。
    戻り値の p は、等速モデルが予測する走路上の位置（垂線の足から C5 向きが正）"""
    sign = -1.0 if r_obs[-1] < r_obs[0] else +1.0              # 接近なら t < t_min
    best = None
    for v in np.linspace(0.5, 6.0, 221):
        q = np.sqrt(np.maximum(r_obs**2 - d**2, 0)) / v
        t_min = np.mean(t - sign * q)
        p = sign * v * (t - t_min)                              # 接近・離反とも奥（C5 側）が正
        r_th = np.hypot(p, d)
        rms = np.sqrt(np.mean((r_obs - r_th)**2))
        if best is None or rms < best[2]:
            best = (v, t_min, rms, p, r_th)
    return best                                                # v, t_min, rms, (T,) p, (T,) R_theory


def main():
    dl = lane_d()
    print("各レーンの最接近距離 d（道幅 4 m 実測・1 m 間隔）: "
          + ", ".join(f"レーン{ln} {dl[ln]:.2f} m" for ln in (1, 2, 3)))

    runs = []
    for n in range(1, N_MAIN + 1):
        mode, lane, dr = condition(n)
        g = pick_ridge(TAGS[n - 1])
        if g is None or len(g["f"]) < 10:
            continue
        own = fit_fixed_d(g["rng"], g["t"], dl[lane])
        others = [fit_fixed_d(g["rng"], g["t"], dl[o])[2] for o in (1, 2, 3) if o != lane]
        runs.append(dict(n=n, mode=mode, lane=lane, dr=dr, r=g["rng"], fit=own, rms_other=min(others)))

    # 表: 申告レーンでの残差と、他レーンで当てはめたときとの差
    print(f"\n{'No.':>4} {'手段':>5} {'レーン':>4} {'向き':>4} {'点数':>4} {'v[m/s]':>7} "
          f"{'平均ずれ[m]':>10} {'rms[m]':>7} {'他レーン最良rms':>14} {'見えた範囲 p[m]':>16}")
    for u in runs:
        v, _, rms, p, r_th = u["fit"]
        bias = np.mean(u["r"] - r_th)
        print(f"{u['n']:4d} {u['mode']:>5} {u['lane']:4d} {u['dr']:>4} {len(u['r']):4d} {v:7.2f} "
              f"{bias:+10.3f} {rms:7.3f} {u['rms_other']:14.3f} {p.min():7.1f}–{p.max():5.1f}")
    rms_all = np.array([u["fit"][2] for u in runs])
    gap = np.array([u["rms_other"] - u["fit"][2] for u in runs])
    print(f"\n  対象 {len(runs)} 本。申告レーンでの rms: 中央値 {np.median(rms_all):.3f} m"
          f"（距離分解能 0.846 m）")
    print(f"  他レーンの d にしたときの rms の増分: 中央値 {np.median(gap):+.4f} m、"
          f"最大 {gap.max():+.4f} m -> レーン間で当てはまりの差はほぼ無い")

    # 図: 上段 = 観測点と3レーンの理論曲線、下段 = 申告レーンに対する残差。列は (手段, レーン)
    cols = [(m, ln) for m in ("walk", "bike") for ln in (1, 2, 3)]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    pp = np.linspace(-S0, 45, 300)                               # C1 の位置（-S0）から奥まで
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for j, (m, ln) in enumerate(cols):
        ax, axr = axes[0, j], axes[1, j]
        for o in (1, 2, 3):
            ax.plot(pp, np.hypot(pp, dl[o]), color=colors[o], lw=1.6 if o == ln else 0.7,
                    ls="-" if o == ln else ":", label=f"lane{o} truth" if j == 0 else None)
        sel = [u for u in runs if u["mode"] == m and u["lane"] == ln]
        for u in sel:
            _, _, _, p, r_th = u["fit"]
            mk = "o" if u["dr"] == "app" else "^"
            ax.plot(p, u["r"], mk, ms=2.5, color="k", alpha=0.5)
            axr.plot(p, u["r"] - r_th, mk, ms=2.5, color="k", alpha=0.5)
        ax.axvline(-S0, color="tab:red", lw=0.6, ls="--")
        ax.axvline(0, color="tab:red", lw=0.6)
        ax.set_title(f"{m} lane{ln}  (n={len(sel)})", fontsize=9)
        ax.set_ylim(18, 52)
        ax.grid(alpha=0.3)
        axr.axhspan(-0.423, 0.423, color="gray", alpha=0.15)    # ±半ビン（距離分解能 0.846 m）
        axr.axhline(0, color=colors[ln], lw=1)
        axr.set_ylim(-2, 2)
        axr.set_xlabel("position along road p [m]\n(0 = closest approach)")
        axr.grid(alpha=0.3)
    axes[0, 0].set_ylabel("range R [m]")
    axes[1, 0].set_ylabel("R obs - R truth [m]")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Observed tracks vs lane truth (d fixed to declared lane; v, t_min fitted). "
                 "o: approach, ^: depart. gray band: +-half range bin", fontsize=10)
    fig.tight_layout()
    out = ROOT / "figures/night_lane_truth.png"
    fig.savefig(out, dpi=120)
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
