# 仮説 2「ML なしで、手法の組み合わせ（MTI・距離ごとの正規化・CFAR・軌跡のつながり）で検出できる」
# のうち、前処理 × CFAR の部分を 10/1 夜のデータで比べる（meeting/2026-10-13.md「ミーティングの着地」）。
#
# 前処理（どれも RD の各セルの電力 (F, D, R) にしてから同じ CA-CFAR に掛ける）:
#   MTI       : 現行（チャープ平均を引く）。DC ビンは消え残りが出るので検出から外す（processing.md §3）
#   no MTI    : MTI なしの RD をそのまま。静止物は DC ビンで CFAR に掛かる
#   bg-sub    : MTI なし → 共通位相を戻して複素の時間中央値を引く（background_recheck.py の最良の方式）。
#               夜は無人背景がないので背景はファイル自身の時間中央値。DC ビンも検出に使う
# CFAR は weekly/0908/postprocess/detect.py の cfar_ca（CA-CFAR、窓は同ファイルの初期値）をそのまま使う。
#
# 評価:
#   CFAR の公称 Pfa は、雑音が指数分布で一様という仮定の値で、静止物の残差がある実測では成り立たない。
#   そこで、しきい値係数（訓練セル平均の何 dB 上か）を振り、実測の誤警報率と検出率の組で比べる。
#   - 真の軌跡: 申告レーンの d に固定した等速軌跡 R(t) = sqrt(v^2 (t - t0)^2 + d^2) の (v, t0) を、
#     3 方式の床超過の平均に lane_visibility.matched_score と同じ総当たりで当てはめる（R 30〜49 m）。
#     1 方式だけで当てはめると、その方式に有利な軌跡になるため。近い帯は当てはめた軌跡の外挿
#   - 検出: 真の軌跡のセル（距離 ±1 ビン、ドップラーは期待の径方向速度 dR/dt の ±1 ビン、折り返し込み）に
#     CFAR の検出が 1 つでもあれば、そのフレームは検出。帯ごとに検出したフレームの割合を出す
#   - 誤警報: 走路の距離帯 R 20〜49 m で、真の軌跡から距離 ±3 ビンより外のセルの検出の割合。
#     静止物（DC ビン）と動体ビンを分けて数える。目標は 1 人なので、それ以外の検出は誤警報とみなす
#   - 対象はレーン 1, 2 の 24 本。レーン 3 は直接波がほぼ見えず、軌跡の当てはめが成り立たない（1013/1001/README.md A-c）
#
# 使い方:
#   python cfar_compare.py

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
W1001 = ROOT.parent.parent / "1001"
sys.path.insert(0, str(W1001.parent / "0727"))
sys.path.insert(0, str(W1001.parent / "0908" / "postprocess"))
sys.path.insert(0, str(W1001))
sys.path.insert(0, str(ROOT))
from analyze_night import N_MAIN, TAGS, condition, npz_path      # noqa: E402
from background_recheck import common_phase, complex_median      # noqa: E402
from detect import _box_mean                                     # noqa: E402
from lane_visibility import T0_GRID, V_GRID                      # noqa: E402
from plot_lane_truth import lane_range                           # noqa: E402

METHODS = ["MTI", "no MTI", "bg-sub"]
LANES = (1, 2)
BANDS = {"near 21-30 m": (21.0, 30.0), "mid 30-40 m": (30.0, 40.0), "far 40-49 m": (40.0, 49.0)}
FIT_R = (30.0, 49.0)          # 真の軌跡を当てはめる帯（どの方式でも目標が見えている帯。README B）
ROAD_R = (20.0, 49.0)         # 誤警報を数える帯
N_TRAIN, N_GUARD = (2, 5), (1, 2)   # detect.py の初期値（ドップラー, 距離の片側セル数）
ALPHA_DB = np.arange(4.0, 26.01, 0.5)   # しきい値係数（訓練セル平均からの dB）
NOMINAL_PFA = 1e-3
FA_REF = 1e-3                 # 検出率を読む誤警報率
GATE_R, GATE_D = 1, 1         # 軌跡のセルの許容（距離・ドップラーのビン）
EXCL_R = 3                    # 誤警報から外す、軌跡のまわりの距離ビン
# 走行速度の探索範囲 [m/s]。lane_visibility の 0.8〜5 m/s のままだと、徒歩でも静止物の残差の帯に
# 合わせた 5 m/s 近い軌跡が選ばれる（対照と差がない）ので、手段ごとに絞る
V_RANGE = {"walk": (0.8, 2.2), "bike": (1.5, 4.5)}
FIT_MIN_DB = 1.0              # 当てはめが時刻反転の対照をこれだけ上回る走行だけを評価に使う


def power_maps(tag):
    """3 方式の RD 電力 (F, D, R)（8ch の電力和）と、検出に使うドップラービンのマスク (D,)、軸"""
    d = np.load(npz_path(tag))
    rd_mti = d["rd"]                                               # (F, TX, RX, D, R)
    rng, vel, t = d["range_m"], d["vel_ms"], d["t_rel_s"]
    rd = np.load(npz_path(tag, nomti=True))["rd"]                  # (F, TX, RX, D, R)
    dc = int(np.abs(vel).argmin())
    phi, _ = common_phase(rd, rng, dc)                             # (F,)
    rd_c = rd * np.exp(-1j * phi)[:, None, None, None, None]

    def pw(x):
        return np.sum(np.abs(x) ** 2, axis=(1, 2))                 # (F, D, R)

    p_mti = pw(rd_mti)
    # MTI の DC ビンは検出に使わない。ただし消え残りが訓練セルに入ると隣の ±1 ビンのしきい値が上がるので、
    # 隣 2 ビンの平均で埋めておく（行ごと消すと ±1 ビンが隣り合ってしまう）
    p_mti[:, dc, :] = 0.5 * (p_mti[:, dc - 1, :] + p_mti[:, dc + 1, :])
    use_mti = np.ones(len(vel), bool)
    use_mti[dc] = False
    maps = {"MTI": p_mti, "no MTI": pw(rd), "bg-sub": pw(rd_c - complex_median(rd_c))}
    use = {"MTI": use_mti, "no MTI": np.ones(len(vel), bool), "bg-sub": np.ones(len(vel), bool)}
    return maps, use, rng, vel, t, dc


def cfar_ratio_db(p):
    """CA-CFAR の「セルの電力 / 訓練セル平均」[dB]。(F, D, R)。しきい値を振るので比のまま持つ"""
    td, tr = N_TRAIN
    gd, gr = N_GUARD
    full = _box_mean(p, td + gd, tr + gr)
    guard = _box_mean(p, gd, gr)
    n_full = (2 * (td + gd) + 1) * (2 * (tr + gr) + 1)
    n_guard = (2 * gd + 1) * (2 * gr + 1)
    train = (full * n_full - guard * n_guard) / (n_full - n_guard)
    return 10 * np.log10(p / np.maximum(train, 1e-30))


def nominal_alpha_db():
    """detect.py の cfar_ca と同じ公称しきい値係数（Pfa = NOMINAL_PFA）[dB]"""
    td, tr = N_TRAIN
    gd, gr = N_GUARD
    n = (2 * (td + gd) + 1) * (2 * (tr + gr) + 1) - (2 * gd + 1) * (2 * gr + 1)
    return 10 * np.log10(n * (NOMINAL_PFA ** (-1.0 / n) - 1.0))


def excess_rt(p, use):
    """距離−時間の床超過 dB (F, R)。検出に使うドップラーの最大 → 距離ごとの時間中央値を床として引く"""
    db = 10 * np.log10(p[:, use, :].max(axis=1) + 1e-30)
    return db - np.median(db, axis=0, keepdims=True)


def fit_truth(ex, rng, t, d, sign, v_grid):
    """lane_visibility.matched_score と同じ総当たりで、FIT_R 内の軌跡上の床超過の平均が最大の (v, t0) を返す"""
    dr = rng[1] - rng[0]
    exd = np.max(np.stack([np.roll(ex, k, axis=1) for k in (-1, 0, 1)]), axis=0)   # (F, R)
    best = (-np.inf, None, None)
    for v in v_grid:
        s = v * (t[None, :] - T0_GRID[:, None])                   # (T0, F)
        r = np.hypot(s, d)
        inb = ((s * sign) >= 0) & (r >= FIT_R[0]) & (r <= FIT_R[1])
        idx = np.clip(np.round((r - rng[0]) / dr).astype(int), 0, len(rng) - 1)
        val = exd[np.arange(len(t))[None, :], idx]
        cnt = inb.sum(axis=1)
        sc = np.where(cnt >= 5, (val * inb).sum(axis=1) / np.maximum(cnt, 1), -np.inf)
        k = int(sc.argmax())
        if sc[k] > best[0]:
            best = (float(sc[k]), float(v), float(T0_GRID[k]))
    return best


def truth_path(v, t0, d, sign, t, rng, vel):
    """真の軌跡のフレームごとの距離ビン・ドップラービン・R・走路上か。(F,) ×4"""
    s = v * (t - t0)                                               # (F,)
    r = np.hypot(s, d)
    on = (s * sign) >= 0                                           # 申告の向きの側だけ（反対側は通っていない）
    vr = v * s / r                                                 # dR/dt。ridge の slope と vel が同符号なのと同じ向き
    dv = vel[1] - vel[0]
    # 速度軸は −4.8〜+4.2 m/s の 16 ビン。範囲外は折り返す
    kd = np.round((vr - vel[0]) / dv).astype(int) % len(vel)
    kr = np.round((r - rng[0]) / (rng[1] - rng[0])).astype(int)
    return kr, kd, r, on


def evaluate(ratio, use, kr, kd, r, on, rng, dc):
    """しきい値ごとの、帯別の検出率の分子・分母と、誤警報（静止 / 動体）の分子・分母"""
    F, D, R = ratio.shape
    nb = len(BANDS)
    hit = np.zeros((len(ALPHA_DB), nb))
    tot = np.zeros(nb)
    fa = np.zeros((len(ALPHA_DB), 2))                              # [:, 0] DC ビン, [:, 1] 動体ビン
    fa_n = np.zeros(2)
    # 軌跡のセルの比の最大（フレームごと）。これがしきい値を超えれば検出
    dd = np.arange(-GATE_D, GATE_D + 1)
    rr = np.arange(-GATE_R, GATE_R + 1)
    best = np.full(F, -np.inf)
    for f in range(F):
        if not on[f]:
            continue
        ks = (kd[f] + dd) % D
        ks = ks[use[ks]]
        ri = np.clip(kr[f] + rr, 0, R - 1)
        if len(ks):
            best[f] = ratio[f][np.ix_(ks, ri)].max()
    for j, (b0, b1) in enumerate(BANDS.values()):
        m = on & (r >= b0) & (r <= b1)
        tot[j] = m.sum()
        hit[:, j] = (best[m][None, :] > ALPHA_DB[:, None]).sum(axis=1)
    # 誤警報: 走路の帯で、軌跡（走路上にいるフレームのみ）から EXCL_R ビンより外
    road = (rng >= ROAD_R[0]) & (rng <= ROAD_R[1])                # (R,)
    mask = np.broadcast_to(road[None, :], (F, R)).copy()
    for f in np.where(on)[0]:
        mask[f, max(kr[f] - EXCL_R, 0):kr[f] + EXCL_R + 1] = False
    for c, dsel in enumerate([np.arange(D) == dc, np.arange(D) != dc]):
        dsel = dsel & use
        if not dsel.any():
            continue
        x = ratio[:, dsel, :][np.broadcast_to(mask[:, None, :], (F, dsel.sum(), R))]
        fa_n[c] = x.size
        fa[:, c] = (x[None, :] > ALPHA_DB[:, None]).sum(axis=1)
    return hit, tot, fa, fa_n


def interp_pd_at_fa(fa_rate, pd, ref):
    """誤警報率 ref での検出率（しきい値を振った曲線を、誤警報率の対数で線形補間）"""
    x = np.log10(np.maximum(fa_rate, 1e-7))[::-1]                  # しきい値が上がるほど誤警報は減る → 逆順で単調増加
    y = pd[::-1]
    lr = np.log10(ref)
    if lr < x[0] or lr > x[-1]:
        return np.nan
    return float(np.interp(lr, x, y))


def main():
    runs = [n for n in range(1, N_MAIN + 1) if condition(n)[1] in LANES]
    a_nom = nominal_alpha_db()
    acc = {m: dict(hit=0, tot=0, fa=0, fa_n=0, hit_c=0) for m in METHODS}
    per_run = {m: [] for m in METHODS}
    example = None
    used = []
    print(f"CA-CFAR 窓: 訓練 {N_TRAIN}・ガード {N_GUARD}（ドップラー, 距離）、公称 Pfa {NOMINAL_PFA:g} の係数 {a_nom:.1f} dB")
    print("\n(1) 真の軌跡の当てはめ（3 方式の床超過の平均、R 30〜49 m）")
    for n in runs:
        mode, lane, dr = condition(n)
        tag = TAGS[n - 1]
        maps, use, rng, vel, t, dc = power_maps(tag)
        d0 = lane_range(lane)[0]
        sign = -1.0 if dr == "app" else +1.0
        ex = np.mean([excess_rt(maps[m], use[m]) for m in METHODS], axis=0)   # (F, R)
        vg = V_GRID[(V_GRID >= V_RANGE[mode][0]) & (V_GRID <= V_RANGE[mode][1])]
        sc, v, t0 = fit_truth(ex, rng, t, d0, sign, vg)
        ctrl = fit_truth(ex[::-1], rng, t, d0, sign, vg)[0]
        ok = sc - ctrl >= FIT_MIN_DB
        print(f"    No.{n:2d} {mode} lane{lane} {dr}: v {v:.2f} m/s, t0 {t0:6.1f} s, "
              f"軌跡上 {sc:.1f} dB（時刻反転の対照 {ctrl:.1f} dB）" + ("" if ok else "  → 除外"))
        if not ok:
            continue
        used.append(n)
        kr, kd, r, on = truth_path(v, t0, d0, sign, t, rng, vel)
        for m in METHODS:
            ratio = cfar_ratio_db(maps[m])                         # (F, D, R)
            hit, tot, fa, fa_n = evaluate(ratio, use[m], kr, kd, r, on, rng, dc)
            # 偶然の当たりの水準: 同じセルの並びを時刻反転して当てる（目標の筋とは向きが合わない）。
            # 静止物の残差の帯（R 21〜30 m の木など）に軌跡が重なると、目標がいなくても当たるため
            hit_c = evaluate(ratio, use[m], kr[::-1], kd[::-1], r[::-1], on[::-1], rng, dc)[0]
            acc[m]["hit_c"] = acc[m]["hit_c"] + hit_c
            for k, x in dict(hit=hit, tot=tot, fa=fa, fa_n=fa_n).items():
                acc[m][k] = acc[m][k] + x
            per_run[m].append((n, hit, tot, fa, fa_n))
            if n == 19 and m == METHODS[0]:
                example = dict(n=n, t=t, rng=rng, kr=kr, on=on, dc=dc, ratios={})
            if example is not None and example["n"] == n:
                example["ratios"][m] = (ratio, use[m])

    # ---- 集計 ----
    res = {}
    for m in METHODS:
        a = acc[m]
        pd = a["hit"] / np.maximum(a["tot"], 1)                    # (A, B)
        fa_all = a["fa"].sum(axis=1) / a["fa_n"].sum()             # (A,)
        fa_dc = a["fa"][:, 0] / max(a["fa_n"][0], 1)
        fa_mv = a["fa"][:, 1] / max(a["fa_n"][1], 1)
        pd_c = a["hit_c"] / np.maximum(a["tot"], 1)                # (A, B) 時刻反転の対照
        res[m] = dict(pd=pd, pd_c=pd_c, fa=fa_all, fa_dc=fa_dc, fa_mv=fa_mv, tot=a["tot"])

    print(f"    評価に使う走行 {len(used)} 本: " + ", ".join(f"No.{n}" for n in used))
    runs = used
    i_nom = int(np.abs(ALPHA_DB - a_nom).argmin())
    print(f"\n(2) 公称しきい値（{ALPHA_DB[i_nom]:.1f} dB）での実測（{len(runs)} 本の合計）")
    print(f"{'':8}{'誤警報率 全体':>14}{'DC ビン':>10}{'動体ビン':>10}" + "".join(f"{b:>15}" for b in BANDS))
    for m in METHODS:
        x = res[m]
        print(f"{m:8}{x['fa'][i_nom]:14.2e}{x['fa_dc'][i_nom]:10.2e}{x['fa_mv'][i_nom]:10.2e}"
              + "".join(f"{x['pd'][i_nom, j]:15.2f}" for j in range(len(BANDS))))
    print(f"    帯ごとの軌跡上のフレーム数: " + ", ".join(
        f"{b} {int(res[METHODS[0]]['tot'][j])}" for j, b in enumerate(BANDS)))

    print(f"\n(3) 誤警報率 {FA_REF:g} に揃えたときの検出率（しきい値を振った曲線から補間）")
    print(f"{'':8}{'しきい値[dB]':>12}" + "".join(f"{b:>15}" for b in BANDS))
    for m in METHODS:
        x = res[m]
        th = interp_pd_at_fa(x["fa"], ALPHA_DB, FA_REF)
        print(f"{m:8}{th:12.1f}" + "".join(
            f"{interp_pd_at_fa(x['fa'], x['pd'][:, j], FA_REF):15.2f}" for j in range(len(BANDS))))
    print("    同じ、ただし誤警報を動体ビンだけで数えたとき（静止物を別の手段で捨てられる場合）")
    for m in METHODS:
        x = res[m]
        th = interp_pd_at_fa(x["fa_mv"], ALPHA_DB, FA_REF)
        print(f"{m:8}{th:12.1f}" + "".join(
            f"{interp_pd_at_fa(x['fa_mv'], x['pd'][:, j], FA_REF):15.2f}" for j in range(len(BANDS))))
    for key, lab in (("fa", "全体"), ("fa_mv", "動体ビンだけ")):
        print(f"    偶然の当たり（時刻反転した軌跡での検出率）、誤警報率を{lab}で揃えたとき")
        for m in METHODS:
            x = res[m]
            print(f"{m:8}{'':12}" + "".join(
                f"{interp_pd_at_fa(x[key], x['pd_c'][:, j], FA_REF):15.2f}" for j in range(len(BANDS))))

    print(f"\n(4) 走行ごとの検出率（誤警報率 {FA_REF:g} のしきい値、近 / 中 / 遠。分母 0 は -）")
    th_ref = {m: interp_pd_at_fa(res[m]["fa"], ALPHA_DB, FA_REF) for m in METHODS}
    for k, n in enumerate(runs):
        mode, lane, dr = condition(n)
        cells = []
        for m in METHODS:
            _, hit, tot, _, _ = per_run[m][k]
            i = int(np.abs(ALPHA_DB - th_ref[m]).argmin()) if np.isfinite(th_ref[m]) else i_nom
            cells.append(f"{m} " + "/".join("-" if tot[j] == 0 else f"{hit[i, j] / tot[j]:.2f}"
                                            for j in range(len(BANDS))))
        print(f"    No.{n:2d} {mode} lane{lane} {dr}: " + " | ".join(cells))

    # ---- 図 1: 帯ごとの検出率 − 誤警報率 ----
    fig, axes = plt.subplots(1, len(BANDS), figsize=(13, 4.2), sharey=True, layout="constrained")
    for j, (ax, b) in enumerate(zip(axes, BANDS)):
        for c, m in enumerate(METHODS):
            x = res[m]
            ax.plot(x["fa"], x["pd"][:, j], "-", color=f"C{c}", label=f"{m} (all bins)")
            ax.plot(x["fa_mv"], x["pd"][:, j], ":", color=f"C{c}", label=f"{m} (FA moving bins only)")
            ax.plot(x["fa"][i_nom], x["pd"][i_nom, j], "o", color=f"C{c}", ms=5)
        ax.axvline(FA_REF, color="k", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false alarm rate per cell (R 20-49 m, off-path)")
        ax.set_title(f"{b} ({int(res[METHODS[0]]['tot'][j])} on-path frames)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("detection rate on true path")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"CA-CFAR after each preprocessing, night 10/1 lanes 1-2 ({len(runs)} runs). "
                 f"Dot: nominal Pfa {NOMINAL_PFA:g} threshold ({a_nom:.1f} dB)", fontsize=10)
    out = ROOT / "figures" / "cfar_compare_roc.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\n-> {out.relative_to(ROOT)}")

    # ---- 図 2: No.19 の距離−時間の検出（誤警報率 FA_REF のしきい値） ----
    ex = example
    keep = ex["rng"] <= 60
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13, 5), sharey=True, layout="constrained")
    for ax, m in zip(axes, METHODS):
        ratio, use = ex["ratios"][m]
        th = th_ref[m] if np.isfinite(th_ref[m]) else ALPHA_DB[i_nom]
        det = ratio > th                                           # (F, D, R)
        dc = ex["dc"]
        mv = np.arange(det.shape[1]) != dc
        # 0: なし、1: 動体ビンで検出、2: DC ビンのみで検出（静止物）
        img = np.where(det[:, mv & use, :].any(axis=1), 1.0,
                       np.where(det[:, dc, :] & use[dc], 2.0, 0.0))   # (F, R)
        ax.pcolormesh(ex["rng"][keep], ex["t"], img[:, keep], shading="auto",
                      cmap=matplotlib.colors.ListedColormap(["white", "black", "tab:orange"]), vmin=-0.5, vmax=2.5)
        on = ex["on"]
        ax.plot(ex["rng"][np.clip(ex["kr"], 0, len(ex["rng"]) - 1)][on], ex["t"][on], "r-", lw=0.8, alpha=0.7)
        ax.set_title(f"{m}: threshold {th:.1f} dB (FA {FA_REF:g})", fontsize=10)
        ax.set_xlabel("range R [m]")
    axes[0].set_ylabel("time [s]")
    mode, lane, dr = condition(ex["n"])
    fig.suptitle(f"CFAR detections on night No.{ex['n']} ({mode}, lane {lane}, {dr}). "
                 "Black: moving bins, orange: DC bin only, red: fitted true path", fontsize=10)
    out = ROOT / "figures" / "cfar_compare_No19.png"
    fig.savefig(out, dpi=110)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
