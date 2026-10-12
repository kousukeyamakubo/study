# 見逃しの (a)「真値から BEV 2 m 以内に CFAR の検出がない」が、本当に見えていないのかを確かめる（README「見逃しの内訳」の続き）。
#
# (a) は「CFAR がその場所で反応しなかった」ことしか言っていない。見えていないと言うには、ほかの理由を消す必要がある:
#   1. 真値のずれ: 真値は等速の仮定で当てはめたもので、人が漕ぐ・歩く以上は崩れうる（独立な真値はない。カメラは夜で暗い）。
#      → 真値のまわりの幅 W を 2 → 4 → 6 → 8 m と広げても CFAR の検出がないフレームを数える。
#        広げても何もなければ、真値が W までずれていても説明がつかない
#   2. MTI で消した: MTI はチャープ平均を引くので、径方向の速度が 0 に近い目標も一緒に消す。
#      → MTI あり・なしの電力を同じセル（DC 以外のドップラー）で比べる。MTI なし・背景差分でも 4 の判定をする
#   3. 周りに隠された（CFAR の訓練セルに強い反射が入ってしきい値が上がる）: 目標のいない背景が要るので、ここではやらない
#   4. 背景と区別できない: CFAR を通さず、真値のまわり W の電力（ドップラーの最大）を、同じ距離ビンで
#      目標が遠く（BEV で FAR_M より離れた）にいるフレームの電力と比べ、その分布の何 % 点かを出す。
#      見えていなければ一様（95% 点を超えるのは 5% 程度）になるはず。
#      検定が効くかを見るため、同じことを対応づいた（検出できた）フレームにも掛ける
#
# 使い方:
#   python miss_check.py > miss_check_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cfar_3gpp as c3                                           # noqa: E402
from cfar_3gpp import cc                                         # noqa: E402

WIDTHS = [2.0, 4.0, 6.0, 8.0]           # 真値のまわりの幅（BEV の走路方向）[m]
FAR_M = 12.0                            # これより目標が離れたフレームを、その距離ビンの背景とみなす [m]
MIN_BG = 20                             # 背景のフレームがこれより少ない走行は 4 から外す
PCT_HI = 95.0


def bin_x(run):
    """各距離ビンの BEV 位置 (R,)"""
    return np.sqrt(np.maximum(run["rng"] ** 2 - run["d"] ** 2, 0.0))


def frame_classes(run, rmax, alpha):
    """目標が区間にいるフレーム (F,)、そのうち対応づいたもの (F,)、(a) (F,)、真値のまわりの検出の有無 {W: (F,)}"""
    _, matched = c3.score_3gpp(rmax, run, [alpha], c3.GAP, return_matched=True)
    matched = matched[0]                                           # (F,)
    tgt = run["on"] & (run["r"] >= c3.ZONE[0]) & (run["r"] <= c3.ZONE[1])
    dx = np.abs(bin_x(run)[None, :] - np.abs(run["s"])[:, None])   # (F, R) 真値からの BEV の離れ
    det = rmax > alpha                                             # (F, R)
    near = {w: (det & (dx <= w)).any(axis=1) for w in WIDTHS}
    a = tgt & ~matched & ~near[2.0]
    return tgt, matched & tgt, a, near, dx


def peak_db(p, use):
    """ドップラー（検出に使うもの）の最大の電力 [dB]。(F, R)"""
    return 10 * np.log10(p[:, use, :].max(axis=1) + 1e-30)


def percentiles(pdb, dx, on, frames, w):
    """frames の各フレームで、真値のまわり w の電力の床超過の最大が、背景フレームの同じ量の分布の何 % 点か。

    pdb: (F, R) 電力 [dB]、dx: (F, R) 真値からの BEV の離れ。背景は「その距離ビンで目標が FAR_M より遠い」フレーム。
    床は距離ビンごとの背景フレームの中央値（目標のいるフレームを床に入れないため）
    """
    bg = on[:, None] & (dx > FAR_M)                                # (F, R)
    floor = np.array([np.median(pdb[bg[:, k], k]) if bg[:, k].sum() >= MIN_BG else np.nan
                      for k in range(pdb.shape[1])])               # (R,)
    ex = pdb - floor[None, :]                                      # (F, R)
    out = []
    for f in np.flatnonzero(frames):
        cells = (dx[f] <= w) & np.isfinite(floor)                  # (R,) このフレームの真値のまわり
        if not cells.any():
            continue
        # 同じ距離ビンの組が、すべて背景になっているフレームを対照にする
        ctrl = bg[:, cells].all(axis=1)
        if ctrl.sum() < MIN_BG:
            continue
        s = ex[f, cells].max()
        s_ctrl = ex[ctrl][:, cells].max(axis=1)
        out.append(100.0 * (s_ctrl < s).mean())
    return np.array(out)


def main():
    a_nom = cc.nominal_alpha_db()
    runs = c3.load_runs()
    print(f"MTI・CA-CFAR 訓練 {cc.N_TRAIN}・ガード {cc.N_GUARD}・隙間 {c3.GAP}・公称 {a_nom:.1f} dB（T1 と同じ）、12 本")

    tot = dict(tgt=0, matched=0, a=0)
    no_det = {w: 0 for w in WIDTHS}
    pct = {m: {"a": {w: [] for w in WIDTHS}, "hit": {w: [] for w in WIDTHS}} for m in cc.METHODS}
    mti_vs_raw = []
    per_run = []
    for run in runs:
        maps, use, rng, vel, t, dc = cc.power_maps(c3.TAGS[run["n"] - 1])
        rmax = c3.collapse(cc.cfar_ratio_db(maps["MTI"]), use["MTI"])     # (F, R)
        tgt, hit, a, near, dx = frame_classes(run, rmax, a_nom)
        tot["tgt"] += tgt.sum()
        tot["matched"] += hit.sum()
        tot["a"] += a.sum()
        for w in WIDTHS:
            no_det[w] += (a & ~near[w]).sum()
        per_run.append((run, a.sum(), [(a & ~near[w]).sum() for w in WIDTHS]))

        # 2. MTI あり・なしの電力を、(a) のフレームの真値のまわり 2 m・DC 以外のドップラーで比べる。
        # MTI ありの DC ビンは cc.power_maps で隣の平均に置き換えてあるので、DC と ±1 は外す（窓の漏れも ±1 に出る）
        # ドップラーの窓（hanning）は MTI の後に掛けるので、MTI なしでは静止物の DC が窓の漏れとして ±2 ビン以上にも出る。
        # 差が「MTI が目標を削った」のか「MTI なしに漏れが足された」のかを分けるため、
        # 同じ距離ビンの背景フレーム（目標が FAR_M より遠い）でも同じ比を取り、ドップラーのビンごとに並べる
        k = np.abs(np.arange(len(vel)) - dc)                       # (D,) DC からのビン数
        cells_a = a[:, None] & (dx <= 2.0)                         # (F, R)
        rows_a = np.flatnonzero(cells_a.any(axis=0))
        cells_bg = run["on"][:, None] & (dx > FAR_M) & np.isin(np.arange(len(rng)), rows_a)[None, :]
        for kk in range(2, int(k.max()) + 1):
            sel = k == kk
            r10 = 10 * np.log10(maps["MTI"][:, sel, :].sum(axis=1) / maps["no MTI"][:, sel, :].sum(axis=1))   # (F, R)
            mti_vs_raw.append((kk, r10[cells_a], r10[cells_bg]))

        # 4. 背景と比べる（3 方式それぞれ）
        for m in cc.METHODS:
            pdb = peak_db(maps[m], use[m])                         # (F, R)
            for w in WIDTHS:
                pct[m]["a"][w].append(percentiles(pdb, dx, run["on"], a, w))
                pct[m]["hit"][w].append(percentiles(pdb, dx, run["on"], hit, w))

    print(f"区間にいるフレーム {tot['tgt']}、対応づいた {tot['matched']}、(a) {tot['a']}")

    print("\n" + "=" * 80)
    print("1. 真値のずれ: (a) のフレームのうち、真値のまわり W に MTI の CFAR の検出（ドップラーを潰した距離ビン）が 1 つもないもの")
    print("=" * 80)
    print("    " + "".join(f"W={w:g} m: {no_det[w]:3d}（{no_det[w] / max(tot['a'], 1):.2f}）  " for w in WIDTHS))
    print("    走行ごと（(a) の数 → W=2/4/6/8 m で検出が 1 つもない数）")
    for run, na, nd in per_run:
        print(f"    No.{run['n']:2d} {run['mode']} lane{run['lane']} {run['dr']}: {na:3d} → " + " / ".join(f"{x:3d}" for x in nd))

    print("\n" + "=" * 80)
    print("2. MTI で消したか: 電力の比 MTI あり / なし [dB]（DC からのビン数ごと）。")
    print("   (a) のフレームの真値のまわり 2 m と、同じ距離ビンの背景フレーム（目標が 12 m より遠い）で比べる")
    print("=" * 80)
    print(f"    {'ビン':>4}{'(a) 中央値':>12}{'5% 点':>8}{'背景 中央値':>13}{'5% 点':>8}{'(a) − 背景':>12}")
    for kk in sorted({z[0] for z in mti_vs_raw}):
        xa = np.concatenate([z[1] for z in mti_vs_raw if z[0] == kk])
        xb = np.concatenate([z[2] for z in mti_vs_raw if z[0] == kk])
        print(f"    {'±' + str(kk):>4}{np.median(xa):+12.2f}{np.percentile(xa, 5):+8.2f}"
              f"{np.median(xb):+13.2f}{np.percentile(xb, 5):+8.2f}{np.median(xa) - np.median(xb):+12.2f}")
    print("    （背景でも同じだけ下がっていれば、差は MTI なしの静止物の漏れで、MTI が目標を削ったのではない）")
    # MTI で目標が消えるのは |v_r| が DC ビンに入るときなので、区間内でどこまで遅くなるかを真値から見る
    vr_min = {}
    for r in runs:
        m = (r["r"] >= c3.ZONE[0]) & (r["r"] <= c3.ZONE[1]) & r["on"]
        if m.any():
            vr_min.setdefault(r["mode"], []).append(float(np.min(r["v"] * np.abs(r["s"][m]) / r["r"][m])))
    for mode, xs in vr_min.items():
        print(f"    {mode}: 区間内の |v_r| の最小 {min(xs):.2f} m/s（走行ごとの最小の範囲 {min(xs):.2f}〜{max(xs):.2f}）。"
              f"ドップラーのビン幅 {vel[1] - vel[0]:.2f} m/s")

    print("\n" + "=" * 80)
    print(f"4. 背景と区別できるか: 真値のまわり W の電力（ドップラーの最大）の床超過の最大が、")
    print(f"   同じ距離ビンで目標が {FAR_M:g} m より遠いフレームの分布の何 % 点か。{PCT_HI:g}% 点を超えたフレームの割合と、% 点の中央値")
    print(f"   見えていなければ、超える割合は約 {100 - PCT_HI:g}%、中央値は約 50 になる")
    print("=" * 80)
    print(f"    {'':10}{'':6}" + "".join(f"{f'W={w:g} m':>22}" for w in WIDTHS))
    for m in cc.METHODS:
        for key, lab in (("a", "(a)"), ("hit", "対応づいた")):
            cells = []
            for w in WIDTHS:
                v = np.concatenate(pct[m][key][w])
                cells.append(f"{(v > PCT_HI).mean():.2f} / {np.median(v):4.0f}（{len(v)}）")
            print(f"    {m:10}{lab:6}" + "".join(f"{c:>22}" for c in cells))
    print("    値は「95% 点を超えた割合 / % 点の中央値（フレーム数）」。背景のフレームが足りないフレームは外した")

    # 図: (a) と対応づいたフレームの % 点の分布（W=2 m と W=6 m、3 方式）
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    bins = np.linspace(0, 100, 21)
    for ax, m in zip(axes, cc.METHODS):
        for key, col, lab in (("a", "tab:blue", "(a) missed, no CFAR detection within 2 m"),
                              ("hit", "tab:orange", "matched (detected)")):
            for w, ls in ((2.0, "-"), (6.0, "--")):
                v = np.concatenate(pct[m][key][w])
                h, _ = np.histogram(v, bins)
                ax.stairs(h / max(len(v), 1), bins, color=col, ls=ls,
                          label=f"{lab}, W={w:g} m" if m == cc.METHODS[0] else None)
        ax.axhline(1 / (len(bins) - 1), color="0.4", lw=0.8, ls=":")
        ax.set_xlabel(f"percentile of power near truth among background frames")
        ax.set_title(m, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of frames")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Is the target distinguishable from background near the fitted path? night 10/1, 12 runs. "
                 f"Dotted: uniform (= indistinguishable)", fontsize=10)
    o = ROOT / "figures" / "miss_check_percentile.png"
    fig.savefig(o, dpi=110)
    print(f"-> {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
