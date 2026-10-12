# 10/1 夜のデータを TS 22.137 / TR 38.765 の数え方（1 フレーム＝1 事象、物体単位、BEV で対応づけ）で測り直す（README の T1・T2）。
#
# 真値の当てはめ・RD の電力・CA-CFAR の比は ../1001/cfar_compare.py の関数をそのまま使う（12 本の選び方も同じ）。
# 前処理は MTI だけ（10/1 の比較で最も良かった。../1001/README.md C）。
#
# 数え方（README T1 の表）:
#   - 物体: 各フレームで CFAR の検出をドップラー方向に潰し（距離ビンごとに「どれかのドップラーで検出」）、
#     距離で隣り合う（GAP ビンまでの隙間は許す）ビンを 1 つの物体にまとめる。位置は比が最大の距離ビン。
#     ドップラーを潰すのは、チャープ数 N を減らしても同じ数え方にするため（N=1 ではドップラー軸がない）
#   - BEV: 目標のレーンにいると仮定し、X = sqrt(R^2 - d^2)（d はレーンまでの横の離れ）。真値も検出も同じ d で写す
#   - 対応づけ: BEV で 2 m 以内の最も近い物体を真値と 1 対 1 で対応づける（目標は 1 人なので最も近い 1 つ）
#   - 見逃し: 目標が評価区間 R 30〜49 m にいるフレームのうち、対応づく物体がないフレームの割合
#   - 誤警報 Type 2: 目標が区間にいて物体があるフレームで「対応づかない物体の数 / 物体の数」を平均（TR 38.765 §4.1）。
#     物体は位置が区間内のもの。ただし区間の端の外で対応づいた物体は、真値の検出なので分母に入れる
#   - 誤警報 Type 1（近似）: 目標が区間の端から 3 m 以上外か走路にいないフレームで、区間に物体が 1 つでも出た割合。
#     目標は視野のどこかにいるので本物の Type 1（無人）ではない。本物は T5 で無人背景から出す
#
# 使い方:
#   python cfar_3gpp.py > run_log.txt

import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "1001"))
import cfar_compare as cc                                        # noqa: E402
from analyze_night import N_MAIN, TAGS, condition                # noqa: E402
from detect import _box_mean                                     # noqa: E402
from plot_lane_truth import lane_range                           # noqa: E402

ZONE = (30.0, 49.0)                     # 評価区間（見える帯。近い帯はビームが当たっていない）
BANDS = {"mid 30-40 m": (30.0, 40.0), "far 40-49 m": (40.0, 49.0)}
GATE_BEV = 2.0                          # 対応づけの BEV 距離 [m]（TS 22.137 カテゴリ 2 の水平精度）
T1_MARGIN = 3.0                         # Type 1 に使うフレーム: 目標が区間の端からこれだけ外
ALPHA_DB = cc.ALPHA_DB                  # 4〜26 dB、0.5 dB 刻み
GAP = 0                                 # T1 の物体のまとめ方（隣り合うビンだけ）
TARGET = 0.05                           # 見逃し・誤警報の目標

# T2 で振るもの（ドップラー, 距離の片側セル数）
TRAINS = [(2, 5), (1, 3), (4, 8)]
GUARDS = [(1, 2), (1, 1)]
GAPS = [0, 1]


# ------------------------------------------------------------------
# 真値と電力の読み込み
# ------------------------------------------------------------------

def load_runs():
    """cfar_compare.main と同じ手順で真値を当てはめ、FIT_MIN_DB を満たす走行だけを返す"""
    runs = []
    for n in [n for n in range(1, N_MAIN + 1) if condition(n)[1] in cc.LANES]:
        mode, lane, dr = condition(n)
        maps, use, rng, vel, t, dc = cc.power_maps(TAGS[n - 1])
        d0 = lane_range(lane)[0]
        sign = -1.0 if dr == "app" else +1.0
        ex = np.mean([cc.excess_rt(maps[m], use[m]) for m in cc.METHODS], axis=0)   # (F, R)
        vg = cc.V_GRID[(cc.V_GRID >= cc.V_RANGE[mode][0]) & (cc.V_GRID <= cc.V_RANGE[mode][1])]
        sc, v, t0 = cc.fit_truth(ex, rng, t, d0, sign, vg)
        ctrl = cc.fit_truth(ex[::-1], rng, t, d0, sign, vg)[0]
        if sc - ctrl < cc.FIT_MIN_DB:
            continue
        s = v * (t - t0)                                           # (F,) 走路上の位置（垂線の足から）
        runs.append(dict(n=n, mode=mode, lane=lane, dr=dr, v=v, t0=t0, d=d0, sign=sign,
                         p=maps["MTI"], use=use["MTI"], rng=rng, vel=vel, t=t, dc=dc,
                         s=s, r=np.hypot(s, d0), on=(s * sign) >= 0))
    return runs


def cfar_ratio(p, n_train, n_guard):
    """cc.cfar_ratio_db と同じ計算を、窓を引数にしたもの [dB]。(F, D, R)"""
    td, tr = n_train
    gd, gr = n_guard
    full = _box_mean(p, td + gd, tr + gr)
    guard = _box_mean(p, gd, gr)
    n_full = (2 * (td + gd) + 1) * (2 * (tr + gr) + 1)
    n_guard = (2 * gd + 1) * (2 * gr + 1)
    train = (full * n_full - guard * n_guard) / (n_full - n_guard)
    return 10 * np.log10(p / np.maximum(train, 1e-30))


def nominal_alpha(n_train, n_guard):
    """cc.nominal_alpha_db と同じ（Pfa = cc.NOMINAL_PFA）を、窓を引数にしたもの [dB]"""
    td, tr = n_train
    gd, gr = n_guard
    n = (2 * (td + gd) + 1) * (2 * (tr + gr) + 1) - (2 * gd + 1) * (2 * gr + 1)
    return 10 * np.log10(n * (cc.NOMINAL_PFA ** (-1.0 / n) - 1.0))


# ------------------------------------------------------------------
# 物体へのまとめと 3GPP の数え方
# ------------------------------------------------------------------

def extract_objects(rmax, alphas, gap):
    """しきい値ごと・フレームごとの物体。

    rmax: (F, R) ドップラーを潰した比 [dB]（検出に使うドップラーの最大）
    返り値: row (K,) = しきい値 a とフレーム f の通し番号 a*F + f、col (K,) = 位置の距離ビン（比が最大）
    """
    F, R = rmax.shape
    A = len(alphas)
    det = rmax[None, :, :] > np.asarray(alphas)[:, None, None]     # (A, F, R)
    if gap >= 1:
        # gap ビンまでの隙間を埋める。埋めたビンは比がしきい値未満なので、物体の位置には選ばれない
        filled = det.copy()
        for g in range(1, gap + 1):
            for k in range(1, g + 1):
                # 左に k、右に g+1-k 以内の検出に挟まれたビン
                left = np.zeros_like(det)
                right = np.zeros_like(det)
                left[..., k:] = det[..., :-k]
                right[..., :R - (g + 1 - k)] = det[..., g + 1 - k:]
                filled |= left & right
        det = filled
    # 末尾に False を 1 列足して平らにし、フレームをまたいで物体がつながらないようにする
    x = np.concatenate([det, np.zeros((A, F, 1), bool)], axis=2).reshape(-1)   # (A*F*(R+1),)
    start = x & ~np.r_[False, x[:-1]]
    idx = np.flatnonzero(x)
    seg = np.cumsum(start)[idx] - 1
    row = idx // (R + 1)
    col = idx % (R + 1)
    val = rmax[row % F, col]
    order = np.lexsort((-val, seg))                                # 物体ごとに比の大きい順
    first = order[np.r_[True, seg[order][1:] != seg[order][:-1]]]
    return row[first], col[first]


def score_3gpp(rmax, run, alphas, gap, zone=ZONE, bands=BANDS, return_matched=False):
    """1 本の走行の、しきい値ごとの見逃し・誤警報の分子と分母。値はすべて (A,) か (A, 帯)。
    return_matched のときは、フレームごとの対応づけの有無 (A, F) も返す（見逃しの内訳に使う）"""
    F, _ = rmax.shape
    A = len(alphas)
    rng, d = run["rng"], run["d"]
    row, col = extract_objects(rmax, alphas, gap)
    r_obj = rng[col]
    x_obj = np.sqrt(np.maximum(r_obj ** 2 - d ** 2, 0.0))          # (K,) BEV の走路方向の位置
    x_true = np.abs(run["s"])                                      # (F,) 真値も同じ d で写すと |s| になる
    in_zone = (r_obj >= zone[0]) & (r_obj <= zone[1])
    dist = np.abs(x_obj - x_true[row % F])
    # フレームごとに最も近い物体（1 対 1 の対応づけ。目標は 1 人）
    nearest = np.full(A * F, -1)
    if len(row):
        o = np.lexsort((dist, row))
        fst = o[np.r_[True, row[o][1:] != row[o][:-1]]]
        nearest[row[fst]] = fst
    has = nearest >= 0
    matched = np.zeros(A * F, bool)
    matched[has] = dist[nearest[has]] <= GATE_BEV
    near_out = np.zeros(A * F, bool)
    near_out[has] = ~in_zone[nearest[has]]
    n_zone = np.bincount(row[in_zone], minlength=A * F)
    n_den = n_zone + (matched & near_out)                          # 区間の外で対応づいた物体も分母に入れる
    n_false = n_den - matched
    matched, n_den, n_false, n_zone = (z.reshape(A, F) for z in (matched, n_den, n_false, n_zone))

    r, on = run["r"], run["on"]
    out = dict(tgt=np.zeros(len(bands)), miss=np.zeros((A, len(bands))),
               t2_sum=np.zeros((A, len(bands))), t2_n=np.zeros((A, len(bands))))
    for j, (b0, b1) in enumerate(bands.values()):
        m = on & (r >= b0) & (r <= b1)                             # (F,)
        out["tgt"][j] = m.sum()
        out["miss"][:, j] = (~matched[:, m]).sum(axis=1)
        frac = np.where(n_den > 0, n_false / np.maximum(n_den, 1), 0.0)   # (A, F)
        ok = m[None, :] & (n_den > 0)
        out["t2_sum"][:, j] = (frac * ok).sum(axis=1)
        out["t2_n"][:, j] = ok.sum(axis=1)
    off = ~on | (r < zone[0] - T1_MARGIN) | (r > zone[1] + T1_MARGIN)
    out["t1_n"] = off.sum()
    out["t1_hit"] = (n_zone[:, off] > 0).sum(axis=1)
    if return_matched:
        return out, matched
    return out


def rates(acc):
    """集計した分子・分母から率を出す。帯ごと (A, B) と全体 (A,)"""
    miss_b = acc["miss"] / np.maximum(acc["tgt"], 1)
    t2_b = acc["t2_sum"] / np.maximum(acc["t2_n"], 1)
    miss = acc["miss"].sum(axis=1) / max(acc["tgt"].sum(), 1)
    t2 = acc["t2_sum"].sum(axis=1) / np.maximum(acc["t2_n"].sum(axis=1), 1)
    t1 = acc["t1_hit"] / max(acc["t1_n"], 1)
    return dict(miss=miss, miss_b=miss_b, t2=t2, t2_b=t2_b, t1=t1)


def add(acc, x):
    for k, v in x.items():
        acc[k] = acc.get(k, 0) + v
    return acc


def collapse(ratio, use):
    """ドップラーを潰す: 検出に使うドップラーでの比の最大 (F, R)"""
    return ratio[:, use, :].max(axis=1)


# ------------------------------------------------------------------
# T1
# ------------------------------------------------------------------

def t1(runs):
    a_nom = cc.nominal_alpha_db()
    print("=" * 80)
    print("T1. 3GPP の数え方（MTI、CA-CFAR 窓は cfar_compare の初期値）")
    print("=" * 80)
    print(f"評価に使う走行 {len(runs)} 本: " + ", ".join(f"No.{r['n']}" for r in runs))

    # 今の数え方（セル単位）を同じ 12 本で出し直し、誤警報率 1e-3 のしきい値を得る
    hit = 0
    tot = 0
    fa = 0
    fa_n = 0
    for run in runs:
        ratio = cc.cfar_ratio_db(run["p"])                         # (F, D, R)
        assert np.allclose(ratio, cfar_ratio(run["p"], cc.N_TRAIN, cc.N_GUARD))
        kr, kd, r, on = cc.truth_path(run["v"], run["t0"], run["d"], run["sign"], run["t"], run["rng"], run["vel"])
        h, tt, f_, fn = cc.evaluate(ratio, run["use"], kr, kd, r, on, run["rng"], run["dc"])
        hit, tot, fa, fa_n = hit + h, tot + tt, fa + f_, fa_n + fn
        run["rmax"] = collapse(ratio, run["use"])                  # (F, R)
    fa_cell = fa.sum(axis=1) / fa_n.sum()                          # (A,)
    pd_cell = hit / np.maximum(tot, 1)                             # (A, 近/中/遠)
    a_cell = cc.interp_pd_at_fa(fa_cell, ALPHA_DB, cc.FA_REF)
    print(f"公称しきい値（Pfa {cc.NOMINAL_PFA:g}）{a_nom:.2f} dB、セル単位の誤警報率 {cc.FA_REF:g} のしきい値 {a_cell:.2f} dB")
    print(f"目標が区間にいるフレーム: " + ", ".join(
        f"{b} {int(sum(((run['on']) & (run['r'] >= lo) & (run['r'] <= hi)).sum() for run in runs))}"
        for b, (lo, hi) in BANDS.items()))
    print(f"Type 1 に使うフレーム（区間の端から {T1_MARGIN:g} m 外か走路にいない）: "
          f"{int(sum((~run['on'] | (run['r'] < ZONE[0] - T1_MARGIN) | (run['r'] > ZONE[1] + T1_MARGIN)).sum() for run in runs))}")

    alphas = np.array([a_nom, a_cell])
    acc = {}
    per_run = []
    for run in runs:
        x = score_3gpp(run["rmax"], run, alphas, GAP)
        add(acc, x)
        per_run.append((run, rates(x)))
    res = rates(acc)

    print("\n(1) 12 本の合計")
    print(f"{'しきい値':>16}{'見逃し':>8}{'(中)':>7}{'(遠)':>7}{'Type 2':>9}{'(中)':>7}{'(遠)':>7}{'Type 1':>9}"
          f"   | 今の数え方の検出率 中 / 遠")
    pd_at = [np.interp(a, ALPHA_DB, pd_cell[:, 1]) for a in alphas], [np.interp(a, ALPHA_DB, pd_cell[:, 2]) for a in alphas]
    for i, lab in enumerate([f"公称 {a_nom:.1f} dB", f"セル1e-3 {a_cell:.1f} dB"]):
        print(f"{lab:>16}{res['miss'][i]:8.2f}{res['miss_b'][i, 0]:7.2f}{res['miss_b'][i, 1]:7.2f}"
              f"{res['t2'][i]:9.2f}{res['t2_b'][i, 0]:7.2f}{res['t2_b'][i, 1]:7.2f}{res['t1'][i]:9.2f}"
              f"   | {pd_at[0][i]:.2f} / {pd_at[1][i]:.2f}（見逃しに直すと {1 - pd_at[0][i]:.2f} / {1 - pd_at[1][i]:.2f}）")
    print("    今の数え方: 軌跡のセル（距離 ±1・ドップラー ±1 ビン）に CFAR の検出が 1 つでもあれば検出（しきい値は線形補間）")
    print(f"    Type 2 の分母（目標が区間にいて物体があるフレーム）: 公称 {int(acc['t2_n'][0].sum())}、"
          f"セル1e-3 {int(acc['t2_n'][1].sum())}")

    print(f"\n(2) 走行ごと（公称しきい値 {a_nom:.1f} dB）。見逃しは 中 / 遠 / 全体、分母 0 は -")
    print(f"{'':28}{'見逃し 中/遠/全体':>22}{'Type 2':>9}{'Type 1':>9}{'区間フレーム':>14}")
    for run, x in per_run:
        tgt = x["miss_b"]
        n_tgt = [int(((run["on"]) & (run["r"] >= lo) & (run["r"] <= hi)).sum()) for lo, hi in BANDS.values()]
        mb = "/".join("-" if n_tgt[j] == 0 else f"{tgt[0, j]:.2f}" for j in range(len(BANDS)))
        print(f"    No.{run['n']:2d} {run['mode']} lane{run['lane']} {run['dr']}:"
              f"{mb + '/' + f'{x['miss'][0]:.2f}':>22}{x['t2'][0]:9.2f}{x['t1'][0]:9.2f}{sum(n_tgt):>14d}")
    # 対応づかない物体がどの距離に出ているか。同じ距離に集まるなら、目標でない決まった反射の消え残り
    hist = np.zeros(len(runs[0]["rng"]), int)
    for run in runs:
        row, col = extract_objects(run["rmax"], [a_nom], GAP)
        r_obj = run["rng"][col]
        x_obj = np.sqrt(np.maximum(r_obj ** 2 - run["d"] ** 2, 0.0))
        false = (np.abs(x_obj - np.abs(run["s"])[row]) > GATE_BEV) & (r_obj >= ZONE[0]) & (r_obj <= ZONE[1])
        np.add.at(hist, col[false], 1)
    top = np.argsort(hist)[::-1][:6]
    print(f"\n(3) 対応づかない物体の位置（公称しきい値、区間内、全フレーム）: 計 {hist.sum()} 個。多い距離ビン: "
          + ", ".join(f"{runs[0]['rng'][k]:.1f} m {hist[k]}" for k in top))
    return dict(a_nom=a_nom, a_cell=a_cell, res=res, per_run=per_run)


def plot_example(run, train, guard, gap, alpha, title, out):
    """1 本の走行の、物体（対応づいた / つかない）と真値の距離−時間"""
    rmax = collapse(cfar_ratio(run["p"], train, guard), run["use"])   # (F, R)
    row, col = extract_objects(rmax, [alpha], gap)
    rng, t = run["rng"], run["t"]
    r_obj = rng[col]
    x_obj = np.sqrt(np.maximum(r_obj ** 2 - run["d"] ** 2, 0.0))
    ok = np.abs(x_obj - np.abs(run["s"])[row]) <= GATE_BEV
    keep = rng <= 60
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    ax.pcolormesh(rng[keep], t, rmax[:, keep], shading="auto", cmap="Greys", vmin=0, vmax=20)
    ax.axvspan(ZONE[0], ZONE[1], color="tab:blue", alpha=0.06)
    on = run["on"]
    ax.plot(run["r"][on], t[on], "-", color="tab:red", lw=0.8, label="true path")
    ax.plot(r_obj[ok], t[row[ok]], "o", mfc="none", color="tab:green", ms=5, label="object, matched (BEV <= 2 m)")
    ax.plot(r_obj[~ok], t[row[~ok]], "x", color="tab:orange", ms=5, label="object, unmatched")
    ax.set_xlim(rng[keep][0], rng[keep][-1])
    ax.set_xlabel("range R [m]  (shaded: evaluation zone)")
    ax.set_ylabel("time [s]")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=9)
    fig.savefig(out, dpi=110)
    print(f"-> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------
# T2
# ------------------------------------------------------------------

def sweep(runs, train, guard, gap):
    """1 つの組み合わせで、しきい値を振った率と走行ごとの率"""
    acc = {}
    per_run = []
    for run in runs:
        rmax = collapse(cfar_ratio(run["p"], train, guard), run["use"])   # (F, R)
        x = score_3gpp(rmax, run, ALPHA_DB, gap)
        add(acc, x)
        per_run.append(rates(x))
    return rates(acc), per_run


def t2(runs):
    print("\n" + "=" * 80)
    print("T2. 窓・物体のまとめ方・しきい値を振る（MTI、12 本）")
    print("=" * 80)
    combos = list(product(TRAINS, GUARDS, GAPS))
    out = []
    print(f"{'訓練':>8}{'ガード':>8}{'隙間':>5} | 公称しきい値: {'しきい値':>7}{'見逃し':>7}{'T1':>6}{'T2':>6}"
          f" | max(見逃し, T1, T2) 最小: {'しきい値':>7}{'見逃し':>7}{'T1':>6}{'T2':>6}{'max':>6}")
    for train, guard, gap in combos:
        res, per_run = sweep(runs, train, guard, gap)
        worst = np.max([res["miss"], res["t1"], res["t2"]], axis=0)       # (A,)
        i = int(worst.argmin())
        a_nom = nominal_alpha(train, guard)
        nom = {k: float(np.interp(a_nom, ALPHA_DB, res[k])) for k in ("miss", "t1", "t2")}
        out.append(dict(train=train, guard=guard, gap=gap, res=res, per_run=per_run, worst=worst, i=i, a_nom=a_nom))
        print(f"{str(train):>8}{str(guard):>8}{gap:>5} | {'':14}{a_nom:7.1f}{nom['miss']:7.2f}{nom['t1']:6.2f}{nom['t2']:6.2f}"
              f" | {'':24}{ALPHA_DB[i]:7.1f}{res['miss'][i]:7.2f}{res['t1'][i]:6.2f}{res['t2'][i]:6.2f}{worst[i]:6.2f}")
    print("    T1 = 誤警報 Type 1（近似）、T2 = 誤警報 Type 2。公称しきい値の値は 0.5 dB 刻みの格子から線形補間")
    # しきい値を下げきったときの見逃し（見逃しの床）。隙間で変わるなら、物体のつながりが床を作っている
    print("    見逃しの最小（しきい値）: " + ", ".join(
        f"{c['train']}{c['guard']}隙間{c['gap']} {c['res']['miss'].min():.2f}（{ALPHA_DB[c['res']['miss'].argmin()]:.1f} dB）"
        for c in out))

    # 2. 両方 5% 以下の組み合わせ
    ok = [(c, k) for c in out for k in range(len(ALPHA_DB)) if c["worst"][k] <= TARGET]
    print(f"\n(2) 見逃し・誤警報（Type 1・Type 2）が両方 {TARGET:g} 以下になる組み合わせ: {len(ok)} 個（しきい値込み）")
    if ok:
        c, k = min(ok, key=lambda z: z[0]["worst"][z[1]])
        print(f"    余裕が最も大きいもの: 訓練 {c['train']}・ガード {c['guard']}・隙間 {c['gap']}・しきい値 {ALPHA_DB[k]:.1f} dB")
    best = min(out, key=lambda c: c["worst"][c["i"]])
    k = best["i"]
    print(f"\n(3) max(見逃し, Type 1, Type 2) が最小の組み合わせ: 訓練 {best['train']}・ガード {best['guard']}・"
          f"隙間 {best['gap']}・しきい値 {ALPHA_DB[k]:.1f} dB")
    r = best["res"]
    print(f"    見逃し {r['miss'][k]:.2f}（中 {r['miss_b'][k, 0]:.2f} / 遠 {r['miss_b'][k, 1]:.2f}）、"
          f"Type 1 {r['t1'][k]:.2f}、Type 2 {r['t2'][k]:.2f}（中 {r['t2_b'][k, 0]:.2f} / 遠 {r['t2_b'][k, 1]:.2f}）")
    # 見逃しと誤警報のどちらを 5% にそろえると、もう片方がどこまで行くか（兼ね合いの両端）
    for key, lab in (("miss", "見逃し"), ("t1", "Type 1"), ("t2", "Type 2")):
        cand = [(c, j) for c in out for j in range(len(ALPHA_DB)) if c["res"][key][j] <= TARGET]
        if not cand:
            print(f"    {lab} ≤ {TARGET:g} になるしきい値はどの組み合わせにもない")
            continue
        c, j = min(cand, key=lambda z: z[0]["worst"][z[1]])
        rr = c["res"]
        print(f"    {lab} ≤ {TARGET:g} の中で max が最小: 訓練 {c['train']}・ガード {c['guard']}・隙間 {c['gap']}・"
              f"{ALPHA_DB[j]:.1f} dB → 見逃し {rr['miss'][j]:.2f}、Type 1 {rr['t1'][j]:.2f}、Type 2 {rr['t2'][j]:.2f}")

    print(f"\n    選んだ組み合わせ（(3)）の走行ごと: 見逃し / Type 1 / Type 2")
    for run, x in zip(runs, best["per_run"]):
        print(f"    No.{run['n']:2d} {run['mode']} lane{run['lane']} {run['dr']}: "
              f"{x['miss'][k]:.2f} / {x['t1'][k]:.2f} / {x['t2'][k]:.2f}")
    return out, best


def plot_t2(out, best):
    """組み合わせごとの見逃し − 誤警報の曲線（しきい値を振る）。Type 1・Type 2 を左右に並べる"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, layout="constrained")
    styles = {0: "-", 1: "--"}
    labels_done = set()
    for c in out:
        col = f"C{TRAINS.index(c['train']) * len(GUARDS) + GUARDS.index(c['guard'])}"
        lab = f"train {c['train']}, guard {c['guard']}"
        for ax, key in zip(axes, ("t1", "t2")):
            ax.plot(c["res"][key], c["res"]["miss"], styles[c["gap"]], color=col, lw=1.2,
                    label=lab if (lab not in labels_done and ax is axes[0] and c["gap"] == 0) else None)
            i_nom = int(np.abs(ALPHA_DB - c["a_nom"]).argmin())
            ax.plot(c["res"][key][i_nom], c["res"]["miss"][i_nom], "o", color=col, ms=4)
        labels_done.add(lab)
    k = best["i"]
    for ax, key, name in zip(axes, ("t1", "t2"), ("false alarm Type 1 (approx., target elsewhere)",
                                                   "false alarm Type 2 (TR 38.765)")):
        ax.plot(best["res"][key][k], best["res"]["miss"][k], "*", color="k", ms=12, zorder=5)
        ax.axvline(TARGET, color="0.4", lw=0.8, ls=":")
        ax.axhline(TARGET, color="0.4", lw=0.8, ls=":")
        ax.fill_between([0, TARGET], 0, TARGET, color="0.85", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("missed detection (R 30-49 m)")
    axes[0].plot([], [], "k-", label="gap 0 bins")
    axes[0].plot([], [], "k--", label="gap 1 bin")
    axes[0].plot([], [], "ko", ms=4, label="nominal threshold (Pfa 1e-3)")
    axes[0].plot([], [], "k*", ms=10, label="min of max(miss, T1, T2)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"3GPP counting (1 frame = 1 event, objects, BEV gate {GATE_BEV:g} m), MTI, night 10/1, 12 runs. "
                 f"Threshold {ALPHA_DB[0]:g}-{ALPHA_DB[-1]:g} dB. Grey box: both <= {TARGET:g}", fontsize=10)
    o = ROOT / "figures" / "t2_tradeoff.png"
    o.parent.mkdir(exist_ok=True)
    fig.savefig(o, dpi=110)
    print(f"-> {o.relative_to(ROOT)}")


# ------------------------------------------------------------------
# T3（10/5、参考値）
# ------------------------------------------------------------------

DATA_1005 = ROOT.parent / "1005" / "data"          # ../1005/axes_maps_1005.py の export() が作る npz
RUNS_1005 = {"135532": "app", "135606": "dep", "143104": "app", "143133": "dep"}   # 自転車の 4 本
ZONE_1005 = (22.0, 37.0)                            # 10/5 で見えた帯。真値の当てはめと評価の両方に使う
BANDS_1005 = {"22-30 m": (22.0, 30.0), "30-37 m": (30.0, 37.0)}
# 10/1 のレーン 1〜3 の d の範囲 ±2 m を総当たり（../collect/check_run_1013.py と同じ考え方）
D_GRID_1005 = np.arange(min(lane_range(k)[0] for k in (1, 2, 3)) - 2.0,
                        max(lane_range(k)[0] for k in (1, 2, 3)) + 2.0 + 1e-9, 0.25)


def npz_1005(tag, nomti=False):
    s = f"atlas_log_20261005_{tag}"
    return DATA_1005 / ("export_nomti" if nomti else "export") / s / f"{s}_rd.npz"


def load_runs_1005():
    """10/5 の自転車 4 本の真値（d も総当たり）と MTI の電力。

    cc.power_maps・cc.fit_truth は 10/1 のファイル名と当てはめ帯（cc.FIT_R）を中で参照しているので、
    その 2 つだけを一時的に 10/5 用に差し替えて呼ぶ（計算そのものは 10/1 と同じにするため）
    """
    keep = (cc.npz_path, cc.FIT_R)
    cc.npz_path, cc.FIT_R = npz_1005, ZONE_1005
    runs = []
    try:
        for tag, dr in RUNS_1005.items():
            maps, use, rng, vel, t, dc = cc.power_maps(tag)
            sign = -1.0 if dr == "app" else +1.0
            ex = np.mean([cc.excess_rt(maps[m], use[m]) for m in cc.METHODS], axis=0)   # (F, R)
            vg = cc.V_GRID[(cc.V_GRID >= cc.V_RANGE["bike"][0]) & (cc.V_GRID <= cc.V_RANGE["bike"][1])]
            fits = [(cc.fit_truth(ex, rng, t, d0, sign, vg), d0) for d0 in D_GRID_1005]
            (sc, v, t0), d0 = max(fits, key=lambda z: z[0][0])
            ctrl = max(cc.fit_truth(ex[::-1], rng, t, d0, sign, vg)[0] for d0 in D_GRID_1005)
            s = v * (t - t0)                                       # (F,)
            runs.append(dict(n=tag, mode="bike", lane="?", dr=dr, v=v, t0=t0, d=d0, sign=sign, sc=sc, ctrl=ctrl,
                             p=maps["MTI"], use=use["MTI"], rng=rng, vel=vel, t=t, dc=dc,
                             s=s, r=np.hypot(s, d0), on=(s * sign) >= 0))
    finally:
        cc.npz_path, cc.FIT_R = keep
    return runs


def t3(best, runs_1001):
    train, guard, gap = best["train"], best["guard"], best["gap"]
    alpha = ALPHA_DB[best["i"]]
    print("\n" + "=" * 80)
    print(f"T3. 10/5 の自転車 4 本に、T2 で選んだ組み合わせをそのまま掛ける（参考値。合否の判定には使わない）")
    print(f"    訓練 {train}・ガード {guard}・隙間 {gap}・しきい値 {alpha:.1f} dB、評価区間 R {ZONE_1005[0]:g}〜{ZONE_1005[1]:g} m")
    print("=" * 80)
    runs = load_runs_1005()
    print(f"(1) 真値の当てはめ（3 方式の床超過の平均、R {ZONE_1005[0]:g}〜{ZONE_1005[1]:g} m、"
          f"d を {D_GRID_1005[0]:.1f}〜{D_GRID_1005[-1]:.1f} m で総当たり）")
    for run in runs:
        print(f"    {run['n']} bike {run['dr']}: d {run['d']:.2f} m, v {run['v']:.2f} m/s, t0 {run['t0']:6.1f} s, "
              f"軌跡上 {run['sc']:.1f} dB（時刻反転の対照 {run['ctrl']:.1f} dB、差 {run['sc'] - run['ctrl']:.1f} dB）")
    acc = {}
    print(f"(2) 走行ごと: 見逃し（{' / '.join(BANDS_1005)} / 全体）・Type 1・Type 2、区間にいるフレーム")
    for run in runs:
        rmax = collapse(cfar_ratio(run["p"], train, guard), run["use"])  # (F, R)
        x = score_3gpp(rmax, run, [alpha], gap, ZONE_1005, BANDS_1005)
        add(acc, x)
        r = rates(x)
        mb = " / ".join("-" if x["tgt"][j] == 0 else f"{r['miss_b'][0, j]:.2f}" for j in range(len(BANDS_1005)))
        print(f"    {run['n']} bike {run['dr']}: 見逃し {mb} / {r['miss'][0]:.2f}、Type 1 {r['t1'][0]:.2f}、"
              f"Type 2 {r['t2'][0]:.2f}、{int(x['tgt'].sum())} フレーム")
    r = rates(acc)
    print(f"    4 本の合計: 見逃し {r['miss'][0]:.2f}（{' / '.join(f'{v:.2f}' for v in r['miss_b'][0])}）、"
          f"Type 1 {r['t1'][0]:.2f}、Type 2 {r['t2'][0]:.2f}。区間にいるフレーム {int(acc['tgt'].sum())}、"
          f"Type 1 に使うフレーム {int(acc['t1_n'])}")
    # 10/1 夜の自転車（レーン 1・2）と並べる。10/1 の評価区間は R 30〜49 m なので帯は違う
    k = best["i"]
    bike = [x for run, x in zip(runs_1001, best["per_run"]) if run["mode"] == "bike"]
    print(f"    （並べる値）10/1 夜の 12 本: 見逃し {best['res']['miss'][k]:.2f}、Type 1 {best['res']['t1'][k]:.2f}、"
          f"Type 2 {best['res']['t2'][k]:.2f}。自転車 {len(bike)} 本の走行ごとの見逃しの中央値 "
          f"{np.median([x['miss'][k] for x in bike]):.2f}")
    return r


# ------------------------------------------------------------------
# 見逃しの内訳（T2 の後に追加）
# ------------------------------------------------------------------
# 見逃しのフレームを 2 つに分ける:
#   (a) 見えていない: 真値から BEV で GATE_BEV 以内の距離ビンに、どのドップラーでも CFAR の検出がない
#   (b) 見えているが位置がずれた: そのビンに検出はあるが、隣の検出とつながった物体の位置（比の最大）が
#       GATE_BEV より離れた
# (a) は信号を強くする（角度方向・積分）、(b) は物体へのまとめ方・位置の出し方で直すもので、手段が違う

SPLIT_ALPHAS = [4.0, 6.0, 8.0, 9.0, 10.0, 12.0]


def miss_split(runs, train, guard, gap, alphas):
    """しきい値ごとの、目標が区間にいるフレーム数と (a)・(b) の数、(a) の不足 [dB] のリスト"""
    alphas = np.asarray(alphas)
    A = len(alphas)
    rows = []
    for run in runs:
        rmax = collapse(cfar_ratio(run["p"], train, guard), run["use"])   # (F, R)
        _, matched = score_3gpp(rmax, run, alphas, gap, return_matched=True)   # (A, F)
        rng, d = run["rng"], run["d"]
        x_bin = np.sqrt(np.maximum(rng ** 2 - d ** 2, 0.0))        # (R,) 各距離ビンの BEV 位置
        gate = np.abs(x_bin[None, :] - np.abs(run["s"])[:, None]) <= GATE_BEV   # (F, R)
        gate_max = np.where(gate, rmax, -np.inf).max(axis=1)       # (F,) 真値の近くの比の最大
        tgt = run["on"] & (run["r"] >= ZONE[0]) & (run["r"] <= ZONE[1])   # (F,)
        seen = gate_max[None, :] > alphas[:, None]                 # (A, F)
        miss = ~matched & tgt[None, :]
        a = miss & ~seen
        b = miss & seen
        rows.append(dict(run=run, tgt=tgt.sum(), a=a.sum(axis=1), b=b.sum(axis=1),
                         short=[alphas[i] - gate_max[a[i]] for i in range(A)]))
    return rows


def print_split(rows, alphas, label):
    tot = sum(x["tgt"] for x in rows)
    a = sum(x["a"] for x in rows)
    b = sum(x["b"] for x in rows)
    print(f"\n  {label}（12 本、区間にいるフレーム {tot}）")
    print(f"    {'しきい値':>8}{'見逃し':>8}{'(a) 見えない':>14}{'(b) 位置ずれ':>14}{'(b) の割合':>10}"
          f"   (a) の不足 dB: 中央値 / 75% / 90%")
    for i, al in enumerate(alphas):
        sh = np.concatenate([x["short"][i] for x in rows])
        q = "  -" if len(sh) == 0 else " / ".join(f"{v:.1f}" for v in np.percentile(sh, [50, 75, 90]))
        print(f"    {al:8.1f}{(a[i] + b[i]) / tot:8.2f}{a[i] / tot:14.2f}{b[i] / tot:14.2f}"
              f"{b[i] / max(a[i] + b[i], 1):10.2f}   {q}")
    return a, b, tot


def split_section(runs, best):
    print("\n" + "=" * 80)
    print("見逃しの内訳: (a) 真値から BEV 2 m 以内のビンに検出がない / (b) 検出はあるが物体の位置が 2 m より離れた")
    print("=" * 80)
    print("  (a) の不足 dB = しきい値 − 真値から 2 m 以内のビンでの比の最大（ドップラーを潰した値）。信号をどれだけ強くすれば拾えるかの目安")
    nom = cc.nominal_alpha_db()
    sets = [(cc.N_TRAIN, cc.N_GUARD, 0, "T1 の窓 訓練 (2, 5)・ガード (1, 2)・隙間 0"),
            (best["train"], best["guard"], 0, f"T2 で選んだ窓 訓練 {best['train']}・ガード {best['guard']}・隙間 0"),
            (best["train"], best["guard"], 1, f"同じ窓で隙間 1")]
    alphas = sorted(set(SPLIT_ALPHAS + [round(nom, 2)]))
    for train, guard, gap, lab in sets:
        print_split(miss_split(runs, train, guard, gap, alphas), alphas, lab)

    # 走行の種類ごと・走行ごと（T1 の窓、公称しきい値）
    rows = miss_split(runs, cc.N_TRAIN, cc.N_GUARD, 0, [nom])
    print(f"\n  走行ごと（T1 の窓、公称 {nom:.1f} dB）: 区間フレーム / 見逃し (a) / (b)、(a) の不足 dB の中央値")
    groups = {}
    for x in rows:
        run = x["run"]
        g = f"{run['mode']} lane{run['lane']}"
        groups.setdefault(g, []).append(x)
        med = f"{np.median(x['short'][0]):.1f}" if len(x["short"][0]) else "-"
        print(f"    No.{run['n']:2d} {g} {run['dr']}: {x['tgt']:3d} / {x['a'][0]:3d} / {x['b'][0]:3d}、{med}")
    print("  種類ごとの合計: 区間フレーム / (a) の割合 / (b) の割合")
    for g, xs in groups.items():
        t = sum(x["tgt"] for x in xs)
        print(f"    {g}: {t:3d} / {sum(x['a'][0] for x in xs) / t:.2f} / {sum(x['b'][0] for x in xs) / t:.2f}")

    # 図: T1 の窓で、しきい値を振ったときの (a)・(b) の割合
    grid = ALPHA_DB
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, gap in zip(axes, (0, 1)):
        rr = miss_split(runs, cc.N_TRAIN, cc.N_GUARD, gap, grid)
        tot = sum(x["tgt"] for x in rr)
        a = sum(x["a"] for x in rr) / tot
        b = sum(x["b"] for x in rr) / tot
        ax.stackplot(grid, a, b, colors=["tab:blue", "tab:orange"],
                     labels=["(a) no detection within 2 m of truth", "(b) detected, but object position > 2 m off"])
        ax.axvline(nom, color="k", lw=0.8, ls="--")
        ax.axhline(TARGET, color="0.4", lw=0.8, ls=":")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("CFAR threshold [dB]  (dashed: nominal Pfa 1e-3)")
        ax.set_title(f"gap {gap} bin(s)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("missed detection (R 30-49 m)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Breakdown of missed detections, MTI, CA-CFAR train {cc.N_TRAIN} guard {cc.N_GUARD}, "
                 f"night 10/1, 12 runs", fontsize=10)
    o = ROOT / "figures" / "miss_split.png"
    fig.savefig(o, dpi=110)
    print(f"-> {o.relative_to(ROOT)}")


def main():
    runs = load_runs()
    r1 = t1(runs)
    ex = next(run for run in runs if run["n"] == 19)
    (ROOT / "figures").mkdir(exist_ok=True)
    plot_example(ex, cc.N_TRAIN, cc.N_GUARD, GAP, r1["a_nom"],
                 f"No.19 (bike, lane 1, app), MTI, CA-CFAR train {cc.N_TRAIN} guard {cc.N_GUARD}, "
                 f"nominal {r1['a_nom']:.1f} dB.\nGrey: CFAR ratio max over Doppler [0-20 dB]",
                 ROOT / "figures" / "t1_objects_No19.png")
    out, best = t2(runs)
    plot_t2(out, best)
    t3(best, runs)
    split_section(runs, best)


if __name__ == "__main__":
    main()
