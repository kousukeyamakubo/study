# visibility.py の続き。「見えている」の上限と下限をそろえる（README「見えているかの上限」の次の段）。
#
# visibility.py の 2 つの弱点を直す:
#   B. 人が区間にいる時間帯を、等速の当てはめではなくデータの筋から決める（レーン 2 は当てはめが筋から外れていた）
#   C. 一番強い反射だけでなく、上位 K 個の反射（RD の局所最大）から筋を探す（消え残りより弱い目標も拾う）
# あわせて A. 条件の数値を振って、結果が数値の選び方に左右されないかを見る。
#
# 判定（K 個の反射それぞれ＝節点ごと）。visibility.py の 3 条件を、反射が複数あっても使える形にした:
#   前後 SLOPE_HALF フレームへ、節点のドップラーの速度で予測した距離に最も近い反射（STEP_M 以内）を順にたどって道をつくり、
#   1 つながる : 道の点が MIN_PTS 以上（途切れたらそこで止める）
#   2 向き     : 道の点への直線の当てはめの dR/dt が走行の向きと合い、MIN_RATE 以上
#   3 ドップラー: 節点のドップラーの速度と dR/dt の差が DOP_TOL 以内
# ドップラーで次の点を選ぶので 3 は通りやすくなるが、その分は偶然の水準（人がいないフレーム）に同じ判定を掛けて測る。
#
# 時間帯（B）: K=1・既定の数値で条件を満たした節点から、走行の向きに人の速さで動く最長の鎖を取り、
#   鎖の点の間は直線でつなぎ、両端は端の 5 点の傾きで区間の外まで延ばす。この筋が区間にある時間を「人が区間にいる」とする。
#   鎖の点は定義上「見えている」ので、評価できるのは鎖の点の間（途切れた所）と延ばした両端。この偏りは上限側に効く。
#   偶然の水準は、筋が区間の端から 3 m 以上外のフレーム（筋のある時間の外を含む）で測る。
#
# 使い方:
#   python visibility2.py --zone 37.5 49 --tag _far > visibility2_far_log.txt   # 消え残りの線（R 31〜37 m）より遠い帯
#   python visibility2.py --zone 30 37.5 --tag _near > visibility2_near_log.txt  # 消え残りの線の帯

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cfar_3gpp as c3                                           # noqa: E402
from analyze_night import condition                              # noqa: E402

ZONE = c3.ZONE                           # main() で --zone に置き換える
FRAME_S = 0.2
V_MAX = {"walk": 2.5, "bike": 5.0}
KS = [1, 2, 3, 5]
DEFAULT = dict(step_scale=1.0, min_pts=5, min_rate=0.5, dop_tol=1.0)
SLOPE_HALF = 3
CHAIN_GAP_S = 5.0          # 鎖の隣り合う点の時間の離れの上限 [s]
MIN_CHAIN = 5              # これより短い鎖の走行は、時間帯を決められないので C・A の評価から外す
OFF_MARGIN = 3.0
SWEEP = {"step_scale": [0.75, 1.0, 1.5], "min_pts": [4, 5, 6], "min_rate": [0.3, 0.5, 0.8], "dop_tol": [0.6, 1.0, 1.5]}


def load_all():
    """レーン 1・2 の 24 本（等速の当てはめで外した走行も入れる。時間帯はデータの筋から決めるので）"""
    runs = []
    for n in [n for n in range(1, c3.N_MAIN + 1) if condition(n)[1] in (1, 2)]:
        mode, lane, dr = condition(n)
        maps, use, rng, vel, t, dc = c3.cc.power_maps(c3.TAGS[n - 1])
        runs.append(dict(n=n, mode=mode, lane=lane, dr=dr, sign=-1.0 if dr == "app" else 1.0,
                         p=maps["MTI"], use=use["MTI"], rng=rng, vel=vel, t=t))
    return runs


def peaks(p, use, rng, k):
    """各フレームの区間内の上位 k 個の局所最大（ドップラー × 距離の 3×3。ドップラーは折り返す）。
    返り値: kr, kd (F, k)。足りないところは -1"""
    F, D, R = p.shape
    zr = (rng >= ZONE[0]) & (rng <= ZONE[1])
    # 局所最大は全距離で取ってから区間に絞る。先に区間の外を 0 にすると、区間の端のビンが偽の局所最大になる
    q = np.where(use[None, :, None], p, 0.0)                       # (F, D, R)
    pad = np.pad(q, ((0, 0), (0, 0), (1, 1)))
    pad = np.concatenate([pad[:, -1:], pad, pad[:, :1]], axis=1)  # ドップラーを折り返す
    nb = np.max(np.stack([pad[:, 1 + i:1 + i + D, 1 + j:1 + j + R] for i in (-1, 0, 1) for j in (-1, 0, 1)
                          if (i, j) != (0, 0)]), axis=0)
    loc = (q > nb) & (q > 0) & zr[None, None, :]
    val = np.where(loc, q, 0.0).reshape(F, -1)
    order = np.argsort(-val, axis=1)[:, :k]                        # (F, k)
    ok = np.take_along_axis(val, order, axis=1) > 0
    kd, kr = np.unravel_index(order, (D, R))
    return np.where(ok, kr, -1), np.where(ok, kd, -1)


def judge_nodes(kr, kd, rng, vel, t, mode, sign, prm):
    """各節点が 3 条件を満たすか (F, k)"""
    F, k = kr.shape
    step = V_MAX[mode] * FRAME_S * prm["step_scale"] + (rng[1] - rng[0])
    rr = np.where(kr >= 0, rng[np.maximum(kr, 0)], np.nan)        # (F, k)
    ok = np.zeros((F, k), bool)
    for f in range(F):
        for j in range(k):
            if kr[f, j] < 0:
                continue
            v = vel[kd[f, j]]
            pts_t, pts_r = [t[f]], [rr[f, j]]
            for direction in (1, -1):
                r_prev, t_prev = rr[f, j], t[f]
                for s in range(1, SLOPE_HALF + 1):
                    g = f + direction * s
                    if g < 0 or g >= F:
                        break
                    pred = r_prev + v * (t[g] - t_prev)
                    cand = rr[g]
                    d = np.abs(cand - pred)
                    d = np.where(np.abs(cand - r_prev) <= step, d, np.inf)
                    i = int(np.nanargmin(np.where(np.isnan(d), np.inf, d))) if np.isfinite(np.nan_to_num(d, nan=np.inf)).any() else -1
                    if i < 0 or not np.isfinite(d[i]):
                        break
                    r_prev, t_prev = cand[i], t[g]
                    pts_t.append(t[g])
                    pts_r.append(cand[i])
            if len(pts_t) < prm["min_pts"]:
                continue
            rate = np.polyfit(pts_t, pts_r, 1)[0]
            ok[f, j] = (rate * sign >= prm["min_rate"]) and (abs(v - rate) <= prm["dop_tol"])
    return ok, rr


def prm_min_rate(dt):
    """鎖の隣り合う点の dR/dt の下限 [m/s]。距離の量子化（1 ビン）の分だけ緩める"""
    return DEFAULT["min_rate"] - 0.85 / dt


def data_track(ok, rr, t, mode, sign):
    """条件を満たした節点の最長の鎖から筋 R(t) (F,) を作る。筋のない時刻は nan"""
    pts = [(t[f], rr[f, j]) for f, j in zip(*np.nonzero(ok))]
    if len(pts) < 2:
        return np.full(len(t), np.nan), []
    pts.sort()
    n = len(pts)
    best = np.ones(n, int)
    prev = -np.ones(n, int)
    for b in range(n):
        for a in range(b):
            dt = pts[b][0] - pts[a][0]
            if dt <= 0 or dt > CHAIN_GAP_S:
                continue
            rate = (pts[b][1] - pts[a][1]) / dt * sign
            # 人の速さで、走行の向きに動いている。止まった（rate ≈ 0）つながりは許さない。
            # R 31〜37 m の消え残りの線は数本並ぶので、0 を許すと線を階段状に渡る鎖ができる
            if prm_min_rate(dt) <= rate <= V_MAX[mode] + 0.9 / dt and best[a] + 1 > best[b]:
                best[b], prev[b] = best[a] + 1, a
    i = int(best.argmax())
    chain = []
    while i >= 0:
        chain.append(pts[i])
        i = prev[i]
    chain = chain[::-1]
    # 同じ時刻の点は平均する
    ct = np.array(sorted({c[0] for c in chain}))
    cr = np.array([np.mean([c[1] for c in chain if c[0] == x]) for x in ct])
    track = np.full(len(t), np.nan)
    if len(ct) < 2:
        return track, chain
    inside = (t >= ct[0]) & (t <= ct[-1])
    track[inside] = np.interp(t[inside], ct, cr)
    # 両端は、鎖全体の平均の速さで延ばす（端の数点の傾きは量子化で暴れる）。見えなくなった後の時間帯はここで決まるので、
    # この部分だけは等速の仮定になる
    a = np.polyfit(ct, cr, 1)[0]
    for end in (0, -1):
        side = t < ct[0] if end == 0 else t > ct[-1]
        ext = cr[end] + a * (t[side] - ct[end])
        track[side] = np.where((ext >= ZONE[0] - 6) & (ext <= ZONE[1] + 6), ext, np.nan)
    return track, chain


def main():
    global ZONE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", type=float, nargs=2, default=list(c3.ZONE))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ZONE = tuple(args.zone)
    runs = load_all()
    print(f"区間 R {ZONE[0]:g}〜{ZONE[1]:g} m")
    print("見えているかの上限と下限（真値の位置に頼らない）。レーン 1・2 の 24 本、MTI、DC 以外のドップラー")
    print(f"既定の数値: {DEFAULT}。前後 {SLOPE_HALF} フレームをドップラーの予測でたどる")

    # ---- B. 時間帯をデータの筋から決める（K=1、既定） ----
    for run in runs:
        kr, kd = peaks(run["p"], run["use"], run["rng"], 1)
        ok, rr = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)
        track, chain = data_track(ok, rr, run["t"], run["mode"], run["sign"])
        run["track"] = track
        run["chain"] = chain
        run["usable"] = len(chain) >= MIN_CHAIN
        run["inz"] = (track >= ZONE[0]) & (track <= ZONE[1]) & run["usable"]
        run["off"] = ~((track >= ZONE[0] - OFF_MARGIN) & (track <= ZONE[1] + OFF_MARGIN))   # nan も外側
        # 筋のある時間の端 1 s は、筋の延ばし方次第なので偶然の水準から外す
        has = np.isfinite(track)
        if has.any():
            f0, f1 = np.flatnonzero(has)[[0, -1]]
            run["off"][max(f0 - 5, 0):f0] = False
            run["off"][f1 + 1:f1 + 6] = False

    print("\nB. データの筋（K=1 で条件を満たした節点の最長の鎖）")
    print("    走行: 鎖の点数、筋が区間にあるフレーム数（等速の当てはめで区間にいたフレーム数。当てはめで外した走行は -）")
    fit = {r["n"]: r for r in c3.load_runs()}
    for run in runs:
        f = fit.get(run["n"])
        n_fit = "-" if f is None else str(int((f["on"] & (f["r"] >= ZONE[0]) & (f["r"] <= ZONE[1])).sum()))
        print(f"    No.{run['n']:2d} {run['mode']} lane{run['lane']} {run['dr']}: 鎖 {len(run['chain']):3d} 点、"
              f"区間 {int(run['inz'].sum()):3d} フレーム（等速 {n_fit}）" + ("" if run["usable"] else f"  → 鎖が {MIN_CHAIN} 点未満で外す"))
    for mode in ("walk", "bike"):
        for lane in (1, 2):
            xs = [r for r in runs if r["mode"] == mode and r["lane"] == lane]
            print(f"    {mode} lane{lane}: 使える走行 {sum(r['usable'] for r in xs)} / {len(xs)}")

    # ---- C. 上位 K 個 ----
    def evaluate(prm, ks, verbose):
        out = {}
        for k in ks:
            groups = {}
            off_n = off_k = 0
            for run in [r for r in runs if r["usable"]]:
                kr, kd = peaks(run["p"], run["use"], run["rng"], k)
                ok, _ = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], prm)
                vis = ok.any(axis=1)                               # (F,) どれかの反射が条件を満たす
                g = f"{run['mode']} lane{run['lane']}"
                a = groups.setdefault(g, [0, 0])
                a[0] += int(run["inz"].sum())
                a[1] += int((vis & run["inz"]).sum())
                off_n += int(run["off"].sum())
                off_k += int((vis & run["off"]).sum())
            tot = [sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())]
            out[k] = dict(groups=groups, tot=tot, chance=off_k / max(off_n, 1), off_n=off_n)
            if verbose:
                print(f"    K={k}: " + "、".join(f"{g} {v[1] / max(v[0], 1):.2f}（{v[0]}）" for g, v in groups.items())
                      + f"、全体 {tot[1] / max(tot[0], 1):.2f}（{tot[0]}）| 偶然 {out[k]['chance']:.2f}（{off_n}）")
        return out

    print("\nC. 上位 K 個の反射のどれかが条件を満たすフレームの割合（人が区間にいる = データの筋が区間にある）。"
          "かっこはフレーム数")
    base = evaluate(DEFAULT, KS, True)

    print("\nA. 数値を 1 つずつ振る（ほかは既定）。全体の割合 / 偶然の水準、K=1 と K=3")
    for key, vals in SWEEP.items():
        for v in vals:
            prm = dict(DEFAULT, **{key: v})
            r = evaluate(prm, [1, 3], False)
            print(f"    {key}={v:<5}: " + "  ".join(
                f"K={k} {r[k]['tot'][1] / max(r[k]['tot'][0], 1):.2f} / {r[k]['chance']:.2f}" for k in (1, 3)))

    # ---- 図: K=3 の反射・条件を満たした節点・データの筋・等速の当てはめ ----
    fig, axes = plt.subplots(4, 6, figsize=(22, 13), sharex=True, sharey=True, layout="constrained")
    for ax, run in zip(axes.ravel(), runs):
        kr, kd = peaks(run["p"], run["use"], run["rng"], 3)
        ok, rr = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)
        tt = np.broadcast_to(run["t"][:, None], rr.shape)
        ax.axhspan(ZONE[0], ZONE[1], color="tab:blue", alpha=0.05)
        ax.plot(tt[~ok], rr[~ok], "x", color="0.7", ms=3)
        ax.plot(tt[ok], rr[ok], "o", color="tab:green", ms=3, mfc="none")
        ax.plot(run["t"], run["track"], "-", color="tab:blue", lw=1.2)
        f = fit.get(run["n"])
        if f is not None:
            ax.plot(f["t"][f["on"]], f["r"][f["on"]], "--", color="tab:red", lw=0.8)
        vis = ok.any(axis=1)
        ax.set_title(f"No.{run['n']} {run['mode']} L{run['lane']} {run['dr']}: "
                     f"{int((vis & run['inz']).sum())}/{int(run['inz'].sum())}", fontsize=8)
        ax.set_ylim(25, 55)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("range R [m]")
    fig.suptitle(f"Top-3 reflections in R {ZONE[0]:g}-{ZONE[1]:g} m (grey x: fail, green o: pass 1+2+3), blue: data track (K=1 chain), "
                 "red dashed: constant-speed fit (excluded runs: none). Title: visible / frames in zone", fontsize=11)
    o = ROOT / "figures" / f"visibility2_top3{args.tag}.png"
    fig.savefig(o, dpi=90)
    print(f"\n-> {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


def gap_lengths(zone=(37.5, 49.0), k=3):
    """データの筋が区間にある時間の中で、見えていないフレームが何フレーム続くか（走行の種類ごと）。
    1〜2 フレームの途切れなら前後のフレームでつなげるが、長い途切れはつなげない"""
    global ZONE
    ZONE = zone
    runs = load_all()
    out = {}
    for run in runs:
        kr, kd = peaks(run["p"], run["use"], run["rng"], 1)
        ok, rr = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)
        track, chain = data_track(ok, rr, run["t"], run["mode"], run["sign"])
        if len(chain) < MIN_CHAIN:
            continue
        inz = (track >= ZONE[0]) & (track <= ZONE[1])
        kr, kd = peaks(run["p"], run["use"], run["rng"], k)
        vis = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)[0].any(axis=1)
        miss = inz & ~vis
        # 連続する見えないフレームの長さ
        lens = []
        n = 0
        for m in np.r_[miss, False]:
            if m:
                n += 1
            elif n:
                lens.append(n)
                n = 0
        out.setdefault(f"{run['mode']} lane{run['lane']}", []).extend(lens)
    print(f"\n見えないフレームの続く長さ（区間 R {ZONE[0]:g}〜{ZONE[1]:g} m、K={k}、データの筋が区間にある時間）")
    for g, lens in out.items():
        lens = np.array(lens)
        tot = lens.sum()
        print(f"    {g}: 見えないフレーム {tot}。そのうち 1〜2 フレームの途切れ {lens[lens <= 2].sum() / max(tot, 1):.2f}、"
              f"3〜5 フレーム {lens[(lens >= 3) & (lens <= 5)].sum() / max(tot, 1):.2f}、"
              f"6 フレーム（1.2 s）以上 {lens[lens >= 6].sum() / max(tot, 1):.2f}")


def gap_structure(zone=(37.5, 49.0), k=3):
    """見えないフレームが「筋の両端」にあるか「見えている期間の途中」にあるかを分け、途中の途切れの位置を走行をまたいで比べる。

    両端にまとまり、途中は毎フレーム見えているなら、見えないのは範囲外（ビームの外など）で説明がつく。
    途中の途切れが走行をまたいで同じ距離に集まるなら、決まった場所の遮蔽（木など）を疑う
    """
    global ZONE
    ZONE = zone
    runs = load_all()
    groups = {}
    print(f"見えないフレームの位置（区間 R {ZONE[0]:g}〜{ZONE[1]:g} m、K={k}、既定の数値、データの筋が区間にある時間）")
    print("  見えている期間 = 区間の中で最初に見えたフレームから最後に見えたフレームまで。その外の見えないフレームが「両端」")
    print("  走行: 区間フレーム / 見えている / 途中の見えない / 両端の見えない（うち筋を延ばした部分）、途中の途切れの回数")
    for run in runs:
        kr, kd = peaks(run["p"], run["use"], run["rng"], 1)
        ok, rr = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)
        track, chain = data_track(ok, rr, run["t"], run["mode"], run["sign"])
        if len(chain) < MIN_CHAIN:
            continue
        t = run["t"]
        inz = (track >= ZONE[0]) & (track <= ZONE[1])
        ext = (t < chain[0][0]) | (t > chain[-1][0])               # 筋を延ばした部分（等速の仮定）
        kr, kd = peaks(run["p"], run["use"], run["rng"], k)
        vis = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)[0].any(axis=1)
        v = np.flatnonzero(vis & inz)
        core = np.zeros(len(t), bool)
        if len(v):
            core[v[0]:v[-1] + 1] = True
        core &= inz
        interior = core & ~vis
        edge = inz & ~core
        # 途中の途切れの回数（連続する見えないフレームを 1 回と数える）
        n_gap = int(np.sum(interior[1:] & ~interior[:-1]) + interior[0])
        g = f"{run['mode']} lane{run['lane']}"
        a = groups.setdefault(g, dict(n=0, vis=0, inter=0, edge=0, edge_ext=0, gaps=0, core=0, r_inter=[], r_core=[], runs=[]))
        a["n"] += int(inz.sum())
        a["vis"] += int((vis & inz).sum())
        a["inter"] += int(interior.sum())
        a["edge"] += int(edge.sum())
        a["edge_ext"] += int((edge & ext).sum())
        a["gaps"] += n_gap
        a["core"] += int(core.sum())
        a["r_inter"].append(track[interior])
        a["r_core"].append(track[core])
        a["runs"].append(run["n"])
        print(f"    No.{run['n']:2d} {g} {run['dr']}: {int(inz.sum()):3d} / {int((vis & inz).sum()):3d} / {int(interior.sum()):3d} / "
              f"{int(edge.sum()):3d}（{int((edge & ext).sum()):3d}）、途切れ {n_gap} 回")

    print("\n  種類ごと: 区間フレームに対する割合（見えている / 途中の見えない / 両端の見えない（うち筋を延ばした部分））、"
          "見えている期間の中で見えている割合、途切れ 1 回あたりの長さ")
    for g, a in groups.items():
        print(f"    {g:11}（{len(a['runs'])} 本）: {a['vis'] / a['n']:.2f} / {a['inter'] / a['n']:.2f} / "
              f"{a['edge'] / a['n']:.2f}（{a['edge_ext'] / a['n']:.2f}）、期間内 {1 - a['inter'] / max(a['core'], 1):.2f}、"
              f"途切れ {a['gaps']} 回・平均 {a['inter'] / max(a['gaps'], 1) * FRAME_S:.1f} s")

    # 途中の途切れが、走行をまたいで同じ距離に集まるか（2 m ごと）。見えている期間のフレーム数で割った途切れの割合
    edges = np.arange(ZONE[0], ZONE[1] + 2, 2.0)
    print("\n  見えている期間の中で、筋の距離ごとの見えない割合（2 m ごと。かっこは期間内のフレーム数）")
    print("    " + f"{'':11}" + "".join(f"{f'{lo:.1f}-{lo + 2:.1f}':>13}" for lo in edges[:-1]))
    for g, a in groups.items():
        ri = np.concatenate(a["r_inter"])
        rc = np.concatenate(a["r_core"])
        hi, _ = np.histogram(ri, edges)
        hc, _ = np.histogram(rc, edges)
        print("    " + f"{g:11}" + "".join(f"{'-' if c == 0 else f'{i / c:.2f}（{c}）':>13}" for i, c in zip(hi, hc)))


def visible_spans(zone=(37.5, 49.0), k=3, near_m=1.7):
    """走行ごとに、目標が何フレーム目から何フレーム目まで見えていて、その間のどのフレームで途切れたかを出す。

    等速の仮定は使わない。目標の筋 = K=1 で条件を満たした点の最長の鎖を、点の間だけ直線でつないだもの（両端の外には延ばさない）。
    「見えた」= 筋から near_m 以内に、上位 k 個の反射のうち条件を満たしたものがあるフレーム。
    見えている期間 = 最初に見えたフレームから最後に見えたフレームまで。その外は「見えていない」とだけ扱う
    """
    global ZONE
    ZONE = zone
    runs = load_all()
    rows = []
    print(f"見えている期間と途切れ（区間 R {ZONE[0]:g}〜{ZONE[1]:g} m、上位 {k} 個、筋から {near_m:g} m 以内）。フレームは 0 始まり、0.2 s ごと")
    print("  走行: 見えている期間（フレーム、R）、期間のフレーム数、途切れたフレーム数、途切れたフレーム")
    for run in runs:
        kr, kd = peaks(run["p"], run["use"], run["rng"], 1)
        ok, rr = judge_nodes(kr, kd, run["rng"], run["vel"], run["t"], run["mode"], run["sign"], DEFAULT)
        _, chain = data_track(ok, rr, run["t"], run["mode"], run["sign"])
        g = f"{run['mode']} lane{run['lane']}"
        if len(chain) < MIN_CHAIN:
            print(f"    No.{run['n']:2d} {g} {run['dr']}: 筋が取れない（鎖 {len(chain)} 点）")
            rows.append(dict(run=run, vis=None))
            continue
        t = run["t"]
        ct = np.array(sorted({c[0] for c in chain}))
        cr = np.array([np.mean([c[1] for c in chain if c[0] == x]) for x in ct])
        track = np.full(len(t), np.nan)
        inside = (t >= ct[0]) & (t <= ct[-1])
        track[inside] = np.interp(t[inside], ct, cr)               # 点の間だけ。両端の外は nan
        kr, kd = peaks(run["p"], run["use"], run["rng"], k)
        okk, rrk = judge_nodes(kr, kd, run["rng"], run["vel"], t, run["mode"], run["sign"], DEFAULT)
        vis = (okk & (np.abs(rrk - track[:, None]) <= near_m)).any(axis=1)   # (F,) nan は False
        f = np.flatnonzero(vis)
        span = np.zeros(len(t), bool)
        span[f[0]:f[-1] + 1] = True
        drop = span & ~vis
        # 途切れたフレームを、続く塊ごとに「a-b」で書く
        blocks = []
        for i in np.flatnonzero(drop):
            if blocks and blocks[-1][1] == i - 1:
                blocks[-1][1] = i
            else:
                blocks.append([i, i])
        txt = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in blocks) or "なし"
        print(f"    No.{run['n']:2d} {g} {run['dr']}: {f[0]:3d}〜{f[-1]:3d}（R {track[f[0]]:.1f} → {track[f[-1]]:.1f} m）、"
              f"{int(span.sum()):3d} フレーム、途切れ {int(drop.sum()):2d}（{len(blocks)} 回）: {txt}")
        rows.append(dict(run=run, vis=vis, span=span, drop=drop, track=track, blocks=blocks))

    print("\n  種類ごと: 筋が取れた走行、期間のフレーム数の合計、そのうち見えた割合、途切れの回数・1 回の平均の長さ・最長")
    for mode in ("walk", "bike"):
        for lane in (1, 2):
            xs = [x for x in rows if x["run"]["mode"] == mode and x["run"]["lane"] == lane]
            ok_ = [x for x in xs if x["vis"] is not None]
            if not ok_:
                print(f"    {mode} lane{lane}: 筋が取れた走行 0 / {len(xs)}")
                continue
            n = sum(int(x["span"].sum()) for x in ok_)
            d = sum(int(x["drop"].sum()) for x in ok_)
            lens = [b - a + 1 for x in ok_ for a, b in x["blocks"]]
            print(f"    {mode} lane{lane}: {len(ok_)} / {len(xs)} 本、{n} フレーム、見えた {1 - d / n:.2f}、"
                  f"途切れ {len(lens)} 回・平均 {np.mean(lens) if lens else 0:.1f} フレーム・最長 {max(lens) if lens else 0} フレーム")

    # 図: 走行ごとのフレームの帯（見えた / 期間の中の途切れ / 期間の外）
    ok_rows = [x for x in rows if x["vis"] is not None]
    img = np.zeros((len(ok_rows), 150))
    for i, x in enumerate(ok_rows):
        img[i] = np.where(x["vis"], 2, np.where(x["drop"], 1, 0))
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(ok_rows) + 1.2), layout="constrained")
    ax.imshow(img, aspect="auto", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(["#eeeeee", "tab:red", "tab:green"]), vmin=-0.5, vmax=2.5)
    ax.set_yticks(range(len(ok_rows)))
    ax.set_yticklabels([f"No.{x['run']['n']} {x['run']['mode']} L{x['run']['lane']} {x['run']['dr']}" for x in ok_rows], fontsize=8)
    ax.set_xlabel("frame (0.2 s)")
    ax.set_title(f"R {ZONE[0]:g}-{ZONE[1]:g} m. Green: visible, red: dropped inside the visible span, grey: outside the span",
                 fontsize=10)
    o = ROOT / "figures" / "visible_spans.png"
    fig.savefig(o, dpi=110)
    print(f"\n-> {o.relative_to(ROOT)}")
    return rows


def drop_positions(zone=(37.5, 49.0)):
    """途切れたフレームで目標がどの距離にいたかを、走行をまたいで並べる（visible_spans と同じ判定）。

    同じ距離に集まれば、その場所に決まった原因（木の遮蔽、決まった反射への埋もれ）があると疑える。
    レーンで横の位置が違うので、斜距離 R に加えて走路沿いの位置 X = sqrt(R^2 - d^2) でも並べる
    （木は走路沿いの決まった位置にあるので、遮蔽ならレーンが違っても X でそろうはず）。d は plot_lane_truth.lane_range
    """
    from plot_lane_truth import lane_range
    rows = [x for x in visible_spans(zone) if x["vis"] is not None]
    # 決まった距離の反射の線: 上位 3 個の反射が、走行の半分以上のフレームで出る距離ビン（全走行で平均）
    occ = []
    for x in rows:
        run = x["run"]
        kr, _ = peaks(run["p"], run["use"], run["rng"], 3)
        h = np.zeros(len(run["rng"]))
        for f in range(len(kr)):
            h[np.unique(kr[f][kr[f] >= 0])] += 1
        occ.append(h / len(kr))
    occ = np.mean(occ, axis=0)
    rng = rows[0]["run"]["rng"]
    lines = rng[(occ >= 0.5) & (rng >= ZONE[0]) & (rng <= ZONE[1])]
    print(f"\n途切れたフレームの位置（区間 R {ZONE[0]:g}〜{ZONE[1]:g} m）")
    print("  決まった距離の反射の線（上位 3 個の反射が半分以上のフレームで出る距離）: "
          + ", ".join(f"{r:.1f} m（{occ[np.argmin(np.abs(rng - r))]:.2f}）" for r in lines))
    edges = np.arange(ZONE[0], ZONE[1] + 1e-9, 1.0)
    print("  1 m ごと: 見えている期間の中で、目標がその距離にいたフレームのうち途切れた割合（途切れ / 期間内のフレーム）")
    print("    " + f"{'':24}" + "".join(f"{lo:>8.1f}" for lo in edges[:-1]))
    for x in rows:
        run = x["run"]
        r = x["track"]
        hd, _ = np.histogram(r[x["drop"]], edges)
        hs, _ = np.histogram(r[x["span"]], edges)
        cells = "".join(f"{'':>8}" if c == 0 else f"{f'{d}/{c}':>8}" for d, c in zip(hd, hs))
        print(f"    No.{run['n']:2d} {run['mode']} L{run['lane']} {run['dr']:3}{'':9}" + cells)

    # 図: 走行ごとに、見えている期間の距離の範囲（緑の線）と途切れた位置（赤）。左が R、右が走路沿いの X
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.36 * len(rows) + 1.6), sharey=True, layout="constrained")
    for i, x in enumerate(rows):
        run = x["run"]
        d = lane_range(run["lane"])[0]
        for ax, conv in zip(axes, (lambda r: r, lambda r: np.sqrt(np.maximum(r ** 2 - d ** 2, 0)))):
            rs = conv(x["track"][x["span"]])
            ax.plot([np.nanmin(rs), np.nanmax(rs)], [i, i], "-", color="tab:green", lw=5, alpha=0.5, solid_capstyle="butt")
            rd = conv(x["track"][x["drop"]])
            ax.plot(rd, np.full(len(rd), i), "|", color="tab:red", ms=12, mew=2)
    for r in lines:
        axes[0].axvline(r, color="0.4", lw=0.8, ls="--")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"No.{x['run']['n']} {x['run']['mode']} L{x['run']['lane']} {x['run']['dr']}" for x in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("slant range R [m]  (dashed: fixed reflections seen in >= half of frames)")
    axes[1].set_xlabel("position along the road X = sqrt(R^2 - d^2) [m]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Where the target dropped out (red) inside its visible span (green), R {ZONE[0]:g}-{ZONE[1]:g} m, night 10/1",
                 fontsize=10)
    o = ROOT / "figures" / "drop_positions.png"
    fig.savefig(o, dpi=110)
    print(f"-> {o.relative_to(ROOT)}")
