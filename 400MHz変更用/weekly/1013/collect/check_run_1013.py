# 週末の収録で、.dat を 1 本撮るごとにその場で確かめる道具。weekly/0928/check_run.py の後継。
#
# 0928 版は「尾根が追えたか」を見たが、今回の収録の問いは 3 つなので、それぞれに判定を 1 行で出す:
#   1. 近い端（R 22〜28 m）が見えたか（H1 の向きの切り分け）
#      自動追跡は別の筋を拾うので使わない。申告レーンの等速軌跡を総当たりする整合フィルタ
#      （weekly/1001/lane_visibility.py）を帯ごとに掛け、時刻反転の対照からの上乗せ [dB] を出す。
#      d（最接近距離）は 10/1 の幾何から出し、置き方を変えた分のずれは ±1.5 m の総当たりで吸収する
#   2. 窓の床: 動体ビン（MTI あり）の距離ごとの床。9/28 の開/閉の値と並べる
#   3. 背景の安定度: 全距離共通の位相の段差（background_recheck.py）と、室内（R 0.5〜5 m）の動き
#      段差があると複素の背景差分が使えない。室内の動きは操作者
#
# 判定の目安（collect/README.md）:
#   近い端: 3 dB 以上で「見えた」（10/1 夜の遠い帯でレーン1の自転車が 4.2 dB）。1.5〜3 dB は「弱い」
#   背景  : 録画開始 5 s 以降に 20 mrad を超える共通位相の跳びがあれば「段差あり」
#
# 使い方:
#   python check_run_1013.py data/raw/atlas_log_xxx.dat --lane 1 --dir dep
#   python check_run_1013.py data/raw/atlas_log_xxx.dat --bg            # 無人背景
#   python check_run_1013.py data/raw/atlas_log_xxx.dat --lane 1 --dir dep --d 22.5   # d を指定

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
W1001 = ROOT.parent.parent / "1001"
sys.path.insert(0, str(W1001.parent / "0727"))
sys.path.insert(0, str(W1001))
from atlas_ridge_track import moving_power_db                    # noqa: E402
from lane_visibility import matched_score                        # noqa: E402
from plot_lane_truth import lane_range                           # noqa: E402

from background_recheck import common_phase                      # noqa: E402

EXPORT = W1001.parent / "0727" / "atlas_export.py"
BANDS = {"near": (22.0, 28.0), "mid": (30.0, 40.0), "far": (40.0, 49.0)}
D_SPAN = np.arange(-1.5, 1.51, 0.5)        # d の総当たり幅 [m]
# 9/28 の無人背景の床（window_recheck.py、MTI あり動体ビンの距離ごと中央値）[dB]
FLOOR_REF = {"13-20": (16.3, 21.6), "20-28": (14.8, 17.1), "40-60": (14.1, 14.4)}
SETTLE_S = 5.0          # 録画開始直後は操作者が動く（10/1 夜 No.1, 31）。この間の段差は数えない
STEP_MRAD = 20.0
VISIBLE_DB, WEAK_DB = 3.0, 1.5


def export(dat, outdir, nomti):
    sub = outdir / ("export_nomti" if nomti else "export") / dat.stem
    npz = sub / f"{dat.stem}_rd.npz"
    if not npz.exists():
        cmd = [sys.executable, str(EXPORT), str(dat), "--outdir", str(sub)] + (["--no-mti"] if nomti else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return npz


def target_scores(ex, rng, t, d0, sign):
    """帯ごとに (軌跡上 − 時刻反転の対照) [dB] と、そのときの d。d は D_SPAN で総当たり"""
    out = {}
    for name, band in BANDS.items():
        on = [matched_score(ex, rng, t, d0 + dd, sign, band) for dd in D_SPAN]
        ctrl = max(matched_score(ex[::-1], rng, t, d0 + dd, sign, band) for dd in D_SPAN)
        i = int(np.argmax(on))
        out[name] = (on[i] - ctrl, d0 + D_SPAN[i])
    return out


def phase_steps(phi, t, w=5):
    """共通位相の段差の時刻 [s] のリスト。

    1 フレームの差では ±20 mrad 程度の揺れも拾うので、前後 w フレーム（1 s）の中央値の差で見る。
    STEP_MRAD を超えた区間ごとに、差が最大の時刻を 1 つ返す"""
    jump = np.zeros(len(phi))                                       # (F,) mrad
    for i in range(w, len(phi) - w + 1):
        jump[i] = abs(np.median(phi[i:i + w]) - np.median(phi[i - w:i])) * 1e3
    hit = (jump > STEP_MRAD) & (t >= SETTLE_S)
    out, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j < len(hit) and hit[j]:
                j += 1
            out.append(float(t[i + int(np.argmax(jump[i:j]))]))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--outdir", type=Path, default=ROOT / "data")
    ap.add_argument("--bg", action="store_true", help="無人背景（目標の判定をしない）")
    ap.add_argument("--lane", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--dir", choices=("dep", "app"), default="dep", help="離反 dep / 接近 app")
    ap.add_argument("--d", type=float, default=None, help="最接近距離 [m]。省略時は 10/1 の幾何から")
    args = ap.parse_args()

    npz = export(args.dat, args.outdir, nomti=False)
    npz_raw = export(args.dat, args.outdir, nomti=True)
    hdr = json.loads(str(np.load(npz)["header"]))
    pw, _, rng, t, v_ax, db, k = moving_power_db(npz)              # pw: (F, R), db: (F, D, R)
    ex = pw - np.median(pw, axis=0, keepdims=True)                  # (F, R) 床超過

    print(f"== {args.dat.name}  {len(t)} フレーム / {t[-1] + 0.2:.1f} s"
          + ("  [無人背景]" if args.bg else f"  [レーン{args.lane} {args.dir}]"))
    # atlas_export.py の --n-fft は 256 決め打ち。GUI でサンプル数を変えると黙って分解能が落ちる
    if hdr.get("n_sample", 256) != 256:
        print(f"  [注意] n_sample = {hdr.get('n_sample')}（256 でない）。--n-fft を合わせて再出力すること")

    # ---- 1. 近い端 ----
    sc = None
    if not args.bg:
        d0 = args.d if args.d is not None else lane_range(args.lane)[0]
        sign = -1.0 if args.dir == "app" else +1.0
        sc = target_scores(ex, rng, t, d0, sign)
        near = sc["near"][0]
        verdict = "見えた" if near >= VISIBLE_DB else ("弱い" if near >= WEAK_DB else "見えない")
        print(f"1. 近い端 R 22-28 m: {near:+.1f} dB → 【{verdict}】"
              f"（中 30-40 m {sc['mid'][0]:+.1f} / 遠 40-49 m {sc['far'][0]:+.1f} dB、d = {sc['near'][1]:.1f} m）")
        if sc["far"][0] < WEAK_DB:
            print("   [注意] 遠い帯でも弱い。走行そのもの（レーン・向き・開始の遅れ）を確かめる")

    # ---- 2. 窓の床 ----
    floor = np.median(db[:, k >= 1, :], axis=(0, 1))                # (R,)
    parts = []
    for key, (ref_open, ref_closed) in FLOOR_REF.items():
        lo, hi = map(float, key.split("-"))
        b = (rng >= lo) & (rng < hi)
        parts.append(f"R {key} m {np.median(floor[b]):.1f}（9/28 開 {ref_open} / 閉 {ref_closed}）")
    print("2. 床 [dB]: " + "  ".join(parts))

    # ---- 3. 背景の安定度 ----
    raw = np.load(npz_raw)
    dc = int(np.abs(raw["vel_ms"]).argmin())
    phi, g = common_phase(raw["rd"], raw["range_m"], dc)            # (F,), (F,)
    steps = phase_steps(phi, t)
    room = db[:, k >= 1, :][:, :, (rng >= 0.5) & (rng <= 5.0)].max(axis=(1, 2))   # (F,)
    moved = t[room > np.median(room) + 6.0]
    stab = "段差あり" if len(steps) else "安定"
    print(f"3. 背景: 共通位相の幅 {np.ptp(phi[t >= SETTLE_S]) * 1e3:.0f} mrad、"
          f"{SETTLE_S:.0f} s 以降の跳び > {STEP_MRAD:.0f} mrad: "
          + (", ".join(f"{s:.1f}" for s in steps[:6]) + " s" if steps else "なし")
          + f" → 【{stab}】")
    # 段差の原因の手がかり: 同じ時刻（±0.6 s）に室内が動いたか。10/1 夜は動いた回と動かない回が両方あった
    for s in steps[:6]:
        w = np.abs(t - s) <= 0.6
        print(f"   {s:.1f} s の段差: 前後の室内 {room[w].max() - np.median(room):+.1f} dB（中央値比）")
    if len(moved):
        late_moved = moved[moved >= SETTLE_S]
        print(f"   室内の動き（R 0.5-5 m が中央値 +6 dB 超）: {len(moved)} フレーム"
              + (f"、うち {SETTLE_S:.0f} s 以降 {len(late_moved)}（{late_moved[0]:.1f} s〜）→ 操作者は動かない"
                 if len(late_moved) else f"（すべて最初の {SETTLE_S:.0f} s）"))

    # ---- 図 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2, 1]}, layout="constrained")
    ax = axes[0]
    keep = rng <= 60
    im = ax.pcolormesh(rng[keep], t, ex[:, keep], shading="auto", cmap="viridis", vmin=0, vmax=20)
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, color="w", alpha=0.08 if name != "near" else 0.18)
    ax.axvline(BANDS["near"][0], color="r", lw=0.8, ls="--")
    ax.axvline(BANDS["near"][1], color="r", lw=0.8, ls="--")
    title = args.dat.stem
    if sc is not None:
        title += f"\nnear {sc['near'][0]:+.1f} / mid {sc['mid'][0]:+.1f} / far {sc['far'][0]:+.1f} dB"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("range R [m]")
    ax.set_ylabel("time [s]")
    fig.colorbar(im, ax=ax, label="moving power over floor [dB]")
    ax = axes[1]
    ax.plot(phi * 1e3, t, "C0")
    ax.axhspan(0, SETTLE_S, color="gray", alpha=0.15)
    ax.set_xlabel("common phase [mrad]", color="C0")
    ax.set_ylim(t[0], t[-1])
    ax2 = ax.twiny()
    ax2.plot(room, t, "C3", lw=0.7)
    ax2.set_xlabel("room motion R 0.5-5 m [dB]", color="C3")
    ax.set_title(f"background: {'STEP' if len(steps) else 'stable'}", fontsize=9)
    out = args.outdir / "check" / f"{args.dat.stem}_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"   図 -> {out}")


if __name__ == "__main__":
    main()
