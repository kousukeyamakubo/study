# README の S（定性版）: 無人背景を引く処理を、MTI の代わりとして試す。窓 開の組（背景 15:00:40 → 自転車 15:09:39）。
#
# MTI（チャープ平均を引く）は N=1 で結果がゼロになり使えない。N を減らす比較では、フレームをまたぐ処理が要るので、
# 無人背景を引く方式が N=16 と N=1 で筋を残せるかを見る。並べるもの（どれもチャープ × 8ch の電力和、ドップラーは使わない）:
#   MTI なし / MTI あり（N=1 では 0）/ 背景差分・位相を戻さない / 背景差分・位相を戻す / フレーム差（1 つ前のフレームを引く）
# フレーム差は背景を撮らなくてよく、背景との時間の差が 0.2 s で済む。背景差分の引き残しが時間の差で増えるかを見る対照
#   指数平均の背景（時定数 EMA_TAU_S）: 過去のフレームの指数平均を背景として引く。固定の背景（揺れに弱い）と
#   フレーム差（止まった目標が消える）の中間。時定数はこの 1 本に合わせず、先に決め打ちした（README S）
# 位相を戻すのは、10/1 夜で静止物の反射に全距離共通の位相の段差（30〜90 mrad）があり、戻さないと MTI に負けたため
# （../1001/README.md B）。10/12 の 4 本にも段差があった（check_run_1013.py、55〜122 mrad）。
#
# 背景と走行の間は約 9 分空いている（直前の背景ではない）。その間の変化は引き残しとして出る。
# 背景は無人背景の t >= 5 s のフレームとチャープの複素平均 (TX, RX, R)。静止物は 1 フレームの中でチャープによらないので、
# チャープ方向にも平均してよい。N=1 では先頭のチャープだけを使う（時分割 ISAC で「先頭から N 本」と同じ間引き方）。
#
# 色の範囲は図ごとに「目標が来ない帯 R 55〜62 m の中央値」から 25 dB にする。処理によって値の大きさが違うので、
# 同じ絶対値の範囲にすると片方が潰れる。雑音の床からの見え方をそろえて比べる。
#
# 使い方:
#   python -X utf8 bg_subtract.py > bg_subtract_log.txt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1011"))
sys.path.insert(0, str(HERE.parent / "collect"))
from range_profile import chirp_profiles, setup_font_ja                         # noqa: E402
from check_run_1013 import export                                              # noqa: E402
from axes_maps import R_HI, R_LO, res_bar                                      # noqa: E402
from bike_on_line import fit_track                                             # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
BG, RUN = "150040", "150939"
SETTLE_S = 5.0
STATIC_R = (8.0, 60.0)          # 共通位相を測る静止物の距離帯（../1001/background_recheck.py と同じ。窓枠を除く）
STATIC_TOP = 0.2                # その帯で背景の電力が上位 2 割のビンを使う（強い静止物ほど位相がよく決まる）
NOISE_BAND = (55.0, 62.0)
SPAN_DB = 25.0
FRAME_S = 0.2
EMA_TAU_S = 1.0                 # 自転車が 1 ビンにいる約 0.5 s より長く、木の揺れ（数 s）には追いつく程度として決め打ち
BANDS = {"R 13-20 m": (13.0, 20.0), "R 20-28 m": (20.0, 28.0), "R 30-40 m": (30.0, 40.0), "R 40-49 m": (40.0, 49.0)}
METHODS = ["nomti", "mti", "bg", "bg_phase", "diff", "ema"]
METHOD_JA = {"nomti": "MTI なし", "mti": "MTI あり（チャープ平均を引く）",
             "bg": "背景差分・位相を戻さない", "bg_phase": "背景差分・位相を戻す",
             "diff": "フレーム差（1 つ前を引く）",
             "ema": f"指数平均の背景（時定数 {EMA_TAU_S:.0f} s）"}


def load_chirps(stem):
    """MTI なしの RD から、チャープごとの距離プロファイル (F, TX, RX, C, R) と軸"""
    d = np.load(export(DATA / f"atlas_log_20261012_{stem}.dat", DATA, True))
    return chirp_profiles(d["rd"]), d["range_m"], d["t_rel_s"]


def frame_phase(x, ref, st):
    """フレームごとの、背景の静止物 ref (TX, RX, R) からの共通位相 [rad]。(F,)

    ../1001/background_recheck.py common_phase と同じ考え方（強い静止物の複素内積を全部足す）だが、
    基準を走行自身の中央値ではなく無人背景にする。走行と背景のずれ（9 分の間の段差）もここで戻すため"""
    s = x.mean(axis=3)                                                  # (F, TX, RX, R) チャープ平均
    z = (s[..., st] * np.conj(ref[..., st])).sum(axis=(1, 2, 3))        # (F,)
    return np.angle(z)


def ema_residual(x):
    """過去のフレームだけから作った指数平均の背景を引く。(F, TX, RX, n, R)

    背景は B_f = (1 − α) B_{f−1} + α s_{f−1}（s はフレームのチャープ平均）、残差は x_f − B_f。
    今のフレームを背景に入れないので、目標が来た瞬間に自分自身を引いてしまうことはない。
    最初のフレームは背景がないので、そのフレーム自身で始める（最初の数フレームは背景が育っていない）"""
    alpha = 1.0 - np.exp(-FRAME_S / EMA_TAU_S)
    s = x.mean(axis=3)                                                  # (F, TX, RX, R)
    b = s[0].copy()
    out = np.empty_like(x)
    for f in range(x.shape[0]):
        out[f] = x[f] - b[:, :, None, :]
        b = (1.0 - alpha) * b + alpha * s[f]
    return out


def to_db(x):
    return 10 * np.log10(np.sum(np.abs(x) ** 2, axis=(1, 2, 3)) + 1e-30)   # (F, R)


def maps_for_n(xb, xr, rng, tb, n):
    """先頭 n チャープで 4 通りの距離−時間 (F, R) [dB]"""
    xb, xr = xb[:, :, :, :n, :], xr[:, :, :, :n, :]                    # (F, TX, RX, n, R)
    keep = tb >= SETTLE_S
    ref = xb[keep].mean(axis=(0, 3))                                    # (TX, RX, R) 位相を戻す前の背景
    p_ref = 10 * np.log10(np.sum(np.abs(ref) ** 2, axis=(0, 1)) + 1e-30)
    band = (rng >= STATIC_R[0]) & (rng <= STATIC_R[1])
    st = band & (p_ref >= np.quantile(p_ref[band], 1 - STATIC_TOP))

    out = {"nomti": to_db(xr)}
    out["mti"] = to_db(xr - xr.mean(axis=3, keepdims=True)) if n > 1 else None
    out["bg"] = to_db(xr - ref[None, :, :, None, :])

    rot = lambda x: x * np.exp(-1j * frame_phase(x, ref, st))[:, None, None, None, None]
    xb_c, xr_c = rot(xb), rot(xr)
    ref_c = xb_c[keep].mean(axis=(0, 3))                                # (TX, RX, R) 位相を戻した背景
    out["bg_phase"] = to_db(xr_c - ref_c[None, :, :, None, :])
    # 先頭のフレームは 1 つ前がないので、2 番目のフレームの値で埋めて行数をそろえる
    d = to_db(xr[1:] - xr[:-1])                                         # (F-1, R)
    out["diff"] = np.vstack([d[:1], d])
    out["ema"] = to_db(ema_residual(xr))
    phi_r = frame_phase(xr, ref, st)
    return out, phi_r, int(st.sum())


def contrast(m, rng, t, a, b):
    """筋の上の強さ: 軌跡の ±1 ビンの最大 − その距離の時間中央値 [dB]、t 1.5〜13 s のフレームの中央値（帯ごと）"""
    med = np.median(m, axis=0)                                          # (R,)
    res = {}
    for name, (lo, hi) in BANDS.items():
        v = []
        for f in np.flatnonzero((t >= 1.5) & (t <= 13.0)):
            r0 = a * t[f] + b
            if not lo <= r0 < hi:
                continue
            k = np.flatnonzero(np.abs(rng - r0) <= 1.0)
            v.append((m[f, k] - med[k]).max())
        res[name] = np.median(v) if v else np.nan
    return res


def draw(ax, m, rng, t, title, label):
    keep = (rng >= R_LO) & (rng <= R_HI)
    nb = (rng >= NOISE_BAND[0]) & (rng <= NOISE_BAND[1])
    lo = np.median(m[:, nb])
    im = ax.pcolormesh(rng[keep], t, m[:, keep], shading="nearest", cmap="viridis", vmin=lo, vmax=lo + SPAN_DB)
    res_bar(ax, "range", label=label)
    ax.set_title(title, fontsize=10)
    return im


def main():
    setup_font_ja()
    FIG.mkdir(exist_ok=True)
    xb, rng, tb = load_chirps(BG)
    xr, _, tr = load_chirps(RUN)
    n_full = xr.shape[3]

    # 軌跡は MTI ありの N=16 で 1 回決め、全部の処理に同じものを使う（bike_on_line.py と同じ当てはめ）
    m16, phi16, nst = maps_for_n(xb, xr, rng, tb, n_full)
    a, b = fit_track({"rng": rng, "t": tr, "sum_mti": m16["mti"]}, "sum_mti")
    print(f"背景 {BG} → 走行 {RUN}。共通位相を測る静止物ビン {nst} 個。軌跡 R = {a:+.2f} t + {b:.1f}")
    print(f"走行の共通位相（背景からの回転）: 中央値 {np.degrees(np.median(phi16)):+.1f}°、"
          f"幅（最大 − 最小） {1e3 * np.ptp(phi16):.0f} mrad")

    rows = {n_full: m16, 1: maps_for_n(xb, xr, rng, tb, 1)[0]}
    fig, axes = plt.subplots(2, 6, figsize=(25, 9), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.3, wspace=0.1))
    for i, (n, ms) in enumerate(rows.items()):
        print(f"\n[N={n}] 引き残しの床（目標が来ない帯 R 55〜62 m からの、t ≥ 5 s の中央値）/ 筋の上の強さ（軌跡上 − 時間中央値）[dB]")
        print("処理".ljust(26) + "".join(k.rjust(20) for k in BANDS))
        for j, k in enumerate(METHODS):
            ax = axes[i, j]
            m = ms[k]
            if m is None:
                ax.text(0.5, 0.5, "N=1 ではチャープ平均を引くと\nすべて 0 になる", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(f"N={n}: {METHOD_JA[k]}", fontsize=10)
                print(METHOD_JA[k].ljust(26) + "（0 になる）".rjust(20))
                continue
            draw(ax, m, rng, tr, f"N={n}: {METHOD_JA[k]}", label=(i == 0 and j == 0))
            nb = (rng >= NOISE_BAND[0]) & (rng <= NOISE_BAND[1])
            noise = np.median(m[tr >= SETTLE_S][:, nb])
            c = contrast(m, rng, tr, a, b)
            cells = []
            for name, (lo, hi) in BANDS.items():
                kr = (rng >= lo) & (rng < hi)
                fl = np.median(m[tr >= SETTLE_S][:, kr]) - noise
                cells.append(f"{fl:+6.1f} / {c[name]:+5.1f}")
            print(METHOD_JA[k].ljust(26) + "".join(s.rjust(20) for s in cells))
    for ax in axes[:, 0]:
        ax.set_ylabel("時刻 [s]")
    for ax in axes[1]:
        ax.set_xlabel("距離 R [m]")
    fig.suptitle("10/12 窓 開・自転車 接近（15:09:39）。背景は 15:00:40 の無人 30 s。"
                 f"電力和（ドップラーなし）、色は各図の R 55〜62 m の中央値から {SPAN_DB:.0f} dB", fontsize=11)
    out = FIG / "bg_subtract.png"
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
