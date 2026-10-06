# 10/1 夜の MTI なしデータで、「静止物の反射が 30 s の中で変わる」原因に当たりをつけ、
# MTI の代わりに背景を引く方式が成り立つかを試す。対象は夜の No.1, 19, 20, 31。
#
# 10/5 の予備分析: 複素 RD からファイル全体の時間中央値を引くと、R < 45 m で残差が MTI より
# 25〜30 dB 大きく目標が埋もれる。No.19 では t ≈ 21 s で全距離が一斉に段差状に変わる。
#
# 原因の候補と、ここでの見分け方:
#   (i)  操作者の動き     → 室内（R 0.5〜5 m）の動体電力（MTI あり）が同じ時刻に跳ねるか
#   (ii) レーダー側の揺らぎ → 全距離・全チャンネルで共通の位相（または振幅）の段差か
#        共通位相は、静止物の強いビン（DC ビン）の「基準からの位相回転」を全部足して出す。
#        レーダー（三脚）が視線方向に δ 動いても位相は 4πδ/λ だけ全距離に一様に乗るので、
#        内部の位相の跳びと三脚の微小な動き（λ = 12.5 mm なので 0.03 rad ≈ 30 µm）はここでは区別できない
#
# 試す背景の引き方（どれも DC ビンを含めて距離−時間の電力にする。MTI の利点は DC を捨てない点だけなので）:
#   MTI      : 現行（チャープ平均を引く、DC ビン除外）。比較の基準
#   complex  : 複素 RD − 時間中央値（10/5 の予備分析と同じ）
#   complex+φ: フレームごとの共通位相を戻してから 複素 RD − 時間中央値
#   power    : 電力領域。セルごとに dB − 時間中央値 dB（背景で割るのと同じ）
# 評価は lane_visibility.py と同じ整合フィルタ（申告レーンの等速軌跡の総当たり、時刻反転の対照からの上乗せ）。
# 背景はファイル自身の時間中央値で代用する（夜は無人背景を撮り忘れた。weekly/1001/README.md）。
#
# 使い方:
#   python background_recheck.py

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
from analyze_night import TAGS, condition, npz_path              # noqa: E402
from atlas_ridge_track import moving_power_db                    # noqa: E402
from lane_visibility import matched_score                        # noqa: E402
from plot_lane_truth import lane_range                           # noqa: E402

RUNS = [1, 19, 20, 31]
NEAR_ROOM = (0.5, 5.0)       # 室内（窓枠・三脚・操作者）の距離帯 [m]
STATIC_MIN_DB = 45.0         # 共通位相を測るのに使う静止物ビンの下限（時間中央値の電力）
STATIC_R = (8.0, 60.0)       # 同上の距離帯。窓枠（< 2 m）は除く。窓自体が動く場合と分けるため
BANDS = {"near 21-30 m": (21.0, 30.0), "mid 30-40 m": (30.0, 40.0), "far 40-49 m": (40.0, 49.0)}


def complex_median(x, axis=0):
    """複素数の中央値（実部・虚部を別に）。目標が通る数フレームに引きずられないように平均は使わない"""
    return np.median(x.real, axis=axis) + 1j * np.median(x.imag, axis=axis)


def common_phase(rd, rng, dc):
    """フレームごとの共通位相 φ(f) [rad] と共通利得 g(f) [dB]。(F,), (F,)

    静止物の強いビンで、各フレームの DC ビンを時間中央値と比べる。全チャンネル・全ビンの
    複素内積を 1 つに足すので、強い静止物ほど重く効く"""
    s = rd[:, :, :, dc, :]                                         # (F, TX, RX, R)
    ref = complex_median(s)                                        # (TX, RX, R)
    p_ref = 20 * np.log10(np.abs(ref).sum(axis=(0, 1)) + 1e-12)   # (R,)
    st = (rng >= STATIC_R[0]) & (rng <= STATIC_R[1]) & (p_ref > STATIC_MIN_DB)
    z = (s[..., st] * np.conj(ref[..., st])).sum(axis=(1, 2, 3))   # (F,)
    phi = np.unwrap(np.angle(z))
    phi -= np.median(phi)
    g = 20 * np.log10(np.abs(s[..., st]).sum(axis=(1, 2)) / np.abs(ref[..., st]).sum(axis=(0, 1)))  # (F, R')
    return phi, np.median(g, axis=1)


def room_motion_db(tag):
    """室内の動体電力（MTI あり、DC 除外、R 0.5〜5 m の最大）。操作者が動くと跳ねる。(F,)"""
    _, _, rng, _, _, db, k = moving_power_db(npz_path(tag))         # db: (F, D, R)
    b = (rng >= NEAR_ROOM[0]) & (rng <= NEAR_ROOM[1])
    return db[:, k >= 1, :][:, :, b].max(axis=(1, 2))


def excess(pw):
    """距離ごとの時間中央値を床として引く。(F, R)"""
    return pw - np.median(pw, axis=0, keepdims=True)


def maps(tag):
    """4 通りの背景の引き方の距離−時間の床超過 dB。dict of (F, R)、ほかに軸と共通位相"""
    pw_mti, _, rng, t, *_ = moving_power_db(npz_path(tag))        # (F, R)
    d = np.load(npz_path(tag, nomti=True))
    rd = d["rd"]                                                   # (F, TX, RX, D, R)
    dc = int(np.abs(d["vel_ms"]).argmin())
    phi, g = common_phase(rd, rng, dc)

    def to_pw(x):                                                  # 8ch 振幅加算 → 全ドップラーの最大
        return 20 * np.log10(np.abs(x).sum(axis=(1, 2)).max(axis=1) + 1e-12)   # (F, R)

    rd_c = rd * np.exp(-1j * phi)[:, None, None, None, None]       # 共通位相を戻す
    db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)        # (F, D, R)
    out = {
        "MTI (current)": excess(pw_mti),
        "complex bg-sub": excess(to_pw(rd - complex_median(rd))),
        "complex bg-sub + phase fix": excess(to_pw(rd_c - complex_median(rd_c))),
        "power bg-norm": excess((db - np.median(db, axis=0, keepdims=True)).max(axis=1)),
    }
    return out, rng, t, phi, g


def main():
    res, keep_maps = {}, {}
    print("(1) 共通位相の段差と室内の動き")
    for n in RUNS:
        tag = TAGS[n - 1]
        mp, rng, t, phi, g = maps(tag)
        room = room_motion_db(tag)
        dphi = np.abs(np.diff(phi))
        i = int(dphi.argmax())
        # 段差の前後 ±1 s に室内の動きがあったか
        w = slice(max(i - 4, 0), i + 6)
        print(f"    No.{n:2d}: 共通位相の幅 {np.ptp(phi):.3f} rad、最大の跳び {dphi[i]:.3f} rad @ t={t[i + 1]:.1f} s"
              f"（その前後の室内 {room[w].max():.1f} dB / 収録中央値 {np.median(room):.1f} dB）、"
              f"共通利得の揺れ {g.std():.2f} dB")
        mode, lane, dr = condition(n)
        d0 = lane_range(lane)[0]
        sign = -1.0 if dr == "app" else +1.0
        res[n] = {}
        for name, ex in mp.items():
            res[n][name] = {b: matched_score(ex, rng, t, d0, sign, band)
                            - matched_score(ex[::-1], rng, t, d0, sign, band) for b, band in BANDS.items()}
        keep_maps[n] = (mp, rng, t, phi, room)

    print("\n(2) 軌跡上の目標強度（時刻反転の対照からの上乗せ [dB]、4 走行の中央値）")
    names = list(next(iter(res.values())).keys())
    print(f"{'':28}" + "".join(f"{b:>15}" for b in BANDS))
    for name in names:
        print(f"{name:28}" + "".join(f"{np.median([res[n][name][b] for n in RUNS]):15.2f}" for b in BANDS))
    print("\n    走行ごと")
    for n in RUNS:
        mode, lane, dr = condition(n)
        print(f"    No.{n:2d} {mode} lane{lane} {dr}: " + " | ".join(
            f"{name.split(' (')[0]} " + "/".join(f"{res[n][name][b]:.1f}" for b in BANDS) for name in names))

    # ---- 図: 上段 = No.19 の共通位相と室内の動き、下段 = No.19 の 4 方式の距離−時間 ----
    n = 19
    mp, rng, t, phi, room = keep_maps[n]
    fig = plt.figure(figsize=(14, 8.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, phi * 1e3, "C0")
    ax.set_ylabel("common phase [mrad]", color="C0")
    ax.set_xlabel("time [s]")
    ax2 = ax.twinx()
    ax2.plot(t, room, "C3", lw=0.8)
    ax2.set_ylabel("room motion R 0.5-5 m [dB]", color="C3")
    ax.set_title(f"No.{n}: step in phase common to all ranges/channels, room is quiet", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2:])
    for m in RUNS:
        _, _, tt, ph, rm = keep_maps[m]
        ax.plot(tt, ph * 1e3, lw=1, label=f"No.{m}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("common phase [mrad]")
    ax.set_title("No.1 / 31: swings in first 5 s coincide with room motion (operator)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    keep = rng <= 60
    for j, (name, ex) in enumerate(mp.items()):
        ax = fig.add_subplot(gs[1, j])
        im = ax.pcolormesh(rng[keep], t, ex[:, keep], shading="auto", cmap="viridis", vmin=0, vmax=20)
        sc = "/".join(f"{res[n][name][b]:.1f}" for b in BANDS)
        ax.set_title(f"{name}\non-path near/mid/far: {sc} dB", fontsize=9)
        ax.set_xlabel("range R [m]")
        if j == 0:
            ax.set_ylabel("time [s]")
    fig.colorbar(im, ax=fig.axes[-4:], label="power over per-range floor [dB]")
    mode, lane, dr = condition(n)
    fig.suptitle(f"Background removal on night No.{n} ({mode}, lane {lane}, {dr})", fontsize=11)
    out = ROOT / "figures" / "background_recheck.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=105)
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
