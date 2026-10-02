# 10/1 夜の収録（36本 + 追加1往復）を定量化する。
#
# 問い: 夜のデータからレーンごとの横方向離隔 d̂ が出せるか（主張(2) の前提）。
# 答えは「出せない」で、本スクリプトはその根拠を3つの数字にする:
#   (A) 手前（最接近点付近）で目標の強度が R^4 則の期待より何 dB 足りないか。
#       MTI あり / なし（フレーム間差分）の両方で出す
#   (A') 手前の区間に目標の反射が残っているかを、外挿した軌跡上で対照と比べる。
#       MTI を外しても戻らないこと（MTI が原因でないこと）をここで示す
#   (B) 観測した各走行で、d を動かしたときの当てはめ残差がどれだけ平らか
#   (C) 実測の幾何で、「見えている最小距離」ごとに d̂ の標準偏差（Cramér–Rao 下限）を出す。
#       今の見え方では d̂ がレーン間差より大きくばらつき、最接近点まで見えれば決まることを示す
#
# 使い方:
#   python analyze_night.py
#   （事前に data/export/ と data/export_nomti/ に atlas_export.py の出力が要る）

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "0727"))
from atlas_ridge_track import build_ridges, moving_power_db   # noqa: E402
from analyze_1001 import profile_d                             # noqa: E402

# 収録開始時刻順。No.1–36 が本測定、37–38 は「一応取っておく」追加の1往復（README §5）
TAGS = ["203951", "204055", "204204", "204304", "204425", "204512", "204604", "204653",
        "204819", "204910", "205002", "205053", "205144", "205235", "205327", "205419",
        "205507", "205603", "205755", "205839", "205930", "210134", "210302", "210352",
        "210514", "210630", "210715", "210809", "210858", "211050", "211133", "211235",
        "211319", "211407", "211450", "211545", "211637", "211733"]
N_MAIN = 36


def condition(n):
    """No.（1始まり）→ (手段, レーン, 向き)。徒歩→自転車、各レーン3往復、往復は接近が先"""
    mode = "walk" if n <= 18 else "bike"
    lane = ((n - 1) % 18) // 6 + 1
    return mode, lane, ("app" if n % 2 else "dep")


# コーンの斜距離 [m]（レーダー位置からレーザー距離計で実測）
CONES = {"C1": 23.282, "C2": 20.943, "C3": 26.774, "C4": 33.698, "C5": 49.405, "C6": 48.404}
L_C1_C5 = 45.818          # C1→C5 の地上実測 [m]。どちらも左（建物側）の縁


def foot_of_perpendicular():
    """左の縁の C1, C5 と区間長から、垂線の足の位置 s0（C1 から奥向き）と横方向離隔 d を出す。

    R^2 = d^2 + (s - s0)^2 を s = 0 (C1), s = L (C5) で連立すると s0 が線形に解ける。
    C1, C5 が同じ縁の一直線上にある前提。C5 の斜距離 0.5 m の誤差で s0 は約 0.5 m 動く
    """
    r1, r5, L = CONES["C1"], CONES["C5"], L_C1_C5
    s0 = (L**2 - (r5**2 - r1**2)) / (2 * L)
    return s0, float(np.sqrt(r1**2 - s0**2))


S0, D_LEFT = foot_of_perpendicular()
NOISE_R = (55.0, 62.0)    # 走路（R ≤ 49.4 m）より遠く目標が来ない帯。「雑音だけ」の基準に使う


def npz_path(tag, nomti=False):
    s = f"atlas_log_20261001_{tag}"
    return ROOT / ("data/export_nomti" if nomti else "data/export") / s / f"{s}_rd.npz"


# ------------------------------------------------------------------
# (A) 距離帯ごとの目標強度
# ------------------------------------------------------------------

def peak_excess_mti(tag):
    """現行の処理（チャープ平均 MTI + DC ビン除外）での、距離ビンごとの「時間方向の最大の床超過 dB」

    被験者は毎回 走路の全域（R 21〜49 m）を通るので、各距離ビンで時間方向の最大を取れば
    尾根の追跡に頼らずに「そこを通ったときの強さ」が取れる。途切れた区間でも値が出る
    """
    pw, _, rng, *_ = moving_power_db(npz_path(tag))             # (F, R)
    ex = pw - np.median(pw, axis=0, keepdims=True)              # (F, R) 床からの超過
    return rng, ex.max(axis=0)                                  # (R,), (R,)


def peak_excess_framediff(tag):
    """MTI なしの RD に、フレーム間差分（0.2 s 前との差）を掛けた場合の同じ量。

    チャープ平均 MTI は 1フレーム内で速度 0 の成分を消すので、最接近点（v_r → 0）の目標も
    消しうる。フレーム間差分は 0.2 s の間に位相が回れば残るので、DC ビンも含めて見られる
    """
    d = np.load(npz_path(tag, nomti=True))
    rd = d["rd"]                                                 # (F, TX, RX, D, R)
    diff = np.abs(np.diff(rd, axis=0)).sum(axis=(1, 2))          # (F-1, D, R)
    pw = 20 * np.log10(diff.max(axis=1) + 1e-12)                 # (F-1, R) DC ビンも含めた最大
    ex = pw - np.median(pw, axis=0, keepdims=True)
    return d["range_m"], ex.max(axis=0)


def part_a():
    runs = range(1, N_MAIN + 1)
    rng, _ = peak_excess_mti(TAGS[0])
    mti = np.array([peak_excess_mti(TAGS[n - 1])[1] for n in runs])          # (Run, R)
    fdf = np.array([peak_excess_framediff(TAGS[n - 1])[1] for n in runs])    # (Run, R)
    mti_m, fdf_m = np.median(mti, axis=0), np.median(fdf, axis=0)            # (R,)

    noise = (rng >= NOISE_R[0]) & (rng <= NOISE_R[1])
    # R^4 則の期待。最も強く安定して見えている 36–45 m 帯に合わせる
    ref = (rng >= 36) & (rng <= 45)

    def r4(m):
        return np.median(m[ref]) + 40 * np.log10(np.median(rng[ref]) / rng)

    print("(A) 距離帯ごとの目標強度（36本の中央値、時間方向最大の床超過 [dB]）")
    print(f"    雑音だけの帯 R {NOISE_R[0]:.0f}–{NOISE_R[1]:.0f} m: "
          f"MTI {np.median(mti_m[noise]):.1f} / 差分 {np.median(fdf_m[noise]):.1f} dB")
    print(f"{'R帯[m]':>9} {'MTI':>6} {'R^4期待':>8} {'不足':>6} | {'差分':>6} {'R^4期待':>8} {'不足':>6}")
    bands = [(21, 24), (24, 27), (27, 30), (30, 33), (33, 36), (36, 39), (39, 42), (42, 45), (45, 48), (48, 51)]
    rows = []
    for lo, hi in bands:
        b = (rng >= lo) & (rng < hi)
        a1, e1 = np.median(mti_m[b]), np.median(r4(mti_m)[b])
        a2, e2 = np.median(fdf_m[b]), np.median(r4(fdf_m)[b])
        rows.append((lo, hi, a1, e1, a2, e2))
        print(f"{lo:4d}–{hi:<4d} {a1:6.1f} {e1:8.1f} {e1 - a1:6.1f} | {a2:6.1f} {e2:8.1f} {e2 - a2:6.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    keep = (rng >= 15) & (rng <= 62)
    ax.plot(rng[keep], mti_m[keep], "o-", ms=3, label="MTI (chirp mean) + DC excluded  [current]")
    ax.plot(rng[keep], fdf_m[keep], "s-", ms=3, label="no MTI, frame difference, DC included\n(near range inflated by clutter flicker)")
    k2 = (rng >= 20) & (rng <= 50)
    ax.plot(rng[k2], r4(mti_m)[k2], "k--", lw=1, label="R^-4 expectation (anchored at 36-45 m)")
    ax.axhline(np.median(mti_m[noise]), color="gray", lw=0.8, ls=":", label="noise-only level (R 55-62 m)")
    ax.axvspan(rng[keep].min(), 31, color="tab:red", alpha=0.07)
    ax.axvline(D_LEFT, color="tab:red", lw=1)
    ax.text(D_LEFT + 0.3, ax.get_ylim()[1] * 0.95, "closest\napproach", color="tab:red", fontsize=8, va="top")
    for name, r in CONES.items():
        ax.text(r, ax.get_ylim()[0] + 0.3, name, fontsize=7, ha="center", color="tab:red")
    ax.set_xlabel("range R [m]")
    ax.set_ylabel("peak power over floor [dB]\n(median of 36 runs)")
    ax.set_title("Target is weakest near the closest approach (opposite to R^-4)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/night_intensity_vs_range.png", dpi=120)
    return rows


def framediff_excess(tag):
    """MTI なし + フレーム間差分の床超過 dB。(F-1, R) と時刻 (F-1,)"""
    d = np.load(npz_path(tag, nomti=True))
    diff = np.abs(np.diff(d["rd"], axis=0)).sum(axis=(1, 2))     # (F-1, D, R)
    pw = 20 * np.log10(diff.max(axis=1) + 1e-12)                 # (F-1, R)
    return pw - np.median(pw, axis=0, keepdims=True), d["t_rel_s"][1:]


def path_score(ex, rng, tt, pr_t, pr_r):
    """予測軌跡 (pr_t, pr_r) 上の床超過を ±1ビン・±1フレームの最大で拾って平均する。
    外挿の速度誤差で軌跡が少しずれても拾えるように近傍を見る"""
    vals = []
    for t0, r0 in zip(pr_t, pr_r):
        i, j = int(np.argmin(abs(tt - t0))), int(np.argmin(abs(rng - r0)))
        vals.append(ex[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].max())
    return float(np.mean(vals))


def part_a2():
    """手前の区間（R ≤ 30 m）に目標の反射が残っているかを、予測軌跡上で調べる。

    (A) の時間方向最大は、手前の強い静止クラッタの揺らぎも拾うので、それだけでは
    「目標が居るか」を言えない。そこで尾根から速度を出し、手前端から C1 まで等速で外挿した
    軌跡の上の強度を、同じ距離の並びを時間だけずらした対照（目標がそこに居ない時刻）と比べる
    """
    d0 = D_LEFT - 0.8            # レーン2相当。レーン間の差 0.8 m は結果をほぼ変えない
    out = {"fd": [], "mt": []}
    for n in range(1, N_MAIN + 1):
        _, _, dr = condition(n)
        g = pick_ridge(TAGS[n - 1])
        if g is None or len(g["f"]) < 10 or g["rng"].min() < d0 + 1:
            continue
        s = np.sqrt(g["rng"]**2 - d0**2)                        # 走路に沿った位置（垂線の足から）
        v = abs(np.polyfit(g["t"], s, 1)[0])
        if not 0.5 < v < 6:
            continue
        i_e = int(g["rng"].argmin())
        te, se = g["t"][i_e], s[i_e]
        ss = np.arange(se, -S0, -v * 0.2)                         # 手前端 → C1
        pt = te + (se - ss) / v if dr == "app" else te - (se - ss) / v
        ok = (pt >= 0.3) & (pt <= 29.7) & (np.hypot(ss, d0) <= 30)
        if ok.sum() < 8:
            continue
        pt, pr = pt[ok], np.hypot(ss[ok], d0)

        fd, tt = framediff_excess(TAGS[n - 1])
        pw, _, rng, *_ = moving_power_db(npz_path(TAGS[n - 1]))
        mt = (pw - np.median(pw, axis=0, keepdims=True))[1:]     # 時刻を差分側に揃える
        span = tt[-1] - tt[0]
        for key, ex in (("fd", fd), ("mt", mt)):
            on = path_score(ex, rng, tt, pt, pr)
            offs = [path_score(ex, rng, tt, (pt + sh - tt[0]) % span + tt[0], pr)
                    for sh in np.arange(6.0, 24.0, 1.0)]
            out[key].append((on, np.mean(offs), np.std(offs)))

    print("\n(A') 手前（R ≤ 30 m）の予測軌跡上に目標が残っているか（床超過 [dB]、本数の中央値）")
    res = {}
    for key, name in (("fd", "MTI なし・フレーム間差分"), ("mt", "現行 MTI + DC 除外    ")):
        a = np.array(out[key])                                   # (Run, 3)
        z = (a[:, 0] - a[:, 1]) / a[:, 2]
        res[key] = (len(a), np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 0] - a[:, 1]), np.sum(z > 2))
        print(f"    {name}: {len(a)} 本 | 軌跡上 {res[key][1]:.1f} / 時間ずらし {res[key][2]:.1f}"
              f" | 差 {res[key][3]:+.2f} dB | 対照より 2σ 超が {res[key][4]} 本")
    print("    -> 奥（36–45 m）では床から 10 dB 以上出ている目標が、手前では 1 dB 未満しか残らない。"
          "MTI を外しても戻らない")
    return res


# ------------------------------------------------------------------
# (B) 観測した走行の d プロファイル
# ------------------------------------------------------------------

def pick_ridge(tag, rmin=20.0, rmax=60.0, thr=20.0):
    """レンジ変化率とドップラーが一致する尾根のうち最長のもの（analyze_1001 と同じ判定）"""
    pw, vel, rng, t, v_ax, *_ = moving_power_db(npz_path(tag))
    keep = (rng >= rmin) & (rng <= rmax)
    gate = int(np.ceil(abs(v_ax).max() * (t[1] - t[0]) / (rng[1] - rng[0])))
    rs = build_ridges(pw[:, keep], vel[:, keep], rng[keep], t, thr, gate,
                      max_miss=3, min_run=5, dop_gate=2.0, v_res=abs(v_ax[1] - v_ax[0]))
    rs = [g for g in rs if abs(g["slope"] - g["vel"]) < 1.0 and abs(g["slope"]) > 0.3]
    return max(rs, key=lambda g: len(g["f"])) if rs else None


def part_b():
    print("\n(B) 観測した走行で d を動かしたときの残差（d = 5 m 〜 min(30, 観測の最小R) m）")
    out = []
    for n in range(1, N_MAIN + 1):
        g = pick_ridge(TAGS[n - 1])
        if g is None or len(g["f"]) < 10:
            continue
        d_grid = np.arange(5.0, min(30.0, g["rng"].min() - 0.5), 2.5)
        rms = np.array([p[2] for p in profile_d(g["rng"], g["t"], d_grid)])
        out.append((n, g["rng"].min(), d_grid, rms))
    ratio = np.array([r.max() / r.min() for *_, r in out])
    rmin = np.array([o[1] for o in out])
    print(f"    対象 {len(out)} 本（尾根 10 点以上）。観測の最小 R: 中央値 {np.median(rmin):.1f} m")
    print(f"    残差の最大/最小比: 中央値 {np.median(ratio):.2f}、最大 {ratio.max():.2f}"
          f"、1.2 倍未満が {np.sum(ratio < 1.2)}/{len(out)} 本")
    print("    -> d を数倍動かしても残差がほとんど変わらない。どの d でも等しく当てはまる")
    return out, ratio


# ------------------------------------------------------------------
# (C) 見えている最小距離と d̂ のばらつき（Cramér–Rao 下限）
# ------------------------------------------------------------------

def crb_sigma_d(r_near, v, d=D_LEFT, s_far=L_C1_C5 - S0, s_min=-S0, sigma=0.5, dt=0.2):
    """R(t) = sqrt(v^2 (t - t_min)^2 + d^2) の (v, t_min, d) を当てはめたときの d̂ の標準偏差の下限。

    見えているのは R ≥ r_near の部分だけとし、走路に沿った位置 s は
      r_near > d なら s ∈ [sqrt(r_near^2 - d^2), s_far]（最接近点を含まない片側）
      r_near ≤ d なら s ∈ [s_min, s_far]（最接近点を含む。s_min は C1 の位置）
    sigma は1フレームの距離の誤差 [m]。量子化（0.846/√12 ≈ 0.24 m）に揺らぎを足して 0.5 m とした
    """
    s_lo = np.sqrt(r_near**2 - d**2) if r_near > d else s_min
    s = np.arange(s_lo, s_far, v * dt)                           # (T,)
    tt = s / v                                                   # t - t_min
    r = np.hypot(s, d)
    jac = np.stack([v * tt**2 / r, -v**2 * tt / r, np.full_like(r, d) / r], axis=1)   # (T, 3)
    cov = sigma**2 * np.linalg.inv(jac.T @ jac)
    return float(np.sqrt(cov[2, 2]))


def lane_delta_d(road_w=4.0, n_lane=3):
    """隣り合うレーンの d の差 [m] の目安。

    手前の C1（左）と C2（右）は同じ X にあるとみなすと、R^2 の差は横方向の水平距離の2乗差だけになる:
      R_C1^2 - R_C2^2 = (a + w)^2 - a^2      a: レーダー直下から右の縁までの水平距離, w: 道幅
    道幅 w は 9/28 下見の 4 m を仮定（未実測）。レーンを道幅の等分点に置いて d(Y) の差を取る
    """
    a = (CONES["C1"]**2 - CONES["C2"]**2 - road_w**2) / (2 * road_w)
    h = np.sqrt(D_LEFT**2 - (a + road_w)**2)
    ys = a + road_w * (np.arange(n_lane) + 0.5) / n_lane          # 右の縁からの水平距離（レーン3→1）
    ds = np.hypot(ys, h)
    return a, h, float(np.mean(np.diff(ds)))


def part_c():
    a, h, dd = lane_delta_d()
    print(f"\n(C) 幾何: 垂線の足は C1 から奥へ {S0:.2f} m、左の縁の d = {D_LEFT:.2f} m")
    print(f"    道幅 4 m を仮定すると a = {a:.1f} m, h = {h:.1f} m、隣接レーンの d の差 ≈ {dd:.2f} m")
    r_nears = np.arange(20.0, 36.01, 0.5)
    print(f"    d̂ の標準偏差の下限（1フレームの距離誤差 0.5 m）")
    print(f"{'見えている最小R[m]':>18} {'徒歩 1.3m/s':>12} {'自転車 2.5m/s':>14}")
    sig = {}
    for v in (1.3, 2.5):
        sig[v] = np.array([crb_sigma_d(r, v) for r in r_nears])
    for r in (21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0):
        i = int(np.argmin(abs(r_nears - r)))
        print(f"{r:18.0f} {sig[1.3][i]:12.2f} {sig[2.5][i]:14.2f}")
    return r_nears, sig, dd


# 距離−時間図に載せる代表12本。上段が徒歩・下段が自転車で、各レーンの接近と離反を1本ずつ。
# 徒歩と自転車で同じ位置に同じレーン・向きが来るように揃え、手段の違いを縦に見比べられるようにする
RANGE_TIME_PICK = [[1, 2, 7, 8, 13, 14],
                   [19, 20, 25, 26, 31, 32]]


def plot_range_time():
    """代表12本（徒歩・自転車の各レーン、接近と離反）の距離−時間。コーンの斜距離を赤破線で重ねる。
    目標が R 31〜33 m より手前で見えなくなることを、処理を挟まずに見せるための図"""
    n_row, n_col = len(RANGE_TIME_PICK), len(RANGE_TIME_PICK[0])
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.3 * n_col, 4.6 * n_row), sharey=True)
    for ax, n in zip(axes.ravel(), sum(RANGE_TIME_PICK, [])):
        mode, lane, dr = condition(n)
        pw, _, rng, t, *_ = moving_power_db(npz_path(TAGS[n - 1]))   # (F, R)
        keep = rng <= 60
        im = ax.pcolormesh(rng[keep], t, pw[:, keep], shading="auto", cmap="viridis", vmin=10, vmax=35)
        for r in CONES.values():
            ax.axvline(r, color="r", lw=0.6, ls="--")
        ax.set_title(f"#{n} {mode} lane{lane} {'approach' if dr == 'app' else 'depart'}", fontsize=9)
        ax.set_xlabel("range R [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("time [s]")
    fig.colorbar(im, ax=axes, label="moving power [dB]")
    fig.savefig(ROOT / "figures/night_range_time.png", dpi=110, bbox_inches="tight")


def main():
    plot_range_time()
    rows = part_a()
    part_a2()
    prof, ratio = part_b()
    r_nears, sig, dd = part_c()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, rmin, d_grid, rms in prof:
        ax1.plot(d_grid, rms / rms.min(), color="gray", lw=0.8, alpha=0.6)
    ax1.set_xlabel("assumed lateral offset d [m]")
    ax1.set_ylabel("fit residual rms / min")
    ax1.set_title(f"Observed runs (n={len(prof)}): residual is flat in d\n"
                  f"median max/min = {np.median(ratio):.2f}")
    ax1.set_ylim(0.95, 1.5)
    ax1.grid(alpha=0.3)

    for v, lab in ((1.3, "walk 1.3 m/s"), (2.5, "bike 2.5 m/s")):
        ax2.semilogy(r_nears, sig[v], label=lab)
    ax2.axhline(dd, color="k", ls="--", lw=1, label=f"adjacent-lane difference in d ({dd:.2f} m)")
    ax2.axvline(D_LEFT, color="tab:red", lw=1)
    ax2.axvspan(30, 33, color="tab:red", alpha=0.12, label="observed near limit (30-33 m)")
    ax2.set_xlabel("nearest visible range [m]")
    ax2.set_ylabel("std of d-hat, lower bound [m]")
    ax2.set_title("d becomes identifiable only if the closest approach is visible")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/night_d_identifiability.png", dpi=120)
    print("\n-> figures/night_range_time.png, figures/night_intensity_vs_range.png, figures/night_d_identifiability.png")


if __name__ == "__main__":
    main()
