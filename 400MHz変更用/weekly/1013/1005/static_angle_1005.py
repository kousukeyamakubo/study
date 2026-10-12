# 10/5 の 13:55 の 2 本と 14:22 以降の 4 本の間で、レーダーの向き（yaw）が変わったかを静止物の角度で確かめる。
#
# static_compare_1005.py で、静止物の強さが距離によって上下に 4〜9 dB 変わり、R 16〜28 m の揺れが増えた。
# 原因の候補は「雨（濡れ・木の揺れ）」と「間にレーダーに触った（置き方が変わった）」の 2 つ。
#   触って yaw が変わったなら、全距離の静止物の角度が u = sin θ の上で同じだけずれる（素子方向 FFT は u で等間隔）。
#   雨なら、静止物の位置（角度）は変わらず、強さと揺れだけが変わる。
# 比べる基準として、同じ回のうちで触っていない組（13:55 の 2 本どうし、14:22 以降の 4 本どうし）のずれも出す。
#
# 手順（1 本ごと）:
#   MTI なしの速度 0 ビンを、全フレームで素子方向 FFT（to_ra）→ 電力の時間中央値 → 静止物の距離−角度マップ (A, R)
# 2 本の間のずれ: R 10〜45 m の距離ビンごとに角度スペクトル（dB）を u 方向に相互相関し、
#   全距離で足し合わせた相関が最大になる u のずれを求める（放物線補間でビン以下まで）。
#   全距離共通の 1 つのずれを求めるのは、yaw の変化なら全距離で同じずれになるから。
# 距離方向のずれ（俯角や位置の変化）も同じ方法で距離プロファイルから出す。
#
# 使い方:
#   cd weekly/1013/1005
#   python static_angle_1005.py      （事前に axes_maps_1005.py で data/export_nomti を作る）
#   python static_angle_1005.py --lang ja   （図中の文字を日本語にした版を *_ja.png で出す）

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
sys.path.insert(0, str(WEEKLY / "0727"))
from atlas_export import to_ra   # noqa: E402

FIG = HERE / "figures"
N_ANT = 256
U = 2 * np.arange(-(N_ANT // 2), N_ANT - N_ANT // 2) / N_ANT   # (A,) u = sin θ。to_ra の出力の並び
DU = U[1] - U[0]
R_LO, R_HI = 10.0, 45.0
U_LIM = 0.9                     # |u| > 0.9（|θ| > 64°）は素子の指向性で弱く、雑音しかないので相関から外す
MAX_SHIFT = 40                  # 探すずれの範囲 [u ビン]（±0.31、約 ±18°）
# (時刻タグ, 回)。回 = 間にレーダーに触っていないはずの組
TAGS = [("135532", "13:55"), ("135606", "13:55"),
        ("142246", "14:22-"), ("142546", "14:22-"), ("143104", "14:22-"), ("143133", "14:22-")]
# 図中の文字。英語版を既定にして、既存の図はそのまま再現できるようにする
TXT = {
    "en": dict(
        db="dB", diff="difference [dB]", shift="estimated shift: {d:+.2f} deg",
        rows=("same session", "across sessions"), range="range R [m]",
        ang="angle [deg] (sign convention unverified)",
        title="10/5 static objects (no MTI, zero-Doppler bin, time median): range-angle map\n"
              "top: two runs at 13:55 (no touch expected);  bottom: 13:55 vs 14:22",
    ),
    "ja": dict(
        db="dB", diff="差 [dB]", shift="推定したずれ: {d:+.2f}°",
        rows=("同じ回", "回をまたぐ"), range="距離 R [m]",
        ang="角度 [°]（符号の向きは未確認）",
        title="10/5 静止物の距離−角度マップ（MTI なし・速度 0 ビン・時間中央値）\n"
              "上: 13:55 の 2 本（触っていないはず）　下: 13:55 と 14:22 の比較",
    ),
}


def static_ra(tag):
    """静止物の距離−角度マップ [dB] (A, R) と距離軸"""
    s = f"atlas_log_20261005_{tag}"
    d = np.load(HERE / "data" / "export_nomti" / s / f"{s}_rd.npz")
    rd, rng = d["rd"], d["range_m"]                                   # rd: (F, TX, RX, D, R)
    hdr = json.loads(str(d["header"]))
    i0 = int(np.argmin(np.abs(d["vel_ms"])))
    # 速度 0 ビンは TDM の位相補正が 0 なので、そのビンだけ切り出して渡してよい
    p = np.stack([np.abs(to_ra(rd[f][:, :, i0:i0 + 1, :], hdr, 1, N_ANT)[0]) ** 2
                  for f in range(len(rd))])                           # (F, A, R)
    # 時間中央値: 動く目標や 1 フレームだけの揺れに引っ張られないように
    return 10 * np.log10(np.median(p, axis=0) + 1e-12), rng           # (A, R)


def best_shift(score):
    """相関 score (S,) の最大位置を放物線補間で返す [ビン]（S の中央がずれ 0）"""
    k = int(np.argmax(score))
    if 0 < k < len(score) - 1:
        a, b, c = score[k - 1], score[k], score[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k - (len(score) - 1) / 2


def u_shift(ra_a, ra_b, rsel):
    """b が a に対して u 方向にいくつずれているか [u]。距離ビンごとの相関を足し合わせる"""
    usel = np.abs(U) <= U_LIM
    a = ra_a[usel][:, rsel]                                            # (A', R')
    b = ra_b[usel][:, rsel]
    # 距離ビンごとに平均を引いて正規化（強い距離ビンだけで決まらないように）
    a = (a - a.mean(0)) / (a.std(0) + 1e-9)
    b = (b - b.mean(0)) / (b.std(0) + 1e-9)
    n = a.shape[0]
    score = np.array([np.sum(a[max(0, -s):n - max(0, s)] * b[max(0, s):n - max(0, -s)]) /
                      (n - abs(s)) for s in range(-MAX_SHIFT, MAX_SHIFT + 1)])
    return best_shift(score) * DU, score


def r_shift(ra_a, ra_b, rsel, rng):
    """距離プロファイル（角度方向に最大）の距離方向のずれ [m]"""
    a = ra_a.max(0)[rsel]
    b = ra_b.max(0)[rsel]
    a, b = a - a.mean(), b - b.mean()
    n, m = len(a), 5
    score = np.array([np.sum(a[max(0, -s):n - max(0, s)] * b[max(0, s):n - max(0, -s)]) / (n - abs(s))
                      for s in range(-m, m + 1)])
    return best_shift(score) * (rng[1] - rng[0])


def main():
    lang = "ja" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "ja" else "en"
    tx = TXT[lang]
    if lang == "ja":
        # Windows 標準の Meiryo（presentations/20261014_ceatec/common.py と同じ）
        plt.rcParams["font.family"] = ["Meiryo"]
        plt.rcParams["axes.unicode_minus"] = False
    maps = {}
    for tag, _ in TAGS:
        maps[tag], rng = static_ra(tag)
    rsel = (rng >= R_LO) & (rng <= R_HI)

    # (1) 全組のずれ。13:55 の 1 本目を基準にした表と、組ごとの表
    print(f"静止物（MTI なしの速度 0 ビン、時間中央値）の角度・距離のずれ。R {R_LO:.0f}〜{R_HI:.0f} m、|u| <= {U_LIM}")
    print("u のずれ → 正面付近の角度に直すと Δθ ≈ Δu [rad]。1 u ビン = "
          f"{DU:.4f}（{np.degrees(DU):.2f}° 相当）\n")
    print("(1) 組ごとのずれ（b − a）")
    print(f"{'a':>7} {'b':>7}  {'組':10s} {'Δu':>8} {'Δθ[°]':>7} {'ΔR[m]':>7}")
    rows = []
    for i, (ta, ga) in enumerate(TAGS):
        for tb, gb in TAGS[i + 1:]:
            du, _ = u_shift(maps[ta], maps[tb], rsel)
            dr = r_shift(maps[ta], maps[tb], rsel, rng)
            kind = "同じ回" if ga == gb else "回をまたぐ"
            rows.append((ta, tb, kind, du, dr))
            print(f"{ta:>7} {tb:>7}  {kind:10s} {du:+8.4f} {np.degrees(du):+7.2f} {dr:+7.2f}")

    print("\n(2) まとめ: |Δθ| [°]")
    for kind in ("同じ回", "回をまたぐ"):
        v = np.array([abs(np.degrees(r[3])) for r in rows if r[2] == kind])
        w = np.array([abs(r[4]) for r in rows if r[2] == kind])
        print(f"  {kind:10s}: Δθ 中央値 {np.median(v):.2f}  最大 {v.max():.2f}  (n={len(v)})   "
              f"|ΔR| 中央値 {np.median(w):.2f} m  最大 {w.max():.2f} m")


    # (3) 図: 代表 2 本（13:55 の 1 本目と、14:22 以降で車が走っていない時間の多い 1 本）の静止物の距離−角度と差
    FIG.mkdir(exist_ok=True)
    ang = np.degrees(np.arcsin(U))
    pairs = [("135532", "135606"), ("135532", "142246")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for row, (ta, tb) in enumerate(pairs):
        a, b = maps[ta][:, rsel], maps[tb][:, rsel]
        for j, (img, lab, cm, lim) in enumerate([(a, ta, "viridis", (0, 40)), (b, tb, "viridis", (0, 40)),
                                                  (b - a, f"{tb} − {ta}", "RdBu_r", (-10, 10))]):
            im = axes[row, j].pcolormesh(ang, rng[rsel], img.T, shading="nearest", cmap=cm,
                                         vmin=lim[0], vmax=lim[1])
            axes[row, j].set_title(lab, fontsize=10)
            if j != 1:
                fig.colorbar(im, ax=axes[row, j], label=tx["db"] if j == 0 else tx["diff"], shrink=0.8)
        du = next(r[3] for r in rows if r[0] == ta and r[1] == tb)
        axes[row, 0].set_ylabel(f"{tx['rows'][row]}\n{tx['range']}")
        axes[row, 2].text(0.02, 0.97, tx["shift"].format(d=np.degrees(du)), transform=axes[row, 2].transAxes,
                          va="top", fontsize=9, bbox=dict(fc="w", alpha=0.8))
    for ax in axes[-1]:
        ax.set_xlabel(tx["ang"])
        ax.set_xlim(-65, 65)
    fig.suptitle(tx["title"], fontsize=11)
    out = FIG / f"static_angle_compare{'_ja' if lang == 'ja' else ''}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
