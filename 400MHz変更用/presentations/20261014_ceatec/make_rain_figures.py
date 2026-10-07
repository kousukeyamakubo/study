# 10/5 の背景の違い（13:55 の 2 本 と 14:22 以降の 4 本）が何によるかを説明するための図（CEATEC 発表の素材）。
#
# 数値の出どころと解釈は weekly/1013/1005/README.md §2「背景」「静止物の位置」・§3。ここでは同じ量を発表用に描き直すだけ:
#   rain_static_fluct_*.png : 距離ごとの静止物の強さと、フレームごとの揺れ（static_compare_1005.py と同じ定義）
#   rain_position_*.png     : 静止物の距離−角度マップ（扇形）の 13:55 / 14:22 / 差（static_angle_1005.py と同じ定義）
#   rain_shift_*.png        : 静止物の角度のずれ（同じ回どうし / 回をまたぐ）。参考に 10/1 → 10/5 の置き直し
#   rain_camera_*.png       : 13:55 と 14:22 のカメラ映像の先頭フレーム（路面の様子を見比べる用）
# 番号は make_sync_video.py と同じ（common.RUNS）。
#
# 使い方:
#   cd presentations/20261014_ceatec
#   python make_rain_figures.py

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
D1005 = ROOT / "weekly" / "1013" / "1005"
sys.path.insert(0, str(ROOT / "weekly" / "0727"))
sys.path.insert(0, str(D1005))
sys.path.insert(0, str(ROOT.parent / "camera_pipeline" / "0817" / "capture"))
from cam_extract import read_cam, split_jpegs                       # noqa: E402
from static_angle_1005 import R_HI as RA_HI, R_LO as RA_LO, U, static_ra, u_shift   # noqa: E402
from static_compare_1005 import profiles                            # noqa: E402

from common import FIG, LABELS, RUNS, setup_font                    # noqa: E402

C_1355, C_1422 = "#2a78d6", "#eb6834"      # 回の色（2 系列）。どちらの回かは凡例でも示す
BAND = (16.0, 28.0)                          # 揺れが増えた木の帯
A_LIM = 64.0
SESSION = {tag: ("1355" if folder == "pre" else "1422") for _, folder, tag, *_ in RUNS}
NO = {tag: no for no, _, tag, *_ in RUNS}


def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"-> figures/{name}")


def fig_static_fluct(lab, lang, prof):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for tag, (rng, st, fl) in prof.items():
        c = C_1355 if SESSION[tag] == "1355" else C_1422
        sel = (rng >= 10) & (rng <= 45)
        axes[0].plot(rng[sel], st[sel], color=c, lw=2, alpha=0.85)
        axes[1].plot(rng[sel], fl[sel], color=c, lw=2, alpha=0.85)
    for ax, yl, tl in ((axes[0], lab["static"], lab["static_t"]), (axes[1], lab["fluct"], lab["fluct_t"])):
        ax.axvspan(*BAND, color="#999", alpha=0.15, lw=0)
        ax.set_ylabel(yl, fontsize=12)
        ax.set_title(tl, fontsize=13, loc="left")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].text(np.mean(BAND), 0.97, lab["band"], transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=11, color="#444")
    axes[1].set_xlabel(lab["range"], fontsize=12)
    axes[1].set_xlim(10, 45)
    axes[0].plot([], [], color=C_1355, lw=2, label=lab["s1355"])
    axes[0].plot([], [], color=C_1422, lw=2, label=lab["s1422"])
    axes[0].legend(loc="lower left", fontsize=11, frameon=False)
    save(fig, f"rain_static_fluct_{lang}.png")


def fan(ax, img, rng, cmap, lim, ink="w"):
    """距離−角度マップ (A, R) を扇形に描く。角度は + を右（make_sync_video.py と同じ向き）"""
    ang = np.degrees(np.arcsin(U))
    asel, rsel = np.abs(ang) <= A_LIM, (rng >= RA_LO) & (rng <= RA_HI)
    rr, tt = np.meshgrid(rng[rsel], np.radians(ang[asel]), indexing="ij")   # (R', A')
    m = ax.pcolormesh(rr * np.sin(tt), rr * np.cos(tt), img[asel][:, rsel].T, shading="gouraud",
                      cmap=cmap, vmin=lim[0], vmax=lim[1])
    for r0 in (10, 20, 30, 40):
        a = np.radians(np.linspace(-A_LIM, A_LIM, 100))
        ax.plot(r0 * np.sin(a), r0 * np.cos(a), color=ink, lw=0.5, alpha=0.5)
        ax.text(0.6, r0 + 0.4, f"{r0} m", fontsize=9, color=ink)
    ax.set_aspect("equal")
    ax.set_xlim(-42, 42)
    ax.set_ylim(0, 47)
    return m


def fig_position(lab, lang, maps, rng):
    a, b = maps["135532"], maps["142246"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    m0 = fan(axes[0], a, rng, "viridis", (0, 40))
    fan(axes[1], b, rng, "viridis", (0, 40))
    m2 = fan(axes[2], b - a, rng, "RdBu_r", (-10, 10), ink="#333")
    for ax, t in zip(axes, (lab["ra_1355"], lab["ra_1422"], lab["ra_diff"])):
        ax.set_title(t, fontsize=13)
        ax.set_xlabel(lab["cross"], fontsize=11)
    axes[0].set_ylabel(lab["down"], fontsize=11)
    fig.colorbar(m0, ax=axes[:2], shrink=0.8, pad=0.02, label=lab["static"])
    fig.colorbar(m2, ax=axes[2], shrink=0.8, pad=0.02, label=lab["diff"])
    save(fig, f"rain_position_{lang}.png")


def fig_shift(lab, lang, maps, rng):
    rsel = (rng >= RA_LO) & (rng <= RA_HI)
    tags = [t for _, _, t, *_ in RUNS]
    same, cross = [], []
    for i, ta in enumerate(tags):
        for tb in tags[i + 1:]:
            d = abs(np.degrees(u_shift(maps[ta], maps[tb], rsel)[0]))
            (same if SESSION[ta] == SESSION[tb] else cross).append(d)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rs = np.random.default_rng(0)
    for y, v, c in ((1, same, "#555"), (0, cross, "#555")):
        ax.scatter(v, y + rs.uniform(-0.12, 0.12, len(v)), s=60, color=c, edgecolor="w", lw=1.5, zorder=3)
        ax.text(max(v) + 0.25, y, f"≤ {max(v):.2f}°", va="center", fontsize=11, color="#333")
    ax.axvspan(5, 10, color="#999", alpha=0.18, lw=0)
    ax.text(7.5, 0.5, lab["relocate"], ha="center", va="center", fontsize=10, color="#444", wrap=True)
    ax.set_yticks([1, 0], [f"{lab['same']}\n(n={len(same)})", f"{lab['cross_s']}\n(n={len(cross)})"], fontsize=11)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel(lab["shift"], fontsize=12)
    ax.set_title(lab["shift_t"], fontsize=13, loc="left")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    save(fig, f"rain_shift_{lang}.png")


def fig_camera(lab, lang, cams):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    for ax, img, t in zip(axes, cams, (lab["cam_1355"], lab["cam_1422"])):
        ax.imshow(img[:, :, ::-1])
        ax.set_title(t, fontsize=13)
        ax.set_axis_off()
    save(fig, f"rain_camera_{lang}.png")


def main():
    FIG.mkdir(exist_ok=True)
    tags = [t for _, _, t, *_ in RUNS]
    prof = {t: profiles(t) for t in tags}
    maps = {}
    for t in tags:
        maps[t], rng = static_ra(t)
    cams = []
    for folder, tag in (("pre", "135532"), ("car", "142246")):
        j = split_jpegs(read_cam(D1005 / folder / f"atlas_log_20261005_{tag}.cam.gz"))[0]
        cams.append(cv2.resize(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), (1280, 720),
                               interpolation=cv2.INTER_AREA))
    for lang in ("ja", "en"):
        setup_font(lang)
        lab = LABELS[lang]
        fig_static_fluct(lab, lang, prof)
        fig_position(lab, lang, maps, rng)
        fig_shift(lab, lang, maps, rng)
        fig_camera(lab, lang, cams)


if __name__ == "__main__":
    main()
