# 10/5 昼の 6 本（自転車単独 2・車 2・止めた車の横を自転車 2）を、1013/1001/axes_maps.py と同じ
# 距離−時間・ドップラー−時間・角度−時間で描く。
#
# axes_maps.py の run_maps は 10/1 夜の No. からファイルを引くので、npz を直接受ける版をここに置く。
# 角度の取り方（各フレームの最強動体セルの角度スペクトル、TDM 位相補正込み）と
# MTI / フレーム間差分の 2 系統は axes_maps.py と同じ。
#
# 期待角度（赤点）は 10/1 夜に自転車レーン1 で当てはめた yaw −35.5°・符号 +1 をそのまま使う。
# 10/5 は俯角が 10/1 と違う（見える距離帯が手前にずれた）。俯角は u に効かないが、
# 置き直したときに yaw も変わっていれば赤点はずれる。レーンは分からないので、すべてレーン1 で描く。
#
# 使い方:
#   cd weekly/1013/1005
#   python axes_maps_1005.py
#   python axes_maps_1005.py --lang ja   （発表向けの日本語版を *_ja.png で出す。ドップラー−時間・時刻タグ・赤点・網掛け・角度の符号の注記なし）

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
sys.path.insert(0, str(WEEKLY / "0727"))
sys.path.insert(0, str(WEEKLY / "1001"))
sys.path.insert(0, str(WEEKLY / "1013" / "1001"))
from axes_maps import (ANG, ANG_LIM, ANG_RES, EXCESS_THR, NEAR, PW_LIM, R_HI, R_LO,   # noqa: E402
                       RANGE_RES, SIGN_NOTE, VEL_RES, bin_edges, expected_angle, res_bar,
                       strongest_cell_angle)

EXPORT = WEEKLY / "0727" / "atlas_export.py"
DATA = HERE / "data"            # 中間生成物（git 管理外）
FIG = HERE / "figures"
GEO = (-35.5, +1)               # 10/1 夜の自転車レーン1 での当てはめ（1013/1001/run_log.txt）
# (フォルダ, 時刻タグ, シナリオ)。シナリオはユーザーの申告（10/5）
RUNS = [
    ("pre", "135532", "bike approach"),
    ("pre", "135606", "bike depart"),
    ("car", "142246", "car depart"),
    ("car", "142546", "car approach"),
    ("car", "143104", "bike approach, parked car"),
    ("car", "143133", "bike depart, parked car"),
]


def export(dat, nomti):
    sub = DATA / ("export_nomti" if nomti else "export") / dat.stem
    npz = sub / f"{dat.stem}_rd.npz"
    if not npz.exists():
        cmd = [sys.executable, str(EXPORT), str(dat), "--outdir", str(sub)] + (["--no-mti"] if nomti else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return npz


def run_maps(dat):
    """axes_maps.run_maps と同じ量を、.dat から直接出す"""
    out = {}
    for kind in ("mti", "fd"):
        d = np.load(export(dat, nomti=(kind == "fd")))
        rd, rng, vel, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]   # rd: (F, TX, RX, D, R)
        hdr = json.loads(str(d["header"]))
        dc = int(np.abs(vel).argmin())
        k = np.abs(np.arange(len(vel)) - dc)                                  # (D,) DC からのビン番号
        if kind == "fd":
            # 0.2 s 前との複素差。最接近点付近（v_r → 0）も残す
            rd = np.diff(rd, axis=0)                                          # (F-1, TX, RX, D, R)
            t = t[1:]
            dop_ok = np.ones(len(vel), bool)
        else:
            dop_ok = k >= 1                                                   # ビン番号で切る
        db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)              # (F, D, R)
        rsel = (rng >= R_LO) & (rng <= R_HI)
        mov = np.where(dop_ok[:, None] & rsel[None, :], db, -np.inf)          # (F, D, R)

        ang = np.empty((len(t), len(ANG)))                                    # (F, A)
        pk_r, pk_v, pk_p = (np.empty(len(t)) for _ in range(3))
        for f in range(len(t)):
            ang[f], i_d, i_r = strongest_cell_angle(rd[f], mov[f], hdr)
            pk_r[f], pk_v[f], pk_p[f] = rng[i_r], vel[i_d], mov[f, i_d, i_r]

        out[kind] = dict(
            t=t, rng=rng, vel=vel, dc=dc,
            rt=mov.max(axis=1),                                               # (F, R)
            dt=np.where(rsel[None, None, :], db, -np.inf).max(axis=2),        # (F, D)
            ang=ang, pk_r=pk_r, pk_p=pk_p,
            pk_ang=ANG[ang.argmax(axis=1)],
            present=pk_p - np.median(pk_p) >= EXCESS_THR,
        )
    return out


def run_figure(m2, title, out):
    """axes_maps.run_figure と同じ配置: 2 行（MTI / フレーム間差分）× 3 列（距離・ドップラー・角度）"""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharey=True)
    for row, kind in enumerate(("mti", "fd")):
        m = m2[kind]
        ax = axes[row, 0]
        keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
        im0 = ax.pcolormesh(m["rng"][keep], m["t"], m["rt"][:, keep], shading="nearest", cmap="viridis",
                            vmin=PW_LIM[0], vmax=PW_LIM[1])
        ax.axvspan(*NEAR, color="w", alpha=0.15)
        res_bar(ax, "range")
        ax.set_xlabel(f"range R [m]\n(bar: 1 bin = {RANGE_RES} m)")
        ax.set_ylabel(f"{'MTI' if kind == 'mti' else 'no MTI, frame diff'}\ntime [s]")
        ax = axes[row, 1]
        ax.pcolormesh(bin_edges(m["vel"]), bin_edges(m["t"]), m["dt"], shading="flat", cmap="viridis",
                      vmin=PW_LIM[0], vmax=PW_LIM[1])
        res_bar(ax, "dop")
        ax.set_xlabel(f"radial velocity [m/s], + = receding\n(bar: 1 bin = {VEL_RES:.2f} m/s)")
        ax = axes[row, 2]
        im2 = ax.pcolormesh(ANG, m["t"], m["ang"], shading="nearest", cmap="viridis",
                            vmin=ANG_LIM[0], vmax=ANG_LIM[1])
        p = m["present"]
        ax.plot(expected_angle(m["pk_r"][p], 1, *GEO), m["t"][p], ".", ms=3, color="tab:red")
        ax.set_xlim(-90, 90)
        res_bar(ax, "ang")
        ax.set_xlabel(f"angle [deg]  ({SIGN_NOTE})\nbars: resolution ({ANG_RES} deg at 0 deg, wider off-axis)")
        fig.colorbar(im0, ax=axes[row, :2], label="power [dB] (same scale both rows)", shrink=0.8)
        fig.colorbar(im2, ax=axes[row, 2], label="angle spectrum [dB] (same scale both rows)", shrink=0.8)
    fig.suptitle(f"{title}  -  shaded: R {NEAR[0]:.0f}-{NEAR[1]:.0f} m;  red: expected angle of lane1 "
                 f"(yaw {GEO[0]} deg fitted on 10/1 night)", fontsize=10)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def overview_figure(maps, kind, out):
    """6 本を横に並べ、3 行（距離−時間・ドップラー−時間・角度−時間）で比べる"""
    fig, axes = plt.subplots(3, len(RUNS), figsize=(3.2 * len(RUNS), 12), sharey=True)
    for j, (_, tag, scen) in enumerate(RUNS):
        m = maps[tag][kind]
        keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
        im0 = axes[0, j].pcolormesh(m["rng"][keep], m["t"], m["rt"][:, keep], shading="nearest", cmap="viridis",
                                    vmin=PW_LIM[0], vmax=PW_LIM[1])
        axes[0, j].axvspan(*NEAR, color="w", alpha=0.15)
        # 細い格子では数値が重なるので、左端の列だけに書く（axes_maps.grid_figure と同じ）
        res_bar(axes[0, j], "range", label=(j == 0))
        axes[0, j].set_xlabel("range R [m]")
        axes[0, j].set_title(f"{tag}\n{scen}", fontsize=9)
        axes[1, j].pcolormesh(bin_edges(m["vel"]), bin_edges(m["t"]), m["dt"], shading="flat", cmap="viridis",
                              vmin=PW_LIM[0], vmax=PW_LIM[1])
        res_bar(axes[1, j], "dop", label=(j == 0))
        axes[1, j].set_xlabel("radial velocity [m/s]")
        im2 = axes[2, j].pcolormesh(ANG, m["t"], m["ang"], shading="nearest", cmap="viridis",
                                    vmin=ANG_LIM[0], vmax=ANG_LIM[1])
        p = m["present"]
        axes[2, j].plot(expected_angle(m["pk_r"][p], 1, *GEO), m["t"][p], ".", ms=2.5, color="tab:red")
        axes[2, j].set_xlim(-90, 90)
        res_bar(axes[2, j], "ang", label=(j == 0))
        axes[2, j].set_xlabel("angle [deg]")
    for ax, lab in zip(axes[:, 0], ("range-time", "Doppler-time", "angle-time")):
        ax.set_ylabel(f"{lab}\ntime [s]")
    src = "MTI (chirp mean)" if kind == "mti" else "no MTI, frame difference (DC included)"
    fig.suptitle(f"10/5 daytime, same setup  [{src}]  red: expected angle of lane1 (10/1 yaw).  {SIGN_NOTE}\n"
                 f"white bars: 1 bin = {RANGE_RES} m / {VEL_RES:.2f} m/s; angular resolution {ANG_RES} deg at 0 deg"
                 f" and {ANG_RES / np.cos(np.radians(30)):.1f} deg at 30 deg (left column labelled)", fontsize=11)
    fig.colorbar(im0, ax=axes[:2], label="power [dB] (same scale for MTI / frame diff)", shrink=0.6)
    fig.colorbar(im2, ax=axes[2], label="angle spectrum [dB] (same scale for MTI / frame diff)", shrink=0.8)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


# ---- 日本語版（発表向け）: 距離−時間と角度−時間だけにし、時刻タグ・期待角度の赤点は描かない ----
SCEN_JA = {
    "135532": "自転車が接近", "135606": "自転車が離反",
    "142246": "車が離反", "142546": "車が接近",
    "143104": "自転車が接近（駐車車両あり）", "143133": "自転車が離反（駐車車両あり）",
}
KIND_JA = {"mti": "MTI あり", "fd": "MTI なし（フレーム間差分）"}
# res_bar の白い線の説明。「バー」「ビン」では伝わらないので、線の長さが何を表すかを書く
RES_NOTE_JA = (f"各図の上端の白い線の長さ = 分解能（距離 {RANGE_RES:.2f} m、角度は正面で {ANG_RES}°・"
               f"30° 方向で {ANG_RES / np.cos(np.radians(30)):.1f}°）")


def setup_font_ja():
    # Windows 標準の Meiryo（presentations/20261014_ceatec/common.py と同じ）。Meiryo の「−」は字形が崩れるので ASCII にする
    plt.rcParams["font.family"] = ["Meiryo"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_rt_ja(ax, m, label=True):
    """距離−時間を 1 枚描く（日本語版）"""
    keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
    im = ax.pcolormesh(m["rng"][keep], m["t"], m["rt"][:, keep], shading="nearest", cmap="viridis",
                       vmin=PW_LIM[0], vmax=PW_LIM[1])
    res_bar(ax, "range", label=label)
    ax.set_xlabel("距離 R [m]")
    return im


def draw_at_ja(ax, m, label=True):
    """角度−時間を 1 枚描く（日本語版、期待角度の赤点なし）"""
    im = ax.pcolormesh(ANG, m["t"], m["ang"], shading="nearest", cmap="viridis",
                       vmin=ANG_LIM[0], vmax=ANG_LIM[1])
    ax.set_xlim(-90, 90)
    res_bar(ax, "ang", label=label)
    ax.set_xlabel("角度 [°]")
    return im


def run_figure_ja(m2, tag, out):
    """1 走行: 2 行（MTI あり / フレーム間差分）× 2 列（距離−時間・角度−時間）"""
    # Meiryo は行の高さが大きいので、段の間を広めに空ける
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True, gridspec_kw=dict(hspace=0.35))
    for row, kind in enumerate(("mti", "fd")):
        m = m2[kind]
        im0 = draw_rt_ja(axes[row, 0], m)
        im2 = draw_at_ja(axes[row, 1], m)
        axes[row, 0].set_ylabel(f"{KIND_JA[kind]}\n時刻 [s]")
        fig.colorbar(im0, ax=axes[row, 0], label="受信電力 [dB]", shrink=0.8)
        fig.colorbar(im2, ax=axes[row, 1], label="受信電力 [dB]", shrink=0.8)
    axes[0, 0].set_title("距離−時間", fontsize=11)
    axes[0, 1].set_title("角度−時間", fontsize=11)
    fig.suptitle(f"{SCEN_JA[tag]}\n{RES_NOTE_JA}", fontsize=10)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def overview_figure_ja(maps, kind, out):
    """6 本を横に並べ、2 行（距離−時間・角度−時間）で比べる（日本語版）"""
    fig, axes = plt.subplots(2, len(RUNS), figsize=(3.2 * len(RUNS), 8.5), sharey=True,
                             gridspec_kw=dict(hspace=0.25))
    for j, (_, tag, _) in enumerate(RUNS):
        m = maps[tag][kind]
        # 細い格子では数値が重なるので、左端の列だけに書く（英語版と同じ）
        im0 = draw_rt_ja(axes[0, j], m, label=(j == 0))
        im2 = draw_at_ja(axes[1, j], m, label=(j == 0))
        axes[1, j].set_xlabel("角度 [°]")
        # 「（駐車車両あり）」は列幅に収まらないので改行する（全列で 2 行にそろえる）
        title = SCEN_JA[tag].replace("（", "\n（") if "（" in SCEN_JA[tag] else SCEN_JA[tag] + "\n"
        axes[0, j].set_title(title, fontsize=10)
    for ax, lab in zip(axes[:, 0], ("距離−時間", "角度−時間")):
        ax.set_ylabel(f"{lab}\n時刻 [s]")
    fig.suptitle(f"10/5 昼の 6 走行（{KIND_JA[kind]}）\n{RES_NOTE_JA}",
                 fontsize=11, y=1.06)
    fig.colorbar(im0, ax=axes[0], label="受信電力 [dB]", shrink=0.8)
    fig.colorbar(im2, ax=axes[1], label="受信電力 [dB]", shrink=0.8)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    ja = "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "ja"
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    gi = DATA / ".gitignore"
    if not gi.exists():
        gi.write_text("# 生データ・中間生成物は git で追跡しない\n*\n", encoding="utf-8")
    maps = {}
    for folder, tag, scen in RUNS:
        dat = HERE / folder / f"atlas_log_20261005_{tag}.dat"
        maps[tag] = run_maps(dat)
    if ja:
        setup_font_ja()
        for _, tag, scen in RUNS:
            run_figure_ja(maps[tag], tag, FIG / f"{tag}_{scen.replace(', ', '_').replace(' ', '_')}_ja.png")
        for kind in ("mti", "fd"):
            overview_figure_ja(maps, kind, FIG / f"overview{'' if kind == 'mti' else '_fd'}_ja.png")
        return
    for _, tag, scen in RUNS:
        run_figure(maps[tag], f"{tag} {scen}", FIG / f"{tag}_{scen.replace(', ', '_').replace(' ', '_')}.png")
    for kind in ("mti", "fd"):
        overview_figure(maps, kind, FIG / f"overview{'' if kind == 'mti' else '_fd'}.png")

    # 角度の当てはまり: 目標がいるフレームの、レーン1 の期待角度からの残差（R 22〜40 m）
    print("\n期待角度（レーン1、10/1 の yaw）からの残差の中央値 [deg]（目標ありフレーム、R 22〜40 m）")
    for _, tag, scen in RUNS:
        m = maps[tag]["mti"]
        p = m["present"] & (m["pk_r"] >= 22) & (m["pk_r"] <= 40)
        r = m["pk_ang"][p] - expected_angle(m["pk_r"][p], 1, *GEO)
        print(f"  {tag} {scen:28s}: {np.median(r):+6.1f}  (n={p.sum()})" if p.any() else f"  {tag}: なし")


if __name__ == "__main__":
    main()
