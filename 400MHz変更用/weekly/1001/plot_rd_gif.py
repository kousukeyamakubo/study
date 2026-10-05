# 4本の RD マップ（距離×ドップラー）を時間順に並べた GIF を作る。
# range_time.png と tracks.png はドップラーを潰しているので、
# 目標がどの速度ビンに居たか・クラッタがどこに出ているかを目で確かめるため。
#
# 使い方:
#   python plot_rd_gif.py            # figures/rd_<tag>.gif を4本出す
#   python plot_rd_gif.py --fps 10   # 2倍速
#   python plot_rd_gif.py --mti-compare   # MTI OFF / ON を左右に並べた figures/rd_mti_<tag>.gif
#     （事前に atlas_export.py --no-mti で data/export_nomti/ を作っておく）

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from analyze_1001 import RMAX, RMIN, ROOT, RUNS, load, pick_ridge
# analyze_1001 が sys.path に 0727/ を足すので、その後で読む
from atlas_ridge_track import moving_power_db   # noqa: E402

R_SHOW = 55.0     # 解析ゲート上限 RMAX に合わせる


def make_gif(tag, label_en, fps, thr):
    pw, vel, rng, t, v_ax, db, k = load(tag)     # db: (Frame, Doppler, Range)
    g = pick_ridge(pw, vel, rng, t, v_ax, thr)
    on_ridge = {int(f): i for i, f in enumerate(g["f"])} if g is not None else {}
    # 尾根のビン番号 g["b"] は解析ゲート内での番号なので、ゲートで切った配列から引く
    vel_gate = vel[:, (rng >= RMIN) & (rng <= RMAX)]                     # (Frame, R_gate)

    keep = rng <= R_SHOW
    dv = v_ax[1] - v_ax[0]
    # pcolormesh 用にビンの境界を作る。ビン中心のままだと色の境目が半ビンずれる
    v_edge = np.r_[v_ax - dv / 2, v_ax[-1] + dv / 2]                     # (D+1,)
    dr = rng[1] - rng[0]
    r_edge = np.r_[rng[keep] - dr / 2, rng[keep][-1] + dr / 2]            # (R+1,)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    mesh = ax.pcolormesh(r_edge, v_edge, db[0][:, keep], cmap="viridis",
                         vmin=10, vmax=40, shading="flat")
    fig.colorbar(mesh, ax=ax, label="power [dB]")
    # 解析で外している領域を薄く示す（近傍 0-25 m は rmin で切っている）
    ax.axvspan(0, 25, color="white", alpha=0.12, lw=0)
    ax.axhline(0, color="white", lw=0.5, alpha=0.5)    # DC ビン（MTI で抑圧済み、解析では除外）
    mark, = ax.plot([], [], "o", ms=14, mfc="none", mec="red", mew=1.8)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("radial velocity [m/s]  (- : approaching)")
    ax.set_xlim(0, R_SHOW)
    title = ax.set_title("")

    def update(f):
        mesh.set_array(db[f][:, keep].ravel())
        if f in on_ridge:
            i = on_ridge[f]
            mark.set_data([g["rng"][i]], [vel_gate[f, g["b"][i]]])
        else:
            mark.set_data([], [])
        title.set_text(f"{label_en}   t = {t[f]:5.1f} s   (red: tracked ridge)")
        return mesh, mark, title

    anim = FuncAnimation(fig, update, frames=len(t), blit=False)
    out = ROOT / "figures" / f"rd_{tag}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def make_mti_gif(tag, label_en, fps):
    """MTI OFF / ON の RD マップを同じ色スケールで並べる。

    MTI（フレーム内チャープ平均の減算）が静止物だけを消し、移動目標の強度は
    変えないことを目で確かめるため。OFF 側の静止物は 80 dB 近くあるので、
    色スケールは両方に共通の 10-60 dB にして、飽和も含めて差が見えるようにする。
    """
    db_on = load(tag)[5]                                                  # (Frame, Doppler, Range)
    npz = ROOT / "data/export_nomti" / f"atlas_log_20261001_{tag}" / f"atlas_log_20261001_{tag}_rd.npz"
    pw, vel, rng, t, v_ax, db_off, k = moving_power_db(npz)

    keep = rng <= R_SHOW
    dv = v_ax[1] - v_ax[0]
    v_edge = np.r_[v_ax - dv / 2, v_ax[-1] + dv / 2]                     # (D+1,)
    dr = rng[1] - rng[0]
    r_edge = np.r_[rng[keep] - dr / 2, rng[keep][-1] + dr / 2]            # (R+1,)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    meshes = []
    for ax, db, name in zip(axes, (db_off, db_on), ("MTI OFF", "MTI ON")):
        m = ax.pcolormesh(r_edge, v_edge, db[0][:, keep], cmap="viridis",
                          vmin=10, vmax=60, shading="flat")
        meshes.append((m, db))
        ax.axhline(0, color="white", lw=0.5, alpha=0.5)
        ax.set_xlim(0, R_SHOW)
        ax.set_xlabel("Range [m]")
        ax.set_title(name)
    axes[0].set_ylabel("radial velocity [m/s]  (- : approaching)")
    fig.colorbar(meshes[0][0], ax=axes, label="power [dB]")
    sup = fig.suptitle("")

    def update(f):
        for m, db in meshes:
            m.set_array(db[f][:, keep].ravel())
        sup.set_text(f"{label_en}   t = {t[f]:5.1f} s")
        return [m for m, _ in meshes] + [sup]

    anim = FuncAnimation(fig, update, frames=len(t), blit=False)
    out = ROOT / "figures" / f"rd_mti_{tag}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=5.0)    # フレーム周期 0.2 s なので 5 で実時間
    ap.add_argument("--thr", type=float, default=22.0)
    ap.add_argument("--mti-compare", action="store_true")
    args = ap.parse_args()
    for tag, (_, label_en) in RUNS.items():
        if args.mti_compare:
            make_mti_gif(tag, label_en, args.fps)
        else:
            make_gif(tag, label_en, args.fps, args.thr)


if __name__ == "__main__":
    main()
