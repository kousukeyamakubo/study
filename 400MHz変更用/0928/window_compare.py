# 窓の開閉 × MTI の ON/OFF が、背景（無人）の誤警報にどう効くかを比べる。
#
# 指標: しきい値を超えたセルの割合 ＝ 誤警報率。
#   無人背景なので、しきい値を超えたセルは定義上すべて誤警報。全フレーム × 全動体
#   ドップラービン × 全距離ビンを1つの数字に畳めるので、どこを切り出すかの恣意性が消える。
#   しきい値を振って曲線にすれば「その条件で使えるしきい値はいくつか」も同時に読める。
#
# 図の読み方:
#   横軸のしきい値を1つ選ぶと、縦軸にそのときの誤警報率が出る。使える条件は
#   「点線（許容誤警報率）より下」かつ「赤い帯より左（歩行者を見逃さない）」の両方を
#   満たす縦位置が存在すること。
#
# 注意: 無人背景だけでは目標対クラッタ比は測れない。ここで言えるのは誤警報側だけ。
#
# 使い方:
#   python window_compare.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THR_NOW = 25.0     # 現行の検出しきい値[dB]（0727/atlas_track.py の既定）
FA_TARGET = 1e-3   # 許容誤警報率の目安
PED_DB = 22.0      # 歩行者の信号レベル[dB]（0728 実測、軌跡上の中央値。0727/atlas_track.py）

CASES = {
    "with window, MTI on":     "data/export/with_window/with_window_rd.npz",
    "with window, MTI off":    "data/export/with_window_nomti/with_window_rd.npz",
    "without window, MTI on":  "data/export/without_window/without_window_rd.npz",
    "without window, MTI off": "data/export/without_window_nomti/without_window_rd.npz",
}


def load_cells(npz_path, rmax):
    """動体セルの電力[dB]を返す。(Frame, Doppler', Range')"""
    d = np.load(npz_path)
    p = np.abs(d["rd"]).sum(axis=(1, 2))                     # (Frame, Doppler, Range) Rx 非コヒーレント加算
    db = 20 * np.log10(p + 1e-12)
    vel, rng = d["vel_ms"], d["range_m"]
    vel_res = float(np.abs(np.diff(vel)).mean())
    moving = np.abs(vel) >= vel_res                          # 静止クラッタ主ローブを外す
    return db[:, moving, :][:, :, rng <= rmax]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--out", default="window_compare.png")
    args = ap.parse_args()

    thr = np.linspace(5, 70, 261)                            # (T,)
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"{'case':26} {'FA@25dB':>9} {'thr for FA=1e-3':>17} {'median':>8}")

    for (name, path), c in zip(CASES.items(), ("C0", "C0", "C1", "C1")):
        flat = load_cells(Path(path), args.rmax).ravel()     # (N,)
        fa = (flat[None, :] > thr[:, None]).mean(axis=1)     # (T,)
        ls = "-" if "MTI on" in name else "--"

        ok = np.where(fa < FA_TARGET)[0]                     # FA_TARGET を下回る最小しきい値
        thr_need = thr[ok[0]] if len(ok) else np.nan
        fa_now = (flat > THR_NOW).mean()
        print(f"{name:26} {fa_now:9.2e} {thr_need:17.1f} {np.median(flat):8.2f}")

        ax.semilogy(thr, np.maximum(fa, 1e-7), c, ls=ls, label=name)
        ax.plot([THR_NOW], [max(fa_now, 1e-7)], c + "o", ms=7)

    # しきい値を歩行者の信号レベルより上げると、誤警報は消えるが歩行者も消える
    ax.axvspan(PED_DB, thr[-1], color="red", alpha=0.07)
    ax.axvline(PED_DB, color="darkred", lw=1.2)
    ax.text(PED_DB + 1.0, 2.5e-7, "threshold above pedestrian level -> missed",
            fontsize=8.5, color="darkred")
    ax.text(PED_DB, 1.6, f"pedestrian {PED_DB:.0f} dB", fontsize=8.5,
            color="darkred", ha="right")

    ax.axvline(THR_NOW, color="k", lw=1.2)
    ax.text(THR_NOW + 0.6, 1.6, f"current thr {THR_NOW:.0f} dB", fontsize=8.5)
    ax.axhline(FA_TARGET, color="gray", lw=1, ls=":")
    ax.text(thr[-1], FA_TARGET * 1.4, "acceptable FA 1e-3 ", fontsize=8.5,
            color="gray", ha="right")

    ax.set_xlabel("detection threshold [dB]")
    ax.set_ylabel("false-alarm rate  (unmanned background)")
    ax.set_title("Usable threshold: window open/closed x MTI on/off")
    ax.set_ylim(1e-7, 3)
    ax.set_xlim(thr[0], thr[-1])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
