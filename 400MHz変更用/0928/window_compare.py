# 窓の開閉が背景（無人）にどう効くかを比べる。
#
# 【2026-09-28 改訂】初版は代表3距離の値を並べていたが、差分カーブの山を見てから
# 距離を選んでいたため選択バイアスがあった。距離を選ばずに済む指標に差し替える。
#
# 指標: しきい値を超えたセルの割合 ＝ 誤警報率。
#   無人背景なので、しきい値を超えたセルは定義上すべて誤警報。全フレーム × 全動体
#   ドップラービン × 全距離ビンを1つの数字に畳めるので、どこを切り出すかの恣意性が消える。
#   しきい値を振って曲線にすれば「その条件で使えるしきい値はいくつか」も同時に読める。
#
# MTI の ON/OFF も並べる。初版は atlas_export の既定（MTI=ON）の出力しか見ておらず、
# 「窓の反射が強い」のか「MTI が窓の反射を消しきれていない」のか区別できていなかった。
#
# 注意: 無人背景だけでは目標対クラッタ比は測れない。ここで言えるのは誤警報側だけで、
# 「窓を開けるべきか」には答えられない（目標ありのデータが要る）。
#
# 使い方:
#   python window_compare.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THR_NOW = 25.0            # 現行の検出しきい値[dB]（0727/atlas_track.py の既定）
FA_TARGET = 1e-3          # 許容誤警報率の目安


def load_cells(npz_path, rmax):
    """動体セルの電力[dB]を平たい配列で返す。(Frame, Doppler', Range') を潰したもの"""
    d = np.load(npz_path)
    p = np.abs(d["rd"]).sum(axis=(1, 2))                    # (Frame, Doppler, Range) Rx 非コヒーレント加算
    db = 20 * np.log10(p + 1e-12)
    vel, rng = d["vel_ms"], d["range_m"]
    vel_res = float(np.abs(np.diff(vel)).mean())
    moving = np.abs(vel) >= vel_res                          # 静止クラッタ主ローブを外す
    return db[:, moving, :][:, :, rng <= rmax], rng[rng <= rmax], db[:, moving, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--out", default="window_compare.png")
    args = ap.parse_args()

    cases = {
        "with window, MTI on":     "export_with_window/with_window_rd.npz",
        "with window, MTI off":    "export_with_window_nomti/with_window_rd.npz",
        "without window, MTI on":  "export_without_window/without_window_rd.npz",
        "without window, MTI off": "export_without_window_nomti/without_window_rd.npz",
    }
    thr = np.linspace(5, 70, 261)                            # (T,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    print(f"{'case':26} {'FA@25dB':>9} {'thr for FA=1e-3':>17} {'median':>8} {'p99.9':>8}")

    for (name, path), c in zip(cases.items(), ("C0", "C0", "C1", "C1")):
        cells, rng, _ = load_cells(Path(path), args.rmax)
        flat = cells.ravel()                                 # (N,)
        fa = (flat[None, :] > thr[:, None]).mean(axis=1)     # (T,) 誤警報率
        ls = "-" if "MTI on" in name else "--"

        # FA_TARGET を下回る最小のしきい値
        ok = np.where(fa < FA_TARGET)[0]
        thr_need = thr[ok[0]] if len(ok) else np.nan

        print(f"{name:26} {(flat > THR_NOW).mean():9.2e} {thr_need:17.1f} "
              f"{np.median(flat):8.2f} {np.percentile(flat, 99.9):8.2f}")

        axes[0].semilogy(thr, np.maximum(fa, 1e-7), c, ls=ls, label=name)
        axes[1].plot(rng, np.median(cells, axis=(0, 1)), c, ls=ls, label=name)

    axes[0].axvline(THR_NOW, color="k", lw=1)
    axes[0].text(THR_NOW, 2e-7, " current thr 25 dB", fontsize=9)
    axes[0].axhline(FA_TARGET, color="gray", lw=1, ls=":")
    axes[0].set_xlabel("threshold [dB]")
    axes[0].set_ylabel("false-alarm rate (unmanned background)")
    axes[0].set_title("(a) false-alarm rate vs threshold")
    axes[0].set_ylim(1e-7, 1.5)

    axes[1].set_xlabel("Range [m]")
    axes[1].set_ylabel("Power [dB]")
    axes[1].set_title("(b) median power vs range (reference)")

    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
