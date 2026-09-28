# 窓の開閉が背景（無人）にどう効くかを比べる。README「窓ガラスの影響検証」の一人でできる部分。
#
# 比較するのは3つだけ:
#   (a) 雑音＋クラッタ床の距離プロファイル … 窓越しで底上げされるか
#   (b) 動体電力の分布                      … 樹木の揺れが窓越しで減るか
#   (c) フレーム平均 RD マップ              … 虚像（ゴースト）が増えていないか
#
# 被験者が居ないので「反射ピークの減衰」はここでは測れない。それは 9/29 に人を入れて測る。
#
# 使い方:
#   python window_compare.py export_with_window/with_window_rd.npz \
#                            export_without_window/without_window_rd.npz

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(npz_path):
    d = np.load(npz_path)
    rd = d["rd"]                                            # (Frame, TX, RX, Doppler, Range)
    # Rx を非コヒーレント加算して電力にする。角度は見ないので位相は捨ててよい
    p = np.abs(rd).sum(axis=(1, 2))                         # (Frame, Doppler, Range)
    return 20 * np.log10(p + 1e-12), d["range_m"], d["vel_ms"], d["t_rel_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("with_npz", type=Path)
    ap.add_argument("without_npz", type=Path)
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--out", default="window_compare.png")
    args = ap.parse_args()

    db_w, rng, vel, _ = load(args.with_npz)
    db_o, _, _, _ = load(args.without_npz)

    # クラッタ主ローブ（|v| < 分解能）は静止物なので動体の議論から外す。
    # MTI が ON でも残留するため、ここで明示的に落とす
    vel_res = float(np.abs(np.diff(vel)).mean())
    moving = np.abs(vel) >= vel_res                          # (Doppler,)
    keep_r = rng <= args.rmax

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a)(b) 距離プロファイル。中央値＝床、95 パーセンタイル＝たまに立つ動体
    for db, name, c in ((db_w, "with window", "C0"), (db_o, "without window", "C1")):
        m = db[:, moving, :][:, :, keep_r]                   # (Frame, Doppler', Range')
        axes[0].plot(rng[keep_r], np.median(m, axis=(0, 1)), c, label=name)
        axes[1].plot(rng[keep_r], np.percentile(m, 95, axis=(0, 1)), c, label=name)

    axes[0].set_title("(a) noise/clutter floor  (median)")
    axes[1].set_title("(b) moving-target power  (95th pct)")
    for ax in axes[:2]:
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Power [dB]")
        ax.grid(alpha=0.3)
        ax.legend()

    # (c) 差分。正なら窓越しのほうが強い＝減衰していない or 虚像が乗っている
    diff = (np.median(db_w[:, moving, :][:, :, keep_r], axis=(0, 1))
            - np.median(db_o[:, moving, :][:, :, keep_r], axis=(0, 1)))
    axes[2].plot(rng[keep_r], diff, "C2")
    axes[2].axhline(0, color="k", lw=1)
    axes[2].set_title("(c) with - without  [dB]")
    axes[2].set_xlabel("Range [m]")
    axes[2].set_ylabel("Delta [dB]")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)

    # 数字も出す。図だけだと報告に書けない
    for db, name in ((db_w, "with window   "), (db_o, "without window")):
        m = db[:, moving, :][:, :, keep_r]
        far = rng[keep_r] >= 13.0                            # 5F から地上目標が現れうる下限（0727 の既定）
        print(f"{name}  floor(median) {np.median(m):6.2f} dB   "
              f"far>=13m {np.median(m[:, :, far]):6.2f} dB   "
              f"peak {m.max():6.2f} dB @ {rng[keep_r][np.unravel_index(m.argmax(), m.shape)[2]]:.1f} m")
    print(f"\ndiff median {np.median(diff):+.2f} dB   max {diff.max():+.2f} dB @ "
          f"{rng[keep_r][diff.argmax()]:.1f} m")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
