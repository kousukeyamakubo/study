# t14re_capture.py で保存した IQ (.npy) から簡易RDマップを描画する
#
# 使い方（例）:
#   python plot_rd_map.py captures/iq_20260716_100000_0000.npy --preset Short
#
# 軸スケールは Cfg説明書のプリセット値（docs/hardware.md 参照）から算出。
# 学習時と同一の処理系（Cyclist_env_RDA_2nano.py）への置き換えは次段階。

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from t14re_capture import PRESETS, quick_rd_map

# プリセットごとの物理スケール（Cfg説明書より）
SCALES = {
    # (距離分解能 m/bin, 速度分解能 m/s/bin)
    "Short":  (0.045, 0.180),
    "Mid":    (0.244, 0.180),
    "100fps": (0.045, 0.718),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy", type=Path)
    ap.add_argument("--preset", choices=SCALES, default="Short")
    ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--save", type=Path, default=None, help="画像の保存先（省略時は表示のみ）")
    args = ap.parse_args()

    x = np.load(args.npy)  # (Frame, Chirpset, Tx, Rx, Sample)
    rd = quick_rd_map(x, tx=args.tx)  # (Doppler, Range)

    dr, dv = SCALES[args.preset]
    n_dop, n_rng = rd.shape
    # レンジFFTそのままなので横軸ビン=距離、縦軸は fftshift 済みで速度0が中央
    extent = [0, n_rng * dr, -n_dop / 2 * dv, n_dop / 2 * dv]

    plt.figure(figsize=(8, 4))
    plt.imshow(20 * np.log10(rd + 1e-6), aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label="Power [dB]")
    plt.xlabel("Range [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"{args.npy.name}  ({args.preset}, TX{args.tx})")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"保存: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
