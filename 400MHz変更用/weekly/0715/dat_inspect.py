# DemoKitApp の録画ファイル (.dat) を解析して RD マップを生成する
#
# .dat の内部フォーマットは非公開だが、実ファイルの解析から以下を確認済み:
#   - 先頭 256 byte: ヘッダ。Cfg 由来のパラメータが double / int32 (LE) で並ぶ
#   - 以降: 196,608 byte × N フレーム（RAW 1フレーム = 16チャープセット×3TX×4RX×256サンプル×IQ×int16）
#   - 実測例: 19,661,056 byte = 256 + 196,608×100（10秒・10fps録画と整合）
#
# ヘッダの判明済みオフセット（本ファイル解析による推定。保証なし）:
#   0x0c: double 開始周波数[GHz]   0x14: double スロープ[MHz/µs]
#   0x1c: int32  サンプル数        0x20: int32  チャープ数/フレーム
#   0x44: int32  TX数              0x48: int32  RX数
#   0x4c: double 距離分解能[m]     0x54: double 最大距離[m]
#   0x5c: double 最大速度[m/s]     0x64: double 速度分解能[m/s]
#
# 使い方:
#   python dat_inspect.py xWR14xx_log_20260715_164450.dat --frame 50 --save rd_frame50.png

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FRAME_BYTES = 196608  # RAW 1フレーム（Short/Mid、98304 word）
HEADER_BYTES = 256


def parse_header(h: bytes) -> dict:
    """256byteヘッダから判明済みパラメータを取り出す（オフセットは実測解析による推定）"""
    d = lambda off: struct.unpack_from("<d", h, off)[0]
    i = lambda off: struct.unpack_from("<i", h, off)[0]
    return dict(
        start_freq_ghz=d(0x0C), slope_mhz_us=d(0x14),
        n_sample=i(0x1C), n_chirp_per_frame=i(0x20),
        n_tx=i(0x44), n_rx=i(0x48),
        range_res_m=d(0x4C), max_range_m=d(0x54),
        max_vel_ms=d(0x5C), vel_res_ms=d(0x64),
    )


def load_frames(path: Path, hdr: dict) -> np.ndarray:
    """データ部を (Frame, Chirpset, Tx, Rx, Sample) complex64 に整形する"""
    raw = np.fromfile(path, dtype="<i2", offset=HEADER_BYTES)
    n_tx, n_rx, n_sp = hdr["n_tx"], hdr["n_rx"], hdr["n_sample"]
    n_cs = hdr["n_chirp_per_frame"] // n_tx  # チャープセット数（48/3=16）
    words_per_frame = n_cs * n_tx * n_rx * n_sp * 2
    n_frame = raw.size // words_per_frame
    raw = raw[: n_frame * words_per_frame]
    iq = raw.reshape(n_frame, n_cs, n_tx, n_rx, n_sp, 2)
    # RAW出力は I が前・Q が後（FFT出力Cfgの場合は 虚→実 なので注意）
    return iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)


def rd_map(x_frame: np.ndarray, tx: int = 0) -> np.ndarray:
    """1フレームぶんの簡易RDマップ (Doppler, Range)。DC除去は静止クラッタ抑制のため"""
    z = x_frame[:, tx, :, :]                        # (Chirpset, Rx, Sample)
    z = z - z.mean(axis=0, keepdims=True)           # チャープ間平均を引き静止成分を抑制
    r = np.fft.fft(z * np.hanning(z.shape[-1]), axis=-1)
    rd = np.fft.fftshift(np.fft.fft(r, axis=0), 0)  # (Doppler, Rx, Range)
    return np.abs(rd).sum(axis=1)                   # (Doppler, Range) Rx非コヒーレント加算


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--frame", type=int, default=None, help="RDマップにするフレーム番号（省略時は中央）")
    ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--no-dc-cut", action="store_true", help="静止クラッタ抑制を無効化")
    args = ap.parse_args()

    b = args.dat.read_bytes()
    print(f"ファイルサイズ: {len(b):,} byte / データ部 {len(b)-HEADER_BYTES:,} = "
          f"{(len(b)-HEADER_BYTES)/FRAME_BYTES:.1f} フレーム相当")

    hdr = parse_header(b[:HEADER_BYTES])
    for k, v in hdr.items():
        print(f"  {k}: {v}")

    x = load_frames(args.dat, hdr)  # (Frame, Chirpset, Tx, Rx, Sample)
    print(f"IQ shape: {x.shape}  dtype: {x.dtype}")
    amp = np.abs(x)
    zero_frames = np.where(amp.reshape(x.shape[0], -1).max(axis=1) == 0)[0]
    print(f"振幅: max={amp.max():.0f} mean={amp.mean():.1f}  All-0フレーム: {list(zero_frames) or 'なし'}")

    fidx = args.frame if args.frame is not None else x.shape[0] // 2
    frame = x[fidx]
    if args.no_dc_cut:
        z = frame[:, args.tx, :, :]
        r = np.fft.fft(z * np.hanning(z.shape[-1]), axis=-1)
        rd = np.abs(np.fft.fftshift(np.fft.fft(r, axis=0), 0)).sum(axis=1)
    else:
        rd = rd_map(frame, tx=args.tx)

    n_dop, n_rng = rd.shape
    dr, dv = hdr["range_res_m"], hdr["vel_res_ms"]
    extent = [0, n_rng * dr, -n_dop / 2 * dv, n_dop / 2 * dv]

    plt.figure(figsize=(9, 4))
    plt.imshow(20 * np.log10(rd + 1e-6), aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label="Power [dB]")
    plt.xlabel("Range [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"{args.dat.name}  frame {fidx} (TX{args.tx})")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"保存: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
