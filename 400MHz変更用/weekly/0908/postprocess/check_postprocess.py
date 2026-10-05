# detect.py / track.py の動作確認。合成 RD テンソルを使うので実測データが無くても検証できる。
#
# 目的: 静止ノイズの中に1目標を置いた合成データで、
#   1) CFAR+NMS がノイズを誤検出せず目標だけを検出するか
#   2) フレームをまたいで1本のトラックにまとまるか
# を確認する。しきい値・窓サイズの妥当性を実測前に見ておくためのもの。
#
# 使い方:
#   python check_postprocess.py

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import detect_peaks, incoherent_power  # noqa: E402
from track import greedy_associate  # noqa: E402

# atlas_export.py のヘッダに合わせた実測相当のビン構成
N_FRAME, N_TX, N_RX, N_DOPPLER, N_RANGE = 50, 2, 4, 16, 129
RANGE_RES_M = 0.8463541666666666
MAX_VEL_MS = 4.79570305006714
RANGE_M = np.arange(N_RANGE) * RANGE_RES_M
VEL_MS = np.linspace(-MAX_VEL_MS, MAX_VEL_MS, N_DOPPLER, endpoint=False)

RNG = np.random.default_rng(0)


def make_synthetic_rd(target_r_idx: int, target_d_idx: int, snr_db: float) -> np.ndarray:
    """静止ノイズ（複素ガウス, 分散1）に、単一目標を1点だけ重ねた RD テンソルを作る。
    目標は全フレーム・全 TX/RX で同じ (range, doppler) ビンに留まる（=等速直線運動）とみなす簡易版"""
    noise = (RNG.standard_normal((N_FRAME, N_TX, N_RX, N_DOPPLER, N_RANGE))
             + 1j * RNG.standard_normal((N_FRAME, N_TX, N_RX, N_DOPPLER, N_RANGE))) / np.sqrt(2)
    amp = 10 ** (snr_db / 20.0)
    rd = noise.astype(np.complex64)
    rd[:, :, :, target_d_idx, target_r_idx] += amp
    return rd


def main():
    print("=" * 76)
    print("[1] 単一目標がノイズの中で検出されるか（SNR を振る）")
    print("=" * 76)
    target_r, target_d = 60, 10
    for snr_db in (20, 15, 10, 6, 3):
        rd = make_synthetic_rd(target_r, target_d, snr_db)
        peaks_by_frame = detect_peaks(rd)
        hit = sum(1 for p in peaks_by_frame
                  if any(int(d) == target_d and int(r) == target_r for d, r in p))
        false_alarms = sum(len(p) for p in peaks_by_frame) - hit
        print(f"  SNR={snr_db:4.0f} dB: 検出フレーム {hit:2d}/{N_FRAME}, "
              f"誤検出（他ビン） {false_alarms}")

    print("\n" + "=" * 76)
    print("[2] トラッキング: 検出が1本のトラックにまとまるか")
    print("=" * 76)
    rd = make_synthetic_rd(target_r, target_d, snr_db=15)
    peaks_by_frame = detect_peaks(rd)
    tracks = greedy_associate(peaks_by_frame, RANGE_M, VEL_MS,
                              max_range_step_m=RANGE_RES_M * 3, max_vel_step_ms=1.0)
    lengths = sorted((len(t.points) for t in tracks), reverse=True)
    print(f"  トラック数: {len(tracks)}, 長さ上位: {lengths[:5]}")
    assert len(tracks) >= 1 and max(lengths) >= N_FRAME * 0.8, \
        "静止目標なのに1本の長いトラックにまとまっていない"
    print("  → 静止目標が1本の長いトラックにまとまっている")

    print("\n" + "=" * 76)
    print("[3] パワーマップの形状確認")
    print("=" * 76)
    power = incoherent_power(rd)
    print(f"  incoherent_power の出力形状: {power.shape}（期待値: "
          f"({N_FRAME}, {N_DOPPLER}, {N_RANGE})）")
    assert power.shape == (N_FRAME, N_DOPPLER, N_RANGE)
    print("  → OK")


if __name__ == "__main__":
    main()
