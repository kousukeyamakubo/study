# 録画10秒間の動体有無と静止シーンの時間変化を確認する解析
# 使い方: python motion_check.py <datファイル>（省略時は7/15の録画）
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dat_inspect import HEADER_BYTES, load_frames, parse_header

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "xWR14xx_log_20260715_164450.dat"
hdr = parse_header(p.read_bytes()[:HEADER_BYTES])
x = load_frames(p, hdr)  # (Frame, Chirpset, Tx, Rx, Sample)

# フレームごとの動体エネルギー: 静止成分（チャープ間平均）除去後の全RD電力
z = x[:, :, 0, :, :] - x[:, :, 0, :, :].mean(axis=1, keepdims=True)
rng_fft = np.fft.fft(z * np.hanning(256), axis=-1)
dop_fft = np.fft.fft(rng_fft, axis=1)
mov = np.abs(dop_fft).sum(axis=(1, 2, 3))  # (Frame,)
top = np.argsort(mov)[::-1][:10]
print("動体エネルギー上位10フレーム:", top.tolist())
print(f"比率 max/min = {mov.max() / mov.min():.2f}")

# 時間×レンジ図①: 静止プロファイル（zero-Doppler）の推移
prof = np.abs(
    np.fft.fft(x[:, :, 0, :, :].mean(axis=1) * np.hanning(256), axis=-1)
).sum(axis=1)  # (Frame, Range)
dr = hdr["range_res_m"]
plt.figure(figsize=(9, 4))
plt.imshow(20 * np.log10(prof + 1e-6), aspect="auto", origin="lower",
           extent=[0, 256 * dr, 0, 10.0])
plt.colorbar(label="Power [dB]")
plt.xlabel("Range [m]")
plt.ylabel("Time [s]")
plt.title("Range profile vs time (zero-Doppler)")
plt.tight_layout()
out1 = Path(__file__).parent / f"range_time_{p.stem}.png"
plt.savefig(out1, dpi=150)
print(f"保存: {out1}")

# 時間×レンジ図②: 動体成分（静止除去後）の推移。歩行なら斜めの軌跡として見える
mov_prof = np.abs(rng_fft).sum(axis=(1, 2))  # (Frame, Range) — rng_fft は静止除去済み
plt.figure(figsize=(9, 4))
plt.imshow(20 * np.log10(mov_prof + 1e-6), aspect="auto", origin="lower",
           extent=[0, 256 * dr, 0, 10.0])
plt.colorbar(label="Power [dB]")
plt.xlabel("Range [m]")
plt.ylabel("Time [s]")
plt.title("Moving component vs time (static removed)")
plt.tight_layout()
out2 = Path(__file__).parent / f"motion_range_time_{p.stem}.png"
plt.savefig(out2, dpi=150)
print(f"保存: {out2}")

# 動体エネルギー最大フレームのRDマップも保存しておく
from dat_inspect import rd_map

best = int(top[0])
m = rd_map(x[best], tx=0)
n_dop, n_rng = m.shape
dv = hdr["vel_res_ms"]
plt.figure(figsize=(9, 4))
plt.imshow(20 * np.log10(m + 1e-6), aspect="auto", origin="lower",
           extent=[0, n_rng * dr, -n_dop / 2 * dv, n_dop / 2 * dv])
plt.colorbar(label="Power [dB]")
plt.xlabel("Range [m]")
plt.ylabel("Velocity [m/s]")
plt.title(f"RD map, most-motion frame {best}")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "rd_most_motion.png", dpi=150)
print(f"保存: 0715/rd_most_motion.png (frame {best})")
