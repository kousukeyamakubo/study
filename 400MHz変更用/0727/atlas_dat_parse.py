# ATLAS 評価キット（AtlasDemoKitApp）の録画 .dat を解析し RD マップ／range-time を生成する
#
# 【役割】フォーマット解読と軸の妥当性検証のための確認用スクリプト。役割は完了しており、
# ここから機能を足さない。解析パイプラインの入口は `atlas_export.py`（複素 RD を保持）。
# 本ファイルの rd_map() は Rx を非コヒーレント加算して位相を捨てるため RAD は作れない。
#
# .dat の内部フォーマットは非公開。本ファイル（0727/atlas_log_20260727_195113.dat、
# 人が約4m先からレーダーへ歩いてくる10秒録画）の解析で以下を確定させた。
# 詳細な根拠は 0727/atlas_dat_format.md を参照。
#
#   [0:256]   ヘッダ。Cfg 由来のパラメータが double / int32 (LE) で並ぶ
#   [256:]    データ部。1フレーム = 32チャープ × 4RX × 256サンプル × int16 = 65,536 word
#             **実数サンプル**（IQ ではない。T14RE の RAW と異なる最重要の差分）
#
# チャープ軸は TDM で TX 交互（偶数index=TX_a / 奇数index=TX_b）。
# 同一 TX の PRI = 650 µs（= チャープ間隔 325 µs × 2）で、これが最大速度 ±4.7957 m/s を決める。
#
# 使い方:
#   python atlas_dat_parse.py atlas_log_20260727_195113.dat --outdir .

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HEADER_BYTES = 256
N_FFT = 1024  # 256サンプルを4倍ゼロ埋め。ヘッダの range_res がこの刻みと一致する（後述）

C_LIGHT = 3e8


def hanning_matlab(n: int) -> np.ndarray:
    """MATLAB の hanning(N)。np.hanning は MATLAB の hann(N) 相当で両端が厳密に 0 になり、
    N=16（チャープ軸）では 16 本のうち 2 本が捨てられ Doppler 主ローブが広がる"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))


def parse_header(h: bytes) -> dict:
    """256byte ヘッダから判明済みパラメータを取り出す（オフセットは実データ解析による推定）"""
    d = lambda off: struct.unpack_from("<d", h, off)[0]
    i = lambda off: struct.unpack_from("<i", h, off)[0]
    return dict(
        start_freq_ghz=d(0x0C),      # 24.06
        slope_mhz_us=d(0x14),        # 0.6923
        n_sample=i(0x1C),            # 256
        n_chirp_per_frame=i(0x20),   # 32（TX2本の合計。1TXあたり16）
        sampling_ksps=i(0x24),       # 1000
        bw_ghz=d(0x28),              # 0.1772308（実効帯域。slope×n_sample/fs と厳密一致）
        frame_period_ms=i(0x30),     # 200
        n_tx=i(0x34),                # 2  ← T14RE(0x44) から 0x10 ずれている
        n_rx=i(0x38),                # 4
        chirp_interval_us=d(0x3C),   # 325.0（TX交互なので同一TXのPRIはこの2倍）
        range_res_m=d(0x44),         # 0.2115885（表示グリッド。真の分解能は c/2B≈0.85m）
        max_range_m=d(0x4C),         # 53.955 = 255 * range_res
        max_vel_ms=d(0x54),          # 4.7957
        vel_res_ms=d(0x5C),          # 0.5995
    )


def load_frames(path: Path, hdr: dict) -> np.ndarray:
    """データ部を (Frame, Chirp, Rx, Sample) の実数配列に整形する"""
    raw = np.fromfile(path, dtype="<i2", offset=HEADER_BYTES).astype(np.float32)
    n_ch, n_rx, n_sp = hdr["n_chirp_per_frame"], hdr["n_rx"], hdr["n_sample"]
    per_frame = n_ch * n_rx * n_sp
    n_frame = raw.size // per_frame
    return raw[: n_frame * per_frame].reshape(n_frame, n_ch, n_rx, n_sp)


def range_fft(x: np.ndarray) -> np.ndarray:
    """実数IF信号のレンジFFT。片側スペクトルのみが意味を持つので前半だけ返す
    (..., Sample) -> (..., N_FFT//4) すなわちヘッダの max_range までの256ビン"""
    return np.fft.rfft(x * hanning_matlab(x.shape[-1]), n=N_FFT, axis=-1)[..., : N_FFT // 4]


def rd_map(frame: np.ndarray, tx: int = 0) -> np.ndarray:
    """1フレームの RD マップ (Doppler, Range)。
    TDM なので同一 TX のチャープだけを抜き出さないと位相が飛んで Doppler が崩れる"""
    z = frame[tx::2]                                   # (16, Rx, Sample) 同一TXの列
    z = z - z.mean(axis=0, keepdims=True)              # チャープ間平均を引き静止クラッタを抑制
    r = range_fft(z)                                   # (16, Rx, Range)
    n_dop = r.shape[0]
    d = np.fft.fftshift(np.fft.fft(r * hanning_matlab(n_dop)[:, None, None], axis=0), axes=0)
    return np.abs(d).sum(axis=1)                       # (Doppler, Range) Rx非コヒーレント加算


def track_from_rd(rd_all: np.ndarray, vel_res: float, range_res: float, max_vel: float,
                  frame_period_s: float, min_bin: int = 3, half_win: int = 3):
    """各フレームの「動いている応答」からレンジと動径速度の時系列を取る。
    Doppler=0 ビンは静止クラッタ残差なので除外する。

    単純な argmax は使えない。真の距離分解能 0.85 m に対し表示刻みが 0.21 m と細かく、
    人体の応答は数ビンに広がってレンジサイドローブや室内マルチパスの尾を引くため、
    副ピークへ飛ぶ（実データで t=1.0s に 6.35m へ飛ぶ現象を確認）。
    そこでフレーム間の連続性で拘束する: 1フレームで移動しうる距離は最大速度で上限が決まる。"""
    n_frame, n_dop, n_rng = rd_all.shape
    dc = n_dop // 2
    mov = np.delete(rd_all, dc, axis=1).max(axis=1)  # (Frame, Range) 動体のみ
    mov[:, :min_bin] = 0                             # 送信リーク（近傍ビン）

    gate = int(np.ceil(max_vel * frame_period_s / range_res))  # フレーム間の物理的な移動上限[bin]
    r_bin = np.zeros(n_frame, dtype=int)
    start = int(mov.max(axis=1).argmax())            # 動体応答が最も強いフレームから両方向へ伸ばす
    r_bin[start] = mov[start].argmax()
    for f in range(start - 1, -1, -1):
        r_bin[f] = _peak_near(mov[f], r_bin[f + 1], gate)
    for f in range(start + 1, n_frame):
        r_bin[f] = _peak_near(mov[f], r_bin[f - 1], gate)

    # 主ローブ内の電力重心でサブビン精度にする
    rng = np.empty(n_frame)
    for f in range(n_frame):
        lo, hi = max(0, r_bin[f] - half_win), min(n_rng, r_bin[f] + half_win + 1)
        w = mov[f, lo:hi]
        rng[f] = (w * np.arange(lo, hi)).sum() / max(w.sum(), 1e-12) * range_res

    # レンジ追跡（mov）は DC を除いているので速度読み出しも揃える。
    # 揃えないとクラッタが強い距離ビンで 0 m/s が返り、両者の基準がずれる
    prof = np.delete(rd_all, dc, axis=1)             # (Frame, Doppler-1, Range)
    d_bin = np.array([prof[f, :, r_bin[f]].argmax() for f in range(n_frame)])
    d_bin[d_bin >= dc] += 1                          # 削除した DC の分をインデックスに戻す
    return rng, (d_bin - dc) * vel_res


def _peak_near(prof: np.ndarray, center: int, gate: int) -> int:
    """前フレームの推定位置から gate ビン以内で最大の応答を返す"""
    lo, hi = max(0, center - gate), min(len(prof), center + gate + 1)
    return lo + int(prof[lo:hi].argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--frame", type=int, default=None, help="RDマップにするフレーム（省略時は動体が最強のフレーム）")
    args = ap.parse_args()

    b = args.dat.read_bytes()
    hdr = parse_header(b[:HEADER_BYTES])
    print(f"ファイル: {args.dat.name}  {len(b):,} byte")
    for k, v in hdr.items():
        print(f"  {k}: {v}")

    x = load_frames(args.dat, hdr)  # (Frame, Chirp, Rx, Sample)
    n_frame = x.shape[0]
    dt = hdr["frame_period_ms"] / 1000.0
    print(f"\nデータ部 shape: {x.shape}  = {n_frame} フレーム × {dt*1000:.0f} ms = {n_frame*dt:.1f} 秒")
    print(f"振幅: min={x.min():.0f} max={x.max():.0f}  （int16 実数サンプル）")

    # 帯域幅から出る真の距離分解能。ヘッダの range_res はゼロ埋め後の表示刻みにすぎない
    bw_hz = hdr["slope_mhz_us"] * 1e6 / 1e-6 * (hdr["n_sample"] / (hdr["sampling_ksps"] * 1e3))
    true_res = C_LIGHT / (2 * bw_hz)
    print(f"\n帯域幅 {bw_hz/1e6:.1f} MHz → 真の距離分解能 c/2B = {true_res:.3f} m")
    print(f"ヘッダ range_res {hdr['range_res_m']:.4f} m = 真の分解能 / {true_res/hdr['range_res_m']:.1f}"
          f"（{N_FFT//hdr['n_sample']}倍ゼロ埋めの表示刻みと一致）")

    rd_all = np.stack([rd_map(x[f]) for f in range(n_frame)])  # (Frame, Doppler, Range)
    rng, vel = track_from_rd(rd_all, hdr["vel_res_ms"], hdr["range_res_m"],
                             hdr["max_vel_ms"], dt)

    # レンジ変化率と Doppler は独立に得られるので、両者の一致が軸の妥当性検証になる。
    # 物体が止まっている区間はピークがノイズに飛ぶため、Doppler 符号が続く最長区間だけで評価する
    t = np.arange(n_frame) * dt
    sign = np.sign(vel)
    best_len, best = 0, slice(0, 0)
    i = 0
    while i < n_frame:
        j = i
        while j + 1 < n_frame and sign[j + 1] == sign[i] and sign[i] != 0:
            j += 1
        if j - i + 1 > best_len:
            best_len, best = j - i + 1, slice(i, j + 1)
        i = j + 1
    slope = np.polyfit(t[best], rng[best], 1)[0]
    print(f"\n運動区間: t = {t[best][0]:.1f} 〜 {t[best][-1]:.1f} s（{best_len} フレーム）")
    print(f"  レンジ変化率（直線近似）: {slope:+.2f} m/s")
    print(f"  Doppler ピーク（中央値）: {np.median(vel[best]):+.2f} m/s "
          f"（分解能 {hdr['vel_res_ms']:.2f} m/s → ビン幅 ±{hdr['vel_res_ms']/2:.2f}）")
    print(f"  → 両者が独立に一致すればレンジ軸・速度軸の刻みが妥当と言える")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- range-time: Doppler≠0 の最大値をとり動体だけを残す ---
    dc = rd_all.shape[1] // 2
    mov = np.delete(rd_all, dc, axis=1).max(axis=1)  # (Frame, Range)
    db = 20 * np.log10(mov + 1e-9)
    db -= db.max()
    max_r = rd_all.shape[2] * hdr["range_res_m"]
    plt.figure(figsize=(9, 4))
    plt.imshow(db, aspect="auto", origin="lower", vmin=-40, vmax=0,
               extent=[0, max_r, 0, n_frame * dt])
    plt.colorbar(label="Power [dB] (rel. max)")
    plt.plot(rng, t, "r.-", lw=1, ms=4, label="moving peak")
    plt.xlim(0, 12)
    plt.xlabel("Range [m]")
    plt.ylabel("Time [s]")
    # 図中は日本語フォント未設定のため英語（0715/ のスクリプトと同じ方針）
    plt.title(f"{args.dat.name}  range-time (static clutter removed)")
    plt.legend()
    plt.tight_layout()
    p = args.outdir / "range_time.png"
    plt.savefig(p, dpi=150)
    print(f"\n保存: {p}")

    # --- 代表フレームの RD マップ ---
    fidx = args.frame if args.frame is not None else int(mov[:, 3:].max(axis=1).argmax())
    m = rd_all[fidx]
    v_max = rd_all.shape[1] / 2 * hdr["vel_res_ms"]
    plt.figure(figsize=(9, 4))
    plt.imshow(20 * np.log10(m + 1e-9), aspect="auto", origin="lower",
               extent=[0, max_r, -v_max, v_max])
    plt.colorbar(label="Power [dB]")
    plt.xlim(0, 12)
    plt.xlabel("Range [m]")
    plt.ylabel("Radial velocity [m/s]")
    plt.title(f"{args.dat.name}  frame {fidx} (t={fidx*dt:.1f}s, TX0)")
    plt.tight_layout()
    p = args.outdir / "rd_map.png"
    plt.savefig(p, dpi=150)
    print(f"保存: {p}")

    print("\nフレームごとの動体ピーク（レンジ / 動径速度）:")
    for f in range(n_frame):
        print(f"  t={f*dt:4.1f}s  {rng[f]:5.2f} m  {vel[f]:+5.2f} m/s")


if __name__ == "__main__":
    main()
