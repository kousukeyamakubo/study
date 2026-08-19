# ハードウェア設定（最大距離・サンプル数）を変えたときに atlas_export.py が追従するかの検証
#
# 実機で設定を変えて録り直す前に、コード側の追従性だけを切り分けて確認するためのもの。
# 既知の正解（目標距離・速度）を埋め込んだ合成 .dat を作り、パイプラインが復元できるかを見る。
#
# 検証したいのは主に 2 点:
#   1) range/velocity 軸がヘッダから正しく再計算されるか（fs・slope・サンプル数を振る）
#   2) --n-fft の既定値 256 がリテラルであることによる取りこぼしが実害を出すか
#      → 分解能ぎりぎりで並べた 2 目標が分離できるかで判定する
#
# 実機でしか検証できないこと（parse_header のオフセット推定が正しいか）は対象外。
#
# 使い方:
#   python verify_config_change.py

import struct
from pathlib import Path

import numpy as np

from atlas_dat_parse import HEADER_BYTES, load_frames, parse_header
from atlas_export import C0, chirp_rate, range_axis_m, rd_tensor, true_range_res_m, velocity_axis_ms

OUT = Path("verify_out")


def make_header(cfg: dict) -> bytes:
    """parse_header が読むオフセットに値を書き戻す。表示系の値（range_res 等）は
    実機と同じ「4倍ゼロ埋めの表示刻み」の慣習に合わせて計算しておく"""
    h = bytearray(HEADER_BYTES)
    kf = cfg["slope_mhz_us"] * 1e12
    fs = cfg["sampling_ksps"] * 1e3
    n_disp = cfg["n_sample"]                       # 表示ビン数（実機は 256 だった）
    range_res = fs * C0 / (2 * kf) / (4 * cfg["n_sample"])   # 4倍ゼロ埋め時の刻み
    pri = cfg["chirp_interval_us"] * 1e-6 * cfg["n_tx"]
    v_max = C0 / (4 * cfg["start_freq_ghz"] * 1e9 * pri)
    n_dop = cfg["n_chirp_per_frame"] // cfg["n_tx"]

    struct.pack_into("<d", h, 0x0C, cfg["start_freq_ghz"])
    struct.pack_into("<d", h, 0x14, cfg["slope_mhz_us"])
    struct.pack_into("<i", h, 0x1C, cfg["n_sample"])
    struct.pack_into("<i", h, 0x20, cfg["n_chirp_per_frame"])
    struct.pack_into("<i", h, 0x24, cfg["sampling_ksps"])
    struct.pack_into("<i", h, 0x30, cfg["frame_period_ms"])
    struct.pack_into("<i", h, 0x34, cfg["n_tx"])
    struct.pack_into("<i", h, 0x38, cfg["n_rx"])
    struct.pack_into("<d", h, 0x3C, cfg["chirp_interval_us"])
    struct.pack_into("<d", h, 0x44, range_res)
    struct.pack_into("<d", h, 0x4C, range_res * (n_disp - 1))
    struct.pack_into("<d", h, 0x54, v_max)
    struct.pack_into("<d", h, 0x5C, 2 * v_max / n_dop)
    return bytes(h)


def synth_frames(cfg: dict, targets: list, n_frame: int = 4, snr_db: float = 30.0,
                 seed: int = 0) -> np.ndarray:
    """実数 IF 信号を合成する。(Frame, Chirp, Rx, Sample) int16

    ATLAS は IQ ではなく実数サンプルなので cos で作る。ここを複素で作ってしまうと
    片側スペクトルの検証にならない（実機との最重要の差分）"""
    rng = np.random.default_rng(seed)
    kf, fs = chirp_rate(cfg_hdr(cfg)), cfg["sampling_ksps"] * 1e3
    f0 = cfg["start_freq_ghz"] * 1e9
    lam = C0 / f0
    n_sp, n_ch, n_rx = cfg["n_sample"], cfg["n_chirp_per_frame"], cfg["n_rx"]
    t_ci = cfg["chirp_interval_us"] * 1e-6

    t = np.arange(n_sp) / fs                                   # fast-time
    out = np.zeros((n_frame, n_ch, n_rx, n_sp))
    for f in range(n_frame):
        for c in range(n_ch):
            # TDM: チャープ index がそのまま送信時刻。TX 交互の遅れもここに含まれる
            t_slow = f * cfg["frame_period_ms"] * 1e-3 + c * t_ci
            for tgt in targets:
                r = tgt["r"] + tgt["v"] * t_slow
                fb = 2 * kf * r / C0                           # ビート周波数
                ph_d = 4 * np.pi * r / lam                     # slow-time 位相（ドップラー）
                # RX 素子間位相差。素子間隔 λ/2 前提
                d_ph = np.pi * np.sin(np.radians(tgt.get("ang", 0.0))) * np.arange(n_rx)
                out[f, c] += tgt["a"] * np.cos(
                    2 * np.pi * fb * t[None, :] + ph_d + d_ph[:, None])

    p_sig = (out ** 2).mean()
    out += rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), out.shape)
    return (out / np.abs(out).max() * 12000).astype("<i2")


def cfg_hdr(cfg: dict) -> dict:
    """chirp_rate() など hdr を取る関数に cfg をそのまま渡すためのアダプタ"""
    return cfg


def write_dat(path: Path, cfg: dict, targets: list) -> Path:
    path.write_bytes(make_header(cfg) + synth_frames(cfg, targets).tobytes())
    return path


def expected_range(tgt: dict, cfg: dict, frame: int) -> float:
    """評価するフレームの中央時刻での距離。目標は動いているので初期距離と比べてはいけない
    （フレーム内の 32 チャープ分だけ進むため、その中央を代表点とする）"""
    t = frame * cfg["frame_period_ms"] * 1e-3 + \
        (cfg["n_chirp_per_frame"] / 2) * cfg["chirp_interval_us"] * 1e-6
    return tgt["r"] + tgt["v"] * t


def nearest_peak(prof: np.ndarray, axis: np.ndarray, r_true: float, win_m: float) -> float:
    """正解位置の近傍で最大となる位置[m]。サイドローブに飛ばないよう窓で拘束する。
    窓は目標間隔の半分より狭くする（広いと 2 目標が同じピークを拾ってしまう）"""
    i = np.where(np.abs(axis - r_true) <= win_m)[0]
    return axis[i[prof[i].argmax()]]


def is_resolved(prof: np.ndarray, axis: np.ndarray, r1: float, r2: float,
                dip_db: float = 3.0) -> bool:
    """2 目標が分離できたかを谷の深さで判定する。単に「ピークが2個取れた」では
    サイドローブを数えてしまうので、間に dip_db 以上の谷があることを要求する"""
    i1, i2 = (int(np.abs(axis - r).argmin()) for r in (r1, r2))
    lo, hi = sorted((i1, i2))
    if hi - lo < 2:
        return False
    seg = 20 * np.log10(prof + 1e-12)
    valley = seg[lo + 1:hi].min()
    return min(seg[lo], seg[hi]) - valley >= dip_db


def run_case(name: str, cfg: dict, targets: list, n_fft_modes: dict):
    dat = write_dat(OUT / f"{name}.dat", cfg, targets)
    hdr = parse_header(dat.read_bytes()[:HEADER_BYTES])
    frames = load_frames(dat, hdr)

    print(f"\n=== {name} ===")
    print(f"  cfg: n_sample={hdr['n_sample']} fs={hdr['sampling_ksps']}ksps "
          f"slope={hdr['slope_mhz_us']:.4f}MHz/us n_chirp={hdr['n_chirp_per_frame']}")

    # ヘッダの往復（書いた値がそのまま読めるか）
    for k in ("n_sample", "sampling_ksps", "n_chirp_per_frame", "n_tx", "n_rx"):
        assert hdr[k] == cfg[k], f"header roundtrip failed: {k}"

    r_max_theory = hdr["sampling_ksps"] * 1e3 * C0 / (4 * chirp_rate(hdr))
    res_true = true_range_res_m(hdr)
    frame_eval = 3                                        # 合成は 4 フレーム。最後を見る
    r_true = [expected_range(t, cfg, frame_eval) for t in targets]
    print(f"  理論: 最大距離 {r_max_theory:.1f} m  真の分解能 {res_true:.3f} m")
    print(f"  正解(frame {frame_eval}): " +
          ", ".join(f"{r:.2f}m/{t['v']:+.1f}m/s" for r, t in zip(r_true, targets)))
    if len(targets) == 2:
        sep = abs(r_true[1] - r_true[0])
        print(f"        目標間隔 {sep:.2f} m → 物理的に分離{'可能' if sep > res_true else '不可能'}"
              f"（分解能 {res_true:.2f} m）")

    n_dop = hdr["n_chirp_per_frame"] // hdr["n_tx"]
    for label, n_fft in n_fft_modes.items():
        rd = rd_tensor(frames, hdr, n_fft=n_fft, n_fft_vel=n_dop)
        r_ax = range_axis_m(hdr, n_fft)
        v_ax = velocity_axis_ms(hdr, n_dop)

        a = np.abs(rd[frame_eval]).sum(axis=(0, 1))       # (Doppler, Range)
        d_i = int(a.max(axis=1).argmax())
        prof = a[d_i]
        win = 2.0 if len(targets) == 1 else 0.45 * abs(r_true[1] - r_true[0])
        found = [nearest_peak(prof, r_ax, r, win) for r in r_true]
        err = max(abs(f - r) for f, r in zip(found, r_true))
        v_err = abs(v_ax[d_i] - targets[0]["v"])

        print(f"  [{label:>14}] n_fft={n_fft:4d}  軸範囲 0〜{r_ax[-1]:.1f}m  刻み {r_ax[1]:.3f}m")
        print(f"                   検出 {['%.2f' % x for x in found]} m（誤差 {err:.2f} m "
              f"= {err/res_true:.2f} 分解能セル）  v={v_ax[d_i]:+.2f} m/s（誤差 {v_err:.2f}）")
        if len(targets) == 2:
            # 正解位置ではなく実際に検出した位置で谷を見る（マージすると2つが同一ビンになる）
            ok = is_resolved(prof, r_ax, *found)
            print(f"                   2目標の分離: {'OK（谷あり）' if ok else 'NG（1山にマージ）'}")


def main():
    OUT.mkdir(exist_ok=True)

    base = dict(start_freq_ghz=24.06, slope_mhz_us=0.6923076923076924, n_sample=256,
                n_chirp_per_frame=32, sampling_ksps=1000, frame_period_ms=200,
                n_tx=2, n_rx=4, chirp_interval_us=325.0)

    # 実機と同じ設定。0.85 m 分解能では 5.0/5.6 m は分離できないのが物理的に正しい
    run_case("A_current", base, [dict(r=5.0, v=1.2, a=1.0), dict(r=6.2, v=1.2, a=1.0)],
             {"既定(256)": 256, "n_sample連動": 256})

    # サンプル数だけ倍：帯域が倍→分解能 0.42 m。既定の 256 だと後半が捨てられる
    cfg_b = dict(base, n_sample=512)
    run_case("B_nsample512", cfg_b, [dict(r=5.0, v=1.2, a=1.0), dict(r=6.2, v=1.2, a=1.0)],
             {"既定(256)": 256, "n_sample連動": 512})

    # 最大距離を倍にする典型操作：fs を倍にする。サンプル数も倍にしないと帯域が縮む
    cfg_c = dict(base, sampling_ksps=2000, n_sample=512)
    run_case("C_fs2x_range2x", cfg_c, [dict(r=20.0, v=1.2, a=1.0)],
             {"既定(256)": 256, "n_sample連動": 512})

    # チャープ数を倍：速度軸の分解能が半分になる。ドップラー側の追従確認
    cfg_d = dict(base, n_chirp_per_frame=64)
    run_case("D_nchirp64", cfg_d, [dict(r=8.0, v=2.4, a=1.0)],
             {"既定(256)": 256, "n_sample連動": 256})


if __name__ == "__main__":
    main()
