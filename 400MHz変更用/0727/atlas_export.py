# ATLAS 録画 .dat → 時刻付き複素 RD テンソルのエクスポータ
#
# カメラ・レーダー連携（0725/camera_radar_labeling_plan.md Step 9）のためのデータ生成が目的。
# 成果物は npz であり、図は確認用の副産物として扱う。
#
# `atlas_dat_parse.py` との役割分担:
#   atlas_dat_parse.py … フォーマット解読と軸の妥当性検証（役割終了・凍結）
#   本ファイル          … 解析パイプラインの入口。位相を保持した複素テンソルを出す
#
# atlas_dat_parse.rd_map() は Rx を非コヒーレント加算して位相を捨てるため RAD を作れない。
# ここでは (TX, RX) 8チャンネルの複素 RD をそのまま保存し、仮想アレイの構成・角度FFTは
# 解析時に回す。TX の順序が未確定（角度の符号が反転しうる）なので、曖昧な判断を
# エクスポート時点に固定しないための設計。
#
# 使い方:
#   python atlas_export.py atlas_log_20260727_195113.dat --outdir out --plot

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from atlas_dat_parse import HEADER_BYTES, load_frames, parse_header

C0 = 3e8   # ベンダーコード（AN24_xx の c0）に合わせる。2.998e8 ではない

# AN24_07 の VirtData 構成（1始まり→0始まり）。物理RXの空間順が 1,0,2,3 であることを意味する。
# (tx, rx) の tx: 0=先行チャープ側, 1=後続チャープ側。素子4のみ重複するため平均する
VIRT_ARRAY = [
    [(0, 1)],
    [(0, 0)],
    [(0, 2)],
    [(1, 1), (0, 3)],   # 重複素子
    [(1, 0)],
    [(1, 2)],
    [(1, 3)],
]


def hanning_matlab(n: int) -> np.ndarray:
    """MATLAB の hanning(N)。np.hanning は MATLAB の hann(N) 相当で両端が厳密に 0 になり、
    N=7（仮想アレイ）では 7 素子のうち 2 素子が消えて実効開口が 5 素子まで落ちる。
    N=256 のレンジ窓では差は無視できるが、素子軸・チャープ軸では効くので揃える"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))


# --------------------------------------------------------------------------
# 物理軸（すべてヘッダから導出する。GUI 設定を変えても自動追従させるため）
# --------------------------------------------------------------------------

def chirp_rate(hdr: dict) -> float:
    """チャープ率 [Hz/s]。1 MHz/µs = 1e12 Hz/s"""
    return hdr["slope_mhz_us"] * 1e12


def range_axis_m(hdr: dict, n_fft: int) -> np.ndarray:
    """AN24_xx 共通: vRange = [0:NFFT-1]/NFFT * fs * c0/(2*kf)
    実数サンプルなので rfft の出力（0〜fs/2 の n_fft//2+1 点）だけが有効"""
    fs = hdr["sampling_ksps"] * 1e3
    k = np.arange(n_fft // 2 + 1)
    return k / n_fft * fs * C0 / (2 * chirp_rate(hdr))


def pri_s(hdr: dict) -> float:
    """同一 TX のチャープ繰り返し間隔。TDM で TX を交互に撃つので TX 数だけ延びる"""
    return hdr["chirp_interval_us"] * 1e-6 * hdr["n_tx"]


def velocity_axis_ms(hdr: dict, n_fft_vel: int) -> np.ndarray:
    """AN24_06: vVel = [-N/2:N/2-1]/N * (1/PRI) * c0/(2*f0)

    MATLAB は中心周波数 fc=(fStrt+fStop)/2 を使うが、ヘッダの max_vel は fStrt を
    使った値と一致する（差 0.4%）。ヘッダとの整合を優先して fStrt を採用"""
    v_freq = np.arange(-(n_fft_vel // 2), n_fft_vel - n_fft_vel // 2) / n_fft_vel / pri_s(hdr)
    return v_freq * C0 / (2 * hdr["start_freq_ghz"] * 1e9)


def angle_axis_deg(n_fft_ant: int = 256) -> np.ndarray:
    """AN24_05/07: vAngDeg = asin(2*[-N/2:N/2-1]/N)/pi*180。素子間隔 λ/2 前提"""
    u = 2 * np.arange(-(n_fft_ant // 2), n_fft_ant - n_fft_ant // 2) / n_fft_ant
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def true_range_res_m(hdr: dict) -> float:
    """真の距離分解能 c/2B。B は 260µs のランプのうち実際にサンプルした 256µs 分。
    ヘッダの range_res はゼロ埋め後の表示刻みであり、これとは別物"""
    fs = hdr["sampling_ksps"] * 1e3
    bw_eff = chirp_rate(hdr) * hdr["n_sample"] / fs
    return C0 / (2 * bw_eff)


# --------------------------------------------------------------------------
# RD テンソル
# --------------------------------------------------------------------------

def rd_tensor(frames: np.ndarray, hdr: dict, n_fft: int = 256,
              n_fft_vel: int | None = None, mti: bool = True) -> np.ndarray:
    """(Frame, Chirp, Rx, Sample) 実数 → (Frame, TX, RX, Doppler, Range) complex64

    n_fft のデフォルトを 256（ゼロ埋めなし）にしてあるのは、1 ビン = 0.846 m = 真の
    分解能セル 1 個という対応を保つため。ヘッダの range_res(0.21 m) は 4 倍ゼロ埋めの
    表示刻みで、分離能力とは無関係なのでパイプラインには持ち込まない"""
    n_tx = hdr["n_tx"]
    n_chirp_tx = frames.shape[1] // n_tx        # 1 TX あたりのチャープ数
    if n_fft_vel is None:
        n_fft_vel = n_chirp_tx                  # ゼロ埋めなし: 1 ビン = 速度分解能 1 個

    # (Frame, TX, Chirp/TX, Rx, Sample) — TDM なので偶数/奇数チャープが別 TX
    x = np.stack([frames[:, tx::n_tx] for tx in range(n_tx)], axis=1)
    x = x.transpose(0, 1, 3, 2, 4)              # (Frame, TX, Rx, Chirp, Sample)

    if mti:
        # チャープ間平均を引き静止クラッタを抑制。センサ固定なのでこれで足りる。
        # 元デモはリアルタイム表示なので入っていない（録画解析では静止物が支配的）
        x = x - x.mean(axis=-2, keepdims=True)

    win_r = hanning_matlab(x.shape[-1])
    rp = np.fft.rfft(x * win_r, n=n_fft, axis=-1) / win_r.sum()   # (..., Chirp, Range)

    win_v = hanning_matlab(rp.shape[-2])
    rd = np.fft.fft(rp * win_v[:, None], n=n_fft_vel, axis=-2) / win_v.sum()
    rd = np.fft.fftshift(rd, axes=-2)                             # (..., Doppler, Range)
    return rd.astype(np.complex64)


def virtual_array_rd(rd_frame: np.ndarray, hdr: dict, n_fft_vel: int,
                     tx_swap: bool = False, doppler_comp: bool = True) -> np.ndarray:
    """(TX, RX, Doppler, Range) → (Virt=7, Doppler, Range)

    doppler_comp: TDM MIMO のドップラー位相補正。TX_b のチャープは TX_a より
    chirp_interval だけ遅いので、動目標では余分な位相 2*pi*f_d*dt が乗り、
    そのまま素子方向 FFT を掛けると角度がずれる。ドップラービン k に対する
    補正量は exp(-j*pi*k/N)（TX 2本の場合）。
    歩行 0.6 m/s では 11 度だが車両 4 m/s では 75 度に達するため、屋外では必須"""
    tx = [rd_frame[1], rd_frame[0]] if tx_swap else [rd_frame[0], rd_frame[1]]

    if doppler_comp:
        k = np.arange(-(n_fft_vel // 2), n_fft_vel - n_fft_vel // 2)
        tx[1] = tx[1] * np.exp(-1j * np.pi * k / n_fft_vel)[None, :, None]

    out = np.empty((len(VIRT_ARRAY),) + rd_frame.shape[-2:], dtype=np.complex64)
    for v, srcs in enumerate(VIRT_ARRAY):
        out[v] = np.mean([tx[t][r] for t, r in srcs], axis=0)   # 重複素子は平均
    return out


def to_ra(rd_frame: np.ndarray, hdr: dict, n_fft_vel: int, n_fft_ant: int = 256,
          **kw) -> np.ndarray:
    """(TX, RX, Doppler, Range) → (Doppler, Angle, Range) 素子方向 FFT（AN24_05/07 の DBF）"""
    virt = virtual_array_rd(rd_frame, hdr, n_fft_vel, **kw)     # (Virt, Doppler, Range)
    win = hanning_matlab(virt.shape[0])
    j = np.fft.fft(virt * win[:, None, None], n=n_fft_ant, axis=0) / win.sum()
    return np.fft.fftshift(j, axes=0).transpose(1, 0, 2)


# --------------------------------------------------------------------------
# 時刻
# --------------------------------------------------------------------------

def frame_times(path: Path, hdr: dict, n_frame: int):
    """フレーム時刻を生成する。

    .dat にはフレームごとのタイムスタンプが無い（atlas_dat_format.md §7-1）ため、
    ファイル名の日時 ＋ frame_period_ms の等間隔で合成するしかない。
    取りこぼしが 1 フレームでもあると以降が period 分ずつずれ、しかもファイルからは
    検出できない。絶対時刻は目安であり、カメラとの対応は同期イベントで取ること"""
    dt = hdr["frame_period_ms"] / 1000.0
    t_rel = np.arange(n_frame) * dt

    m = re.search(r"(\d{8})_(\d{6})", path.name)
    if m is None:
        return t_rel, None
    t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return t_rel, t0


def check_frame_integrity(path: Path, hdr: dict) -> tuple[int, bool]:
    """データ部がフレーム長で割り切れるかを見る。割り切れなければ書き込み途中か破損。
    ただし「等間隔で落ちた」ドロップはこの検査では検出できない（同期イベントが必要）"""
    n_byte = path.stat().st_size - HEADER_BYTES
    per_frame = hdr["n_chirp_per_frame"] * hdr["n_rx"] * hdr["n_sample"] * 2
    return n_byte // per_frame, (n_byte % per_frame == 0)


def find_sync_events(rd: np.ndarray, vel_ms: np.ndarray, range_m: np.ndarray,
                     vel_res: float, r_max: float = 6.0, k_mad: float = 5.0,
                     min_gap: int = 3):
    """同期イベント（金属板を振る）のフレーム番号を返す。

    板は近距離で速く振るので、近距離ゲート内の非ゼロドップラー成分が突発的に跳ねる。
    中央値と MAD で外れ値として拾う。録画の先頭と末尾の 2 回入れておけば、
    その間のフレーム数が期待値と一致することでドロップの有無を判定できる"""
    keep_d = np.abs(vel_ms) >= vel_res      # クラッタ主ローブ相当（±分解能）を除外
    keep_r = range_m <= r_max
    a = np.abs(rd)[..., keep_d, :][..., keep_r]
    e = a.sum(axis=(1, 2, 3, 4))            # (Frame,) 動体エネルギー

    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-12
    z = (e - med) / (1.4826 * mad)

    cand = np.where(z > k_mad)[0]
    events = []
    for i in cand:
        if events and i - events[-1][0] <= min_gap:
            if z[i] > events[-1][1]:        # 連続候補は最大の 1 点にまとめる
                events[-1] = (int(i), float(z[i]))
        else:
            events.append((int(i), float(z[i])))
    return events, e, z


# --------------------------------------------------------------------------

def export_npz(out: Path, rd, hdr, range_m, vel_ms, angle_deg, t_rel, t0,
               sync_frames, meta: dict):
    np.savez_compressed(
        out,
        rd=rd,                              # (Frame, TX, RX, Doppler, Range) complex64
        range_m=range_m,
        vel_ms=vel_ms,
        angle_deg=angle_deg,                # to_ra を掛けたときの角度軸（参考値）
        t_rel_s=t_rel,
        t0_iso=("" if t0 is None else t0.isoformat()),
        frame_period_s=hdr["frame_period_ms"] / 1000.0,
        sync_frames=np.array(sync_frames, dtype=int),
        virt_array=json.dumps(VIRT_ARRAY),
        header=json.dumps(hdr),
        meta=json.dumps(meta),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--n-fft", type=int, default=256,
                    help="レンジFFT点数。既定 256=ゼロ埋めなし（1ビン=真の分解能 0.846m）")
    ap.add_argument("--n-fft-vel", type=int, default=None,
                    help="ドップラーFFT点数。既定はチャープ数（ゼロ埋めなし）")
    ap.add_argument("--no-mti", action="store_true", help="静止クラッタ抑制を無効化（ON/OFF比較用）")
    ap.add_argument("--tx-swap", action="store_true", help="TX の順序を入れ替える（角度の符号が反転）")
    ap.add_argument("--no-doppler-comp", action="store_true", help="TDM ドップラー位相補正を無効化")
    ap.add_argument("--sync-rmax", type=float, default=6.0, help="同期イベント探索の距離ゲート[m]")
    ap.add_argument("--plot", action="store_true", help="確認用の図も出す")
    ap.add_argument("--frame", type=int, default=None, help="RD/RA 図にするフレーム")
    args = ap.parse_args()

    hdr = parse_header(args.dat.read_bytes()[:HEADER_BYTES])
    n_frame_hdr, ok = check_frame_integrity(args.dat, hdr)
    frames = load_frames(args.dat, hdr)      # (Frame, Chirp, Rx, Sample)
    n_frame = frames.shape[0]

    print(f"{args.dat.name}")
    print(f"  frames={n_frame}  chirps/frame={hdr['n_chirp_per_frame']}  "
          f"TX={hdr['n_tx']} RX={hdr['n_rx']}  period={hdr['frame_period_ms']}ms")
    if not ok:
        print("  [警告] データ部がフレーム長で割り切れない。書き込み途中か破損の可能性")

    n_tx = hdr["n_tx"]
    n_fft_vel = args.n_fft_vel or (hdr["n_chirp_per_frame"] // n_tx)
    mti = not args.no_mti

    rd = rd_tensor(frames, hdr, n_fft=args.n_fft, n_fft_vel=n_fft_vel, mti=mti)
    range_m = range_axis_m(hdr, args.n_fft)
    vel_ms = velocity_axis_ms(hdr, n_fft_vel)
    angle_deg = angle_axis_deg()
    vel_res = hdr["vel_res_ms"]

    print(f"  RD tensor {rd.shape} (Frame, TX, RX, Doppler, Range) complex64  "
          f"{rd.nbytes/1e6:.1f} MB")
    print(f"  range  {range_m[0]:.2f}〜{range_m[-1]:.2f} m  刻み {range_m[1]-range_m[0]:.4f} m"
          f"（真の分解能 {true_range_res_m(hdr):.3f} m）")
    print(f"  vel    {vel_ms[0]:+.2f}〜{vel_ms[-1]:+.2f} m/s  刻み {vel_ms[1]-vel_ms[0]:.4f} m/s"
          f"（ヘッダ vel_res {vel_res:.4f}）")
    print(f"  MTI={'ON' if mti else 'OFF'}  tx_swap={args.tx_swap}  "
          f"doppler_comp={not args.no_doppler_comp}")

    t_rel, t0 = frame_times(args.dat, hdr, n_frame)
    print(f"  t0={'不明（ファイル名から取得できず）' if t0 is None else t0.isoformat()}"
          f"  期間 {t_rel[-1]:.1f} s")

    events, energy, z = find_sync_events(rd, vel_ms, range_m, vel_res, r_max=args.sync_rmax)
    if events:
        print(f"  同期イベント候補 {len(events)} 件（近距離 <{args.sync_rmax:.0f}m の動体エネルギー外れ値）:")
        for f, zz in events:
            print(f"    frame {f:4d}  t={t_rel[f]:5.2f} s  z={zz:.1f}")
        if len(events) >= 2:
            df = events[-1][0] - events[0][0]
            print(f"    先頭〜末尾: {df} フレーム = {df*hdr['frame_period_ms']/1000:.2f} s")
            print("    → カメラ側の同期イベント間隔と一致すればドロップ無しと言える")
    else:
        print(f"  同期イベントは検出されず。カメラとの時刻対応は付けられない")
        print("    → 次回録画では先頭と末尾に金属板を振る動作を必ず入れること")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / (args.dat.stem + "_rd.npz")
    meta = dict(n_fft=args.n_fft, n_fft_vel=n_fft_vel, mti=mti, tx_swap=args.tx_swap,
                doppler_comp=not args.no_doppler_comp,
                true_range_res_m=true_range_res_m(hdr),
                axes="rd: (Frame, TX, RX, Doppler, Range) complex64",
                note="range_m は真の分解能グリッド。ヘッダ range_res は表示刻みなので使わない")
    export_npz(out, rd, hdr, range_m, vel_ms, angle_deg, t_rel, t0,
               [f for f, _ in events], meta)
    print(f"\n保存: {out}  ({out.stat().st_size/1e6:.1f} MB)")

    if args.plot:
        _plots(args, hdr, rd, range_m, vel_ms, angle_deg, t_rel, vel_res, energy, events, n_fft_vel)


def _plots(args, hdr, rd, range_m, vel_ms, angle_deg, t_rel, vel_res, energy, events, n_fft_vel):
    """確認用。データ成果物は npz であり、図はあくまで人間が目で確かめるためのもの"""
    import matplotlib.pyplot as plt

    keep_d = np.abs(vel_ms) >= vel_res
    mov = np.abs(rd)[..., keep_d, :].sum(axis=(1, 2, 3))     # (Frame, Range) 動体のみ

    # --- range-time ---
    db = 20 * np.log10(mov + 1e-12)
    db -= db.max()
    plt.figure(figsize=(9, 4))
    plt.imshow(db, aspect="auto", origin="lower", vmin=-40, vmax=0,
               extent=[range_m[0], range_m[-1], t_rel[0], t_rel[-1]])
    plt.colorbar(label="Power [dB] (rel. max)")
    for f, _ in events:
        plt.axhline(t_rel[f], color="w", ls="--", lw=1)
    plt.xlim(0, min(30, range_m[-1]))
    plt.xlabel("Range [m]")
    plt.ylabel("Time [s]")
    # 図中は日本語フォント未設定のため英語（0715/ のスクリプトと同じ方針）
    plt.title(f"{args.dat.name}  range-time (dashed = sync candidates)")
    plt.tight_layout()
    p = args.outdir / "export_range_time.png"
    plt.savefig(p, dpi=150)
    print(f"保存: {p}")

    fidx = args.frame if args.frame is not None else int(mov.max(axis=1).argmax())

    # --- RD（TX/RX 非コヒーレント加算。可視化用であって解析用ではない）---
    m = np.abs(rd[fidx]).sum(axis=(0, 1))                    # (Doppler, Range)
    plt.figure(figsize=(9, 4))
    plt.imshow(20 * np.log10(m + 1e-12), aspect="auto", origin="lower",
               extent=[range_m[0], range_m[-1], vel_ms[0], vel_ms[-1]])
    plt.colorbar(label="Power [dB]")
    plt.xlim(0, min(30, range_m[-1]))
    plt.xlabel("Range [m]")
    plt.ylabel("Radial velocity [m/s]")
    plt.title(f"{args.dat.name}  RD  frame {fidx} (t={t_rel[fidx]:.1f}s)")
    plt.tight_layout()
    p = args.outdir / "export_rd.png"
    plt.savefig(p, dpi=150)
    print(f"保存: {p}")

    # --- RA（最も強い動体ドップラービンを選んでから DBF）---
    ra = to_ra(rd[fidx], hdr, n_fft_vel, tx_swap=args.tx_swap,
               doppler_comp=not args.no_doppler_comp)        # (Doppler, Angle, Range)
    p_dr = np.abs(rd[fidx]).sum(axis=(0, 1)).copy()
    p_dr[~keep_d] = 0
    d_i = int(np.unravel_index(p_dr.argmax(), p_dr.shape)[0])
    j = 20 * np.log10(np.abs(ra[d_i]) + 1e-12)
    j -= j.max()
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(angle_deg, range_m, np.clip(j, -18, 0).T, shading="auto", cmap="jet")
    plt.colorbar(label="normalized [dB] (clipped at -18, as in AN24_07)")
    plt.xlabel("Angle [deg]")
    plt.ylabel("Range [m]")
    plt.xlim(-60, 60)
    plt.ylim(0, min(30, range_m[-1]))
    plt.title(f"{args.dat.name}  RA  frame {fidx}, v={vel_ms[d_i]:+.2f} m/s")
    plt.tight_layout()
    p = args.outdir / "export_ra.png"
    plt.savefig(p, dpi=150)
    print(f"保存: {p}")


if __name__ == "__main__":
    main()
