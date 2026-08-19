# ベンダー提供 MATLAB デモコード（0727/matlab/AN24_02,05,06,07.m）の Python 移植
#
# 元コードは USB 経由で実機から取得する（Brd.BrdGetData()）ため、ファイル読み込み部は存在しない。
# ここでは取得部を `.dat` 読み込み（atlas_dat_parse.load_frames）に差し替え、
# 信号処理チェーンだけをそのまま移植している。
#
# 移植で意図的に変えた点:
#   - 実機ストリームの軸順 (Rx が最も遅い) と `.dat` の軸順 (Chirp, Rx, Sample) は異なる。
#     .dat 側は実測で確定させた（atlas_dat_format.md §3.1）。MATLAB の reshape をそのまま
#     持ってくると壊れるので注意
#   - Brd.FuSca（ADCコード → 電圧の換算係数）は Class フォルダにあり未入手のため 1.0。
#     絶対値 [dBV] は出せないが、相対値の議論には影響しない
#   - 静止クラッタ除去（MTI）を追加。元デモはリアルタイム表示なので入っていないが、
#     録画データの解析では静止物が支配的になるため必要
#
# MATLAB→Python の対応で注意した点:
#   - MATLAB は列優先・1始まり。仮想アレイの並べ替えは 0 始まりに変換済み
#   - MATLAB の fft(X, N, 1) は列方向 → numpy では axis=-2 相当。軸を明示して移植した
#   - `np.hanning` は MATLAB の `hann` 相当（両端が 0）で `hanning` とは別物。
#     N=7 の素子軸では 7 素子中 2 素子が消えるため `hanning_matlab()` を使う

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atlas_dat_parse import load_frames, parse_header

# --- ベンダーコードの Cfg そのまま【文書】: AN24_02/05/06/07 で共通 ---
C0 = 3e8              # ベンダーは 2.998e8 ではなく 3e8 を使う
F_STRT = 24.06e9
F_STOP = 24.24e9
T_RAMP_UP = 260e-6    # チャープの掃引時間。帯域 180MHz はこの時間で掃く
T_P = 325e-6          # チャープ繰り返し間隔（.dat ヘッダ 0x3C と一致）
N_SAMPLE = 256
FS = 1e6
FU_SCA = 1.0          # 本来は Brd.FuSca（未入手）

KF = (F_STOP - F_STRT) / T_RAMP_UP   # チャープ率 = 0.6923 MHz/µs

# 実効帯域は 180MHz ではない。260µs のうち 256µs しかサンプルしないため
# （元コードのコメント "Effective bandwidth is reduced as only 256 us are sampled"）
BW_EFF = KF * N_SAMPLE / FS
TRUE_RANGE_RES = C0 / (2 * BW_EFF)   # ≈ 0.846 m。これが真の距離分解能

# 仮想アレイの並べ替え【文書】: AN24_07 の VirtData 構成（1始まり→0始まりに変換）
# 物理RXの空間順が 1,0,2,3 であることを意味する。7素子・素子間隔 λ/2
# (tx, rx) の tx: 0=先行チャープ側, 1=後続チャープ側
VIRT_ARRAY = [
    [(0, 1)],            # VirtData(:,1) = DataTx1(:,2)
    [(0, 0)],            # VirtData(:,2) = DataTx1(:,1)
    [(0, 2)],            # VirtData(:,3) = DataTx1(:,3)
    [(1, 1), (0, 3)],    # VirtData(:,4) = 0.5*(DataTx2(:,2) + DataTx1(:,4)) 重複素子の平均
    [(1, 0)],            # VirtData(:,5) = DataTx2(:,1)
    [(1, 2)],            # VirtData(:,6) = DataTx2(:,3)
    [(1, 3)],            # VirtData(:,7) = DataTx2(:,4)
]

# SIMO（1TX）の並べ替え【文書】: AN24_05 の VirtData 構成。MIMO 側と同じ置換
VIRT_ARRAY_SIMO = [1, 0, 2, 3]


def hanning_matlab(n: int) -> np.ndarray:
    """MATLAB の hanning(N)（両端非ゼロ）。np.hanning は MATLAB の hann(N) 相当で
    両端が厳密に 0 になり、7 素子の仮想アレイでは実効開口が 5 素子まで落ちる"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))


def range_axis(n_fft: int) -> np.ndarray:
    """AN24_02/05/06/07 共通: vRange = [0:NFFT-1]/NFFT * fs * c0/(2*kf)
    全長は fs に対応する 216.67 m。実数サンプルなので前半（fs/2 = 108.3 m）までが有効"""
    return np.arange(n_fft) / n_fft * FS * C0 / (2 * KF)


def velocity_axis(n_fft_vel: int, pri: float) -> np.ndarray:
    """AN24_06: vFreqVel = [-N/2:N/2-1]/N * (1/Tp), vVel = vFreqVel*c0/(2*fc)

    注意: MATLAB は中心周波数 fc=(fStrt+fStop)/2 を使うが、DemoKitApp のヘッダ値
    (max_vel=4.79570) は fStrt=24.06GHz を使ったときの値と一致する。
    ヘッダとの整合を優先し fStrt を採用した（差は 0.4%）。"""
    v_freq = np.arange(-n_fft_vel // 2, n_fft_vel // 2) / n_fft_vel * (1 / pri)
    return v_freq * C0 / (2 * F_STRT)


def range_profile(x: np.ndarray, n_fft: int = 2 ** 12, mti: bool = True) -> np.ndarray:
    """AN24_02/05/06 のレンジプロファイル。
    RP = fft(MeasChn .* Win2D, NFFT, 1) .* FuSca / ScaWin

    x: (..., Chirp, Sample) -> (..., Chirp, n_fft//2) 実数サンプルなので片側のみ返す"""
    if mti:
        # チャープ間平均を引いて静止クラッタを抑制（元デモには無い追加処理）
        x = x - x.mean(axis=-2, keepdims=True)
    win = hanning_matlab(x.shape[-1])
    sca_win = win.sum()
    rp = np.fft.fft(x * win, n=n_fft, axis=-1) * FU_SCA / sca_win
    return rp[..., : n_fft // 2]


def range_doppler(frame: np.ndarray, tx: int = 0, n_fft: int = 2 ** 12,
                  n_fft_vel: int = 2 ** 8) -> np.ndarray:
    """AN24_06 の RD マップ。TDM なので同一 TX のチャープだけを使う
    frame: (Chirp, Rx, Sample) -> (Doppler, Range)"""
    z = frame[tx::2].transpose(1, 0, 2)          # (Rx, Chirp, Sample)
    rp = range_profile(z, n_fft)                 # (Rx, Chirp, Range)
    n_chirp = rp.shape[-2]
    win_vel = hanning_matlab(n_chirp)
    rd = np.fft.fft(rp * win_vel[:, None], n=n_fft_vel, axis=-2) / win_vel.sum()
    rd = np.fft.fftshift(rd, axes=-2)            # (Rx, Doppler, Range)
    return np.abs(rd).sum(axis=0)                # Rx 非コヒーレント加算


def virtual_array(frame: np.ndarray) -> np.ndarray:
    """AN24_07: 2TX×4RX の生データから7素子の仮想等間隔リニアアレイを構成する。
    frame: (Chirp, Rx, Sample) -> (Virt, Chirp/2, Sample)

    TDM なので偶数チャープと奇数チャープが別 TX。ペアにして 1 スナップショットとする。
    注意: どちらの TX が AN24_07 の Tx1 かは不明。逆だとアレイが反転し角度の符号が反転する"""
    tx = [frame[0::2], frame[1::2]]              # 各 (Chirp/2, Rx, Sample)
    n_pair = min(tx[0].shape[0], tx[1].shape[0])
    out = np.empty((len(VIRT_ARRAY), n_pair, frame.shape[-1]), dtype=frame.dtype)
    for v, srcs in enumerate(VIRT_ARRAY):
        # 重複素子は平均（元コードの 0.5*(...+...)）
        out[v] = np.mean([tx[t][:n_pair, r] for t, r in srcs], axis=0)
    return out


def rd_per_channel(frame: np.ndarray, n_fft: int = 2 ** 12,
                   n_fft_vel: int = 2 ** 8) -> np.ndarray:
    """仮想アレイの各素子について RD を作る。frame: (Chirp, Rx, Sample) -> (Virt, Doppler, Range)

    元デモ（AN24_07）は 2 チャープしか取らず Doppler 処理をしないが、録画データでは静止物が
    支配的なので Doppler で分離しないと動体の角度が読めない。"""
    virt = virtual_array(frame)                  # (Virt, Chirp, Sample)
    rp = range_profile(virt, n_fft)              # (Virt, Chirp, Range)
    win_vel = hanning_matlab(rp.shape[1])
    rd = np.fft.fft(rp * win_vel[:, None], n=n_fft_vel, axis=1) / win_vel.sum()
    return np.fft.fftshift(rd, axes=1)


def dbf(rd_virt: np.ndarray, n_fft_ant: int = 256) -> np.ndarray:
    """AN24_05/07 の DBF。JOpt = fftshift(fft(RPExt .* WinAnt2D, NFFTAnt, 2)/ScaWinAnt, 2)
    素子方向の FFT。rd_virt: (Virt, Range) -> (Range, Angle)

    注意: 素子方向にコヒーレント加算する前に、必ず単一の Doppler ビンを選んでおくこと。
    動体は MTI 後にチャープ間で位相が回るため、チャープ方向に平均すると打ち消し合う"""
    win_ant = hanning_matlab(rd_virt.shape[0])
    j = np.fft.fft(rd_virt * win_ant[:, None], n=n_fft_ant, axis=0) / win_ant.sum()
    return np.fft.fftshift(j, axes=0).T


def angle_axis(n_fft_ant: int = 256) -> np.ndarray:
    """AN24_05/07: vAngDeg = asin(2*[-N/2:N/2-1]/N)/pi*180。素子間隔 λ/2 前提"""
    u = 2 * np.arange(-n_fft_ant // 2, n_fft_ant // 2) / n_fft_ant
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--rmax", type=float, default=10.0, help="表示する最大レンジ[m]（元コードは0.5-10m）")
    args = ap.parse_args()

    hdr = parse_header(args.dat.read_bytes()[:256])
    x = load_frames(args.dat, hdr)  # (Frame, Chirp, Rx, Sample)
    print(f"{args.dat.name}: {x.shape} (Frame, Chirp, Rx, Sample)")
    print(f"kf = {KF/1e12:.6f} MHz/us  実効帯域 {BW_EFF/1e6:.2f} MHz  真の分解能 {TRUE_RANGE_RES:.4f} m")

    # 元コードは 4096 ビン全体の軸を作って RMin/RMax で切り出すが、実数サンプルなので
    # 有効なのは前半（fs/2 = 108.3 m）まで。range_profile が返すのもその前半だけ
    v_rng = range_axis(2 ** 12)[: 2 ** 11]
    keep = (v_rng >= 0.5) & (v_rng <= args.rmax)   # 元コードの RMin=0.5, RMax=10
    v_ang = angle_axis()

    # フレームごとに「最も強い動体セル」を選び、その Doppler ビンで DBF する
    n_fft_vel = 2 ** 8
    v_vel = velocity_axis(n_fft_vel, 2 * T_P)   # TDM なので同一TXの PRI は 2*Tp
    ra, picks = [], []
    for f in range(x.shape[0]):
        rd = rd_per_channel(x[f], n_fft_vel=n_fft_vel)      # (Virt, Doppler, Range)
        p = np.abs(rd).sum(axis=0)[:, keep]                 # (Doppler, Range)
        # 静止クラッタ残差を除外。ノッチ幅は固定値ではなく速度分解能を基準にする。
        # ゼロ埋め後の刻み(0.037 m/s)で ±0.15 だけ潰してもクラッタ主ローブ(±0.6 m/s 相当)は
        # 消えず、argmax が静止物に張り付く。GUI 設定を変えても追従するようヘッダ値を使う
        p[np.abs(v_vel) < hdr["vel_res_ms"]] = 0
        d_i, r_i = np.unravel_index(p.argmax(), p.shape)
        j = np.abs(dbf(rd[:, d_i, :][:, keep]))             # (Range, Angle)
        ra.append(j)
        picks.append((v_rng[keep][r_i], v_vel[d_i], v_ang[j[r_i].argmax()]))
    ra = np.stack(ra)

    f_best = int(np.array([p[0] for p in picks]).argmin())  # 最も近づいたフレーム
    r_pk, v_pk, a_pk = picks[f_best]
    print(f"最接近フレーム: {f_best} (t={f_best*hdr['frame_period_ms']/1000:.1f}s)")
    print(f"  R = {r_pk:.2f} m, v = {v_pk:+.2f} m/s, Angle = {a_pk:+.1f} deg")
    m = 20 * np.log10(ra[f_best] + 1e-12)
    m -= m.max()

    plt.figure(figsize=(7, 5))
    plt.pcolormesh(v_ang, v_rng[keep], np.clip(m, -18, 0), shading="auto", cmap="jet")
    plt.colorbar(label="normalized [dB] (clipped at -18, as in AN24_07)")
    plt.xlabel("Angle [deg]")
    plt.ylabel("Range [m]")
    plt.xlim(-60, 60)
    plt.title(f"{args.dat.name}  DBF (AN24_07 port)  frame {f_best}")
    plt.tight_layout()
    p = args.outdir / "range_angle_dbf.png"
    plt.savefig(p, dpi=150)
    print(f"保存: {p}")

    # 角度ピークの時間変化。ボアサイト方向の歩行なら 0° 付近に留まるはず
    print("\nフレームごとの動体セルと角度:")
    for f in range(0, x.shape[0], 2):
        r_i, v_i, a_i = picks[f]
        print(f"  t={f*hdr['frame_period_ms']/1000:4.1f}s  R={r_i:5.2f} m  "
              f"v={v_i:+5.2f} m/s  Ang={a_i:+6.1f} deg")
    walk = [p[2] for p in picks[5:44]]   # 歩行区間のみ
    print(f"\n歩行区間の角度: 中央値 {np.median(walk):+.1f} deg, "
          f"標準偏差 {np.std(walk):.1f} deg（Azimuth分解能 14.5° と比較）")


if __name__ == "__main__":
    main()
