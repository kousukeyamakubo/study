# atlas_export.py が出した npz → フレームごとの RD マップを並べた GIF
#
# npz の rd は (Frame, TX, RX, Doppler, Range) と 1 フレーム 1 枚の RD マップを 50 枚
# 持っているが、静止画では 1 フレームしか見えない。目標の接近／離反がドップラー軸の
# 符号として動く様子を目で追うための確認用ツール。
#
# 使い方:
#   python atlas_rd_gif.py export_out/atlas_log_20260727_195113_rd.npz

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # GUI の無い環境でも回るように保存専用バックエンド
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def rd_power_db(rd: np.ndarray) -> np.ndarray:
    """(Frame, TX, RX, Doppler, Range) complex → (Frame, Doppler, Range) dB

    TX/RX は非コヒーレント加算する。可視化では角度情報を使わないので位相は不要で、
    8ch 足したほうが SNR が稼げる（角度が要るときは to_ra を使うこと）"""
    a = np.abs(rd).sum(axis=(1, 2))
    return 20 * np.log10(a + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="出力 GIF。既定は npz と同じ場所に <stem>_rd.gif")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="既定 5 fps。frame_period 200ms の実時間再生に相当")
    ap.add_argument("--rmax", type=float, default=30.0, help="表示するレンジ上限[m]")
    ap.add_argument("--rmin-peak", type=float, default=0.0,
                    help="ピーク探索の下限[m]。表示範囲は変えない。5F 設置では窓枠と室内の"
                         "操作者が近距離に出るので 13 を指定して地上目標だけを追う")
    # 目標のピークはフレーム間で 46〜78 dB と 31 dB も振れる（r^4 減衰）。40 dB 幅だと
    # 遠方フレームがほぼ真っ黒になるので、雑音床(約15 dB)が見える程度まで広く取る
    ap.add_argument("--dyn-range", type=float, default=60.0, help="カラースケールの幅[dB]")
    # 録画をまたいで比べるには絶対値で固定する必要がある。既定の「最大値から dyn-range 下まで」
    # ではファイルごとに基準がずれ、弱い目標の録画が相対的に明るく見えてしまう
    ap.add_argument("--vmin", type=float, default=None,
                    help="カラースケール下限[dB]（絶対値）。--vmax と併せて指定すると録画間で統一できる")
    ap.add_argument("--vmax", type=float, default=None, help="カラースケール上限[dB]（絶対値）")
    args = ap.parse_args()

    d = np.load(args.npz)
    rd = d["rd"]
    range_m, vel_ms, t_rel = d["range_m"], d["vel_ms"], d["t_rel_s"]
    vel_res = json.loads(str(d["header"]))["vel_res_ms"]
    sync = d["sync_frames"]

    db = rd_power_db(rd)                        # (Frame, Doppler, Range)

    # 全フレーム共通のカラースケール。フレームごとに正規化すると目標が居ないフレームで
    # ノイズが持ち上がり、強度が変化しているように誤読するため
    if args.vmax is not None or args.vmin is not None:
        vmax = args.vmax if args.vmax is not None else db.max()
        vmin = args.vmin if args.vmin is not None else vmax - args.dyn_range
        scale_note = "fixed"          # 録画間で統一（絶対 dB）
    else:
        vmax = db.max()
        vmin = vmax - args.dyn_range
        scale_note = "per-file"       # このファイルの最大を基準にする

    keep_r = range_m <= args.rmax
    db = db[..., keep_r]
    r_view = range_m[keep_r]

    # imshow の extent はセル中心ではなく端を指すので半ビンずつ広げる
    dr = r_view[1] - r_view[0]
    dv = vel_ms[1] - vel_ms[0]
    extent = [r_view[0] - dr / 2, r_view[-1] + dr / 2,
              vel_ms[0] - dv / 2, vel_ms[-1] + dv / 2]

    # 各フレームの動体ピーク。静止クラッタ（|v| < 速度分解能）を除いてから探す
    keep_d = np.abs(vel_ms) >= vel_res
    masked = np.where(keep_d[:, None], db, -np.inf)
    masked[:, :, r_view < args.rmin_peak] = -np.inf   # 近距離ゲート（表示は残す）
    peak = [np.unravel_index(f.argmax(), f.shape) for f in masked]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # ドップラーは 16 ビンしかないので補間せずビンの粗さをそのまま見せる
    im = ax.imshow(db[0], aspect="auto", origin="lower", extent=extent,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    # ピーク位置の白丸は出さない。単純な最大値探索なので MTI なしでは近距離クラッタの
    # 窓漏れ（|v|=1ビン, 約70dB）に全フレーム張り付き、目標の軌跡だと誤読させる
    fig.colorbar(im, ax=ax, label=f"Power [dB] ({scale_note}: {vmin:.0f} to {vmax:.0f})")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Radial velocity [m/s]")
    # 図中は日本語フォント未設定のため英語（atlas_export.py と同じ方針）
    # tight_layout は現在の文字列で余白を決めるので、空文字のままだとタイトルが切れる
    title = ax.set_title(" ", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    def update(f):
        im.set_data(db[f])
        d_i, r_i = peak[f]
        s = f" [sync]" if f in sync else ""
        title.set_text(f"{args.npz.stem}  frame {f:3d}  t={t_rel[f]:5.2f}s"
                       f"   peak {r_view[r_i]:5.2f}m {vel_ms[d_i]:+.2f}m/s{s}")
        return im, title

    out = args.out or args.npz.with_name(args.npz.stem + ".gif")
    ani = FuncAnimation(fig, update, frames=len(db), blit=False)
    ani.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"保存: {out}  ({out.stat().st_size/1e6:.1f} MB)  "
          f"{len(db)} フレーム @ {args.fps} fps")
    print(f"  カラースケール {vmin:.1f}〜{vmax:.1f} dB"
          f"（{'録画間で統一（絶対dB）' if scale_note == 'fixed' else 'このファイルの最大基準'}）"
          f"  このファイルの最大 {db.max():.1f} dB")


if __name__ == "__main__":
    main()
