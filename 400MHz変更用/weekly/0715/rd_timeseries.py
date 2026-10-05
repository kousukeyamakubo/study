# 録画(.dat)の全フレームをRDマップ化し、時刻付きの一覧図とアニメーションGIFを生成する
#
# 使い方:
#   python rd_timeseries.py xWR14xx_log_20260715_164450.dat
# 出力: rd_grid.png（1秒おき10枚の一覧）、rd_anim.gif（全フレーム）

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dat_inspect import HEADER_BYTES, load_frames, parse_header

FRAME_PERIOD_S = 0.1  # フレーム周期100ms（Short/Mid共通）


def all_rd_maps(x: np.ndarray, tx: int = 0, dc_cut: bool = True) -> np.ndarray:
    """全フレームのRDマップ (Frame, Doppler, Range) を一括計算"""
    z = x[:, :, tx, :, :]  # (Frame, Chirpset, Rx, Sample)
    if dc_cut:
        # 静止クラッタ抑制。完全なゼロ化ではなく減衰に留め、表示スケールの崩壊を防ぐ
        z = z - 0.95 * z.mean(axis=1, keepdims=True)
    r = np.fft.fft(z * np.hanning(z.shape[-1]), axis=-1)
    rd = np.fft.fftshift(np.fft.fft(r, axis=1), axes=1)  # (Frame, Doppler, Rx, Range)
    return np.abs(rd).sum(axis=2)  # (Frame, Doppler, Range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--no-dc-cut", action="store_true")
    ap.add_argument("--grid-sec", type=float, default=1.0, help="一覧図のスナップショット間隔[s]")
    # 既定値は 7/15 録画の分布から決めた絶対値。録画をまたいだ比較を成立させるため、
    # 以降の計測でも同じ値を使う（変える場合は全図を作り直すこと）
    ap.add_argument("--vmin", type=float, default=48.0, help="表示下限 [dB]")
    ap.add_argument("--vmax", type=float, default=85.0, help="表示上限 [dB]")
    args = ap.parse_args()

    hdr = parse_header(args.dat.read_bytes()[:HEADER_BYTES])
    x = load_frames(args.dat, hdr)
    maps = all_rd_maps(x, tx=args.tx, dc_cut=not args.no_dc_cut)  # (Frame, Doppler, Range)
    maps_db = 20 * np.log10(maps + 1e-6)

    n_frame, n_dop, n_rng = maps_db.shape
    dr, dv = hdr["range_res_m"], hdr["vel_res_ms"]
    extent = [0, n_rng * dr, -n_dop / 2 * dv, n_dop / 2 * dv]
    # 全フレーム・全録画共通の固定カラースケール（コマ間・録画間の比較を成立させるため必須）
    vmin, vmax = args.vmin, args.vmax
    p5, p998 = np.percentile(maps_db, [5, 99.8])
    print(f"この録画の分布: 5%={p5:.1f}dB 99.8%={p998:.1f}dB / 表示レンジ [{vmin}, {vmax}] dB")

    # --- 一覧図: grid-sec おきのスナップショット ---
    step = max(1, int(round(args.grid_sec / FRAME_PERIOD_S)))
    idxs = list(range(0, n_frame, step))
    ncol = 5
    nrow = int(np.ceil(len(idxs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow),
                             sharex=True, sharey=True)
    for ax, fi in zip(np.ravel(axes), idxs):
        ax.imshow(maps_db[fi], aspect="auto", origin="lower", extent=extent,
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {fi * FRAME_PERIOD_S:.1f} s", fontsize=10)
    for ax in np.ravel(axes)[len(idxs):]:
        ax.axis("off")
    fig.supxlabel("Range [m]")
    fig.supylabel("Velocity [m/s]")
    fig.tight_layout()
    # 共通スケールであることを図自体に示す（誤解防止）
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cax, label="Power [dB]")
    out_grid = args.dat.parent / f"rd_grid_{args.dat.stem}.png"
    fig.savefig(out_grid, dpi=130)
    print(f"保存: {out_grid}")

    # --- アニメーションGIF: 全フレーム ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    im = ax2.imshow(maps_db[0], aspect="auto", origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax)
    fig2.colorbar(im, ax=ax2, label="Power [dB]")
    ax2.set_xlabel("Range [m]")
    ax2.set_ylabel("Velocity [m/s]")
    title = ax2.set_title("t = 0.0 s")
    fig2.tight_layout()

    # PillowWriter はフレームごとに適応パレットを作り直すため、同じdB値の色が
    # フレーム間で微妙に変動する。全フレームを先頭フレームの共通パレットに
    # 量子化して色を完全に固定する（カラーバーが全色域を含むため先頭で十分）
    # パレット作成(median-cut)と割り当て(最近傍)はアルゴリズムが異なるため、
    # 先頭フレームも含め全フレームを同一の palette= 経路で量子化して色を統一する
    frames_pil = []
    palette_base = None
    for fi in range(n_frame):
        im.set_data(maps_db[fi])
        title.set_text(f"t = {fi * FRAME_PERIOD_S:.1f} s")
        fig2.canvas.draw()
        rgb = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
        img = Image.fromarray(rgb)
        if palette_base is None:
            palette_base = img.quantize(colors=256, dither=Image.Dither.NONE)
        frames_pil.append(img.quantize(palette=palette_base, dither=Image.Dither.NONE))

    out_gif = args.dat.parent / f"rd_anim_{args.dat.stem}.gif"
    # duration=100ms/コマ = 実時間と同じ10fps再生
    frames_pil[0].save(out_gif, save_all=True, append_images=frames_pil[1:],
                       duration=100, loop=0)
    print(f"保存: {out_gif}")


if __name__ == "__main__":
    main()
