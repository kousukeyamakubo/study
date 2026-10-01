# 収録した .dat 1本を、その場で最後まで通して確認する。
#
# 夜の現地で「12本撮ってから尾根が追えていないと分かる」のを避けるための道具。
# .dat -> npz -> 尾根追跡 -> 当てはめ を一息でやり、次の判断に要る数字だけ出す。
#
# 見るところは3つ:
#   1. 尾根が走行の全区間にわたって続いているか（途切れていたら遮蔽か追跡の失敗）
#   2. レンジ変化率とドップラーが一致しているか（別物を拾っていないかの独立チェック）
#   3. 当てはめの rms が小さいか（大きければ等速でないか、尾根が飛んでいる）
#
# 使い方:
#   python check_run.py data/raw/run01.dat
#   python check_run.py data/raw/run01.dat --thr 22 --rmin 10

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0727"))
from atlas_ridge_track import build_ridges, moving_power_db   # noqa: E402

from fit_trajectory import fit_const_speed                     # noqa: E402

EXPORT = Path(__file__).resolve().parent.parent / "0727" / "atlas_export.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("data/export"))
    ap.add_argument("--rmin", type=float, default=13.0, help="近傍ゲート[m]。窓枠・三脚を外す")
    ap.add_argument("--rmax", type=float, default=100.0)
    ap.add_argument("--thr", type=float, default=25.0, help="検出しきい値[dB]")
    ap.add_argument("--top", type=int, default=3, help="表示する尾根の本数")
    args = ap.parse_args()

    out = args.outdir / args.dat.stem
    npz = out / f"{args.dat.stem}_rd.npz"
    if not npz.exists():
        subprocess.run([sys.executable, str(EXPORT), str(args.dat), "--outdir", str(out)],
                       check=True)

    pw, vel, rng, t, v_ax, _, _ = moving_power_db(npz)   # (F,R), (F,R), (R,), (F,), (D,)
    v_res = abs(v_ax[1] - v_ax[0])
    keep = (rng >= args.rmin) & (rng <= args.rmax)
    # 1フレームで動きうる距離。これを超える結びつきは非物理なので繋がない
    gate_bin = int(np.ceil(abs(v_ax).max() * (t[1] - t[0]) / (rng[1] - rng[0])))

    ridges = build_ridges(pw[:, keep], vel[:, keep], rng[keep], t,
                          args.thr, gate_bin, max_miss=3, min_run=5,
                          dop_gate=2.0, v_res=v_res)
    if not ridges:
        print(f"尾根なし（thr={args.thr} dB, rmin={args.rmin} m）。"
              f"しきい値を下げるか近傍ゲートを見直す")
        return

    print(f"{args.dat.name}  収録 {t[-1]:.1f} s / {len(t)} フレーム")
    print(f"{'#':>2} {'frames':>7} {'時間':>12} {'R範囲[m]':>14} "
          f"{'dR/dt':>7} {'doppler':>8} {'強度':>7}")
    for i, g in enumerate(ridges[:args.top]):
        print(f"{i:2d} {len(g['f']):7d} {g['t'][0]:5.1f}-{g['t'][-1]:5.1f}s "
              f"{g['rng'].min():6.2f}-{g['rng'].max():6.2f} "
              f"{g['slope']:+7.2f} {g['vel']:+8.2f} {np.median(g['lev']):7.1f}")

    g = ridges[0]                                               # 最も強い尾根を走行とみなす
    v, t_min, d, rms = fit_const_speed(g["rng"], g["t"])
    print(f"\n最強の尾根に当てはめ:")
    print(f"  v = {v:.2f} m/s   最接近 t = {t_min:.1f} s   d = {d:.3f} m   rms = {rms:.3f} m")

    # レンジ変化率とドップラーが合わないなら、別の動体を拾っている疑い
    if abs(g["slope"] - g["vel"]) > 1.0:
        print(f"  [注意] レンジ変化率 {g['slope']:+.2f} とドップラー {g['vel']:+.2f} が一致しません")
    if len(g["f"]) < 0.5 * len(t):
        print(f"  [注意] 尾根が収録の半分未満です。遮蔽か追跡の失敗を疑ってください")
    if rms > 0.5:
        print(f"  [注意] rms が大きいです。等速でないか、尾根が途中で飛んでいます")


if __name__ == "__main__":
    main()
