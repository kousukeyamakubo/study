# npz から移動目標の軌跡を「複数本」抽出する。フレーム間連続性で拘束する版。
#
# atlas_track.py との違いと、それが必要になった理由:
#   atlas_track.py は毎フレーム独立に argmax を取るため、応答が弱い目標では
#   別の動体（樹木の揺れ・室内の操作者）へ飛ぶ。0729 の遅い自転車走行（173342）で
#   実際に失敗し、レンジ変化率 +6.11 m/s（最大速度 4.80 を超える非物理値）を返した。
#   本スクリプトは 1 フレームの移動上限（最大速度 × フレーム周期）で候補を繋ぐ。
#
# 複数本出すのは、直接波の他に建物壁を経由した反射（より長い斜距離に同じ動きの尾根が出る）
# を探すため。0725/camera_radar_labeling_plan.md の実験目的そのもの。
#
# 使い方:
#   python atlas_ridge_track.py export_out_bike_173342/atlas_log_20260729_173342_rd.npz

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_power_db(npz):
    """(Frame, Range) の動体電力[dB]と、各セルで最強だったドップラー値を返す"""
    d = np.load(npz)
    rd, r, v, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]
    db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)     # (Frame, Doppler, Range)

    # 速度軸は 0.5995 m/s 刻み。|v|>=0.6 と書くと浮動小数で ±1 ビン目が漏れるので
    # DC からのビン番号で切る（atlas_track.py と同じ理由）
    dc = int(np.abs(v).argmin())
    k = np.abs(np.arange(len(v)) - dc)
    mov = np.where((k >= 1)[:, None], db, -np.inf)

    pw = mov.max(axis=1)                                        # (Frame, Range)
    vel = v[mov.argmax(axis=1)]                                 # (Frame, Range)
    return pw, vel, r, t, v, db, k


def local_peaks(prof: np.ndarray, thr: float) -> np.ndarray:
    """しきい値を超える極大のレンジビン。目標応答は真の分解能 0.846 m の
    主ローブ 1 個に収まるので、隣接ビンより大きいことだけを条件にする"""
    hi = prof > thr
    hi[1:-1] &= (prof[1:-1] >= prof[:-2]) & (prof[1:-1] >= prof[2:])
    hi[0] = hi[-1] = False
    return np.where(hi)[0]


def build_ridges(pw, vel, r, t, thr, gate_bin, max_miss, min_run, dop_gate, v_res):
    """しきい値超えの極大を、フレーム間の移動上限 gate_bin 以内で貪欲に繋ぐ。
    max_miss フレームまでは見失いを許して同じ尾根として繋ぐ（応答が一時的に落ちるため）

    レンジの連続性だけでは足りない。静止物の MTI 残差（33〜35m の縞）はドップラー方向に
    平坦なので、毎フレーム別のドップラービンで極大を作り、レンジは動かないまま尾根として
    居座る。そこへ本物の目標が同じレンジに到達すると吸収され、1本の軌跡が2本に割れる
    （0729 の 173441 で実際に発生）。目標のドップラーは安定しているので、
    ドップラーの連続性 dop_gate も条件に加えて切り分ける"""
    n_frame = pw.shape[0]
    open_r, done = [], []
    for f in range(n_frame):
        cand = list(local_peaks(pw[f], thr))

        # (尾根, 候補) の組を全部作り、ドップラー不一致の小さい順に確定する。
        # 「直近点が強い尾根から順に選ぶ」方式だと、静止物残差の尾根がたまたま強いだけで
        # 本物の目標を奪う。173441 では目標が 2 フレーム しきい値を下回って消え、復帰した
        # 候補を 33.9m に居座っていた残差の尾根が先に取り、軌跡が 2 本に割れた。
        # レンジは動くがドップラーは緩やかにしか変わらないので、後者を第一の手がかりにする
        props = []
        for ri, ridge in enumerate(open_r):
            miss = f - ridge["f"][-1]
            reach = gate_bin * miss                             # 見失っている間も動いている
            v_last = vel[ridge["f"][-1], ridge["b"][-1]]
            for b in cand:
                if abs(b - ridge["b"][-1]) > reach:
                    continue
                dv = abs(vel[f, b] - v_last)
                # ドップラーのゲートは miss でスケールさせない（速度は保存量に近い）
                if dv > dop_gate * v_res:
                    continue
                props.append((dv, -pw[f, b], ri, b))
        props.sort()                                            # 不一致が小さく、強い組を優先

        used, taken = set(), set()
        for _, _, ri, b in props:
            if ri in taken or b in used:
                continue
            taken.add(ri); used.add(b)
            open_r[ri]["f"].append(f); open_r[ri]["b"].append(b)

        for b in cand:                                          # 未割り当ては新しい尾根の種
            if b not in used:
                open_r.append(dict(f=[f], b=[b]))
        keep = []
        for ridge in open_r:
            if f - ridge["f"][-1] > max_miss:
                done.append(ridge)
            else:
                keep.append(ridge)
        open_r = keep
    done += open_r

    out = []
    for g in done:
        if len(g["f"]) < min_run:
            continue
        fi, bi = np.array(g["f"]), np.array(g["b"])
        rng, tt = r[bi], t[fi]
        slope = np.polyfit(tt, rng, 1)[0]                       # レンジ変化率[m/s]
        out.append(dict(f=fi, b=bi, rng=rng, t=tt, slope=slope,
                        vel=np.median(vel[fi, bi]), lev=pw[fi, bi]))
    return sorted(out, key=lambda g: -np.median(g["lev"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--rmin", type=float, default=13.0,
                    help="既定 13m = 5F(h≈15m)から地上目標が現れうる下限。室内の操作者と窓枠を除ける")
    ap.add_argument("--rmax", type=float, default=100.0,
                    help="既定は物理上限 108m 近く。建物経由の長い経路を探すため広く取る")
    ap.add_argument("--thr", type=float, default=25.0, help="検出しきい値[dB]（雑音床 ≈13dB）")
    ap.add_argument("--min-run", type=int, default=5, help="尾根として認める最小フレーム数")
    ap.add_argument("--max-miss", type=int, default=3, help="許容する連続見失いフレーム数")
    ap.add_argument("--dop-gate", type=float, default=2.0,
                    help="フレーム間で許容するドップラー変化[ビン]。既定2。"
                         "静止物残差（ドップラーが平坦）を目標と切り分けるために使う")
    ap.add_argument("--top", type=int, default=4, help="表示する尾根の本数")
    # 既定は雑音床基準（そのファイル内で弱い応答まで見たいとき）。録画をまたいで強度を
    # 比べるには絶対値で固定する必要がある。既定の上限は雑音床+22dB なので、
    # 40dB 級の目標は飽和して録画間の強度差が消える
    ap.add_argument("--vmin", type=float, default=None, help="カラースケール下限[dB]（絶対値）")
    ap.add_argument("--vmax", type=float, default=None, help="カラースケール上限[dB]（絶対値）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pw, vel, r, t, v_ax, db, k = moving_power_db(args.npz)
    dt = t[1] - t[0]
    res = r[1] - r[0]
    v_res = abs(v_ax[1] - v_ax[0])
    v_max = abs(v_ax).max()

    band = (r >= args.rmin) & (r <= args.rmax)
    rb, pwb, velb = r[band], pw[:, band], vel[:, band]

    # 1 フレームで動きうる距離。これを超える結びつきは非物理なので繋がない
    gate = int(np.ceil(v_max * dt / res))
    nf = np.median(db[:, k >= 1, :][:, :, r > 60])

    print(f"{args.npz.name}")
    print(f"  雑音床(>60m) {nf:.1f} dB  しきい値 {args.thr:.0f} dB  "
          f"ゲート {gate} bin (={gate*res/dt:.1f} m/s)")

    ridges = build_ridges(pwb, velb, rb, t, args.thr, gate, args.max_miss, args.min_run,
                          args.dop_gate, v_res)
    if not ridges:
        print("  尾根なし")
        return

    for i, g in enumerate(ridges[: args.top]):
        dv = abs(g["slope"] - g["vel"])
        ok = dv < v_res                                          # ビン幅以内なら一致とみなす
        print(f"  尾根{i+1}: t={g['t'][0]:.1f}〜{g['t'][-1]:.1f}s（{len(g['f'])}フレーム）  "
              f"{g['rng'][0]:.1f} → {g['rng'][-1]:.1f} m")
        print(f"    レンジ変化率 {g['slope']:+.2f} m/s  vs  Doppler中央値 {g['vel']:+.2f} m/s  "
              f"差 {dv:.2f}（分解能の {dv/v_res:.2f} 倍）")
        print(f"    レベル {g['lev'].min():.1f}〜{g['lev'].max():.1f} dB"
              f"（中央値 {np.median(g['lev']):.1f}）")
        print(f"    → {'実在の移動目標と判定' if ok else '一致せず。副ローブ／別の動体の可能性'}")

    vmin = args.vmin if args.vmin is not None else nf - 1
    vmax = args.vmax if args.vmax is not None else nf + 22
    fixed = args.vmin is not None or args.vmax is not None

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pw, aspect="auto", origin="lower",
                   extent=[r[0], r[-1], t[0], t[-1]], vmin=vmin, vmax=vmax)
    fig.colorbar(im, label=f"Moving-target power [dB] "
                           f"({'fixed' if fixed else 'noise-floor ref'}: "
                           f"{vmin:.0f} to {vmax:.0f})")
    for i, g in enumerate(ridges[: args.top]):
        c = ["r", "lime", "cyan", "magenta"][i % 4]
        ax.plot(g["rng"], g["t"], ".", color=c, ms=5,
                label=f"ridge{i+1} {g['slope']:+.2f} m/s (Dop {g['vel']:+.2f})")
    ax.axvline(args.rmin, color="orange", ls=":", lw=1, label=f"gate {args.rmin:.0f} m")
    ax.set_xlim(0, min(args.rmax, r[-1]))
    # 図中は日本語フォント未設定のため英語（他のスクリプトと同じ方針）
    ax.set_xlabel("Slant range [m]"); ax.set_ylabel("Time [s]")
    ax.set_title(f"{args.npz.stem}  ridges")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = args.out or args.npz.with_name("ridge_track.png")
    fig.savefig(out, dpi=150)
    print(f"  保存: {out}")


if __name__ == "__main__":
    main()
