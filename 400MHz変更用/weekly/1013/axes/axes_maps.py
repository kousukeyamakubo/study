# 10/1 夜の 36 走行を、距離−時間以外の軸（ドップラー−時間・角度−時間）で見直す。
#
# 10/2 の指摘「ドップラー−時間、角度−時間のマップも作る」への対応。問いは 3 つ:
#   a. 近い端（R 22〜28 m）を通る時間帯に、ドップラー・角度で目標の痕跡があるか
#      （距離−時間図では見えなかった。痕跡があれば「目標が弱い」以外に原因を絞れる）
#   b. 角度の軌跡がレーンごとに違うか（幾何から出した期待角度と重ねる）
#   c. レーン 3 は角度でも見えないか
#
# 図は 2 系統で作る:
#   MTI あり（data/export）           … 現行の処理。|v_r| < 0.3 m/s の目標は削られる
#   MTI なし＋フレーム間差分（data/export_nomti）
#                                     … 0.2 s 前の複素 RD との差。最接近点付近（v_r → 0）でも、
#                                       0.2 s の間に位相が回れば残る（docs/realworld.md §5 の対処）。
#                                       MTI なしの生の RD は静止物（DC ビン 40〜63 dB）が全域で最大になり
#                                       目標が見えないので、差分を取ってから同じ図を描く
#
# 角度は、各フレームで最も強い動体セル（R 15〜55 m）の角度スペクトルを並べたもの。
# atlas_export.to_ra（TDM の位相補正込み、素子間隔 λ/2 前提）をそのセルの距離ビンに掛ける。
# 角度の符号の向き（どちらが +）は未検証。
#
# 使い方:
#   cd weekly/1013/axes
#   python axes_maps.py

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEKLY = HERE.parent.parent
sys.path.insert(0, str(WEEKLY / "0727"))
sys.path.insert(0, str(WEEKLY / "1001"))
from analyze_night import CONES, L_C1_C5, N_MAIN, S0, TAGS, condition, lane_delta_d, npz_path   # noqa: E402
from atlas_export import angle_axis_deg, to_ra                                                   # noqa: E402
from atlas_ridge_track import moving_power_db                                                     # noqa: E402
from plot_lane_truth import LANE_OFFSET                                                           # noqa: E402

FIG = HERE / "figures"
R_LO, R_HI = 15.0, 55.0          # 道路上の目標が来うる範囲（地上は R ≳ 13 m、走路の奥の端は R ≈ 49 m）
NEAR = (22.0, 28.0)              # 距離−時間図で見えなかった近い端の帯
N_ANT = 256
ANG = angle_axis_deg(N_ANT)      # (A,)
# 格子の列: 既存の距離−時間図（analyze_night.RANGE_TIME_PICK）と同じく、レーン1 接近・離反、レーン2 …
COLS = [(1, "app"), (1, "dep"), (2, "app"), (2, "dep"), (3, "app"), (3, "dep")]
SIGN_NOTE = "angle sign convention unverified"
EXCESS_THR = 6.0                 # 「目標がいる」とみなす、最強セルの電力の走行内中央値からの超過 [dB]


# ------------------------------------------------------------------
# 1 走行ぶんの量
# ------------------------------------------------------------------

def strongest_cell_angle(rd_frame, power, hdr):
    """power (D, R) の最大セルで、角度スペクトル [dB] を返す。

    to_ra は距離軸を独立に扱うので、最大セルの距離ビンだけを切り出して渡す。
    ドップラー軸は切らない（TDM の位相補正はドップラービン番号 k から決まるため）
    """
    i_d, i_r = np.unravel_index(np.argmax(power), power.shape)
    ra = to_ra(rd_frame[..., i_r:i_r + 1], hdr, rd_frame.shape[-2], N_ANT)   # (D, A, 1)
    return 20 * np.log10(np.abs(ra[i_d, :, 0]) + 1e-12), i_d, i_r             # (A,)


def run_maps(n):
    """No.n の (MTI あり / フレーム間差分) それぞれについて、距離−時間・ドップラー−時間・角度−時間を出す"""
    out = {}
    for kind in ("mti", "fd"):
        d = np.load(npz_path(TAGS[n - 1], nomti=(kind == "fd")))
        rd, rng, vel, t = d["rd"], d["range_m"], d["vel_ms"], d["t_rel_s"]   # rd: (F, TX, RX, D, R)
        hdr = json.loads(str(d["header"]))
        dc = int(np.abs(vel).argmin())
        k = np.abs(np.arange(len(vel)) - dc)                                  # (D,) DC からのビン番号
        if kind == "fd":
            # 0.2 s 前との複素差。線形なので to_ra の位相補正もそのまま効く
            rd = np.diff(rd, axis=0)                                          # (F-1, TX, RX, D, R)
            t = t[1:]
            dop_ok = np.ones(len(vel), bool)                                  # DC も含める（最接近点を見るため）
        else:
            # 速度ビンは値でなくビン番号で切る（docs/realworld.md §5）
            dop_ok = k >= 1
        db = 20 * np.log10(np.abs(rd).sum(axis=(1, 2)) + 1e-12)              # (F, D, R) moving_power_db と同じ定義
        rsel = (rng >= R_LO) & (rng <= R_HI)
        mov = np.where(dop_ok[:, None] & rsel[None, :], db, -np.inf)          # (F, D, R)

        ang = np.empty((len(t), N_ANT))                                       # (F, A)
        pk_r, pk_v, pk_p = (np.empty(len(t)) for _ in range(3))
        for f in range(len(t)):
            ang[f], i_d, i_r = strongest_cell_angle(rd[f], mov[f], hdr)
            pk_r[f], pk_v[f], pk_p[f] = rng[i_r], vel[i_d], mov[f, i_d, i_r]

        out[kind] = dict(
            t=t, rng=rng, vel=vel, dc=dc,
            rt=mov.max(axis=1),                                               # (F, R) 距離−時間
            dt=np.where(rsel[None, None, :], db, -np.inf).max(axis=2),        # (F, D) ドップラー−時間（全ビン表示）
            ang=ang, pk_r=pk_r, pk_v=pk_v, pk_p=pk_p,
            pk_ang=ANG[ang.argmax(axis=1)],
            # 最強セルは目標がいなくても雑音の最大として必ず選ばれるので、走行内の中央値からの超過で目標の有無を見る
            present=pk_p - np.median(pk_p) >= EXCESS_THR,
        )
    return out


# ------------------------------------------------------------------
# 期待角度（幾何）
# ------------------------------------------------------------------

GEO_A, GEO_H, _ = lane_delta_d(road_w=4.0)     # a: レーダー直下→右の縁の水平距離, h: 設置高


def lane_d(lane):
    """レーンの横方向離隔 d（最接近距離）と、レーダー直下からの横方向の水平距離 x"""
    x = GEO_A + LANE_OFFSET[lane]
    return float(np.hypot(x, GEO_H)), x


def expected_angle(r, lane, yaw_deg, sign):
    """斜距離 r にいるときの期待角度 [deg]。

    アレイの軸が水平（ロールなし）なら、素子方向 FFT が測るのはアレイ軸方向の方向余弦
    u = (到来方向の単位ベクトル)·(アレイ軸) で、俯角（縦の向き）には依らない。
    アレイ軸が走路方向から yaw だけ回っているとすると
        u = sign * (X cos(yaw) + x sin(yaw)) / R,   X = sqrt(R^2 - d^2)（走路方向、垂線の足から奥が正）
    走路は垂線の足から手前へ 2.2 m しかない（S0）ので、X ≥ 0 の枝だけを使う。
    yaw と sign は未知なので、データ（自転車レーン1）に当てはめて決める
    """
    d, x = lane_d(lane)
    r = np.asarray(r, float)
    X = np.sqrt(np.maximum(r**2 - d**2, 0.0))
    u = sign * (X * np.cos(np.radians(yaw_deg)) + x * np.sin(np.radians(yaw_deg))) / np.maximum(r, d)
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def fit_yaw(maps):
    """自転車レーン1（最も濃く見える）の、目標がいるフレームの (R, 角度) に yaw と sign を当てはめる。

    外れ値（静止物の残差の帯、他の動体）が混ざるので、誤差の中央値を最小にする
    """
    rs, angs = [], []
    for n in range(1, N_MAIN + 1):
        mode, lane, _ = condition(n)
        if mode != "bike" or lane != 1:
            continue
        m = maps[n]["mti"]
        sel = m["present"] & (m["pk_r"] >= 28) & (m["pk_r"] <= 50)
        rs.append(m["pk_r"][sel])
        angs.append(m["pk_ang"][sel])
    rs, angs = np.concatenate(rs), np.concatenate(angs)
    best = None
    for sign in (1, -1):
        for yaw in np.arange(-90, 90.1, 0.5):
            err = np.median(np.abs(angs - expected_angle(rs, 1, yaw, sign)))
            if best is None or err < best[0]:
                best = (err, yaw, sign)
    return best, rs, angs


# ------------------------------------------------------------------
# 図
# ------------------------------------------------------------------

def run_grid(mode):
    """mode の 18 本を 3 行（繰り返し）× 6 列（レーン・向き）に置く。戻り値は {(row, col): No.}"""
    pos = {}
    for n in range(1, N_MAIN + 1):
        m, lane, dr = condition(n)
        if m != mode:
            continue
        col = COLS.index((lane, dr))
        row = sum(1 for (r, c) in pos if c == col)
        pos[(row, col)] = n
    return pos


def bin_edges(c):
    """中心 c の等間隔軸のビン境界（pcolormesh 用）"""
    h = (c[1] - c[0]) / 2
    return np.concatenate([c - h, [c[-1] + h]])


def grid_figure(maps, mode, kind, what, geo, out):
    """手段 mode の 18 本を、ドップラー−時間（what="dop"）または角度−時間（what="ang"）で並べる"""
    pos = run_grid(mode)
    fig, axes = plt.subplots(3, 6, figsize=(3.0 * 6, 4.2 * 3), sharey=True, sharex=True)
    for (row, col), n in pos.items():
        ax = axes[row, col]
        m = maps[n][kind]
        lane, dr = COLS[col]
        if what == "dop":
            vmin, vmax = (10, 40) if kind == "mti" else (15, 45)
            im = ax.pcolormesh(bin_edges(m["vel"]), bin_edges(m["t"]), m["dt"], shading="flat",
                               cmap="viridis", vmin=vmin, vmax=vmax)
            if kind == "mti":
                ax.axvline(m["vel"][m["dc"]], color="w", lw=0.5, ls=":")   # DC は MTI で消えている
        else:
            vmin, vmax = (5, 30) if kind == "mti" else (10, 35)
            im = ax.pcolormesh(ANG, m["t"], m["ang"], shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
            p = m["present"]
            ax.plot(expected_angle(m["pk_r"][p], lane, *geo), m["t"][p], ".", ms=2.5, color="tab:red",
                    label="expected angle at the peak range (declared lane)")
            ax.set_xlim(-90, 90)
        ax.set_title(f"#{n} {mode} lane{lane} {'approach' if dr == 'app' else 'depart'}", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("time [s]")
    for ax in axes[-1]:
        ax.set_xlabel("radial velocity [m/s]" if what == "dop" else "angle [deg]")
    src = "MTI (chirp mean)" if kind == "mti" else "no MTI, frame difference (DC included)"
    if what == "dop":
        title = f"{mode}: Doppler-time, max over R {R_LO:.0f}-{R_HI:.0f} m  [{src}]  + = receding"
        label = "power [dB]"
    else:
        title = (f"{mode}: angle-time, angle spectrum of the strongest cell in R {R_LO:.0f}-{R_HI:.0f} m  [{src}]\n"
                 f"red dots: expected angle (yaw {geo[0]:.1f} deg, sign {geo[1]:+d}, fitted on bike lane1)"
                 f" at frames with a target.  {SIGN_NOTE}")
        label = "angle spectrum [dB]"
    fig.suptitle(title, fontsize=10)
    fig.colorbar(im, ax=axes, label=label, shrink=0.6)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def run_figure(maps, n, geo, out):
    """1 走行を 2 行（MTI / フレーム間差分）× 3 列（距離・ドップラー・角度）。時間軸は共通。
    近い端の帯 R 22〜28 m を通った時間帯に、ドップラー・角度で痕跡があるかを見るための図"""
    mode, lane, dr = condition(n)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharey=True)
    for row, kind in enumerate(("mti", "fd")):
        m = maps[n][kind]
        lo = 10 if kind == "mti" else 15
        ax = axes[row, 0]
        keep = (m["rng"] >= R_LO) & (m["rng"] <= R_HI)
        im0 = ax.pcolormesh(m["rng"][keep], m["t"], m["rt"][:, keep], shading="nearest", cmap="viridis",
                            vmin=lo, vmax=lo + 30)
        ax.axvspan(*NEAR, color="w", alpha=0.15)
        for r in CONES.values():
            ax.axvline(r, color="r", lw=0.5, ls="--")
        ax.set_xlabel("range R [m]")
        ax.set_ylabel(f"{'MTI' if kind == 'mti' else 'no MTI, frame diff'}\ntime [s]")
        ax = axes[row, 1]
        ax.pcolormesh(bin_edges(m["vel"]), bin_edges(m["t"]), m["dt"], shading="flat", cmap="viridis", vmin=lo, vmax=lo + 30)
        ax.set_xlabel("radial velocity [m/s]  (+ = receding)")
        ax = axes[row, 2]
        im2 = ax.pcolormesh(ANG, m["t"], m["ang"], shading="nearest", cmap="viridis", vmin=lo - 5, vmax=lo + 20)
        p = m["present"]
        ax.plot(expected_angle(m["pk_r"][p], lane, *geo), m["t"][p], ".", ms=3, color="tab:red")
        # 最強セルが近い端の帯にあるフレームを印す（距離−時間図で見えていない区間の痕跡）
        nr = p & (m["pk_r"] >= NEAR[0]) & (m["pk_r"] <= NEAR[1])
        ax.plot(np.full(nr.sum(), 85), m["t"][nr], ">", ms=4, color="w")
        ax.set_xlim(-90, 90)
        ax.set_xlabel(f"angle [deg]  ({SIGN_NOTE})")
        fig.colorbar(im0, ax=axes[row, :2], label="power [dB]", shrink=0.8)
        fig.colorbar(im2, ax=axes[row, 2], label="angle spectrum [dB]", shrink=0.8)
    fig.suptitle(f"No.{n} {mode} lane{lane} {'approach' if dr == 'app' else 'depart'}  -  shaded: R {NEAR[0]:.0f}-{NEAR[1]:.0f} m;"
                 f"  red: expected angle at the peak range;  white >: peak cell inside R {NEAR[0]:.0f}-{NEAR[1]:.0f} m",
                 fontsize=10)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def angle_vs_range_figure(maps, geo, fit_err, out):
    """時刻を使わずに、目標がいるフレームの (R, 角度) をレーン別に散布し、3 レーンの期待角度と重ねる。
    plot_lane_truth.py と同じく、映像がなく時刻つきの真値が無いため、R を介して比べる"""
    rr = np.linspace(20, 52, 200)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for row, mode in enumerate(("walk", "bike")):
        for lane in (1, 2, 3):
            ax = axes[row, lane - 1]
            for n in range(1, N_MAIN + 1):
                mo, ln, dr = condition(n)
                if mo != mode or ln != lane:
                    continue
                m = maps[n]["mti"]
                p = m["present"]
                ax.plot(m["pk_r"][p], m["pk_ang"][p], "o" if dr == "app" else "^", ms=3, alpha=0.5,
                        color="k" if dr == "app" else "tab:purple")
            for o in (1, 2, 3):
                ax.plot(rr, expected_angle(rr, o, *geo), color=colors[o], lw=1.8 if o == lane else 0.8,
                        label=f"expected lane{o}")
            ax.axvspan(*NEAR, color="tab:red", alpha=0.07)
            n_pts = sum(int(maps[n]["mti"]["present"].sum()) for n in range(1, N_MAIN + 1)
                        if condition(n)[:2] == (mode, lane))
            ax.set_title(f"{mode} lane{lane}  ({n_pts} frames with a target, 6 runs)", fontsize=9)
            ax.set_xlim(R_LO, R_HI)
            ax.set_ylim(-90, 90)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"peak angle [deg]\n({SIGN_NOTE})")
    for ax in axes[-1]:
        ax.set_xlabel("peak range R [m]")
    axes[0, 0].plot([], [], "o", color="k", label="approach")
    axes[0, 0].plot([], [], "^", color="tab:purple", label="depart")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"strongest moving cell per frame (MTI), frames {EXCESS_THR:.0f} dB above the run median.  "
                 f"expected: yaw {geo[0]:.1f} deg, sign {geo[1]:+d} fitted on bike lane1 (median |err| {fit_err:.1f} deg).  "
                 f"angular resolution 14.5 deg", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


RES_BAND = (30.0, 45.0)    # 直接波がどのレーンでも安定して見えている帯
RES_CLIP = 8.0             # これより外れた点は別の反射（近い端の時間帯に R 33〜37 m に出る負の角度の成分など）とみなす


def lane_residual(maps, geo):
    """各走行で、目標がいるフレームの角度から「レーン1 の期待角度」を引いた残差の中央値 [deg]。

    レーンの違いは R にはほぼ出ない（奥行き 1 m あたり 0.3〜0.6 m）が、角度には
    (横方向の水平距離の差) × sin(yaw) / R の形で 1 レーンあたり約 1.6° 出る（R 30〜45 m）。
    分解能 14.5° より小さいが、軌跡に沿って多数のフレームを平均すれば見える可能性がある。
    yaw はレーン1 自転車で決めたので、レーン1 自転車の残差 ≈ 0 は自明。徒歩レーン1 と他のレーンは独立な確認
    """
    out = {}
    for n in range(1, N_MAIN + 1):
        m = maps[n]["mti"]
        p = m["present"] & (m["pk_r"] >= RES_BAND[0]) & (m["pk_r"] <= RES_BAND[1])
        r = m["pk_ang"][p] - expected_angle(m["pk_r"][p], 1, *geo)
        r = r[np.abs(r) < RES_CLIP]
        if len(r):
            out[n] = (float(np.median(r)), len(r))
    return out


def lane_residual_figure(res, geo, out):
    """走行ごとの角度残差を収録順に並べる。レーンは収録順と交絡しているので、横軸を収録順にして
    時間的なずれ（ドリフト）では説明できないことを見せる（徒歩レーン2 の後に自転車レーン1 が戻る）"""
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    exp = {ln: float(np.median(expected_angle(np.linspace(*RES_BAND, 20), ln, *geo)
                               - expected_angle(np.linspace(*RES_BAND, 20), 1, *geo))) for ln in (1, 2, 3)}
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for ln in (1, 2, 3):
        ax.axhline(exp[ln], color=colors[ln], lw=1, ls="--", label=f"expected lane{ln} ({exp[ln]:+.1f} deg)")
    for n, (v, cnt) in res.items():
        mode, ln, dr = condition(n)
        ax.scatter(n, v, s=12 + 3 * cnt, color=colors[ln], marker="o" if mode == "walk" else "s",
                   edgecolor="k", lw=0.5)
    ax.axvline(18.5, color="gray", lw=0.8)
    ax.text(9, ax.get_ylim()[1], "walk", ha="center", va="top")
    ax.text(27, ax.get_ylim()[1], "bike", ha="center", va="top")
    ax.set_xlabel("run No. (recording order)")
    ax.set_ylabel("median angle residual vs lane1 expectation [deg]")
    ax.set_title(f"per-run angle offset, R {RES_BAND[0]:.0f}-{RES_BAND[1]:.0f} m, frames with a target (marker size = n frames)\n"
                 f"yaw fitted on bike lane1 (No.19-24).  {SIGN_NOTE}.  lane3 has 1 frame per run at most", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"-> {out.relative_to(HERE)}")


def main():
    FIG.mkdir(exist_ok=True)
    (FIG / "runs").mkdir(exist_ok=True)
    maps = {n: run_maps(n) for n in range(1, N_MAIN + 1)}

    (err, yaw, sign), *_ = fit_yaw(maps)
    geo = (yaw, sign)
    print(f"期待角度の当てはめ（自転車レーン1）: yaw {yaw:.1f} deg, sign {sign:+d}, 誤差の中央値 {err:.1f} deg")
    rr = np.array([22.0, 28.0, 35.0, 42.0, 49.0])
    for lane in (1, 2, 3):
        print(f"  レーン{lane} の期待角度 R={rr.tolist()}: {np.round(expected_angle(rr, lane, *geo), 1).tolist()}")

    for mode in ("walk", "bike"):
        for kind in ("mti", "fd"):
            sfx = "" if kind == "mti" else "_fd"
            grid_figure(maps, mode, kind, "dop", geo, FIG / f"doppler_time_{mode}{sfx}.png")
            grid_figure(maps, mode, kind, "ang", geo, FIG / f"angle_time_{mode}{sfx}.png")
    angle_vs_range_figure(maps, geo, err, FIG / "angle_vs_range.png")
    res = lane_residual(maps, geo)
    lane_residual_figure(res, geo, FIG / "angle_residual_by_run.png")
    print("\n走行ごとの角度残差（レーン1 の期待角度から、R 30〜45 m の中央値 [deg]、フレーム数）")
    for n, (v, cnt) in res.items():
        print(f"  No.{n:2d} {'/'.join(map(str, condition(n)))}: {v:+.2f} ({cnt})")
    for n in range(1, N_MAIN + 1):
        mode, lane, dr = condition(n)
        run_figure(maps, n, geo, FIG / "runs" / f"No{n:02d}_{mode}_lane{lane}_{dr}.png")
    print(f"-> figures/runs/ に {N_MAIN} 枚")

    # 近い端の帯に最強セルが来たフレーム数（目標がいるフレームのみ）。a の手がかり
    print("\n近い端 R 22〜28 m に最強セルが来た「目標あり」フレーム数（MTI / 差分）")
    for n in range(1, N_MAIN + 1):
        mode, lane, dr = condition(n)
        c = []
        for kind in ("mti", "fd"):
            m = maps[n][kind]
            c.append(int((m["present"] & (m["pk_r"] >= NEAR[0]) & (m["pk_r"] <= NEAR[1])).sum()))
        print(f"  No.{n:2d} {mode} lane{lane} {dr}: {c[0]:3d} / {c[1]:3d}")


if __name__ == "__main__":
    main()
