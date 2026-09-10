# homography.py の検証。合成カメラで真値が分かる状況を作り、精度と誤差伝搬を測る。
#
# 【なぜ合成データで先にやるか】
# 実映像はまだ無いが、精度が足りるかどうかは幾何と雑音だけで決まるので先に評価できる。
# ここで「マーカーを何点どこに置けばよいか」を出しておけば、現地作業が一発で済む。
#
# 【判定基準】
# ラベルは最終的にレーダーの斜距離ビン（真の分解能 0.846 m）に量子化される。
# 地上座標の誤差がこれに対して十分小さければラベルとして使える。
# 目安として **0.85 m の 1/3 = 0.28 m** を目標精度に置く。
#
# 使い方:
#   python check_homography.py

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from homography import (apply_h, estimate_homography, pose_from_homography,  # noqa: E402
                        reprojection_error)

OUT_DIR = Path(__file__).resolve().parent
RANGE_BIN = 0.8463541666666666      # ATLAS の真の距離分解能[m]
TARGET_ACC = RANGE_BIN / 3          # 目標精度[m]
RNG_SEED = 42

# 想定する設置（0727/0729 の条件に合わせる）
H_CAM = 15.0            # 設置高[m]
DEPRESSION = 20.0       # 俯角[deg]
IMG_W, IMG_H = 1920, 1080
FOV_H = 60.0            # 水平画角[deg]


def make_camera(h=H_CAM, dep=DEPRESSION, fov=FOV_H, img_w=IMG_W, img_h=IMG_H):
    """内部パラメータ K と、地上点→画像座標の射影関数を返す。

    世界座標: X=横方向, Y=建物から離れる方向, Z=上。カメラは (0,0,h) にあり
    俯角 dep で +Y を向く。地面は Z=0。fov は水平画角[deg]"""
    fx = fy = (img_w / 2) / np.tan(np.radians(fov / 2))
    K = np.array([[fx, 0, img_w / 2], [0, fy, img_h / 2], [0, 0, 1.0]])
    d = np.radians(dep)
    # カメラ座標 x=右, y=下, z=前方（CV の慣例）
    R = np.array([[1, 0, 0],
                  [0, -np.sin(d), -np.cos(d)],
                  [0, np.cos(d), -np.sin(d)]])
    C = np.array([0.0, 0.0, h])

    def project(xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        pw = np.hstack([xy, np.zeros((len(xy), 1))])
        pc = (pw - C) @ R.T                       # (N,3) カメラ座標
        uv = (pc @ K.T)
        return uv[:, :2] / uv[:, 2:3]

    return K, project


def visible(uv):
    """画角内に入っているか"""
    return (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)


# --------------------------------------------------------------------------

def eval_layout(project, markers, targets, px_noise_marker, px_noise_target,
                n_mc=400, rng=None):
    """マーカー配置を与え、目標位置での地上座標誤差[m]を返す (n_mc, n_target)"""
    rng = rng or np.random.default_rng(RNG_SEED)
    uv_m = project(markers)
    uv_t = project(targets)
    err = np.zeros((n_mc, len(targets)))
    for i in range(n_mc):
        # マーカーの画像座標を拾う誤差（人がクリックする精度）
        uv_m_n = uv_m + rng.normal(0, px_noise_marker, uv_m.shape)
        H = estimate_homography(uv_m_n, markers)      # 画像 → 地上
        # 目標の接地点を拾う誤差（YOLO の bbox 下辺のばらつき）
        uv_t_n = uv_t + rng.normal(0, px_noise_target, uv_t.shape)
        err[i] = np.linalg.norm(apply_h(H, uv_t_n) - targets, axis=1)
    return err


def grid_markers(y_lo, y_hi, x_half, n_y=3, n_x=2):
    """奥行き n_y × 横 n_x のマーカー配置"""
    ys = np.linspace(y_lo, y_hi, n_y)
    xs = np.linspace(-x_half, x_half, n_x)
    return np.array([[x, y] for y in ys for x in xs])


def main():
    rng = np.random.default_rng(RNG_SEED)
    K, project = make_camera()

    print("=" * 76)
    print(f"合成カメラ: h={H_CAM} m, 俯角 {DEPRESSION}°, {IMG_W}x{IMG_H}, 水平画角 {FOV_H}°")
    print(f"判定基準: 地上座標の誤差 < {TARGET_ACC:.2f} m "
          f"（レーダー距離分解能 {RANGE_BIN:.2f} m の 1/3）")
    print("=" * 76)

    # --- 0. 見える範囲と画素あたりの地上距離 ---
    print("\n[0] 画素あたりの地上距離（この幾何の限界を先に把握する）")
    print(f"  {'Y[m]':>6} {'v[px]':>8} {'dY/dv[m/px]':>13}  可視")
    for y in (15, 20, 30, 40, 50, 70, 100):
        uv = project([[0.0, y]])
        uv2 = project([[0.0, y + 0.01]])
        dv = abs(uv2[0, 1] - uv[0, 1])
        dydv = 0.01 / dv if dv > 0 else np.inf
        print(f"  {y:6.0f} {uv[0,1]:8.1f} {dydv:13.3f}  "
              f"{'○' if visible(uv)[0] else '×（画角外）'}")
    print("  ※ 俯角が浅いほど遠方が1画素に潰れる。ここが精度の物理的な上限を決める")

    # --- 1. 基本精度 ---
    targets = np.array([[0.0, y] for y in (20, 25, 30, 35, 40, 45, 50)])
    markers = grid_markers(20, 50, 8.0, n_y=3, n_x=2)      # 6点、目標範囲を囲む
    print(f"\n[1] 基本精度（マーカー6点: Y=20〜50 m, X=±8 m で目標範囲を囲む）")
    print(f"  {'目標Y[m]':>8}", end="")
    for pn in (1.0, 2.0, 5.0):
        print(f" {'σ=' + str(pn) + 'px':>12}", end="")
    print()
    err_by_noise = {}
    for pn in (1.0, 2.0, 5.0):
        err_by_noise[pn] = eval_layout(project, markers, targets, pn, pn, rng=rng)
    for j, t in enumerate(targets):
        print(f"  {t[1]:8.0f}", end="")
        for pn in (1.0, 2.0, 5.0):
            e = np.median(err_by_noise[pn][:, j])
            print(f" {e:9.3f} m{'*' if e < TARGET_ACC else ' '}", end="")
        print()
    print("  * = 目標精度を満たす")

    # --- 2. 外挿の危険性 ---
    print(f"\n[2] マーカーの外側は使えるか（マーカーを Y=20〜35 m に限定）")
    near = grid_markers(20, 35, 8.0, n_y=3, n_x=2)
    e_near = eval_layout(project, near, targets, 2.0, 2.0, rng=rng)
    e_wide = err_by_noise[2.0]
    print(f"  {'目標Y[m]':>8} {'20-50mに配置':>14} {'20-35mに配置':>14}  倍率")
    for j, t in enumerate(targets):
        a, b = np.median(e_wide[:, j]), np.median(e_near[:, j])
        mark = "  ← 外挿" if t[1] > 35 else ""
        print(f"  {t[1]:8.0f} {a:12.3f} m {b:12.3f} m  {b/a:5.1f}x{mark}")
    print("  ※ 外挿の代償は 1.2〜1.4 倍程度で、距離とともに緩やかに増える。"
          "「囲めないと使えない」ほどではない")
    print("     遠方にマーカーを置けない場合でも実用範囲。ただし目標範囲の中心付近は覆うこと")

    # --- 3. マーカー点数 ---
    print(f"\n[3] マーカーの点数（σ=2px、Y=40 m での誤差）")
    j40 = int(np.where(targets[:, 1] == 40)[0][0])
    for n_y, n_x, label in ((2, 2, "4点（最小）"), (3, 2, "6点"), (4, 2, "8点"), (4, 3, "12点")):
        m = grid_markers(20, 50, 8.0, n_y=n_y, n_x=n_x)
        e = eval_layout(project, m, targets, 2.0, 2.0, rng=rng)
        print(f"  {label:>10}: {np.median(e[:, j40]):.3f} m")
    print("  ※ 点数を増やすと最小二乗で誤差が平均化される。地物を選ぶだけならコストは低い")

    # --- 4. 横方向の広がり ---
    print(f"\n[4] マーカーの横の広がり（6点、σ=2px、Y=40 m での誤差）")
    for xh in (2.0, 5.0, 8.0, 15.0):
        m = grid_markers(20, 50, xh, n_y=3, n_x=2)
        e = eval_layout(project, m, targets, 2.0, 2.0, rng=rng)
        print(f"  X=±{xh:4.1f} m: {np.median(e[:, j40]):.3f} m")
    print("  ※ この幾何では横の広がりはほとんど効かない（±2 m でも実用になる）。"
          "道幅が取れなくても問題にならない")
    print("     ただし4点が一直線に並ぶと解けなくなるので、横のずれ自体は必要")

    # --- 5. マーカー座標の実測誤差 ---
    print(f"\n[5] マーカーの地上座標をメジャーで測る誤差の影響（σ=2px、Y=40 m）")
    uv_m = project(markers)
    uv_t = project(targets)
    for m_err in (0.00, 0.02, 0.05, 0.10, 0.30):
        acc = []
        for _ in range(400):
            mk = markers + rng.normal(0, m_err, markers.shape)   # 測り間違い
            H = estimate_homography(uv_m + rng.normal(0, 2.0, uv_m.shape), mk)
            p = apply_h(H, uv_t + rng.normal(0, 2.0, uv_t.shape))
            acc.append(np.linalg.norm(p - targets, axis=1)[j40])
        print(f"  メジャー誤差 {m_err*100:4.0f} cm → 地上誤差 {np.median(acc):.3f} m")
    print("  ※ 巻尺で cm オーダーに収まれば画素雑音に埋もれる。過剰な精度は不要")

    # --- 6. h と俯角の復元 ---
    print(f"\n[6] ホモグラフィ分解による h・俯角の復元（内部パラメータ既知）")
    print(f"  {'画素雑音':>10} {'h[m]':>16} {'俯角[deg]':>18}")
    for pn in (0.0, 1.0, 2.0, 5.0):
        hs, ds = [], []
        for _ in range(200):
            Hh = estimate_homography(uv_m + rng.normal(0, pn, uv_m.shape), markers)
            h_e, d_e, _, _ = pose_from_homography(Hh, K)
            hs.append(h_e); ds.append(d_e)
        hs, ds = np.array(hs), np.array(ds)
        print(f"  σ={pn:4.1f}px  {np.median(hs):7.2f} (±{np.std(hs):5.2f})  "
              f"{np.median(ds):9.2f} (±{np.std(ds):5.2f})   真値 h={H_CAM}, {DEPRESSION}°")
    print("  ※ 内部パラメータがあれば h は現地で測らなくても出る")

    # --- 7. 残差による異常点の検出 ---
    print(f"\n[7] 残差で測り間違いを検出できるか（マーカー1点を 50 cm ずらす）")
    bad = markers.copy()
    bad[2] += np.array([0.5, 0.0])
    Hb = estimate_homography(uv_m, bad)
    res = reprojection_error(Hb, uv_m, bad)
    for i, r in enumerate(res):
        print(f"  マーカー{i+1}: 残差 {r:.3f} m {'  ← ずらした点' if i == 2 else ''}")
    print("  ※ 6点あれば残差に現れる。4点だと厳密解になり残差が 0 で検出できない")

    # --- 図 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for pn, c in zip((1.0, 2.0, 5.0), ("C0", "C1", "C2")):
        ax.plot(targets[:, 1], np.median(err_by_noise[pn], axis=0), "o-", color=c,
                label=f"pixel noise {pn:.0f} px")
    ax.axhline(TARGET_ACC, color="r", ls=":", label=f"target {TARGET_ACC:.2f} m")
    ax.axhline(RANGE_BIN, color="k", ls="--", label=f"radar bin {RANGE_BIN:.2f} m")
    # 図中は日本語フォント未設定のため英語（0727/ のスクリプトと同じ方針）
    ax.set_xlabel("Target ground range Y [m]"); ax.set_ylabel("Ground error [m]")
    ax.set_title("Accuracy vs range"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(targets[:, 1], np.median(e_wide, axis=0), "o-", label="markers 20-50 m")
    ax.plot(targets[:, 1], np.median(e_near, axis=0), "s-", label="markers 20-35 m")
    ax.axvspan(35, 50, color="r", alpha=.08)
    ax.axhline(TARGET_ACC, color="r", ls=":")
    ax.set_xlabel("Target ground range Y [m]"); ax.set_ylabel("Ground error [m]")
    ax.set_title("Extrapolation beyond markers"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_yscale("log")

    ax = axes[2]
    uv_m_p, uv_t_p = project(markers), project(targets)
    ax.plot(uv_m_p[:, 0], uv_m_p[:, 1], "o", label="markers")
    ax.plot(uv_t_p[:, 0], uv_t_p[:, 1], "x", label="targets")
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
    ax.set_title("Image-plane layout"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.tight_layout()
    p = OUT_DIR / "homography_check.png"
    fig.savefig(p, dpi=150)
    print(f"\n保存: {p}")


if __name__ == "__main__":
    main()
