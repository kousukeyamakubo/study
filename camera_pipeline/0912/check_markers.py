# 2026-09-12 の実測マーカーの健全性診断。README.md の数値を再現する。
#
# 【なぜ要るか】
# markers.csv は「クリックした画素」と「距離2本から計算した地上座標」の対でできている。
# 距離2本から座標を出す方式は、点がアンカーA-Bの直線に近いと Y が決まらなくなる
# （寄り道 ra+rb-基線 が数cmしか無く、メジャーの読み取り誤差に埋もれる）。
# この退化は残差を見ただけでは「どの点が悪いのか」が分かりにくいので、
# (1)測定値自体の成立性 (2)Yの感度 (3)割り当てのずれ を分けて出す。
#
# 使い方:
#   python check_markers.py            # 既定の 0912 データを診断

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0817" / "lib"))
from homography import apply_h, estimate_homography, reprojection_error  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MARKERS = ROOT / "experiment_0912" / "atlas_log_20260912_164900.markers.csv"
BASELINE = 35.40

# docs/assets/measurement_sheet_0912.md に手書きで記録した実測値 (Aまで, Bまで)[m]。
# 現地で紙に書いた値が唯一の一次記録なので、ここに転記して突き合わせの基準にする
SHEET = [(3.10, 38.35), (3.30, 32.30), (8.15, 28.00), (9.15, 26.25), (14.50, 21.00),
         (15.80, 20.30), (16.35, 19.05), (20.90, 14.40), (21.85, 13.95), (24.50, 10.85),
         (28.05, 8.50), (30.90, 4.75), (33.45, 2.45)]
# markers.csv の各クリックで入力したシート行番号（1始まり）。P8/P10 は弾かれ P13 は未入力
TYPED = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12]
WELL_CONDITIONED = [3, 6, 9, 11]     # Y が安定して決まる行（遠い側の縁に置いた点）


def trilaterate_xy(r_a, r_b, baseline=BASELINE):
    """距離2本 → (X, Y)。Y が虚数になる（＝三角不等式を満たさない）場合は Y=None"""
    x = (r_a ** 2 - r_b ** 2 + baseline ** 2) / (2 * baseline)
    y2 = r_a ** 2 - x ** 2
    return x, (float(np.sqrt(y2)) if y2 > 0 else None)


def load_clicks(path=MARKERS):
    rows = list(csv.DictReader(open(path)))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    return uv[0], uv[1], uv[2:]          # A, B, クリックした点


def fit(anchor_a, anchor_b, pts, xys):
    """アンカー2点＋与えられた対応から画像→地上のHを推定する"""
    return estimate_homography(np.vstack([anchor_a, anchor_b] + list(pts)),
                               np.vstack([[0.0, 0.0], [BASELINE, 0.0]] + list(xys)))


def main():
    sheet_xy = [trilaterate_xy(*s) for s in SHEET]
    A, B, clicks = load_clicks()

    print("=== 1. 測定値そのものの成立性 ===")
    for k, ((ra, rb), (X, Y)) in enumerate(zip(SHEET, sheet_xy), 1):
        if Y is None:
            print(f"  P{k:<2d} ra+rb={ra + rb:.2f} < 基線{BASELINE} → 平面上に存在しない（破棄される）")
        elif Y < 0.10:
            print(f"  P{k:<2d} X={X:7.3f} Y={Y:.6f} → AB線上。Yはゼロで頭打ちになっている")

    print("\n=== 2. Yの感度（メジャーを±5cm動かしたときのYの振れ幅）===")
    for k, ((ra, rb), (X, Y)) in enumerate(zip(SHEET, sheet_xy), 1):
        if Y is None:
            continue
        ys = [trilaterate_xy(ra + d, rb)[1] for d in (-0.05, 0.05)]
        if None in ys:
            print(f"  P{k:<2d} Y={Y:5.3f} → ±5cmで測定不能側に落ちる（退化）")
        else:
            print(f"  P{k:<2d} Y={Y:5.3f} → 振れ幅 {abs(ys[0] - ys[1]):.3f} m")

    print("\n=== 3. 割り当てのずれ（「コーン#mが不可視」13通りの仮説）===")
    # X は距離の差から決まるため全13行で信頼でき、どの仮説でも同じ10点で公平に採点できる
    scores = []
    for m in range(1, 14):
        true_idx = [(j if j < m else j + 1) for j in TYPED]
        if max(true_idx) > 13:
            continue
        usable = [k for k, t in enumerate(true_idx) if sheet_xy[t - 1][1] is not None]
        if len(usable) < 3:
            continue
        H = fit(A, B, [clicks[k] for k in usable],
                [[sheet_xy[true_idx[k] - 1][0], sheet_xy[true_idx[k] - 1][1]] for k in usable])
        pred = apply_h(H, clicks)
        dx = np.array([abs(pred[k, 0] - sheet_xy[true_idx[k] - 1][0]) for k in range(len(clicks))])
        scores.append((float(np.sqrt((dx ** 2).mean())), m))
        print(f"  コーン#{m:<2d} が不可視 → X残差RMS {scores[-1][0]:7.3f} m")
    scores.sort()
    print(f"  → 最も整合するのは #{scores[0][1]} が不可視（2番目より {scores[1][0] / scores[0][0]:.1f} 倍良い）")

    print("\n=== 4. 採用するH（Yが信頼できる点だけ）===")
    idx = [TYPED.index(j) for j in WELL_CONDITIONED]
    H = fit(A, B, [clicks[k] for k in idx],
            [[sheet_xy[j - 1][0], sheet_xy[j - 1][1]] for j in WELL_CONDITIONED])
    used = "A, B, " + ", ".join(f"P{j}" for j in WELL_CONDITIONED)
    print(f"  使用点: {used}")
    for k, j in enumerate(TYPED):
        X, Y = sheet_xy[j - 1]
        px, py = apply_h(H, clicks[k][None, :])[0]
        tag = "(Hに使用)" if j in WELL_CONDITIONED else ""
        print(f"  P{j:<2d} 実測({X:7.3f},{Y:6.3f}) 予測({px:7.3f},{py:6.3f}) "
              f"ズレ {np.hypot(px - X, py - Y):.3f} m {tag}")

    print("\n=== 5. 全点で当てはめた場合との比較 ===")
    for label, keep in (("全10点", TYPED), ("P4,P7を除く8点", [j for j in TYPED if j not in (4, 7)])):
        idx = [TYPED.index(j) for j in keep]
        pts = [clicks[k] for k in idx]
        xys = [[sheet_xy[j - 1][0], sheet_xy[j - 1][1]] for j in keep]
        Hk = fit(A, B, pts, xys)
        err = reprojection_error(Hk, np.vstack([A, B] + pts),
                                 np.vstack([[0.0, 0.0], [BASELINE, 0.0]] + xys))
        print(f"  {label}: RMS残差 {np.sqrt((err ** 2).mean()):.3f} m / 最大 {err.max():.3f} m")


if __name__ == "__main__":
    main()
