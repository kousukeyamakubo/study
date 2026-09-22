# マーカーの leave-one-out hold-out による H の精度評価。
#
# 【なぜ要るか】
# 2026-09-12 までの評価は in-sample の残差だけだった（`../0912/check_markers.py`）。
# 残差は「H を当てはめるのに使った点が、そのHでどれだけ当たるか」なので、
# 点を増やすほど良く見える一方で、未知の点に対する性能は何も言っていない。
# ラベルとして使うのは「Hに使っていない位置に立っている人」なので、
# 知りたいのは未知点での誤差のほうになる。
#
# 【やり方】
# アンカー A, B は地上座標系の定義そのもの（(0,0) と (b,0)）なので常に当てはめに含め、
# それ以外の点を1つずつ抜く。抜いた点は H の推定に一切使わず、その点の画素座標を
# H で写した結果と、現地でA・Bからの距離2本を測って得た座標とを比べる。
# 画素座標は markers.csv に全点ぶん残っているので、Hに使わなかった点も予測できる。
#
# 【歪み補正】
# レンズ歪み補正はパイプライン本体では未適用（`../docs/calibration.md` の未解決項目）。
# 掛けると当てはめが良くなるのか、それとも差が出ないのかを数字にするため、
# 補正あり・なしの両方を同じ手順で回して並べる。
#
# 【測量側の感度も一緒に出す】
# hold-out の ΔY が大きいとき、それが H の誤差なのか真値の誤差なのかは
# hold-out の数字だけでは分けられない。分ける材料として、
# 「メジャーの読みが5cm違ったら真値の座標が何m動くか」を同じスクリプトで出す。
# これは誤差の仮定ではなく、距離2本→座標の式が持っている感度そのもの。
#
# 使い方:
#   python holdout_eval.py                    # 既定の 0912 データ
#   python holdout_eval.py <markers.csv>      # 他の markers.csv

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0817" / "lib"))
from homography import apply_h, estimate_homography, spread_axes  # noqa: E402

MARKERS = ROOT / "experiment_0912" / "atlas_log_20260912_164900.markers.csv"
K_NPZ = ROOT / "0817" / "calib" / "K.npz"

# markers.csv の各行に対応する現地での呼び名。0912 は P8・P10 が三角不等式を満たさず
# 破棄され、P13 は街灯に隠れて未入力なので、行番号と現地の番号がずれる
# （`../0912/check_markers.py` の TYPED と同じ）。表を 0912/README.md と突き合わせるために持つ
LABELS_0912 = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9", "P11", "P12"]

# 測量側の感度を出すときの前提。いずれも現場で決まっている値で、誤差の仮定ではない
READING = 0.05      # メジャーの目盛り。5cm 刻みで読んでいる
ROAD_WIDTH = 4.0    # 歩道の幅[m]。2026-09-12 の現場で実測。Y はこの中にしか置けない
TARGET = 0.28       # ラベルに要求される精度[m]


def load_markers(path):
    """markers.csv → (アンカー2点の画素, アンカーの地上座標, 残り点の画素, 残り点の地上座標)。

    先頭2行がアンカー A, B という並びは pick_markers.py の出力仕様。
    X,Y 列は現地で測った距離2本を trilaterate した値で、これが今回の真値になる。
    """
    rows = list(csv.DictReader(open(path)))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows], float)
    xy = np.array([[float(r["X"]), float(r["Y"])] for r in rows], float)
    return uv[:2], xy[:2], uv[2:], xy[2:]


def undistort(uv, k_npz=K_NPZ):
    """レンズ歪みを除いた画素座標を返す。P=K を渡して画素のまま戻す（正規化座標にしない）"""
    import cv2

    d = np.load(k_npz)
    K, dist = d["K"], d["dist"]
    out = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K)
    return out.reshape(-1, 2)


def leave_one_out(anchor_uv, anchor_xy, uv, xy):
    """1点ずつ抜いて H を作り直し、抜いた点での予測誤差を返す。

    戻り値は (予測座標, 誤差ベクトル, 当てはめに使った点の副軸方向の広がり) の配列。
    副軸の広がりも返すのは、点を抜いたことで配置が退化して
    「数字は出るが信用できないH」になっていないかを見るため（spread_axes 参照）。
    """
    pred = np.zeros_like(xy)
    minor = np.zeros(len(xy))
    for i in range(len(xy)):
        keep = [j for j in range(len(xy)) if j != i]
        src = np.vstack([anchor_uv, uv[keep]])
        dst = np.vstack([anchor_xy, xy[keep]])
        H = estimate_homography(src, dst)
        pred[i] = apply_h(H, uv[i][None, :])[0]
        minor[i] = spread_axes(dst)[1]
    return pred, pred - xy, minor


def report(title, xy, err, minor, labels):
    print(f"\n=== {title} ===")
    print("  点    真値(X, Y)           予測(X, Y)           ΔX      ΔY      距離   当てはめ副軸")
    for lab, t, e, m in zip(labels, xy, err, minor):
        p = t + e
        print(f"  {lab:<4s} ({t[0]:7.3f},{t[1]:6.3f})  ({p[0]:7.3f},{p[1]:6.3f})  "
              f"{e[0]:7.3f} {e[1]:7.3f} {np.hypot(*e):7.3f}   {m:5.3f} m")

    rms = np.sqrt((err ** 2).mean(axis=0))
    dist = np.hypot(err[:, 0], err[:, 1])
    worst = int(np.argmax(dist))
    print(f"  RMS  ΔX {rms[0]:.3f} m / ΔY {rms[1]:.3f} m / 距離 {np.sqrt((dist ** 2).mean()):.3f} m")
    print(f"  最大 距離 {dist[worst]:.3f} m（{labels[worst]}）")
    # 符号付きの平均。誤差が散らばっているのか一方向に寄っているのかで意味が変わる
    print(f"  偏り ΔX {err[:, 0].mean():+.3f} m / ΔY {err[:, 1].mean():+.3f} m")
    return rms, dist


def survey_sensitivity(baseline, X, Y, delta=READING):
    """メジャーの読みが ±delta 違ったとき、距離2本から出る座標が動く幅の半分[m]。

    誤差モデルの仮定ではなく、trilaterate の式が持っている感度そのもの。
    ra, rb の両方に ±delta を与えた4通りの最悪幅を見る。平方根の中身が負になる
    （＝三角不等式を満たさず棄却される）組み合わせは Y=0 に潰れたものとして数える。
    """
    r_a, r_b = np.hypot(X, Y), np.hypot(baseline - X, Y)
    pts = []
    for d_a in (-delta, delta):
        for d_b in (-delta, delta):
            x = ((r_a + d_a) ** 2 - (r_b + d_b) ** 2 + baseline ** 2) / (2 * baseline)
            y2 = (r_a + d_a) ** 2 - x ** 2
            pts.append((x, float(np.sqrt(y2)) if y2 > 0 else 0.0))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2


def usable_band(baseline, target=TARGET, width=ROAD_WIDTH, delta=READING):
    """道幅 width のうち、Y の感度が target 以下に収まる帯の下端[m]。無ければ None。

    基線中央（Y の感度が最も悪くなる位置）で判定する。
    """
    for y in np.arange(0.05, width + 1e-9, 0.01):
        if survey_sensitivity(baseline, baseline / 2, y, delta)[1] <= target:
            return float(y)
    return None


def report_survey_limits(baseline):
    """なぜ真値の Y だけ壊れるかを、基線長と道幅の関係として出す。

    【なぜ要るか】hold-out の ΔY を「H の誤差」と読まないための材料。
    道幅が4mしかない現場では、アンカー2本も点もほぼ一直線に並ぶしかなく、
    距離の測定が Y 方向にほとんど感度を持たない（∂ra/∂Y = Y/ra）。
    """
    print(f"\n=== 測量側の感度（読み ±{READING * 100:.0f} cm、道幅 {ROAD_WIDTH:.1f} m）===")
    print(f"  基線中央での Y 感度と、目標 {TARGET} m を満たす帯")
    print("   基線[m]   Y=1.0   Y=2.0   Y=3.0   Y=4.0   | 使える帯")
    for b in (10.0, 15.0, 20.0, 25.0, baseline):
        row = "  ".join(f"{survey_sensitivity(b, b / 2, y)[1]:6.3f}" for y in (1.0, 2.0, 3.0, 4.0))
        low = usable_band(b)
        band = f"Y {low:.2f}〜{ROAD_WIDTH:.2f} m（幅 {ROAD_WIDTH - low:.2f} m）" if low else "無し"
        print(f"   {b:6.1f}  {row}  | {band}")

    # 基線を詰めれば帯は広がるが、基線の外側に置いた点では逆に破綻する。
    # 「基線はマーカー場を跨いでいないといけない」ことをここで示す
    print(f"\n  基線を詰めた場合（b=15 m、Y=2.0 m）に X をずらしたときの感度")
    print("     X[m]    X感度   Y感度")
    for x in (7.5, 13.0, 20.0, 25.0, 30.0):
        s_x, s_y = survey_sensitivity(15.0, x, 2.0)
        tag = "" if x <= 15.0 else "  ← 基線の外側"
        print(f"   {x:6.1f}   {s_x:6.3f}  {s_y:6.3f}{tag}")


def measurable(xy, baseline):
    """各点の真値がその位置で測れているか（Y 感度が目標以下か）を返す。

    点ごとに実際の (X, Y) で感度を出す。基線中央に近いほど Y 感度が悪くなるので、
    同じ Y でも X によって判定が変わる。
    """
    return np.array([survey_sensitivity(baseline, x, y)[1] <= TARGET for x, y in xy])


def report_fit_contamination(anchor_uv, anchor_xy, uv, xy, labels, baseline):
    """真値が測れていない点を当てはめ集合に入れると H がどれだけ汚れるかを見る。

    【なぜ要るか】0919 の LOO は markers.csv の全点で当てはめる（run_pipeline.py と同じ）。
    しかし真値の Y が壊れている点を当てはめに入れると、DLT は全点を等しく信頼するので
    H 自体が線に引き寄せられる（`../0912/xy_estimation.md` の「最小二乗を汚す」）。
    hold-out 誤差のうち「抜いた点の真値が悪い」ぶんと「当てはめが汚れている」ぶんを
    分けるため、当てはめ集合から測れていない点を外した場合と並べる。
    """
    ok = measurable(xy, baseline)
    print(f"\n=== 真値の測りやすさ（読み ±{READING * 100:.0f} cm、各点の実際の位置で評価）===")
    for lab, (x, y), good in zip(labels, xy, ok):
        s = survey_sensitivity(baseline, x, y)[1]
        print(f"  {lab:<4s} Y={y:6.3f}  Y感度 {s:6.3f} m  "
              f"{'測れている' if good else '退化（真値が信用できない）'}")

    # 【この表の読み方の注意】
    # 「測れている点だけ」で当てはめると、測れている点自身を抜いたときに
    # 残りがアンカー2点＋1〜2点まで落ちて DLT の最小構成（4点）になる。
    # そこで出る数字は当てはめの退化を見ているだけなので比較にならない。
    # 意味があるのは、抜いた点が測れていない点である行（当てはめ側は満杯のまま）。
    print(f"\n=== 当てはめ集合を変えたときの hold-out 誤差 ===")
    print("  抜いた点  真値のY   全点で当てはめ   測れている点だけ（当てはめ点数）")
    for i, lab in enumerate(labels):
        row, sizes = [], []
        for keep in ([j for j in range(len(xy)) if j != i],
                     [j for j in range(len(xy)) if j != i and ok[j]]):
            H = estimate_homography(np.vstack([anchor_uv, uv[keep]]),
                                    np.vstack([anchor_xy, xy[keep]]))
            row.append(float(np.hypot(*(apply_h(H, uv[i][None, :])[0] - xy[i]))))
            sizes.append(len(keep) + 2)
        # 当てはめが5点を切ると DLT が最小構成になり、数字が意味を持たない
        mark = "  ← 当てはめが最小構成。比較不能" if sizes[1] <= 4 else ""
        print(f"  {lab:<8s} {xy[i][1]:6.3f}   {row[0]:11.3f}   {row[1]:14.3f} ({sizes[1]}点){mark}")
    print(f"  ※ 意味があるのは当てはめ点数が {int(ok.sum()) + 2} 点のままの行")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else MARKERS
    a_uv, a_xy, uv, xy = load_markers(path)
    labels = LABELS_0912 if (path == MARKERS and len(xy) == len(LABELS_0912)) \
        else [f"#{i + 1}" for i in range(len(xy))]

    print(f"markers: {path}")
    print(f"アンカー2点を固定し、残り {len(xy)} 点を1点ずつ hold-out する")
    print(f"基線 A-B = {np.hypot(*(a_xy[1] - a_xy[0])):.2f} m")

    # アンカーも同じ変換を通さないと座標系が混ざるので、全点まとめて補正してから分ける
    ud = undistort(np.vstack([a_uv, uv]))
    variants = (("歪み補正なし（run_pipeline.py の現状）", a_uv, uv),
                ("歪み補正あり（K.npz）", ud[:2], ud[2:]))

    results = {}
    for title, anchors, pts in variants:
        pred, err, minor = leave_one_out(anchors, a_xy, pts, xy)
        results[title] = report(title, xy, err, minor, labels)

    baseline = float(np.hypot(*(a_xy[1] - a_xy[0])))
    report_fit_contamination(a_uv, a_xy, uv, xy, labels, baseline)
    report_survey_limits(baseline)

    print("\n=== 補正あり / なし の比較 ===")
    print("  条件                             ΔX RMS   ΔY RMS   距離RMS")
    for title, (rms, dist) in results.items():
        print(f"  {title:<32s} {rms[0]:7.3f}  {rms[1]:7.3f}  {np.sqrt((dist ** 2).mean()):7.3f}")


if __name__ == "__main__":
    main()
