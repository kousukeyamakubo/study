# 画像上でマーカーの対応点をクリックして作る。
#
# 【--xy: カラーコーン等の個別マーカー用（2026-09-10 追加、09-11 改訂）】
# 舗装タイルの格子（--pitch、既定）は「測るのは1回、あとは格子番号を数えるだけ」だが、
# 実運用では数える作業自体が負担だった。カラーコーンのように個別に置いたマーカーは
# 格子に乗らないので数えられない。
#
# ただし現地でメジャーで直接測れるのは2点間の「距離」であって「X方向・Y方向」ではない
# ため、当初案の「実測X,Yをそのまま入力する」は現地では機能しない（方向をどう決めるか
# 分からない）。かわりに、最初にクリックした2点を固定アンカーA・Bとし
# （Aが原点、A→Bの向きが+X軸、A-B間の実測距離だけ最初に1回入力する）、
# 3点目以降は「アンカーA・Bまでの距離」の2本だけを入力すれば、方向を意識せず
# `homography.trilaterate()` が座標を計算する。
#
# 【アンカーは離れた2点を選ぶこと】
# アンカー間隔が測る対象までの距離に対して狭いと、メジャー誤差が座標に大きく増幅される
# （`trilaterate()` のdocstring参照）。目安は10m以上、理想はマーカー全体の横の広がり
# （±8m=16m）程度。狭いアンカー間隔は起動時に警告する。
#
# 【--auto: 色による自動検出（2026-09-10 追加）】
# タイルの角は「角」という点自体が曖昧（拡大しないと精度良く指せない）。
# カラーコーンのように背景から色で際立つ物体なら、ラフにクリックした周辺の色を
# サンプリングして同色領域を検出し、外接矩形の**下端中央**を候補点として出せる。
# コーンは軸対称なので下端中央は接地点（地面との接点）の良い近似になる
# （`detections.foot_of()` の bbox 下辺中央と同じ考え方）。
# ラフクリックの精度で足りる分、クリック自体の負担が減る。誤検出時は
# 手動クリックでいつでも上書きできるようにしてある（自動検出は候補の提示に留める）。
#
# 【2段クリックにしている理由（--auto 未指定時、従来どおり）】
# 4K を画面に収めると 1 表示px = 4 画像px になり、指す精度が 3〜4 cm 落ちる。
# 粗く指してから拡大窓で確定することで、1 画像px（≒1 cm）で指せるようにする。
#
# 【平面性の検査を兼ねている】
# 歩道に排水勾配があると平面の仮定が崩れる。再投影残差が**系統的に偏る**か
# ランダムに散るかで判定できるので、点を指した時点で自動的に分かる。
#
# 依存: opencv-python, numpy
#
# 使い方:
#   python pick_markers.py frames/00025.jpg --out markers.csv
#   python pick_markers.py frames/00025.jpg --check markers.csv   # 指し直さず再評価だけ
#   python pick_markers.py frames/00025.jpg --xy --auto --out markers.csv   # カラーコーン運用

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import sys
from pathlib import Path

# lib/ を import 可能にする。各スクリプトは直接実行される前提なので、
# パッケージ化せずパスを通す方式にしている
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from homography import apply_h, estimate_homography, reprojection_error, trilaterate

TILE_M = 0.20          # タイル1辺[m]（明るい正方形。4タイル一括の実測から）
ZOOM = 8               # 拡大窓の倍率
HALF = 60              # 拡大窓に映す元画像の半幅[px]
COLS = ["u", "v", "i", "j", "X", "Y"]


def fit_scale(shape, max_w=1600, max_h=900) -> float:
    h, w = shape[:2]
    return min(max_w / w, max_h / h, 1.0)


def detect_marker_auto(im, u0: int, v0: int, patch_r: int = 6, win: int = HALF * 2,
                       hue_tol: int = 12, sat_tol: int = 80, val_tol: int = 80,
                       min_area: int = 20, max_area_frac: float = 0.5) -> tuple[float, float] | None:
    """ラフクリック点周辺の色でマーカー（カラーコーン等）を自動検出する。

    クリック点そのものの色を「このマーカーの色」のテンプレートとして使うので、
    コーンの色を事前にハードコードする必要がない（現地でどの色を使っても動く）。
    戻り値は外接矩形の**下端中央**＝地面との接点の近似（コーンは軸対称なので、
    `detections.foot_of()` の bbox 下辺中央と同じ理屈が使える）。見つからなければ None"""
    h, w = im.shape[:2]
    x0, y0 = max(0, u0 - win), max(0, v0 - win)
    x1, y1 = min(w, u0 + win), min(h, v0 + win)
    roi = im[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    # クリック点周辺の小パッチをテンプレート色にする（中央値で外れ値に強くする）
    px0, py0 = max(0, u0 - patch_r), max(0, v0 - patch_r)
    px1, py1 = min(w, u0 + patch_r), min(h, v0 + patch_r)
    patch_hsv = cv2.cvtColor(im[py0:py1, px0:px1], cv2.COLOR_BGR2HSV)
    ref = np.median(patch_hsv.reshape(-1, 3), axis=0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo = np.array([max(0, ref[0] - hue_tol), max(0, ref[1] - sat_tol), max(0, ref[2] - val_tol)])
    hi = np.array([min(179, ref[0] + hue_tol), 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    # クリックが背景に外れると「背景全体」が同色領域として検出されてしまう。
    # マーカーは局所的な物体である前提なので、ROI の大部分を占める塊は誤検出とみなす
    if area > max_area_frac * roi.shape[0] * roi.shape[1]:
        return None
    bx, by, bw, bh = cv2.boundingRect(c)
    return x0 + bx + bw / 2, y0 + by + bh


def refine(im, u0: int, v0: int, auto: bool = False) -> tuple[int, int] | None:
    """粗く指した点の周りを拡大表示し、1 画像px の精度で確定させる。

    auto=True なら色による自動検出結果を候補点として緑十字で提示する。
    Enter で採用、クリックすればいつでも手動指定で上書きできる（自動検出は誤ることが
    あるので、確定させず候補の提示に留める）"""
    h, w = im.shape[:2]
    x0, y0 = max(0, u0 - HALF), max(0, v0 - HALF)
    x1, y1 = min(w, u0 + HALF), min(h, v0 + HALF)
    crop = cv2.resize(im[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                      interpolation=cv2.INTER_NEAREST)
    # 中心の十字は「今の候補位置」。ここを目標の角に合わせてクリックする
    cx, cy = (u0 - x0) * ZOOM, (v0 - y0) * ZOOM
    cv2.drawMarker(crop, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 40, 1)

    suggestion = detect_marker_auto(im, u0, v0) if auto else None
    if suggestion is not None:
        su, sv = suggestion
        sx, sy = int((su - x0) * ZOOM), int((sv - y0) * ZOOM)
        cv2.drawMarker(crop, (sx, sy), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        msg = "green=auto候補(Enterで採用)  /  click=手動指定  /  r=redo  /  esc=cancel"
    else:
        msg = "click the corner  /  r=redo  /  esc=cancel"
    cv2.putText(crop, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    got = {}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            got["p"] = (x0 + x / ZOOM, y0 + y / ZOOM)

    cv2.namedWindow("zoom")
    cv2.setMouseCallback("zoom", on_mouse)
    while True:
        cv2.imshow("zoom", crop)
        k = cv2.waitKey(20) & 0xFF
        if "p" in got:
            cv2.destroyWindow("zoom")
            return int(round(got["p"][0])), int(round(got["p"][1]))
        if suggestion is not None and k in (13, 10):        # Enter = 自動候補を採用
            cv2.destroyWindow("zoom")
            return int(round(suggestion[0])), int(round(suggestion[1]))
        if k in (27, ord("r")):                 # esc / r
            cv2.destroyWindow("zoom")
            return None


def evaluate(rows: list[dict]) -> None:
    """再投影残差を出す。マーカーの数え間違いと歩道の非平面性の両方がここに出る"""
    if len(rows) < 4:
        print(f"  点が {len(rows)} 個。ホモグラフィには 4 点以上必要")
        return
    uv = np.array([[r["u"], r["v"]] for r in rows], float)
    xy = np.array([[r["X"], r["Y"]] for r in rows], float)
    H = estimate_homography(uv, xy)
    e = reprojection_error(H, uv, xy)

    # 突出した残差は「数え間違い」を示す。全体が小さくても 1 点だけ浮くのが典型
    outlier = (e > 3 * np.median(e)) & (e > 0.05)

    print(f"\n再投影残差[m]  平均 {e.mean():.3f} / 最大 {e.max():.3f}")
    for r, err, o in zip(rows, e, outlier):
        print(f"  ({r['i']:>3},{r['j']:>3})  ({r['X']:6.2f},{r['Y']:6.2f}) m  "
              f"残差 {err:.3f} m{'  ★' if o else ''}")

    # 判定: レーダー距離分解能 0.846 m の 1/3 を目標にしている（README §1）
    if outlier.any():
        print(f"→ ★ の {int(outlier.sum())} 点だけ残差が突出している。"
              "実測値の入力ミスや自動検出の誤りを疑うこと"
              "（格子モードなら格子番号の数え間違い＝1 枚ずれで 0.20 m）。"
              "全体が目標内でも、この点は直すべき")
    elif e.max() < 0.05:
        print("→ 残差はほぼゼロ。指し間違いも無く、歩道は平面とみなせる")
    elif e.max() < 0.28:
        print("→ 目標 0.28 m 以内。ラベルとして使える")
    else:
        print("→ 目標 0.28 m を超えている。特定の点が突出しているわけではないので、"
              "指し誤差の蓄積か、歩道が平面でない可能性")

    # 平面性: 残差ベクトルが一方向に揃っていれば勾配、散っていれば単なる指し誤差
    d = apply_h(H, uv) - xy
    if len(rows) >= 6 and e.mean() > 0.02:
        bias = np.linalg.norm(d.mean(axis=0)) / max(e.mean(), 1e-9)
        print(f"  残差の偏り {bias:.2f}（0 に近い=ランダム、1 に近い=系統的）")
        if bias > 0.5:
            print("  → 系統的。歩道に勾配がある可能性。ただし目標内なら実用上は問題ない")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("markers.csv"))
    ap.add_argument("--check", type=Path, default=None,
                    help="既存 CSV を読んで評価だけ行う")
    ap.add_argument("--pitch", type=float, default=TILE_M, help="タイル1辺[m]（格子モード）")
    ap.add_argument("--xy", action="store_true",
                    help="格子番号でなく、アンカー2点(1・2点目)からの実測距離を入力し、"
                         "trilaterateで座標を計算する（カラーコーン等、格子に乗らない個別マーカー用）")
    ap.add_argument("--auto", action="store_true",
                    help="ラフクリック周辺の色でマーカーを自動検出し、候補点を提示する"
                         "（Enterで採用、クリックで手動上書き）")
    ap.add_argument("--annotate", action="store_true",
                    help="格子番号でなくラベルを付ける（歩道の縁・街灯などの図示用）。"
                         "地上座標は ground_plot.py 側で H を使って求める")
    args = ap.parse_args()
    if args.annotate and args.xy:
        raise SystemExit("--annotate と --xy は同時指定できない")

    if args.check:
        with open(args.check, newline="", encoding="utf-8") as f:
            rows = [{k: float(v) if k not in ("i", "j") else int(v)
                     for k, v in r.items()} for r in csv.DictReader(f)]
        print(f"{args.check}: {len(rows)} 点")
        evaluate(rows)
        return

    im = cv2.imread(str(args.image))
    if im is None:
        raise FileNotFoundError(args.image)
    s = fit_scale(im.shape)
    view = cv2.resize(im, None, fx=s, fy=s)
    rows: list[dict] = []

    print(f"{args.image.name}: {im.shape[1]}x{im.shape[0]}  表示倍率 {s:.2f}")
    if args.xy:
        auto_note = "（緑=自動候補、Enterで採用）" if args.auto else ""
        print(f"マーカーをクリック{auto_note} → 拡大窓で確定")
        print("  1点目=アンカーA(原点) / 2点目=アンカーB(A-B間の実測距離を入力)")
        print("  3点目以降=アンカーA・Bまでの実測距離を入力（trilaterateが座標を計算）")
    else:
        print("タイルの角を粗くクリック → 拡大窓で確定 → 端末で格子番号を入力")
    print("  u=直前を取り消し / s=保存 / q=保存して終了")

    pending = {}
    last: dict[str, str] = {}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pending["p"] = (int(x / s), int(y / s))

    cv2.namedWindow("pick")
    cv2.setMouseCallback("pick", on_mouse)

    while True:
        disp = view.copy()
        for n, r in enumerate(rows):
            p = (int(r["u"] * s), int(r["v"] * s))
            cv2.drawMarker(disp, p, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            if args.annotate:
                tag = r["label"]
            elif args.xy:
                tag = f"({r['X']:.2f},{r['Y']:.2f})"
            else:
                tag = f"({r['i']},{r['j']})"
            cv2.putText(disp, f"{n}:{tag}", (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("pick", disp)
        k = cv2.waitKey(20) & 0xFF

        if "p" in pending:
            u0, v0 = pending.pop("p")
            p = refine(im, u0, v0, auto=args.auto)
            if p is not None:
                if args.annotate:
                    # 同じラベルを続けて打てるよう、空入力なら直前のラベルを引き継ぐ
                    lab = input(f"  ({p[0]},{p[1]}) のラベル > ").strip() or last.get("l", "")
                    if not lab:
                        print("  → ラベルが空。この点は破棄")
                        continue
                    last["l"] = lab
                    rows.append(dict(u=p[0], v=p[1], label=lab))
                    print(f"  #{len(rows)-1} 追加 {lab}")
                    continue
                if args.xy:
                    # 格子に乗らない個別マーカー用。メジャーで測れるのは「距離」なので、
                    # 1点目=アンカーA(原点)、2点目=アンカーB(+X軸上)を固定し、
                    # 3点目以降はA・Bまでの距離2本だけを入力して trilaterate に投げる
                    if len(rows) == 0:
                        rows.append(dict(u=p[0], v=p[1], i=0, j=0, X=0.0, Y=0.0))
                        print(f"  #0 追加 → アンカーA（原点）")
                        continue
                    if len(rows) == 1:
                        try:
                            d_ab = float(input(f"  ({p[0]},{p[1]}) アンカーA-B間の実測距離[m] > "))
                        except ValueError:
                            print("  → 入力が不正。この点は破棄")
                            continue
                        if d_ab < 10.0:
                            print(f"  ⚠ アンカー間隔 {d_ab:.1f} m は狭い。"
                                  "遠方の点でメジャー誤差が大きく増幅される（10m以上を推奨）")
                        rows.append(dict(u=p[0], v=p[1], i=1, j=0, X=d_ab, Y=0.0))
                        print(f"  #1 追加 → アンカーB ({d_ab:.2f}, 0.00) m")
                        continue
                    baseline = rows[1]["X"]
                    try:
                        t = input(f"  ({p[0]},{p[1]}) アンカーAまで, Bまでの実測距離[m] > "
                                 ).replace(",", " ").split()
                        r_a, r_b = float(t[0]), float(t[1])
                    except (ValueError, IndexError):
                        print("  → 入力が不正。この点は破棄")
                        continue
                    got = trilaterate(r_a, r_b, baseline)
                    if got is None:
                        print("  → 距離が矛盾している（三角不等式を満たさない）。測り直すこと。この点は破棄")
                        continue
                    X, Y = got
                    rows.append(dict(u=p[0], v=p[1], i=len(rows), j=0, X=X, Y=Y))
                    print(f"  #{len(rows)-1} 追加 ({X:.2f}, {Y:.2f}) m")
                    continue
                # 格子番号は整数。原点は最初に指した点にするのが分かりやすい
                try:
                    t = input(f"  ({p[0]},{p[1]}) の格子番号 i,j > ").replace(",", " ").split()
                    i, j = int(t[0]), int(t[1])
                except (ValueError, IndexError):
                    print("  → 入力が不正。この点は破棄")
                    continue
                rows.append(dict(u=p[0], v=p[1], i=i, j=j,
                                 X=i * args.pitch, Y=j * args.pitch))
                print(f"  #{len(rows)-1} 追加 ({i},{j}) = "
                      f"({i*args.pitch:.2f}, {j*args.pitch:.2f}) m")

        if k == ord("u") and rows:
            print(f"  #{len(rows)-1} を取り消し")
            rows.pop()
        if k in (ord("s"), ord("q")):
            cols = ["u", "v", "label"] if args.annotate else COLS
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, cols)
                w.writeheader()
                w.writerows(rows)
            print(f"\n保存: {args.out} ({len(rows)} 点)")
            if not args.annotate:
                evaluate(rows)
            if k == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
