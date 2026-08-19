# 画像上でタイルの角をクリックしてマーカーの対応点を作る。
#
# 【地上座標を測らない理由】
# 舗装が 20 cm の規則格子なので、地上座標は「格子番号 × ピッチ」で決まる。
# 巻尺で測るのは 1 回（4 タイル一括）で済み、あとは数えるだけ。
# → 累積誤差は 20 m 先で ±5 cm 程度で、目標 0.28 m に対して十分小さい。
#
# 【2段クリックにしている理由】
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

from homography import apply_h, estimate_homography, reprojection_error

TILE_M = 0.20          # タイル1辺[m]（明るい正方形。4タイル一括の実測から）
ZOOM = 8               # 拡大窓の倍率
HALF = 60              # 拡大窓に映す元画像の半幅[px]
COLS = ["u", "v", "i", "j", "X", "Y"]


def fit_scale(shape, max_w=1600, max_h=900) -> float:
    h, w = shape[:2]
    return min(max_w / w, max_h / h, 1.0)


def refine(im, u0: int, v0: int) -> tuple[int, int] | None:
    """粗く指した点の周りを拡大表示し、1 画像px の精度で確定させる"""
    h, w = im.shape[:2]
    x0, y0 = max(0, u0 - HALF), max(0, v0 - HALF)
    x1, y1 = min(w, u0 + HALF), min(h, v0 + HALF)
    crop = cv2.resize(im[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                      interpolation=cv2.INTER_NEAREST)
    # 中心の十字は「今の候補位置」。ここを目標の角に合わせてクリックする
    cx, cy = (u0 - x0) * ZOOM, (v0 - y0) * ZOOM
    cv2.drawMarker(crop, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 40, 1)
    cv2.putText(crop, "click the corner  /  r=redo  /  esc=cancel", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

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
              "格子番号の数え間違いを疑うこと（1 枚ずれ = 0.20 m）。"
              "全体が目標内でも、この点は直すべき")
    elif e.max() < 0.05:
        print("→ 残差はほぼゼロ。格子の数え間違いも無く、歩道は平面とみなせる")
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
    ap.add_argument("--pitch", type=float, default=TILE_M, help="タイル1辺[m]")
    ap.add_argument("--annotate", action="store_true",
                    help="格子番号でなくラベルを付ける（歩道の縁・街灯などの図示用）。"
                         "地上座標は ground_plot.py 側で H を使って求める")
    args = ap.parse_args()

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
            tag = r["label"] if args.annotate else f"({r['i']},{r['j']})"
            cv2.putText(disp, f"{n}:{tag}", (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("pick", disp)
        k = cv2.waitKey(20) & 0xFF

        if "p" in pending:
            u0, v0 = pending.pop("p")
            p = refine(im, u0, v0)
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
