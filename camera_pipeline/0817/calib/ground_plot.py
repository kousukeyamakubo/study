# 画像上で指した地物をホモグラフィで地上に落とし、XY 平面に俯瞰図として描く。
#
# 【何のためか】
# ホモグラフィの推定に**使っていない**情報で答え合わせをするため。
# 残差はマーカー同士の無矛盾性しか見ておらず、変換が正しい保証にならない。
# 一方、歩道の縁・街灯の並びは「まっすぐ」「等間隔」という既知の性質を持つので、
# 変換後にそれが再現されるかで検証できる。
#
# 特に **レンズ歪み** はこれまで一度も検証していない。遠方まで伸びる直線が
# 曲がって出れば歪みの証拠になり、チェスボード較正の要否が決まる。
#
# 入力:
#   markers.csv     pick_markers.py が出した対応点（H の推定に使う）
#   annots.csv      pick_markers.py --annotate が出した u,v,label
#   （任意）bike_ground.csv  detections の地上座標
#
# 使い方:
#   python ground_plot.py markers.csv annots.csv --out ground.png
#   python ground_plot.py markers.csv annots.csv --det bike_ground.csv

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

# lib/ を import 可能にする。各スクリプトは直接実行される前提なので、
# パッケージ化せずパスを通す方式にしている
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from homography import apply_h, estimate_homography

TARGET_M = 0.28        # 目標精度[m]。レーダー距離分解能 0.846 m の 1/3
PIX_NOISE = 1.0        # マーカーの指し誤差[px]。不確かさの評価に使う

# ラベルの接頭辞で描き方を変える。線で結ぶべきものと点で置くべきものがある
LINE_PREFIXES = ("edge", "line", "curb", "側溝", "縁")
# 閉じた領域として塗るラベル。芝生や植栽帯など「面」で持ちたいものに使う
AREA_PREFIXES = ("area", "poly", "grass", "芝")


def load_markers(path: Path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    xy = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
    return uv, xy


def load_annots(path: Path) -> dict[str, np.ndarray]:
    g = defaultdict(list)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        g[r["label"]].append([float(r["u"]), float(r["v"])])
    return {k: np.array(v) for k, v in g.items()}


def straightness(p: np.ndarray) -> float:
    """点列が直線からどれだけ外れるか[m]。最小二乗直線への最大距離。

    歩道の縁は直線なので、これが大きければレンズ歪みか非平面を疑う"""
    if len(p) < 3:
        return float("nan")
    c = p - p.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = np.array([-vt[0, 1], vt[0, 0]])              # 主方向の法線
    return float(np.abs(c @ n).max())


def uncertainty(uv_m, xy_m, uv_q, n=200) -> np.ndarray:
    """マーカーを 1px 揺らしたときの、問い合わせ点の地上座標のばらつき[m]"""
    rng = np.random.default_rng(0)
    est = np.array([apply_h(estimate_homography(uv_m + rng.normal(0, PIX_NOISE, uv_m.shape),
                                                xy_m), uv_q)
                    for _ in range(n)])                     # (n, Q, 2)
    return np.sqrt(((est - est.mean(0)) ** 2).sum(-1).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markers", type=Path)
    ap.add_argument("annots", type=Path)
    ap.add_argument("--det", type=Path, default=None, help="bike_ground.csv 等")
    ap.add_argument("--out", type=Path, default=Path("ground.png"))
    args = ap.parse_args()

    uv_m, xy_m = load_markers(args.markers)
    H = estimate_homography(uv_m, xy_m)
    groups = load_annots(args.annots)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xy_m[:, 0], xy_m[:, 1], c="k", marker="+", s=90, zorder=5,
               label=f"markers ({len(xy_m)})")

    print("地物の変換結果")
    for lab, uv in sorted(groups.items()):
        g = apply_h(H, uv)
        unc = uncertainty(uv_m, xy_m, uv)
        line = lab.lower().startswith(LINE_PREFIXES)
        area = lab.lower().startswith(AREA_PREFIXES)
        if area:
            ax.fill(g[:, 0], g[:, 1], alpha=0.35, zorder=0, label=lab)
            print(f"  {lab:12s} {len(g):2d}点  面として描画"
                  f"  / 位置の不確かさ 平均 ±{unc.mean():.3f} m")
        elif line:
            o = np.argsort(g[:, 0])                       # 走行方向に並べてから結ぶ
            ax.plot(g[o, 0], g[o, 1], "-o", ms=3, lw=1.4, label=lab)
            s = straightness(g)
            note = ("" if len(g) >= 3 else
                    "  ★ 点が足りない（直線性の判定には 3 点以上、帯を描くには各縁 6〜8 点）")
            sv = f"{s:.3f} m" if len(g) >= 3 else "―"
            print(f"  {lab:12s} {len(g):2d}点  直線からのずれ 最大 {sv}"
                  f"  / 位置の不確かさ 平均 ±{unc.mean():.3f} m{note}")
        else:
            ax.scatter(g[:, 0], g[:, 1], s=45, label=lab)
            print(f"  {lab:12s} {len(g):2d}点  位置の不確かさ 平均 ±{unc.mean():.3f} m")
        for p, u in zip(g, unc):
            if u > TARGET_M:                              # 目標を割った点は目立たせる
                ax.plot(*p, "x", c="red", ms=10, mew=2, zorder=6)

    # 道幅: 2 本の縁があれば、距離によらず一定であるべき
    edges = [k for k in groups if k.lower().startswith(LINE_PREFIXES)]
    if len(edges) == 2 and any(len(groups[e]) < 2 for e in edges):
        print("\n道幅は計算できない: 縁ラベルの点が 1 個しかない。"
              "各縁を手前から遠方まで 6〜8 点クリックすること")
    if len(edges) == 2 and all(len(groups[e]) >= 2 for e in edges):
        a, b = (apply_h(H, groups[e]) for e in edges)
        # 2 本の縁で囲まれた領域を舗装面として塗る。図が「道」に見えるようにするため
        a_s, b_s = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
        poly = np.vstack([a_s, b_s[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color="0.75", alpha=0.5, zorder=0,
                label="walkway")
        # a の各点から b の直線までの距離を幅とみなす
        d = b[-1] - b[0]
        d = d / np.linalg.norm(d)
        nvec = np.array([-d[1], d[0]])
        w = np.abs((a - b[0]) @ nvec)
        print(f"\n道幅 {w.mean():.3f} ± {w.std():.3f} m  (min {w.min():.3f} / max {w.max():.3f})")
        print("  → 幅が距離によらず一定なら、ホモグラフィは正しく効いている")

    if args.det:
        rows = list(csv.DictReader(open(args.det, newline="", encoding="utf-8")))
        d = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
        ax.scatter(d[:, 0], d[:, 1], c="red", marker="*", s=120, zorder=6,
                   label=f"detection ({len(d)}f)")
        print(f"\n検出: ({d[:,0].mean():.2f}, {d[:,1].mean():.2f}) m")

    ax.set_xlabel("X [m]  (along walkway)")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"bird's-eye view   red x = uncertainty > {TARGET_M} m")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
