# 現地実測（既知位置の協力者等）との突き合わせで、パイプライン全体の精度をRMSEで出す。
#
# 【真値の作り方】
# pick_markers.py --xy で使ったのと同じ2本のアンカー（動かしていないこと）を使い、
# 評価対象（静止した協力者等）の足元位置までの距離2本を測って trilaterate で (X,Y) を出す。
# 対象が写ったフレームに対して pick_markers.py --xy を実行し、1点だけ拾えば済む
# （アンカーA・Bも同じ2点をクリックして距離を入れ直す。座標は同じ値になるはず）。
# これを name・track_id 付きでまとめたものが --gt。
#
# 【track_id で紐付ける理由】
# 評価対象は静止しているので、そのtrackの全フレームが同じ真値と比較できる。
# フレーム数だけサンプルが増える上、検出のフレーム間ジッタもここで見える
# （`docs/detection.md` の track_id=-1 の扱いに注意。-1 のフレームは対象から除く）。
#
# 依存: numpy, pandas
#
# 使い方:
#   python eval_ground_truth.py gt.csv video.ground.csv
#   python eval_ground_truth.py gt.csv video.ground.csv --out errors.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANGE_BIN = 0.8463541666666666      # ATLAS の距離分解能[m]（check_homography.py と同じ）
TARGET_ACC = RANGE_BIN / 3          # 目標精度[m]
# 距離帯の区切りはこのフォルダの想定運用（20-50m）に合わせた仮の目安。
# 実際のマーカー配置に応じて変えてよい
BINS = [(0, 25, "near"), (25, 40, "mid"), (40, float("inf"), "far")]


def evaluate(gt: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """真値(gt)ごとに、対応するtrack_idの全フレームの検出と突き合わせて誤差を出す"""
    rows = []
    for _, g in gt.iterrows():
        sub = det[det["track_id"] == g["track_id"]]
        if len(sub) == 0:
            print(f"  ⚠ track_id={g['track_id']}（{g['name']}）が検出CSVに無い。"
                  "track_idの取り違えか、その対象が検出されていない")
            continue
        for _, d in sub.iterrows():
            rows.append(dict(
                name=g["name"], frame=d["frame"],
                X_true=g["X_true"], Y_true=g["Y_true"], X=d["X"], Y=d["Y"],
                ex=float(d["X"] - g["X_true"]), ey=float(d["Y"] - g["Y_true"]),
                err=float(np.hypot(d["X"] - g["X_true"], d["Y"] - g["Y_true"])),
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path, help="name,track_id,X_true,Y_true の真値CSV")
    ap.add_argument("det", type=Path,
                    help="to_ground() 済みの検出CSV（frame,track_id,X,Y 列を含む）")
    ap.add_argument("--out", type=Path, default=None, help="点ごとの誤差を保存するCSV")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    det = pd.read_csv(args.det)
    r = evaluate(gt, det)
    if len(r) == 0:
        print("突き合わせできる点が無い。track_id を確認すること")
        return

    rmse = float(np.sqrt((r["err"] ** 2).mean()))
    bias = r[["ex", "ey"]].mean().to_numpy()
    print(f"点数 {len(r)}（対象 {r['name'].nunique()} 個、目標精度 {TARGET_ACC:.2f} m）")
    print(f"RMSE 全体: {rmse:.3f} m {'（目標内）' if rmse < TARGET_ACC else '（目標超過）'}")
    # bias が大きいと「ランダムな雑音」でなく「系統的なずれ」（docs/detection.md の
    # bbox下辺=手前端など）が支配的という意味になる
    print(f"平均符号付き誤差(bias): X {bias[0]:+.3f} m / Y {bias[1]:+.3f} m "
          f"（|bias|={np.hypot(*bias):.3f} m）")

    print("\n距離帯別（Y_true基準）:")
    for lo, hi, label in BINS:
        sub = r[(r["Y_true"] >= lo) & (r["Y_true"] < hi)]
        if len(sub) == 0:
            continue
        rmse_b = float(np.sqrt((sub["err"] ** 2).mean()))
        hi_s = f"{hi:.0f}" if np.isfinite(hi) else "∞"
        print(f"  {label:>4} ({lo:.0f}-{hi_s}m): RMSE {rmse_b:.3f} m  n={len(sub)}")

    print("\n対象別:")
    for name, sub in r.groupby("name"):
        rmse_n = float(np.sqrt((sub["err"] ** 2).mean()))
        print(f"  {name:>10}: RMSE {rmse_n:.3f} m  "
              f"真値=({sub['X_true'].iloc[0]:.2f},{sub['Y_true'].iloc[0]:.2f}) m  n={len(sub)}")

    if args.out:
        r.to_csv(args.out, index=False)
        print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
