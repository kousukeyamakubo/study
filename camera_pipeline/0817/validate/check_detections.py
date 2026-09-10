# detections.py の検証。合成 bbox を使うので YOLO も映像も要らない。
#
# 主目的は cyclist 統合の動作確認ではなく、**接地点の取り方が地上座標にどれだけ効くか**の
# 定量化。画素雑音（check_homography.py で 0.13 m）より遥かに大きいことを確かめる。
#
# 使い方:
#   python check_detections.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from check_homography import DEPRESSION, H_CAM, grid_markers, make_camera  # noqa: E402
from detections import (CLS_CYCLIST, COCO_BICYCLE, COCO_PERSON,           # noqa: E402
                        merge_cyclist, to_ground)
from homography import estimate_homography                                 # noqa: E402

RANGE_BIN = 0.8463541666666666
TARGET_ACC = RANGE_BIN / 3

# 自転車に乗った人の寸法[m]（幅, 長さ, 高さ範囲）
BIKE = dict(w=0.6, l=1.8, z0=0.0, z1=1.1)
RIDER = dict(w=0.5, l=0.6, z0=0.8, z1=1.9)
CAR = dict(w=1.8, l=4.5, z0=0.0, z1=1.5)


def box_to_bbox(project, cx, cy, spec) -> np.ndarray:
    """地上位置 (cx,cy) に置いた直方体を画像に投影し、外接矩形を返す。
    YOLO が出す bbox は物体の外接矩形なので、それを模している"""
    xs = [cx - spec["w"] / 2, cx + spec["w"] / 2]
    ys = [cy - spec["l"] / 2, cy + spec["l"] / 2]
    zs = [spec["z0"], spec["z1"]]
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                pts.append([x, y, z])
    uv = project_3d(project, np.array(pts))
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def project_3d(project, pts3):
    """地上投影関数を高さ付きに拡張する。高さ z の点は、地面上で
    Y' = Y*h/(h-z) の位置にある点と画像上で同じところに写る（相似）"""
    out = []
    for x, y, z in pts3:
        s = H_CAM / (H_CAM - z)
        out.append(project([[x * s, y * s]])[0])
    return np.array(out)


def main():
    K, project = make_camera()
    markers = grid_markers(20, 50, 8.0, n_y=3, n_x=2)
    H = estimate_homography(project(markers), markers)     # 雑音なしの理想変換

    print("=" * 76)
    print("接地点の取り方が地上座標に与える誤差")
    print("=" * 76)

    # --- 0. そもそも検出できる大きさか ---
    # 映像の仕様で最も効くのはここ。画角が広い（＝焦点距離が短い）と遠方の目標が
    # 数画素になり、YOLO が原理的に検出できない
    print(f"\n[0] 目標が画像上で何画素になるか（水平画角 60°, 1920x1080 の場合）")
    print(f"  {'Y[m]':>6} {'cyclist':>14} {'vehicle':>14}   検出可否の目安")
    for y in (20, 30, 40, 50, 70):
        sizes = []
        for spec in (BIKE, CAR):
            bb = box_to_bbox(project, 0.0, float(y), spec)
            sizes.append((bb[2] - bb[0], bb[3] - bb[1]))
        ok = "○" if sizes[0][0] >= 20 else ("△ 限界" if sizes[0][0] >= 12 else "× 小さすぎ")
        print(f"  {y:6.0f} {sizes[0][0]:6.0f}x{sizes[0][1]:<6.0f} "
              f"{sizes[1][0]:6.0f}x{sizes[1][1]:<6.0f}   {ok}")
    print("  ※ YOLO が安定して検出できるのは概ね 20 px 以上。"
          "画角を狭める（望遠寄りにする）と遠方が稼げる")

    # --- 1. 高さ方向のずれがどれだけ増幅されるか ---
    print("\n[1] 接地点の高さが Δz ずれたときの地上距離の誤差")
    print("    （地面より Δz 高い点は、画像上では Y*h/(h-Δz) の地面と同じ場所に写る）")
    print(f"  {'Y[m]':>6}", end="")
    for dz in (0.1, 0.3, 0.8):
        print(f" {'Δz=' + str(dz) + 'm':>12}", end="")
    print("   増幅率")
    for y in (20, 30, 40, 50):
        print(f"  {y:6.0f}", end="")
        for dz in (0.1, 0.3, 0.8):
            print(f" {y * dz / (H_CAM - dz):10.2f} m", end="")
        print(f"   {y / H_CAM:5.2f}x")
    print(f"  ※ 増幅率は Y/h。40 m では **高さ 1 cm のずれが地上 2.7 cm** になる")
    print(f"  ※ 目標精度 {TARGET_ACC:.2f} m を 40 m で満たすには "
          f"接地点の高さを {TARGET_ACC*H_CAM/40:.2f} m 以内で当てる必要がある")

    # --- 2. rider の下辺を使ってしまうと ---
    print(f"\n[2] cyclist の接地点に person の下辺を使った場合の誤差")
    print(f"    person の bbox 下辺は「ペダル上の足」= 地上 {RIDER['z0']:.1f} m")
    for y in (20, 30, 40, 50):
        e = y * RIDER["z0"] / (H_CAM - RIDER["z0"])
        print(f"  Y={y:3.0f} m → {e:5.2f} m のずれ "
              f"（画素雑音 0.13 m の {e/0.13:.0f} 倍、レーダー分解能の {e/RANGE_BIN:.1f} 倍）")
    print("  → **bicycle の下辺を使うのが必須**。detections.merge_cyclist はそうしている")

    # --- 3. 統合の動作確認 ---
    print(f"\n[3] merge_cyclist の動作確認（Y=40 m に自転車＋乗り手）")
    bike_bb = box_to_bbox(project, 0.0, 40.0, BIKE)
    rider_bb = box_to_bbox(project, 0.0, 40.0, RIDER)
    raw = pd.DataFrame([
        dict(frame=0, t_s=0.0, track_id=1, cls=COCO_BICYCLE, conf=0.9,
             x1=bike_bb[0], y1=bike_bb[1], x2=bike_bb[2], y2=bike_bb[3],
             foot_u=0, foot_v=0),
        dict(frame=0, t_s=0.0, track_id=2, cls=COCO_PERSON, conf=0.9,
             x1=rider_bb[0], y1=rider_bb[1], x2=rider_bb[2], y2=rider_bb[3],
             foot_u=0, foot_v=0),
    ])
    m = merge_cyclist(raw)
    print(f"  入力 2 件 → 出力 {len(m)} 件, クラス {m['cls'].tolist()}")
    assert len(m) == 1 and m["cls"].iloc[0] == CLS_CYCLIST
    print(f"  接地点 v = {m['foot_v'].iloc[0]:.1f} px "
          f"(bicycle 下辺 {bike_bb[3]:.1f} / person 下辺 {rider_bb[3]:.1f})")
    assert abs(m["foot_v"].iloc[0] - bike_bb[3]) < 1e-6, "bicycle の下辺が選ばれていない"
    print("  → bicycle の下辺が採られている")

    # --- 4. 端から端まで通す ---
    print(f"\n[4] 通しの精度（真値: 物体の中心。合成 bbox → 統合 → 地上座標）")
    print(f"  {'クラス':>8} {'Y真値':>7} {'Y推定':>7} {'誤差':>7}   要因")
    for label, spec, ys in (("cyclist", BIKE, (20, 30, 40, 50)),
                            ("vehicle", CAR, (20, 30, 40, 50))):
        for y in ys:
            bb = box_to_bbox(project, 0.0, float(y), spec)
            df = pd.DataFrame([dict(frame=0, t_s=0.0, track_id=1,
                                    cls=COCO_BICYCLE if label == "cyclist" else 2,
                                    conf=0.9, x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3],
                                    foot_u=0, foot_v=0)])
            g = to_ground(merge_cyclist(df), H)
            est = float(g["Y"].iloc[0])
            print(f"  {label:>8} {y:6.1f} m {est:6.1f} m {est-y:+6.2f} m"
                  f"   物体長 {spec['l']:.1f} m の手前端を見ている")
    print(f"  ※ bbox の下辺は物体の**手前端**の接地線であり、中心ではない。")
    print(f"     ずれは概ね −(物体長/2) で、cyclist −0.9 m、vehicle −2.25 m")

    print("\n" + "=" * 76)
    print("結論")
    print("=" * 76)
    print("""
  誤差の大きさは次の順。画素雑音は最小で、無視してよい。

    1. 物体長による偏り（bbox 下辺 = 手前端）  cyclist 0.9 m / vehicle 2.2 m
    2. 接地点の高さの取り違え                   person の下辺を使うと 40 m で 2.2 m
    3. 画素雑音                                 40 m で 0.13 m

  1 は **向きが一定なら定数の偏り**なので、共通座標系でラベルを定義しておけば
  「モデルが学習する対象」に吸収できる（8/14 の議論のとおり）。ただし目標が横切ると
  見かけの長さが変わるため、完全な定数ではない。実測で確認すべき最初の量。
""")


if __name__ == "__main__":
    main()
