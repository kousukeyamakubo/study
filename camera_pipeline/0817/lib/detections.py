# 物体検出の結果を扱う。検出器（YOLO）には依存しないので、映像が無くても検証できる。
#
# 【役割分担】
#   detect_yolo.py  映像 → 検出結果 CSV（ultralytics に依存）
#   detections.py   検出結果 → 接地点 → 地上座標（numpy/pandas のみ）  ← 本ファイル
#   homography.py   (u,v) → (X,Y) の変換
#
# 【接地点の取り方が本ファイルの中心】
# ラベルは「地上座標 + クラス」なので、bbox から地面に接している1点を決める必要がある。
# 0725/camera_radar_labeling_plan.md Step 4 は「下辺中央」としているが、
# cyclist ではこれをそのまま使うと誤る（下記）。

from __future__ import annotations

import numpy as np
import pandas as pd

# COCO のクラス ID（YOLO の既定の学習セット）
COCO_PERSON, COCO_BICYCLE, COCO_MOTORCYCLE = 0, 1, 3
COCO_CAR, COCO_BUS, COCO_TRUCK = 2, 5, 7

# 本研究の対象クラス。RD マップ側の {背景=0, cyclist=1, vehicle=2} に対応させる
# pedestrian は 9/8 に向けて追加（実測での識別対象が vehicle → cyclist/pedestrian に変更されたため）
CLS_CYCLIST, CLS_VEHICLE, CLS_PEDESTRIAN = "cyclist", "vehicle", "pedestrian"
VEHICLE_COCO = {COCO_CAR, COCO_BUS, COCO_TRUCK}

COLUMNS = ["frame", "t_s", "track_id", "cls", "conf",
           "x1", "y1", "x2", "y2", "foot_u", "foot_v"]


# --------------------------------------------------------------------------
# 幾何の小道具
# --------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """2つの bbox [x1,y1,x2,y2] の IoU"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(ua, 1e-9))


def foot_of(box: np.ndarray) -> tuple[float, float]:
    """bbox の下辺中央。地面と接している点の推定値"""
    return float((box[0] + box[2]) / 2), float(box[3])


# --------------------------------------------------------------------------
# cyclist の統合
# --------------------------------------------------------------------------

def _dedup_by_conf(rows: pd.DataFrame, iou_thr: float) -> pd.DataFrame:
    """信頼度が高い順に見て、既に採用した箱と IoU が閾値を超えるものを間引く（NMS）。

    bicycle/motorcycle は低信頼度だと YOLO が両方の候補を出すことがあるが、
    ultralytics の NMS はクラスごとに独立して行われるため、別クラス間の重複は
    自動では消えない。ここで同一物体の二重検出を除く。
    """
    if len(rows) <= 1:
        return rows
    rows = rows.sort_values("conf", ascending=False)
    kept_idx, kept_boxes = [], []
    for idx, r in rows.iterrows():
        box = r[["x1", "y1", "x2", "y2"]].to_numpy(float)
        if all(_iou(box, kb) <= iou_thr for kb in kept_boxes):
            kept_idx.append(idx)
            kept_boxes.append(box)
    return rows.loc[kept_idx]


def merge_cyclist(df: pd.DataFrame, iou_thr: float = 0.15, dup_iou_thr: float = 0.5) -> pd.DataFrame:
    """COCO の person + bicycle を1つの cyclist にまとめ、person 単独は pedestrian とする。

    【なぜ必要か】
    YOLO は自転車に乗った人を person と bicycle の2つの箱で出す。
    そのままだと1つの目標が2つのラベルになり、レーダー側の1目標と対応しない。

    【接地点は自転車の下辺を使う】
    person の下辺は「ペダル上の足」であって地面ではない。地面に接しているのは
    自転車のタイヤなので、統合後の接地点には **bicycle の下辺中央** を採る。
    これを person の下辺で代用すると、俯角設置では地上距離が数十 cm ずれる。

    【対応が取れなかった場合】
      - bicycle のみ（乗り手を見逃し）→ cyclist とする。接地点はそのまま
      - person のみ → **pedestrian とする**（以前は誤ラベル回避のため捨てていたが、
        識別対象が vehicle → cyclist/pedestrian に変わったため、クラスを立てて拾う）

    【bicycle/motorcycle の二重検出】
    低信頼度のフレームでは同一の自転車が bicycle と motorcycle の両方として
    検出されることがある（実測 0816 データで確認）。dup_iou_thr で重なりの強い
    ものを間引く。9/8 計画の「近接した複数目標の分離」を壊さないよう、person-bike
    の対応付け閾値（iou_thr=0.15）より高い値（既定 0.5）にして、本当に別々の
    自転車まで誤統合しないようにしている。
    """
    out = []
    for frame, g in df.groupby("frame", sort=True):
        persons = g[g["cls"] == COCO_PERSON]
        bikes = _dedup_by_conf(g[g["cls"].isin([COCO_BICYCLE, COCO_MOTORCYCLE])], dup_iou_thr)
        used_p = set()

        for _, b in bikes.iterrows():
            bb = b[["x1", "y1", "x2", "y2"]].to_numpy(float)
            # 最も重なる person を相方にする
            best, best_iou = None, iou_thr
            for pi, p in persons.iterrows():
                if pi in used_p:
                    continue
                v = _iou(bb, p[["x1", "y1", "x2", "y2"]].to_numpy(float))
                if v > best_iou:
                    best, best_iou = pi, v
            box = bb.copy()
            conf = float(b["conf"])
            if best is not None:
                used_p.add(best)
                p = persons.loc[best]
                pb = p[["x1", "y1", "x2", "y2"]].to_numpy(float)
                box = np.array([min(bb[0], pb[0]), min(bb[1], pb[1]),
                                max(bb[2], pb[2]), max(bb[3], pb[3])])
                conf = float(min(conf, p["conf"]))
            fu, fv = foot_of(bb)                     # ← 接地点は自転車の下辺
            out.append(dict(frame=frame, t_s=b["t_s"], track_id=b["track_id"],
                            cls=CLS_CYCLIST, conf=conf,
                            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                            foot_u=fu, foot_v=fv))

        for pi, p in persons.iterrows():              # 自転車と対応が取れなかった person は pedestrian
            if pi in used_p:
                continue
            pb = p[["x1", "y1", "x2", "y2"]].to_numpy(float)
            fu, fv = foot_of(pb)
            out.append(dict(frame=frame, t_s=p["t_s"], track_id=p["track_id"],
                            cls=CLS_PEDESTRIAN, conf=float(p["conf"]),
                            x1=pb[0], y1=pb[1], x2=pb[2], y2=pb[3],
                            foot_u=fu, foot_v=fv))

        for _, v in g[g["cls"].isin(VEHICLE_COCO)].iterrows():
            vb = v[["x1", "y1", "x2", "y2"]].to_numpy(float)
            fu, fv = foot_of(vb)
            out.append(dict(frame=frame, t_s=v["t_s"], track_id=v["track_id"],
                            cls=CLS_VEHICLE, conf=float(v["conf"]),
                            x1=vb[0], y1=vb[1], x2=vb[2], y2=vb[3],
                            foot_u=fu, foot_v=fv))

    return pd.DataFrame(out, columns=COLUMNS) if out else pd.DataFrame(columns=COLUMNS)


# --------------------------------------------------------------------------
# 地上座標へ
# --------------------------------------------------------------------------

def to_ground(df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """接地点 (foot_u, foot_v) を地上座標 (X, Y) に変換して列を足す"""
    from homography import apply_h                    # 循環 import を避けて遅延

    d = df.copy()
    if len(d) == 0:
        d["X"], d["Y"] = [], []
        return d
    xy = apply_h(H, d[["foot_u", "foot_v"]].to_numpy(float))
    d["X"], d["Y"] = xy[:, 0], xy[:, 1]
    return d


def add_slant_range(df: pd.DataFrame, radar_xy, h: float) -> pd.DataFrame:
    """斜距離の列を足す。**学習データを組み立てるときだけ**呼ぶこと。

    ラベルの正本は地上座標 (X,Y) にしておく。h の推定値が変わっても
    ラベルを作り直さずに再計算できるようにするため"""
    from homography import ground_to_slant

    d = df.copy()
    d["R_slant"] = ground_to_slant(d[["X", "Y"]].to_numpy(float), radar_xy, h)
    return d


# --------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------

def save(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)


def load(path) -> pd.DataFrame:
    return pd.read_csv(path)
