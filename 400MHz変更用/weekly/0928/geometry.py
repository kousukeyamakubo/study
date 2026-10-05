# 走行ルートの幾何コア。
#
# レーダー基準の座標系で、足元の地上座標 (X, Y) から斜距離 R・俯角・径方向速度 v_r を出す。
# 0928/README.md の O1〜O3 はすべてここの呼び出し方が違うだけ。
#
# 座標系（README「座標系・区間」と同じ）:
#   原点 … レーダー位置を道路の手前側境界に投影した点
#   X    … 道路の延長方向。レーダーから遠ざかる向きが正
#   Y    … 道路の横断方向。奥が正（走行ラインは Y = 0.5, 1.5, 2.5, 3.5 m）
#   レーダーの設置座標は (0, -y_r, h)。y_r は道路手前側境界からの引き（セットバック）
#
# 注意: ここで返す R は「足元」基準。実際の反射中心は地上 z_ref ≈ 1 m にあるため、
# 実測と重ねると系統的にずれる。その扱いは 9/24 の議論どおり後回し（README 参照）。

import numpy as np

# レーダー諸元。すべて 0727/atlas_dat_parse.py がヘッダから読んだ実測値
RANGE_RES_M = 0.846      # 真の距離分解能 c/2B（B = 177.2 MHz）。ヘッダの range_res は表示刻みなので使わない
VEL_RES_MS = 0.5995      # ドップラー分解能。速度軸は 16 ビンしかない
MAX_VEL_MS = 4.7957      # 折り返し限界
MAX_RANGE_M = 53.955
FRAME_PERIOD_S = 0.2


def lateral_offset(y_line, y_r, h):
    """走行ラインまでの横方向離隔 d = sqrt((Y + y_r)^2 + h^2)。

    R は X と d だけで決まるので、ライン識別の難しさは実質この d の差に集約される。
    h が大きいほど平方根の中で h^2 が支配し、Y の差が d に伝わらなくなる。
    """
    return np.hypot(np.asarray(y_line, dtype=float) + y_r, h)


def slant_range(x, y_line, y_r, h):
    """足元 (x, y_line) までの斜距離 R [m]"""
    return np.hypot(np.asarray(x, dtype=float), lateral_offset(y_line, y_r, h))


def depression_deg(x, y_line, y_r, h):
    """俯角 [deg]。垂直ビーム幅の外に出ていないかの確認に使う（近距離端の決定）"""
    d = lateral_offset(y_line, y_r, h)
    horiz = np.sqrt(np.maximum(d**2 - h**2, 0.0) + np.asarray(x, dtype=float) ** 2)
    return np.degrees(np.arctan2(h, horiz))


def radial_velocity(v, x, y_line, y_r, h):
    """径方向速度 v_r = v * X / R [m/s]。正が離反。

    道路に沿って（X 方向に）速さ v で動く場合。v_r = dR/dt そのものなので、
    ドップラーは距離−時間軌跡と同等の幾何情報しか持たない（README 参照）。
    """
    r = slant_range(x, y_line, y_r, h)
    return np.asarray(v, dtype=float) * np.asarray(x, dtype=float) / r


def trajectory(t, cone_x, cone_t, y_line, y_r, h):
    """コーン通過時刻から理論軌道 R(t), v_r(t) を作る（O1）。

    区間ごとに線形内挿する。20 m を始点・終点の 1 区間で内分すると速度変動 ±10% で
    中間が 1 m 近くずれ、見たい反射位置のずれと同オーダーになって埋もれるため、
    コーン 5 本ぶんの区間に切って内挿する。

    t       : 評価したい時刻 [s]        (T,)
    cone_x  : コーンの X 座標 [m]        (K,)
    cone_t  : その通過時刻 [s]           (K,)
    戻り値  : (R(t), v_r(t), X(t))       各 (T,)
    """
    t = np.asarray(t, dtype=float)
    cone_x = np.asarray(cone_x, dtype=float)
    cone_t = np.asarray(cone_t, dtype=float)
    order = np.argsort(cone_t)
    cone_t, cone_x = cone_t[order], cone_x[order]

    x = np.interp(t, cone_t, cone_x)                       # (T,) 区間ごとの等速直線運動
    v = np.gradient(cone_x, cone_t)                        # (K,) 区間速度
    v_t = np.interp(t, cone_t, v)                          # (T,)

    r = slant_range(x, y_line, y_r, h)                     # (T,)
    v_r = v_t * x / r                                      # (T,)
    return r, v_r, x
