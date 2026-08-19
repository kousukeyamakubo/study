# 画像座標 (u,v) → 地上座標 (X,Y) の変換。カメラ由来ラベル生成の中核。
#
# 【役割】実装のみ。検証・精度評価は check_homography.py に置く。
#
# 【なぜホモグラフィで足りるか】
# 目標は地面（平面）の上を動く。平面上の点と画像の間は射影変換1枚で厳密に対応するので、
# カメラの内部パラメータや姿勢を知らなくても、4点以上の対応があれば (u,v)→(X,Y) が引ける。
# ラベル生成に必要なのはこの写像だけであり、設置高 h も俯角も要らない。
#
# 【地上座標系は自分で定義してよい】
# 必要なのは「マーカー同士の位置関係」だけで、カメラからの距離ではない。
# マーカー1を原点、1→2の向きを X 軸と決めれば、相互距離から全点の (X,Y) が決まる。
# → 5階から地上への距離を測る必要が無い（レーザーが届かなくても成立する）。
#
# 【内部パラメータが要る場面】
# (1) レンズ歪みの補正、(2) ホモグラフィから h・俯角を取り出す分解。
# いずれもラベル生成の本筋には不要で、精度改善と幾何の把握のための追加要素。
#
# 依存: numpy のみ（OpenCV は使わない）

import numpy as np


# --------------------------------------------------------------------------
# 推定
# --------------------------------------------------------------------------

def _normalize(pts: np.ndarray):
    """Hartley 正規化。重心を原点に、原点からの平均距離を sqrt(2) に揃える。

    DLT は座標のスケールに敏感で、画素値（〜10^3）と実距離（〜10^1）をそのまま
    混ぜると設計行列の条件数が悪化して解が暴れる。前処理として必須"""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    return (pts - c) * s, T


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src(N,2) → dst(N,2) の射影変換 H(3x3) を DLT で最小二乗推定する。

    N>=4。4点なら厳密解、5点以上なら残差最小の解になる（誤差が平均化されるので
    実運用では 6〜8 点を推奨）"""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError(f"4点以上必要（与えられたのは {len(src)} 点）")

    sn, Ts = _normalize(src)
    dn, Td = _normalize(dst)

    # 各対応から 2 本の線形方程式。h を 9 次元ベクトルとして A h = 0 を解く
    A = np.zeros((2 * len(sn), 9))
    for i, ((x, y), (X, Y)) in enumerate(zip(sn, dn)):
        A[2 * i] = [-x, -y, -1, 0, 0, 0, X * x, X * y, X]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]

    # 最小特異値に対応する右特異ベクトルが |Ah| 最小の解
    Hn = np.linalg.svd(A)[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts                    # 正規化を戻す
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) に H を適用する。同次座標での割り算を含む"""
    pts = np.atleast_2d(np.asarray(pts, float))
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """各対応点の残差[dst の単位]。マーカーの測り間違いはここに大きく出るので、
    残差を見て異常な点を弾くのが実運用での使い方"""
    return np.linalg.norm(apply_h(H, src) - np.asarray(dst, float), axis=1)


# --------------------------------------------------------------------------
# ラベル生成
# --------------------------------------------------------------------------

def bbox_to_ground(H: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """YOLO の bbox (N,4)=[x1,y1,x2,y2] → 地上座標 (N,2)。

    接地点として **下辺中央** を使う（0725/camera_radar_labeling_plan.md Step 4）。
    bbox の縦方向は物体の高さであり、レーダーは高さを分離しないので対応物が無い。
    地面と接している下辺だけが平面上の点として扱える"""
    b = np.atleast_2d(np.asarray(bboxes, float))
    foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
    return apply_h(H, foot)


def ground_to_slant(xy: np.ndarray, radar_xy: np.ndarray, h: float) -> np.ndarray:
    """地上座標 → レーダーから見た斜距離[m]。学習データ組み立て時にのみ使う。

    ラベルの正本は地上座標にしておく（h の推定値が変わっても作り直さずに済む）。
    斜距離はレーダーの「位置」だけで決まり「向き」には依存しないので、
    カメラとレーダーのボアサイトを精密に合わせる必要は無い"""
    xy = np.atleast_2d(np.asarray(xy, float))
    return np.sqrt(((xy - np.asarray(radar_xy, float)) ** 2).sum(axis=1) + h ** 2)


# --------------------------------------------------------------------------
# 内部パラメータがある場合の分解（h・俯角を得る）
# --------------------------------------------------------------------------

def pose_from_homography(H_img2ground: np.ndarray, K: np.ndarray):
    """地上→画像のホモグラフィを分解し、カメラの設置高 h と俯角を返す。

    平面 Z=0 に対して H_ground2img ∝ K [r1 r2 t] が成り立つことを使う。
    r1, r2 は回転行列の第1・2列なのでノルムが 1 という条件からスケールが決まる。

    戻り値: (h[m], 俯角[deg], R(3x3), C(カメラ中心, 世界座標))"""
    A = np.linalg.inv(K) @ np.linalg.inv(H_img2ground)      # ∝ [r1 r2 t]
    # r1, r2 のノルムは本来等しいので平均を取る（雑音への耐性）
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if A[2, 2] * lam < 0:                                   # 被写体がカメラ前方に来る符号を選ぶ
        lam = -lam
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]

    # 雑音で r1,r2 が直交しなくなるので、最も近い正規直交基底に落とす
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))
    R = U @ Vt

    C = -R.T @ t                                            # 世界座標でのカメラ中心
    fwd = R.T @ np.array([0.0, 0.0, 1.0])                   # 光軸の向き（世界座標）
    depression = np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1)))
    return float(C[2]), float(depression), R, C
