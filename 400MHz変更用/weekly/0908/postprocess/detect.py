# 実測 RD テンソルに対する素朴な検出器（ピーク抽出 + NMS）。モデル非依存。
#
# 【なぜ今書けるか】
# 9/8 時点で角度情報の渡し方（DBF で潰すか生の位相のまま渡すか）は未決定。
# この検出器は角度を一切使わず、7 素子（virt_array）をインコヒーレントに合成した
# パワーだけで動く。DBF を先取りしないので、角度の判断が決まる前でも着手できる。
#
# 【役割分担】
#   detect.py   RD テンソル → (Doppler,Range) パワーマップ → CA-CFAR → NMS  ← 本ファイル
#   track.py    フレームごとのピーク列 → 時間方向のトラック
#
# 入力は atlas_export.py が吐く .npz の "rd" キー
# （axes: "rd: (Frame, TX, RX, Doppler, Range) complex64"）を想定する。

from __future__ import annotations

import numpy as np


def incoherent_power(rd: np.ndarray) -> np.ndarray:
    """(Frame, TX, RX, Doppler, Range) の複素RDから、素子間の位相を捨てた
    パワーマップ (Frame, Doppler, Range) を作る。位相（角度情報）を使わないのは、
    DBF を先取りして角度の判断材料を汚さないため。
    """
    return np.sum(np.abs(rd) ** 2, axis=(1, 2)).astype(np.float32)  # (F,TX,RX,D,R) -> (F,D,R)


def _box_mean(power: np.ndarray, half_d: int, half_r: int) -> np.ndarray:
    """各セルを中心とする (2*half_d+1) x (2*half_r+1) 窓の平均。最後の2軸 (D,R) に対して適用する。

    Doppler は FFT ビンなので折り返す（wrap）。Range は折り返らないので端を延長する（edge）。
    積分画像（累積和）を使い、窓サイズによらず一定コストで平均を求める。
    """
    pad_d = [(0, 0)] * (power.ndim - 2) + [(half_d, half_d), (0, 0)]
    p = np.pad(power, pad_d, mode="wrap")
    pad_r = [(0, 0)] * (power.ndim - 2) + [(0, 0), (half_r, half_r)]
    p = np.pad(p, pad_r, mode="edge")

    pad_ii = [(0, 0)] * (power.ndim - 2) + [(1, 0), (1, 0)]
    ii = np.cumsum(np.cumsum(np.pad(p, pad_ii), axis=-2), axis=-1)  # 積分画像

    D, R = power.shape[-2:]
    kd, kr = 2 * half_d + 1, 2 * half_r + 1
    window_sum = (ii[..., kd:kd + D, kr:kr + R] - ii[..., 0:D, kr:kr + R]
                  - ii[..., kd:kd + D, 0:R] + ii[..., 0:D, 0:R])
    return window_sum / (kd * kr)


def cfar_ca(power: np.ndarray, n_train: tuple[int, int], n_guard: tuple[int, int],
            pfa: float) -> np.ndarray:
    """CA-CFAR（Cell-Averaging CFAR）。周囲のトレーニングセル平均から
    しきい値を立て、それを超えたセルを検出とする真偽値マスクを返す。

    n_train, n_guard は (Doppler方向, Range方向) の片側セル数。
    power は (..., Doppler, Range) — フレーム軸をまとめて渡してよい。
    """
    td, tr = n_train
    gd, gr = n_guard
    full_mean = _box_mean(power, td + gd, tr + gr)
    guard_mean = _box_mean(power, gd, gr)
    n_full = (2 * (td + gd) + 1) * (2 * (tr + gr) + 1)
    n_guard_cells = (2 * gd + 1) * (2 * gr + 1)
    n_train_cells = n_full - n_guard_cells
    train_mean = (full_mean * n_full - guard_mean * n_guard_cells) / n_train_cells

    # CA-CFAR の標準しきい値係数（指数分布ノイズを仮定した false alarm rate から導出）
    alpha = n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0)
    return power > train_mean * alpha


def nms_2d(power_frame: np.ndarray, mask_frame: np.ndarray,
           sep_d: int, sep_r: int) -> np.ndarray:
    """CFAR マスクは主ローブの広がりで1目標が複数セルにまたがるので、
    パワーが強い順に採用し、既に採ったピークの近傍（sep_d, sep_r 以内）を間引く。
    Doppler は循環軸として距離を測る。

    戻り値: (N, 2) の [doppler_idx, range_idx] 配列
    """
    n_doppler = power_frame.shape[0]
    locs = np.argwhere(mask_frame)
    if len(locs) == 0:
        return np.empty((0, 2), dtype=int)

    scores = power_frame[locs[:, 0], locs[:, 1]]
    order = np.argsort(-scores)
    locs = locs[order]

    kept: list[tuple[int, int]] = []
    for d, r in locs:
        d, r = int(d), int(r)
        collide = False
        for kd, kr in kept:
            dd = min(abs(d - kd), n_doppler - abs(d - kd))
            if dd <= sep_d and abs(r - kr) <= sep_r:
                collide = True
                break
        if not collide:
            kept.append((d, r))
    return np.array(kept, dtype=int)


def detect_peaks(rd: np.ndarray, n_train=(2, 5), n_guard=(1, 2), pfa=1e-3,
                  sep_d=1, sep_r=2) -> list[np.ndarray]:
    """rd (Frame,TX,RX,Doppler,Range) から、フレームごとの検出ピーク [doppler_idx, range_idx] を返す。

    しきい値・NMS の窓サイズは experiments/eval_cfar の sim 向けパラメータを実測の
    ビン数（Doppler=16, Range=129）に合わせて縮小した初期値。実測で見ながら調整する。
    """
    power = incoherent_power(rd)                 # (F, D, R)
    mask = cfar_ca(power, n_train, n_guard, pfa)  # (F, D, R)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return [nms_2d(power_db[f], mask[f], sep_d, sep_r) for f in range(power.shape[0])]
