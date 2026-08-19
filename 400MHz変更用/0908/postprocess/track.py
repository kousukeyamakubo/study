# フレームごとの検出ピークを時間方向に安定させる（追跡）。
#
# 8/17 の決定「出力が点ならラベルは点、後処理はトラッキング」に対応する部品。
# 単純な greedy 最近傍対応付け（Kalman フィルタ等は使わない）。素朴な検出器
# （detect.py）に合わせて、まずは最小構成で成立させることを優先している。

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    # 各要素は (frame_idx, range_m, vel_ms)
    points: list[tuple[int, float, float]] = field(default_factory=list)


def greedy_associate(peaks_by_frame: list[np.ndarray], range_m: np.ndarray, vel_ms: np.ndarray,
                      max_range_step_m: float, max_vel_step_ms: float) -> list[Track]:
    """フレーム間でゲート内の最近傍と対応付け、対応が取れなければ新規トラックにする。

    フレーム周期は 0.2 s（atlas_export.py の meta.frame_period_s）と短く、対象の
    フレーム間移動量は小さいので、ゲートは物理量（m, m/s）で固定してよい。
    対応が取れなかったトラックはその場で打ち切る（見失い猶予は設けない）。
    """
    tracks: list[Track] = []
    active: dict[int, tuple[float, float]] = {}  # track_id -> (直近の range_m, vel_ms)
    next_id = 0

    for f, peaks in enumerate(peaks_by_frame):
        used = set()
        for d_idx, r_idx in peaks:
            r, v = float(range_m[r_idx]), float(vel_ms[d_idx])

            best_id, best_dist2 = None, None
            for tid, (pr, pv) in active.items():
                if tid in used:
                    continue
                if abs(r - pr) > max_range_step_m or abs(v - pv) > max_vel_step_ms:
                    continue
                dist2 = (r - pr) ** 2 + (v - pv) ** 2
                if best_dist2 is None or dist2 < best_dist2:
                    best_id, best_dist2 = tid, dist2

            if best_id is None:
                best_id = next_id
                next_id += 1
                tracks.append(Track(track_id=best_id))

            tracks[best_id].points.append((f, r, v))
            active[best_id] = (r, v)
            used.add(best_id)

        for tid in list(active):                 # 今フレームで対応が取れなかったトラックは打ち切り
            if tid not in used:
                del active[tid]

    return tracks
