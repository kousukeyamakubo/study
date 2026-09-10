# カメラ検出とレーダー検出（0908/postprocess/detect.py の出力）を1フレームごとに対応付ける
#
# 【背景】meeting/2026-09-08.md S3。レンジだけでは同一レンジビンの複数目標
# （9/8シナリオD/Eが本命に据えている「分離」のケース）を区別できないため、
# SNRが十分なフレームに限り角度ゲートを併用する。0727実測で低SNR時の角度標準偏差が
# 20.8°まで悪化した（atlas_dat_format.md §6.5）ため、無条件には使えない。
#
# 対応が一意に決まらない場合は「対応なし」を返す。誤対応（クラスの取り違え）は
# 無対応より有害という前提（S6の lone_person_as_cyclist=False と同じ考え方）。
#
# 【未検証】実測の同期データがまだ無いため、ゲート幅（RANGE_GATE_M）は暫定値。
# 本番データが揃ったら add_slant_range() の誤差実測から調整すること。

from __future__ import annotations

from dataclasses import dataclass

RANGE_GATE_M = 1.0     # 暫定値。実測のレンジ誤差が分かり次第調整する
ANGLE_GATE_DEG = 20.0  # DBFの主ローブ幅（Azimuth分解能14.5°と整合。atlas_dat_format.md §6.5）
SNR_GATE_DB = 30.0     # これ以上で角度std1桁台、未満は20.8°まで悪化し使えない（同§6.5）


@dataclass
class CamDetection:
    track_id: int
    cls: str
    range_m: float     # add_slant_range() で計算済みの斜距離
    angle_deg: float   # カメラの地上座標から見た、レーダー正面を0とする方位角


@dataclass
class RadarPeak:
    range_m: float
    vel_ms: float
    snr_db: float
    angle_deg: float | None = None  # DBFを掛けた場合のみ値が入る。掛けなければ None


@dataclass
class Match:
    cam: CamDetection
    peak: RadarPeak | None   # 対応するレーダー検出。無ければ None（未検出・遮蔽・対応不明）
    used_angle_gate: bool    # 角度ゲートで絞り込んだか（レンジだけで一意なら False）


def _angle_diff(a: float, b: float) -> float:
    """角度差を [-180, 180) に正規化する"""
    return (a - b + 180) % 360 - 180


def associate_frame(cam_dets: list[CamDetection], peaks: list[RadarPeak]) -> list[Match]:
    """1フレーム分の対応付け。

    手順: まずレンジゲートで候補を絞る。候補が複数残り、かつ候補のSNRが十分なら
    角度ゲートでさらに絞る。それでも一意に決まらなければ「対応なし」とする
    （無理に1つを選んで誤対応を作らない）。
    """
    results = []
    used_peaks = set()

    for cam in cam_dets:
        cands = [i for i, p in enumerate(peaks)
                 if i not in used_peaks and abs(p.range_m - cam.range_m) <= RANGE_GATE_M]

        used_angle_gate = False
        if len(cands) > 1:
            gated = [i for i in cands
                     if peaks[i].snr_db >= SNR_GATE_DB and peaks[i].angle_deg is not None
                     and abs(_angle_diff(peaks[i].angle_deg, cam.angle_deg)) <= ANGLE_GATE_DEG]
            if gated:
                cands = gated
                used_angle_gate = True

        if len(cands) == 1:
            idx = cands[0]
            used_peaks.add(idx)
            results.append(Match(cam, peaks[idx], used_angle_gate))
        else:
            # 候補0件（未検出）、または複数のまま絞れなかった（対応不明）場合は対応なし
            results.append(Match(cam, None, used_angle_gate))

    return results
