# associate.py の動作確認。合成データを使うので実測の同期データが無くても検証できる。
#
# 目的: 以下の4パターンで、対応付けが「誤対応より無対応を選ぶ」設計通りに動くか確認する。
#   1) 単独目標: レンジだけで一意に決まる
#   2) 同一レンジ・角度違い・高SNR: 角度ゲートで正しく分離できる
#   3) 同一レンジ・角度違い・低SNR: 角度が信用できず、どちらも対応なしになる
#   4) 対応するレーダー検出が無い（遮蔽）: 対応なしになる
#
# 使い方:
#   python check_associate.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CamDetection, RadarPeak, associate_frame  # noqa: E402


def main():
    print("=" * 76)
    print("[1] 単独目標: レンジだけで一意に決まる")
    print("=" * 76)
    cams = [CamDetection(track_id=1, cls="cyclist", range_m=15.0, angle_deg=10.0)]
    peaks = [RadarPeak(range_m=15.1, vel_ms=2.0, snr_db=25.0)]  # 角度不明でも対応できるはず
    m = associate_frame(cams, peaks)[0]
    assert m.peak is not None and not m.used_angle_gate
    print(f"  対応: track1 -> range={m.peak.range_m}m（角度ゲート未使用）  OK")

    print("\n" + "=" * 76)
    print("[2] 同一レンジ・角度違い・高SNR: 角度ゲートで正しく分離できる")
    print("=" * 76)
    cams = [
        CamDetection(track_id=1, cls="cyclist", range_m=15.0, angle_deg=-15.0),
        CamDetection(track_id=2, cls="pedestrian", range_m=15.2, angle_deg=15.0),
    ]
    peaks = [
        RadarPeak(range_m=15.0, vel_ms=2.0, snr_db=35.0, angle_deg=-14.0),
        RadarPeak(range_m=15.3, vel_ms=0.5, snr_db=32.0, angle_deg=16.0),
    ]
    matches = associate_frame(cams, peaks)
    m1, m2 = matches
    assert m1.peak is not None and m2.peak is not None, "対応が付いていない"
    assert any(m.used_angle_gate for m in matches), \
        "角度ゲートが一度も使われていない（貪欲法の処理順で偶然分離できただけの可能性）"
    # 取り違えていないか（cyclist側の角度に近いピークに繋がっているか）を確認
    assert m1.peak.angle_deg == -14.0 and m2.peak.angle_deg == 16.0, \
        "クラスとピークの対応が入れ替わっている"
    print("  対応: track1(-15°)->peak(-14°), track2(+15°)->peak(+16°)  クラス取り違えなし  OK")

    print("\n" + "=" * 76)
    print("[3] 同一レンジ・角度違い・低SNR: 角度が信用できず対応なしになる")
    print("=" * 76)
    cams = [
        CamDetection(track_id=1, cls="cyclist", range_m=15.0, angle_deg=-15.0),
        CamDetection(track_id=2, cls="pedestrian", range_m=15.2, angle_deg=15.0),
    ]
    peaks = [
        RadarPeak(range_m=15.0, vel_ms=2.0, snr_db=18.0, angle_deg=-14.0),  # SNR不足
        RadarPeak(range_m=15.3, vel_ms=0.5, snr_db=20.0, angle_deg=16.0),   # SNR不足
    ]
    matches = associate_frame(cams, peaks)
    assert all(m.peak is None for m in matches), \
        "低SNRなのに対応付けてしまっている（誤対応のリスクを取っている）"
    print("  低SNRのため両方とも対応なし（誤対応より欠損を選んだ）  OK")

    print("\n" + "=" * 76)
    print("[4] 対応するレーダー検出が無い（遮蔽）: 対応なしになる")
    print("=" * 76)
    cams = [CamDetection(track_id=1, cls="cyclist", range_m=15.0, angle_deg=10.0)]
    peaks = [RadarPeak(range_m=25.0, vel_ms=1.0, snr_db=30.0)]  # レンジゲート外
    m = associate_frame(cams, peaks)[0]
    assert m.peak is None
    print("  レンジゲート外のため対応なし（遮蔽・未検出と解釈）  OK")

    print("\n全パターン OK")


if __name__ == "__main__":
    main()
