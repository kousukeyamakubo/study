# CEATEC 素材のスクリプトで共通に使う、走行の一覧・文言（日本語／英語）・フォント。

from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

# (番号, フォルダ, 時刻タグ, シナリオ, 向き)。シナリオごとに 接近 → 離反 の順にそろえる（撮った順とは違う）。
# シナリオはユーザーの申告（10/5、weekly/1013/1005/README.md §1）
RUNS = [
    (1, "pre", "135532", "bike", "app"),
    (2, "pre", "135606", "bike", "dep"),
    (3, "car", "142546", "car", "app"),
    (4, "car", "142246", "car", "dep"),
    (5, "car", "143104", "bike_parked", "app"),
    (6, "car", "143133", "bike_parked", "dep"),
]

LABELS = {
    "ja": dict(
        scen={"bike": "自転車", "car": "車", "bike_parked": "止めた車の横を自転車"},
        dir={"app": "接近", "dep": "離反"},
        kind={"nomti": "MTI なし", "mti": "MTI あり（静止物を除去）"},
        camera="カメラ映像",
        profile="Range Profile",
        ra="Range-Angle マップ",
        range="距離 R [m]",
        power="受信電力 [dB]",
        power_rel="受信電力 [dB]",
        bg="この走行の時間中央値（背景）",
        now="現在のフレーム",
        cross="横方向 R sin θ [m]",
        down="奥行き方向 R cos θ [m]",
        note="24 GHz FMCW レーダー（TX 2 × RX 4）・2026/10/5 昼・R は斜距離",
        # 雨の材料
        s1355="13:55（1・2）", s1422="14:22 以降（3〜6）",
        static="静止物の強さ [dB]", fluct="フレームごとの揺れ [dB]",
        static_t="静止物の強さ（MTI なし・速度 0 の時間中央値）",
        fluct_t="フレームごとの揺れ（MTI なし・0.2 s 前との差）",
        band="木の帯 R 16〜28 m",
        same="同じ回どうし", cross_s="13:55 と 14:22 以降",
        shift="角度のずれ |Δθ| [°]",
        shift_t="静止物の位置のずれ（2 本ずつ比べる）",
        relocate="参考: 10/1 → 10/5 の置き直し（5〜10°）",
        ra_1355="13:55（番号 1）", ra_1422="14:22（番号 4）", ra_diff="差（14:22 − 13:55）",
        diff="差 [dB]",
        cam_1355="13:55（番号 1）の映像", cam_1422="14:22（番号 4）の映像",
    ),
    "en": dict(
        scen={"bike": "Bicycle", "car": "Car", "bike_parked": "Bicycle passing a parked car"},
        dir={"app": "approaching", "dep": "departing"},
        kind={"nomti": "without MTI", "mti": "with MTI (static clutter removed)"},
        camera="Camera",
        profile="Range profile",
        ra="Range-angle map",
        range="Range R [m]",
        power="Received power [dB]",
        power_rel="Received power [dB]",
        bg="Time median of this run (background)",
        now="Current frame",
        cross="Cross-range R sin θ [m]",
        down="Down-range R cos θ [m]",
        note="24 GHz FMCW radar (2 TX × 4 RX), Oct 5 2026 daytime, R = slant range",
        s1355="13:55 (runs 1-2)", s1422="14:22 onward (runs 3-6)",
        static="Static return [dB]", fluct="Frame-to-frame fluctuation [dB]",
        static_t="Static return (no MTI, zero-Doppler, time median)",
        fluct_t="Frame-to-frame fluctuation (no MTI, difference from 0.2 s before)",
        band="Tree band R 16-28 m",
        same="Within a session", cross_s="13:55 vs 14:22 onward",
        shift="Angle shift |Δθ| [deg]",
        shift_t="Shift of static objects (pairwise)",
        relocate="Ref.: re-placement Oct 1 → Oct 5 (5-10 deg)",
        ra_1355="13:55 (run 1)", ra_1422="14:22 (run 4)", ra_diff="Difference (14:22 − 13:55)",
        diff="Difference [dB]",
        cam_1355="Camera at 13:55 (run 1)", cam_1422="Camera at 14:22 (run 4)",
    ),
}


def setup_font(lang):
    # 日本語は Windows 標準の Meiryo。英語は既定のまま（Meiryo の英字は幅が広く、軸の数字が詰まるため）
    matplotlib.rcParams["font.family"] = ["Meiryo"] if lang == "ja" else ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
