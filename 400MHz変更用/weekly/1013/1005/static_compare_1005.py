# 10/5 の 6 本で、静止物の強さとフレームごとの揺れを距離ビンごとに並べる。
#
# 自転車単独（13:55）と車あり（14:22〜）の図の見え方の違いが、どの距離で・何によって
# 起きているかを確かめるため。距離帯をまとめて平均すると局所的な変化が埋もれるので、ビンごとに出す。
#   静止物: MTI なしの速度 0 ビンの振幅（8ch 加算）の時間中央値
#   揺れ  : MTI なしの 0.2 s 前との複素差分（8ch 加算、速度方向に最大）の時間中央値
# 静止物が変わらずに揺れだけ増えれば「同じ反射がフレームごとに変わっている」ことになる。
#
# 使い方:
#   cd weekly/1013/1005
#   python static_compare_1005.py      （事前に axes_maps_1005.py で data/export_nomti を作る）

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ["135532", "135606", "142246", "142546", "143104", "143133"]
R_LO, R_HI = 10.0, 45.0


def profiles(tag):
    s = f"atlas_log_20261005_{tag}"
    d = np.load(HERE / "data" / "export_nomti" / s / f"{s}_rd.npz")
    rd = d["rd"]                                                   # (F, TX, RX, D, R)
    i0 = int(np.argmin(np.abs(d["vel_ms"])))                      # 速度 0 のビン番号
    amp = np.abs(rd).sum(axis=(1, 2))                              # (F, D, R)
    static = 20 * np.log10(np.median(amp[:, i0, :], axis=0))       # (R,)
    diff = np.abs(np.diff(rd, axis=0)).sum(axis=(1, 2)).max(axis=1)  # (F-1, R)
    fluct = 20 * np.log10(np.median(diff, axis=0))                 # (R,)
    return d["range_m"], static, fluct


def main():
    res = [profiles(t) for t in TAGS]
    rng = res[0][0]
    print("静止物 [dB]" + " " * 33 + "| 揺れ [dB]")
    print("R [m] " + " ".join(f"{t:>6}" for t in TAGS) + " | " + " ".join(f"{t:>6}" for t in TAGS))
    for k in np.where((rng >= R_LO) & (rng <= R_HI))[0]:
        st = " ".join(f"{r[1][k]:6.1f}" for r in res)
        fl = " ".join(f"{r[2][k]:6.1f}" for r in res)
        print(f"{rng[k]:5.1f} {st} | {fl}")


if __name__ == "__main__":
    main()
