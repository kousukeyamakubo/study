# 複数の .dat 録画のヘッダを byte 単位で比較し、設定を1つ変えたときにどこが変わるかを特定する
#
# 【背景】meeting/2026-09-08.md H8。ヘッダ 0x64〜0xB3 は未同定領域で、Remove Static Clutter の
# フラグがこの中にあると推定されている（0727/atlas_dat_format.md §2）。fs・サンプル数・PRI が
# アプリの設定で変更可能かどうかも同じ手法（設定を振って差分を見る）で確認する。
# atlas_provenance_check.py は1ファイルの内部整合性チェックであり、複数ファイル間の比較はしない。
#
# ヘッダの読み方は 0727/atlas_dat_parse.py の parse_header() と同一だが、
# このスクリプトは256byteしか読まないので numpy/matplotlib への依存を避けて独立に持つ
# （元ファイルは RD マップ描画のために両ライブラリを読み込む）。
#
# 使い方:
#   python header_diff.py raw/atlas_log_..._clutteroff.dat raw/atlas_log_..._clutteron.dat

import argparse
import struct
from pathlib import Path

HEADER_BYTES = 256

# 0x64〜0xB3（80 byte = int32 20個）。atlas_dat_format.md で「未同定」と記録された領域
UNKNOWN_START = 0x64
UNKNOWN_END = 0xB4  # exclusive


def parse_header(h: bytes) -> dict:
    """256byte ヘッダから判明済みパラメータを取り出す（0727/atlas_dat_parse.py と同一定義）"""
    d = lambda off: struct.unpack_from("<d", h, off)[0]
    i = lambda off: struct.unpack_from("<i", h, off)[0]
    return dict(
        start_freq_ghz=d(0x0C),
        slope_mhz_us=d(0x14),
        n_sample=i(0x1C),
        n_chirp_per_frame=i(0x20),
        sampling_ksps=i(0x24),
        bw_ghz=d(0x28),
        frame_period_ms=i(0x30),
        n_tx=i(0x34),
        n_rx=i(0x38),
        chirp_interval_us=d(0x3C),
        range_res_m=d(0x44),
        max_range_m=d(0x4C),
        max_vel_ms=d(0x54),
        vel_res_ms=d(0x5C),
    )


def unknown_region(raw: bytes) -> list:
    return [struct.unpack_from("<i", raw, off)[0]
            for off in range(UNKNOWN_START, UNKNOWN_END, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dats", type=Path, nargs="+", help=".dat ファイル（2つ以上）")
    args = ap.parse_args()
    if len(args.dats) < 2:
        ap.error("比較には2ファイル以上が必要です")

    headers, unknowns = [], []
    for p in args.dats:
        raw = p.read_bytes()[:HEADER_BYTES]
        headers.append(parse_header(raw))
        unknowns.append(unknown_region(raw))

    names = [p.name for p in args.dats]
    print("比較対象:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")

    print("\n===== 既知フィールドの差分 =====")
    n_changed_known = 0
    for k in headers[0]:
        vals = [h[k] for h in headers]
        changed = len(set(vals)) > 1
        if changed:
            n_changed_known += 1
        mark = "  <- CHANGED" if changed else ""
        print(f"  {k:20s} " + "  ".join(f"{v}" for v in vals) + mark)

    print("\n===== 未同定領域 0x64-0xB3 の差分（int32, offsetごと）=====")
    n_changed_unknown = 0
    for i in range(len(unknowns[0])):
        off = UNKNOWN_START + i * 4
        vals = [u[i] for u in unknowns]
        changed = len(set(vals)) > 1
        if changed:
            n_changed_unknown += 1
        mark = "  <- CHANGED" if changed else ""
        print(f"  0x{off:02X}  " + "  ".join(f"{v:6d}" for v in vals) + mark)

    print(f"\n既知フィールドの変化: {n_changed_known} 件 / 未同定領域の変化: {n_changed_unknown} 件")
    if n_changed_unknown == 0:
        print("→ Remove Static Clutter のフラグはこの領域には無いか、この2ファイルの条件差では表れていない")
    elif n_changed_unknown == 1:
        print("→ 変化したoffsetが1箇所のみ。他の設定を変えていなければ、それがフラグ位置の有力候補")
    else:
        print("→ 複数箇所が変化。設定が複数連動して変わっている可能性があるため、条件を絞って再検証が必要")


if __name__ == "__main__":
    main()
