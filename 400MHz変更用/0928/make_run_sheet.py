# 走行記録表を生成する。記入するのは 5F の記録係なので、その場で埋められる列だけを前に置く。
#
# 設計（README「9/29 実施分における割り切りの内容と根拠」）:
#   2ライン（Y=0.5, 3.5）× 中速のみ × 2方向 × 3反復 = 12本（6往復）＋ 背景2本
#   1往復で離反・接近の2ファイルが撮れるため、シャッフルの単位は「ブロック内のライン順」。
#   疲労・日照・熱ドリフトとの交絡を避けるため反復ごとにブロック化する。
#
# 低速水準（任意）は本来の設計に含まれるが 9/29 は時間優先で落とす。時間が余った場合に
# 追加できるよう、同じ構成の行を末尾に付ける。
#
# 使い方:
#   python make_run_sheet.py            # run_sheet.csv を出力
#   python make_run_sheet.py --seed 7

import argparse
import csv
import random

LINES = [0.5, 3.5]          # パイロットは最外側2本のみ
DIRECTIONS = ["離反", "接近"]   # 往復走行なので必ずこの順（X=0 から出発）
N_BLOCK = 3                 # 反復数

# 記録係がその場で埋める列 → 後から映像で埋める列、の順に並べる
HEADER = ["No", "区分", "ブロック", "ライン Y[m]", "速度水準", "方向",
          "ファイル名", "開始時刻", "備考（ふらつき・停止等）",
          "始点コーン frame", "中央コーン frame", "終点コーン frame", "実速度[m/s]"]


def rows(seed):
    rnd = random.Random(seed)
    out, n = [], 0

    n += 1
    out.append([n, "背景", "-", "-", "-", "-", "", "", "無人30秒（頭）", "", "", "", ""])

    for speed, section in (("中速", "本測定"), ("低速", "任意（時間が余れば）")):
        for b in range(1, N_BLOCK + 1):
            order = LINES[:]
            rnd.shuffle(order)                      # ブロック内でライン順をシャッフル
            for y in order:
                for d in DIRECTIONS:                # 1往復 = 離反 + 接近
                    n += 1
                    out.append([n, section, b, y, speed, d, "", "", "", "", "", "", ""])

    n += 1
    out.append([n, "背景", "-", "-", "-", "-", "", "", "無人30秒（尻）", "", "", "", ""])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=928, help="実施順のシャッフル種。再現性のため固定")
    ap.add_argument("--out", default="run_sheet.csv")
    args = ap.parse_args()

    data = rows(args.seed)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:   # Excel 用に BOM 付き
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(data)

    for r in data:
        print(f"{str(r[0]):>3} {r[1]:<20} B{r[2]} Y={str(r[3]):<4} {r[4]:<4} {r[5]:<4} {r[8]}")
    main_n = sum(1 for r in data if r[1] == "本測定")
    print(f"\n本測定 {main_n} 本 ＋ 任意 {sum(1 for r in data if r[1].startswith('任意'))} 本 "
          f"＋ 背景 2 本  -> {args.out}")


if __name__ == "__main__":
    main()
