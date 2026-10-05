# 週末の研究室収録（1 回）の走行記録表を生成する。記入するのは 5F の記録係なので、
# その場で埋める列を前に置く（weekly/0928/make_run_sheet.py と同じ方針）。
#
# 何を撮るか（優先順。根拠は collect/README.md）:
#   H2 窓 開/閉: 置き方 a のまま、開→閉→閉→開（ABBA）で自転車レーン1の離反。時間による変化を打ち消す
#   H1 向き a/b/c: 各置き方でレーン1の離反 ×2 ＋ レーン3の離反 ×1。ブロック（置き方）内で順をシャッフル
#      a のレーン1 は H2 の「開」の 2 本で兼ねる
#   H3 背景: 置き方・窓を変えるたびに、その前後で無人 30 s（直前の「後」と直後の「前」は兼ねない。
#      置き方を変える作業で三脚に触るので、触った後の背景が別に要る）
#   H4 車: 最後に、近い端が最もよく見えた置き方で徐行 1 往復（使えれば）
# 置き方 a → b → c の順は固定する（c は b の向きのまま位置を変えるだけなので戻す手間が省ける）。
# 時間との交絡は、b で見えたら c を撮らない判断と、時間が余れば最後に a に戻す 1 本で確かめる。
#
# 列「所要[分]」は見積もり。自転車 1 本 ≈ 1 分（10/1 夜の実績: 6 本で約 6 分）、背景 30 s ＋ 準備で 1.5 分。
#
# 使い方:
#   python make_run_sheet.py            # run_sheet.csv を出力
#   python make_run_sheet.py --seed 7

import argparse
import csv
import random

T_BIKE = 1.0      # 自転車 1 本（収録 30 s ＋ 戻り）[分]
T_BG = 1.5        # 無人背景 30 s ＋ 被験者を走路の外へ出す [分]
T_WINDOW = 3.0    # 窓の開閉 [分]
T_TILT = 10.0     # 向き b: 下へ 15〜20° 傾け、角度を測る [分]
T_MOVE = 15.0     # 置き方 c: 開口部の外側へ寄せ、コーン C1・C5 の斜距離を測り直す [分]
T_CAR = 10.0      # 車の段取り（1 往復 = 2 ファイル）[分]

# 記録係がその場で埋める列 → 後から埋める列 の順
HEADER = ["No", "優先", "実験", "置き方", "窓", "対象", "レーン", "向き", "ファイル名", "開始時刻",
          "確認（check_run_1013 の判定）", "備考（ふらつき・停止・通行人・風）", "所要[分]"]


def rows(seed):
    rnd = random.Random(seed)
    out = []

    def add(prio, exp, place, win, obj, lane="-", dr="-", note="", t=0.0):
        out.append([len(out) + 1, prio, exp, place, win, obj, lane, dr, "", "", "", note, t])

    def bg(prio, place, win, when):
        add(prio, "H3", place, win, "無人 30 s", note=f"背景（{when}）。操作者は録画開始から終了まで動かない", t=T_BG)

    def bike_block(prio, exp, place, win, lanes):
        order = lanes[:]
        rnd.shuffle(order)                                   # ブロック内でレーン順をシャッフル
        for ln in order:
            p = prio if ln == 1 else "B"                     # レーン3 は時間が足りなければ削る
            add(p, exp, place, win, "自転車", ln, "離反",
                note="C1 の手前から助走し、C1 を巡航速度で通過。録画開始から 5 s 待って出発", t=T_BIKE)

    # ---- H2: 窓 開/閉（置き方 a） ABBA ----
    bg("A", "a", "開", "H2 前")
    bike_block("A", "H2", "a", "開", [1])
    add("A", "-", "a", "開→閉", "作業", note="窓を閉める（半分開けていた側）", t=T_WINDOW)
    bg("A", "a", "閉", "閉の前")
    bike_block("A", "H2", "a", "閉", [1])
    bike_block("B", "H2", "a", "閉", [1])                    # 2 回目は削れる（ABBA → AB）
    bg("A", "a", "閉", "閉の後")
    add("A", "-", "a", "閉→開", "作業", note="窓を開ける（9/28・10/1 と同じ半開き）", t=T_WINDOW)
    bg("A", "a", "開", "開に戻した後")
    bike_block("B", "H2", "a", "開", [1])

    # ---- H1: 向き a/b/c ----
    # 置き方 a のレーン1 は H2 の「開」の 2 本で兼ねる（同じ条件）。ここはレーン3 だけ
    bike_block("B", "H1", "a", "開", [3])
    bg("A", "a", "開", "a の後")
    add("A", "-", "a→b", "開", "作業", note="下へ 15〜20° 傾ける。角度を記録", t=T_TILT)
    bg("A", "b", "開", "b の前")
    bike_block("A", "H1", "b", "開", [1, 1, 3])
    bg("A", "b", "開", "b の後")
    add("A'", "-", "b→c", "開", "作業",
        note="b で近い端が見えたら c は撮らず H4 へ。c: 向き b のまま開口部の外側へ寄せ、C1・C5 を測り直す", t=T_MOVE)
    bg("A'", "c", "開", "c の前")
    bike_block("A'", "H1", "c", "開", [1, 1, 3])
    bg("A'", "c", "開", "c の後")

    # ---- H4: 車（使えれば） ----
    for dr in ("接近", "離反"):
        add("C", "H4", "最良", "開", "車 徐行 15 km/h", 1, dr,
            note="15 km/h を超えると速度が折り返す（±4.8 m/s）", t=T_CAR / 2)

    # ---- 任意: a に戻して時間による変化を確かめる ----
    add("D", "-", "→a", "開", "作業", note="時間が余れば。a に戻す", t=T_TILT)
    bike_block("D", "H1", "a", "開", [1])
    bg("D", "a", "開", "最後")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1013, help="実施順のシャッフル種。再現性のため固定")
    ap.add_argument("--out", default="run_sheet.csv")
    args = ap.parse_args()

    data = rows(args.seed)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:   # Excel 用に BOM 付き
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(data)

    for r in data:
        print(f"{r[0]:>3} {r[1]:<2} {r[2]:<3} {r[3]:<4} {r[4]:<5} {r[5]:<14} L{r[6]} {r[7]:<3} {r[12]:4.1f}分  {r[11]}")
    n_file = sum(1 for r in data if r[5] != "作業")
    t_all = sum(r[12] for r in data)
    print(f"\n全 {n_file} ファイル、計 {t_all:.0f} 分（設置・コーン測量の 30 分は別）")
    # 削る順に積み上げたときの所要
    for cut in ("D", "C", "B", "A'"):
        t_all -= sum(r[12] for r in data if r[1] == cut)
        print(f"  優先 {cut} まで削ると {t_all:.0f} 分")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
