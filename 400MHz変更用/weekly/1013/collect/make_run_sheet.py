# 週末の研究室収録（1 回）の走行記録表を生成する。記入するのは 5F の記録係なので、
# その場で埋める列を前に置く（weekly/0928/make_run_sheet.py と同じ方針）。
#
# 何を撮るか（優先順。根拠は collect/README.md）:
#   A  確かめる用の走行: 置き方 a・窓 開（10/1 夜と同じ条件）に固定し、徒歩・自転車 × レーン 1・2 × 接近・離反を各 2 本。
#      10/1 夜で決めた処理が、別のデータでも見逃し・誤警報 5% に届くかを確かめる（weekly/1013/1008/ T3）。
#      順はブロック全体でシャッフルする（時間による変化と条件が重ならないように）。
#      無人背景を最初・途中・最後に 30 s ずつ撮り、目標がいないときの誤警報（Type 1）の分母にする
#   B  窓 開/閉: 窓を閉めて自転車レーン1の離反 ×2、開け直して ×1。開の比較相手は A の同じ条件の走行
#   D  コーンの差分測定（任意）: 無人背景 → 反射板つきコーンを置く → 30 s。仰角方向のパターンを見る材料
#   C  置き方 b: 下へ 15〜20° 傾けて自転車レーン1の離反 ×2。近い帯が見えるようになるか
#   車は撮らない（カテゴリ 2 の対象は人。近い帯が見えないのはビームのせいだと 10/5 で分かった）。
#   レーン 3 も撮らない（10/1 夜で直接波がほぼ見えず、真値の軌跡が当てはめられない）。
# 実施順は A → B → D → C（置き方を変える C は最後。a に戻す必要がない）。削る順は C → D → B。A は削らない。
#
# 列「所要[分]」は見積もり。自転車 1 本 ≈ 1 分（10/1 夜の実績: 6 本で約 6 分）、徒歩は 1.5 分、背景 30 s ＋ 準備で 1.5 分。
#
# 使い方:
#   python make_run_sheet.py            # run_sheet.csv を出力
#   python make_run_sheet.py --seed 7

import argparse
import csv
import random

T_BIKE = 1.0      # 自転車 1 本（収録 30 s ＋ 戻り）[分]
T_WALK = 1.5      # 徒歩 1 本（60 m を 1.4 m/s で約 45 s ＋ 戻り）[分]
T_BG = 1.5        # 無人背景 30 s ＋ 被験者を走路の外へ出す [分]
T_RECORD = 3.0    # 高さ・傾きを測って記録する [分]
T_WINDOW = 3.0    # 窓の開閉 [分]
T_CONE = 5.0      # 反射板つきコーンを置く・片付ける [分]
T_TILT = 10.0     # 置き方 b: 下へ 15〜20° 傾け、角度を測る [分]
N_REP = 2         # A の各条件の本数

# 記録係がその場で埋める列 → 後から埋める列 の順
HEADER = ["No", "優先", "実験", "置き方", "窓", "対象", "レーン", "向き", "ファイル名", "開始時刻",
          "確認（check_run_1013 の判定）", "備考（ふらつき・停止・通行人・風）", "所要[分]"]

NOTE_BIKE = "C1 の手前から助走し、C1 を巡航速度で通過。録画開始から 5 s 待って出発"
NOTE_WALK = "ふだんの速さで一定に歩く。録画開始から 5 s 待って出発"


def rows(seed):
    rnd = random.Random(seed)
    out = []

    def add(prio, exp, place, win, obj, lane="-", dr="-", note="", t=0.0):
        out.append([len(out) + 1, prio, exp, place, win, obj, lane, dr, "", "", "", note, t])

    def bg(prio, exp, place, win, when):
        add(prio, exp, place, win, "無人 30 s", note=f"背景（{when}）。操作者は録画開始から終了まで動かない", t=T_BG)

    def run(prio, exp, place, win, obj, lane, dr):
        if obj == "自転車":
            add(prio, exp, place, win, obj, lane, dr, note=NOTE_BIKE, t=T_BIKE)
        else:
            add(prio, exp, place, win, obj, lane, dr, note=NOTE_WALK, t=T_WALK)

    # ---- 設置の記録（置き方 a） ----
    add("A", "-", "a", "開", "作業",
        note="高さ: 窓からレーザーで真下の地面まで。傾き: スマホを板の面に当てて水準器アプリで読む。天気・風も書く",
        t=T_RECORD)

    # ---- A: 確かめる用の走行（置き方 a・窓 開に固定） ----
    conds = [(obj, lane, dr) for obj in ("徒歩", "自転車") for lane in (1, 2) for dr in ("接近", "離反")] * N_REP
    rnd.shuffle(conds)                                       # 時間による変化と条件が重ならないように
    half = len(conds) // 2
    bg("A", "V", "a", "開", "A の最初")
    for c in conds[:half]:
        run("A", "V", "a", "開", *c)
    bg("A", "V", "a", "開", "A の途中")
    for c in conds[half:]:
        run("A", "V", "a", "開", *c)
    bg("A", "V", "a", "開", "A の最後")

    # ---- B: 窓 開/閉（置き方 a のまま。開の比較相手は A の自転車レーン1の離反） ----
    add("B", "-", "a", "開→閉", "作業", note="窓を閉める（半分開けていた側）。レーダーに触らない", t=T_WINDOW)
    bg("B", "W", "a", "閉", "閉の前")
    run("B", "W", "a", "閉", "自転車", 1, "離反")
    run("B", "W", "a", "閉", "自転車", 1, "離反")
    bg("B", "W", "a", "閉", "閉の後")
    add("B", "-", "a", "閉→開", "作業", note="窓を開ける（9/28・10/1 と同じ半開き）", t=T_WINDOW)
    run("B", "W", "a", "開", "自転車", 1, "離反")       # 開け直した後。時間による変化の確認を兼ねる

    # ---- D: コーンの差分測定（任意。置き方 a のうちに） ----
    bg("D", "K", "a", "開", "コーンを置く前")
    add("D", "-", "a", "開", "作業",
        note="反射板（アルミ or コーナーリフレクタ）つきコーンを C3 に置く。置いた人は視野から出て 5 s 待つ", t=T_CONE)
    add("D", "K", "a", "開", "コーン 30 s", note="コーンの斜距離をレーザーで測って書く。操作者は動かない", t=T_BG)

    # ---- C: 置き方 b（最後。a には戻さない） ----
    add("C", "-", "a→b", "開", "作業", note="下へ 15〜20° 傾ける。傾きを測って書く", t=T_TILT)
    bg("C", "T", "b", "開", "b の前")
    run("C", "T", "b", "開", "自転車", 1, "離反")
    run("C", "T", "b", "開", "自転車", 1, "離反")
    bg("C", "T", "b", "開", "b の後")
    add("C", "-", "b", "開", "作業", note="終わりの傾きを測って書く（触っていないことの確認）", t=T_RECORD)
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
        print(f"{r[0]:>3} {r[1]:<2} {r[2]:<2} {r[3]:<4} {r[4]:<5} {r[5]:<10} L{r[6]} {r[7]:<3} {r[12]:4.1f}分  {r[11]}")
    n_file = sum(1 for r in data if r[5] != "作業")
    t_all = sum(r[12] for r in data)
    print(f"\n全 {n_file} ファイル、計 {t_all:.0f} 分（設置・コーン測量の 30 分は別）")
    # 削る順に積み上げたときの所要
    for cut in ("C", "D", "B"):
        t_all -= sum(r[12] for r in data if r[1] == cut)
        print(f"  優先 {cut} まで削ると {t_all:.0f} 分")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
