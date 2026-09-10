# 映像 + マーカーCSV → 地上座標ラベルCSV を1コマンドで完結させる。
#
# 【なぜ要るか】
# detect_yolo.py（検出）・homography.py（変換）・detections.py（統合・地上座標化）は
# 役割ごとにファイルを分けてある（0817/README.md参照）が、実際に1本の映像を処理するときは
# 毎回同じ順で手動で繋ぐ必要があった。ここでは呼び出し順を1本化するだけで、
# 各モジュールの役割分担・実装そのものは変えない。
#
# 【camera_pipeline直下に置いている理由】
# 実体は 0817/detect・0817/lib にあるが、日付フォルダ（0817/0908/0910）はレーダー側の
# 運用に合わせた「作業単位」の区切りであり、実際に毎回叩く入口をその中に埋めると
# 「どこから実行するか」を都度探す必要が出る。呼び出し順をまとめる本スクリプトだけは
# 直下に置き、パス解決も内部で完結させる。
#
# 【マーカーCSVについて】
# `0817/calib/pick_markers.py`（--xy --auto 推奨）で作った markers.csv を渡す。
# 動画1本（=1撮影セッション）につき、代表フレーム1枚でマーカーを拾えば十分
# （地物・カメラは録画中動かない前提。`camera_pipeline/docs/history.md` 2026-08-17参照）。
#
# カメラの設置位置・向きは撮影ごとに変わるので、Hも markers.csv も撮影ごとに別物になる。
# つまり markers.csv は基本的に毎回作るものなので、引数を省略すると
# 「既存のどれかを使う / 新規作成する」を選べるようにしてある。
# 新規作成を選ぶと、展開済みの代表フレームに対して pick_markers.py（--xy --auto）が起動し、
# そのまま検出まで続けて走る。保存名は `<映像名>.markers.csv` に固定している
# （共通の `markers.csv` 1本を使い回すと、カメラを動かした後に古いHが黙って当たって
# ラベルが全部ずれる。名前を撮影セッションに紐付けておけば取り違えに気付ける）。
#
# 【.cam 入力について】
# ATLAS の `.cam`（`.cam.gz`）はJPEGの束であって動画ではないので、YOLO には直接渡せない。
# 渡された場合は内部で `0817/capture/cam_extract.py` を呼んで連番JPEGに展開し、
# そのディレクトリを検出に回す（dtも 0.2 s に自動設定）。展開先は
# `<cam名>_frames/`（cam_extract.py を直接叩いたときと同じ場所・同じ連番）。
#
# 依存: 他の依存はすべて detect_yolo.py / detections.py 側の requirements.txt に従う
#
# 【出力先について】
# 生成される地上座標ラベルCSVは、入力動画と同じ場所ではなく `data/outputs/` に集約する
# （--out 未指定時）。入力データと生成物を混在させないため。ファイル名は動画名基準なので、
# 同じ動画を再実行すると上書き更新される。
#
# 【俯瞰図について】
# ラベル（地上座標）をXY平面に描いたPNGも `data/outputs/` に出す（--no-plot で止められる）。
# CSVの数字だけでは変換の破綻に気付けないので、既定で出す側にしてある。
# 描画は `0817/calib/ground_plot.py` の plot_tracks() を呼ぶ。
#
# --plot-frames を付けると、全trackを重ねた1枚に加えて**フレーム1枚ごとの俯瞰図**も出す
# （`<映像名>_ground_frames/00000.png` … と、それを繋いだ `<映像名>.ground.mp4`）。
# 連番はフレーム番号と1対1（`.cam` の連番＝レーダーのフレーム番号と同じ並び）なので、
# レーダー側のフレームからそのまま引ける。既定で出さないのは枚数分の描画に時間がかかるため。
#
# 【--overlay について】
# bbox・接地点・track_id を重ねた確認用動画も `data/outputs/` に出す。
# 描画は `0817/detect/overlay_detections.py` の render() をそのまま呼ぶ（実装は複製しない）。
# ラベルCSVの数字だけでは接地点のずれ（支配的な誤差要因）と 5 fps での track_id 切れが
# 見えないため、ラベルを人に渡す前の目視確認はこれで行う。4K は重いので --overlay-scale 0.5 も可。
#
# 【引数無し実行について】
# video を省略するとファイル選択ダイアログで選べる（コマンドを毎回打つのが面倒なため）。
# markers はダイアログではなく上記の選択式（一覧に無いファイルを選ぶときだけダイアログ）。
# 連番画像ディレクトリ（--dt 併用）を渡す場合はダイアログでは選べないので
# CLI引数で指定すること。ダイアログは camera_pipeline直下の `data/` を既定で開く
# （日付フォルダはコードのひとまとめ単位であって、日付をまたいで積み上がるデータの
# 置き場ではないため、データはここに集約する。8/16の過去データのみ `0817/data/` に残っている）
#
# 使い方（camera_pipeline直下から）:
#   python run_pipeline.py                                       # ダイアログで選ぶ
#   python run_pipeline.py video.mp4 markers.csv --out video.ground.csv
#   python run_pipeline.py frames/ markers.csv --dt 0.2 --out video.ground.csv
#   python run_pipeline.py atlas_log_20260820_155929.cam.gz markers.csv   # 展開込み
#   python run_pipeline.py video.mp4 markers.csv --overlay                # 確認用動画も出す
#   python run_pipeline.py atlas_log.cam --new-markers                    # markersを必ず作り直す
#   python run_pipeline.py video.mp4 markers.csv --no-plot                # 俯瞰図を出さない
#   python run_pipeline.py atlas_log.cam --plot-frames                    # フレーム毎の俯瞰図も

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent
LIB_ROOT = PIPELINE_ROOT / "0817"
# markers.csv の新規作成はこのスクリプトを起動する（下記 create_markers の理由参照）
PICK_MARKERS = LIB_ROOT / "calib" / "pick_markers.py"
sys.path.insert(0, str(LIB_ROOT / "lib"))
sys.path.insert(0, str(LIB_ROOT / "detect"))
sys.path.insert(0, str(LIB_ROOT / "capture"))

from cam_extract import FRAME_PERIOD_S, dat_frame_count, read_cam, split_jpegs   # noqa: E402
from detect_yolo import detect_video                                        # noqa: E402
from detections import add_slant_range, merge_cyclist, save, to_ground      # noqa: E402
from homography import estimate_homography, reprojection_error, spread_axes  # noqa: E402

TARGET_ACC = 0.8463541666666666 / 3   # check_homography.py と同じ目標精度[m]
# マーカー配置の副軸方向の広がり[m]の下限。これを下回ると一直線とみなして止める。
# 一直線だと H が退化して Y が常に 0 のラベルができるのに、再投影残差は 0 に近づいて
# 「精度良好」に見えてしまう（2026-09-10 に実データで発生。homography.spread_axes 参照）。
# 0.10 m は「コーンを意図的に横へずらせば必ず超える」値として置いた
MIN_MINOR_SPREAD = 0.10
# camera_pipeline直下のdata/。日付フォルダ（0817等）はコードのひとまとめ単位であって
# データの置き場ではないため、日付をまたいで積み上がるデータは別に集約している
DATA_DIR = PIPELINE_ROOT / "data"
# 生成物（地上座標ラベルCSV）は入力動画と同じ場所ではなく、ここに集約する。
# 入力データと生成物が混在すると汚くなる上、再実行のたびに同じ名前で上書き更新されるのが
# 分かりやすい（動画ファイル名基準。同名の動画を別フォルダから処理すると衝突する点に注意）
OUTPUT_DIR = DATA_DIR / "outputs"


def pick_file(title: str, filetypes: list[tuple[str, str]]) -> Path | None:
    """ファイル選択ダイアログを開く。tkinterが無い/ディスプレイが無い環境（最小構成の
    Python、SSH等）では開けないので、その場合はパスの手入力にフォールバックする"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # 他のウィンドウの裏に隠れないように
        initialdir = str(DATA_DIR) if DATA_DIR.is_dir() else str(Path.cwd())
        path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir)
        root.destroy()
        return Path(path) if path else None
    except Exception as e:
        print(f"  （ダイアログを開けなかった: {e}。パスを手入力する）")
        p = input(f"{title}のパスを入力（キャンセルは空Enter） > ").strip()
        return Path(p) if p else None


def cam_stem(cam: Path) -> str:
    """`.cam`/`.cam.gz` から拡張子を落とした名前。`.gz` は二重拡張子なので Path.stem では足りない"""
    return cam.name.removesuffix(".gz").removesuffix(".cam")


def is_cam(path: Path) -> bool:
    return path.name.endswith((".cam", ".cam.gz"))


def extract_cam(cam: Path, dat: Path | None = None) -> Path:
    """`.cam`/`.cam.gz` を連番JPEGに展開し、そのディレクトリを返す。

    【なぜここで展開するか】
    展開処理そのものは cam_extract.py の関数をそのまま呼ぶだけ（実装は複製しない）。
    ただし「.camを渡す→展開→検出」は毎回同じ順で必ず挟むので、呼び出し順を1本化する
    本スクリプトの役目に含める。展開先・連番は cam_extract.py を直接叩いた場合と同じ。

    【.dat との枚数照合】
    枚数一致は index 対応（レーダー1フレーム = JPEG 1枚）が崩れていないかの唯一の検査
    （`0908/cam_sync/README.md`）なので、同 stem の `.dat` が隣にあれば自動で照合する。
    ペアは同じ stem（`xxx.cam.gz` と `xxx.dat`）という命名前提。名前が違う場合は
    別セッションの `.dat` と取り違えると「一致」の誤判定が出るため、推測はせず
    --dat で明示させる（8/20 のように1日に複数セッション撮ることがある）"""
    jpegs = split_jpegs(read_cam(cam))
    if not jpegs:
        raise SystemExit(f"{cam.name} からJPEGが取り出せなかった。ファイルが壊れている疑い")

    frames = cam.parent / f"{cam_stem(cam)}_frames"
    frames.mkdir(parents=True, exist_ok=True)
    for i, jpg in enumerate(jpegs):
        (frames / f"{i:05d}.jpg").write_bytes(jpg)
    # 前回より短い映像を同じ名前で展開し直したとき、古い連番が残って時系列の末尾に
    # 混ざるのを防ぐ（ディレクトリ入力は sorted() で読むだけなので古い分も拾われる）
    for stale in sorted(frames.glob("*.jpg"))[len(jpegs):]:
        stale.unlink()
    print(f"  .cam展開: {len(jpegs)} 枚 → {frames}")

    dat = dat if dat is not None else cam.parent / f"{cam_stem(cam)}.dat"
    if dat.is_file():
        n_dat = dat_frame_count(dat)
        if n_dat == len(jpegs):
            print(f"  .dat({dat.name}) フレーム数 {n_dat} と一致")
        else:
            print(f"  ★不一致★ .dat({dat.name}) は {n_dat} フレーム、JPEG は {len(jpegs)} 枚。"
                  "カメラ側が間引かれていて index 対応が崩れている"
                  "（このまま進めるが、レーダーとの対応付けには使えない）")
    else:
        print(f"  ⚠ 同名の .dat（{dat.name}）が無いので枚数照合をしていない。"
              "別セッションの .dat を取り違えないよう推測はしない。"
              "照合するなら --dat でパスを渡すこと")
    return frames


def representative_frame(src: Path, stem: str, index: int | None) -> Path:
    """マーカーを指すための代表フレームを1枚用意する。

    地物・カメラは録画中動かない前提なので1枚で足りる。既定を中央のフレームにしてあるのは、
    先頭はコーンを置いた本人が写り込んでいることがあるため。隠れていたら --marker-frame で選ぶ"""
    if src.is_dir():
        files = sorted(q for q in src.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise SystemExit(f"{src} に画像が無い")
        return files[len(files) // 2 if index is None else index]

    # 動画入力。切り出した1枚は生成物なので outputs/ に置く。
    # どのフレームで指したかが後から分かる（Hの由来の記録になる）
    import cv2

    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = (n // 2) if index is None else index
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{src} の {i} フレーム目が読めない")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{stem}_frame{i:05d}.jpg"
    cv2.imwrite(str(out), im)
    return out


def source_frame_count(src: Path) -> int:
    """入力のフレーム数。フレーム毎の俯瞰図を「検出が無いフレームも含めて」出すために要る
    （連番をフレーム番号と1対1に保てば、レーダー側のフレームからそのまま引ける）"""
    if src.is_dir():
        return sum(1 for q in src.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    import cv2

    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def create_markers(src: Path, out: Path, stem: str, index: int | None) -> Path:
    """pick_markers.py を起動して markers.csv を作らせる。

    【なぜ subprocess か】pick_markers.py の対話ループ（ラフクリック→拡大窓で確定→
    端末で実測距離を入力）は main() と一体で書かれている。関数に切り出すと現地で使う
    唯一の較正手段の挙動を壊す余地があるので、実績のあるスクリプトを同じ python で
    そのまま叩く。cv2のウィンドウも端末入力も子プロセスのまま動く"""
    frame = representative_frame(src, stem, index)
    print(f"  代表フレーム: {frame}")
    print(f"  pick_markers.py を起動する（--xy --auto: カラーコーン運用）。"
          "指し終わったら q で保存・終了")
    r = subprocess.run([sys.executable, str(PICK_MARKERS), str(frame),
                        "--xy", "--auto", "--out", str(out)])
    if r.returncode != 0 or not out.is_file():
        raise SystemExit("markers.csv が作られなかった。中止")
    # 4点未満だとHが求まらない。Hの推定時ではなくここで止めた方が原因が分かりやすい
    n = sum(1 for _ in csv.DictReader(open(out, newline="", encoding="utf-8")))
    if n < 4:
        raise SystemExit(f"{out.name} は {n} 点。ホモグラフィには4点以上必要"
                         "（コーンが隠れているフレームなら --marker-frame で別のフレームにする）")
    print(f"  作成: {out}（{n} 点）")
    return out


def choose_markers(src: Path, stem: str, index: int | None, force_new: bool) -> Path:
    """markers.csv を「既存を使う / 新規作成」から選ばせる。

    カメラ位置が撮影ごとに変わるので毎回作り直すのが基本だが、同じ設置のまま連続で撮った
    2本目（8/20 の 155904 と 155929 のような25秒差の2本）では前のものが使える。
    どちらも起こるので、既定を押し付けずに選ばせる"""
    session = src.parent / f"{stem}.markers.csv"
    if force_new:
        return create_markers(src, session, stem, index)

    # このセッション名のものを先頭に、同じフォルダの他の markers を新しい順で並べる
    cands = [session] if session.is_file() else []
    cands += sorted((q for q in src.parent.glob("*markers*.csv") if q != session),
                    key=lambda q: q.stat().st_mtime, reverse=True)

    print("\nmarkers.csv をどうするか:")
    for n, q in enumerate(cands, 1):
        tag = "この映像用" if q == session else "別セッション"
        print(f"  [{n}] {q.name}（{tag}）")
    print("  [n] 新規作成（代表フレームでマーカーを指す）")
    print("  [f] 一覧に無いファイルを選ぶ（ダイアログ）")
    default = "1" if cands else "n"
    try:
        ans = input(f"選択 [{default}] > ").strip().lower() or default
    except EOFError:              # 対話端末でない場合（バッチ実行等）
        ans = default
    if ans == "n":
        return create_markers(src, session, stem, index)
    if ans == "f":
        got = pick_file("markers.csv を選択", [("csv", "*.csv"), ("all", "*.*")])
        if got is None:
            raise SystemExit("markers.csv が選択されなかった。中止")
        return got
    if ans.isdigit() and 1 <= int(ans) <= len(cands):
        got = cands[int(ans) - 1]
        if got != session:
            # カメラを動かしていれば別セッションのHは使えない。ここで気付けるようにする
            print(f"  ⚠ {got.name} はこの映像用（{session.name}）ではない。"
                  "撮影の間にカメラを動かしていないことを確認すること")
        return got
    raise SystemExit(f"選択 '{ans}' が不正。中止")


def load_markers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    xy = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
    return uv, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, nargs="?", default=None,
                    help="映像ファイル、または --dt 指定時は連番画像ディレクトリ。"
                         "省略するとダイアログで選ぶ")
    ap.add_argument("markers", type=Path, nargs="?", default=None,
                    help="pick_markers.py の出力（markers.csv）。"
                         "省略すると既存の再利用／新規作成を選べる")
    ap.add_argument("--new-markers", action="store_true",
                    help="markers.csv を必ず作り直す（選択を飛ばして pick_markers.py を起動）")
    ap.add_argument("--marker-frame", type=int, default=None,
                    help="markers を指すフレーム番号（既定: 中央）。"
                         "コーンが隠れているフレームに当たったときに変える")
    ap.add_argument("--out", type=Path, default=None,
                    help="既定: data/outputs/<video名>.ground.csv（同名なら上書き）")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dt", type=float, default=None,
                    help="連番画像ディレクトリ入力の場合は必須（.cam由来なら0.2）。"
                         ".cam を直接渡した場合は自動で 0.2 になる")
    ap.add_argument("--dat", type=Path, default=None,
                    help=".cam 入力時、枚数照合に使う .dat。"
                         "既定は .cam と同名（見つからなければ照合しない）")
    ap.add_argument("--no-plot", action="store_true",
                    help="地上座標の俯瞰図（data/outputs/<video名>.ground.png）を出さない")
    ap.add_argument("--plot-frames", action="store_true",
                    help="フレーム1枚ごとの俯瞰図も出す"
                         "（data/outputs/<video名>_ground_frames/ と .ground.mp4）")
    ap.add_argument("--overlay", action="store_true",
                    help="bbox・接地点・track_id を重ねた確認用動画も出す"
                         "（data/outputs/<video名>_overlay.mp4）")
    ap.add_argument("--overlay-scale", type=float, default=1.0,
                    help="--overlay の書き出し倍率。4K が重いとき 0.5 など")
    ap.add_argument("--radar-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="指定時のみ斜距離R_slant列を追加（学習データ組み立て時のみ使う）")
    ap.add_argument("--h", type=float, default=None, help="--radar-xy と併用。カメラ設置高[m]")
    args = ap.parse_args()

    if args.video is None:
        args.video = pick_file("映像ファイルを選択",
                               [("video/cam", "*.mp4 *.avi *.mov *.cam *.cam.gz"),
                                ("all", "*.*")])
        if args.video is None:
            raise SystemExit("映像が選択されなかった。中止")
    if args.markers is not None and args.new_markers:
        raise SystemExit("markers を指定しているのに --new-markers。どちらか一方にする")

    # --- 1. .cam なら先に展開する。markers を新規作成するとき、指す対象の
    #        代表フレームが展開済みでないと始められないため ---
    if is_cam(args.video):
        label_stem = cam_stem(args.video)
        print("[1/4] .cam を展開 ...")
        args.video = extract_cam(args.video, args.dat)
        # .cam の時刻は frame index × frame periodicity で作る決まり（fpsからは作れない）
        if args.dt is None:
            args.dt = FRAME_PERIOD_S
    else:
        label_stem = args.video.stem
        kind = "連番画像" if args.video.is_dir() else "動画"
        print(f"[1/4] 入力は{kind}。展開は不要")

    # --- 2. markers.csv を決めてHを推定し、先に精度を確認する（時間のかかるYOLO推論の前に） ---
    if args.markers is None:
        args.markers = choose_markers(args.video, label_stem,
                                      args.marker_frame, args.new_markers)
    uv_m, xy_m = load_markers(args.markers)
    H = estimate_homography(uv_m, xy_m)
    e = reprojection_error(H, uv_m, xy_m)
    print(f"\n[2/4] {args.markers.name} の{len(uv_m)}点からH推定。"
          f"再投影残差 平均{e.mean():.3f}m / 最大{e.max():.3f}m")
    # 残差より先に配置の退化を見る。残差はこの異常を検出できないため
    major, minor = spread_axes(xy_m)
    print(f"  マーカー配置の広がり 主軸 {major:.2f} m / 副軸 {minor:.2f} m")
    if minor < MIN_MINOR_SPREAD:
        raise SystemExit(
            f"★ マーカーが一直線に並んでいる（副軸方向 {minor:.3f} m）。このまま進めると"
            "Hが退化して Y が常に 0 のラベルができる（再投影残差は 0 に近づくので"
            "残差では気付けない）。マーカーを横方向にもばらして指し直すこと"
            "（--new-markers で作り直せる）。中止")
    if e.max() > TARGET_ACC:
        print(f"  ⚠ 目標精度 {TARGET_ACC:.2f}m を超えている点がある。"
              "このまま進めてよいが、markers.csv の測り間違いを疑うこと"
              "（`0817/calib/pick_markers.py --check` で個別残差を確認できる）")

    # --- 3. YOLO検出（時間がかかる本体） ---
    print(f"\n[3/4] YOLO検出・追跡 ...")
    raw = detect_video(args.video, args.model, args.conf, args.imgsz, dt=args.dt)
    print(f"  検出 {len(raw)} 件 / 追跡ID {raw['track_id'].nunique()} 個")

    # --- 4. cyclist統合 → 地上座標 ---
    merged = merge_cyclist(raw)
    labeled = to_ground(merged, H)
    if args.radar_xy is not None:
        if args.h is None:
            raise SystemExit("--radar-xy 指定時は --h も必須")
        labeled = add_slant_range(labeled, radar_xy=tuple(args.radar_xy), h=args.h)

    if args.out is not None:
        out = args.out
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # 展開後のディレクトリ名（*_frames）ではなく元の .cam / 動画名を基準にする
        # （同じ撮影セッションの成果物だと分かる名前にしたいため）
        out = OUTPUT_DIR / (label_stem + ".ground.csv")
    save(labeled, out)
    print(f"\n[4/4] 保存: {out} ({len(labeled)} 行)")
    print(labeled.groupby("cls").size().rename("件数"))

    # 変換が破綻していてもCSVの数字だけでは気付けないので、既定で俯瞰図も出す。
    # matplotlib の import はここまで遅らせる（図が不要な実行で待たされないため）
    if not args.no_plot:
        sys.path.insert(0, str(LIB_ROOT / "calib"))
        from ground_plot import plot_tracks
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # --out 指定時はまだ無いことがある
        png = plot_tracks(labeled, xy_m, OUTPUT_DIR / f"{label_stem}.ground.png",
                          title=label_stem, radar_xy=args.radar_xy)
        print(f"  俯瞰図: {png}")
        if args.plot_frames:
            from ground_plot import plot_tracks_per_frame
            d = OUTPUT_DIR / f"{label_stem}_ground_frames"
            mp4 = OUTPUT_DIR / f"{label_stem}.ground.mp4"
            n = plot_tracks_per_frame(labeled, xy_m, d, source_frame_count(args.video),
                                      dt=args.dt or 1 / 30, title=label_stem,
                                      radar_xy=args.radar_xy, mp4=mp4)
            print(f"  フレーム毎の俯瞰図: {d} ({n} 枚) / {mp4.name}")

    # 4K の書き出しは時間がかかるので、ラベルCSVを保存し切ってから最後に回す
    # （描画で落ちてもCSVは残る）。cv2 の import もここまで遅らせる
    if args.overlay:
        from overlay_detections import render, summarize
        ov = OUTPUT_DIR / f"{label_stem}_overlay.mp4"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # --out 指定時はまだ無いことがある
        # 動画入力で --dt 未指定なら、元動画の fps に合わせる（既定値を当てると再生速度が
        # 変わって track_id の切れ方が実時間と対応しなくなる）
        if args.dt is not None:
            dt = args.dt
        else:
            import cv2
            cap = cv2.VideoCapture(str(args.video))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            dt = 1.0 / (fps if fps > 0 else 30.0)
        # 描くのは merge_cyclist 後の3クラス。merged を渡すので検出CSVの書き出し・読み直しは不要
        n = render(args.video, merged, ov, dt=dt, scale=args.overlay_scale)
        print(f"\n[overlay] {ov}: {n} フレーム")
        summarize(merged, n)


if __name__ == "__main__":
    main()
