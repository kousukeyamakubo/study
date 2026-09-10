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
# 依存: 他の依存はすべて detect_yolo.py / detections.py 側の requirements.txt に従う
#
# 【出力先について】
# 生成される地上座標ラベルCSVは、入力動画と同じ場所ではなく `data/outputs/` に集約する
# （--out 未指定時）。入力データと生成物を混在させないため。ファイル名は動画名基準なので、
# 同じ動画を再実行すると上書き更新される。
#
# 【引数無し実行について】
# video・markers を省略するとファイル選択ダイアログで選べる（コマンドを毎回打つのが
# 面倒なため）。連番画像ディレクトリ（--dt 併用）を渡す場合はダイアログでは選べないので
# CLI引数で指定すること。ダイアログは camera_pipeline直下の `data/` を既定で開く
# （日付フォルダはコードのひとまとめ単位であって、日付をまたいで積み上がるデータの
# 置き場ではないため、データはここに集約する。8/16の過去データのみ `0817/data/` に残っている）
#
# 使い方（camera_pipeline直下から）:
#   python run_pipeline.py                                       # ダイアログで選ぶ
#   python run_pipeline.py video.mp4 markers.csv --out video.ground.csv
#   python run_pipeline.py frames/ markers.csv --dt 0.2 --out video.ground.csv

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent
LIB_ROOT = PIPELINE_ROOT / "0817"
sys.path.insert(0, str(LIB_ROOT / "lib"))
sys.path.insert(0, str(LIB_ROOT / "detect"))

from detect_yolo import detect_video                                        # noqa: E402
from detections import add_slant_range, merge_cyclist, save, to_ground      # noqa: E402
from homography import estimate_homography, reprojection_error             # noqa: E402

TARGET_ACC = 0.8463541666666666 / 3   # check_homography.py と同じ目標精度[m]
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
                    help="pick_markers.py の出力（markers.csv）。省略するとダイアログで選ぶ")
    ap.add_argument("--out", type=Path, default=None,
                    help="既定: data/outputs/<video名>.ground.csv（同名なら上書き）")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dt", type=float, default=None,
                    help="連番画像ディレクトリ入力の場合は必須（.cam由来なら0.2）")
    ap.add_argument("--radar-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="指定時のみ斜距離R_slant列を追加（学習データ組み立て時のみ使う）")
    ap.add_argument("--h", type=float, default=None, help="--radar-xy と併用。カメラ設置高[m]")
    args = ap.parse_args()

    if args.video is None:
        args.video = pick_file("映像ファイルを選択", [("video", "*.mp4 *.avi *.mov"), ("all", "*.*")])
        if args.video is None:
            raise SystemExit("映像が選択されなかった。中止")
    if args.markers is None:
        args.markers = pick_file("markers.csv を選択", [("csv", "*.csv"), ("all", "*.*")])
        if args.markers is None:
            raise SystemExit("markers.csv が選択されなかった。中止")

    # --- 1. マーカーからHを推定し、先に精度を確認する（時間のかかるYOLO推論の前に） ---
    uv_m, xy_m = load_markers(args.markers)
    H = estimate_homography(uv_m, xy_m)
    e = reprojection_error(H, uv_m, xy_m)
    print(f"[1/3] マーカー{len(uv_m)}点からH推定。再投影残差 平均{e.mean():.3f}m / 最大{e.max():.3f}m")
    if e.max() > TARGET_ACC:
        print(f"  ⚠ 目標精度 {TARGET_ACC:.2f}m を超えている点がある。"
              "このまま進めてよいが、markers.csv の測り間違いを疑うこと"
              "（`0817/calib/pick_markers.py --check` で個別残差を確認できる）")

    # --- 2. YOLO検出（時間がかかる本体） ---
    print(f"\n[2/3] YOLO検出・追跡 ...")
    raw = detect_video(args.video, args.model, args.conf, args.imgsz, dt=args.dt)
    print(f"  検出 {len(raw)} 件 / 追跡ID {raw['track_id'].nunique()} 個")

    # --- 3. cyclist統合 → 地上座標 ---
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
        out = OUTPUT_DIR / (args.video.stem + ".ground.csv")
    save(labeled, out)
    print(f"\n[3/3] 保存: {out} ({len(labeled)} 行)")
    print(labeled.groupby("cls").size().rename("件数"))


if __name__ == "__main__":
    main()
