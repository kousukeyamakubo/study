# AtlasDemoKitApp の `.cam` を連番 JPEG に展開する。
#
# 【なぜ動画に変換しないか】
# `.cam` の中身はもともと JPEG なので、バイト列をそのまま書き出せば無劣化で取り出せる。
# mp4 に再エンコードすると YOLO に入る前に一度画質を落とすことになる。
#
# 【フォーマット】詳細は 0817/camera_capture.md。要点だけ:
#   .NET BinaryFormatter の List<MemoryStream> で、各要素が JPEG 1 枚。
#   1 要素 = レーダー 1 フレーム。要素は固定長スロットに置かれ後ろはゼロ埋め。
#   タイムスタンプは入っていないので、時刻は frame index × frame periodicity で作る。
#
# 依存: 標準ライブラリのみ（--mp4 を使うときだけ opencv-python）
#
# 使い方:
#   python cam_extract.py atlas_log_20260816_154056.cam
#   python cam_extract.py atlas_log.cam.gz --out frames/ --dat atlas_log.dat
#   python cam_extract.py atlas_log.cam --mp4          # 目視確認用の動画も出す

import argparse
import gzip
import re
from pathlib import Path

# レーダーの frame periodicity [s]（既定 200 ms）。動画の fps はこれの逆数にする。
# こうすると再生時間が実時間と一致する（50 フレーム → 10 秒）
FRAME_PERIOD_S = 0.2

# JPEG の SOI / EOI。スロット内でこの区間だけを取り出す
SOI = b"\xff\xd8\xff"
EOI = b"\xff\xd9"

# .dat 1 フレームのバイト長（0727/atlas_dat_format.md）。フレーム数の突き合わせに使う
DAT_HEADER_BYTES = 256
DAT_FRAME_BYTES = 65536


def read_cam(path: Path) -> bytes:
    """`.cam` を読む。`.gz` で固めてあってもそのまま扱える"""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def split_jpegs(raw: bytes) -> list[bytes]:
    """`.cam` のバイト列を JPEG のリストに分解する。

    シリアライズを解釈せずマーカー走査で切る。スロットの境界を次の SOI で区切り、
    その範囲内の EOI までを 1 枚とする。区間を限ることで、後続スロットの EOI を
    誤って拾うことがない（実測 50 枚で最初/最後の EOI が完全一致）"""
    starts = [m.start() for m in re.finditer(SOI, raw)]
    bounds = starts[1:] + [len(raw)]
    out = []
    for s, nxt in zip(starts, bounds):
        e = raw.find(EOI, s, nxt)
        if e < 0:                       # EOI が無い = 書き込み途中で切れたスロット
            continue
        out.append(raw[s:e + 2])
    return out


def write_mp4(jpegs: list[bytes], out: Path, fps: float, scale: float = 1.0) -> None:
    """目視確認用の動画を書き出す。

    【注意】ここは再エンコードなので画質が落ちる。YOLO に入れるのは連番 JPEG のほう。
    この動画は「何が写っているか」「フレームが飛んでいないか」を人が見るためだけのもの"""
    import cv2
    import numpy as np

    frames = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]
    h, w = frames[0].shape[:2]                       # (H, W, 3)
    w, h = int(w * scale), int(h * scale)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.resize(f, (w, h)) if scale != 1.0 else f)
    vw.release()


def dat_frame_count(dat: Path) -> int:
    """`.dat` のフレーム数。`.cam` の枚数と一致すべき値"""
    return (dat.stat().st_size - DAT_HEADER_BYTES) // DAT_FRAME_BYTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="出力ディレクトリ（既定: <cam名>_frames/）")
    ap.add_argument("--dat", type=Path, default=None,
                    help="対応する .dat。フレーム数の一致を検証する")
    ap.add_argument("--mp4", action="store_true",
                    help="目視確認用の動画も出す（再エンコードするので YOLO には使わない）")
    ap.add_argument("--dt", type=float, default=FRAME_PERIOD_S,
                    help="フレーム間隔[s]。動画の fps = 1/dt")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="動画のみ縮小して書き出す（4K が重いとき 0.25 など）")
    args = ap.parse_args()

    raw = read_cam(args.cam)
    jpegs = split_jpegs(raw)

    stem = args.cam.name.removesuffix(".gz").removesuffix(".cam")
    out = args.out or args.cam.parent / f"{stem}_frames"
    out.mkdir(parents=True, exist_ok=True)
    # ファイル名の連番 = レーダーのフレーム index。ソート順がそのまま時系列になる
    for i, jpg in enumerate(jpegs):
        (out / f"{i:05d}.jpg").write_bytes(jpg)

    sizes = [len(j) for j in jpegs]
    print(f"{args.cam.name}: {len(jpegs)} 枚 → {out}")
    print(f"  JPEG 1枚 {min(sizes)/1024:.0f}〜{max(sizes)/1024:.0f} KB "
          f"(平均 {sum(sizes)/len(sizes)/1024:.0f} KB)")

    if args.mp4:
        mp4 = out.parent / f"{stem}.mp4"
        write_mp4(jpegs, mp4, fps=1.0 / args.dt, scale=args.scale)
        print(f"  動画 {mp4.name}: {1/args.dt:.1f} fps, "
              f"{len(jpegs)*args.dt:.1f} 秒（実時間と一致）")

    # フレーム数の一致は index 対応が保てているかの唯一の検査。崩れたら映像は使えない
    if args.dat:
        n_dat = dat_frame_count(args.dat)
        ok = n_dat == len(jpegs)
        print(f"  .dat フレーム数 {n_dat} と {'一致' if ok else '★不一致★'}")
        if not ok:
            print("  → カメラ側が間引かれている。index 対応が崩れるので解像度を下げて撮り直す")

    print(f"\n次: python detect_yolo.py {out} --dt 0.2 --out {stem}.detections.csv")


if __name__ == "__main__":
    main()
