# CPU / CUDA で YOLO 推論がどれだけ速くなるかの計測。
#
# 【なぜ detect_yolo.py に --device を足すのではなく別スクリプトにしたか】
# 知りたいのは「どの区間が速くなるか」であって検出結果ではない。ultralytics は
# device 未指定なら CUDA を自動で使うので、本番の detect_yolo.py 側に手を入れる
# 必要はない。ここは計測専用の使い捨て（結果が出たら README に数値だけ残す）。
#
# 【predict と track を分けて測る理由】
# GPU で速くなるのは推論だけで、ByteTrack の対応付け（lap の線形割当）と NMS 後処理は
# CPU 側に残る。パイプライン全体の短縮率は推論単体の短縮率より必ず小さくなるので、
# 「推論だけ」と「追跡込み（本番と同じ経路）」の両方を出す。
#
# 【フレームを先にメモリへ載せる理由】
# .cam 由来の連番 JPEG は 1 枚のデコードが数 ms かかる。GPU で推論が速くなると
# デコードが支配的になり、計測がストレージ・CPU 律速になってしまう。純粋な
# 推論時間を見たいので、デコードは計測ループの外に出す。
#
# 使い方（デスクトップ側、camera_pipeline/ 直下から）:
#   .venv\Scripts\python.exe 0910/bench/bench_device.py frames/ --devices cpu,cuda:0
#   .venv\Scripts\python.exe 0910/bench/bench_device.py video.mp4 --devices cuda:0 --half
#   .venv\Scripts\python.exe 0910/bench/bench_device.py --synthetic 3840x2160  # 実データ無しの目安

import argparse
import statistics
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def report_env():
    """torch が CPU ビルドかどうかを最初に晒す。ここが +cpu だと GPU は使われない"""
    import torch
    print(f"torch          : {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda.is_available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} = {p.name}, {p.total_memory / 1024**3:.1f} GiB, "
                  f"SM {p.major}.{p.minor}")
    elif "+cpu" in torch.__version__:
        print("  ※ +cpu ビルドです。GPU があっても使えません（README 2.5 の導入手順のまま）")
    print()


def load_frames(source: Path | None, n: int, synthetic: tuple[int, int] | None):
    """計測に使うフレームを ndarray のリストで返す（デコードを計測外に出すため）"""
    import numpy as np

    if synthetic is not None:
        w, h = synthetic
        # 一様乱数だと低信頼度の箱が大量に出て NMS が重くなり、実データと傾向がずれる。
        # 緩やかな勾配＋弱いノイズにして箱がほぼ出ない状態で測る（＝推論コストの下限）
        rng = np.random.default_rng(0)
        base = np.linspace(40, 200, h, dtype=np.uint8)[:, None, None]
        base = np.repeat(np.repeat(base, w, axis=1), 3, axis=2)
        frames = [np.clip(base + rng.integers(-8, 8, (h, w, 3)), 0, 255).astype(np.uint8)
                  for _ in range(n)]
        print(f"合成フレーム: {w}x{h} x {n} 枚（検出はほぼ出ないので推論コストの下限側）")
        return frames

    import cv2
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)[:n]
        if not files:
            raise FileNotFoundError(f"{source} に画像がありません")
        frames = [cv2.imread(str(p)) for p in files]
    else:
        cap = cv2.VideoCapture(str(source))
        frames = []
        while len(frames) < n:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
        cap.release()
        if not frames:
            raise FileNotFoundError(f"{source} からフレームが読めません")
    h, w = frames[0].shape[:2]
    print(f"入力: {source} → {w}x{h} x {len(frames)} 枚をメモリに展開")
    return frames


def sync(device: str):
    """GPU は非同期実行なので、同期しないと計測が投入時間になってしまう"""
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def bench(frames, device: str, mode: str, model_name: str, conf: float,
          imgsz: int, half: bool, warmup: int):
    """1 デバイス・1 モード分を計測して統計を返す"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "0817" / "lib"))
    from detections import COCO_BICYCLE, COCO_MOTORCYCLE, COCO_PERSON, VEHICLE_COCO
    from ultralytics import YOLO

    wanted = sorted({COCO_PERSON, COCO_BICYCLE, COCO_MOTORCYCLE} | VEHICLE_COCO)
    # track は内部状態を持つので、モード・デバイスごとに作り直してトラッカーを初期化する
    model = YOLO(model_name)
    # ultralytics 8.4.120 で half は deprecated になり、渡すと推論のたびに警告が出る。
    # 後継の quantize はビット幅指定で 16=FP16、未指定(None)=FP32。None は cfg 検証で
    # 素通りするので、FP32 側は「渡さない」ではなく None で表現してよい
    kw = dict(conf=conf, imgsz=imgsz, classes=wanted, device=device,
              quantize=16 if half else None, verbose=False)
    call = ((lambda im: model.track(im, persist=True, tracker="bytetrack.yaml", **kw))
            if mode == "track" else (lambda im: model.predict(im, **kw)))

    # 初回は重み転送・cuDNN のアルゴリズム選択が入るので捨てる
    for i in range(warmup):
        call(frames[i % len(frames)])
    sync(device)

    per_frame, n_box = [], 0
    parts = {"preprocess": [], "inference": [], "postprocess": []}
    for im in frames:
        t0 = time.perf_counter()
        res = call(im)[0]
        sync(device)
        per_frame.append((time.perf_counter() - t0) * 1e3)
        n_box += 0 if res.boxes is None else len(res.boxes)
        for k in parts:
            parts[k].append(res.speed[k])
    return dict(ms_median=statistics.median(per_frame),
                ms_mean=statistics.fmean(per_frame),
                ms_p90=sorted(per_frame)[int(len(per_frame) * 0.9) - 1],
                parts={k: statistics.fmean(v) for k, v in parts.items()},
                n_box=n_box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, nargs="?", default=None,
                    help="動画または連番画像ディレクトリ。--synthetic なら省略可")
    ap.add_argument("--synthetic", default=None, metavar="WxH",
                    help="実データ無しで測る場合の解像度（例 1920x1080）")
    ap.add_argument("--devices", default="cpu,cuda:0",
                    help="比較するデバイス（カンマ区切り）")
    ap.add_argument("--frames", type=int, default=60, help="計測フレーム数")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="本番と同じ 1280。640 との差も見るなら別途実行する")
    ap.add_argument("--half", action="store_true",
                    help="CUDA で FP16。CPU には効かないので cuda のみに適用")
    ap.add_argument("--modes", default="predict,track")
    args = ap.parse_args()

    if args.source is None and args.synthetic is None:
        ap.error("source か --synthetic のどちらかが必要です")

    report_env()
    syn = None
    if args.synthetic:
        w, h = args.synthetic.lower().split("x")
        syn = (int(w), int(h))
    frames = load_frames(args.source, args.frames, syn)
    print(f"設定: model={args.model}, imgsz={args.imgsz}, conf={args.conf}, "
          f"warmup={args.warmup}\n")

    rows = {}
    for mode in args.modes.split(","):
        for dev in args.devices.split(","):
            half = args.half and dev.startswith("cuda")
            tag = f"{dev}{'+fp16' if half else ''}"
            try:
                r = bench(frames, dev, mode, args.model, args.conf,
                          args.imgsz, half, args.warmup)
            except Exception as e:                      # GPU 無し環境で cuda を指定した等
                print(f"[{mode}] {tag}: 失敗 — {type(e).__name__}: {e}")
                continue
            rows[(mode, tag)] = r
            p = r["parts"]
            print(f"[{mode}] {tag:12s} {r['ms_median']:7.1f} ms/frame "
                  f"({1e3 / r['ms_median']:5.1f} fps)  p90 {r['ms_p90']:6.1f} ms  "
                  f"pre {p['preprocess']:5.1f} / inf {p['inference']:6.1f} / "
                  f"post {p['postprocess']:5.1f} ms  箱 {r['n_box']}")

    # CPU を基準に短縮率を出す。predict と track で率が違うことが見たい点
    print("\n--- CPU 比 ---")
    for mode in args.modes.split(","):
        base = rows.get((mode, "cpu"))
        if base is None:
            continue
        for (m, tag), r in rows.items():
            if m != mode or tag == "cpu":
                continue
            print(f"[{mode}] {tag}: 全体 {base['ms_median'] / r['ms_median']:.2f}x, "
                  f"推論のみ {base['parts']['inference'] / r['parts']['inference']:.2f}x")
    print("\n※ 箱の数がデバイス間でずれる場合、FP16 で検出が変わっている"
          "（CSV はバイト一致しなくなる）")


if __name__ == "__main__":
    main()
