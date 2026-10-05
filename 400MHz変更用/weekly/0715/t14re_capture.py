# TITAN T14RE 生IQデータ取り込みスクリプト（ドラフト・実機未検証）
#
# 根拠ドキュメント: 0715/hw_docs/ の「Pythonサンプルコードスタートガイド」
# 「ソフトウェア説明書」。仕様の要約は docs/hardware.md を参照。
#
# 使い方（例）:
#   python t14re_capture.py --port COM3 --cam-index 1 \
#       --cfg T14RE_2R5D_Short.cfg --out captures/ --num-data 10
#
# ポート番号とキャプチャデバイス index は付属ツールで確認:
#   t14retool -deviceInfo
#
# 依存: pyserial, numpy, opencv-python（要インストール確認）

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import serial

# プリセット Cfg ごとのデータ構造（Cfg説明書より）。
# データにはヘッダが無く、reshape に必要な次元数はホスト側で管理する必要がある
PRESETS = {
    # (Frame, Chirpset, Tx, Rx, Sample)
    "Short":  dict(frame=1,  chirpset=16, tx=3, rx=4, sample=256),
    "Mid":    dict(frame=1,  chirpset=16, tx=3, rx=4, sample=256),
    "100fps": dict(frame=10, chirpset=4,  tx=3, rx=4, sample=256),
    "500fps": dict(frame=50, chirpset=1,  tx=4, rx=4, sample=256),
}


def send_cfg(port: str, cfg_path: Path, baud: int = 115200, timeout_s: float = 5.0):
    """Cfg ファイルを1行ずつ MMIC コマンドポートへ送信する（公式 T14reSample.py 準拠）。

    応答はプロンプト "mmwDemo:/>" まで読み切ってから次行を送る。
    空行・コメント行(%)・sensorReset は "Done" を返さない仕様のため、
    Done 確認はそれ以外の行に限定する（全行で待つとタイムアウトする）。
    """
    with serial.Serial(port, baud, timeout=timeout_s) as ser:
        ser.reset_output_buffer()
        ser.reset_input_buffer()
        for line in cfg_path.read_text().splitlines():
            cmd = line.strip()
            ser.write((cmd + "\n").encode())
            resp = ser.read_until(b"\nmmwDemo:/>")
            if cmd and not cmd.startswith("%") and cmd != "sensorReset":
                rets = resp.decode(errors="replace").split("\n")
                if len(rets) < 2 or rets[-2].strip() != "Done":
                    raise RuntimeError(f"Cfg送信でエラー応答: {cmd}\n応答: {resp!r}")
                print(f"  OK: {cmd}")
    print("Cfg送信完了（センサー自動起動）")


def open_capture(cam_index: int, dims: dict) -> cv2.VideoCapture:
    """UVC（カメラ）として見えるデータIFを開く。

    Windows では MSMF バックエンド必須（DirectShow は RGB 自動変換が
    無効化できずデータが壊れる）。CONVERT_RGB=0 / FORMAT=-1 も必須。
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    # 転送画像サイズ: 幅 = Rx × Sample × 2 [pixel]、高さ = Frame × Chirpset × Tx [line]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, dims["sample"] * 2 * dims["rx"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dims["tx"] * dims["chirpset"] * dims["frame"])
    if not cap.isOpened():
        raise RuntimeError(f"キャプチャデバイス {cam_index} を開けない")
    return cap


def decode_raw(buf: np.ndarray, dims: dict) -> np.ndarray:
    """int16 バッファを複素IQ配列に整形する。

    次元順（遅い→速い）: Frame → Chirpset → Tx → Rx → Sample → (I, Q)
    RAW 出力は I が先、Q が後（FFT出力モードは 虚→実 の順で異なるので注意）。
    """
    d = dims
    iq = buf.astype(np.int16).reshape(
        d["frame"], d["chirpset"], d["tx"], d["rx"], d["sample"], 2
    )
    # (Frame, Chirpset, Tx, Rx, Sample) complex64
    return (iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32))


def quick_rd_map(x: np.ndarray, tx: int = 0) -> np.ndarray:
    """パイロット計測の定性確認用の簡易RDマップ。

    学習時と同一の処理（Cyclist_env_RDA_2nano.py 系）に置き換える前の
    暫定版。Doppler 軸はチャープセット方向（同一TXの繰り返し、PRI 660µs）。
    """
    # (Chirpset, Rx, Sample) — 先頭フレーム・指定TXのみ
    z = x[0, :, tx, :, :]
    r = np.fft.fft(z, axis=-1)                      # レンジFFT → (Chirpset, Rx, Range)
    rd = np.fft.fftshift(np.fft.fft(r, axis=0), 0)  # DopplerFFT → (Doppler, Rx, Range)
    # Rx 方向は位相を捨てて電力を非コヒーレント加算（角度は初回計測では扱わない）
    return np.abs(rd).sum(axis=1)                   # (Doppler, Range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="MMICコマンドIFのCOMポート（例 COM3）")
    ap.add_argument("--cam-index", type=int, required=True, help="データIFのキャプチャデバイスindex")
    ap.add_argument("--cfg", type=Path, required=True, help="RAW出力用Cfgファイル（_fft無し版）")
    ap.add_argument("--preset", choices=PRESETS, default="Short")
    ap.add_argument("--out", type=Path, default=Path("captures"))
    ap.add_argument("--num-data", type=int, default=10, help="取得するデータ数（1データ=100ms）")
    args = ap.parse_args()

    dims = PRESETS[args.preset]
    args.out.mkdir(parents=True, exist_ok=True)

    send_cfg(args.port, args.cfg)
    cap = open_capture(args.cam_index, dims)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    try:
        while saved < args.num_data:
            ret, data = cap.read()
            if not ret:
                print("read失敗、リトライ")
                continue
            buf = np.frombuffer(data.tobytes(), dtype="<i2")
            # 起動直後は All-0 データが来ることがあるため捨てる
            if not buf.any():
                print("All-0フレームをスキップ")
                continue
            x = decode_raw(buf, dims)  # (Frame, Chirpset, Tx, Rx, Sample)
            np.save(args.out / f"iq_{stamp}_{saved:04d}.npy", x)
            saved += 1
            print(f"保存 {saved}/{args.num_data}  max|IQ|={np.abs(x).max():.0f}")
    finally:
        cap.release()

    # 取得条件をデータと並べて残す（データ自体にメタデータが無いため必須）
    meta = dict(preset=args.preset, cfg=str(args.cfg), dims=dims,
                timestamp=stamp, num_data=saved)
    (args.out / f"meta_{stamp}.json").write_text(json.dumps(meta, indent=2))
    print(f"完了: {args.out} に {saved} データ + メタ情報を保存")


if __name__ == "__main__":
    main()
