# .dat の解釈が正しいことを、その .dat 自身で反証可能な形に落として検査する
#
# 【なぜ必要か】
# atlas_dat_format.md のヘッダオフセットは非公開フォーマットの推定であり、
# 「そう読んだら妥当な値が出た」だけでは循環論法になる。そこで、推定に使っていない
# 独立な量どうしの一致を見る。ヘッダには冗長なフィールド（max_vel, vel_res, range_res,
# max_range）があり、これらは他のフィールドから計算できる。**もしオフセットの推定が
# 間違っていれば、この冗長性は成立しない**。したがって一致は推定の裏付けになる。
#
# 各検査は「失敗しうる」ように書く。通ることではなく、落ちたら解釈が誤りだと分かることが目的。
#
# 使い方:
#   python atlas_provenance_check.py dat/atlas_log_20260729_173342.dat

import argparse
import struct
from pathlib import Path

import numpy as np

from atlas_dat_parse import HEADER_BYTES, load_frames, parse_header
from atlas_export import C0, chirp_rate, pri_s, true_range_res_m

TOL = 1e-3   # 相対誤差の許容。ヘッダは double なので本来は小数9桁まで合う


def check(name: str, got: float, want: float, unit: str, basis: str, tol: float = TOL):
    rel = abs(got - want) / max(abs(want), 1e-12)
    ok = rel < tol
    print(f"  [{'OK ' if ok else 'NG '}] {name}")
    print(f"         ヘッダ {want:.6f} {unit}  vs  再計算 {got:.6f} {unit}  "
          f"相対誤差 {rel:.2e}")
    print(f"         根拠: {basis}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat", type=Path)
    args = ap.parse_args()

    raw = args.dat.read_bytes()
    hdr = parse_header(raw[:HEADER_BYTES])
    print(f"===== {args.dat.name} =====")

    results = []

    # --- 1. マジックワード（フォーマット同定） -----------------------------
    magic = struct.unpack_from("<4H", raw, 0)
    ok = magic == (0x1022, 0x3040, 0x5060, 0x7080)
    print(f"\n1. マジックワード")
    print(f"  [{'OK ' if ok else 'NG '}] {tuple(hex(m) for m in magic)}")
    print(f"         根拠: 0727 と同一。DemoKitApp 系である同定に使う")
    results.append(ok)

    # --- 2. ヘッダ内部の冗長性 -------------------------------------------
    # ここが本体。max_vel / vel_res / range_res / max_range は他フィールドから決まる。
    # オフセット推定が誤っていればまず合わない。
    print(f"\n2. ヘッダの冗長フィールドが他フィールドから再現できるか")
    pri = pri_s(hdr)                            # chirp_interval(0x3C) × n_tx(0x34)
    lam = C0 / (hdr["start_freq_ghz"] * 1e9)    # start_freq(0x0C)
    n_dop = hdr["n_chirp_per_frame"] // hdr["n_tx"]

    results.append(check(
        "max_vel = λ/(4·PRI)", lam / (4 * pri), hdr["max_vel_ms"], "m/s",
        "0x3C(chirp_interval)・0x34(n_tx)・0x0C(start_freq) から。"
        "λ/(4·PRI) は TDM の最大明確速度"))

    results.append(check(
        "vel_res = λ/(2·N_dop·PRI)", lam / (2 * n_dop * pri), hdr["vel_res_ms"], "m/s",
        "上記に加え 0x20(n_chirp)・0x34(n_tx)。"
        "32チャープを TX2本で割って 16 本という解釈がここで検証される"))

    fs = hdr["sampling_ksps"] * 1e3
    results.append(check(
        "実効帯域 = slope·n_sample/fs", chirp_rate(hdr) * hdr["n_sample"] / fs / 1e9,
        hdr["bw_ghz"], "GHz",
        "0x14(slope)・0x1C(n_sample)・0x24(fs) から。260µs のランプのうち実際に"
        "サンプルした 256µs 分だけが帯域になる（MATLAB のコメントと一致）"))

    n_fft_disp = 4 * hdr["n_sample"]            # 4倍ゼロ埋め（表示刻みの慣習）
    results.append(check(
        "range_res = fs·c0/(2·kf·NFFT)", fs * C0 / (2 * chirp_rate(hdr) * n_fft_disp),
        hdr["range_res_m"], "m",
        "0x24(fs)・0x14(slope)・0x1C(n_sample) から。MATLAB AN24_02 の "
        "vRange=[0:NFFT-1]/NFFT*fs*c0/(2*kf) と同一式"))

    results.append(check(
        "max_range = 255 × range_res", 255 * hdr["range_res_m"], hdr["max_range_m"], "m",
        "0x44 と 0x4C の関係。アプリが先頭256ビンを表示していることの傍証"))

    # --- 3. データ部の長さ -------------------------------------------------
    print(f"\n3. データ部がフレーム長で割り切れるか")
    per_frame = hdr["n_chirp_per_frame"] * hdr["n_rx"] * hdr["n_sample"] * 2
    n_byte = len(raw) - HEADER_BYTES
    n_frame, rem = divmod(n_byte, per_frame)
    ok = rem == 0
    print(f"  [{'OK ' if ok else 'NG '}] {n_byte:,} / {per_frame:,} = {n_frame} 余り {rem}")
    print(f"         根拠: (Chirp,Rx,Sample) 各軸の長さ推定が全部正しくないと割り切れない。"
          f"余り 0 は 4 個の整数の積が一致したことを意味する")
    results.append(ok)

    dur = n_frame * hdr["frame_period_ms"] / 1000
    print(f"  [参考] {n_frame} × {hdr['frame_period_ms']}ms = {dur:.1f} 秒"
          f"（録画長の申告と照合すること）")

    frames = load_frames(args.dat, hdr)

    # --- 4. IF が実数か複素IQか -------------------------------------------
    # I+jQ と解釈してFFTすると、実数信号なら正負の周波数が対称になる。
    print(f"\n4. IF サンプルは実数か（複素IQとして解釈したときの対称性）")
    z = frames[0, 0, 0].astype(np.float64)
    sp = np.abs(np.fft.fft(z[0::2] + 1j * z[1::2]))    # 偶数=I, 奇数=Q と仮定
    n = len(sp)
    pos, neg = sp[1:n // 2], sp[n // 2 + 1:][::-1]
    sym = np.corrcoef(pos, neg)[0, 1]
    ok = sym > 0.9
    print(f"  [{'OK ' if ok else 'NG '}] 正負周波数の相関 {sym:.4f}")
    print(f"         根拠: 実数信号のスペクトルは共役対称なので相関 ≈ 1 になる。"
          f"真の複素IQなら片側に寄り相関は低い。→ rfft の片側だけを使う根拠")
    results.append(ok)

    # --- 5. TX は交互かブロックか -----------------------------------------
    # 同一TXのチャープ列は位相がコヒーレントなので、隣接チャープの位相差は揃う。
    # 誤った並びで抜き出すと TX 間をまたいで位相が飛ぶ。
    print(f"\n5. チャープ軸の並び（TX交互 vs 前半後半ブロック）")
    n_tx = hdr["n_tx"]
    n_half = hdr["n_chirp_per_frame"] // n_tx
    r0 = np.fft.rfft(frames[0, :, 0, :].astype(np.float64), axis=-1)
    peak = int(np.abs(r0).mean(axis=0)[3:].argmax()) + 3   # 送信リークを避けて最強レンジ
    stds = {}
    for label, sl in [("交互 [::2]", slice(None, None, 2)),
                      ("ブロック [:16]", slice(0, n_half))]:
        ph = np.angle(r0[sl, peak])
        stds[label] = float(np.std(np.diff(np.unwrap(ph))))
    best = min(stds, key=stds.get)
    ok = best.startswith("交互")
    print(f"  [{'OK ' if ok else 'NG '}] 隣接チャープ位相差の標準偏差: " +
          "  ".join(f"{k}={v:.3f} rad" for k, v in stds.items()))
    print(f"         → 小さい方が物理的に正しい並び: {best}")
    print(f"         根拠: 同一TXならコヒーレント。TX をまたぐと位相が飛ぶ。"
          f"MATLAB AN24_07 の DataTx1/DataTx2 分割と同じ結論になるかを見ている")
    results.append(ok)

    # --- 6. 真の分解能と表示刻みの比 ---------------------------------------
    print(f"\n6. 真の距離分解能とヘッダ range_res の関係")
    tr = true_range_res_m(hdr)
    ratio = tr / hdr["range_res_m"]
    ok = abs(ratio - 4.0) < 0.01
    bw = chirp_rate(hdr) * hdr["n_sample"] / fs
    print(f"  [{'OK ' if ok else 'NG '}] 実効帯域 {bw/1e6:.1f} MHz → c/2B = {tr:.4f} m"
          f"  = range_res × {ratio:.3f}")
    print(f"         根拠: 比が 4 = ゼロ埋め倍率と一致することが、"
          f"range_res が分解能でなく表示刻みである証拠。MATLAB のコメントと一致")
    results.append(ok)

    # --- 7. 物理上限 -------------------------------------------------------
    print(f"\n7. 最大距離（実数サンプルなので fs/2 で決まる）")
    r_phys = (fs / 2) * C0 / (2 * chirp_rate(hdr))
    n_bin = hdr["n_sample"] // 2
    print(f"  [参考] c·(fs/2)/(2·kf) = {r_phys:.2f} m  = {n_bin} ビン × {tr:.3f} m"
          f" = {n_bin * tr:.2f} m")
    print(f"         ヘッダ max_range {hdr['max_range_m']:.2f} m は表示窓であり物理上限ではない")

    print(f"\n===== {sum(results)} / {len(results)} 項目が OK =====")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
