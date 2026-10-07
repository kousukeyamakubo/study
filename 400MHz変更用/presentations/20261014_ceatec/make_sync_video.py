# 10/5 の 6 本について、カメラ映像・Range Profile・Range-Angle マップを同期させた動画を作る（CEATEC 発表の素材）。
#
# 1 フレーム = レーダー 1 フレーム（0.2 s）= カメラ 1 枚（.cam と .dat は 1:1 で対応、weekly/0908 で確認済み）。
# 再生は 5 fps で実時間と一致させる。
# MTI なし／あり、日本語／英語で別の動画にする（6 本 × 2 × 2 = 24 本）。
# あわせて、目標が最も強いフレームの静止画も出す（スライドに貼る用）。
#
# 量の定義:
#   Range Profile : 8ch × 全ドップラービンの電力和 [dB]（(F, R)）。薄い線 = その走行の時間中央値（静止物の背景）
#   Range-Angle   : 全ドップラービンで素子方向 FFT（to_ra、TDM 位相補正込み）した電力の和 [dB]（(F, A, R)）
#   MTI あり = チャープ平均を引いた RD（atlas_export.py の既定）。MTI なし = --no-mti
# Range-Angle は扇形（x = R sin θ, y = R cos θ）で描く。R は斜距離なので、地面の上の位置そのものではない。
# 角度は「+ を右」に描く。接近する目標は角度が + → − に動き、カメラ映像の中では右 → 左に動くので、
# この向きでカメラと左右がそろう（10/5 の接近 3 本で確認。角度の符号の定義そのものは未確認）。
#
# 使い方:
#   cd presentations/20261014_ceatec
#   python make_sync_video.py              # 全 24 本
#   python make_sync_video.py --only 3     # 番号 3 だけ（試し用）
#   python make_sync_video.py --jobs 4     # 並列数（既定 6）

import argparse
import json
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
D1005 = ROOT / "weekly" / "1013" / "1005"
sys.path.insert(0, str(ROOT / "weekly" / "0727"))
sys.path.insert(0, str(ROOT.parent / "camera_pipeline" / "0817" / "capture"))
from atlas_export import angle_axis_deg, to_ra   # noqa: E402
from cam_extract import read_cam, split_jpegs     # noqa: E402

from common import FIG, LABELS, RUNS, setup_font   # noqa: E402

N_ANT = 256
ANG = angle_axis_deg(N_ANT)                 # (A,)
A_LIM = 64.0                                # |θ| > 64°（|u| > 0.9）は素子の指向性で弱いので描かない
R_SHOW = (8.0, 50.0)                        # 描く距離。走路は R 約 13〜49 m。R < 8 m は窓枠などの直近の反射で、色の範囲を食うので外す
FPS = 5.0                                   # = 1 / frame period。実時間で再生する
CAM_W, CAM_H = 1280, 720                    # カメラは 4K なので縮小してから貼る
DR_SHOW = 45.0                              # Range-Angle の色の幅 [dB]（上限から下へ）。MTI ありの目標（MTI なしの上限より 15〜20 dB 下）が床から浮く幅


def load(tag, folder, nomti):
    """レーダーの (Range Profile, Range-Angle, 距離軸, 時刻) を返す"""
    s = f"atlas_log_20261005_{tag}"
    npz = D1005 / "data" / ("export_nomti" if nomti else "export") / s / f"{s}_rd.npz"
    d = np.load(npz)
    rd, rng, t = d["rd"], d["range_m"], d["t_rel_s"]                  # rd: (F, TX, RX, D, R)
    hdr = json.loads(str(d["header"]))
    prof = 10 * np.log10((np.abs(rd) ** 2).sum(axis=(1, 2, 3)) + 1e-12)   # (F, R)
    ra = np.stack([(np.abs(to_ra(rd[f], hdr, rd.shape[-2], N_ANT)) ** 2).sum(axis=0)
                   for f in range(len(rd))])                           # (F, A, R)
    return prof, 10 * np.log10(ra + 1e-12), rng, t


def load_camera(tag, folder):
    """カメラ画像のリスト（BGR, CAM_H × CAM_W）"""
    jpgs = split_jpegs(read_cam(D1005 / folder / f"atlas_log_20261005_{tag}.cam.gz"))
    return [cv2.resize(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), (CAM_W, CAM_H),
                       interpolation=cv2.INTER_AREA) for j in jpgs]


def peak_frame(tag, folder):
    """目標が最も強いフレーム（MTI ありの Range Profile、R 15〜45 m の最大）。静止画に使う"""
    prof, _, rng, _ = load(tag, folder, nomti=False)
    sel = (rng >= 15) & (rng <= 45)
    return int(prof[:, sel].max(axis=1).argmax())


class Frame:
    """1 本ぶんの図。軸は 1 回だけ作り、フレームごとに中身だけ差し替える（速さのため）"""

    def __init__(self, lab, run_title, kind, rng, ra_lim, prof_lim):
        # 16:9（1920 × 1080）。上段 = カメラと Range-Angle（どちらも横長 約 1.75:1）、下段 = Range Profile を横長に
        self.fig = plt.figure(figsize=(16, 9), dpi=120)
        self.ax_cam = self.fig.add_axes([0.012, 0.36, 0.50, 0.55])
        self.ax_ra = self.fig.add_axes([0.555, 0.36, 0.38, 0.55])
        cax = self.fig.add_axes([0.945, 0.40, 0.012, 0.47])
        self.ax_pr = self.fig.add_axes([0.06, 0.08, 0.875, 0.21])
        self.fig.text(0.012, 0.955, f"{run_title}   [{lab['kind'][kind]}]", fontsize=20, va="center")

        self.ax_cam.set_axis_off()
        self.im_cam = self.ax_cam.imshow(np.zeros((CAM_H, CAM_W, 3), np.uint8))
        self.ax_cam.set_title(lab["camera"], fontsize=13)
        self.txt_t = self.ax_cam.text(0.015, 0.97, "", transform=self.ax_cam.transAxes, va="top",
                                      fontsize=15, color="w", bbox=dict(fc="k", alpha=0.55, lw=0))

        rsel = (rng >= R_SHOW[0]) & (rng <= R_SHOW[1])
        self.rsel = rsel
        self.bg, = self.ax_pr.plot(rng[rsel], np.zeros(rsel.sum()), color="#9a9a9a", lw=1.5,
                                   label=lab["bg"])
        self.ln, = self.ax_pr.plot(rng[rsel], np.zeros(rsel.sum()), color="#2a78d6", lw=2.0,
                                   label=lab["now"])
        self.ax_pr.set_xlim(*R_SHOW)
        self.ax_pr.set_ylim(*prof_lim)
        self.ax_pr.set_xlabel(lab["range"], fontsize=12)
        self.ax_pr.set_ylabel(lab["power"], fontsize=12)
        self.ax_pr.set_title(lab["profile"], fontsize=13)
        self.ax_pr.grid(alpha=0.25, lw=0.6)
        self.ax_pr.legend(loc="upper right", fontsize=10, frameon=False, ncol=2)

        asel = np.abs(ANG) <= A_LIM
        self.asel = asel
        th = np.radians(ANG[asel])
        rr, tt = np.meshgrid(rng[rsel], th, indexing="ij")             # (R', A')
        self.mesh = self.ax_ra.pcolormesh(rr * np.sin(tt), rr * np.cos(tt), np.zeros(rr.shape),
                                          shading="gouraud", cmap="viridis", vmin=ra_lim[0], vmax=ra_lim[1])
        # 等距離の円弧と角度の線（読み取りの目安）
        for r0 in (10, 20, 30, 40, 50):
            a = np.radians(np.linspace(-A_LIM, A_LIM, 100))
            self.ax_ra.plot(r0 * np.sin(a), r0 * np.cos(a), color="w", lw=0.5, alpha=0.45)
            # 距離の目盛りは 0° の線の右に置く（扇の端に置くと角度の目盛りと重なる）
            self.ax_ra.text(0.6, r0 + 0.4, f"{r0} m", fontsize=9, color="w", alpha=0.8, va="bottom")
        for a0 in (-60, -30, 0, 30, 60):
            a = np.radians(a0)
            self.ax_ra.plot([0, R_SHOW[1] * np.sin(a)], [0, R_SHOW[1] * np.cos(a)], color="w", lw=0.5, alpha=0.45)
            self.ax_ra.text(1.05 * R_SHOW[1] * np.sin(a), 1.05 * R_SHOW[1] * np.cos(a), f"{a0:+d}°" if a0 else "0°",
                            fontsize=10, color="#444", ha="center", va="center")
        self.ax_ra.set_aspect("equal")
        self.ax_ra.set_xlim(-50, 50)
        self.ax_ra.set_ylim(0, 56)
        self.ax_ra.set_xlabel(lab["cross"], fontsize=12)
        self.ax_ra.set_ylabel(lab["down"], fontsize=12)
        self.ax_ra.set_title(lab["ra"], fontsize=13)
        cb = self.fig.colorbar(self.mesh, cax=cax)
        cb.set_label(lab["power_rel"], fontsize=11)
        self.fig.text(0.985, 0.012, lab["note"], ha="right", fontsize=10, color="#555")

    def update(self, cam, t, prof, bg, ra):
        self.im_cam.set_data(cam[:, :, ::-1])                          # BGR → RGB
        self.txt_t.set_text(f"t = {t:4.1f} s")
        self.ln.set_ydata(prof[self.rsel])
        self.bg.set_ydata(bg[self.rsel])
        self.mesh.set_array(ra[self.asel][:, self.rsel].T)             # (R', A')

    def to_bgr(self):
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def open_writer(path, size):
    # avc1（H.264）は PowerPoint でそのまま再生できる。opencv の ffmpeg 版では作れないが、
    # Windows の Media Foundation に切り替わって作られる。作れない環境では mp4v に落とす。
    # Media Foundation はパスに日本語（このリポジトリの「400MHz変更用」）があると開けないので、
    # ASCII だけの一時ファイルに書いて、閉じた後に移す（write_video）
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, size)
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    return vw


def write_video(frames, out):
    """BGR 画像の列を mp4 に書く。一時フォルダに書いてから out に移す"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "v.mp4"
        vw = None
        for img in frames:
            h, w = img.shape[:2]
            h, w = h - h % 2, w - w % 2                                # H.264 は縦横とも偶数
            if vw is None:
                vw = open_writer(tmp, (w, h))
            vw.write(np.ascontiguousarray(img[:h, :w]))
        vw.release()
        shutil.move(str(tmp), str(out))


def render_one(job):
    """1 走行 × MTI なし／あり の 1 組ぶん（日本語・英語の 2 本と静止画）を作る。並列に呼ばれる"""
    no, kind, lim = job
    _, folder, tag, scen, direc = next(r for r in RUNS if r[0] == no)
    out_v = HERE / "video"
    prof, ra, rng, t = load(tag, folder, nomti=(kind == "nomti"))
    cams = load_camera(tag, folder)
    k_peak = peak_frame(tag, folder)
    assert len(cams) == len(prof), f"{tag}: カメラ {len(cams)} 枚とレーダー {len(prof)} フレームが合わない"
    bg = np.median(prof, axis=0)                                       # (R,)
    msgs = []
    for lang in ("ja", "en"):
        setup_font(lang)
        lab = LABELS[lang]
        title = f"{no}. {lab['scen'][scen]}（{lab['dir'][direc]}）" if lang == "ja" else                 f"{no}. {lab['scen'][scen]} ({lab['dir'][direc]})"
        fr = Frame(lab, title, kind, rng, lim["ra"], lim["prof"])
        stem = f"{no}_{scen}_{direc}_{kind}_{lang}"

        def frames():
            for f in range(len(prof)):
                fr.update(cams[f], t[f], prof[f], bg, ra[f])
                img = fr.to_bgr()
                if f == k_peak:
                    # cv2.imwrite はパスに日本語があると黙って失敗するので、エンコードしてから書く
                    (FIG / f"still_{stem}.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
                yield img

        write_video(frames(), out_v / f"{stem}.mp4")
        plt.close(fr.fig)
        msgs.append(f"-> video/{stem}.mp4  (静止画 t = {t[k_peak]:.1f} s)")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="番号（1〜6）を 1 つだけ作る")
    ap.add_argument("--jobs", type=int, default=6,
                    help="並列数。時間の大半は matplotlib の描画（1 コアで動く）なので、組（走行 × MTI の有無）ごとに分けて並列にする")
    args = ap.parse_args()
    (HERE / "video").mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    # 色の範囲（Range-Angle）と縦軸（Range Profile）は、MTI なし／あり・6 本すべてで共通にする。
    # 並べて見たときに、色や高さの違いがそのまま dB の違いになるように（MTI の有無で dB の基準は同じ）。
    # 共通にすると、目標（MTI ありの山）が静止物（MTI なしの山）より弱いことがそのまま見える
    data = {}
    for no, folder, tag, *_ in RUNS:
        for kind in ("nomti", "mti"):
            data[no, kind] = load(tag, folder, nomti=(kind == "nomti"))
    # 描く範囲（R_SHOW、|θ| ≤ A_LIM）だけで決める
    rsel = (data[1, "mti"][2] >= R_SHOW[0]) & (data[1, "mti"][2] <= R_SHOW[1])
    asel = np.abs(ANG) <= A_LIM
    prof_all = np.concatenate([data[k][0][:, rsel].ravel() for k in data])
    ra_all = np.concatenate([data[k][1][:, asel][:, :, rsel].ravel() for k in data])
    hi = np.percentile(ra_all, 99.9)
    lim = dict(ra=(hi - DR_SHOW, hi),
               prof=(np.percentile(prof_all, 1) - 3, np.percentile(prof_all, 99.99) + 3))
    print(f"Range-Angle {lim['ra'][0]:.1f}〜{lim['ra'][1]:.1f} dB, "
          f"Range Profile {lim['prof'][0]:.1f}〜{lim['prof'][1]:.1f} dB（MTI なし／ありで共通）")
    del data

    jobs = [(no, kind, lim) for no, *_ in RUNS if not args.only or no == args.only for kind in ("nomti", "mti")]
    with Pool(min(args.jobs, len(jobs))) as pool:
        for msgs in pool.imap(render_one, jobs):
            print("\n".join(msgs), flush=True)


if __name__ == "__main__":
    main()
