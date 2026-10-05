# ポスター Experiments 帯用: 代表サンプル 1 件の 2 クラス重ね描き 3D 可視化
# voxel_3d_visualization.py の推論キャッシュを流用し、
# 「両クラスとも peak-GT <=1 voxel・高確信」の無難な代表例を 1 つ選んで描く
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# パスはスクリプト自身の位置基準（リポジトリ規約に合わせる）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "..", "0705meeting", "results", "voxel_records_cache.npz")
OUT = os.path.join(SCRIPT_DIR, "voxel_3d_poster.png")

data = np.load(CACHE, allow_pickle=True)
recs = {"cyclist": list(data["cyclist"]), "vehicle": list(data["vehicle"])}

# sample_idx で両クラスを突き合わせ
by_idx = {}
for label in ("cyclist", "vehicle"):
    for r in recs[label]:
        if r["gt"] is None:
            continue
        by_idx.setdefault(r["sample_idx"], {})[label] = r

# 候補: 両クラスとも peak-GT <=1, peak_prob >=0.95。GT のレンジ差は中庸 (5..15 bin)
cands = []
for idx, d in by_idx.items():
    if "cyclist" not in d or "vehicle" not in d:
        continue
    cy, ve = d["cyclist"], d["vehicle"]
    if cy["gt_dist"] <= 1 and ve["gt_dist"] <= 1 and cy["peak_prob"] >= 0.95 and ve["peak_prob"] >= 0.95:
        r_gap = abs(int(cy["gt"][2]) - int(ve["gt"][2]))
        if 5 <= r_gap <= 15:
            cands.append((idx, cy, ve, r_gap))

# ボクセル総数が候補の中央値に近いもの＝「典型的な見え方」を代表に選ぶ
n_vox = [len(c[1]["probs"]) + len(c[2]["probs"]) for c in cands]
med = np.median(n_vox)
idx_best = int(np.argmin([abs(n - med) for n in n_vox]))
sid, cy, ve, r_gap = cands[idx_best]
print(f"selected sample_idx={sid}, r_gap={r_gap}, "
      f"cy: n={len(cy['probs'])} p={cy['peak_prob']:.2f} dist={cy['gt_dist']:.1f}, "
      f"ve: n={len(ve['probs'])} p={ve['peak_prob']:.2f} dist={ve['gt_dist']:.1f}, "
      f"候補数={len(cands)}")

# ===== 描画 =====
plt.rcParams.update({"font.size": 13})
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection="3d")

mappables = {}
for rec, cmap, label in ((cy, "Blues", "Cyclist"), (ve, "Oranges", "Vehicle")):
    c, p = rec["coords"], rec["probs"]
    # 色の濃さ＝当該クラスの確率（元のミーティング図と同じエンコード）。サイズにも重畳
    mappables[label] = ax.scatter(c[:, 2], c[:, 1], c[:, 0], c=p, cmap=cmap,
                                  vmin=0, vmax=1, s=30 + 130 * p,
                                  alpha=0.9, depthshade=False)
    gt, peak = rec["gt"], rec["peak"]
    ax.scatter([gt[2]], [gt[1]], [gt[0]], marker="*", s=500, color="red",
               edgecolors="black", linewidths=1.0, depthshade=False, zorder=5)
    ax.scatter([peak[2]], [peak[1]], [peak[0]], marker="x", s=150,
               color="black", linewidths=2.5, depthshade=False, zorder=6)

pts = np.vstack([cy["coords"], ve["coords"], cy["gt"][None, :], ve["gt"][None, :]])
pad = 3
ax.set_xlim(pts[:, 2].min() - pad, pts[:, 2].max() + pad)
ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
ax.set_zlim(max(pts[:, 0].min() - 1, -0.5), min(pts[:, 0].max() + 1, 9.5))
ax.zaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Range bin", fontsize=14, labelpad=6)
ax.set_ylabel("Doppler bin", fontsize=14, labelpad=6)
ax.set_zlabel("Azimuth ch", fontsize=14, labelpad=10)
ax.tick_params(labelsize=11)
# 3D 図は bbox_inches="tight" でも z ラベルが切れやすいので右余白を明示的に確保
fig.subplots_adjust(left=0.0, right=0.88, top=0.94, bottom=0.16)

legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue", markersize=11, label="Cyclist voxels"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:orange", markersize=11, label="Vehicle voxels"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="red", markeredgecolor="black", markersize=16, label="Ground truth"),
    Line2D([0], [0], marker="x", color="black", linestyle="none", markersize=10, markeredgewidth=2.5, label="Peak voxel"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=12, framealpha=0.9)
ax.set_title("Example inference on an unseen two-target scene", fontsize=15, pad=10)

# 確率のカラーバー（クラス別に 2 本、下側にコンパクトに横置き）
for i, label in enumerate(("Cyclist", "Vehicle")):
    cax = fig.add_axes([0.16 + 0.38 * i, 0.09, 0.28, 0.022])
    cb = fig.colorbar(mappables[label], cax=cax, orientation="horizontal")
    cb.set_label(f"{label} probability", fontsize=11)
    cb.ax.tick_params(labelsize=9)

plt.savefig(OUT, dpi=300)
print(f"saved: {OUT}")
