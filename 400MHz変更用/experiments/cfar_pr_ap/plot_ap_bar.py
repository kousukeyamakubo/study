"""plot_ap_bar.py
for_paper/cfar_pr_ap.csv から AP 上位 10 設定の棒グラフを生成する。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH   = "./for_paper/cfar_pr_ap.csv"
OUTPUT_DIR = "./for_paper"

FONT_LABEL = 18
FONT_TICK  = 14
FONT_VAL   = 13

df   = pd.read_csv(CSV_PATH)
top10 = df.nlargest(10, "ap")
ap_max = top10["ap"].max()

labels = [
    f"ta={int(r.n_train_a)}, td={int(r.n_train_d)}, "
    f"tr={int(r.n_train_r)}, gr={int(r.n_guard_r)}, pfa={r.pfa:.0e}"
    for _, r in top10.iterrows()
]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(len(top10)), top10["ap"].values, color="steelblue", alpha=0.7)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(labels, fontsize=FONT_TICK)
ax.invert_yaxis()
ax.set_xlabel("AP", fontsize=FONT_LABEL)
ax.tick_params(axis="x", labelsize=FONT_TICK)
ax.set_title("Top-10 CFAR Configurations by AP", fontsize=FONT_LABEL)
ax.grid(True, axis="x")

# AP 数値をバー内側右端に配置してはみ出しを防ぐ
for bar, val in zip(bars, top10["ap"].values):
    ax.text(val - ap_max * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="right",
            fontsize=FONT_VAL, color="white", fontweight="bold")
ax.set_xlim(0, ap_max * 1.05)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "cfar_ap_bar.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"保存: {out}")
