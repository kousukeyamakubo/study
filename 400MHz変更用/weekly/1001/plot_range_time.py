# 収録した複数の .dat を距離-時間で並べて見る。尾根追跡の前に、生の分布を確認するため。
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0727"))
from atlas_ridge_track import moving_power_db   # noqa: E402

npzs = sorted(Path(sys.argv[1]).glob("*/*_rd.npz"))
rmax = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

fig, axes = plt.subplots(1, len(npzs), figsize=(4.2 * len(npzs), 4.6), sharey=True)
for ax, npz in zip(np.atleast_1d(axes), npzs):
    pw, vel, rng, t, *_ = moving_power_db(npz)      # (F,R), (F,R), (R,), (F,)
    keep = rng <= rmax
    im = ax.pcolormesh(rng[keep], t, pw[:, keep], shading="auto",
                       cmap="viridis", vmin=10, vmax=40)
    ax.set_title(npz.stem.replace("_rd", "")[-6:], fontsize=10)
    ax.set_xlabel("Range [m]")
np.atleast_1d(axes)[0].set_ylabel("time [s]")
fig.colorbar(im, ax=axes, label="moving power [dB]")
fig.savefig("range_time_1001.png", dpi=120, bbox_inches="tight")
print("-> range_time_1001.png")
