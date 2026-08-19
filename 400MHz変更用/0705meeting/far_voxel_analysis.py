"""
far_voxel_analysis.py

voxel_3d_relative.png（GT中心の重ね描き）で遠方ボクセルが帯状に見えた点の深掘り:
自クラス GT から遠い（>5 voxel）「誤警報」ボクセルが、もう一方のクラスの
物体位置の周辺に集中しているかを確認する。

先に voxel_3d_visualization.py を実行してキャッシュ（results/voxel_records_cache.npz）
を生成しておくこと。

結果（2026-07-05 時点、300 件）:
  cyclist の遠方ボクセルの 92.9% が vehicle GT から 10 voxel 以内（3 voxel 以内は 0%）
  vehicle の遠方ボクセルの 78.5% が cyclist GT から 10 voxel 以内（3 voxel 以内は 0%）
  → 誤警報は無関係な場所ではなく「他方のターゲット応答の周辺リング」で発生している
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "results", "voxel_records_cache.npz")

# 「遠方」とみなす自クラス GT からの距離しきい値 [voxel]
# （NMS 分析②より本体の広がりは 2〜3 voxel なので、5 超は本体外とみなせる）
FAR_TH = 5.0

data = np.load(CACHE_PATH, allow_pickle=True)
recs = {"cyclist": list(data["cyclist"]), "vehicle": list(data["vehicle"])}
gt_map = {k: {r["sample_idx"]: r["gt"] for r in v if r["gt"] is not None}
          for k, v in recs.items()}

for label, other in [("cyclist", "vehicle"), ("vehicle", "cyclist")]:
    n_far = 0
    d_other_all = []
    for r in recs[label]:
        if r["gt"] is None:
            continue
        d_own = np.sqrt(((r["coords"] - r["gt"][None, :])**2).sum(axis=1))
        m = d_own > FAR_TH
        if not m.any():
            continue
        og = gt_map[other].get(r["sample_idx"])
        if og is None:
            continue
        # 遠方ボクセルから「他クラス GT」までの距離
        d_other = np.sqrt(((r["coords"][m] - og[None, :])**2).sum(axis=1))
        n_far += int(m.sum())
        d_other_all.extend(d_other.tolist())

    d_other_all = np.array(d_other_all)
    total_vox = sum(len(r["probs"]) for r in recs[label])
    print(f"--- {label} ---")
    print(f"  total voxels: {total_vox}, far from own GT (>{FAR_TH:.0f}): {n_far} ({100*n_far/total_vox:.1f}%)")
    if len(d_other_all):
        print(f"  dist to OTHER-class GT: "
              f"p10={np.percentile(d_other_all,10):.1f}, p25={np.percentile(d_other_all,25):.1f}, "
              f"median={np.median(d_other_all):.1f}, p75={np.percentile(d_other_all,75):.1f}, "
              f"p90={np.percentile(d_other_all,90):.1f}, max={d_other_all.max():.1f}")
        for th in (3, 5, 10, 20):
            n = (d_other_all <= th).sum()
            print(f"    <={th:2d} voxels: {n:4d} ({100*n/len(d_other_all):.1f}%)")
