# RawPhaseGroundNet の形状確認。学習はしない。
#
# 実測npz（atlas_log_20260727_195113_rd.npz）があればそれで、無ければ合成データで確認する。
#
# 使い方:
#   python check_prototype.py

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (GRID_X, GRID_Y, RawPhaseGroundNet, load_virt_array,  # noqa: E402
                   select_virtual_elements, to_input_channels)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_NPZ = REPO_ROOT / "atlas_log_20260727_195113_rd.npz"


def check_with_real_data():
    print("[1] 実測npzでの形状確認")
    if not SAMPLE_NPZ.exists():
        print(f"  {SAMPLE_NPZ.name} が無いのでスキップ")
        return
    d = np.load(SAMPLE_NPZ, allow_pickle=True)
    virt_array = load_virt_array(d)
    print(f"  virt_array: {len(virt_array)} 素子")

    rd_virt = select_virtual_elements(d["rd"], virt_array)   # (F, n_virt, D, R)
    x = to_input_channels(rd_virt)                            # (F, 2*n_virt, D, R)
    print(f"  入力テンソル形状: {x.shape}")

    model = RawPhaseGroundNet(n_virt=len(virt_array))
    with torch.no_grad():
        y = model(torch.from_numpy(x))
    print(f"  出力形状: {tuple(y.shape)} (期待値: (F, 3, {GRID_Y}, {GRID_X}))")
    assert y.shape == (x.shape[0], 3, GRID_Y, GRID_X)
    print("  → OK")


def check_with_synthetic_data():
    print("\n[2] 合成データでの形状確認（npz が無い場合の保険）")
    n_frame, n_virt, n_doppler, n_range = 8, 7, 16, 129
    rng = np.random.default_rng(0)
    rd_virt = (rng.standard_normal((n_frame, n_virt, n_doppler, n_range))
              + 1j * rng.standard_normal((n_frame, n_virt, n_doppler, n_range)))
    x = to_input_channels(rd_virt)
    print(f"  入力テンソル形状: {x.shape}")

    model = RawPhaseGroundNet(n_virt=n_virt)
    with torch.no_grad():
        y = model(torch.from_numpy(x))
    print(f"  出力形状: {tuple(y.shape)} (期待値: ({n_frame}, 3, {GRID_Y}, {GRID_X}))")
    assert y.shape == (n_frame, 3, GRID_Y, GRID_X)
    print("  → OK")


if __name__ == "__main__":
    check_with_real_data()
    check_with_synthetic_data()
