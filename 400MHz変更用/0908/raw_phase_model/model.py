# プロトタイプ: 複素RD（生位相）から地上 (X,Y) グリッド上のクラスマップを直接出すモデル。
#
# 【意図（次回チャットに引き継ぐための記録）】
# 9/8 の論点「角度情報を DBF で潰すか、生の位相のまま渡すか」（meeting/2026-09-08.md）のうち、
# 後者を検証するための最小プロトタイプ。学習は一切していない。形状を通しただけ。
#
# 既存モデル（docs/model.md の RadarUNet3DSoftmax）は
#   入力: (B,1,N_FIXED,H,W) ← DBF で作った角度グリッドを depth 軸にした RD マップ
#   出力: (B,3,N_FIXED,H,W) ← 角度・レンジ・ドップラーのボクセルごとに3クラス
# という「角度をグリッド化してから detector に渡す」設計。DBF が 14.5° の角度分解能で
# 頭打ちになる問題（meeting/2026-09-08.md「RAD テンソルをどう扱うか」）がそのまま乗る。
#
# 本プロトタイプは逆に、
#   入力: (B, 2*n_virt, Doppler, Range) ← 7素子の複素RDを実部・虚部14chに展開しただけ（DBF なし）
#   出力: (B, n_classes, grid_y, grid_x) ← 地上平面のクラスマップを直接
# とし、「角度→位置」の変換ごとネットワークに学習させる。
#
# 【(Doppler,Range) と (Y,X) は単純なアップサンプリングでは繋がらない】
# Range は斜距離（h・俯角で地上 Y に変換）、角度は素子間位相差から求まる（地上 X に対応）。
# どちらも非線形な変換なので、エンコーダで一度小さい特徴に圧縮し、デコーダで
# 地上グリッドサイズへ再展開する構成にしている（対応する画素をそのまま引き伸ばすのではない）。

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 地上グリッドの範囲・分解能。中身は未検証のプレースホルダ（h・俯角の確定後に見直す）
GRID_X_RANGE_M = (-20.0, 20.0)
GRID_Y_RANGE_M = (0.0, 50.0)
GRID_RES_M = 0.5
GRID_X = int((GRID_X_RANGE_M[1] - GRID_X_RANGE_M[0]) / GRID_RES_M)   # 80
GRID_Y = int((GRID_Y_RANGE_M[1] - GRID_Y_RANGE_M[0]) / GRID_RES_M)   # 100

# 本研究の対象クラス。../camera_pipeline/0817/lib/detections.py の CLS_* と合わせる
N_CLASSES = 3  # 背景 / cyclist / pedestrian


def select_virtual_elements(rd: np.ndarray, virt_array: list) -> np.ndarray:
    """(Frame,TX,RX,Doppler,Range) から仮想素子ごとの (Frame,n_virt,Doppler,Range) を取り出す。

    virt_array は atlas_export.py の npz に入っている素子マッピング
    （例: [[[0,1]], [[0,0]], ..., [[1,1],[0,3]], ...]）。1つの仮想素子位置に
    複数の物理 (TX,RX) 対が対応する場合（MIMO の冗長素子）は平均する。
    """
    out = []
    for pairs in virt_array:
        chans = [rd[:, tx, rx] for tx, rx in pairs]
        out.append(np.mean(chans, axis=0) if len(chans) > 1 else chans[0])
    return np.stack(out, axis=1)  # (F, n_virt, D, R)


def to_input_channels(rd_virt: np.ndarray) -> np.ndarray:
    """(Frame,n_virt,Doppler,Range) complex → (Frame, 2*n_virt, Doppler, Range) float32。

    実部・虚部をチャネル方向に並べる。DBF のような線形変換すら通さず、
    位相情報をそのままネットワークに渡すのがこのプロトタイプの核心。
    """
    f, n_virt, d, r = rd_virt.shape
    out = np.empty((f, 2 * n_virt, d, r), dtype=np.float32)
    out[:, 0::2] = rd_virt.real.astype(np.float32)
    out[:, 1::2] = rd_virt.imag.astype(np.float32)
    return out


def load_virt_array(npz) -> list:
    """atlas_export.py の npz に文字列（JSON 形式）で入っている virt_array を読む"""
    return json.loads(str(npz["virt_array"]))


class RawPhaseGroundNet(nn.Module):
    """複素RD（生位相, 14ch）→ 地上グリッドのクラスマップ。学習未実施のプロトタイプ。

    層の種類・段数・チャネル数は何も決めていない。ここにあるのは
    「この入出力形状で学習させる」という設計方針だけを示す最小限のプレースホルダで、
    中身（アーキテクチャ）は学習に着手する段階で作り直す前提。
    """

    def __init__(self, n_virt: int = 7, n_classes: int = N_CLASSES,
                 grid_y: int = GRID_Y, grid_x: int = GRID_X, ch: int = 32):
        super().__init__()
        self.grid_y, self.grid_x = grid_y, grid_x
        self.feature = nn.Conv2d(n_virt * 2, ch, 3, padding=1)  # 仮の1層。中身は未定
        self.head = nn.Conv2d(ch, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*n_virt, Doppler, Range)
        feat = F.relu(self.feature(x))
        # (Doppler,Range) と (Y,X) は物理的に別の空間（角度・斜距離 → 地上座標は非線形変換）
        # なので、対応する画素をそのまま引き伸ばすのではなく、形状を合わせるためだけに補間する
        feat = F.interpolate(feat, size=(self.grid_y, self.grid_x),
                             mode="bilinear", align_corners=False)
        return self.head(feat)                                    # (B, n_classes, grid_y, grid_x)
