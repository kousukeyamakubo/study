import numpy as np
from config import Config
from typing import Tuple

def get_motion_model(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    設定に基づいて F (遷移行列) と Q (プロセスノイズ) を生成する
    """
    dt = cfg.dt
    q_std = cfg.process_noise_std
    
    if cfg.motion_model == "CV":
        # --- 等速直線運動モデル (Constant Velocity) ---
        # 状態: [x, y, vx, vy]
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 簡易的なQの生成（離散ホワイトノイズモデル）
        # 必要に応じて、より厳密な区分的定数白色加速度モデルなどに変更可能
        Q = np.eye(4) * (q_std ** 2)
        # より物理的に正しいQの例:
        # q = q_std**2
        # Q = np.array([
        #     [0.25*dt**4, 0, 0.5*dt**3, 0],
        #     [0, 0.25*dt**4, 0, 0.5*dt**3],
        #     [0.5*dt**3, 0, dt**2, 0],
        #     [0, 0.5*dt**3, 0, dt**2]
        # ]) * q

    elif cfg.motion_model == "CA":
        # --- (例) 等加速度運動モデル ---
        # 状態: [x, y, vx, vy, ax, ay] など
        # ここに実装を追加すれば、Configの書き換えだけで切り替え可能に！
        raise NotImplementedError("CA model not implemented yet")
        
    else:
        raise ValueError(f"Unknown motion model: {cfg.motion_model}")
        
    return F, Q

def get_measurement_model(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    設定に基づいて H (観測行列) と R (観測ノイズ) を生成する
    """
    r_std = cfg.measurement_noise_std
    
    # 現状は位置観測 (x, y) のみと仮定
    # 将来的に "Polar" (距離・角度) などを追加する場合もここで分岐
    
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    R = np.eye(2) * (r_std ** 2)
    
    return H, R