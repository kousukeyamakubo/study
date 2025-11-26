from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    """シミュレーションおよびアルゴリズム設定用のコンフィグクラス"""
    # --- シミュレーション基本設定 ---
    seed: int = 42
    num_steps: int = 40
    dt: float = 0.5
    
    # --- 観測データ設定 ---
    use_csv_mode : bool = False
    
    # --- システム構成（ここを変えると使うクラスが変わる） ---
    tracker_type: str = "JPDA"        # "JPDA", "GNN", "PDA" ...
    gating_type: str = "Ellipsoidal"  # "Ellipsoidal", "Rectangular" ...
    
    # --- JPDAF / Gating 設定 ---
    gate_threshold: float = 9.21  # カイ二乗分布の閾値
    clutter_density: float = 0.01 # 空間密度
    detection_prob: float = 0.9   # 検出確率 P_D
    
    # --- 別のアルゴリズム用パラメータ設定 ---
    # ここに他のアルゴリズム固有のパラメータ

    # --- クラッタ生成パラメータ ---
    # mutableなデフォルト値はfield(default_factory=...)を使う必要があります
    clutter_params: dict = field(default_factory=lambda: {
        'type': 'around_target',
        'num_clutter': [1, 8], 
        'clutter_spatial_density': [0.5, 1.0]
    })
    
    # -- 運動・観測モデル設定 ---
    motion_model: str = "CV"          # "CV" (等速), "CA" (等加速) など
    process_noise_std: float = 0.5    # プロセスノイズの標準偏差 (Qの元)
    measurement_noise_std: float = 0.5 # 観測ノイズの標準偏差 (Rの元)