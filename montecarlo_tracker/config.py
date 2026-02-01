from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    """シミュレーションおよびアルゴリズム設定用のコンフィグクラス"""
    # --- シミュレーション基本設定 ---
    seed: int = 42
    num_steps: int = 31
    dt: float = 0.05
    
    # --- 観測データ設定 ---
    use_csv_mode : bool = True
    
    # --- システム構成（ここを変えると使うクラスが変わる） ---
    tracker_type: str = "JPDA"        # "JPDA", "GNN", "PDA" ...
    gating_type: str = "Ellipsoidal"  # "Ellipsoidal", "Rectangular" ...
    filter_type: str = "KF"           # "KF" (Linear), "EKF" (Extended)
    
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
        #'num_clutter': [1, 8],
        'num_clutter': 0,
        'clutter_spatial_density': [0.5, 1.0]
    })
    
    # -- 運動・観測モデル設定 ---
    motion_model: str = "CV"          # "CV" (等速), "CA" (等加速) など
    # デフォルトはどちらも0.5
    #process_noise_std: float = 0.5    # プロセスノイズの標準偏差 (Qの元)
    #measurement_noise_std: float = 0.5 # 観測ノイズの標準偏差 (Rの元)
    # パラメータの再調整
    #process_noise_std: float = 0.05    # 0.5 -> 0.05 (動きの滑らかさを重視)
    #measurement_noise_std: float = 0.25 # 0.5 -> 0.25 (実測データに合わせる)
    # 推奨設定
    #process_noise_std: float = 0.1      # 0.05 -> 0.1 (動きの変化を許容し、遅れを減らす)
    #measurement_noise_std: float = 0.23 # 0.25 -> 0.23 (実測データ標準偏差に合わせる)
    # SeMIのとき
    #process_noise_std: float = 0.3      # 0.05 -> 0.1 (動きの変化を許容し、遅れを減らす)
    #measurement_noise_std: float = 0.18 # 0.25 -> 0.23 (実測データ標準偏差に合わせる)

    process_noise_std: float = 1.0
    measurement_noise_std: float = 1.0
