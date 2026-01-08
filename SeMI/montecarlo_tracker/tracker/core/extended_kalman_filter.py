import numpy as np
from .state import GaussianState
from interfaces.filter import IFilter

# 実装の雛形（プレースホルダー）
class ExtendedKalmanFilter(IFilter):
    """拡張カルマンフィルタ (EKF)"""
    
    def __init__(self, config):
        # 非線形モデルの設定などを受け取る
        self.cfg = config
        # self.R = ...
        pass

    @property
    def H(self) -> np.ndarray:
        # EKFの場合、Hは状態に依存して変化する（ヤコビアン）ため
        # ここで固定値を返す設計だと厳密には不十分ですが、
        # 既存コードとの互換性のためにダミーまたは代表値を返す実装などが考えられます
        return np.eye(2) # 仮

    @property
    def R(self) -> np.ndarray:
        return np.eye(2) # 仮

    def predict(self, state: GaussianState) -> GaussianState:
        # TODO: 非線形予測ステップの実装
        # x = f(x)
        # F = jacobian(f, x)
        # P = F P F.T + Q
        raise NotImplementedError("EKF is not implemented yet.")

    def update(self, predicted_state: GaussianState, measurement: np.ndarray) -> GaussianState:
        # TODO: 非線形更新ステップの実装
        raise NotImplementedError("EKF is not implemented yet.")

    def likelihood(self, predicted_state: GaussianState, measurement: np.ndarray) -> float:
        # TODO: 尤度計算
        return 0.0