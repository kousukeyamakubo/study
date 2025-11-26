import numpy as np
from .state import GaussianState

class LinearKalmanFilter:
    """線形カルマンフィルタ"""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Args:
            F: 状態遷移行列
            H: 観測行列
            Q: プロセスノイズ共分散
            R: 観測ノイズ共分散
        """
        self.F = np.asarray(F)
        self.H = np.asarray(H)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
    
    def predict(self, state: GaussianState) -> GaussianState:
        """予測ステップ"""
        predicted_mean = self.F @ state.mean
        predicted_cov = self.F @ state.covariance @ self.F.T + self.Q
        return GaussianState(predicted_mean, predicted_cov)
    
    def update(self, predicted_state: GaussianState, measurement: np.ndarray) -> GaussianState:
        """更新ステップ"""
        measurement = np.asarray(measurement)
        
        # イノベーション
        innovation = measurement - self.H @ predicted_state.mean
        
        # イノベーション共分散
        S = self.H @ predicted_state.covariance @ self.H.T + self.R
        
        # カルマンゲイン
        K = predicted_state.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 状態更新
        updated_mean = predicted_state.mean + K @ innovation
        updated_cov = (np.eye(len(predicted_state.mean)) - K @ self.H) @ predicted_state.covariance
        
        return GaussianState(updated_mean, updated_cov)
    
    def likelihood(self, predicted_state: GaussianState, measurement: np.ndarray) -> float:
        """観測尤度を計算"""
        measurement = np.asarray(measurement)
        innovation = measurement - self.H @ predicted_state.mean
        S = self.H @ predicted_state.covariance @ self.H.T + self.R
        
        # ガウス分布の確率密度
        dim = len(measurement)
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        
        likelihood = (1.0 / np.sqrt((2 * np.pi) ** dim * det_S)) * \
                    np.exp(-0.5 * innovation.T @ inv_S @ innovation)
        
        return likelihood