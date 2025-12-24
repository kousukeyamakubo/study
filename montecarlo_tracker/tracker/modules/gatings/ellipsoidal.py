import numpy as np
from typing import List
from interfaces.gating import IGating
from core.state import GaussianState

class EllipsoidalGating(IGating):
    """
    マハラノビス距離を用いた楕円ゲーティング
    """
    
    def __init__(self, H: np.ndarray, R: np.ndarray, threshold: float = 9.21):
        """
        Args:
            H: 観測行列
            R: 観測ノイズ共分散
            threshold: ゲーティング閾値（カイ2乗分布）
        """
        self.H = np.asarray(H)
        self.R = np.asarray(R)
        self.threshold = threshold
    
    def validate_measurements(self, predicted_state: GaussianState, 
                            measurements: List[np.ndarray]) -> List[int]:
        """
        ゲーティング処理を実行 (IGatingの実装)
        
        Args:
            predicted_state: 予測状態
            measurements: 観測リスト
            
        Returns:
            有効な観測のインデックスリスト
        """
        # S = H * P * H^T + R
        S = self.H @ predicted_state.covariance @ self.H.T + self.R
        
        # 数値安定性のための微小項（オプション）
        # S += np.eye(S.shape[0]) * 1e-9
        
        try:
            inv_S = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 万が一特異行列になった場合のフェイルセーフ（空リストを返すなど）
            print("Warning: Singular matrix in gating.")
            return []

        validated_indices = []
        for i, z in enumerate(measurements):
            z = np.asarray(z)
            innovation = z - self.H @ predicted_state.mean
            
            # マハラノビス距離 d^2 = v^T * S^-1 * v
            mahalanobis = innovation.T @ inv_S @ innovation
            
            if mahalanobis <= self.threshold:
                validated_indices.append(i)
        
        return validated_indices