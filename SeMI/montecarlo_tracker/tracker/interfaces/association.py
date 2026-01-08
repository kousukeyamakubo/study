from abc import ABC, abstractmethod
from typing import List
import numpy as np
from core.state import GaussianState

class IAssociation(ABC):
    """データ割り当て（アソシエーション）計算のインターフェース"""
    
    @abstractmethod
    def calculate_probabilities(self, predicted_states: List[GaussianState], 
                              measurements: List[np.ndarray],
                              validation_matrix: np.ndarray) -> np.ndarray:
        """
        Args:
            predicted_states: 全ターゲットの予測状態リスト
            measurements: 全観測リスト
            validation_matrix: [Target x Meas] のゲーティング行列 (0 or 1)
            
        Returns:
            beta: アソシエーション確率行列（JPDAの場合）
                  または割り当て行列（GNNの場合など、形式は実装によるが基本は行列）
        """
        pass