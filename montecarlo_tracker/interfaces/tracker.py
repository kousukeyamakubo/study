from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
from core.state import GaussianState

class ITracker(ABC):
    """全ての追跡アルゴリズムが継承すべき基底クラス"""

    @abstractmethod
    def predict(self, states: List[GaussianState]) -> List[GaussianState]:
        """状態予測を行う"""
        pass

    @abstractmethod
    def update(self, predicted_states: List[GaussianState], 
               measurements: List[np.ndarray]) -> Tuple[List[GaussianState], Any]:
        """
        観測を用いて状態更新を行う
        Return:
            updated_states: 更新後の状態リスト
            info: アルゴリズム依存の付加情報（JPDAならバリデーション行列など。保存用）
        """
        pass