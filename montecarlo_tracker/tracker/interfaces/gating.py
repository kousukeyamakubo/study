from abc import ABC, abstractmethod
from typing import List
import numpy as np
from core.state import GaussianState  # パスが変わったので注意

class IGating(ABC):
    """ゲーティング（有効な観測の選別）を行うインターフェース"""
    
    @abstractmethod
    def validate_measurements(self, predicted_state: GaussianState, 
                            measurements: List[np.ndarray]) -> List[int]:
        """
        Args:
            predicted_state: ターゲットの予測状態
            measurements: その時刻の全観測リスト
        Returns:
            ゲート内に入っている観測のインデックスリスト
        """
        pass