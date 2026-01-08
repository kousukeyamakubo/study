from abc import ABC, abstractmethod
from core.state import GaussianState
import numpy as np

class IFilter(ABC):
    """フィルタ（KF, EKF等）の共通インターフェース"""
    
    @property
    @abstractmethod
    def H(self) -> np.ndarray:
        """観測行列（線形近似含む）"""
        pass

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """観測ノイズ共分散"""
        pass

    @abstractmethod
    def predict(self, state: GaussianState) -> GaussianState:
        pass

    @abstractmethod
    def update(self, predicted_state: GaussianState, measurement: np.ndarray) -> GaussianState:
        pass
    
    @abstractmethod
    def likelihood(self, predicted_state: GaussianState, measurement: np.ndarray) -> float:
        pass