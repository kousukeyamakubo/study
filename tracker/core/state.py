import numpy as np
from dataclasses import dataclass

@dataclass
class GaussianState:
    """ガウス分布で表現される状態"""
    mean: np.ndarray  # 状態ベクトル
    covariance: np.ndarray  # 共分散行列
    
    def __post_init__(self):
        """検証"""
        self.mean = np.asarray(self.mean)
        self.covariance = np.asarray(self.covariance)
        
        if self.mean.ndim != 1:
            raise ValueError("mean must be 1-dimensional")
        
        if self.covariance.shape != (len(self.mean), len(self.mean)):
            raise ValueError("covariance shape must match mean dimension")