from interfaces.association import IAssociation
from core.state import GaussianState
import numpy as np
from typing import List

class GNNAssociation(IAssociation):
    """GNN (Global Nearest Neighbor) データ割り当て"""
    
    def __init__(self, kf):
        self.kf = kf

    def calculate_probabilities(self, predicted_states: List[GaussianState], 
                              measurements: List[np.ndarray],
                              validation_matrix: np.ndarray) -> np.ndarray:
        """
        GNNでは確率ではなく「0か1」の行列を返すことで、
        JPDAFの更新式をそのまま流用して「確定割り当て」を実現できる。
        """
        num_targets = len(predicted_states)
        num_measurements = len(measurements)
        beta = np.zeros((num_targets, num_measurements + 1))
        
        # TODO: ここにMunkres法（ハンガリアン法）などで
        # コスト（尤度や距離）を最小化する割り当てを計算するロジックを実装
        # assign_matrix = linear_sum_assignment(cost_matrix) ...
        
        # 仮の実装: 全て未検出扱い (beta[:, 0] = 1)
        beta[:, 0] = 1.0 
        
        return beta