import numpy as np
from typing import List, Dict
from itertools import product

# 新しいパスからのインポート
from interfaces.association import IAssociation
from core.state import GaussianState
from core.kalman_filter import LinearKalmanFilter

class JPDAAssociation(IAssociation):
    """
    JPDA (Joint Probabilistic Data Association) の実装クラス
    """

    def __init__(self, kf: LinearKalmanFilter, clutter_density: float = 1e-4, 
                 detection_prob: float = 0.9):
        """
        Args:
            kf: カルマンフィルタ (尤度計算に使用)
            clutter_density: クラッタ密度 (lambda_c)
            detection_prob: 検出確率 (P_D)
        """
        self.kf = kf
        self.clutter_density = max(clutter_density, 1e-20) # ゼロ除算防止
        self.detection_prob = detection_prob
        self.not_detection_prob = 1.0 - detection_prob
        
        # 仮説キャッシュ (形状と内容が同じなら再利用)
        self._hypothesis_cache: Dict[tuple, list] = {}

    def _calculate_likelihood_matrix(self, 
                                     predicted_states: List[GaussianState], 
                                     measurements: List[np.ndarray]) -> np.ndarray:
        """
        尤度行列 L を計算 (内部ヘルパー)
        L[i, j] = ターゲットi が 観測j を生成した尤度
        """
        num_targets = len(predicted_states)
        num_measurements = len(measurements)
        
        # L[i, 0] はダミー、L[i, j] (j>0) を計算
        likelihood_matrix = np.zeros((num_targets, num_measurements + 1))
        
        for i in range(num_targets):
            for j in range(num_measurements):
                # KFの尤度関数を利用
                likelihood_matrix[i, j+1] = self.kf.likelihood(
                    predicted_states[i], measurements[j]
                )
        return likelihood_matrix

    def _generate_hypotheses(self, validation_matrix: np.ndarray) -> list:
        """
        妥当化行列から、全ての可能な（競合しない）仮説を生成する
        """
        num_targets, num_measurements = validation_matrix.shape
        
        # キャッシュキー作成
        cache_key = (num_targets, num_measurements, tuple(validation_matrix.flatten()))
        if cache_key in self._hypothesis_cache:
            return self._hypothesis_cache[cache_key]

        # 各ターゲットの選択肢 (0:未検出, j+1:観測j)
        target_choices = []
        for i in range(num_targets):
            choices = [0] # 未検出は常にあり得る
            for j in range(num_measurements):
                if validation_matrix[i, j] == 1:
                    choices.append(j + 1)
            target_choices.append(choices)

        # 全組み合わせ (直積)
        # ※ターゲット数が増えるとここがボトルネックになる可能性があります
        all_combinations = product(*target_choices)

        # 競合チェック (同じ観測 j>0 を複数のターゲットが共有していないか)
        valid_hypotheses = []
        for hypothesis in all_combinations:
            measurements_assigned = [j for j in hypothesis if j > 0]
            if len(measurements_assigned) == len(set(measurements_assigned)):
                valid_hypotheses.append(hypothesis)
        
        self._hypothesis_cache[cache_key] = valid_hypotheses
        return valid_hypotheses

    def calculate_probabilities(self, 
                                predicted_states: List[GaussianState], 
                                measurements: List[np.ndarray],
                                validation_matrix: np.ndarray) -> np.ndarray:
        """
        IAssociation インターフェースの実装
        """
        num_targets = len(predicted_states)
        num_measurements = len(measurements)
        
        # 観測がない場合
        if num_measurements == 0:
            beta = np.zeros((num_targets, 1))
            beta[:, 0] = 1.0
            return beta

        # 1. 尤度行列の計算
        likelihood_matrix = self._calculate_likelihood_matrix(predicted_states, measurements)

        # 2. 仮説の列挙
        hypotheses = self._generate_hypotheses(validation_matrix)

        # 3. 各仮説の確率計算
        hypothesis_probs = []
        total_prob_sum = 0.0

        for hypothesis in hypotheses:
            prob = 1.0
            for i in range(num_targets):
                j = hypothesis[i]
                if j == 0:
                    prob *= self.not_detection_prob
                else:
                    prob *= self.detection_prob * likelihood_matrix[i, j]
            
            # クラッタ確率
            assigned_count = len([j for j in hypothesis if j > 0])
            num_clutter = num_measurements - assigned_count
            prob *= (self.clutter_density ** num_clutter)

            hypothesis_probs.append(prob)
            total_prob_sum += prob

        # 4. 正規化
        if total_prob_sum > 0:
            normalized_probs = [p / total_prob_sum for p in hypothesis_probs]
        else:
            fallback_prob = 1.0 / len(hypotheses) if len(hypotheses) > 0 else 0
            normalized_probs = [fallback_prob] * len(hypotheses)

        # 5. 周辺化 (各ターゲット・各観測ごとの確率 beta を算出)
        beta = np.zeros((num_targets, num_measurements + 1))
        
        for k, hypothesis in enumerate(hypotheses):
            prob = normalized_probs[k]
            for i in range(num_targets):
                j = hypothesis[i]
                beta[i, j] += prob
        
        return beta