import numpy as np
from typing import List, Tuple, Any

# 必要なモジュールのインポート
from interfaces.tracker import ITracker
from interfaces.gating import IGating
from interfaces.association import IAssociation
from core.state import GaussianState
from core.kalman_filter import LinearKalmanFilter

class JPDAF(ITracker):
    """
    Joint Probabilistic Data Association Filter
    部品（Gating, Association）を組み合わせて追跡を行うクラス
    """
    
    def __init__(self, kf: LinearKalmanFilter, 
                 gating_module: IGating, 
                 association_calculator: IAssociation):
        """
        Args:
            kf: カルマンフィルタ（状態予測・更新の計算式）
            gating_module: ゲーティング処理を行う部品
            association_calculator: データ割り当て確率計算を行う部品
        """
        self.kf = kf
        # 外部から注入された部品を保持
        self.gating_module = gating_module
        self.association_calculator = association_calculator
        
        self.H = kf.H
        self.R = kf.R

    def predict(self, states: List[GaussianState]) -> List[GaussianState]:
        """
        全ターゲットの予測ステップ
        """
        # 各ターゲットに対して単純にKFの予測を実行
        predicted_states = [self.kf.predict(s) for s in states]
        return predicted_states

    def update(self, predicted_states: List[GaussianState], 
                 measurements: List[np.ndarray]) -> Tuple[List[GaussianState], Any]:
        """
        JPDAFの更新ステップ
        Return:
            (更新後の状態リスト, バリデーション情報)
        """
        num_targets = len(predicted_states)
        num_measurements = len(measurements)
        
        # 観測がない場合は予測状態をそのまま返す
        if num_measurements == 0:
            return predicted_states, []
            
        # --- ステップ1: ゲーティング (部品に委譲) ---
        validation_matrix = np.zeros((num_targets, num_measurements), dtype=int)
        validated_measurements_per_target = []
        
        for i, state in enumerate(predicted_states):
            # 注入された GatingModule を使用
            indices = self.gating_module.validate_measurements(state, measurements)
            validated_measurements_per_target.append(indices)
            
            for j in indices:
                validation_matrix[i, j] = 1

        # --- ステップ2: アソシエーション確率計算 (部品に委譲) ---
        # 注入された AssociationCalculator を使用
        beta = self.association_calculator.calculate_probabilities(
            predicted_states, measurements, validation_matrix
        )
        
        # --- ステップ3: 各ターゲットの状態更新 ---
        updated_states = []
        
        for i in range(num_targets):
            state = predicted_states[i] # P(k|k-1)
            beta_i = beta[i]            # ターゲットiのアソシエーション確率ベクトル
            
            # --- KFのゲイン計算 (ターゲットごとに異なる) ---
            # S = H * P * H.T + R
            S = self.H @ state.covariance @ self.H.T + self.R
            S_inv = np.linalg.inv(S)
            # K = P * H.T * S_inv
            K = state.covariance @ self.H.T @ S_inv
            
            # ゲート内の観測インデックス
            validated_indices = validated_measurements_per_target[i]
            
            # イノベーション計算
            innovations = [measurements[j] - self.H @ state.mean for j in validated_indices]
            
            # 複合イノベーション (v_i)
            combined_innovation = np.zeros(self.H.shape[0]) 
            if validated_indices:
                # beta_i[j+1] は観測jに対応する確率 (index 0は未検出)
                combined_innovation = sum(beta_i[j+1] * innovations[k] 
                                          for k, j in enumerate(validated_indices))
            
            # 状態平均の更新: x(k|k) = x(k|k-1) + K * v_i
            updated_mean = state.mean + K @ combined_innovation
            
            # 共分散の更新
            # P_c: 正しい観測が得られたと仮定した場合の共分散
            P_c = (np.eye(len(state.mean)) - K @ self.H) @ state.covariance
            
            # スプレッド項 (P_tilde): 観測の不確かさによる広がり
            spread_term = np.zeros((len(state.mean), len(state.mean))) 
            if validated_indices:
                spread_sum = sum(beta_i[j+1] * np.outer(innovations[k], innovations[k]) 
                                 for k, j in enumerate(validated_indices))
                spread_term = K @ (spread_sum - np.outer(combined_innovation, combined_innovation)) @ K.T
            
            # 最終的な共分散: P(k|k) = beta_0 * P(k|k-1) + (1 - beta_0) * P_c + P_tilde
            updated_cov = beta_i[0] * state.covariance + (1 - beta_i[0]) * P_c + spread_term
            
            updated_states.append(GaussianState(updated_mean, updated_cov))
            
        return updated_states, validated_measurements_per_target