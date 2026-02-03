import numpy as np
from typing import List, Tuple, Any

# 必要なモジュールのインポート
from interfaces.tracker import ITracker
from interfaces.gating import IGating
from interfaces.association import IAssociation
from core.state import GaussianState
from interfaces.filter import IFilter

class StandardTracker(ITracker):
    """
    Standard Tracker
    部品（Gating, Association）を組み合わせて追跡を行うクラス
    """
    
    def __init__(self, filter: IFilter, 
             gating_module: IGating, 
             association_calculator: IAssociation,
             max_miss_count: int = 5):  # ← 追加
        """
        Args:
            filter: カルマンフィルタ（状態予測・更新の計算式）
            gating_module: ゲーティング処理を行う部品
            association_calculator: データ割り当て確率計算を行う部品
            max_miss_count: 連続未検出でトラック削除する閾値
        """
        self.filter = filter
        self.gating_module = gating_module
        self.association_calculator = association_calculator
        self.max_miss_count = max_miss_count  # ← 追加
        self.next_track_id = 0  # トラックIDの初期値(追加)
        
        self.H = filter.H
        self.R = filter.R

    def predict(self, states: List[GaussianState]) -> List[GaussianState]:
        """
        全ターゲットの予測ステップ
        """
        # 各ターゲットに対して単純にKFの予測を実行
        predicted_states = [self.filter.predict(s) for s in states]
        return predicted_states

    def update(self, predicted_states: List[GaussianState], 
             measurements: List[np.ndarray]) -> Tuple[List[GaussianState], Any]:
        """
        Standard Trackerの更新ステップ
        Return:
            (更新後の状態リスト, バリデーション情報)
        """
        num_targets = len(predicted_states)
        # 修正
        measurements_xy = [z[2:4] for z in measurements]
        num_measurements = len(measurements)
        
        # --- (A) トラックがまだ無い場合の処理 (追加) ---
        if num_targets == 0:
            # 全ての観測を新規トラックとして登録して返す
            new_tracks = []
            for z in measurements:
                new_tracks.append(self._init_new_track(z))
            return new_tracks, [] # infoは空で返す
        
        # 観測がない場合は miss_count をインクリメントして閾値チェック
        if num_measurements == 0:
            updated_states = []
            for state in predicted_states:
                new_miss_count = state.miss_count + 1
                # miss_countが閾値未満なら保持
                if new_miss_count < self.max_miss_count:
                    updated_states.append(GaussianState(state.mean, state.covariance, new_miss_count, track_id=state.track_id))
            return updated_states, []
            
        # --- ステップ1: ゲーティング (部品に委譲) ---
        validation_matrix = np.zeros((num_targets, num_measurements), dtype=int)
        validated_measurements_per_target = []
        
        for i, state in enumerate(predicted_states):
            # 注入された GatingModule を使用
            indices = self.gating_module.validate_measurements(state, measurements_xy)
            validated_measurements_per_target.append(indices)
            
            for j in indices:
                validation_matrix[i, j] = 1

        # --- ステップ2: アソシエーション確率計算 (部品に委譲) ---
        # 注入された AssociationCalculator を使用
        beta = self.association_calculator.calculate_probabilities(
            predicted_states, measurements_xy, validation_matrix
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
            innovations = [measurements_xy[j] - self.H @ state.mean for j in validated_indices]
            
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
            
            # miss_countの更新（デバッグ情報追加版）
            if validated_indices:  # ゲート内に観測があった場合
                new_miss_count = 0  # リセット
            else:  # 観測がなかった場合
                new_miss_count = state.miss_count + 1
            
            updated_states.append(GaussianState(updated_mean, updated_cov, new_miss_count, track_id=state.track_id))

        # --- (B) 新規トラック生成 (追加) ---
        # どのトラックのゲートにも入らなかった観測 (validation_matrixの列の和が0) を探す
        # validation_matrix: [Target x Measurement]
        # 列方向に足して 0 なら、誰のゲートにも入っていない
        
        is_assigned = validation_matrix.sum(axis=0) # [M]
        for j in range(num_measurements):
            if is_assigned[j] == 0:
                # 誰とも紐付かなかった観測 j -> 新規トラックへ
                z = measurements[j]
                new_track = self._init_new_track(z)
                updated_states.append(new_track)

        # --- (C) トラック削除処理 ---
        # miss_countが閾値以上のトラックを削除
        updated_states = [s for s in updated_states if s.miss_count < self.max_miss_count]
        return updated_states, validated_measurements_per_target

    def _init_new_track(self, z: np.ndarray) -> GaussianState:
            """
            観測値から新しいトラックを生成するヘルパーメソッド
            (ここを追加し忘れていたのがエラーの原因です)
            """
            if len(z) >= 5:
                x = z[2]
                y = z[3]
                velocity = z[4]
                #angle = np.deg2rad(z[1])
                angle = z[1]
                
                # 速度成分の初期化
                # ここでは単純に観測された速度(Velocity)が観測角度(Angle)方向に向いていると仮定して分解します
                # ※ターゲットの移動方向が位置ベクトルと一致しない場合（横切る場合など）は誤差になりますが、
                #   情報がない場合の初期値としては0よりは有効な場合があります。
                vx = velocity * np.cos(np.radians(angle))
                vy = velocity * np.sin(np.radians(angle))
                init_mean = np.array([x, y, vx, vy])
            else:
                # 従来のフォールバック（または情報が足りない場合）
                # z が [x, y] の場合など
                if len(z) == 2:
                    init_mean = np.array([z[0], z[1], 0.0, 0.0])
                else:
                    # z が [Range, Angle, X, Y] 等で Velocity がない場合
                    # あるいは安全策として
                    init_mean = np.array([z[2], z[3], 0.0, 0.0])
            
            # 共分散の初期値
            # 位置の分散は観測ノイズR程度、速度の分散は「わからない」ので大きくする
            # R = [[r, 0], [0, r]] と仮定して対角成分を取り出す
            r_var_x = self.R[0, 0]
            r_var_y = self.R[1, 1]
            v_var = 100.0 # 速度の初期不確かさ（大きめに設定）
            
            init_cov = np.diag([r_var_x, r_var_y, v_var, v_var])
            # 修正
            # 新しいトラックIDを割り当て
            new_track_id = self.next_track_id
            self.next_track_id += 1  # インクリメント
            return GaussianState(init_mean, init_cov, miss_count=0, track_id=new_track_id)