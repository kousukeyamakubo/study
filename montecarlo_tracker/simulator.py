import numpy as np
from typing import List, Tuple, Optional

# 新しいパスからのインポート
from core.state import GaussianState
from interfaces.tracker import ITracker
# utilsフォルダにあると仮定（後ほど修正確認）
from utils.data_saver import ResultSaver 

class TrackingSimulator:
    """
    汎用的なトラッキングシミュレータ
    ITrackerインターフェースを持つ任意のアルゴリズムを実行可能
    """
    
    def __init__(self, tracker: ITracker):
        """
        Args:
            tracker: ITrackerを実装した追跡アルゴリズム（JPDAFなど）
        """
        self.tracker = tracker
    
    def run(self,
            initial_states: List[np.ndarray],
            num_steps: int, 
            measurements_list: List[List[np.ndarray]],
            true_trajectories: List[np.ndarray] = None,
            verbose: bool = True,
            USE_CSV_MODE: bool = False) -> Tuple:
        """
        シミュレーションを実行
        """
        
        if USE_CSV_MODE:
            if verbose:
                print("Using CSV mode, skipping true trajectory generation...")
        else:
            if verbose:
                print("Using provided true trajectories...")

        # --- 初期推定値の作成 ---
        # 真の状態に少しノイズを乗せて初期推定値とする
        # (注: 本来はここもConfigの初期共分散パラメータを使うべきですが、簡易的に固定値または引数化)
        initial_estimates = [
            GaussianState(
                mean=state + np.random.randn(len(state)) * 0.5,
                covariance=np.eye(len(state)) * 2.0
            ) for state in initial_states
        ]
        
        if verbose:
            print("Running tracking simulation...")
        
        # 結果格納用
        # estimated_trajectories: [target_id][step] -> GaussianState
        estimated_trajectories = [[est] for est in initial_estimates]
        
        # バリデーション情報（デバッグ・解析用）
        # info_list: [step] -> info content
        info_list = [] 
        
        current_estimates = initial_estimates
        all_estimated_states = [current_estimates] # [step][target_idx]
        
        # --- メインループ ---
        for i in range(num_steps):
            meas_at_t = measurements_list[i]
            
            # 1. 予測 (Trackerに委譲)
            predicted_states = self.tracker.predict(current_estimates)
            
            # 2. 更新 (Trackerに委譲)
            # 戻り値の info はアルゴリズム依存（JPDAなら有効観測indexなど）
            updated_states, info = self.tracker.update(predicted_states, meas_at_t)
            
            info_list.append(info)
            all_estimated_states.append(updated_states)
            current_estimates = updated_states
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Step {i + 1}/{num_steps}")
        
        # --- 結果の整形 ---
        # [Time, Target] の形式を [Target, Time] に変換
        #final_estimated_trajectories = [[] for _ in range(len(initial_states))]
        
        # 修正ポイント(12/22)
        step_states_list = all_estimated_states[1:]
        
        
        # 修正後
        max_targets =0
        # 修正ポイント(12/22)
        #for states_at_t in all_estimated_states:
        for states_at_t in step_states_list:
            max_targets = max(max_targets, len(states_at_t))
            
        final_estimated_trajectories = [[] for _ in range(max_targets)]
        # 修正ポイント(12/22)
        """for states_at_t in all_estimated_states:
            for target_idx, state in enumerate(states_at_t):
                # 念のため、インデックスが範囲外なら拡張する処理を入れておく（安全策）
                while len(final_estimated_trajectories) <= target_idx:
                    final_estimated_trajectories.append([])
                final_estimated_trajectories[target_idx].append(state)
        """
        for states_at_t in step_states_list:
            for target_idx in range(max_targets):
                # ターゲットが存在する場合はその状態を、存在しない場合は None を追加
                if target_idx < len(states_at_t):
                    final_estimated_trajectories[target_idx].append(states_at_t[target_idx])
                else:
                    final_estimated_trajectories[target_idx].append(None)
                    
        #　オクルージョンが発生するとRMSE計算ができないのでいったんコメントアウト
        #if verbose and true_trajectories is not None:
            #self._print_results(true_trajectories, final_estimated_trajectories)
        
        return final_estimated_trajectories, info_list
    
    def _print_results(self, true_trajectories: List[np.ndarray], 
                      estimated_trajectories: List[List[GaussianState]]):
        """結果のRMSEなどを表示"""
        print("\nFinal Results:")
        
        num_targets = len(true_trajectories)
        if num_targets == 0:
            print("No targets to evaluate.")
            return
            
        num_steps = len(true_trajectories[0])
        total_squared_error = 0.0
        
        for i in range(num_targets):
            # 位置(x, y)のみの誤差を評価と仮定（先頭2次元）
            true_final = true_trajectories[i][-1][:2]
            est_final = estimated_trajectories[i][-1].mean[:2]
            
            target_squared_error = 0.0
            for t in range(num_steps):
                true_pos = true_trajectories[i][t][:2]
                est_pos = estimated_trajectories[i][t].mean[:2]
                squared_error = np.sum((true_pos - est_pos) ** 2)
                target_squared_error += squared_error
            
            mse_target = target_squared_error / num_steps
            rmse_target = np.sqrt(mse_target)
            total_squared_error += target_squared_error
            
            print(f"  Target {i}:")
            print(f"    RMSE: {rmse_target:.3f}")
            
        mse_total = total_squared_error / (num_targets * num_steps)
        rmse_total = np.sqrt(mse_total)
        
        print("-" * 20)
        print(f"  Overall Position RMSE: {rmse_total:.3f}")