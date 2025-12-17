import csv
from pathlib import Path
import numpy as np
from typing import List

# ★ パス修正
from core.state import GaussianState

class ResultSaver:
    """結果をCSVファイルに保存するクラス"""
    
    def __init__(self, output_dir: str = "csv_result"):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True)
    
    def save_true_trajectory(self, trajectories: List[np.ndarray], 
                             filename: str = "true_trajectory.csv"):
        """複数の真の軌道を保存"""
        filepath = self.output_path / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "target_id", "x", "y","range", "angle", "velocity"])
            for i, trajectory in enumerate(trajectories):
                for t, state in enumerate(trajectory):
                    writer.writerow([t, i, state[0], state[1], state[2], state[3], state[4]])
        
        print(f"True trajectory saved to {filepath}")
    
    def save_estimated_trajectory(self, trajectories: List[List[GaussianState]], 
                                  filename: str = "estimated_trajectory.csv"):
        """複数の推定軌道を保存"""
        filepath = self.output_path / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "target_id", "x", "y", "vx", "vy", "cov_xx", "cov_yy"])
            for i, trajectory in enumerate(trajectories):
                for t, state in enumerate(trajectory):
                    writer.writerow([t, i, state.mean[0], state.mean[1], state.mean[2], 
                                   state.mean[3], state.covariance[0, 0], state.covariance[1, 1]])
        
        print(f"Estimated trajectory saved to {filepath}")
    
    def save_measurements(self, measurements_list: List[List[np.ndarray]], 
                         filename: str = "measurements.csv"):
        """観測データを保存"""
        filepath = self.output_path / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "range", "angle", "x","y", "velocity"])
            # JPDAFでは time=0 は観測がないため t+1??????????
            for t, measurements in enumerate(measurements_list):
                for j, z in enumerate(measurements):
                    writer.writerow([t, z[0], z[1],z[2],z[3],z[4]])
        
        print(f"Measurements saved to {filepath}")
    
    def save_validated_measurements(self, validated_list: List[List[int]], 
                                   measurements_list: List[List[np.ndarray]],
                                   filename: str = "validated_measurements.csv"):
        """
        ゲーティング処理後の有効な観測を保存
        (注: JPDAFでは T x M の行列になるため、この形式は不正確になるが互換性のため残す)
        """
        filepath = self.output_path / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "measurement_id", "x", "y", "is_validated_for_T0"])
            
            time_offset = 1 # 観測はt=1から始まる想定 (measurements.csv と合わせる)

            if len(validated_list) != len(measurements_list):
                print(f"Warning (save_validated): Mismatch length between measurements ({len(measurements_list)}) and validated_list ({len(validated_list)})")

            # zip は短い方に合わせる
            for t, (measurements, validated_indices_at_t) in enumerate(zip(measurements_list, validated_list)):
                if not measurements:
                    continue
                
                # validated_indices_at_t は (T, M_validated) のリスト
                # e.g., [[0, 2], [1, 2], [0, 1, 3]]
                
                indices_for_t0 = set()
                if validated_indices_at_t: # 空リストでない場合
                    try:
                        indices_for_t0 = set(validated_indices_at_t[0])
                    except IndexError:
                        # ターゲットが存在しない (T=0) か、形式が違う場合
                        pass 
                            
                for j, z in enumerate(measurements):
                    is_validated = 1 if j in indices_for_t0 else 0
                    writer.writerow([t + time_offset, j, z[0], z[1], is_validated])
        
        print(f"Validated measurements saved to {filepath} (Simplified for JPDA, showing T0 only)")

    
    def save_all(self, true_trajectory: List[np.ndarray], 
                 estimated_trajectory: List[List[GaussianState]],
                 measurements_list: List[List[np.ndarray]], 
                 validated_list: List[List[int]] = None):
        """全ての結果を保存"""
        self.save_true_trajectory(true_trajectory)
        self.save_estimated_trajectory(estimated_trajectory)
        self.save_measurements(measurements_list)
        
        if validated_list is not None:
            # JPDAFでは validated_list の意味合いが変わるため注意
            # simulator.py が (N_steps-1, T, M_validated) のリストを渡す
            self.save_validated_measurements(validated_list, measurements_list) # コメントアウト解除
        
        print(f"\nAll results saved to {self.output_path.absolute()}")