"""
MOTA (Multiple Object Tracking Accuracy) Calculator

MOTA = 1 - (FP + FN + IDSW) / GT

FP: False Positives (存在しないターゲットの検出)
FN: False Negatives (存在するターゲットの未検出)
IDSW: ID Switches (トラックIDの切り替わり)
GT: Ground Truth (真のターゲット数の合計)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple
from scipy.optimize import linear_sum_assignment

class MOTACalculator:
    def __init__(self, csv_dir: str = "csv_result", distance_threshold: float = 2.0):
        """
        Args:
            csv_dir: CSVファイルのディレクトリ
            distance_threshold: マッチング距離閾値（メートル）
        """
        self.csv_dir = Path(csv_dir)
        self.distance_threshold = distance_threshold
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """真値と推定値のCSVを読み込む"""
        true_path = self.csv_dir / "true_trajectory.csv"
        est_path = self.csv_dir / "estimated_trajectory.csv"
        
        true_traj = pd.read_csv(true_path)
        est_traj = pd.read_csv(est_path)
        
        return true_traj, est_traj
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """2点間のユークリッド距離を計算"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def match_tracks_at_time(self, true_at_t: pd.DataFrame, 
                            est_at_t: pd.DataFrame) -> Dict[int, int]:
        """
        特定時刻での真値と推定値のマッチングを実行
        
        Returns:
            dict: {true_target_id: estimated_track_id} のマッピング
        """
        if len(true_at_t) == 0 or len(est_at_t) == 0:
            return {}
        
        # 距離行列を作成
        n_true = len(true_at_t)
        n_est = len(est_at_t)
        
        cost_matrix = np.zeros((n_true, n_est))
        
        for i, (_, true_row) in enumerate(true_at_t.iterrows()):
            true_pos = np.array([true_row['x'], true_row['y']])
            for j, (_, est_row) in enumerate(est_at_t.iterrows()):
                est_pos = np.array([est_row['x'], est_row['y']])
                cost_matrix[i, j] = self.calculate_distance(true_pos, est_pos)
        
        # ハンガリアンアルゴリズムで最適マッチング
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 距離閾値以内のマッチングのみ採用
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.distance_threshold:
                true_id = true_at_t.iloc[i]['target_id']
                est_id = est_at_t.iloc[j]['target_id']
                matches[true_id] = est_id
        
        return matches
    
    def calculate_mota(self) -> Dict[str, float]:
        """MOTAおよび関連メトリクスを計算"""
        true_traj, est_traj = self.load_data()
        
        # 時刻のリストを取得
        all_times = sorted(set(true_traj['time'].unique()) | set(est_traj['time'].unique()))
        
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        total_idsw = 0  # ID Switches
        total_gt = 0  # Ground Truth objects
        
        # 前回のマッチング結果を保存（IDスイッチ検出用）
        prev_matches = {}
        
        for t in all_times:
            true_at_t = true_traj[true_traj['time'] == t]
            est_at_t = est_traj[est_traj['time'] == t]
            
            n_true = len(true_at_t)
            n_est = len(est_at_t)
            
            total_gt += n_true
            
            # マッチング実行
            matches = self.match_tracks_at_time(true_at_t, est_at_t)
            
            # FN: マッチしなかった真のターゲット
            total_fn += n_true - len(matches)
            
            # FP: マッチしなかった推定トラック
            total_fp += n_est - len(matches)
            
            # ID Switch検出
            for true_id, est_id in matches.items():
                if true_id in prev_matches:
                    # 同じ真のターゲットが前回と異なる推定IDにマッチした
                    if prev_matches[true_id] != est_id:
                        total_idsw += 1
            
            prev_matches = matches.copy()
        
        # MOTA計算
        if total_gt > 0:
            mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt
        else:
            mota = 0.0
        
        # MOTP (Multiple Object Tracking Precision) も計算
        total_matched_distance = 0
        total_matches = 0
        
        for t in all_times:
            true_at_t = true_traj[true_traj['time'] == t]
            est_at_t = est_traj[est_traj['time'] == t]
            
            matches = self.match_tracks_at_time(true_at_t, est_at_t)
            
            for true_id, est_id in matches.items():
                true_pos = true_at_t[true_at_t['target_id'] == true_id][['x', 'y']].values[0]
                est_pos = est_at_t[est_at_t['target_id'] == est_id][['x', 'y']].values[0]
                total_matched_distance += self.calculate_distance(true_pos, est_pos)
                total_matches += 1
        
        motp = total_matched_distance / total_matches if total_matches > 0 else 0.0
        
        results = {
            'MOTA': mota,
            'MOTP': motp,
            'FP': total_fp,
            'FN': total_fn,
            'IDSW': total_idsw,
            'GT': total_gt,
            'Total Matches': total_matches
        }
        
        return results
    
    def print_results(self, results: Dict[str, float]):
        """結果を整形して表示"""
        print("\n" + "="*50)
        print("MOTA Calculation Results")
        print("="*50)
        print(f"MOTA (Multiple Object Tracking Accuracy): {results['MOTA']:.4f} ({results['MOTA']*100:.2f}%)")
        print(f"MOTP (Multiple Object Tracking Precision): {results['MOTP']:.4f} m")
        print("-"*50)
        print(f"False Positives (FP):  {results['FP']:6d}")
        print(f"False Negatives (FN):  {results['FN']:6d}")
        print(f"ID Switches (IDSW):    {results['IDSW']:6d}")
        print(f"Ground Truth (GT):     {results['GT']:6d}")
        print(f"Total Matches:         {results['Total Matches']:6d}")
        print("="*50 + "\n")


def main():
    """メイン関数"""
    calculator = MOTACalculator(csv_dir="csv_result", distance_threshold=3.0)
    
    print("Loading trajectories...")
    results = calculator.calculate_mota()
    
    calculator.print_results(results)


if __name__ == "__main__":
    main()