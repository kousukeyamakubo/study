"""
A-MOTA (Association-based MOTA) Calculator

IDが割り当てられていない観測データ（measurements.csv）と
真値（true_trajectory.csv）の比較を行う

A-MOTA = 1 - (FP + FN) / GT

FP: False Positives (存在しないターゲットの検出)
FN: False Negatives (存在するターゲットの未検出)
GT: Ground Truth (真のターゲット数の合計)

※ IDスイッチ（IDSW）は考慮しない
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.optimize import linear_sum_assignment

class AMOTACalculator:
    def __init__(self, csv_dir: str = "csv_result", distance_threshold: float = 2.0):
        """
        Args:
            csv_dir: CSVファイルのディレクトリ
            distance_threshold: マッチング距離閾値（メートル）
        """
        self.csv_dir = Path(csv_dir)
        self.distance_threshold = distance_threshold
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """真値と観測データのCSVを読み込む"""
        true_path = self.csv_dir / "true_trajectory.csv"
        meas_path = self.csv_dir / "measurements.csv"
        
        true_traj = pd.read_csv(true_path)
        measurements = pd.read_csv(meas_path)
        
        return true_traj, measurements
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """2点間のユークリッド距離を計算"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def match_detections_at_time(self, true_at_t: pd.DataFrame, 
                                 meas_at_t: pd.DataFrame) -> Tuple[int, List[float]]:
        """
        特定時刻での真値と観測のマッチングを実行
        
        Returns:
            tuple: (マッチ数, マッチした距離のリスト)
        """
        if len(true_at_t) == 0 or len(meas_at_t) == 0:
            return 0, []
        
        # 距離行列を作成
        n_true = len(true_at_t)
        n_meas = len(meas_at_t)
        
        cost_matrix = np.zeros((n_true, n_meas))
        
        for i, (_, true_row) in enumerate(true_at_t.iterrows()):
            true_pos = np.array([true_row['x'], true_row['y']])
            for j, (_, meas_row) in enumerate(meas_at_t.iterrows()):
                meas_pos = np.array([meas_row['x'], meas_row['y']])
                cost_matrix[i, j] = self.calculate_distance(true_pos, meas_pos)
        
        # ハンガリアンアルゴリズムで最適マッチング
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 距離閾値以内のマッチングのみ採用
        num_matches = 0
        matched_distances = []
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.distance_threshold:
                num_matches += 1
                matched_distances.append(cost_matrix[i, j])
        
        return num_matches, matched_distances
    
    def calculate_a_mota(self) -> Dict[str, float]:
        """A-MOTAおよび関連メトリクスを計算"""
        true_traj, measurements = self.load_data()
        
        # 時刻のリストを取得
        all_times = sorted(set(true_traj['time'].unique()) | set(measurements['time'].unique()))
        
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        total_gt = 0  # Ground Truth objects
        total_matches = 0
        all_matched_distances = []
        
        # 時刻ごとの詳細結果
        time_details = []
        
        for t in all_times:
            true_at_t = true_traj[true_traj['time'] == t]
            meas_at_t = measurements[measurements['time'] == t]
            
            n_true = len(true_at_t)
            n_meas = len(meas_at_t)
            
            total_gt += n_true
            
            # マッチング実行
            num_matches, matched_distances = self.match_detections_at_time(true_at_t, meas_at_t)
            
            # FN: マッチしなかった真のターゲット
            fn = n_true - num_matches
            total_fn += fn
            
            # FP: マッチしなかった観測
            fp = n_meas - num_matches
            total_fp += fp
            
            total_matches += num_matches
            all_matched_distances.extend(matched_distances)
            
            # 時刻ごとの詳細を記録
            time_details.append({
                'time': t,
                'n_true': n_true,
                'n_meas': n_meas,
                'matches': num_matches,
                'FN': fn,
                'FP': fp
            })
        
        # A-MOTA計算
        if total_gt > 0:
            a_mota = 1.0 - (total_fp + total_fn) / total_gt
        else:
            a_mota = 0.0
        
        # A-MOTP (平均マッチング距離)
        a_motp = np.mean(all_matched_distances) if all_matched_distances else 0.0
        
        # 検出率とFP率
        detection_rate = total_matches / total_gt if total_gt > 0 else 0.0
        fp_rate = total_fp / total_gt if total_gt > 0 else 0.0
        
        results = {
            'A-MOTA': a_mota,
            'A-MOTP': a_motp,
            'Detection Rate': detection_rate,
            'FP Rate': fp_rate,
            'FP': total_fp,
            'FN': total_fn,
            'GT': total_gt,
            'Total Matches': total_matches,
            'Time Details': time_details
        }
        
        return results
    
    def print_results(self, results: Dict[str, float], show_details: bool = False):
        """結果を整形して表示"""
        print("\n" + "="*60)
        print("A-MOTA Calculation Results")
        print("(Measurements vs True Trajectory)")
        print("="*60)
        print(f"A-MOTA (Association-based MOTA): {results['A-MOTA']:.4f} ({results['A-MOTA']*100:.2f}%)")
        print(f"A-MOTP (Average Matching Distance): {results['A-MOTP']:.4f} m")
        print("-"*60)
        print(f"Detection Rate:        {results['Detection Rate']:.4f} ({results['Detection Rate']*100:.2f}%)")
        print(f"False Positive Rate:   {results['FP Rate']:.4f} ({results['FP Rate']*100:.2f}%)")
        print("-"*60)
        print(f"False Positives (FP):  {results['FP']:6d}")
        print(f"False Negatives (FN):  {results['FN']:6d}")
        print(f"Ground Truth (GT):     {results['GT']:6d}")
        print(f"Total Matches:         {results['Total Matches']:6d}")
        print("="*60)
        
        if show_details:
            print("\nTime-wise Details:")
            print("-"*60)
            print(f"{'Time':>6} {'GT':>5} {'Meas':>5} {'Match':>6} {'FN':>4} {'FP':>4}")
            print("-"*60)
            for detail in results['Time Details']:
                print(f"{detail['time']:6d} {detail['n_true']:5d} {detail['n_meas']:5d} "
                      f"{detail['matches']:6d} {detail['FN']:4d} {detail['FP']:4d}")
            print("="*60 + "\n")


def main():
    """メイン関数"""
    # 距離閾値を設定（デフォルト: 2.0m）
    calculator = AMOTACalculator(csv_dir="csv_result", distance_threshold=3.0)
    
    print("Loading trajectories and measurements...")
    results = calculator.calculate_a_mota()
    
    # 詳細表示を有効にする場合は show_details=True
    calculator.print_results(results, show_details=True)


if __name__ == "__main__":
    main()