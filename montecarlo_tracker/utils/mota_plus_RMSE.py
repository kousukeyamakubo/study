"""
MOTA (Multiple Object Tracking Accuracy) Calculator with RMSE

真値 vs 推定値（MOTA）、および 真値 vs 観測（A-MOTA）の両方を計算可能
さらに、マッチングペア間での位置・速度RMSEも計算

MOTA = 1 - (FP + FN + IDSW) / GT
A-MOTA = 1 - (FP + FN) / GT  ← IDSWを除外（ID情報がない場合に使用）

FP: False Positives (存在しないターゲットの検出)
FN: False Negatives (存在するターゲットの未検出)
IDSW: ID Switches (トラックIDの切り替わり)
GT: Ground Truth (真のターゲット数の合計)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
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
        
    def load_data(self, include_measurements: bool = False) -> Tuple:
        """真値と推定値のCSVを読み込む
        
        Args:
            include_measurements: measurements.csvも読み込むか
            
        Returns:
            include_measurements=False: (true_traj, est_traj)
            include_measurements=True:  (true_traj, est_traj, measurements)
        """
        true_path = self.csv_dir / "true_trajectory.csv"
        est_path = self.csv_dir / "estimated_trajectory.csv"
        
        true_traj = pd.read_csv(true_path)
        est_traj = pd.read_csv(est_path)
        
        if include_measurements:
            meas_path = self.csv_dir / "measurements.csv"
            measurements = pd.read_csv(meas_path)
            return true_traj, est_traj, measurements
        
        return true_traj, est_traj
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """2点間のユークリッド距離を計算"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_velocity_magnitude(self, row: pd.Series) -> Optional[float]:
        """
        速度の大きさを取得
        velocity列があればそれを使用、なければvx,vyから計算
        
        Args:
            row: データ行
            
        Returns:
            速度の大きさ、またはNone（データがない場合）
        """
        if 'velocity' in row.index and pd.notna(row['velocity']):
            return float(row['velocity'])
        elif 'vx' in row.index and 'vy' in row.index and pd.notna(row['vx']) and pd.notna(row['vy']):
            return np.sqrt(row['vx']**2 + row['vy']**2)
        else:
            return None
    
    def match_tracks_at_time(self, true_at_t: pd.DataFrame, 
                            est_at_t: pd.DataFrame,
                            has_target_id: bool = True) -> Dict:
        """
        特定時刻での真値と推定値/観測のマッチングを実行
        
        Args:
            true_at_t: 真値データ（時刻tでフィルタ済み）
            est_at_t: 推定値または観測データ（時刻tでフィルタ済み）
            has_target_id: est_at_tにtarget_idカラムがあるか
        
        Returns:
            dict: {true_target_id: (est_track_id or est_index, distance)} のマッピング
        """
        if len(true_at_t) == 0 or len(est_at_t) == 0:
            return {}
        
        # 距離行列を作成
        n_true = len(true_at_t)
        n_est = len(est_at_t)
        
        cost_matrix = np.zeros((n_true, n_est))
        
        true_at_t_reset = true_at_t.reset_index(drop=True)
        est_at_t_reset = est_at_t.reset_index(drop=True)
        
        for i in range(n_true):
            true_pos = np.array([true_at_t_reset.loc[i, 'x'], 
                                true_at_t_reset.loc[i, 'y']])
            for j in range(n_est):
                est_pos = np.array([est_at_t_reset.loc[j, 'x'], 
                                   est_at_t_reset.loc[j, 'y']])
                cost_matrix[i, j] = self.calculate_distance(true_pos, est_pos)
        
        # ハンガリアンアルゴリズムで最適マッチング
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 距離閾値以内のマッチングのみ採用
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.distance_threshold:
                true_id = true_at_t_reset.loc[i, 'target_id']
                if has_target_id:
                    est_id = est_at_t_reset.loc[j, 'target_id']
                else:
                    # 観測の場合はインデックスを使用
                    est_id = j
                matches[true_id] = (est_id, cost_matrix[i, j])
        
        return matches
    
    def calculate_rmse_from_pairs(self, true_data: pd.DataFrame, 
                                  est_data: pd.DataFrame,
                                  matches: Dict,
                                  has_target_id: bool = True) -> Tuple[List[np.ndarray], List[float]]:
        """
        マッチングペア間の位置・速度RMSEを計算
        
        Args:
            true_data: 真値データ（時刻tでフィルタ済み）
            est_data: 推定値/観測データ（時刻tでフィルタ済み）
            matches: マッチング結果 {true_id: (est_id, distance)}
            has_target_id: est_dataにtarget_idがあるか
            
        Returns:
            (position_errors, velocity_errors): 誤差のリスト
        """
        position_errors = []
        velocity_errors = []
        
        true_data_reset = true_data.reset_index(drop=True)
        est_data_reset = est_data.reset_index(drop=True)
        
        for true_id, (est_id, _) in matches.items():
            # 真値データの取得
            true_row = true_data_reset[true_data_reset['target_id'] == true_id]
            if len(true_row) == 0:
                continue
            true_row = true_row.iloc[0]
            
            # 推定値/観測データの取得
            if has_target_id:
                est_row = est_data_reset[est_data_reset['target_id'] == est_id]
                if len(est_row) == 0:
                    continue
                est_row = est_row.iloc[0]
            else:
                # 観測の場合はインデックスで取得
                est_row = est_data_reset.iloc[est_id]
            
            # 位置誤差
            pos_error = np.array([true_row['x'] - est_row['x'], 
                                 true_row['y'] - est_row['y']])
            position_errors.append(pos_error)
            
            # 速度誤差（velocity列またはvx,vyから計算）
            true_vel = self.get_velocity_magnitude(true_row)
            est_vel = self.get_velocity_magnitude(est_row)
            
            if true_vel is not None and est_vel is not None:
                vel_error = true_vel - est_vel
                velocity_errors.append(vel_error)
        
        return position_errors, velocity_errors
    
    def calculate_mota_with_id(self, true_traj: pd.DataFrame, 
                               est_traj: pd.DataFrame) -> Dict[str, float]:
        """推定値（IDあり）とのMOTA計算 + RMSE計算"""
        
        all_times = sorted(set(true_traj['time'].unique()) | set(est_traj['time'].unique()))
        
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_gt = 0
        
        track_history = {}
        
        total_matched_distance = 0
        total_matches = 0
        
        # RMSE計算用
        all_position_errors = []
        all_velocity_errors = []
        times_without_fn = []  # 検出漏れがない時刻のリスト
        
        for t in all_times:
            true_at_t = true_traj[true_traj['time'] == t]
            est_at_t = est_traj[est_traj['time'] == t]
            
            n_true = len(true_at_t)
            n_est = len(est_at_t)
            
            total_gt += n_true
            
            matches = self.match_tracks_at_time(true_at_t, est_at_t, has_target_id=True)
            
            fn_at_t = n_true - len(matches)
            total_fn += fn_at_t
            total_fp += n_est - len(matches)
            
            # 検出漏れがない時刻のみRMSE計算対象
            if fn_at_t == 0 and len(matches) > 0:
                times_without_fn.append(t)
                pos_errors, vel_errors = self.calculate_rmse_from_pairs(
                    true_at_t, est_at_t, matches, has_target_id=True)
                all_position_errors.extend(pos_errors)
                all_velocity_errors.extend(vel_errors)
            
            # ID Switch検出
            for true_id, (est_id, dist) in matches.items():
                total_matched_distance += dist
                total_matches += 1
                
                if true_id in track_history:
                    if est_id not in track_history[true_id]:
                        total_idsw += 1
                        track_history[true_id].add(est_id)
                else:
                    track_history[true_id] = {est_id}
        
        # RMSEの計算
        position_rmse = self._compute_rmse(all_position_errors) if all_position_errors else np.nan
        velocity_rmse = self._compute_rmse_scalar(all_velocity_errors) if all_velocity_errors else np.nan
        
        mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt if total_gt > 0 else 0.0
        motp = total_matched_distance / total_matches if total_matches > 0 else 0.0
        
        return {
            'MOTA': mota,
            'MOTP': motp,
            'FP': total_fp,
            'FN': total_fn,
            'IDSW': total_idsw,
            'GT': total_gt,
            'Total Matches': total_matches,
            'Position RMSE': position_rmse,
            'Velocity RMSE': velocity_rmse,
            'Times without FN': len(times_without_fn),
            'Total Times': len(all_times)
        }
    
    def calculate_amota_without_id(self, true_traj: pd.DataFrame, 
                                   measurements: pd.DataFrame) -> Dict[str, float]:
        """観測（IDなし）とのA-MOTA計算 + RMSE計算
        
        A-MOTA (Average MOTA) = 1 - (FP + FN) / GT
        
        観測にはIDが無いため、ID Switchesを考慮せずFPとFNのみで評価します。
        これは各時刻での検出性能の平均を表します。
        """
        
        all_times = sorted(set(true_traj['time'].unique()) | set(measurements['time'].unique()))
        
        total_fp = 0
        total_fn = 0
        total_gt = 0
        
        total_matched_distance = 0
        total_matches = 0
        
        # RMSE計算用
        all_position_errors = []
        all_velocity_errors = []
        times_without_fn = []  # 検出漏れがない時刻のリスト
        
        for t in all_times:
            true_at_t = true_traj[true_traj['time'] == t]
            meas_at_t = measurements[measurements['time'] == t]
            
            n_true = len(true_at_t)
            n_meas = len(meas_at_t)
            
            total_gt += n_true
            
            matches = self.match_tracks_at_time(true_at_t, meas_at_t, has_target_id=False)
            
            fn_at_t = n_true - len(matches)
            total_fn += fn_at_t
            total_fp += n_meas - len(matches)
            
            # 検出漏れがない時刻のみRMSE計算対象
            if fn_at_t == 0 and len(matches) > 0:
                times_without_fn.append(t)
                pos_errors, vel_errors = self.calculate_rmse_from_pairs(
                    true_at_t, meas_at_t, matches, has_target_id=False)
                all_position_errors.extend(pos_errors)
                all_velocity_errors.extend(vel_errors)
            
            # マッチング距離の集計（MOTPのため）
            for true_id, (meas_idx, dist) in matches.items():
                total_matched_distance += dist
                total_matches += 1
        
        # RMSEの計算
        position_rmse = self._compute_rmse(all_position_errors) if all_position_errors else np.nan
        velocity_rmse = self._compute_rmse_scalar(all_velocity_errors) if all_velocity_errors else np.nan
        
        # A-MOTA計算（IDSWを含めない）
        amota = 1.0 - (total_fp + total_fn) / total_gt if total_gt > 0 else 0.0
        motp = total_matched_distance / total_matches if total_matches > 0 else 0.0
        
        return {
            'A-MOTA': amota,
            'MOTP': motp,
            'FP': total_fp,
            'FN': total_fn,
            'IDSW': 0,  # 観測にはIDがないため常に0
            'GT': total_gt,
            'Total Matches': total_matches,
            'Position RMSE': position_rmse,
            'Velocity RMSE': velocity_rmse,
            'Times without FN': len(times_without_fn),
            'Total Times': len(all_times)
        }
    
    def _compute_rmse(self, errors: List[np.ndarray]) -> float:
        """位置誤差（2次元ベクトル）からRMSEを計算"""
        if not errors:
            return np.nan
        errors_array = np.array(errors)  # shape: (N, 2)
        squared_errors = np.sum(errors_array ** 2, axis=1)  # 各ペアでx^2 + y^2
        return np.sqrt(np.mean(squared_errors))
    
    def _compute_rmse_scalar(self, errors: List[float]) -> float:
        """速度誤差（スカラー）からRMSEを計算"""
        if not errors:
            return np.nan
        errors_array = np.array(errors)
        return np.sqrt(np.mean(errors_array ** 2))
    
    def calculate_all_mota(self) -> Dict[str, Dict[str, float]]:
        """真値 vs 推定値（MOTA）と 真値 vs 観測（A-MOTA）の両方を計算"""
        
        true_traj, est_traj, measurements = self.load_data(include_measurements=True)
        
        print("Calculating MOTA for True vs Estimated...")
        results_est = self.calculate_mota_with_id(true_traj, est_traj)
        
        print("Calculating A-MOTA for True vs Measurements...")
        results_meas = self.calculate_amota_without_id(true_traj, measurements)
        
        return {
            'True vs Estimated': results_est,
            'True vs Measurements': results_meas
        }
    
    def print_results(self, all_results: Dict[str, Dict[str, float]]):
        """結果を整形して表示"""
        print("\n" + "="*70)
        print("MOTA / A-MOTA Calculation Results with RMSE")
        print("="*70)
        
        for comparison_name, results in all_results.items():
            print(f"\n【{comparison_name}】")
            print("-"*70)
            
            # MOTAかA-MOTAかを判定
            if 'MOTA' in results:
                print(f"MOTA (Multiple Object Tracking Accuracy): {results['MOTA']:.4f} ({results['MOTA']*100:.2f}%)")
            elif 'A-MOTA' in results:
                print(f"A-MOTA (Average MOTA, ID-free):           {results['A-MOTA']:.4f} ({results['A-MOTA']*100:.2f}%)")
            
            print(f"MOTP (Multiple Object Tracking Precision): {results['MOTP']:.4f} m")
            print(f"False Positives (FP):  {results['FP']:6d}")
            print(f"False Negatives (FN):  {results['FN']:6d}")
            
            if results['IDSW'] > 0 or 'MOTA' in results:
                print(f"ID Switches (IDSW):    {results['IDSW']:6d}")
            else:
                print(f"ID Switches (IDSW):    N/A (ID-free)")
            
            print(f"Ground Truth (GT):     {results['GT']:6d}")
            print(f"Total Matches:         {results['Total Matches']:6d}")
            
            # RMSE情報
            print("-"*70)
            print(f"Position RMSE:         {results['Position RMSE']:.4f} m")
            if not np.isnan(results['Velocity RMSE']):
                print(f"Velocity RMSE:         {results['Velocity RMSE']:.4f} m/s")
            else:
                print(f"Velocity RMSE:         N/A (no velocity data)")
            print(f"RMSE calculated from:  {results['Times without FN']} / {results['Total Times']} time steps (FN=0 only)")
        
        print("\n" + "="*70 + "\n")


def main():
    """メイン関数"""
    calculator = MOTACalculator(csv_dir="csv_result", distance_threshold=3.0)
    
    print("Loading trajectories and measurements...")
    all_results = calculator.calculate_all_mota()
    
    calculator.print_results(all_results)


if __name__ == "__main__":
    main()