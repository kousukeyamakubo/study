import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional


class RMSEComparator:
    """真値と推定値・測定値のRMSEを比較するクラス"""
    
    def __init__(self, csv_dir: str = "csv_result", distance_threshold: float = 5.0):
        """
        Args:
            csv_dir: CSVファイルのディレクトリ
            distance_threshold: マッチング閾値（メートル）。この距離より近いものがなければ検出漏れとする
        """
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"Directory {csv_dir} does not exist")
        self.distance_threshold = distance_threshold
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """CSVファイルからデータを読み込む"""
        true_traj = pd.read_csv(self.csv_dir / "true_trajectory.csv")
        est_traj = pd.read_csv(self.csv_dir / "estimated_trajectory.csv")
        measurements = pd.read_csv(self.csv_dir / "measurements.csv")
        
        # 測定値にvx, vyがない場合は計算
        if 'vx' not in measurements.columns and 'velocity' in measurements.columns and 'angle' in measurements.columns:
            measurements['vx'] = measurements['velocity'] * np.cos(np.radians(measurements['angle']))
            measurements['vy'] = measurements['velocity'] * np.sin(np.radians(measurements['angle']))
        
        return true_traj, est_traj, measurements
    
    def match_with_threshold(self, true_pos: np.ndarray, est_pos: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        閾値ベースのマッチングを行う
        
        Args:
            true_pos: 真値の位置 (N, 2)
            est_pos: 推定値の位置 (M, 2)
        
        Returns:
            マッチングされたペアのリスト [(true_idx, est_idx, distance), ...]
            閾値以内のマッチがない真値は含まれない（検出漏れとして扱う）
        """
        if len(true_pos) == 0 or len(est_pos) == 0:
            return []
        
        matches = []
        used_est_indices = set()
        
        # 各真値に対して最も近い推定値を探す
        for i, tp in enumerate(true_pos):
            min_dist = float('inf')
            best_j = -1
            
            for j, ep in enumerate(est_pos):
                if j in used_est_indices:
                    continue  # 既にマッチング済み
                
                dist = np.linalg.norm(tp - ep)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
            
            # 閾値以内であればマッチング
            if best_j >= 0 and min_dist <= self.distance_threshold:
                matches.append((i, best_j, min_dist))
                used_est_indices.add(best_j)
        
        return matches
    
    def calculate_rmse_for_dataset(self, true_traj: pd.DataFrame, 
                                   compare_traj: pd.DataFrame,
                                   dataset_name: str) -> Dict:
        """
        データセット全体のRMSEを計算
        
        Args:
            true_traj: 真値の軌道
            compare_traj: 比較対象の軌道（推定値または測定値）
            dataset_name: データセット名
        
        Returns:
            RMSE結果の辞書
        """
        position_errors = []
        velocity_errors = []
        matched_counts = []
        missed_detections = []
        false_alarms = []
        
        # 全タイムステップを取得
        true_times = set(true_traj['time'].unique())
        comp_times = set(compare_traj['time'].unique())
        all_times = sorted(true_times | comp_times)
        
        for t in all_times:
            # 各タイムステップのデータを取得
            true_t = true_traj[true_traj['time'] == t]
            comp_t = compare_traj[compare_traj['time'] == t]
            
            n_true = len(true_t)
            n_comp = len(comp_t)
            
            if n_true == 0:
                # 真値がない場合（通常は発生しない）
                false_alarms.append(n_comp)
                continue
            
            if n_comp == 0:
                # 推定値/測定値がない場合は全て検出漏れ
                missed_detections.append(n_true)
                continue
            
            # 位置データを取得
            true_pos = true_t[['x', 'y']].values
            comp_pos = comp_t[['x', 'y']].values
            
            # 閾値ベースのマッチング
            matches = self.match_with_threshold(true_pos, comp_pos)
            
            n_matches = len(matches)
            matched_counts.append(n_matches)
            
            # 検出漏れ数（マッチしなかった真値の数）
            n_missed = n_true - n_matches
            if n_missed > 0:
                missed_detections.append(n_missed)
            
            # 誤警報数（マッチしなかった推定値/測定値の数）
            n_false_alarm = n_comp - n_matches
            if n_false_alarm > 0:
                false_alarms.append(n_false_alarm)
            
            # マッチングしたペアの誤差を計算
            for true_idx, comp_idx, dist in matches:
                true_row = true_t.iloc[true_idx]
                comp_row = comp_t.iloc[comp_idx]
                
                # 位置誤差
                pos_error = np.sqrt((true_row['x'] - comp_row['x'])**2 + 
                                  (true_row['y'] - comp_row['y'])**2)
                position_errors.append(pos_error)
                
                # 速度誤差（両方にvx, vyがある場合のみ）
                if 'vx' in comp_t.columns and 'vy' in comp_t.columns:
                    # 真値の速度を計算（velocityとangleから）
                    if 'velocity' in true_t.columns and 'angle' in true_t.columns:
                        true_vx = true_row['velocity'] * np.cos(np.radians(true_row['angle']))
                        true_vy = true_row['velocity'] * np.sin(np.radians(true_row['angle']))
                    else:
                        continue
                    
                    vel_error = np.sqrt((true_vx - comp_row['vx'])**2 + 
                                      (true_vy - comp_row['vy'])**2)
                    velocity_errors.append(vel_error)
        
        # RMSEを計算
        results = {
            'dataset': dataset_name,
            'distance_threshold': self.distance_threshold,
            'position_rmse': np.sqrt(np.mean(np.array(position_errors)**2)) if position_errors else None,
            'velocity_rmse': np.sqrt(np.mean(np.array(velocity_errors)**2)) if velocity_errors else None,
            'total_matches': sum(matched_counts),
            'total_missed_detections': sum(missed_detections),
            'total_false_alarms': sum(false_alarms),
            'timesteps_with_matches': len(matched_counts),
            'total_timesteps': len(all_times)
        }
        
        return results
    
    def compare_all(self) -> pd.DataFrame:
        """全データセットのRMSEを比較"""
        true_traj, est_traj, measurements = self.load_data()
        
        results = []
        
        # 推定値のRMSE
        print(f"Calculating RMSE for estimated trajectory (threshold: {self.distance_threshold}m)...")
        est_results = self.calculate_rmse_for_dataset(true_traj, est_traj, "Estimated")
        results.append(est_results)
        
        # 測定値のRMSE
        print(f"Calculating RMSE for measurements (threshold: {self.distance_threshold}m)...")
        meas_results = self.calculate_rmse_for_dataset(true_traj, measurements, "Measurements")
        results.append(meas_results)
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def print_results(self, results_df: pd.DataFrame):
        """結果を見やすく表示"""
        print("\n" + "="*70)
        print("RMSE Comparison Results")
        print("="*70)
        
        for _, row in results_df.iterrows():
            print(f"\n{row['dataset']}:")
            print(f"  Distance threshold: {row['distance_threshold']:.2f} m")
            print(f"  Position RMSE: {row['position_rmse']:.4f} m" if row['position_rmse'] is not None else "  Position RMSE: N/A")
            print(f"  Velocity RMSE: {row['velocity_rmse']:.4f} m/s" if row['velocity_rmse'] is not None else "  Velocity RMSE: N/A")
            print(f"  Matched pairs: {row['total_matches']}")
            print(f"  Missed detections: {row['total_missed_detections']}")
            print(f"  False alarms: {row['total_false_alarms']}")
            print(f"  Timesteps with matches: {row['timesteps_with_matches']}/{row['total_timesteps']}")
        
        print("\n" + "="*70)


def main():
    """メイン関数"""
    # 閾値を設定（デフォルト: 5.0メートル）
    # 必要に応じて変更してください
    DISTANCE_THRESHOLD = 3.0
    
    comparator = RMSEComparator(csv_dir="csv_result", distance_threshold=DISTANCE_THRESHOLD)
    
    # RMSE比較を実行
    results_df = comparator.compare_all()
    
    # 結果を表示
    comparator.print_results(results_df)
    
    # 結果をCSVに保存
    output_path = Path("csv_result") / "rmse_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()