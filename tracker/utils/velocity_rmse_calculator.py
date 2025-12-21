import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

class VelocityRMSECalculator:
    """
    視線速度 (Radial Velocity) のRMSEを計算するクラス
    真値 vs センシング結果 (Measurements)
    真値 vs トラッキング結果 (Estimated Trajectory)
    """

    def __init__(self, csv_dir: str = "csv_result", radar_pos: Tuple[float, float, float] = (250, -18, 50)):
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.exists():
            # プロジェクトルートからの相対パスの可能性も考慮
            self.csv_dir = Path("tracker") / csv_dir
        
        self.radar_pos = np.array(radar_pos)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            true_df = pd.read_csv(self.csv_dir / "true_trajectory.csv")
            est_df = pd.read_csv(self.csv_dir / "estimated_trajectory.csv")
            meas_df = pd.read_csv(self.csv_dir / "measurements.csv")
            return true_df, est_df, meas_df
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            return None, None, None

    def calculate_estimated_radial_velocity(self, row, radar_pos):
        """推定結果(x, y, vx, vy)から視線速度を計算する"""
        # ターゲット位置 (z=0と仮定)
        tx, ty = row['x'], row['y']
        
        # Target -> Radar へのベクトル (接近方向を正とする)
        dx = radar_pos[0] - tx
        dy = radar_pos[1] - ty
        dz = radar_pos[2] - 0 
        
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 速度ベクトル
        vx = row['vx']
        vy = row['vy']
        
        # 視線速度 = 速度ベクトルの、レーダー方向への射影
        if dist > 1e-6:
            v_r = (vx * dx + vy * dy) / dist
        else:
            v_r = 0.0
        return v_r

    def calculate_metrics(self, target_id: int = 0):
        true_df, est_df, meas_df = self.load_data()
        if true_df is None:
            return

        # --- ターゲットIDでフィルタリング ---
        true_target = true_df[true_df['target_id'] == target_id].sort_values('time')
        est_target = est_df[est_df['target_id'] == target_id].sort_values('time')

        if true_target.empty:
            print(f"No true data found for target_id {target_id}")
            return

        # ---------------------------------------------------------
        # 1. Tracking Result RMSE (真値 vs 推定値)
        # ---------------------------------------------------------
        # 時間でマージ
        merged_est = pd.merge(true_target, est_target, on='time', suffixes=('_true', '_est'))
        
        rmse_tracking = None
        if not merged_est.empty:
            # 推定軌道の視線速度を計算
            est_radial_vels = merged_est.apply(
                lambda row: self.calculate_estimated_radial_velocity(
                    # マージ後のカラム名に合わせてデータを渡す
                    pd.Series({'x': row['x_est'], 'y': row['y_est'], 'vx': row['vx'], 'vy': row['vy']}),
                    self.radar_pos
                ), axis=1
            )
            
            # 真値の視線速度 (velocityカラム) との差分
            # 注: true_trajectory.csvのvelocityは視線速度である前提
            errors_tracking = est_radial_vels - merged_est['velocity']
            mse_tracking = (errors_tracking ** 2).mean()
            rmse_tracking = np.sqrt(mse_tracking)
        else:
            print("No overlapping time steps between True and Estimated trajectories.")

        # ---------------------------------------------------------
        # 2. Sensing Result RMSE (真値 vs 観測値)
        # ---------------------------------------------------------
        # 観測データにはIDがない場合が多いため、位置が最も近いデータを紐付ける
        sensing_diffs = []

        for _, true_row in true_target.iterrows():
            t = true_row['time']
            true_pos = np.array([true_row['x'], true_row['y']])
            true_vel = true_row['velocity'] # 真の視線速度

            # 同じ時刻の観測データを抽出
            meas_at_t = meas_df[meas_df['time'] == t]
            
            if meas_at_t.empty:
                continue

            # 位置 (x, y) が最も近い観測点を探す (Nearest Neighbor Association)
            dists = np.sqrt((meas_at_t['x'] - true_pos[0])**2 + (meas_at_t['y'] - true_pos[1])**2)
            
            # 閾値を設けることも可能だが、ここでは単純に最小値を選択
            closest_idx = dists.idxmin()
            closest_meas = meas_at_t.loc[closest_idx]

            # 観測された視線速度
            meas_vel = closest_meas['velocity']
            
            sensing_diffs.append(meas_vel - true_vel)

        rmse_sensing = None
        if len(sensing_diffs) > 0:
            sensing_diffs = np.array(sensing_diffs)
            mse_sensing = (sensing_diffs ** 2).mean()
            rmse_sensing = np.sqrt(mse_sensing)
        else:
            print("No corresponding measurements found for the true trajectory.")

        # ---------------------------------------------------------
        # 結果表示
        # ---------------------------------------------------------
        print(f"--- Radial Velocity RMSE Results (Target {target_id}) ---")
        if rmse_sensing is not None:
            print(f"Sensing RMSE  : {rmse_sensing:.4f} m/s")
        else:
            print("Sensing RMSE  : N/A")

        if rmse_tracking is not None:
            print(f"Tracking RMSE : {rmse_tracking:.4f} m/s")
        else:
            print("Tracking RMSE : N/A")
            
        # 改善率などの表示（オプション）
        if rmse_sensing is not None and rmse_tracking is not None and rmse_sensing > 0:
            improvement = (1 - rmse_tracking / rmse_sensing) * 100
            print(f"Improvement   : {improvement:.2f}%")

if __name__ == "__main__":
    # 使用例: スクリプトとして直接実行された場合
    calculator = VelocityRMSECalculator(csv_dir="csv_result")
    calculator.calculate_metrics(target_id=0)