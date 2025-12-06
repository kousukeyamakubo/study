import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 色分けのため
from matplotlib.patches import Ellipse
from pathlib import Path
from typing import Tuple


class ResultVisualizer:
    """CSVファイルから結果を読み込んで可視化するクラス (JPDA対応)"""
    
    def __init__(self, csv_dir: str = "csv_result"):
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"Directory {csv_dir} does not exist")
            
        self.colors = sns.color_palette("bright", 10) # ターゲットごとの色
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """CSVファイルからデータを読み込む"""
        true_traj = pd.read_csv(self.csv_dir / "true_trajectory.csv")
        est_traj = pd.read_csv(self.csv_dir / "estimated_trajectory.csv")
        measurements = pd.read_csv(self.csv_dir / "measurements.csv")
        
        validated_path = self.csv_dir / "validated_measurements.csv"
        validated = None
        if validated_path.exists():
            validated = pd.read_csv(validated_path)
        
        return true_traj, est_traj, measurements, validated
    
    def plot_trajectory_2d(self, save_path: str = None):
        """2次元軌道をプロット (複数ターゲット対応)"""
        true_traj, est_traj, measurements, _ = self.load_data()
        
        plt.figure(figsize=(12, 10))
        
        # 観測データ（全ターゲット共通）
        #plt.scatter(measurements['x'], measurements['y'], c='red', alpha=0.3, 
        #                   s=140, label='Measurements')
        
        num_targets = true_traj['target_id'].nunique()
        
        for i in range(num_targets):
            color = self.colors[i]
            
            # 真の軌道
            # 線あり
            true_t = true_traj[true_traj['target_id'] == i]
            #plt.plot(true_t['x'], true_t['y'], '-', linewidth=2, color=color,
            #        label=f'True Target {i}', marker='o', markersize=4)
            # 線なし
            plt.plot(true_t['x'], true_t['y'], 'o', linewidth=2, color=color,
                    label=f'True Target {i}', marker='o', markersize=10)
            
            #推定軌道
            est_t = est_traj[est_traj['target_id'] == i]
            plt.plot(est_t['x'], est_t['y'], '--', linewidth=2, color='red',
                    label=f'Estimated Target {i}', marker='s', markersize=10)
            
            # 開始点と終了点
            plt.plot(true_t['x'].iloc[0], true_t['y'].iloc[0], 'o', color=color,
                    markersize=20, label=f'Start T{i}', alpha=0.5)
            plt.plot(true_t['x'].iloc[-1], true_t['y'].iloc[-1], 'X', color=color,
                    markersize=20, label=f'End T{i}', alpha=0.5)
        
        plt.tick_params(labelsize=22)
        plt.xlabel('X Position', fontsize=30)
        plt.ylabel('Y Position', fontsize=30)
        #plt.title('JPDAF Tracking Result - 2D Trajectory', fontsize=20, fontweight='bold')
        plt.legend(loc='best', fontsize=20, bbox_to_anchor=(1.05, 1))
        plt.grid(True, alpha=0.3)
        #plt.axis('equal')
        #plt.tick_params(axis='both', labelsize=25)
        # ここに追加します
        plt.xlim(-40, -30)
        plt.ylim(14,16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D trajectory plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

    # (plot_trajectory_with_gates はJPDAFでは複雑になるため省略)
    
    def plot_position_error(self, save_path: str = None):
        """位置誤差の時間変化をプロット (複数ターゲット対応)"""
        true_traj, est_traj, _, _ = self.load_data()
        
        num_targets = true_traj['target_id'].nunique()
        
        fig, axes = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), squeeze=False)
        
        for i in range(num_targets):
            true_t = true_traj[true_traj['target_id'] == i].reset_index()
            est_t = est_traj[est_traj['target_id'] == i].reset_index()
            
            # 位置誤差を計算
            error_x = est_t['x'] - true_t['x']
            error_y = est_t['y'] - true_t['y']
            error_norm = np.sqrt(error_x**2 + error_y**2)
            
            ax = axes[i, 0]
            ax.tick_params(axis='both', labelsize=20)
            ax.plot(true_t['time'], error_norm, '-', linewidth=2, color=self.colors[i])
            ax.set_ylabel(f'Euclidean Error (Target {i})', fontsize=11)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title('Position Estimation Error', fontsize=14, fontweight='bold')
            if i == num_targets - 1:
                ax.set_xlabel('Time Step', fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

    # (plot_position_components, plot_velocity_components も同様に target_id で分割可能)

def main_visualize():
    """可視化メイン関数"""
    # 変更前: デフォルト (csv_result_jpda) を読み込んでいた
    # visualizer = ResultVisualizer(csv_dir="csv_result_jpda")
    
    # 変更後: 今回のシナリオ結果 (csv_result) を指定する
    visualizer = ResultVisualizer(csv_dir="csv_result")
    
    print("Plotting 2D trajectory...")
    visualizer.plot_trajectory_2d()
    
    #print("Plotting position error...")
    #visualizer.plot_position_error()

if __name__ == "__main__":
    # 可視化を実行する場合は、main.py とは別に実行してください
    main_visualize()