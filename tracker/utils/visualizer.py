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
        true_traj_path = self.csv_dir / "true_trajectory.csv"
        
        # まず普通に読み込む
        true_traj = pd.read_csv(true_traj_path)
        '''
        # データがずれている場合（y列が文字列として認識されている場合）の補正処理
        # 本来数値であるはずの 'y' が object 型（文字列等）になっているかで判定
        if true_traj['y'].dtype == 'object':
            print("Warning: Detected malformed CSV (columns shift). Reloading with correction.")
            # ヘッダー行(0行目)を無視し、7列分の名前を強制的に割り当てて読み直す
            true_traj = pd.read_csv(true_traj_path, 
                                  names=['time', 'target_id','x', 'y','range', 'angle', 'velocity'], 
                                  header=0)
        '''
        est_traj = pd.read_csv(self.csv_dir / "estimated_trajectory.csv")
        measurements = pd.read_csv(self.csv_dir / "measurements.csv")
        
        validated_path = self.csv_dir / "validated_measurements.csv"
        validated = None
        if validated_path.exists():
            validated = pd.read_csv(validated_path)
        
        return true_traj, est_traj, measurements, validated
    
    def plot_trajectory_2d(self, save_path: str = None, show_measurements: bool = True):
        """2次元軌道をプロット (複数ターゲット対応)"""
        true_traj, est_traj, measurements, _ = self.load_data()
        
        plt.figure(figsize=(12, 10))
        
        # 存在するターゲットIDのリストを取得してループする
        unique_target_ids = true_traj['target_id'].unique()
        # 色の割り当て用にenumerateを使う
        for idx, target_id in enumerate(unique_target_ids):
            # 色を循環させる (ターゲットIDが大きくてもエラーにならないように)
            color = self.colors[idx % len(self.colors)]
            
            # 真の軌道 (target_id を使用)
            true_t = true_traj[true_traj['target_id'] == target_id]
            
            # データが存在しない場合のガード（念のため）
            if true_t.empty:
                continue

            # 線なし (フォーマット文字列 'o' を削除して warning を解消)
            plt.plot(true_t['x'], true_t['y'], linewidth=2, color=color,
                    label=f'True Target {target_id}', marker='o', markersize=10)
            
            
            
            # 推定軌道
            est_t = est_traj[est_traj['target_id'] == target_id]
            if not est_t.empty:
                plt.plot(est_t['x'], est_t['y'], '--', linewidth=2, color='red',
                        label=f'Estimated Target {target_id}', marker='s', markersize=10)
            
            # 開始点と終了点 (iloc[0]のエラー箇所)
            plt.plot(true_t['x'].iloc[0], true_t['y'].iloc[0], marker='o', color=color,
                    markersize=20, label=f'Start T{target_id}', alpha=0.5, linestyle='None')
            plt.plot(true_t['x'].iloc[-1], true_t['y'].iloc[-1], marker='X', color=color,
                    markersize=20, label=f'End T{target_id}', alpha=0.5, linestyle='None')
            
            # measurements はループ外で一度だけプロットする（凡例重複回避のため）
        if show_measurements and measurements is not None and not measurements.empty:
            # measurements を他のプロットより前面に出し、見やすく大きめの緑丸にする
            plt.scatter(measurements['x'], measurements['y'], s=140, c='green', marker='o',
                        edgecolors='k', linewidths=0.5, alpha=0.95, label='Measurements', zorder=10)
        plt.tick_params(labelsize=22)
        plt.xlabel('X Position', fontsize=30)
        plt.ylabel('Y Position', fontsize=30)
        
        # 凡例の重複を避ける等の処理が必要ならここで行う
        plt.legend(loc='best', fontsize=20, bbox_to_anchor=(1.05, 1))
        plt.grid(True, alpha=0.3)
        
        # 範囲指定（データに合わせて調整が必要かもしれません）
        plt.xlim(-40, -30)
        plt.ylim(10, 20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D trajectory plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
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

    def plot_vx_comparison(self, save_path: str = None):
        """vxを時間軸で比較するプロット"""
        true_traj, est_traj, measurements, _ = self.load_data()

        plt.figure(figsize=(12, 8))

        # 真の軌道のvx
        plt.plot(true_traj['time'], true_traj['velocity'], '-', linewidth=2, label='True vx', color='blue')

        # 推定軌道のvx
        plt.plot(est_traj['time'], est_traj['vx'], '--', linewidth=2, label='Estimated vx', color='red')

        # 観測データのvx
        plt.plot(measurements['time'], measurements['velocity'], ':', linewidth=2, label='Measurements vx', color='green')

        plt.xlabel('Time', fontsize=20)
        plt.ylabel('vx', fontsize=20)
        plt.legend(loc='best', fontsize=15)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"vx comparison plot saved to {save_path}")

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
    
    print("Plotting vx comparison...")
    visualizer.plot_vx_comparison()

if __name__ == "__main__":
    # 可視化を実行する場合は、main.py とは別に実行してください
    main_visualize()