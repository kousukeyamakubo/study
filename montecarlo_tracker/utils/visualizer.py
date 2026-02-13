import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 色分けのため
from matplotlib.patches import Ellipse
from pathlib import Path
from typing import Tuple
import math


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

        # --- 観測データの分解: 単一の速度成分 (velocity) と角度 (angle) がある場合、
        #     推定値と同様に x,y 方向の速度成分 `vx`/`vy` を追加する。
        def _decompose(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            df = df.copy()
            # velocity と angle が存在し、vx/vy が無ければ計算する
            if 'vx' not in df.columns and 'vy' not in df.columns:
                if 'velocity' in df.columns and 'angle' in df.columns:
                    # 角度はラジアンである想定
                    print(df['velocity'][0])
                    print('--------')
                    print(df['angle'][0])
                    df['vx'] = df['velocity'] * np.cos(np.radians(df['angle']))
                    df['vy'] = df['velocity'] * np.sin(np.radians(df['angle']))
            return df

        measurements = _decompose(measurements)
        if validated is not None:
            validated = _decompose(validated)

        return true_traj, est_traj, measurements, validated
    
    def plot_trajectory_2d(self, save_path: str = None, show_measurements: bool = True):
        """2次元軌道をプロット (複数ターゲット対応)"""
        true_traj, est_traj, measurements, _ = self.load_data()
        
        plt.figure(figsize=(10, 8))
        
        # IDごとに使用するカラーリストを定義（真値と推定値で色被りなし、違いがはっきり）
        # 真値用：明るく鮮やかな色
        true_colors = [
            '#1f77b4',  # 青
            '#1f77b4',  # 青
            '#17becf',  # シアン
            '#2ca02c',  # 緑
            '#ff7f0e',  # オレンジ
            '#9467bd',  # 紫
            '#bcbd22',  # 黄緑
            '#e377c2',  # ピンク
            '#7f7f7f',  # グレー
            '#8c564b',  # 茶色
            '#00bfff',  # ディープスカイブルー
        ]
        
        # 推定値用：暗めまたは異なる系統の色（真値と被らない）
        est_colors = [
            '#d62728',  # 赤
            '#d62728',  # 赤
            '#8b0000',  # ダークレッド
            '#ff1493',  # ディープピンク
            '#4b0082',  # インディゴ
            '#8b4513',  # サドルブラウン
            '#006400',  # ダークグリーン
            '#00008b',  # ダークブルー
            '#ff4500',  # オレンジレッド
            '#800080',  # パープル
            '#2f4f4f',  # ダークスレートグレー
        ]
        
        # 真値と推定値の両方のトラックIDを取得
        true_target_ids = set(true_traj['target_id'].unique())
        est_target_ids = set(est_traj['target_id'].unique())
        all_target_ids = sorted(true_target_ids | est_target_ids)
        
        # トラックIDごとに真値用と推定値用の色を割り当てる
        true_color_map = {tid: true_colors[i % len(true_colors)] for i, tid in enumerate(all_target_ids)}
        est_color_map = {tid: est_colors[i % len(est_colors)] for i, tid in enumerate(all_target_ids)}
        
        # 真の軌道をプロット
        for target_id in true_target_ids:
            color = true_color_map[target_id]
            true_t = true_traj[true_traj['target_id'] == target_id]
            
            if true_t.empty:
                continue

            plt.plot(true_t['x'], true_t['y'], linewidth=3, color=color,
                    label=f'True Target {target_id}', marker='o', markersize=8)
            
            # 開始点と終了点
            plt.plot(true_t['x'].iloc[0], true_t['y'].iloc[0], marker='o', color=color,
                    markersize=15, alpha=0.7, linestyle='None')
            plt.plot(true_t['x'].iloc[-1], true_t['y'].iloc[-1], marker='X', color=color,
                    markersize=15, alpha=0.7, linestyle='None')
        
        # 推定軌道をプロット
        for target_id in est_target_ids:
            color = est_color_map[target_id]
            est_t = est_traj[est_traj['target_id'] == target_id]
            
            if not est_t.empty:
                # 破線で表示
                plt.plot(est_t['x'], est_t['y'], '--', linewidth=3, color=color,
                        label=f'Estimated Track {target_id}', marker='s', markersize=8)

        # measurements は一度だけプロット
        if show_measurements and measurements is not None and not measurements.empty:
            plt.scatter(measurements['x'], measurements['y'], s=140, c='green', marker='x',
                edgecolors='k', linewidths=3.0, alpha=0.95, label='Measurements', zorder=10)
            
            # 各測定点の近くにタイムステップを数字で表示
            #for _, row in measurements.iterrows():
            #    # x座標を少し左に、y座標を少し上にオフセット
            #    plt.text(row['x'] - 0.05, row['y'] + 0.05, f"{int(row['time'])}", 
            #        fontsize=10, ha='right', va='bottom', 
            #        color='darkgreen', fontweight='bold')

        plt.tick_params(labelsize=22)
        plt.xlabel('X[m]', fontsize=40)
        plt.ylabel('Y[m]', fontsize=40)
        plt.grid(True, alpha=0.3)
        
        plt.tick_params(axis='both', labelsize=30, pad=10)
        
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
            ax.tick_params(axis='both', labelsize=40)
            ax.plot(true_t['time'], error_norm, '-', linewidth=2, color=self.colors[i])
            ax.set_ylabel(f'Euclidean Error (Target {i})', fontsize=11)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title('Position Estimation Error', fontsize=40, fontweight='bold')
            if i == num_targets - 1:
                ax.set_xlabel('Time Step', fontsize=40)
        
        # --- ★追加: 目盛りの文字サイズ変更 ---
        # labelsize=30 で数値を大きくします
        # pad=10 で軸と数値の間に少し隙間を空けて見やすくします
        plt.tick_params(axis='both', labelsize=30, pad=10) 
        # -------------------------------------
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

    def plot_vx_comparison(self, save_path: str = None):
        """vxを時間軸で比較するプロット（ID=0とそれ以外を分けて表示）"""
        true_traj, est_traj, measurements, _ = self.load_data()

        # 真値と推定値の両方のトラックIDを取得
        true_target_ids = sorted(true_traj['target_id'].unique())
        est_target_ids = sorted(est_traj['target_id'].unique())
        all_target_ids = sorted(set(true_target_ids) | set(est_target_ids))
        
        # 2つのサブプロットを作成（上: ID=0, 下: ID≠0）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # レーダー座標
        radar_pos = np.array([250, -18, 50])
        
        # トラックIDごとに異なる色を割り当て（測定値用）
        colors = sns.color_palette("husl", len(all_target_ids))
        
        for idx, target_id in enumerate(all_target_ids):
            color = colors[idx]
            
            # プロット先のaxを選択（ID=0なら上のグラフ、それ以外なら下のグラフ）
            ax = ax1 if target_id == 0 else ax2
            
            # データのフィルタリング
            true_t = true_traj[true_traj['target_id'] == target_id]
            est_t = est_traj[est_traj['target_id'] == target_id]
            
            if 'target_id' in measurements.columns:
                meas_t = measurements[measurements['target_id'] == target_id]
            else:
                meas_t = measurements.copy()
            
            # 推定値の視線速度を計算
            est_radial_vel = []
            for _, row in est_t.iterrows():
                tx, ty = row['x'], row['y']
                dx = radar_pos[0] - tx
                dy = radar_pos[1] - ty
                dz = radar_pos[2] - 0 
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                vx = row['vx']
                vy = row['vy']
                
                if dist > 1e-6:
                    v_r = (vx * dx + vy * dy) / dist
                else:
                    v_r = 0.0
                est_radial_vel.append(v_r)
            
            est_t = est_t.copy()
            est_t['radial_velocity'] = est_radial_vel
            
            # プロット（真値は青色、推定値は赤色に統一）
            if not true_t.empty:
                ax.plot(true_t['time'], true_t['velocity'], '-', linewidth=3, 
                       label=f'True ID={target_id}', color='blue')
            
            if not est_t.empty:
                ax.plot(est_t['time'], est_t['radial_velocity'], '--', linewidth=3, 
                       label=f'Est. ID={target_id}', color='red', alpha=0.8)
            
            if not meas_t.empty and 'vx' in meas_t.columns:
                ax.plot(meas_t['time'], meas_t['vx'], ':', linewidth=2.5, 
                       label=f'Meas. ID={target_id}', color=color, alpha=0.6)
    
        # 上のグラフ（ID=0）の設定
        ax1.set_ylabel('Velocity [m/s]', fontsize=35)
        ax1.legend(loc='best', fontsize=20, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=28, pad=10)
        ax1.set_title('Velocity Comparison (ID=0)', fontsize=38, fontweight='bold')
        """
        # 下のグラフ（ID≠0）の設定
        ax2.set_xlabel('Time [s]', fontsize=35)
        ax2.set_ylabel('Velocity [m/s]', fontsize=35)
        ax2.legend(loc='best', fontsize=20, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=28, pad=10)
        ax2.set_title('Velocity Comparison (ID≠0)', fontsize=38, fontweight='bold')
        """
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"vx comparison plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

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