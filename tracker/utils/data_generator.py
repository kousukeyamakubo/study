import numpy as np
import pandas as pd # CSV読み込み用 (将来)
from typing import List, Tuple, Optional,Dict
from abc import ABC, abstractmethod
from pathlib import Path

# 以前の MeasurementGenerator は、"SimulationScenario" 内の部品として使います
class MeasurementGenerator:
    """観測データにノイズを乗せるクラス（既存のまま）"""
    def __init__(self, H: np.ndarray, R: np.ndarray, detection_prob: float = 0.9):
        self.H = H
        self.R = R
        self.detection_prob = detection_prob
    
    def generate_from_state(self, true_state: np.ndarray) -> Optional[np.ndarray]:
        # 単一ターゲットの1ステップ分の観測生成
        if np.random.rand() < self.detection_prob:
            true_pos = self.H @ true_state
            noise = np.random.multivariate_normal(np.zeros(len(true_pos)), self.R)
            return true_pos + noise
        return None
    
    def _generate_clutter(self, clutter_params: Dict) -> List[np.ndarray]:
        """クラッタを生成（元のコードから復元）"""
        clutter_list = []
        ctype = clutter_params.get('type', 'uniform')
        
        if ctype == 'uniform':
            # 従来通りの均一分布
            num_clutter = clutter_params.get('num_clutter', 5)
            x_range, y_range = clutter_params.get('range', [[-50, 50], [-50, 50]])
            
            if isinstance(num_clutter, list):
                num_clutter = int(np.mean(num_clutter))

            for _ in range(num_clutter):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                clutter_list.append(np.array([x, y]))
        
        elif ctype == 'around_target':
            true_positions = clutter_params.get('true_positions', [])
            
            num_clutter_param = clutter_params.get('num_clutter', 2)
            density_param = clutter_params.get('clutter_spatial_density', 10.0)
            
            for i, pos in enumerate(true_positions):
                # --- ターゲットごとのパラメータ決定 ---
                if isinstance(num_clutter_param, list):
                    idx = min(i, len(num_clutter_param) - 1)
                    n_clutter = num_clutter_param[idx]
                else:
                    n_clutter = num_clutter_param
                
                if isinstance(density_param, list):
                    idx = min(i, len(density_param) - 1)
                    density = density_param[idx]
                else:
                    density = density_param
                # ------------------------------------
                
                for _ in range(n_clutter):
                    offset = np.random.randn(len(pos)) * density
                    clutter_list.append(pos + offset)
                    
        return clutter_list


class ScenarioProvider(ABC):
    """
    データの供給源となる基底クラス
    真の軌道(Ground Truth)と、観測(Measurements)の両方を提供する責務を持つ
    """
    @abstractmethod
    def get_data(self) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Returns:
            true_trajectories: List[np.ndarray] (形状: [Target数, Time, StateDim])
            measurements_list: List[List[np.ndarray]] (形状: [Time, Measurement数, ObsDim])
        """
        pass


class OvertakingScenario(ScenarioProvider):
    """
    【旧 main.py のロジック】
    数式ベースで「追い越し」軌道を生成し、MeasurementGeneratorでノイズを付加する
    """
    def __init__(self, num_steps: int, dt: float, measurement_gen: MeasurementGenerator, clutter_params: dict):
        self.num_steps = num_steps
        self.dt = dt
        self.measurement_gen = measurement_gen
        self.clutter_params = clutter_params

    def _generate_ground_truth(self) -> List[np.ndarray]:
        """内部で真の軌道を作成"""
        # --- ターゲット1: 自転車 ---
        traj_bike = np.zeros((self.num_steps, 4))
        speed_bike = 2.0
        for i in range(self.num_steps):
            traj_bike[i] = [0.0, speed_bike * i * self.dt, 0.0, speed_bike]
            
        # --- ターゲット2: 自動車 ---
        traj_car = np.zeros((self.num_steps, 4))
        speed_car = 10.0
        start_y_car = -20.0
        fixed_lane_x = 1.5
        for i in range(self.num_steps):
            t = i * self.dt
            y = start_y_car + speed_car * t
            traj_car[i] = [fixed_lane_x, y, 0.0, speed_car]
            
        return [traj_bike, traj_car]

    def get_data(self) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        # 1. 真の軌道を生成
        true_trajectories = self._generate_ground_truth()
        
        # 2. 観測を生成 (MeasurementGeneratorを利用)
        measurements_list = []
        
        # クラッタ生成用に真の位置情報を整形して渡す処理などが必要ですが
        # 基本的には既存の simulator.py のループ内でやっていた処理をここで一括で行います
        # あるいは、Simulator側で「都度生成」したい場合は設計を少し変える必要がありますが、
        # 「事前にデータを確定させる」方針であればここで全ステップ分作ってしまいます。
        
        for t in range(self.num_steps):
            # 時刻 t における全ターゲットの真の状態
            current_true_states = [traj[t] for traj in true_trajectories]
            
            step_measurements = []
            true_positions_at_t = []

            # ターゲットごとの観測生成
            for state in current_true_states:
                z = self.measurement_gen.generate_from_state(state)
                true_positions_at_t.append(self.measurement_gen.H @ state) # クラッタ用
                if z is not None:
                    step_measurements.append(z)
            
            # クラッタ生成
            # (注: MeasurementGenerator内に_generate_clutterがある前提)
            self.clutter_params['true_positions'] = true_positions_at_t
            clutter = self.measurement_gen._generate_clutter(self.clutter_params)
            step_measurements.extend(clutter)
            
            # シャッフル
            np.random.shuffle(step_measurements)
            measurements_list.append(step_measurements)
            
        return true_trajectories, measurements_list


class CSVScenario(ScenarioProvider):
    """
    【将来用】
    CSVファイルから真値と観測値を読み込む
    """
    def __init__(self, true_csv_path: str, meas_csv_path: str):
        self.true_csv_path = true_csv_path
        self.meas_csv_path = meas_csv_path

    def get_data(self) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        # ここに pd.read_csv 等の実装が入る
        true_trajectories = []
        measurements_list = []

        # --- 1. 観測データ (measurements.csv) の読み込み ---
        meas_path = Path(self.meas_csv_path)
        if meas_path.exists():
            print(f"Loading measurements from {meas_path}...")
            try:
                df_meas = pd.read_csv(meas_path)
                
                # カラム名の揺らぎ吸収 (X, Y または x, y)
                col_x = 'X' if 'X' in df_meas.columns else 'x'
                col_y = 'Y' if 'Y' in df_meas.columns else 'y'
                col_time = 'Time' if 'Time' in df_meas.columns else 'time'

                if col_time not in df_meas.columns:
                    raise ValueError(f"Column '{col_time}' not found in {self.meas_csv_path}")

                # シミュレーションのステップ数を決定 (Timeの最大値 + 1)
                max_time = int(df_meas[col_time].max())
                measurements_list = [[] for _ in range(max_time + 1)]

                # タイムステップごとにグループ化
                for t, group in df_meas.groupby(col_time):
                    step_measurements = []
                    for _, row in group.iterrows():
                        # [x, y] のnumpy配列を作成
                        z = np.array([row[col_x], row[col_y]], dtype=float)
                        
                        # --- ★追加修正: nanが含まれているデータはリストに追加しない ---
                        if not np.isnan(z).any():
                            step_measurements.append(z)
                        # -------------------------------------------------------
                        
                        #修正前
                        #step_measurements.append(z)
                    
                    # Timeが整数の場合、そのままインデックスとして使用
                    # (measurements.csvのTimeが1始まりなら、measurements_list[1]に入る)
                    t_idx = int(t)
                    if 0 <= t_idx < len(measurements_list):
                        measurements_list[t_idx] = step_measurements
            
            except Exception as e:
                print(f"Error loading measurements CSV: {e}")
                # エラー時は空リストなどを返すか、raiseするか
                measurements_list = []
        else:
            print(f"Measurements file not found: {self.meas_csv_path}")

        # --- 2. 真値データ (true_trajectory.csv) の読み込み ---
        # ※ファイルがない場合は空リストを返す（予測のみのテスト用）
        true_path = Path(self.true_csv_path)
        if true_path.exists():
            print(f"Loading true trajectories from {true_path}...")
            try:
                df_true = pd.read_csv(true_path)
                
                # 想定フォーマット: time, target_id, x, y, vx, vy
                # カラム名の揺らぎ吸収
                t_col = 'time' if 'time' in df_true.columns else 'Time'
                id_col = 'target_id' if 'target_id' in df_true.columns else 'id'
                x_col = 'x' if 'x' in df_true.columns else 'X'
                y_col = 'y' if 'y' in df_true.columns else 'Y'
                vx_col = 'vx' if 'vx' in df_true.columns else 'Vx'
                vy_col = 'vy' if 'vy' in df_true.columns else 'Vy'

                # ターゲットIDごとに分割
                for target_id, group in df_true.groupby(id_col):
                    # 時間順にソート
                    group = group.sort_values(t_col)
                    
                    # 状態ベクトル [x, y, vx, vy] を作成
                    # vx, vyがない場合は0で埋めるなどの処理が必要だが、ここではあると仮定
                    if vx_col in group.columns and vy_col in group.columns:
                        traj = group[[x_col, y_col, vx_col, vy_col]].to_numpy()
                    else:
                        # 位置のみの場合 [x, y, 0, 0]
                        pos = group[[x_col, y_col]].to_numpy()
                        traj = np.hstack([pos, np.zeros_like(pos)])
                    
                    true_trajectories.append(traj)

            except Exception as e:
                print(f"Error loading true trajectory CSV: {e}")
        else:
            print(f"True trajectory file not found (or not specified): {self.true_csv_path}")
            # 真値がない場合でも動作するように空リストを返す

        return true_trajectories, measurements_list