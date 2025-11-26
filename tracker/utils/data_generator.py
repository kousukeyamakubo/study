import numpy as np
import pandas as pd # CSV読み込み用 (将来)
from typing import List, Tuple, Optional,Dict
from abc import ABC, abstractmethod

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
        print(f"Loading data from {self.true_csv_path} and {self.meas_csv_path}...")
        # ダミーリターン
        return [], []