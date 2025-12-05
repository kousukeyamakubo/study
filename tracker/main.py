import numpy as np

# 各種モジュールのインポート
from config import Config
from factory import create_tracker, create_filter
from simulator import TrackingSimulator
from core.models import get_motion_model, get_measurement_model
from utils.data_saver import ResultSaver
from utils.data_generator import MeasurementGenerator, OvertakingScenario, CSVScenario

def main():
    # ---------- 1. 設定の読み込み -----------------------
    cfg = Config()
    np.random.seed(cfg.seed)
    print(f"--- Config Loaded ---")
    print(f"filter Type : {cfg.filter_type}")
    print(f"Tracker Type: {cfg.tracker_type}")
    print(f"Gating Type : {cfg.gating_type}")
    print(f"Motion Model: {cfg.motion_model}")
    # --------------------------------------------------

    # ------2. 行列の生成 (core/models.py に委譲) --------
    # Configのパラメータ(dt, stdなど)を元にF, Q, H, Rを作る
    # get_motion_modelはcore/models.pyに定義
    F, Q = get_motion_model(cfg)
    # get_measurement_modelはcore/models.pyに定義
    H, R = get_measurement_model(cfg)
    # --------------------------------------------------
    
    # -------- 3. カルマンフィルタの構築 -----------------
    # LinearKalmanFilterはcore/kalman_filter.pyに定義
    filter = create_filter(cfg, F, H, Q, R)
    # --------------------------------------------------
    
    # ------ 4. トラッカーの生成 (factory.py に委譲) -----
    # JPDAFかGNNかなどはConfig文字列で決まる
    # create_trackerはfactory.pyに定義
    tracker = create_tracker(cfg, filter)
    # --------------------------------------------------
    
    # ------ 5. データ生成 (シナリオ) -------------------
    if cfg.use_csv_mode:
        # CSVファイルからデータを読み込むモード
        # CSVScenarioはutils/data_generator.pyに定義
        scenario = CSVScenario("true_data.csv", "measurements.csv")
        scenario.get_data()
        print("Data loaded from CSV files.")
    else:
        # ユーザによる指定モード
        # MeasurementGenerator にも生成した行列H, Rを渡す
        # MeasurementGeneratorはutils/data_generator.pyに定義
        meas_gen = MeasurementGenerator(H, R, detection_prob=cfg.detection_prob)
        
        # OvertakingScenarioはutils/data_generator.pyに定義
        scenario = OvertakingScenario(
            num_steps=cfg.num_steps, 
            dt=cfg.dt, 
            measurement_gen=meas_gen, 
            clutter_params=cfg.clutter_params
        )

    # get_dataメソッドで真値と観測値を取得
    true_trajs, measurements_list = scenario.get_data()
    
    # 初期状態 (真値の初項)
    #initial_states = [traj[0] for traj in true_trajs]
    initial_states = []
    # --------------------------------------------------
    
    # ------ 6. シミュレータの構築と実行 -----------------
    # シミュレータは「tracker」の中身を知らずに回すだけ
    # TrackingSimulatorはsimulator.pyに定義
    simulator = TrackingSimulator(tracker)
    
    est_trajs, info_list = simulator.run(
        initial_states=initial_states, 
        num_steps=cfg.num_steps,
        measurements_list=measurements_list,
        true_trajectories=true_trajs,
        verbose=True,
        USE_CSV_MODE=cfg.use_csv_mode
    )
    # --------------------------------------------------

    # ------ 7. 結果の保存 -----------------------------
    # ResultSaverはutils/data_saver.pyに定義
    saver = ResultSaver(output_dir="csv_result")
    
    # バリデーション情報(info_list)の形式がアルゴリズムによって異なるため
    # JPDAFの場合は validated_indices として保存する処理を入れる
    validated_indices = info_list if cfg.tracker_type == "JPDA" else None

    # save_allメソッドはutils/data_saver.pyに定義
    saver.save_all(true_trajs, est_trajs, measurements_list, validated_indices)
    print("Done.")
    # --------------------------------------------------

if __name__ == "__main__":
    main()