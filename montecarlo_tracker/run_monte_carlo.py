import os
import shutil
import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm  # 進捗バー表示用 (pip install tqdm)

def parse_position_rmse(output_str):
    """
    utils/rmse_calculator.py の出力からOverall RMSEを抽出する
    """
    # 正規表現で数値を抽出
    meas_match = re.search(r"measurements vs true:\s+([0-9.]+)", output_str)
    est_match = re.search(r"estimated vs true:\s+([0-9.]+)", output_str)
    
    meas_rmse = float(meas_match.group(1)) if meas_match else None
    est_rmse = float(est_match.group(1)) if est_match else None
    return meas_rmse, est_rmse

def parse_velocity_rmse(output_str):
    """
    utils/velocity_rmse_calculator.py の出力からRMSEを抽出する
    """
    sensing_match = re.search(r"Sensing RMSE\s+:\s+([0-9.]+)", output_str)
    tracking_match = re.search(r"Tracking RMSE\s+:\s+([0-9.]+)", output_str)
    
    sensing_rmse = float(sensing_match.group(1)) if sensing_match else None
    tracking_rmse = float(tracking_match.group(1)) if tracking_match else None
    return sensing_rmse, tracking_rmse

def main():
    # --- 設定 ---
    num_runs = 10  # 実行回数
    measurements_dir = "2GHz_montecarlo_measurements"  # CSVがあるディレクトリ
    # ----------------
    
    # 結果格納用リスト
    results = {
        "pos_rmse_meas": [],
        "pos_rmse_est": [],
        "vel_rmse_meas": [],
        "vel_rmse_est": []
    }

    print(f"Starting Monte Carlo Simulation with {num_runs} runs...")

    # tqdmを使って進捗バーを表示
    for i in tqdm(range(num_runs)):
        # 1. ファイルの準備
        src_file = os.path.join(measurements_dir, f"measurements_{i}.csv")
        dst_file = "measurements.csv" # main.pyが読み込むファイル名
        
        if not os.path.exists(src_file):
            print(f"Warning: File {src_file} not found. Skipping...")
            continue
            
        shutil.copy(src_file, dst_file)
        
        # 2. main.py の実行 (トラッキング処理)
        # 出力が多い場合は capture_output=True で抑制できます
        subprocess.run(["python", "main.py"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Position RMSE の計算
        ret_pos = subprocess.run(["python", "utils/rmse_calculator.py"], capture_output=True, text=True)
        pos_meas, pos_est = parse_position_rmse(ret_pos.stdout)
        
        # 4. Velocity RMSE の計算
        ret_vel = subprocess.run(["python", "utils/velocity_rmse_calculator.py"], capture_output=True, text=True)
        vel_meas, vel_est = parse_velocity_rmse(ret_vel.stdout)
        
        # 結果の保存
        if pos_meas is not None: results["pos_rmse_meas"].append(pos_meas)
        if pos_est is not None: results["pos_rmse_est"].append(pos_est)
        if vel_meas is not None: results["vel_rmse_meas"].append(vel_meas)
        if vel_est is not None: results["vel_rmse_est"].append(vel_est)

    # --- 最終結果の集計と表示 ---
    print("\n" + "="*40)
    print("   MONTE CARLO SIMULATION RESULTS   ")
    print("="*40)
    
    def print_stats(name, data_meas, data_est, unit=""):
        if not data_meas or not data_est:
            print(f"No data for {name}")
            return

        mean_meas = np.mean(data_meas)
        std_meas = np.std(data_meas)
        mean_est = np.mean(data_est)
        std_est = np.std(data_est)
        
        print(f"\n--- {name} RMSE ({len(data_meas)} samples) ---")
        print(f"Measurements (Sensing): {mean_meas:.4f} ± {std_meas:.4f} {unit}")
        print(f"Estimated (Tracking)  : {mean_est:.4f} ± {std_est:.4f} {unit}")
        
        improvement = mean_meas - mean_est
        print(f"Improvement (Mean)    : {improvement:.4f} {unit}")

    print_stats("POSITION", results["pos_rmse_meas"], results["pos_rmse_est"], unit="m")
    print_stats("VELOCITY", results["vel_rmse_meas"], results["vel_rmse_est"], unit="m/s")
    
    # 必要であれば結果をCSVに保存
    df_res = pd.DataFrame(results)
    df_res.to_csv("montecarlo_summary.csv", index_label="run_id")
    print("\nDetailed results saved to 'montecarlo_summary.csv'")

if __name__ == "__main__":
    main()