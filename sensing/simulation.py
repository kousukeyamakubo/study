import numpy as np
import pandas as pd
import os
import time
from config import Radar_setting, Sim_Setting
from data_loader import get_snapshot_data
from doa import MUSIC_method_improved
from range_doppler import FFT
from detection import CFAR
from visualization import RD_MAP
from utils import derectory_deletor, print_true_values


import time

def main_simulation(i):
    # --- 保存先ディレクトリの設定 ---
    base_dir = "./montecarlo"
    rd_map_dir = os.path.join(base_dir, "rd_maps")
    
    os.makedirs(rd_map_dir, exist_ok=True)
    
    # 実行用設定
    T_symbol,f_carrier,Tc,N_ant,BW,BW_sub,N_sample,rx_sample, mu, l_speed, lam, env = Radar_setting()
    N_chirp = 89
    print("DEBUG: ", T_symbol * N_chirp)
    N_trans = N_chirp
    tx = env.tx()
    p_bs, num_cy, num_ve, num_rp, start_cy_idx, start_ve_idx, end_cy_idx, end_ve_idx, cy_v_value, ve_v_value, cy_interval, ve_interval = Sim_Setting()
    cy_idx, ve_idx = start_cy_idx, start_ve_idx
    measurement_history = []
    frame_count = 0
    
    # 画像フォルダの中身を削除 (必要に応じてコメントアウト)
    # 毎回削除すると、前の回の画像が消えます。
    derectory_deletor(rd_map_dir)
    
    T_sensing = N_chirp * T_symbol + 813 * Tc
    
    real_cy_coordinates_all, real_cy_ranges_all, real_cy_angles_all, real_cy_velocities_all, real_ve_coordinates_all, real_ve_ranges_all, real_ve_angles_all, real_ve_velocities_all = [], [], [], [], [], [], [], []

    while cy_idx <= end_cy_idx and ve_idx <= end_ve_idx:
        x_pos = list(np.arange(-450, -149)/10)
        
        Y, phys_quantities, real_cy_coordinates, real_vel_coordinates, real_rp_coordinates = get_snapshot_data(cy_idx,ve_idx,x_pos,p_bs, tx, cy_v_value, ve_v_value, num_cy, num_ve, num_rp, T_symbol, N_trans, Tc, env)
        t0 = time.perf_counter()
        print('使っているインデックス:', cy_idx, ve_idx)
        print_true_values(num_cy, num_ve, num_rp, phys_quantities, real_cy_coordinates, real_vel_coordinates, real_rp_coordinates,
                      real_cy_coordinates_all, real_cy_ranges_all, real_cy_angles_all, real_cy_velocities_all,
                      real_ve_coordinates_all, real_ve_ranges_all, real_ve_angles_all, real_ve_velocities_all) 
        
        est_ang, resp = MUSIC_method_improved(Y, N_ant, num_cy + num_ve + num_rp, frame_count)

        t1 = time.perf_counter()
        print(f"信号処理1: MUSIC  {(t1-t0)*1000:.3f} ms")
        rd_maps = FFT(N_trans, N_ant, rx_sample, Y, tx, N_sample, est_ang, T_symbol, lam, Tc, env)
        t2 = time.perf_counter()
        print(f"信号処理2: FFT  {(t2-t1)*1000:.3f} ms")
        detected_objects = CFAR(rd_maps, est_ang, N_trans, T_symbol, lam, Tc, env)
        t3 = time.perf_counter()
        print(f"信号処理3: CFAR  {(t3-t2)*1000:.3f} ms")
        # 画像保存 (シミュレーション番号 i をファイル名に含めると区別可能です)
        # ただし derectory_deletor が有効だと消えてしまうので注意
        RD_MAP(rd_maps, frame_count, T_symbol, Tc, save_dir=rd_map_dir)
        
        if len(detected_objects) > 0:
            for obj in detected_objects:
                measurement_history.append({
                    "Time": frame_count,
                    "Range": obj[0],
                    "Angle": obj[1],
                    "X": obj[2],
                    "Y": obj[3],
                    "Velocity": obj[4],
                    "Power": obj[5]
                })
        else:
            measurement_history.append({
                "Time": frame_count,
                "Range": np.nan,
                "Angle": np.nan,
                "X": np.nan,
                "Y": np.nan,
                "Velocity": np.nan,
                "Power": np.nan
            })

        angle_list = []
        power_list = []

        for obj in detected_objects:
            if obj[1] in angle_list:
                continue
            else:
                angle_list.append(obj[1])
            if obj[5] in power_list:
                continue
            else:
                power_list.append(obj[5])
        # --- 角度分解に成功したかを表示 ---
        if len(est_ang) > 1:
            print(len(est_ang))
            print(len(angle_list))
            if len(est_ang) == len(angle_list):
                print("成功，強度差は ->", np.abs(power_list[0]-power_list[1]))
                for obj in detected_objects:
                    print(obj)
            else:
                print("処理上で何らかの問題が発生")
        else:
            print("失敗")
        
        cy_idx += cy_interval
        ve_idx += ve_interval
        frame_count += 1
        
    # --- CSV保存 (番号付き) ---
    df_measurements = pd.DataFrame(measurement_history)
    csv_filename = os.path.join(base_dir, f"measurements_{i}.csv") # ここを修正
    df_measurements.to_csv(csv_filename, index=False)

    print("\nData generation complete.")
    print(f"Saved '{csv_filename}'.")

    real_positions = []
    csv_filename_true = os.path.join(base_dir, f"true_information_{i}.csv") # ここを修正
    
    # 【重要修正】 変数名を i から k に変更 (引数 i との衝突回避)
    for k in range(max(len(real_cy_coordinates_all), len(real_ve_coordinates_all))):
        if num_cy > 0:
            real_positions.append({
                "Time": k,
                "target_id":0,
                "X": real_cy_coordinates_all[k][0][0],
                "Y": real_cy_coordinates_all[k][0][1],
                "Range": real_cy_ranges_all[k],
                "Angle": real_cy_angles_all[k],
                "Velocity": real_cy_velocities_all[k],
            })

        if num_ve > 0:
            real_positions.append({
                "Time": k,
                "target_id":1,
                "X": real_ve_coordinates_all[k][0][0],
                "Y": real_ve_coordinates_all[k][0][1],
                "Range": real_ve_ranges_all[k],
                "Angle": real_ve_angles_all[k],
                "Velocity": real_ve_velocities_all[k]
            })
            
    df_real_positions = pd.DataFrame(real_positions)
    df_real_positions.to_csv(csv_filename_true, index=False)
    print(f"Saved true positions to '{csv_filename_true}'.")
    return

def montecarlo_simulation():
    np.random.seed(42)  # 再現性のためのシード設定
    num_simulations = 1
    for i in range(num_simulations):
        print(f"\n--- Simulation Run {i+1}/{num_simulations} ---")
        main_simulation(i)