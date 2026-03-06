import numpy as np
from Cyclist_env_RDA_2nano import cyclist_env

def Radar_setting():
    ###### symbol time & carrier frequency ######
    T_symbol = 5.575 * 1e-6              # symbol duration, with CP time
    T_OFDM = 5.2125 * 1e-6
    f_carrier = 28 * 1e+9
    Tc = 2.545*1e-9                      # sampling time

    ###### tx/rx ######
    N_ant = 16                       # the number of antennas
    # BW = 1.966080e+9                      # chirp bandwidth
    BW = 400*1e+6                    # chirp bandwidth
    BW_sub = BW/N_ant
    N_sample = int(np.floor(T_symbol/Tc))           # the number of samples of single chirp
    rx_sample = 813 + N_sample

    ###### Radar setting ######

    mu = BW_sub/T_symbol * 0.98
    l_speed = 299792458

    lam = l_speed/f_carrier
    env = cyclist_env(f_carrier, N_ant, BW, BW_sub, N_sample, rx_sample, Tc, mu, l_speed)
    return T_symbol,f_carrier,Tc,N_ant,BW,BW_sub,N_sample,rx_sample, mu, l_speed, lam, env

def Sim_Setting():
    # シミュレーション設定
    p_bs = np.array([250, -18, 50])#基地局の位置

    # 各車両の数
    num_cy = 1
    num_ve = 1
    num_rp = 0

    
    """
    # シミュレーション範囲の設定
    start_cy_idx = 150
    start_ve_idx = 100
    end_cy_idx = 240
    end_ve_idx =250
    """
    
    #実験用
    start_cy_idx = 150
    start_ve_idx = 128

    end_cy_idx = start_cy_idx
    end_ve_idx = start_ve_idx
    
    
    cy_interval = 3
    ve_interval = 5

    cy_v_value = 6
    ve_v_value = 10
    
    return p_bs, num_cy, num_ve, num_rp, start_cy_idx, start_ve_idx, end_cy_idx, end_ve_idx, cy_v_value, ve_v_value, cy_interval, ve_interval