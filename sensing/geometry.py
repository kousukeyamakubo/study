import numpy as np

def stevec(N_ant, angle):
    # 入力された角度に対するステアリングベクトルの計算
    resp = (np.arange(N_ant)-N_ant*0.5+0.5).reshape([-1,1])
    resp = np.exp(1j*resp*np.pi*np.sin(angle))
    return np.matrix(resp)

def idx_to_xy(range_idx, angle_deg, Tc, env):
    """
    range_idx: cy_idxs_pred の値 (例: 496)
    angle_deg: MUSIC法で推定した角度 [度] (例: est_ang[0])
    """
    p_bs = np.array([250, -18, 50])
    refangle = np.arctan(33/265)/2
    angle_offset = - 0.00005
    refangle += angle_offset
    offset = 620
    # (1) 距離の計算 [m]
    # サンプル番号 = インデックス + オフセット
    sample_idx = range_idx + offset
    # 距離 = 往復時間 * c / 2
    R = (sample_idx * Tc * env.c) / 2

    # 三次元距離を二次元距離に変換
    height_diff = p_bs[2] - 1
    if R > height_diff:
        R_horizontal = np.sqrt(R**2 - height_diff**2)
    else:
        R_horizontal = 0

    # (2) 角度の計算 [rad]
    # 推定角度をラジアンに変換し、基準角度(refangle)を足す
    theta_world = np.deg2rad(angle_deg) + refangle

    # (3) XY座標の計算 [m]
    dx = R_horizontal * np.cos(theta_world)
    dy = R_horizontal * np.sin(theta_world)

    # 世界座標（絶対座標）に変換
    x = p_bs[0] - dx
    y = p_bs[1] + dy

    return x, y, R

def convert_to_physical(d_idx, r_idx, N_trans, sym_duration, lam, current_angle, idx_to_xy_func, Tc, env):
    """
    インデックスを物理的な距離(m)と速度(m/s)に変換
    """
    # 速度計算
    vel_res = lam / (2.0 * sym_duration * N_trans)
    if d_idx < N_trans / 2:
        velocity = d_idx * vel_res
    else:
        velocity = (d_idx - N_trans) * vel_res
        
    # 距離・座標計算 (既存の関数を利用)
    x, y, r = idx_to_xy_func(r_idx, current_angle, Tc, env)
    
    return r, velocity, x, y