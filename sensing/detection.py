import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from geometry import idx_to_xy, convert_to_physical  # ← 参照先が変わる！

def ca_cfar_improved(x_db, n_train=(1,7), n_guard=(1,5), pfa=1e-6, convert_from_db=False):
    """
    2次元 CA-CFAR (Cell Averaging CFAR)
    x_db: (B, 1, H, W) 入力マップ (通常はdB)
    戻り値: (B, 1, H, W) のBool型テンソル (Trueが検出位置)
    """
    # 修正
    n_guard = (1,5)
    pfa = 1e-4

    if convert_from_db:
        x = 10 ** (x_db / 10.0)
    else:
        x = x_db

    B, C, H, W = x.shape
    th, tw = n_train
    gh, gw = n_guard

    # カーネル作成 (ガードセルを0、参照セルを1にする)
    kh = 2 * (th + gh) + 1
    kw = 2 * (tw + gw) + 1
    kernel = torch.ones((1, 1, kh, kw), device=x.device, dtype=x.dtype)

    #kernel[:, :, th-gh:th+gh+1, tw-gw:tw+gw+1] = 0
    # 修正案
    kernel[:, :, th : th + 2*gh + 1, tw : tw + 2*gw + 1] = 0

    N_train = kernel.sum().item()
    
    # 閾値係数 alpha の計算
    alpha = N_train * (pfa ** (-1.0 / N_train) - 1.0)

    # 修正案
    # ドップラー方向(H)は「circular」(循環)、レンジ方向(W)は「replicate」(端の値を複製)したい場合
    # PyTorchのF.padは一度に1つのモードしか指定できないため、2段階で適用します

    # 1. レンジ方向（左右）のパディング（非循環）
    pad_w = kw // 2
    x_pad_w = F.pad(x, (pad_w, pad_w, 0, 0), mode="replicate") # または "constant"

    # 2. ドップラー方向（上下）のパディング（循環：上端と下端をつなげる）
    pad_h = kh // 2
    x_pad = F.pad(x_pad_w, (0, 0, pad_h, pad_h), mode="circular")
    
    train_sum = F.conv2d(x_pad, kernel)
    noise_est = train_sum / N_train

    threshold_map = alpha * noise_est
    detections = x > threshold_map
    
    return detections

def extract_peaks_scipy(radar_db, detections_bool, kernel_size=5, tar_thres=-25):
    """
    SciPyを用いた可読性の高いNMS（局所最大値検出）。
    既存の thresholder_improved と cleansing_improved を代替します。
    
    Args:
        radar_db (Tensor or ndarray): dB単位のRDマップ
        detections_bool (Tensor or ndarray): CFARの検出結果マスク
        kernel_size (int): 近傍とみなす範囲（奇数推奨, 例: 5なら5x5マス）
        tar_thres (float): パワーの閾値
        
    Returns:
        list: [[dop_idx, rng_idx, power], ...] 
        (後段の処理との互換性を維持したリスト形式)
    """
    # 1. データ型をNumPyに統一 (Tensorが来ても対応)
    if hasattr(radar_db, 'cpu'):
        img = radar_db.cpu().numpy()
    else:
        img = radar_db
        
    if hasattr(detections_bool, 'cpu'):
        mask = detections_bool.cpu().numpy()
    else:
        mask = detections_bool

    # 2. Maximum Filter で「近傍の最大値」を計算
    # これが「その範囲内で一番強い値」のマップになります
    local_max = maximum_filter(img, size=kernel_size, mode='constant', cval=-999)

    # 3. ピークの特定 (Boolean Indexing)
    # 条件A: CFARで検出されている
    # 条件B: 自分の値が近傍最大値と一致する (つまり自分がピーク)
    # 条件C: 閾値を超えている
    is_peak = mask & (img == local_max) & (img > tar_thres)
    #is_peak = mask & (img > tar_thres)

    # 4. インデックスの抽出とリスト化
    # np.argwhere で True の場所の (d_idx, r_idx) を一括取得
    peak_indices = np.argwhere(is_peak)
    
    results = []
    for d_idx, r_idx in peak_indices:
        power = float(img[d_idx, r_idx])
        results.append([d_idx, r_idx, power])
    
    # パワー順にソート (既存処理との整合性のため)
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results

def remove_ghosts_improved(all_detections, range_tol=0.4, ang_tol=0.01, vel_tol=11.0, power_ratio_thres_db=2.0, dist_xy_tol=2.0):
    """
    物理的な重複判定とサイドローブ除去を区別して処理する関数
    
    Args:
        all_detections: [[range, angle, x, y, vel, power], ...]
        range_tol: 同一物体とみなす距離差 [m]
        ang_tol: 同一物体とみなす角度差 [deg]
        vel_tol: 同一物体とみなす速度差 [m/s]
        power_ratio_thres_db: 角度が離れている場合、このdB以上弱ければゴーストとみなす [dB]
        dist_xy_tol: XY座標上での距離許容値 [m] (物理的な近さ)
    """
    if not all_detections:
        return []
    
    # 角度ごとの検出数を記録
    angle_group_counts = {}
    for det in all_detections:
        angle = det[1]
        # 角度を丸めてグループ化 (ang_tol の半分程度の精度で)
        angle_key = round(angle / (ang_tol * 0.5)) * (ang_tol * 0.5)
        angle_group_counts[angle_key] = angle_group_counts.get(angle_key, 0) + 1

    def get_angle_group_count(angle):
        """この角度が属するグループの検出数を返す"""
        angle_key = round(angle / (ang_tol * 0.5)) * (ang_tol * 0.5)
        return angle_group_counts.get(angle_key, 1)

    # 1. パワーが強い順に並べ替える (index 5 = power)
    sorted_dets = sorted(all_detections, key=lambda x: x[5], reverse=True)
    
    final_unique_objects = []
    
    while sorted_dets:
        # 最も強い検出を取り出す（これが「本物」の可能性が最も高い）
        best = sorted_dets.pop(0)
        final_unique_objects.append(best)
        
        remaining = []
        for other in sorted_dets:
            # 各差分を計算
            r_diff = abs(best[0] - other[0])
            a_diff = abs(best[1] - other[1])
            v_diff = abs(best[4] - other[4])
            
            # XY座標での距離 (物理的に同じ場所か？)
            # best[2], best[3] が X, Y
            xy_dist = np.sqrt((best[2] - other[2])**2 + (best[3] - other[3])**2)
            
            # パワー差 (dB)
            p_diff = best[5] - other[5] # sorted済みなので必ず正の値

            # --- 判定ロジック ---

            # A. 「完全に同一の物体」の結合
            # 距離・角度・速度がすべて近い、またはXY座標が非常に近い場合
            is_same_target = (r_diff < range_tol and a_diff < ang_tol and v_diff < vel_tol) or (xy_dist < dist_xy_tol)
            is_same_target = (r_diff < range_tol and v_diff < vel_tol)

            # B. 「サイドローブ（ゴースト）」の判定
            # 距離と速度は同じだが、角度が違う場合。
            # ただし、パワーが「圧倒的に弱い（power_ratio_thres_db以上差がある）」場合のみゴーストとする。
            # ※ パワー差が小さいなら、それは並走している別の車両や自転車の可能性が高い。
            is_sidelobe = (r_diff < range_tol and v_diff < vel_tol) and (p_diff < power_ratio_thres_db)
            is_sidelobe = (a_diff < ang_tol) and (p_diff < power_ratio_thres_db)
            
            other_angle_count = get_angle_group_count(other[1])
            if other_angle_count == 1:
                is_sidelobe = False
                is_same_target = False
            # どちらにも該当しなければ、リストに残す（次のループで採用される候補）
            if is_same_target:
                print("同一物体として認識されたため除去")
                print(other)
                continue 
            elif is_sidelobe:
                print("サイドローブとして認識されたため除去")
                continue
            else:
                remaining.append(other)
        
        sorted_dets = remaining

    return final_unique_objects

def CFAR(rd_maps, est_ang, N_trans, sym_duration, lam, Tc, env):
    detected_objects = []
    for k in range(0, len(rd_maps), 2):
        rdresp_single = rd_maps[k]
        angle = rd_maps[k+1]
        # 1. CFAR実行
        # radar_inの作成 (Batch, Channel, Height, Width)
        radar_in = torch.tensor(np.abs(rdresp_single)).unsqueeze(0).unsqueeze(0)
        detections_bool = ca_cfar_improved(radar_in, convert_from_db=False)

        # 2. dBマップの準備 (Thresholder用)
        radar_db = 20 * torch.log10(radar_in[0,0])

        # 3. 閾値判定 (速度方向も見てフィルタリング)
        clean_targets = extract_peaks_scipy(radar_db, detections_bool[0,0], kernel_size=3, tar_thres=-24)

        # 5. 結果の保存
        if len(clean_targets) > 0:
            print(f"角度 {est_ang[angle]:.2f} 度 のマップからの検出:")
            for tgt in clean_targets:
                d_idx, r_idx, power = tgt
                
                # 物理量への変換 (別途 sym_duration, lam の定義が必要)
                r_est, v_est, x_est, y_est = convert_to_physical(
                    d_idx, r_idx, N_trans, sym_duration, lam, est_ang[angle], idx_to_xy, Tc, env
                )
                #detected_objects.append([r_est, v_est, current_angle, x_est, y_est]) # v_estを追加
                # 【変更点】 power もリストに追加します (末尾に追加)
                detected_objects.append([r_est, est_ang[angle], x_est, y_est, v_est, power])
                print(f"  -> Index: ({d_idx}, {r_idx}), Power: {power:.1f}dB, Range: {r_est:.2f}m, Vel: {v_est:.2f}m/s")
        else:
            print("ターゲット不検出")
        # --- ここまで ---
    # 【追加】 ここで全角度の検出結果をまとめてクリーニングします
    #final_objects = remove_ghosts_across_angles(detected_objects)    
    final_objects = remove_ghosts_improved(detected_objects)
    return final_objects