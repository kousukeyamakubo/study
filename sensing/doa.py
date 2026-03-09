import numpy as np
import time
from geometry import stevec          # ← stevec の参照先が変わる！
from visualization import MUSIC_method_Debug, plot_roots_on_unit_circle

def MUSIC_method_improved(Y, N_ant, num_signals, frame_count):
    '''
    修正前
    Y_music = np.mean(Y, axis=0)
    R_yy = Y_music@np.conjugate(np.transpose(Y_music))
    '''
    # 修正後: 
    # チャープ軸(0)とサンプル軸(2)の両方を「スナップショット」として扱います。
    # (N_trans, N_ant, N_sample) -> (N_ant, N_trans * N_sample) に変形
    # これにより、N_trans(チャープ数)が増えるほど、スナップショット数が増え、
    # 共分散行列 R_yy の推定精度が向上します。
    t0 = time.perf_counter()
    N_trans, N_ant_data, N_sample = Y.shape
    # データを (アンテナ数 x 全サンプル数) の2次元行列に変形
    Y_music = Y.transpose(1, 0, 2).reshape(N_ant_data, -1)
    t1 = time.perf_counter()
    
    # 共分散行列の計算
    # (行列サイズは アンテナ数 x アンテナ数 のまま変わらないので計算負荷は軽微です)
    R_yy = (Y_music @ np.conjugate(Y_music.T)) / Y_music.shape[1]
    t2 = time.perf_counter()
    
    # 1.固有値分解
    eig_val, eig_vec = np.linalg.eig(R_yy)
    t3 = time.perf_counter()
    
    # 2.固有値をソートして雑音部分空間を取得
    idx = np.abs(eig_val).argsort()[::-1]
    # インデックスによって固有値と固有ベクトルの対応関係を維持しつつソート
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    # 雑音部分空間Uを取得

    # num_signalsを固有値から推定?
    
    # デバッグ用
    num_signals = 5
    
    U = eig_vec[:, num_signals:]
    t4 = time.perf_counter()
    

    # 3.雑音部分空間とvalごとのステアリングベクトルの内積を計算
    resp = []
    argm = (np.arange(1800)-900)/100
    for val in argm:
        # ステアリングベクトルの計算
        stv = stevec(N_ant, val*np.pi/180)
        # 雑音部分空間との内積
        p = stv.T@U
        # L2ノルムの二乗を計算(pp^H)
        pp = p*np.conjugate(np.transpose(p))
        # A1は行列を1次元配列に変換するメソッド
        # respに逆数を追加
        # 分子は角度推定の上では不要なので省略している
        resp.append(1/(np.abs(pp)**2).A1)
    t5 = time.perf_counter()
    
    # 4.ピーク検出
    M = np.max(resp)
    
    # デバッグ用
    #MUSIC_method_Debug(eig_val, N_ant, num_signals, resp, argm, M, frame_count)
    
    est_ang = []
    # 三点比較によるピーク検出
    a, b = resp[0][0], resp[1][0]
    for i in range(1,len(argm)-1):
        c = resp[i+1][0]
        #if a < b and b > c and b > 0.2 * M:
        if a < b and b > c and b > 0.00001 * M:
            est_ang.append(argm[i])
        a, b = b, c
    t6 = time.perf_counter()

    
    print(f"[MUSIC] 1. データ整形:        {(t1-t0)*1000:.3f} ms")
    print(f"[MUSIC] 2. 共分散行列:         {(t2-t1)*1000:.3f} ms")
    print(f"[MUSIC] 3. 固有値分解:         {(t3-t2)*1000:.3f} ms")
    print(f"[MUSIC] 4. ソート＆部分空間:   {(t4-t3)*1000:.3f} ms")
    print(f"[MUSIC] 5. スペクトル走査:     {(t5-t4)*1000:.3f} ms  ← グリッド数: {len(argm)}")
    print(f"[MUSIC] 6. ピーク検出:         {(t6-t5)*1000:.3f} ms")
    print(f"[MUSIC] 合計:                  {(t6-t0)*1000:.3f} ms")
    return est_ang