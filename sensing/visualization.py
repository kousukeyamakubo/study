import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def MUSIC_method_Debug(eig_val, N_ant, num_signals, resp, argm, M, frame_count):
    """
    MUSIC法の状態を診断するためのデバッグ関数
    """
    
    # --- 1. 数値情報の出力 ---
    # 固有値の絶対値（大きい順）
    eig_abs = np.abs(eig_val)
    print(eig_abs)

    # --- 2. グラフ描画 (重要) ---
    try:
        # --- ピークのグラフだけ表示 ---
        fig, ax = plt.subplots(figsize=(8, 5))
        resp_array = np.array(resp).flatten() # 形状を1次元に統一
        
        ax.plot(argm, resp_array, label='MUSIC Spectrum')
        ax.set_title(f'MUSIC Spectrum (N_ant={N_ant}, K={num_signals})')
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('Spectrum Power')
        ax.set_yscale('log') # スペクトルも対数で見ると弱いピークが見つけやすい
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
        # 閾値の線を引く (赤の点線)
        threshold_val = 0.2 * M
        #ax.axhline(y=threshold_val, color='r', linestyle='--', label='Threshold (20%)')
        
        # もし閾値を5%に下げたらどう見えるかも参考として表示 (緑の点線)
        #ax.axhline(y=0.00001 * M, color='g', linestyle=':', label='Threshold (0.001%)')
        
        #ax.legend()
        
        #plt.tight_layout()
        #plt.show()
        print("-> 診断用グラフを描画しました")
        save_dir = "./MUSIC/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"MUSIC_spectrum_{frame_count}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        
        # --- クローズ処理 ---
        # show() ではなく close() することで、画面表示せずバックグラウンドで処理できます
        # 表示もしたい場合は plt.show() の前に plt.savefig() を置いてください
        plt.close()

        
    except Exception as e:
        print(f"-> グラフ描画中にエラーが発生しました: {e}")
        
    print("="*30 + "\n")
    return

def plot_roots_on_unit_circle(roots, selected_roots):
    plt.figure(figsize=(6,6))
    
    # 単位円
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.5)
    
    # 全ての根
    plt.scatter(np.real(roots), np.imag(roots), marker='x', color='gray', label='All Roots')
    
    # 選択された根
    plt.scatter(np.real(selected_roots), np.imag(selected_roots), marker='o', color='red', label='Selected')
    
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid()
    plt.legend()
    plt.title("Root Constellation")
    plt.show()
    return

def RD_MAP(rd_maps, frame_id, T_symbol, Tc, save_dir="./rd_maps/"):
    l_speed = 299792458 
    f_carrier = 28 * 1e+9

    lam = l_speed / f_carrier
    
    
    # ディレクトリがなければ作成
    os.makedirs(save_dir, exist_ok=True)
    
    rdresp_list = np.array(rd_maps[::2])
    #idx = np.array(rd_maps[1::2])
    
    # 距離分解能 (m)
    range_res = l_speed * Tc / 2
    
    # 横軸 (距離):
    start_sample_idx = 620
    num_range_bins = 190

    r_min = start_sample_idx * range_res
    r_max = (start_sample_idx + num_range_bins) * range_res

    v_range = lam / (2 * T_symbol)  # 最大速度範囲

    v_min_val = -v_range / 2
    v_max_val = v_range / 2
    
    i = 0

    for rd_resp in rdresp_list:
        #output = rd_resp[:100,:] + 1e-10
        #output = rd_resp + 1e-10
        #output = np.fft.fftshift(rd_resp, axes=0) + 1e-10
        # 1. ゼロ周波数を中央にシフト (縦軸方向)
        output_shifted = np.fft.fftshift(rd_resp, axes=0)
        # 2. dB変換 (対数)
        # 【重要】絶対値をとってから微小値を足す (0除算防止 & ノイズフロア確保)
        output_db = 20 * np.log10(np.abs(output_shifted) + 1e-20)

        #plt.imshow(20*np.log10(np.abs(output)), aspect='auto', interpolation='nearest',
        #     extent = [r_min, r_max, v_min_val, v_max_val], origin='lower', vmin=-30, vmax = -10)
        plt.imshow(output_db, 
                   aspect='auto', 
                   interpolation='nearest', # ドットを潰さない
                   extent=[r_min, r_max, v_min_val, v_max_val], 
                   origin='lower', 
                   vmin=-30,   # 引数で指定した値を使用
                   vmax=10,   # 引数で指定した値を使用
        )
        plt.xlabel('Range [m]')
        plt.ylabel('Velocity [m/s]') # 注: DFTのシフトをしていないため、軸の解釈に注意が必要
        plt.colorbar(label='Power [dB]')
        plt.title('Range-Doppler Map')
        plt.grid(False) # 任意
        plt.xlim(r_min,r_max)
        plt.ylim(-30,30)
        # --- 保存処理 ---
        # ファイル名を決定 (例: ./rd_maps/rd_map_12345.png)
        save_path = os.path.join(save_dir, f"rd_map_{frame_id}_for_{i}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        
        # --- クローズ処理 ---
        # show() ではなく close() することで、画面表示せずバックグラウンドで処理できます
        # 表示もしたい場合は plt.show() の前に plt.savefig() を置いてください
        plt.close()
        i += 1
    
    # 保存する
    return
    
def plot_range_power_with_miss_counts(file_path):
    # 1. CSVファイルを読み込む
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりませんでした。")
        return

    # 必要な列の確認
    required_cols = ['Time', 'Range', 'Power']
    if not all(col in df.columns for col in required_cols):
        print(f"エラー: 必要な列 {required_cols} がCSVに含まれていません。")
        return

    # 2. データの整理と欠損の特定
    # Timeをインデックスに設定
    df = df.sort_values('Time')
    df = df.set_index('Time')

    # 時間ステップ（間隔）を自動判定（最頻値を使用）
    # データが1行しかない場合などは1と仮定
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        dt = time_diffs.mode()[0]  # 最も頻出する間隔をステップとする
    else:
        dt = 1 

    # 完全な時間の並びを作成（最初から最後までの等間隔な時刻）
    full_time_idx = range(int(df.index.min()), int(df.index.max()) + int(dt), int(dt))
    
    # データを再構築（欠損していた時刻にはNaNが入る）
    df_reindexed = df.reindex(full_time_idx)

    # 検出漏れの判定（RangeまたはPowerがNaNの行）
    is_missed = df_reindexed['Range'].isna() | df_reindexed['Power'].isna()
    miss_count = is_missed.sum()

    print(f"検出漏れ回数: {miss_count} 回")

    # 3. プロット用のデータ準備
    # 検出漏れ箇所をプロットするために補間（Interpolate）を行う
    # Time（インデックス）に基づいて線形補間
    df_interp = df_reindexed.interpolate(method='index')

    # プロット開始
    plt.figure(figsize=(10, 6))

    # (A) 検出成功データのプロット（青い線と丸）
    # 欠損箇所で線が切れるように、補間していない元のdf_reindexedを使用
    plt.plot(df_reindexed['Range'], df_reindexed['Power'], 
             'o-', color='blue', alpha=0.7, label='Detected Signal')

    # (B) 検出漏れ箇所のプロット（赤い×）
    # 補間データのうち、検出漏れだった箇所だけを抽出してプロット
    if miss_count > 0:
        missed_data = df_interp[is_missed]
        plt.scatter(missed_data['Range'], missed_data['Power'], 
                    color='red', marker='x', s=100, zorder=5, 
                    label=f'Missed Detection ({miss_count} times)')
        
        # オプション: 検出漏れ箇所を赤い点線で結ぶ場合
        # plt.plot(df_interp['Range'], df_interp['Power'], 'r:', alpha=0.4)

    # 4. グラフの装飾
    plt.title(f'Range vs Power (Total Misses: {miss_count})')
    plt.xlabel('Distance (Range) [m]')
    plt.ylabel('Intensity (Power) [dB]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(-28, -7)
    # 表示
    plt.show()

    if __name__ == "__main__":
        plot_range_power_with_miss_counts('montecarlo/measurements_0.csv')

    return

def create_gif(music_folder="MUSIC", output_path="output.gif"):
    # ノートブック末尾のGIF生成コードを関数化してここに置く
    # MUSICフォルダ内の画像ファイルを取得
    music_folder = "MUSIC"
    image_files = sorted(glob.glob(os.path.join(music_folder, "*.png"))) + \
                sorted(glob.glob(os.path.join(music_folder, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(music_folder, "*.jpeg")))

    if not image_files:
        print("画像ファイルが見つかりません")
        exit()

    # 画像を読み込む
    images = []
    for file in image_files:
        img = Image.open(file)
        images.append(img)
        print(f"読み込み: {file}")

    # GIFとして保存
    output_path = "output.gif"
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 各フレームの表示時間（ミリ秒）
        loop=0  # 0は無限ループ
    )

    print(f"GIFを作成しました: {output_path}")
    return