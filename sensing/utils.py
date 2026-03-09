import os
import math

def derectory_deletor(directory_path):
    try:
        # ディレクトリ内のすべてのエントリ（ファイルとフォルダ）を取得
        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)

            # ファイルかどうかをチェック
            if os.path.isfile(full_path):
                # ファイルを削除
                os.remove(full_path)
            
            # サブディレクトリの場合はスキップ
            elif os.path.isdir(full_path):
                print(f"⏭️ サブディレクトリはスキップしました: {full_path}")
                # サブディレクトリ内のファイルも削除したい場合は、
                # ここで再帰的にこの関数を呼び出すなどの処理が必要です
                # (例: delete_all_files_in_directory(full_path))
    except FileNotFoundError:
        print(f"エラー: ディレクトリが見つかりません - {directory_path}")
    except Exception as e:
        print(f"ファイル削除中にエラーが発生しました: {e}")
    return

def print_true_values(num_cy, num_ve, num_rp, phys_quantities, real_cy_coordinates, real_vel_coordinates, real_rp_coordinates,
                      real_cy_coordinates_all, real_cy_ranges_all, real_cy_angles_all, real_cy_velocities_all,
                      real_ve_coordinates_all, real_ve_ranges_all, real_ve_angles_all, real_ve_velocities_all):
    if num_cy > 0:
        print("Cyclist True Position:", real_cy_coordinates)
        print("Cyclist True Range:", phys_quantities["range"]["cyclists"][0])
        print("Cyclist True Angle:", math.degrees(phys_quantities["angle"]["cyclists"][0]))
        a = math.degrees(phys_quantities["angle"]["cyclists"][0])
        real_cy_coordinates_all.append(real_cy_coordinates)
        real_cy_ranges_all.append(phys_quantities["range"]["cyclists"][0])
        real_cy_angles_all.append(phys_quantities["angle"]["cyclists"][0])
        real_cy_velocities_all.append(phys_quantities["relative_velocity"]["cyclists"][0])
    if num_ve > 0:
        print("Vehicle True Position:", real_vel_coordinates)
        print("Vehicle True Range:", phys_quantities["range"]["vehicles"][0])
        print("Vehicle True Angle:", math.degrees(phys_quantities["angle"]["vehicles"][0]))
        b= math.degrees(phys_quantities["angle"]["vehicles"][0])
        real_ve_coordinates_all.append(real_vel_coordinates)
        real_ve_ranges_all.append(phys_quantities["range"]["vehicles"][0])
        real_ve_angles_all.append(phys_quantities["angle"]["vehicles"][0])
        real_ve_velocities_all.append(phys_quantities["relative_velocity"]["vehicles"][0])
    if num_rp > 0:
        print("Ramp Post True Position:", real_rp_coordinates)
        print("Ramp Post True Range:", phys_quantities["range"]["ramposts"])
        print("Ramp Post True Angle:", math.degrees(phys_quantities["angle"]["ramposts"]))

    #print('angle difference:', np.abs(a-b))
    
    return