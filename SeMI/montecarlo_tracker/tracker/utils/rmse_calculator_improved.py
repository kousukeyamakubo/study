import math
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 設定: 成功とみなす誤差の閾値 (m)
SUCCESS_THRESHOLD = 1.0 
# ==========================================

def find_col(df, hints):
    cols = {c.lower(): c for c in df.columns}
    for h in hints:
        if h in cols:
            return cols[h]
    for h in hints:
        for lc, orig in cols.items():
            if h in lc:
                return orig
    return None

def load_csv(path):
    return pd.read_csv(path)

def compute_metrics(diffs, threshold):
    """
    RMSE, Median, Accuracy(閾値以内の割合)を計算
    """
    diffs = np.array(diffs)
    if diffs.size == 0:
        return float('nan'), float('nan'), 0.0
    
    sq_errors = np.sum(diffs ** 2, axis=1)
    distances = np.sqrt(sq_errors)
    
    rmse = math.sqrt(np.mean(sq_errors))
    median = np.median(distances)
    
    # Accuracy: 検出できたものの中で、誤差が閾値以内の割合
    accuracy_count = np.sum(distances < threshold)
    accuracy_rate = (accuracy_count / len(distances)) * 100.0
    
    return rmse, median, accuracy_rate

def main():
    base = Path(__file__).resolve().parent.parent
    
    # ファイル読み込み
    try:
        true_df = load_csv(base / "true_data.csv")
        meas_df = load_csv(base / "measurements.csv")
        est_df  = load_csv(base / "csv_result" / "estimated_trajectory.csv")
    except FileNotFoundError as e:
        print(f"Error: ファイルが見つかりません。\n{e}")
        return

    # カラム特定
    t_time = find_col(true_df, ["time"])
    t_id   = find_col(true_df, ["target_id", "targetid", "id"])
    t_x    = find_col(true_df, ["x"])
    t_y    = find_col(true_df, ["y"])

    m_time = find_col(meas_df, ["time"])
    m_x    = find_col(meas_df, ["x"])
    m_y    = find_col(meas_df, ["y"])

    e_time = find_col(est_df, ["time"])
    e_id   = find_col(est_df, ["target_id", "targetid", "id"])
    e_x    = find_col(est_df, ["x"])
    e_y    = find_col(est_df, ["y"])

    true_df = true_df.dropna(subset=[t_time])
    meas_df = meas_df.dropna(subset=[m_time])
    est_df  = est_df.dropna(subset=[e_time])

    targets = sorted(pd.unique(true_df[t_id])) if t_id else sorted(pd.unique(est_df[e_id]))
    results = []

    # 全体集計用
    all_diffs_meas = []
    all_diffs_est = []
    total_samples = 0
    total_detected_meas = 0
    total_detected_est = 0

    print(f"Processing {len(targets)} targets...")

    for tid in targets:
        true_sub = true_df[true_df[t_id] == tid]
        times = sorted(pd.unique(true_sub[t_time]))
        n_samples = len(times)
        total_samples += n_samples

        diffs_meas = []
        diffs_est = []

        for t in times:
            trow = true_sub[true_sub[t_time] == t]
            if trow.empty: continue
            tx = float(trow.iloc[0][t_x])
            ty = float(trow.iloc[0][t_y])

            # --- Estimated ---
            est_row = est_df[(est_df[e_id] == tid) & (est_df[e_time] == t)]
            if not est_row.empty:
                ex = float(est_row.iloc[0][e_x])
                ey = float(est_row.iloc[0][e_y])
                diffs_est.append([ex - tx, ey - ty])
                all_diffs_est.append([ex - tx, ey - ty])

            # --- Measurements ---
            meas_rows = meas_df[meas_df[m_time] == t]
            meas_rows = meas_rows.dropna(subset=[m_x, m_y])
            if not meas_rows.empty:
                mx = meas_rows[m_x].astype(float).values
                my = meas_rows[m_y].astype(float).values
                d2 = (mx - tx) ** 2 + (my - ty) ** 2
                idx = int(np.argmin(d2))
                mx_sel = float(mx[idx])
                my_sel = float(my[idx])
                diffs_meas.append([mx_sel - tx, my_sel - ty])
                all_diffs_meas.append([mx_sel - tx, my_sel - ty])

        # 指標計算
        # 1. 検出率 (Detection Rate) = (検出できた数 / 全サンプル数)
        det_rate_m = (len(diffs_meas) / n_samples) * 100.0
        det_rate_e = (len(diffs_est) / n_samples) * 100.0
        total_detected_meas += len(diffs_meas)
        total_detected_est += len(diffs_est)

        # 2. 精度 (RMSE, Median, Accuracy)
        rmse_m, med_m, acc_m = compute_metrics(diffs_meas, SUCCESS_THRESHOLD)
        rmse_e, med_e, acc_e = compute_metrics(diffs_est, SUCCESS_THRESHOLD)
        
        results.append({
            'id': tid, 'n': n_samples,
            'm': {'det': det_rate_m, 'rmse': rmse_m, 'med': med_m, 'acc': acc_m},
            'e': {'det': det_rate_e, 'rmse': rmse_e, 'med': med_e, 'acc': acc_e}
        })

    # 全体評価
    ov_rmse_m, ov_med_m, ov_acc_m = compute_metrics(all_diffs_meas, SUCCESS_THRESHOLD)
    ov_rmse_e, ov_med_e, ov_acc_e = compute_metrics(all_diffs_est, SUCCESS_THRESHOLD)
    ov_det_m = (total_detected_meas / total_samples * 100.0) if total_samples > 0 else 0
    ov_det_e = (total_detected_est / total_samples * 100.0) if total_samples > 0 else 0

    # ==========================
    # 結果表示
    # ==========================
    print("\n" + "=" * 95)
    print(f"EVALUATION REPORT (Threshold < {SUCCESS_THRESHOLD}m)")
    print("  * Det%: Detection Rate (Availability)")
    print("  * Acc%: Position Accuracy within threshold (Precision)")
    print("=" * 95)
    
    # Header
    print(f"{'Target':<6} | {'N':<3} || {'MEASUREMENTS (Sensing)':^38} || {'ESTIMATED (Tracking)':^38}")
    print(f"{'ID':<6} | {'':<3} || {'Det%':<6} {'RMSE':<6} {'Median':<6} {'Acc%':<6} || {'Det%':<6} {'RMSE':<6} {'Median':<6} {'Acc%':<6}")
    print("-" * 95)
    
    def fmt(val, is_pct=False):
        if math.isnan(val): return "-"
        return f"{val:5.1f}%" if is_pct else f"{val:.3f}"

    for r in results:
        print(f"{r['id']:<6} | {r['n']:<3} || "
              f"{fmt(r['m']['det'], True):<6} {fmt(r['m']['rmse']):<6} {fmt(r['m']['med']):<6} {fmt(r['m']['acc'], True):<6} || "
              f"{fmt(r['e']['det'], True):<6} {fmt(r['e']['rmse']):<6} {fmt(r['e']['med']):<6} {fmt(r['e']['acc'], True):<6}")

    print("-" * 95)
    print("OVERALL SUMMARY:")
    print(f"Measurements: Detection={ov_det_m:.1f}%, RMSE={ov_rmse_m:.3f}, Acc(within {SUCCESS_THRESHOLD}m)={ov_acc_m:.1f}%")
    print(f"Estimated:    Detection={ov_det_e:.1f}%, RMSE={ov_rmse_e:.3f}, Acc(within {SUCCESS_THRESHOLD}m)={ov_acc_e:.1f}%")
    print("=" * 95)

if __name__ == '__main__':
    main()