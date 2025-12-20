import math
import numpy as np
import pandas as pd
from pathlib import Path


def find_col(df, hints):
    cols = {c.lower(): c for c in df.columns}
    for h in hints:
        if h in cols:
            return cols[h]
    # fallback: find any column containing hint
    for h in hints:
        for lc, orig in cols.items():
            if h in lc:
                return orig
    return None


def load_csv(path):
    return pd.read_csv(path)


def compute_rmse(diffs):
    diffs = np.array(diffs)
    if diffs.size == 0:
        return float('nan')
    return math.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))


def main():
    base = Path(__file__).resolve().parent.parent
    """
    true_path = base + "\\true_data.csv"
    meas_path = base + "\\measurements.csv"
    est_path = base + "\\csv_result\\estimated_trajectory.csv"
    """
    true_path = base / "true_data.csv"
    meas_path  = base / "measurements.csv"
    est_path   = base / "csv_result" / "estimated_trajectory.csv"
    true_df = load_csv(true_path)
    meas_df = load_csv(meas_path)
    est_df = load_csv(est_path)

    # find column names
    t_time = find_col(true_df, ["time"])
    t_id = find_col(true_df, ["target_id", "targetid", "id"])
    t_x = find_col(true_df, ["x"])
    t_y = find_col(true_df, ["y"])

    m_time = find_col(meas_df, ["time"])
    m_x = find_col(meas_df, ["x"])
    m_y = find_col(meas_df, ["y"])

    e_time = find_col(est_df, ["time"])
    e_id = find_col(est_df, ["target_id", "targetid", "id"])
    e_x = find_col(est_df, ["x"])
    e_y = find_col(est_df, ["y"])

    # standardize column access
    true_df = true_df.dropna(subset=[t_time])
    meas_df = meas_df.dropna(subset=[m_time])
    est_df = est_df.dropna(subset=[e_time])

    targets = sorted(pd.unique(true_df[t_id])) if t_id else sorted(pd.unique(est_df[e_id]))

    results = []
    all_diffs_meas = []
    all_diffs_est = []

    for tid in targets:
        # times where true exists for this target
        true_sub = true_df[true_df[t_id] == tid]
        times = sorted(pd.unique(true_sub[t_time]))

        diffs_meas = []
        diffs_est = []

        for t in times:
            trow = true_sub[true_sub[t_time] == t]
            if trow.empty:
                continue
            tx = float(trow.iloc[0][t_x])
            ty = float(trow.iloc[0][t_y])

            # estimated
            est_row = est_df[(est_df[e_id] == tid) & (est_df[e_time] == t)]
            if not est_row.empty:
                ex = float(est_row.iloc[0][e_x])
                ey = float(est_row.iloc[0][e_y])
                diffs_est.append([ex - tx, ey - ty])
                all_diffs_est.append([ex - tx, ey - ty])

            # measurements: there may be multiple measurements at same time; choose closest to true
            meas_rows = meas_df[meas_df[m_time] == t]
            # drop rows without X/Y
            meas_rows = meas_rows.dropna(subset=[m_x, m_y])
            if not meas_rows.empty:
                # compute distances to true
                mx = meas_rows[m_x].astype(float).values
                my = meas_rows[m_y].astype(float).values
                d2 = (mx - tx) ** 2 + (my - ty) ** 2
                idx = int(np.argmin(d2))
                mx_sel = float(mx[idx])
                my_sel = float(my[idx])
                diffs_meas.append([mx_sel - tx, my_sel - ty])
                all_diffs_meas.append([mx_sel - tx, my_sel - ty])

        rmse_meas = compute_rmse(diffs_meas)
        rmse_est = compute_rmse(diffs_est)
        results.append((tid, len(times), rmse_meas, rmse_est))

    overall_meas = compute_rmse(all_diffs_meas)
    overall_est = compute_rmse(all_diffs_est)

    # print results
    print("Per-target RMSE (positions):")
    print("target_id, samples, rmse(measurements vs true), rmse(estimated vs true)")
    for tid, n, rmeas, rest in results:
        print(f"{tid}, {n}, {rmeas:.6f}, {rest:.6f}")

    print("")
    print("Overall RMSE:")
    print(f"measurements vs true: {overall_meas:.6f}")
    print(f"estimated vs true:   {overall_est:.6f}")


if __name__ == '__main__':
    main()
