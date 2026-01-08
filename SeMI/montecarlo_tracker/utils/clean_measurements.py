import pandas as pd
import os


def clean(path_in, path_out):
    df = pd.read_csv(path_in)

    # normalize column names to lower
    df.columns = [c.strip().lower() for c in df.columns]

    # prefer columns names
    time_col = None
    for c in ['time', 't', 'timestamp']:
        if c in df.columns:
            time_col = c
            break

    x_col = None
    y_col = None
    for c in ['x', 'xx']:
        if c in df.columns:
            x_col = c
            break
    for c in ['y', 'yy']:
        if c in df.columns:
            y_col = c
            break

    # drop rows where both x and y are missing
    if x_col and y_col:
        df = df.dropna(subset=[x_col, y_col], how='all')

    # ensure time is integer-like
    if time_col:
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce').astype('Int64')

    # round numeric columns for readability
    for c in df.select_dtypes(include=['float', 'float64', 'int']).columns:
        df[c] = df[c].round(3)

    # standardize column order
    cols = []
    if time_col:
        cols.append(time_col)
    for c in ['x', 'y', 'range', 'angle', 'velocity']:
        if c in df.columns:
            cols.append(c)
    # append any remaining columns
    for c in df.columns:
        if c not in cols:
            cols.append(c)

    df = df[cols]

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_csv(path_out, index=False)
    return df


def main():
    base = r"c:\Users\kosuke\study\tracker"
    in_root = base + r"\measurements.csv"
    in_result = base + r"\csv_result\measurements.csv"
    out_path = base + r"\csv_result\measurements_cleaned.csv"

    # prefer csv_result version if present
    path_in = in_result if os.path.exists(in_result) else in_root

    df = clean(path_in, out_path)
    print(f"Cleaned measurements saved to: {out_path}")
    print(df.head(40).to_string(index=False))


if __name__ == '__main__':
    main()
