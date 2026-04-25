import pandas as pd


def load_logs(path="data/raw/test_logs.csv"):
    """
    원본 로그 csv 로드
    """
    df = pd.read_csv(path)
    return df


def add_rolling_features(df):
    """
    이상 탐지에 쓸 rolling feature 추가
    scenario별로 따로 계산해야 함
    """
    result = []

    for scenario_id, group in df.groupby("scenario_id"):
        g = group.sort_values("timestamp").copy()

        # rolling mean / std
        g["vt_shift_roll_mean"] = g["vt_shift"].rolling(window=20, min_periods=1).mean()
        g["vt_shift_roll_std"] = g["vt_shift"].rolling(window=20, min_periods=1).std().fillna(0)

        g["leakage_curr_roll_mean"] = g["leakage_curr"].rolling(window=20, min_periods=1).mean()
        g["test_time_roll_mean"] = g["test_time_ms"].rolling(window=20, min_periods=1).mean()

        # baseline 대비 차이
        g["vt_shift_diff"] = g["vt_shift"] - g["vt_shift_roll_mean"]
        g["test_time_diff"] = g["test_time_ms"] - g["test_time_roll_mean"]

        result.append(g)

    out = pd.concat(result, ignore_index=True)
    return out


def preprocess_logs(path="data/raw/test_logs.csv"):
    """
    전체 전처리 파이프라인
    """
    df = load_logs(path)
    df = add_rolling_features(df)
    return df
