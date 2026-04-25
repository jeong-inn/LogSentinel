import pandas as pd

DATASET_LABELS = {
    "ambient_temperature_system_failure": [
        "2013-12-22 20:00:00",
        "2014-04-13 09:00:00",
    ],
    "machine_temperature_system_failure": [
        "2013-12-11 06:00:00",
        "2013-12-16 17:25:00",
        "2014-01-28 13:55:00",
        "2014-02-08 14:30:00",
    ],
    "cpu_utilization_asg_misconfiguration": [
        "2014-07-12 02:04:00",
        "2014-07-14 21:44:00",
    ],
    "ec2_request_latency_system_failure": [
        "2014-03-14 09:06:00",
        "2014-03-18 22:41:00",
        "2014-03-21 03:01:00",
    ],
    "rogue_agent_key_hold": [
        "2014-07-15 08:30:00",
        "2014-07-17 09:50:00",
    ],
}


def _build_ground_truth_windows(df, label_timestamps, window_radius=12):
    """
    라벨 시점 주변 +/- window_radius row를 anomaly window로 간주
    """
    df = df.copy()
    df["ground_truth"] = 0

    for label_ts in label_timestamps:
        ts = pd.Timestamp(label_ts)
        nearest_idx = (df["timestamp"] - ts).abs().idxmin()

        start_idx = max(0, nearest_idx - window_radius)
        end_idx = min(len(df) - 1, nearest_idx + window_radius)

        df.loc[start_idx:end_idx, "ground_truth"] = 1

    return df


def load_real_temperature_dataset(dataset_name, path):
    if dataset_name not in DATASET_LABELS:
        raise ValueError(f"지원하지 않는 dataset_name: {dataset_name}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"value": "vt_shift"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["scenario_id"] = f"REAL_{dataset_name.upper()}"
    df["leakage_curr"] = df["vt_shift"].rolling(window=5, min_periods=1).mean()
    df["test_time_ms"] = 100.0
    df["test_error_code"] = 0
    df["ooc_flag"] = 0
    df["time_index"] = range(len(df))

    df = _build_ground_truth_windows(
        df,
        DATASET_LABELS[dataset_name],
        window_radius=12
    )

    return df