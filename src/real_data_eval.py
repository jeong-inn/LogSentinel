import pandas as pd


def preprocess_real_df(df):
    out = df.copy()

    out["sensor_1_roll_mean"] = out["sensor_1"].rolling(window=30, min_periods=1).mean()
    out["sensor_1_roll_std"] = out["sensor_1"].rolling(window=30, min_periods=1).std().fillna(0.0001)
    out["sensor_1_diff"] = out["sensor_1"] - out["sensor_1_roll_mean"]
    out["sensor_1_zscore"] = out["sensor_1_diff"] / out["sensor_1_roll_std"]

    out["sensor_1_ewm"] = out["sensor_1"].ewm(span=20, adjust=False).mean()
    out["sensor_1_ewm_diff"] = out["sensor_1"] - out["sensor_1_ewm"]

    return out


def detect_real_anomalies(df, z_thresh=3.0, ewm_thresh_scale=2.0, cooldown=10):
    """
    robust detector:
    - rolling z-score
    - ewm deviation
    - cooldown
    """
    out = df.copy()

    std_all = max(out["sensor_1"].std(), 0.0001)
    ewm_thresh = std_all * ewm_thresh_scale

    anomaly_flags = []
    anomaly_reasons = []

    cooldown_left = 0

    for _, row in out.iterrows():
        reasons = []

        is_z = abs(row["sensor_1_zscore"]) >= z_thresh
        is_ewm = abs(row["sensor_1_ewm_diff"]) >= ewm_thresh

        if cooldown_left > 0:
            cooldown_left -= 1
            anomaly_flags.append(0)
            anomaly_reasons.append("cooldown")
            continue

        if is_z:
            reasons.append("zscore_anomaly")
        if is_ewm:
            reasons.append("ewm_shift")

        if reasons:
            anomaly_flags.append(1)
            anomaly_reasons.append("|".join(reasons))
            cooldown_left = cooldown
        else:
            anomaly_flags.append(0)
            anomaly_reasons.append("normal")

    out["predicted_anomaly"] = anomaly_flags
    out["predicted_reason"] = anomaly_reasons
    return out


def score_real_detection(df, dataset_name):
    gt = df["ground_truth"]
    pred = df["predicted_anomaly"]

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return pd.DataFrame([{
        "dataset": dataset_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "predicted_positive_count": int(pred.sum()),
        "ground_truth_positive_count": int(gt.sum()),
    }])