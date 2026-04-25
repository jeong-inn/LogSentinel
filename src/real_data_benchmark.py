import pandas as pd
from real_data_eval import preprocess_real_df, detect_real_anomalies


def detect_baseline_fixed_threshold(df):
    """
    Baseline 1: 고정 threshold 기반 탐지
    """
    out = preprocess_real_df(df.copy())

    threshold = out["sensor_1"].mean() + 2.0 * out["sensor_1"].std()

    pred = []
    reason = []

    for _, row in out.iterrows():
        if row["sensor_1"] >= threshold:
            pred.append(1)
            reason.append("fixed_threshold")
        else:
            pred.append(0)
            reason.append("normal")

    out["predicted_anomaly"] = pred
    out["predicted_reason"] = reason
    return out


def detect_baseline_zscore_only(df, z_thresh=3.0):
    """
    Baseline 2: rolling z-score 기반 탐지
    """
    out = preprocess_real_df(df.copy())

    pred = []
    reason = []

    for _, row in out.iterrows():
        if abs(row["sensor_1_zscore"]) >= z_thresh:
            pred.append(1)
            reason.append("zscore_anomaly")
        else:
            pred.append(0)
            reason.append("normal")

    out["predicted_anomaly"] = pred
    out["predicted_reason"] = reason
    return out


def detect_ablation_zscore_ewm(df, z_thresh=3.0, ewm_thresh_scale=1.0):
    """
    Ablation: z-score + ewm, cooldown 없음
    """
    out = preprocess_real_df(df.copy())

    std_all = max(out["sensor_1"].std(), 0.0001)
    ewm_thresh = std_all * ewm_thresh_scale

    pred = []
    reason = []

    for _, row in out.iterrows():
        reasons = []

        if abs(row["sensor_1_zscore"]) >= z_thresh:
            reasons.append("zscore_anomaly")
        if abs(row["sensor_1_ewm_diff"]) >= ewm_thresh:
            reasons.append("ewm_shift")

        if reasons:
            pred.append(1)
            reason.append("|".join(reasons))
        else:
            pred.append(0)
            reason.append("normal")

    out["predicted_anomaly"] = pred
    out["predicted_reason"] = reason
    return out


def _event_hit_metrics(df):
    """
    ground_truth anomaly window 단위 hit rate 계산
    ground_truth==1 인 연속 구간을 하나의 event로 본다.
    """
    gt = df["ground_truth"].tolist()
    pred = df["predicted_anomaly"].tolist()

    events = []
    in_event = False
    start = None

    for i, val in enumerate(gt):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
            start = None

    if in_event:
        events.append((start, len(gt) - 1))

    total_events = len(events)
    hit_events = 0

    for s, e in events:
        if any(pred[s:e + 1]):
            hit_events += 1

    event_hit_rate = hit_events / total_events if total_events > 0 else 0.0

    return total_events, hit_events, round(event_hit_rate, 3)


def score_detection(df, dataset_name, method_name):
    gt = df["ground_truth"]
    pred = df["predicted_anomaly"]

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    total_events, hit_events, event_hit_rate = _event_hit_metrics(df)

    return pd.DataFrame([{
        "dataset": dataset_name,
        "method": method_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "predicted_positive_count": int(pred.sum()),
        "ground_truth_positive_count": int(gt.sum()),
        "total_events": total_events,
        "hit_events": hit_events,
        "event_hit_rate": event_hit_rate,
    }])


def run_method(df, dataset_name, method_name, **kwargs):
    if method_name == "fixed_threshold":
        scored = detect_baseline_fixed_threshold(df)
    elif method_name == "zscore_only":
        scored = detect_baseline_zscore_only(df, z_thresh=kwargs.get("z_thresh", 3.0))
    elif method_name == "zscore_ewm":
        scored = detect_ablation_zscore_ewm(
            df,
            z_thresh=kwargs.get("z_thresh", 3.0),
            ewm_thresh_scale=kwargs.get("ewm_thresh_scale", 1.0)
        )
    elif method_name == "full_pipeline":
        scored = detect_real_anomalies(
            preprocess_real_df(df.copy()),
            z_thresh=kwargs.get("z_thresh", 3.0),
            ewm_thresh_scale=kwargs.get("ewm_thresh_scale", 1.0),
            cooldown=kwargs.get("cooldown", 0)
        )
    else:
        raise ValueError(f"지원하지 않는 method_name: {method_name}")

    metric_df = score_detection(scored, dataset_name, method_name)
    return scored, metric_df


def build_benchmark_table(raw_df, dataset_name, tuned_params):
    """
    baseline + ablation + proposed 비교표 생성
    """
    rows = []

    methods = [
        ("fixed_threshold", {}),
        ("zscore_only", {"z_thresh": tuned_params["z_thresh"]}),
        ("zscore_ewm", {
            "z_thresh": tuned_params["z_thresh"],
            "ewm_thresh_scale": tuned_params["ewm_thresh_scale"]
        }),
        ("full_pipeline", tuned_params),
    ]

    for method_name, params in methods:
        _, metric_df = run_method(raw_df, dataset_name, method_name, **params)
        rows.append(metric_df)

    return pd.concat(rows, ignore_index=True)