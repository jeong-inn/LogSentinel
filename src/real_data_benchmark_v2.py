import pandas as pd
import numpy as np


# ============================================================
# Feature engineering
# ============================================================

def add_basic_features(df):
    out = df.copy()

    out["roll_mean_30"] = out["vt_shift"].rolling(window=30, min_periods=1).mean()
    out["roll_std_30"] = out["vt_shift"].rolling(window=30, min_periods=1).std().fillna(0.0001)
    out["diff_30"] = out["vt_shift"] - out["roll_mean_30"]
    out["zscore_30"] = out["diff_30"] / out["roll_std_30"]

    out["ewm_20"] = out["vt_shift"].ewm(span=20, adjust=False).mean()
    out["ewm_diff"] = out["vt_shift"] - out["ewm_20"]

    return out


def add_detrended_features(df, seasonal_window=288):
    out = df.copy()

    out["baseline_long"] = out["vt_shift"].rolling(
        window=seasonal_window,
        min_periods=1,
        center=True
    ).median()

    out["residual"] = out["vt_shift"] - out["baseline_long"]
    out["residual_roll_mean"] = out["residual"].rolling(window=30, min_periods=1).mean()
    out["residual_roll_std"] = out["residual"].rolling(window=30, min_periods=1).std().fillna(0.0001)
    out["residual_zscore"] = (
        (out["residual"] - out["residual_roll_mean"]) / out["residual_roll_std"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    out["residual_ewm"] = out["residual"].ewm(span=20, adjust=False).mean()
    out["residual_ewm_diff"] = out["residual"] - out["residual_ewm"]

    return out


# ============================================================
# Post-processing: persistence + expand
# ============================================================

def _apply_persistence(flags, persistence_min=3):
    flags = np.array(flags, dtype=int)
    persistent = np.zeros(len(flags), dtype=int)

    streak = 0
    for i, flag in enumerate(flags):
        if flag == 1:
            streak += 1
        else:
            streak = 0

        if streak >= persistence_min:
            persistent[i] = 1

    for i in range(len(persistent)):
        if persistent[i] == 1:
            start = max(0, i - persistence_min + 1)
            persistent[start:i + 1] = 1

    return persistent


def _expand_events(flags, expand_window=6):
    flags = np.array(flags, dtype=int)
    idxs = np.where(flags == 1)[0]

    for idx in idxs:
        start = max(0, idx - expand_window)
        end = min(len(flags) - 1, idx + expand_window)
        flags[start:end + 1] = 1

    return flags


# ============================================================
# Event consolidation
# ============================================================

def consolidate_events(flags, merge_gap=10):
    """
    연속된 anomaly point를 하나의 event로 묶는다.
    gap이 merge_gap 이하이면 같은 event로 합친다.

    Returns:
        list of (start_idx, end_idx) tuples
    """
    flags = np.array(flags, dtype=int)
    events = []
    in_event = False
    start = 0

    for i, val in enumerate(flags):
        if val == 1 and not in_event:
            start = i
            in_event = True
        elif val == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(flags) - 1))

    # merge close events
    if len(events) <= 1:
        return events

    merged = [events[0]]
    for s, e in events[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= merge_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    return merged


# ============================================================
# Detectors
# ============================================================

def detect_fixed_threshold(df, threshold_scale=2.5):
    out = df.copy()
    threshold = out["vt_shift"].mean() + threshold_scale * out["vt_shift"].std()
    out["predicted_anomaly"] = (out["vt_shift"] >= threshold).astype(int)
    out["predicted_reason"] = np.where(out["predicted_anomaly"] == 1, "fixed_threshold", "normal")
    return out


def detect_isolation_forest(df, contamination=0.05):
    """
    Baseline: Isolation Forest (unsupervised ML)
    sensor_1의 rolling feature를 입력으로 사용
    """
    from sklearn.ensemble import IsolationForest

    out = add_basic_features(df)

    features = out[["vt_shift", "zscore_30", "ewm_diff"]].fillna(0).values

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
    )
    preds = model.fit_predict(features)

    # IsolationForest: -1 = anomaly, 1 = normal
    out["predicted_anomaly"] = (preds == -1).astype(int)
    out["predicted_reason"] = np.where(out["predicted_anomaly"] == 1, "isolation_forest", "normal")
    return out


def detect_zscore_only(df, z_thresh=3.0):
    out = add_basic_features(df)
    out["predicted_anomaly"] = (out["zscore_30"].abs() >= z_thresh).astype(int)
    out["predicted_reason"] = np.where(out["predicted_anomaly"] == 1, "zscore_only", "normal")
    return out


def detect_v1_pipeline(df, z_thresh=3.0, ewm_thresh_scale=0.8, cooldown=10):
    out = add_basic_features(df)

    std_all = max(out["vt_shift"].std(), 0.0001)
    ewm_thresh = std_all * ewm_thresh_scale

    flags = []
    reasons = []
    cooldown_left = 0

    for _, row in out.iterrows():
        if cooldown_left > 0:
            cooldown_left -= 1
            flags.append(0)
            reasons.append("cooldown")
            continue

        is_z = abs(row["zscore_30"]) >= z_thresh
        is_ewm = abs(row["ewm_diff"]) >= ewm_thresh

        reason_parts = []
        if is_z:
            reason_parts.append("zscore")
        if is_ewm:
            reason_parts.append("ewm")

        if reason_parts:
            flags.append(1)
            reasons.append("|".join(reason_parts))
            cooldown_left = cooldown
        else:
            flags.append(0)
            reasons.append("normal")

    out["predicted_anomaly"] = flags
    out["predicted_reason"] = reasons
    return out


def detect_v2_pipeline(df, z_thresh=2.5, ewm_thresh_scale=1.2, persistence_min=3, expand_window=6):
    out = add_detrended_features(df)

    std_res = max(out["residual"].std(), 0.0001)
    ewm_thresh = std_res * ewm_thresh_scale

    raw_flags = []
    raw_reasons = []

    for _, row in out.iterrows():
        reason_parts = []

        if abs(row["residual_zscore"]) >= z_thresh:
            reason_parts.append("residual_zscore")
        if abs(row["residual_ewm_diff"]) >= ewm_thresh:
            reason_parts.append("residual_ewm")

        if reason_parts:
            raw_flags.append(1)
            raw_reasons.append("|".join(reason_parts))
        else:
            raw_flags.append(0)
            raw_reasons.append("normal")

    persistent_flags = _apply_persistence(raw_flags, persistence_min=persistence_min)
    final_flags = _expand_events(persistent_flags, expand_window=expand_window)

    out["predicted_anomaly"] = final_flags
    out["predicted_reason"] = [
        raw_reasons[i] if final_flags[i] == 1 else "normal"
        for i in range(len(out))
    ]
    return out


# ============================================================
# Scoring — event-level 중심
# ============================================================

def _extract_segments(arr):
    """연속된 1 구간을 (start, end) 리스트로 추출"""
    segments = []
    in_seg = False
    start = 0

    for i, val in enumerate(arr):
        if val == 1 and not in_seg:
            start = i
            in_seg = True
        elif val == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(arr) - 1))

    return segments


def _segments_overlap(seg_a, seg_b):
    """두 segment가 겹치는지 확인"""
    return seg_a[0] <= seg_b[1] and seg_b[0] <= seg_a[1]


def score_detection(df, dataset_name, method_name, merge_gap=10):
    """
    Point-level + Event-level metric 모두 산출.
    Event-level이 메인 지표.
    """
    gt = df["ground_truth"].values
    pred = df["predicted_anomaly"].values

    # --- point-level (부록용) ---
    tp_pt = int(((gt == 1) & (pred == 1)).sum())
    fp_pt = int(((gt == 0) & (pred == 1)).sum())
    fn_pt = int(((gt == 1) & (pred == 0)).sum())

    pt_precision = tp_pt / (tp_pt + fp_pt) if (tp_pt + fp_pt) > 0 else 0.0
    pt_recall = tp_pt / (tp_pt + fn_pt) if (tp_pt + fn_pt) > 0 else 0.0
    pt_f1 = (2 * pt_precision * pt_recall / (pt_precision + pt_recall)) if (pt_precision + pt_recall) > 0 else 0.0

    # --- event-level (메인 지표) ---
    gt_events = _extract_segments(gt)
    pred_events = consolidate_events(pred, merge_gap=merge_gap)

    # GT event 중 predicted event와 겹치는 것 = hit
    hit_events = 0
    for gt_seg in gt_events:
        for pred_seg in pred_events:
            if _segments_overlap(gt_seg, pred_seg):
                hit_events += 1
                break

    # Predicted event 중 GT event와 하나도 안 겹치는 것 = false alarm
    false_alarm_events = 0
    for pred_seg in pred_events:
        matched = False
        for gt_seg in gt_events:
            if _segments_overlap(pred_seg, gt_seg):
                matched = True
                break
        if not matched:
            false_alarm_events += 1

    total_gt_events = len(gt_events)
    total_pred_events = len(pred_events)

    event_recall = hit_events / total_gt_events if total_gt_events > 0 else 0.0
    event_precision = hit_events / total_pred_events if total_pred_events > 0 else 0.0
    event_f1 = (
        (2 * event_precision * event_recall / (event_precision + event_recall))
        if (event_precision + event_recall) > 0 else 0.0
    )

    return pd.DataFrame([{
        "dataset": dataset_name,
        "method": method_name,

        # event-level (메인)
        "gt_events": total_gt_events,
        "pred_events": total_pred_events,
        "hit_events": hit_events,
        "false_alarm_events": false_alarm_events,
        "event_precision": round(event_precision, 3),
        "event_recall": round(event_recall, 3),
        "event_f1": round(event_f1, 3),

        # point-level (부록)
        "pt_tp": tp_pt,
        "pt_fp": fp_pt,
        "pt_fn": fn_pt,
        "pt_precision": round(pt_precision, 3),
        "pt_recall": round(pt_recall, 3),
        "pt_f1": round(pt_f1, 3),
        "predicted_positive_count": int(pred.sum()),
        "ground_truth_positive_count": int(gt.sum()),
    }])


# ============================================================
# Run all methods
# ============================================================

def run_all_methods(df, dataset_name):
    results = []
    scored = {}

    methods = [
        ("fixed_threshold", detect_fixed_threshold(df)),
        ("zscore_only", detect_zscore_only(df)),
        ("v1_pipeline", detect_v1_pipeline(df)),
        ("v2_pipeline", detect_v2_pipeline(df)),
    ]

    for method_name, scored_df in methods:
        scored[method_name] = scored_df
        results.append(score_detection(scored_df, dataset_name, method_name))

    summary_df = pd.concat(results, ignore_index=True)
    return summary_df, scored