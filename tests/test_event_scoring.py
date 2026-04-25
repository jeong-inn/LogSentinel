"""
test_event_scoring.py — event-level scoring 단위 테스트
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from real_data_benchmark_v2 import consolidate_events, score_detection


# ============================================================
# consolidate_events
# ============================================================

def _make_df(gt, pred):
    return pd.DataFrame({
        "ground_truth": gt,
        "predicted_anomaly": pred,
        "vt_shift": np.ones(len(gt)) * 50,
    })


def test_consolidate_single_event():
    flags = [0, 0, 1, 1, 1, 0, 0]
    events = consolidate_events(flags, merge_gap=5)
    assert len(events) == 1
    assert events[0] == (2, 4)


def test_consolidate_merge_close_events():
    flags = [1, 1, 0, 0, 1, 1]
    events = consolidate_events(flags, merge_gap=5)
    assert len(events) == 1  # gap=2 <= merge_gap=5


def test_consolidate_no_merge_far_events():
    flags = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    events = consolidate_events(flags, merge_gap=3)
    assert len(events) == 2  # gap=8 > merge_gap=3


def test_consolidate_empty():
    assert consolidate_events([0, 0, 0], merge_gap=5) == []


def test_consolidate_all_ones():
    events = consolidate_events([1, 1, 1, 1], merge_gap=5)
    assert len(events) == 1
    assert events[0] == (0, 3)


# ============================================================
# score_detection
# ============================================================

def test_score_perfect_recall():
    """GT event와 완전히 겹치는 예측 → recall=1.0."""
    gt =   [0, 0, 1, 1, 1, 0, 0, 0, 0]
    pred = [0, 0, 1, 1, 1, 0, 0, 0, 0]
    df = _make_df(gt, pred)
    result = score_detection(df, "test", "method")
    assert result.iloc[0]["event_recall"] == 1.0


def test_score_zero_recall():
    """GT event를 하나도 못 잡으면 recall=0."""
    gt =   [0, 0, 1, 1, 0, 0, 0, 0, 0]
    pred = [0, 0, 0, 0, 0, 1, 1, 0, 0]  # 겹치지 않음
    df = _make_df(gt, pred)
    result = score_detection(df, "test", "method")
    assert result.iloc[0]["event_recall"] == 0.0


def test_score_false_alarm_count():
    """GT event 없이 예측만 있으면 FA = 예측 event 수 (merge_gap=1로 분리 보장)."""
    gt =   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  # gap=9 > merge_gap=3
    df = _make_df(gt, pred)
    result = score_detection(df, "test", "method", merge_gap=3)
    assert result.iloc[0]["event_recall"] == 0.0
    assert result.iloc[0]["false_alarm_events"] == 2


def test_score_output_columns():
    gt =   [0, 1, 1, 0]
    pred = [0, 1, 1, 0]
    df = _make_df(gt, pred)
    result = score_detection(df, "test_ds", "test_method")
    required = ["dataset", "method", "event_recall", "event_precision",
                "event_f1", "gt_events", "pred_events", "hit_events", "false_alarm_events"]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_score_f1_perfect():
    """완벽한 탐지 → F1=1.0."""
    gt =   [0, 1, 1, 1, 0]
    pred = [0, 1, 1, 1, 0]
    df = _make_df(gt, pred)
    result = score_detection(df, "ds", "m")
    assert result.iloc[0]["event_f1"] == 1.0
