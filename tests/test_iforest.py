"""
test_iforest.py — Isolation Forest detector 단위 테스트
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from real_data_benchmark_v2 import detect_isolation_forest, detect_fixed_threshold, detect_zscore_only


def _make_test_df(n=300, anomaly_ratio=0.05, seed=42):
    """
    정상 구간 + anomaly 구간이 있는 테스트용 DataFrame 생성.
    vt_shift 컬럼 사용.
    """
    np.random.seed(seed)
    n_anom = int(n * anomaly_ratio)
    data = np.random.normal(50, 3, n)

    anom_idx = np.random.choice(n, n_anom, replace=False)
    data[anom_idx] += 20  # 강한 anomaly 주입

    gt = np.zeros(n, dtype=int)
    gt[anom_idx] = 1

    df = pd.DataFrame({
        "vt_shift": data,
        "leakage_curr": np.ones(n) * 30,
        "test_time_ms": np.ones(n) * 100,
        "ground_truth": gt,
        "timestamp": range(n),
    })
    return df


# ============================================================
# detect_isolation_forest
# ============================================================

def test_iforest_returns_required_columns():
    df = _make_test_df()
    result = detect_isolation_forest(df)
    assert "predicted_anomaly" in result.columns
    assert "predicted_reason" in result.columns


def test_iforest_binary_flags():
    df = _make_test_df()
    result = detect_isolation_forest(df)
    assert set(result["predicted_anomaly"].unique()).issubset({0, 1})


def test_iforest_detects_some_anomalies():
    """주입된 anomaly 중 일부는 탐지해야 한다."""
    df = _make_test_df(n=500, anomaly_ratio=0.1, seed=7)
    result = detect_isolation_forest(df, contamination=0.1)

    gt = df["ground_truth"].values
    pred = result["predicted_anomaly"].values

    tp = int(((gt == 1) & (pred == 1)).sum())
    assert tp > 0, "Isolation Forest should detect at least some injected anomalies"


def test_iforest_contamination_affects_count():
    """contamination이 높을수록 더 많은 anomaly를 예측해야 한다."""
    df = _make_test_df(n=500, seed=0)
    low = detect_isolation_forest(df, contamination=0.02)["predicted_anomaly"].sum()
    high = detect_isolation_forest(df, contamination=0.15)["predicted_anomaly"].sum()
    assert high > low, "higher contamination should flag more points"


# ============================================================
# detect_fixed_threshold
# ============================================================

def test_fixed_threshold_all_normal_low_signal():
    """정상 신호에서는 anomaly 비율이 낮아야 한다."""
    np.random.seed(42)
    vals = np.random.normal(50, 3, 200)  # mean=50, σ=3, threshold≈57.5
    df = pd.DataFrame({
        "vt_shift": vals,
        "leakage_curr": np.ones(200) * 30,
        "test_time_ms": np.ones(200) * 100,
        "ground_truth": np.zeros(200, dtype=int),
    })
    result = detect_fixed_threshold(df, threshold_scale=2.5)
    # 정상 데이터 중 극소수만 threshold 초과 (3σ 이상은 0.6%)
    anomaly_ratio = result["predicted_anomaly"].mean()
    assert anomaly_ratio < 0.05, f"too many anomalies on normal data: {anomaly_ratio:.2%}"


def test_fixed_threshold_detects_spike():
    """명확한 spike는 탐지해야 한다."""
    vals = [50.0] * 98 + [200.0, 200.0]
    df = pd.DataFrame({
        "vt_shift": vals,
        "leakage_curr": [30.0] * 100,
        "test_time_ms": [100.0] * 100,
        "ground_truth": [0] * 98 + [1, 1],
    })
    result = detect_fixed_threshold(df, threshold_scale=2.5)
    assert result["predicted_anomaly"].iloc[-1] == 1


# ============================================================
# detect_zscore_only
# ============================================================

def test_zscore_only_detects_outlier():
    """z > 3인 포인트는 탐지해야 한다."""
    data = [50.0] * 99 + [150.0]  # z >> 3
    df = pd.DataFrame({
        "vt_shift": data,
        "leakage_curr": [30.0] * 100,
        "test_time_ms": [100.0] * 100,
        "ground_truth": [0] * 100,
    })
    result = detect_zscore_only(df, z_thresh=3.0)
    assert result["predicted_anomaly"].iloc[-1] == 1
