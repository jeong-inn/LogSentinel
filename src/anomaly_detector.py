import pandas as pd


def detect_anomalies(df):
    """
    규칙 기반 이상 탐지 (Sort Test 파라미터 기준)
    결과:
    - anomaly_flag: 0 or 1
    - anomaly_reason: 왜 이상인지
    """
    df = df.copy()

    anomaly_flags = []
    anomaly_reasons = []

    for _, row in df.iterrows():
        reasons = []

        # 1) vt_shift 절대 임계값 초과 (Vt 고수준 이상)
        if row["vt_shift"] >= 65:
            reasons.append("vt_high")

        # 2) vt_shift 급격 변동 (Vt 스파이크)
        if abs(row["vt_shift_diff"]) >= 8:
            reasons.append("vt_spike")

        # 3) test_time_ms 높음 (ATE 사이클 타임 초과)
        if row["test_time_ms"] >= 135:
            reasons.append("test_time_high")

        # 4) test_time_ms rolling 대비 급증 (사이클 타임 점프)
        if row["test_time_diff"] >= 15:
            reasons.append("test_time_jump")

        # 5) test_error_code 발생 (장비 에러)
        if row["test_error_code"] != 0:
            reasons.append("test_error_detected")

        # 6) leakage_curr dropout (프로브 이상 / 신호 손실)
        if row["leakage_curr"] <= 1:
            reasons.append("leakage_dropout")

        # 7) ooc_flag 발생 (Out-of-Control 이벤트)
        if row["ooc_flag"] == 1:
            reasons.append("ooc_event")

        if reasons:
            anomaly_flags.append(1)
            anomaly_reasons.append("|".join(reasons))
        else:
            anomaly_flags.append(0)
            anomaly_reasons.append("normal")

    df["anomaly_flag"] = anomaly_flags
    df["anomaly_reason"] = anomaly_reasons
    return df


def summarize_anomalies(df):
    """
    시나리오별 이상 개수 요약
    """
    summary = (
        df.groupby("scenario_id")["anomaly_flag"]
        .sum()
        .reset_index()
        .rename(columns={"anomaly_flag": "anomaly_count"})
    )
    return summary
