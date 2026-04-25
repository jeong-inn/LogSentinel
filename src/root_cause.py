import pandas as pd


def analyze_root_causes(df, judge_df):
    """
    scenario별 root cause 구조화
    출력 컬럼:
    - primary_cause
    - secondary_signal
    - confidence
    - evidence
    """
    rows = []

    for _, jrow in judge_df.iterrows():
        scenario_id = jrow["scenario_id"]
        g = df[df["scenario_id"] == scenario_id].copy()

        reason_text = str(jrow["all_reasons"])

        has_dropout = "leakage_dropout" in reason_text
        has_error = "test_error_detected" in reason_text
        has_event = "ooc_event" in reason_text
        has_resp_jump = "test_time_jump" in reason_text
        has_resp_high = "test_time_high" in reason_text
        has_sensor_high = "vt_high" in reason_text
        has_sensor_spike = "vt_spike" in reason_text

        if has_dropout:
            primary_cause = "Leakage signal dropout (probe contact failure)"
            secondary_signal = "leakage_curr unavailable"
            confidence = "High"
        elif has_sensor_high and (has_resp_jump or has_resp_high):
            primary_cause = "Persistent Vt anomaly with test cycle time degradation"
            secondary_signal = "test_time_ms increase"
            confidence = "High"
        elif has_error and has_event:
            primary_cause = "Combined fault with OOC event-triggered instability"
            secondary_signal = "repeated test_error + ooc_flag"
            confidence = "High"
        elif has_resp_jump or has_resp_high:
            primary_cause = "ATE test cycle time degradation"
            secondary_signal = "test_time_ms abnormal trend"
            confidence = "Medium"
        elif has_sensor_high:
            primary_cause = "Persistent Vt shift above spec limit"
            secondary_signal = "vt_shift above threshold"
            confidence = "Medium"
        elif has_sensor_spike:
            primary_cause = "Transient Vt spike (electrical noise or process excursion)"
            secondary_signal = "short Vt abnormal excursion"
            confidence = "Low"
        elif has_error:
            primary_cause = "Repeated test error code pattern (contact or handler issue)"
            secondary_signal = "test_error_code recurrence"
            confidence = "Medium"
        else:
            primary_cause = "No significant abnormality"
            secondary_signal = "none"
            confidence = "Low"

        evidence_parts = []

        if has_sensor_high:
            evidence_parts.append("vt_high_detected")
        if has_sensor_spike:
            evidence_parts.append("vt_spike_detected")
        if has_resp_jump or has_resp_high:
            evidence_parts.append("test_time_abnormal")
        if has_error:
            evidence_parts.append("test_error_detected")
        if has_event:
            evidence_parts.append("ooc_event_detected")
        if has_dropout:
            evidence_parts.append("leakage_dropout_detected")

        evidence_parts.append(f"warning_ratio={jrow['warning_ratio']}")
        evidence_parts.append(f"critical_ratio={jrow['critical_ratio']}")
        evidence_parts.append(f"fail_ratio={jrow['fail_ratio']}")

        rows.append({
            "scenario_id": scenario_id,
            "primary_cause": primary_cause,
            "secondary_signal": secondary_signal,
            "confidence": confidence,
            "evidence": " | ".join(evidence_parts)
        })

    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
