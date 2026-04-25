import pandas as pd


def assign_states(df):
    """
    anomaly 결과를 바탕으로 각 row의 상태를 분류한다.
    상태:
    - NORMAL
    - WARNING
    - CRITICAL
    - RECOVERY
    - FAIL
    """
    result = []

    for scenario_id, group in df.groupby("scenario_id"):
        g = group.sort_values("timestamp").copy()
        states = []

        prev_state = "NORMAL"
        critical_streak = 0

        for _, row in g.iterrows():
            reason = row["anomaly_reason"]

            is_dropout = "leakage_dropout" in reason
            is_error = "test_error_detected" in reason
            is_event = "ooc_event" in reason
            is_resp_high = "test_time_high" in reason or "test_time_jump" in reason
            is_sensor_high = "vt_high" in reason or "vt_spike" in reason

            if row["anomaly_flag"] == 0:
                if prev_state in ["WARNING", "CRITICAL", "FAIL", "RECOVERY"]:
                    state = "RECOVERY"
                else:
                    state = "NORMAL"
                critical_streak = 0

            else:
                # 강한 이상 조건
                if is_dropout:
                    state = "FAIL"
                    critical_streak += 1

                elif is_error and (is_resp_high or is_sensor_high or is_event):
                    state = "CRITICAL"
                    critical_streak += 1

                elif is_resp_high and is_sensor_high:
                    state = "CRITICAL"
                    critical_streak += 1

                elif is_error:
                    state = "WARNING"
                    critical_streak = 0

                elif is_resp_high or is_sensor_high or is_event:
                    state = "WARNING"
                    critical_streak = 0

                else:
                    state = "WARNING"
                    critical_streak = 0

            # CRITICAL이 오래 지속되면 FAIL 승격
            if critical_streak >= 30:
                state = "FAIL"

            states.append(state)
            prev_state = state

        g["state"] = states
        result.append(g)

    out = pd.concat(result, ignore_index=True)
    return out


def summarize_states(df):
    """
    시나리오별 상태 개수 요약
    """
    summary = (
        df.groupby(["scenario_id", "state"])
        .size()
        .reset_index(name="count")
        .sort_values(["scenario_id", "state"])
    )
    return summary


def final_state_per_scenario(df):
    """
    시나리오별 마지막 상태 추출
    """
    final_df = (
        df.sort_values(["scenario_id", "timestamp"])
        .groupby("scenario_id")
        .tail(1)[["scenario_id", "state"]]
        .rename(columns={"state": "final_state"})
        .reset_index(drop=True)
    )
    return final_df
