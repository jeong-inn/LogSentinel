import pandas as pd
from policy_engine import recommend_action_policy


def judge_scenarios(df):
    """
    시나리오별 전체 상태 흐름을 바탕으로 최종 판정 결정
    """

    results = []

    for scenario_id, group in df.groupby("scenario_id"):
        g = group.sort_values("timestamp").copy()
        total_count = len(g)

        state_counts = g["state"].value_counts().to_dict()

        fail_count = state_counts.get("FAIL", 0)
        critical_count = state_counts.get("CRITICAL", 0)
        warning_count = state_counts.get("WARNING", 0)

        fail_ratio = fail_count / total_count
        critical_ratio = critical_count / total_count
        warning_ratio = warning_count / total_count

        all_reasons = "|".join(g["anomaly_reason"].astype(str).tolist())

        has_dropout = "leakage_dropout" in all_reasons
        has_error = "test_error_detected" in all_reasons
        has_event = "ooc_event" in all_reasons
        has_resp = ("test_time_high" in all_reasons) or ("test_time_jump" in all_reasons)
        has_sensor = ("vt_high" in all_reasons) or ("vt_spike" in all_reasons)

        # 최종 결과 판정
        if fail_count > 0:
            final_result = "FAIL"

            if has_dropout:
                final_reason = "Leakage 신호 dropout 발생으로 최종 FAIL 판정"
            elif has_error and has_event:
                final_reason = "복합 이상(test_error/ooc_event) 발생으로 최종 FAIL 판정"
            elif has_resp:
                final_reason = "테스트 사이클 타임 누적 초과로 최종 FAIL 판정"
            elif has_sensor:
                final_reason = "Vt 고수준 이상 상태 지속으로 최종 FAIL 판정"
            else:
                final_reason = "치명 이상 상태 발생으로 최종 FAIL 판정"

        elif warning_ratio >= 0.40:
            final_result = "FAIL"

            if has_sensor and has_resp:
                final_reason = "Vt 이상과 사이클 타임 초과가 장시간 지속되어 최종 FAIL 판정"
            elif has_sensor:
                final_reason = "Vt 이상 상태가 장시간 지속되어 최종 FAIL 판정"
            elif has_resp:
                final_reason = "테스트 사이클 타임 초과 상태가 장시간 지속되어 최종 FAIL 판정"
            else:
                final_reason = "경고 상태가 장시간 지속되어 최종 FAIL 판정"

        elif critical_ratio >= 0.02:
            final_result = "FAIL"

            if has_error and has_event:
                final_reason = "복합 치명 이상이 반복 발생하여 최종 FAIL 판정"
            else:
                final_reason = "치명 이상 상태가 반복 발생하여 최종 FAIL 판정"

        elif (has_sensor and has_resp and has_error) or (has_sensor and has_event):
            final_result = "FAIL"
            final_reason = "복합 이상 조합이 확인되어 최종 FAIL 판정"

        elif critical_count > 0 or warning_count > 0:
            final_result = "PASS_WITH_WARNING"

            if critical_count > 0:
                final_reason = "치명 이상 징후가 있었으나 최종 회복되어 PASS_WITH_WARNING 판정"
            else:
                final_reason = "경고 수준 이상 징후가 있었으나 최종 회복되어 PASS_WITH_WARNING 판정"

        else:
            final_result = "PASS"
            final_reason = "이상 징후 없이 정상 범위를 유지하여 PASS 판정"

        recommended_action_actual = recommend_action_policy(
            final_result=final_result,
            has_dropout=has_dropout,
            has_error=has_error,
            has_event=has_event,
            has_resp=has_resp,
            has_sensor=has_sensor,
            warning_ratio=warning_ratio,
            critical_ratio=critical_ratio,
            fail_ratio=fail_ratio
        )

        results.append({
            "scenario_id": scenario_id,
            "total_count": total_count,
            "fail_count": fail_count,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "fail_ratio": round(fail_ratio, 3),
            "critical_ratio": round(critical_ratio, 3),
            "warning_ratio": round(warning_ratio, 3),
            "final_result": final_result,
            "final_reason": final_reason,
            "recommended_action_actual": recommended_action_actual,
            "all_reasons": all_reasons
        })

    return pd.DataFrame(results).sort_values("scenario_id").reset_index(drop=True)
