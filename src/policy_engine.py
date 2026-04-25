import pandas as pd


ACTION_PRIORITY = {
    "NO_ACTION": 0,
    "MONITOR": 1,
    "RETEST_RECOMMENDED": 2,
    "RETEST_AND_ANALYZE": 3,
    "ENGINEERING_REVIEW": 4,
    "FAIL_LOCK": 5,
    "SHUTDOWN_REQUIRED": 6,
}


def recommend_action_policy(final_result, has_dropout, has_error, has_event, has_resp, has_sensor,
                            warning_ratio, critical_ratio, fail_ratio):
    if has_dropout:
        return "FAIL_LOCK"

    if final_result == "FAIL":
        if has_error and has_event:
            return "SHUTDOWN_REQUIRED"
        if fail_ratio >= 0.05 and has_resp:
            return "SHUTDOWN_REQUIRED"
        if critical_ratio >= 0.10:
            return "SHUTDOWN_REQUIRED"
        if has_sensor and has_resp:
            return "ENGINEERING_REVIEW"
        if has_resp:
            return "RETEST_AND_ANALYZE"
        if has_sensor:
            return "ENGINEERING_REVIEW"
        return "ENGINEERING_REVIEW"

    if final_result == "PASS_WITH_WARNING":
        if critical_ratio > 0:
            return "RETEST_RECOMMENDED"
        if warning_ratio >= 0.03:
            return "MONITOR"
        return "NO_ACTION"

    return "NO_ACTION"


def action_gap(expected_action, actual_action):
    e = ACTION_PRIORITY.get(expected_action, 0)
    a = ACTION_PRIORITY.get(actual_action, 0)
    return a - e


def decide_gate(final_result, overall_match, warning_ratio, critical_ratio, fail_ratio):
    if final_result == "FAIL":
        return "BLOCKED"

    if not overall_match:
        return "REVIEW_REQUIRED"

    if critical_ratio > 0 or fail_ratio > 0:
        return "REVIEW_REQUIRED"

    if warning_ratio >= 0.03:
        return "MONITORING_REQUIRED"

    return "READY"


def summarize_quality(validation_df):
    total = len(validation_df)
    match_count = int(validation_df["overall_match"].sum())
    mismatch_count = total - match_count
    match_rate = round(match_count / total, 3) if total > 0 else 0.0

    gate_counts = validation_df["release_gate"].value_counts().to_dict()

    return pd.DataFrame([{
        "scenario_total": total,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "match_rate": match_rate,
        "ready_count": gate_counts.get("READY", 0),
        "monitoring_required_count": gate_counts.get("MONITORING_REQUIRED", 0),
        "review_required_count": gate_counts.get("REVIEW_REQUIRED", 0),
        "blocked_count": gate_counts.get("BLOCKED", 0),
    }])