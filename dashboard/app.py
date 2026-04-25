import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LogSentinel Dashboard", layout="wide")

st.title("LogSentinel Validation Dashboard")

judge_df = pd.read_csv("data/processed/scenario_judgement.csv")
validation_df = pd.read_csv("data/processed/validation_result.csv")
root_df = pd.read_csv("data/processed/root_cause_analysis.csv")
quality_df = pd.read_csv("data/processed/quality_summary.csv")
logs_df = pd.read_csv("data/processed/analyzed_logs_with_states.csv")

with open("data/processed/operator_report.json", "r", encoding="utf-8") as f:
    reports = json.load(f)

with open("data/processed/llm_reports.json", "r", encoding="utf-8") as f:
    llm_reports = json.load(f)

merged = judge_df.merge(validation_df, on="scenario_id", how="left")
merged = merged.merge(root_df, on="scenario_id", how="left")

st.subheader("Quality Summary")
st.dataframe(quality_df)

st.subheader("Scenario Summary")
st.dataframe(merged)

scenario_ids = merged["scenario_id"].tolist()
selected = st.selectbox("시나리오 선택", scenario_ids)

left, right = st.columns([1, 1])

row = merged[merged["scenario_id"] == selected].iloc[0]
report = next(r for r in reports if r["scenario_id"] == selected)
llm_report = next((r for r in llm_reports if r["scenario_id"] == selected), None)
scenario_log = logs_df[logs_df["scenario_id"] == selected].copy()

with left:
    st.subheader("Final Judgement")
    st.write(f"**Final Result:** {row['final_result']}")
    st.write(f"**Final Reason:** {row['final_reason']}")
    st.write(f"**Recommended Action:** {row['recommended_action_actual']}")
    st.write(f"**Validation Score:** {row['validation_score']}")
    st.write(f"**Release Gate:** {row['release_gate']}")
    st.write(f"**Overall Match:** {row['overall_match']}")
    st.write(f"**Mismatch Detail:** {row['mismatch_detail']}")

    st.subheader("Root Cause")
    st.write(f"**Primary Cause:** {row['primary_cause']}")
    st.write(f"**Secondary Signal:** {row['secondary_signal']}")
    st.write(f"**Confidence:** {row['confidence']}")
    st.write(f"**Evidence:** {row['evidence']}")

with right:
    st.subheader("Structured Operator Report")
    st.json(report)

    st.subheader("LLM Operator Report")
    if llm_report is not None:
        st.write(f"**Summary:** {llm_report['summary']}")
        st.write("**Judgement Basis:**")
        for item in llm_report["judgement_basis"]:
            st.write(f"- {item}")

        st.write(f"**Root Cause Hypothesis:** {llm_report['root_cause_hypothesis']}")
        st.write("**Recommended Actions:**")
        for item in llm_report["recommended_actions"]:
            st.write(f"- {item}")

        st.write(f"**Retest Needed:** {llm_report['retest_needed']}")
        st.write(f"**Engineering Review Needed:** {llm_report['engineering_review_needed']}")
        st.write(f"**Operator Note:** {llm_report['operator_note']}")
    else:
        st.write("이 시나리오에 대한 LLM 리포트는 아직 생성되지 않았습니다.")

st.subheader("Timeline")
st.line_chart(
    scenario_log.set_index("timestamp")[["sensor_1", "response_time"]]
)

st.subheader("State Distribution")
state_counts = scenario_log["state"].value_counts().reset_index()
state_counts.columns = ["state", "count"]
st.bar_chart(state_counts.set_index("state"))