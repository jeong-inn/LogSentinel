import os
import pandas as pd

from preprocess import preprocess_logs
from anomaly_detector import detect_anomalies, summarize_anomalies
from state_engine import assign_states, summarize_states, final_state_per_scenario
from judge import judge_scenarios
from validator import load_scenario_specs, validate_against_specs
from root_cause import analyze_root_causes
from operator_report import build_operator_report, save_llm_prompts
from llm_reporter import generate_llm_reports
from policy_engine import summarize_quality
from spc import analyze_spc_by_scenario
from real_data_loader import load_real_temperature_dataset, DATASET_LABELS
from real_data_benchmark_v2 import run_all_methods
from real_data_cross_eval import run_multi_dataset_benchmark


# Sort Test 파라미터 규격 한계 (USL/LSL)
VT_SHIFT_USL   = 70.0;   VT_SHIFT_LSL   = 30.0   # vt_shift: 정상 30~70 mV
LEAKAGE_USL    = 40.0;   LEAKAGE_LSL    = 20.0   # leakage_curr: 정상 20~40 nA
TEST_TIME_USL  = 120.0;  TEST_TIME_LSL  = 80.0   # test_time_ms: 정상 80~120 ms


def main():
    os.makedirs("data/processed", exist_ok=True)

    # =========================================================
    # 1. Synthetic scenario pipeline
    # =========================================================
    df = preprocess_logs("data/raw/test_logs.csv")
    df = detect_anomalies(df)
    df = assign_states(df)

    row_save_path = "data/processed/analyzed_logs_with_states.csv"
    df.to_csv(row_save_path, index=False)

    anomaly_summary = summarize_anomalies(df)
    state_summary = summarize_states(df)
    final_states = final_state_per_scenario(df)

    # 2. Judgement
    judge_df = judge_scenarios(df)
    judge_save_path = "data/processed/scenario_judgement.csv"
    judge_df.to_csv(judge_save_path, index=False)

    # 3. Validation against scenario specs
    spec_df = load_scenario_specs("data/scenarios/scenario_specs.json")
    validation_df = validate_against_specs(judge_df, spec_df)
    validation_save_path = "data/processed/validation_result.csv"
    validation_df.to_csv(validation_save_path, index=False)

    # 4. Quality summary
    quality_df = summarize_quality(validation_df)
    quality_save_path = "data/processed/quality_summary.csv"
    quality_df.to_csv(quality_save_path, index=False)

    # 5. Root cause analysis
    root_cause_df = analyze_root_causes(df, judge_df)
    root_save_path = "data/processed/root_cause_analysis.csv"
    root_cause_df.to_csv(root_save_path, index=False)

    # 6. SPC analysis — 3개 파라미터 공정 능력 + WE Rules
    spc_vt  = analyze_spc_by_scenario(df, col="vt_shift",     usl=VT_SHIFT_USL,  lsl=VT_SHIFT_LSL)
    spc_lkg = analyze_spc_by_scenario(df, col="leakage_curr", usl=LEAKAGE_USL,   lsl=LEAKAGE_LSL)
    spc_tt  = analyze_spc_by_scenario(df, col="test_time_ms", usl=TEST_TIME_USL, lsl=TEST_TIME_LSL)

    # 파라미터 이름 컬럼 추가 후 합치기
    for _df, _param in [(spc_vt, "vt_shift"), (spc_lkg, "leakage_curr"), (spc_tt, "test_time_ms")]:
        _df.insert(1, "param", _param)

    spc_df = pd.concat([spc_vt, spc_lkg, spc_tt], ignore_index=True)
    spc_save_path = "data/processed/spc_analysis.csv"
    spc_df.to_csv(spc_save_path, index=False)

    # 7. Structured operator report
    records = build_operator_report(
        judge_df=judge_df,
        validation_df=validation_df,
        root_cause_df=root_cause_df,
        output_path="data/processed/operator_report.json"
    )

    # 8. LLM prompt and LLM report
    prompt_path = save_llm_prompts(
        records,
        output_path="data/processed/llm_prompts.txt"
    )

    llm_reports = generate_llm_reports(
        records,
        output_path="data/processed/llm_reports.json",
        model="gpt-5",
        max_reports=1
    )

    # =========================================================
    # 9. Real-data multi-dataset benchmark
    # =========================================================
    EXTERNAL_DIR = "data/external"
    datasets = {}

    for ds_name in DATASET_LABELS.keys():
        csv_path = os.path.join(EXTERNAL_DIR, f"{ds_name}.csv")
        if os.path.exists(csv_path):
            datasets[ds_name] = load_real_temperature_dataset(ds_name, csv_path)
            print(f"  [load] {ds_name}: {len(datasets[ds_name])} rows, "
                  f"{len(DATASET_LABELS[ds_name])} labels", flush=True)
        else:
            print(f"  [skip] {ds_name}: {csv_path} not found", flush=True)

    results = run_multi_dataset_benchmark(datasets)

    # 파일 저장
    results["summary_df"].to_csv("data/processed/all_dataset_benchmark.csv", index=False)
    results["cross_eval_df"].to_csv("data/processed/cross_dataset_eval.csv", index=False)

    for ds_name, ds_result in results["per_dataset"].items():
        short = ds_name.replace("_system_failure", "").replace("_misconfiguration", "")
        ds_result["tuning_df"].to_csv(f"data/processed/{short}_tuning.csv", index=False)
        ds_result["benchmark_df"].to_csv(f"data/processed/{short}_benchmark.csv", index=False)

    # =========================================================
    # 10. Print summaries
    # =========================================================

    print("=" * 70)
    print("  SYNTHETIC SCENARIO RESULTS")
    print("=" * 70)

    print("\n[최종 판정]")
    print(judge_df[[
        "scenario_id", "final_result", "recommended_action_actual",
        "warning_ratio", "critical_ratio", "fail_ratio"
    ]].to_string(index=False))

    print("\n[검증 결과]")
    print(validation_df[[
        "scenario_id", "expected_final_result", "actual_final_result",
        "expected_action", "actual_action", "action_gap",
        "validation_score", "overall_match", "release_gate"
    ]].to_string(index=False))

    print("\n[품질 요약]")
    print(quality_df.to_string(index=False))

    print("\n[원인 분석]")
    print(root_cause_df.to_string(index=False))

    print("\n[SPC 분석 — 3개 파라미터 공정 능력]")
    spc_display_cols = ["scenario_id", "param", "cpk", "ppk", "rule1_count", "ooc_count", "ucl", "lcl"]
    print(spc_df[spc_display_cols].to_string(index=False))

    # --- Real-data 출력 ---
    event_cols = [
        "dataset", "method",
        "gt_events", "pred_events", "hit_events", "false_alarm_events",
        "event_precision", "event_recall", "event_f1",
    ]

    print("\n" + "=" * 70)
    print("  REAL-DATA EVALUATION (EVENT-LEVEL)")
    print("=" * 70)

    # dataset별 결과
    for ds_name, ds_result in results["per_dataset"].items():
        short = ds_name.replace("_system_failure", "").replace("_misconfiguration", "")
        print(f"\n[{short} — baseline + v2 self-tuned]")
        cols = [c for c in event_cols if c in ds_result["benchmark_df"].columns]
        print(ds_result["benchmark_df"][cols].to_string(index=False))

        best = ds_result["best"]
        print(f"  best: z={best['z_thresh']}, ewm={best['ewm_thresh_scale']}, "
              f"persist={int(best['persistence_min'])}, expand={int(best['expand_window'])} "
              f"-> recall={best['event_recall']}, FA={int(best['false_alarm_events'])}")

    # 전체 요약: v2_self_tuned만 모아서 비교
    print("\n" + "-" * 50)
    print("[전체 데이터셋 v2_self_tuned 요약]")
    summary = results["summary_df"]
    v2_summary = summary[summary["method"] == "v2_self_tuned"].copy()
    if not v2_summary.empty:
        cols = [c for c in event_cols if c in v2_summary.columns]
        print(v2_summary[cols].to_string(index=False))

    # Cross-dataset 평가
    if not results["cross_eval_df"].empty:
        print("\n[Cross-dataset evaluation (event-level)]")
        cross_cols = ["dataset", "method", "tuned_on",
                      "gt_events", "hit_events", "false_alarm_events",
                      "event_precision", "event_recall", "event_f1"]
        cross_cols = [c for c in cross_cols if c in results["cross_eval_df"].columns]
        print(results["cross_eval_df"][cross_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("  파일 저장 완료")
    print("=" * 70)
    print("  분석 로그: data/processed/analyzed_logs_with_states.csv")
    print("  판정: data/processed/scenario_judgement.csv")
    print("  검증: data/processed/validation_result.csv")
    print("  품질 요약: data/processed/quality_summary.csv")
    print("  원인 분석: data/processed/root_cause_analysis.csv")
    print("  SPC 분석 (3 params): data/processed/spc_analysis.csv")
    print("  운영자 리포트: data/processed/operator_report.json")
    print(f"  LLM 프롬프트: {prompt_path}")
    print("  LLM 리포트: data/processed/llm_reports.json")
    print("  전체 benchmark: data/processed/all_dataset_benchmark.csv")
    print("  Cross eval: data/processed/cross_dataset_eval.csv")


if __name__ == "__main__":
    main()
