import pandas as pd
from real_data_benchmark_v2 import (
    detect_fixed_threshold,
    detect_isolation_forest,
    detect_zscore_only,
    detect_v1_pipeline,
    detect_v2_pipeline,
    score_detection,
)


def tune_v2_on_dataset(df, dataset_name):
    """
    한 데이터셋에서 v2 파라미터 튜닝
    기준: event_recall 최대 -> false_alarm_events 최소 -> event_f1 최대
    """
    candidates = []

    z_values = [1.5, 2.0, 2.5, 3.0]
    ewm_values = [0.8, 1.0, 1.2, 1.5]
    persistence_values = [2, 3, 4, 5]
    expand_values = [3, 6, 9, 12]

    for z_thresh in z_values:
        for ewm_thresh_scale in ewm_values:
            for persistence_min in persistence_values:
                for expand_window in expand_values:
                    scored_df = detect_v2_pipeline(
                        df.copy(),
                        z_thresh=z_thresh,
                        ewm_thresh_scale=ewm_thresh_scale,
                        persistence_min=persistence_min,
                        expand_window=expand_window
                    )

                    score_df = score_detection(
                        scored_df,
                        dataset_name=dataset_name,
                        method_name="v2_pipeline_tuning"
                    )

                    row = score_df.iloc[0].to_dict()
                    row["z_thresh"] = z_thresh
                    row["ewm_thresh_scale"] = ewm_thresh_scale
                    row["persistence_min"] = persistence_min
                    row["expand_window"] = expand_window
                    candidates.append(row)

    tuning_df = pd.DataFrame(candidates)

    max_event_recall = tuning_df["event_recall"].max()
    filtered = tuning_df[tuning_df["event_recall"] == max_event_recall].copy()

    filtered = filtered.sort_values(
        by=["false_alarm_events", "event_f1", "event_precision"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    best_row = filtered.iloc[0].to_dict()

    return tuning_df, best_row


def _build_baseline_benchmark(df, dataset_name):
    """
    특정 데이터셋에 대해 baseline + v1 benchmark 생성
    """
    frames = []
    scored_map = {}

    fixed_df = detect_fixed_threshold(df.copy(), threshold_scale=2.5)
    scored_map["fixed_threshold"] = fixed_df
    frames.append(score_detection(fixed_df, dataset_name, "fixed_threshold"))

    iforest_df = detect_isolation_forest(df.copy(), contamination=0.05)
    scored_map["isolation_forest"] = iforest_df
    frames.append(score_detection(iforest_df, dataset_name, "isolation_forest"))

    zscore_df = detect_zscore_only(df.copy(), z_thresh=3.0)
    scored_map["zscore_only"] = zscore_df
    frames.append(score_detection(zscore_df, dataset_name, "zscore_only"))

    v1_df = detect_v1_pipeline(df.copy(), z_thresh=3.0, ewm_thresh_scale=0.8, cooldown=10)
    scored_map["v1_pipeline"] = v1_df
    frames.append(score_detection(v1_df, dataset_name, "v1_pipeline"))

    return frames, scored_map


def benchmark_single_dataset(df, dataset_name):
    """
    하나의 데이터셋에 대해:
    - baseline 3종
    - v2 자체 튜닝
    전부 수행하고 결과 반환
    """
    print(f"  [benchmark] {dataset_name} - baseline + tuning ...", flush=True)

    tuning_df, best = tune_v2_on_dataset(df.copy(), dataset_name)

    frames, scored_map = _build_baseline_benchmark(df, dataset_name)

    v2_df = detect_v2_pipeline(
        df.copy(),
        z_thresh=best["z_thresh"],
        ewm_thresh_scale=best["ewm_thresh_scale"],
        persistence_min=int(best["persistence_min"]),
        expand_window=int(best["expand_window"])
    )
    scored_map["v2_self_tuned"] = v2_df
    v2_score = score_detection(v2_df, dataset_name, "v2_self_tuned")
    frames.append(v2_score)

    benchmark_df = pd.concat(frames, ignore_index=True)

    return {
        "tuning_df": tuning_df,
        "best": best,
        "benchmark_df": benchmark_df,
        "scored_map": scored_map,
    }


def run_multi_dataset_benchmark(datasets):
    """
    N개 데이터셋에 대해 전부 benchmark 수행.

    Parameters
    ----------
    datasets : dict
        {dataset_name: DataFrame} 형태

    Returns
    -------
    dict with:
        - per_dataset: {name: benchmark_single_dataset result}
        - summary_df: 모든 데이터셋의 event-level 결과를 하나로 합친 표
        - cross_eval_df: cross-dataset 평가 결과 (첫 번째 데이터셋 파라미터를 나머지에 적용)
    """
    per_dataset = {}

    for name, df in datasets.items():
        per_dataset[name] = benchmark_single_dataset(df, name)

    # 전체 요약표
    all_benchmarks = []
    for name, result in per_dataset.items():
        all_benchmarks.append(result["benchmark_df"])
    summary_df = pd.concat(all_benchmarks, ignore_index=True)

    # Cross-dataset eval: 각 데이터셋의 best 파라미터를 다른 모든 데이터셋에 적용
    cross_frames = []
    dataset_names = list(datasets.keys())

    for tune_name in dataset_names:
        best = per_dataset[tune_name]["best"]

        for eval_name in dataset_names:
            if eval_name == tune_name:
                continue

            eval_df = datasets[eval_name]
            scored_df = detect_v2_pipeline(
                eval_df.copy(),
                z_thresh=best["z_thresh"],
                ewm_thresh_scale=best["ewm_thresh_scale"],
                persistence_min=int(best["persistence_min"]),
                expand_window=int(best["expand_window"])
            )
            score = score_detection(scored_df, eval_name, f"v2_cross({tune_name[:8]})")
            score["tuned_on"] = tune_name
            cross_frames.append(score)

    cross_eval_df = pd.concat(cross_frames, ignore_index=True) if cross_frames else pd.DataFrame()

    return {
        "per_dataset": per_dataset,
        "summary_df": summary_df,
        "cross_eval_df": cross_eval_df,
    }