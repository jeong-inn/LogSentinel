import pandas as pd
from real_data_eval import preprocess_real_df, detect_real_anomalies, score_real_detection


def tune_real_detector(raw_df, dataset_name):
    """
    z_thresh / ewm_thresh_scale / cooldown 조합을 grid search 해서
    F1 기준 best 설정을 찾는다.
    """
    candidates = []

    z_values = [1.5, 2.0, 2.5, 3.0]
    ewm_values = [0.8, 1.0, 1.5, 2.0]
    cooldown_values = [0, 3, 5, 10]

    for z_thresh in z_values:
        for ewm_thresh_scale in ewm_values:
            for cooldown in cooldown_values:
                df = preprocess_real_df(raw_df.copy())
                df = detect_real_anomalies(
                    df,
                    z_thresh=z_thresh,
                    ewm_thresh_scale=ewm_thresh_scale,
                    cooldown=cooldown
                )

                score_df = score_real_detection(df, dataset_name)
                row = score_df.iloc[0].to_dict()
                row["z_thresh"] = z_thresh
                row["ewm_thresh_scale"] = ewm_thresh_scale
                row["cooldown"] = cooldown
                candidates.append(row)

    result_df = pd.DataFrame(candidates)

    # 우선순위: f1 > recall > precision
    result_df = result_df.sort_values(
        by=["f1", "recall", "precision"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_row = result_df.iloc[0].to_dict()
    return result_df, best_row


def apply_best_detector(raw_df, dataset_name, best_row):
    """
    best parameter로 다시 탐지
    """
    df = preprocess_real_df(raw_df.copy())
    df = detect_real_anomalies(
        df,
        z_thresh=best_row["z_thresh"],
        ewm_thresh_scale=best_row["ewm_thresh_scale"],
        cooldown=int(best_row["cooldown"])
    )
    score_df = score_real_detection(df, dataset_name)
    return df, score_df