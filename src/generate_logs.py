"""
generate_logs.py — Sort Test 시뮬레이션 로그 생성

구조: 1 Lot × 5 Wafers × 8 Sites × 25 measurements = 1000 rows per scenario
각 row = 1개 die 측정 결과

반도체 Sort Test 파라미터:
- vt_shift: MOSFET 임계전압 변화 (mV), 정상범위 30~70mV
- leakage_curr: 누설 전류 (nA), 정상범위 20~40nA
- test_time_ms: ATE 사이클 타임 (ms), 정상범위 85~115ms
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# 구조 상수
N_WAFERS = 5
N_SITES = 8
N_MEAS = 25   # wafer당 site당 측정 수
LOT_ID = "LOT_A001"


def _make_index():
    """Lot-Wafer-Site-측정 순서로 인덱스 생성 (1000 rows)"""
    records = []
    t = 0
    for w in range(1, N_WAFERS + 1):
        for s in range(1, N_SITES + 1):
            for m in range(N_MEAS):
                records.append({
                    "timestamp": t,
                    "lot_id": LOT_ID,
                    "wafer_id": f"W{w:02d}",
                    "site_id": f"S{s:01d}",
                })
                t += 1
    return pd.DataFrame(records)


def make_base_log(scenario_id="S1"):
    """
    기본 정상 Sort Test 로그 생성.
    Lot-Wafer-Site 계층 구조 포함.
    """
    idx = _make_index()
    n = len(idx)  # 1000

    idx["scenario_id"] = scenario_id
    idx["vt_shift"] = 50 + np.random.normal(0, 1.5, n)
    idx["leakage_curr"] = 30 + np.random.normal(0, 1.0, n)
    idx["test_time_ms"] = 100 + np.random.normal(0, 3, n)
    idx["test_error_code"] = 0
    idx["ooc_flag"] = 0

    return idx.reset_index(drop=True)


def _wafer_mask(df, wafer_from=1, wafer_to=None):
    """특정 wafer 범위에 해당하는 row mask 반환"""
    wafer_nums = df["wafer_id"].str.extract(r"W(\d+)")[0].astype(int)
    if wafer_to is None:
        wafer_to = N_WAFERS
    return (wafer_nums >= wafer_from) & (wafer_nums <= wafer_to)


def _site_mask(df, site_ids):
    """특정 site 번호 리스트에 해당하는 row mask 반환"""
    return df["site_id"].isin([f"S{s}" for s in site_ids])


def inject_spike(df, start=300, end=340, magnitude=20):
    df = df.copy()
    df.loc[start:end, "vt_shift"] += magnitude
    return df


def inject_vt_drift_from_wafer(df, wafer_from=3, slope=0.005):
    """
    특정 wafer 이후 vt_shift 점진 상승 (공정 드리프트 — 실제 웨이퍼 간 drift 모사)
    """
    df = df.copy()
    mask = _wafer_mask(df, wafer_from=wafer_from)
    drift_idx = df.index[mask]
    drift_vals = np.arange(len(drift_idx)) * slope * 40
    df.loc[drift_idx, "vt_shift"] += drift_vals
    return df


def inject_test_time_drift(df, wafer_from=3, slope=0.15):
    """
    특정 wafer 이후 test_time_ms 점진 증가 (ATE 성능 저하 — 특정 wafer 이후 시작)
    """
    df = df.copy()
    mask = _wafer_mask(df, wafer_from=wafer_from)
    drift_idx = df.index[mask]
    drift_vals = np.arange(len(drift_idx)) * slope
    df.loc[drift_idx, "test_time_ms"] += drift_vals
    return df


def inject_error_repeat(df, positions):
    """특정 위치에 반복 test_error_code 삽입 (접촉 불량 / 장비 에러)"""
    df = df.copy()
    for p in positions:
        if 0 <= p < len(df):
            df.loc[p, "test_error_code"] = 21
    return df


def inject_site_dropout(df, site_ids, wafer_from=2, wafer_to=3):
    """
    특정 site + wafer 구간에서 leakage_curr 신호 손실 (프로브 이상 — site 특정)
    """
    df = df.copy()
    mask = _site_mask(df, site_ids) & _wafer_mask(df, wafer_from, wafer_to)
    df.loc[mask, "leakage_curr"] = 0
    return df


def inject_noise_burst(df, start=200, end=260, scale=8):
    """전기적 노이즈 burst 주입"""
    df = df.copy()
    noise = np.random.normal(0, scale, end - start + 1)
    df.loc[start:end, "vt_shift"] += noise
    return df


def inject_vt_high_wafers(df, wafer_from=2, wafer_to=5, offset=22):
    """특정 wafer 범위 전체에서 vt_shift 높은 수준 지속 (공정 이탈)"""
    df = df.copy()
    mask = _wafer_mask(df, wafer_from=wafer_from, wafer_to=wafer_to)
    df.loc[mask, "vt_shift"] += offset
    return df


def build_scenarios():
    scenarios = []

    # S1: 정상 Sort Test 통과 구간 (5 wafers 전부 정상)
    s1 = make_base_log("S1")
    scenarios.append(s1)

    # S2: W3에서 Vt 순간 스파이크 후 자가 회복
    s2 = make_base_log("S2")
    # W3 시작 index ≈ (3-1)*8*25 = 400
    s2 = inject_spike(s2, 400, 440, 18)
    scenarios.append(s2)

    # S3: W2 이후 Vt 지속 상승 — 공정 드리프트 (lot-level degradation)
    s3 = make_base_log("S3")
    s3 = inject_vt_high_wafers(s3, wafer_from=2, wafer_to=5, offset=22)
    scenarios.append(s3)

    # S4: W3 이후 테스트 사이클 타임 점진 증가 — ATE 성능 저하
    s4 = make_base_log("S4")
    s4 = inject_test_time_drift(s4, wafer_from=3, slope=0.15)
    scenarios.append(s4)

    # S5: 반복 테스트 에러 — 여러 wafer에 걸쳐 산발적 접촉 불량
    s5 = make_base_log("S5")
    s5 = inject_error_repeat(s5, [200, 280, 360, 440, 520])
    scenarios.append(s5)

    # S6: W2~W3 Site 5에서 Leakage 측정 신호 손실 — 프로브 이상 (site 특정)
    s6 = make_base_log("S6")
    s6 = inject_site_dropout(s6, site_ids=[5], wafer_from=2, wafer_to=3)
    scenarios.append(s6)

    # S7: W2에서 전기적 노이즈 burst 후 W3부터 회복
    s7 = make_base_log("S7")
    s7 = inject_noise_burst(s7, 200, 295, 10)
    scenarios.append(s7)

    # S8: W2 이후 복합 이상 — Vt + 사이클 타임 + 에러 + OOC
    s8 = make_base_log("S8")
    s8 = inject_vt_drift_from_wafer(s8, wafer_from=2, slope=0.008)
    s8 = inject_test_time_drift(s8, wafer_from=2, slope=0.12)
    s8 = inject_error_repeat(s8, [220, 260, 300, 340, 380])
    wafer2_5_mask = _wafer_mask(s8, wafer_from=2, wafer_to=5)
    s8.loc[wafer2_5_mask, "ooc_flag"] = 1
    scenarios.append(s8)

    # S9: W3에서 Vt 이상 감지 → W4~W5 재테스트 통과
    s9 = make_base_log("S9")
    s9 = inject_spike(s9, 400, 510, 15)
    s9 = inject_error_repeat(s9, [420, 460])
    scenarios.append(s9)

    # S10: W2 이후 Vt 이상 + 사이클 타임 증가 후 회복 실패 — Lot 판정 FAIL
    s10 = make_base_log("S10")
    s10 = inject_vt_high_wafers(s10, wafer_from=2, wafer_to=5, offset=18)
    s10 = inject_test_time_drift(s10, wafer_from=2, slope=0.10)
    s10 = inject_error_repeat(s10, [250, 330, 410, 490])
    scenarios.append(s10)

    return pd.concat(scenarios, ignore_index=True)


def main():
    os.makedirs("data/raw", exist_ok=True)

    df = build_scenarios()
    save_path = "data/raw/test_logs.csv"
    df.to_csv(save_path, index=False)

    print("파일 저장 완료:", save_path)
    print(f"\n컬럼: {list(df.columns)}")
    print("\n상위 3행:")
    print(df.head(3).to_string())
    print(f"\n총 row 수: {len(df)}")
    print("\n시나리오별 wafer 구성 (S1 예시):")
    s1 = df[df["scenario_id"] == "S1"]
    print(s1.groupby(["wafer_id", "site_id"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
