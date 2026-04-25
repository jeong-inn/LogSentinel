# LogSentinel — 반도체 Sort Test 로그 판정·검증·리포트 자동화 시스템

반도체 Sort/Final Test 공정에서 발생하는 ATE(Automated Test Equipment) 로그를 자동으로 분석하고, 이상 탐지 → 상태 분류 → 판정 → 정책 검증 → SPC 분석 → 운영자 리포트까지 수행하는 검증 플랫폼.

---

## 프로젝트 배경

Sort Test 로그 판정은 현장에서 엔지니어가 수동으로 수행하는 경우가 많아, 판정 기준의 일관성과 추적성이 부족합니다. LogSentinel은 이 과정을 자동화하여:

- **판정 일관성**: 동일 이상 패턴에 대해 항상 동일한 기준으로 판정
- **검증 가능성**: expected vs. actual 비교로 파이프라인 자체를 검증
- **조기 경고**: SPC 관리한계 및 Western Electric Rules로 공정 이탈 조기 감지
- **운영자 지원**: 원인 분석 + 후속 조치 추천 + LLM 기반 리포트 자동 생성

을 목표로 함.

---

## 주요 기능

### 1. Sort Test 파라미터 SPC (Statistical Process Control)
3개 핵심 파라미터에 대한 공정 능력 분석:

| 파라미터 | 의미 | 정상 범위 (USL/LSL) |
|---------|------|---------------------|
| `vt_shift` | MOSFET 임계전압 변화 (mV) | 30 ~ 70 mV |
| `leakage_curr` | 누설 전류 (nA) | 20 ~ 40 nA |
| `test_time_ms` | ATE 사이클 타임 (ms) | 80 ~ 120 ms |

- **Cpk / Ppk**: 공정 능력 지수 (Cpk ≥ 1.33 = 양호)
- **UCL / LCL**: X-bar 관리한계 (mean ± 3σ)
- **Western Electric Rules**: 8가지 OOC 패턴 탐지 (런/트렌드/층화 포함)

### 2. Lot-Wafer-Site 계층 구조
```
LOT_A001
├── W01 ~ W05 (5 Wafers)
│   └── S1 ~ S8 (8 Sites per Wafer)
│       └── 25 measurements per Site
→ 1,000 rows per scenario (10 scenarios = 10,000 rows)
```

### 3. 이상 탐지 파이프라인 (Synthetic)
```
preprocess → anomaly_detect → state_assign → judge → validate → SPC → root_cause → report
```

상태: `NORMAL` → `WARNING` → `CRITICAL` → `RECOVERY` / `FAIL`
판정: `PASS` / `PASS_WITH_WARNING` / `FAIL`
Release Gate: `READY` / `MONITORING_REQUIRED` / `REVIEW_REQUIRED` / `BLOCKED`

### 4. 실데이터 벤치마크 (NAB 5종)
5가지 detector를 5개 NAB 데이터셋에서 event-level로 비교:

| Detector | 설명 |
|---------|------|
| `fixed_threshold` | mean + 2.5σ 고정 임계값 |
| `isolation_forest` | sklearn IsolationForest |
| `zscore_only` | rolling z-score ≥ 3.0 |
| `v1_pipeline` | z-score + ewm + cooldown |
| `v2_pipeline` | detrended residual + z + ewm + persistence + expand |

**v2 self-tuned 결과 (event recall):**

| Dataset | GT Events | Hit | FA | Event Recall | Event F1 |
|---------|-----------|-----|----|-------------|----------|
| ambient_temperature | 2 | 2 | 15 | 1.0 | 0.211 |
| machine_temperature | 4 | 4 | 121 | 1.0 | 0.062 |
| cpu_utilization | 2 | 2 | 11 | 1.0 | 0.267 |
| ec2_request_latency | 3 | 3 | 0 | **1.0** | **1.000** |
| rogue_agent_key_hold | 2 | 2 | 6 | 1.0 | 0.400 |

5개 전 데이터셋 event recall 1.0 달성. ec2에서 FA 0, F1 1.0.

---

## 프로젝트 구조

```
LogSentinel/
├── src/
│   ├── main.py                    # 전체 파이프라인 진입점
│   ├── generate_logs.py           # Synthetic 시나리오 로그 생성 (seed=42)
│   ├── preprocess.py              # 로그 전처리 (rolling features)
│   ├── anomaly_detector.py        # Rule-based 이상 탐지
│   ├── state_engine.py            # 상태 분류 (NORMAL/WARNING/CRITICAL/RECOVERY/FAIL)
│   ├── judge.py                   # 최종 판정 (PASS/PASS_WITH_WARNING/FAIL)
│   ├── validator.py               # Expected vs. actual 검증
│   ├── policy_engine.py           # 후속 조치 추천 + quality summary
│   ├── root_cause.py              # 원인 분석
│   ├── spc.py                     # SPC 모듈 (Cpk/Ppk, UCL/LCL, WE Rules 8종)
│   ├── operator_report.py         # 구조화 운영자 리포트 + LLM prompt 생성
│   ├── llm_reporter.py            # OpenAI API 연동 (API key 없으면 fallback)
│   ├── real_data_loader.py        # NAB 데이터 로드 + ground truth 생성
│   ├── real_data_benchmark_v2.py  # Detector 5종 + event-level scoring
│   └── real_data_cross_eval.py    # Multi-dataset benchmark + cross-dataset eval
├── tests/
│   ├── test_spc.py                # SPC 모듈 단위 테스트 (17개)
│   ├── test_event_scoring.py      # Event-level scoring 단위 테스트 (10개)
│   └── test_iforest.py            # Detector 단위 테스트 (7개)
├── data/
│   ├── raw/test_logs.csv          # Synthetic 로그 (10,000 rows, seed=42)
│   ├── external/                  # NAB 실데이터 CSV 5종
│   ├── scenarios/scenario_specs.json  # 시나리오별 expected 판정 기준
│   └── processed/                 # main.py 실행 시 자동 생성
├── app.py                         # Streamlit 대시보드
└── requirements.txt
```

---

## 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 전체 파이프라인 실행 (약 2~5분, real-data grid search 포함)
python3 src/main.py

# 단위 테스트 (34개)
pytest tests/

# 대시보드 실행
streamlit run app.py
```

> **참고:** OpenAI API key가 없으면 LLM report는 fallback으로 생성됨. 나머지 파이프라인은 정상 동작함.

---

## Streamlit 대시보드

4개 탭으로 구성:

| 탭 | 내용 |
|----|------|
| **타임라인** | 시나리오별 vt_shift/leakage_curr/test_time_ms 시계열 + 이상 마킹 |
| **SPC 관리도** | 파라미터 선택 → X-bar Chart (UCL/LCL) + WE Rule 위반 + Cpk 비교 |
| **Wafer 분석** | Wafer별 평균 bar chart + Site×Wafer heatmap + 이상 비율 |
| **판정 요약** | 시나리오별 최종 판정 + 검증 결과 + Release Gate 현황 |

---

## 10개 시나리오

| ID | 시나리오 | 판정 | Release Gate |
|----|---------|------|--------------|
| S1 | 정상 Sort Test 통과 구간 | PASS | READY |
| S2 | W3에서 Vt 순간 스파이크 후 자가 회복 | PASS_WITH_WARNING | MONITORING_REQUIRED |
| S3 | W2 이후 Vt 지속 상승 (공정 드리프트) | FAIL | BLOCKED |
| S4 | W3 이후 테스트 사이클 타임 점진 증가 | PASS_WITH_WARNING | REVIEW_REQUIRED |
| S5 | 반복 테스트 에러 (접촉 불량 의심) | PASS_WITH_WARNING | READY |
| S6 | W2~W3 Site 5 Leakage 신호 손실 (프로브 이상) | FAIL | BLOCKED |
| S7 | W2 전기적 노이즈 burst 후 W3 회복 | PASS_WITH_WARNING | REVIEW_REQUIRED |
| S8 | W2 이후 Vt + 사이클 타임 + 에러 복합 이상 | FAIL | BLOCKED |
| S9 | W3 Vt 이상 감지 후 재테스트 통과 | PASS_WITH_WARNING | REVIEW_REQUIRED |
| S10 | W2 이후 Vt 이상 + 사이클 타임 회복 실패 | FAIL | BLOCKED |

---

## 기술 스택

- **Python 3.9**, pandas, numpy, scikit-learn
- **Streamlit** — 대시보드
- **Plotly** — 인터랙티브 차트
- **pytest** — 단위 테스트 (34개)
- **OpenAI API** — LLM 리포트 (선택)
