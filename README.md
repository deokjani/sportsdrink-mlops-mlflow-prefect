# SportsDrink MLOps Pipeline: MLflow + Prefect 기반 모델 자동화 및 모니터링 시스템

이 프로젝트는 스포츠음료 검색 데이터를 기반으로 한 머신러닝 모델의 학습, 예측, 평가를 자동화하고,  
FastAPI 및 Prometheus-Grafana를 통해 실시간 모니터링 가능한 엔드-투-엔드 MLOps 파이프라인입니다.

---

## 프로젝트 개요

- Prefect를 활용해 모델 학습, 예측, 평가 단계를 하나의 Flow로 자동화
- MLflow로 실험 및 모델 버전을 체계적으로 관리
- FastAPI로 모델 서빙 API 제공, Streamlit으로 대시보드 시각화 구현
- Prometheus + Grafana로 실시간 상태 모니터링 구성
- MinIO(S3 호환) 기반의 모델 Artifact 저장소 구성

---

## 전체 파이프라인 흐름

```
Feature 데이터 생성
→ 모델 학습 (MLflow 로깅)
→ 예측
→ 평가 (시각화 포함)
→ 모델 Registry 등록
→ FastAPI 모델 서빙 (REST API)
→ Streamlit 대시보드 시각화
→ Prometheus 수집
→ Grafana 대시보드 모니터링
```

---

## 사용 기술 스택

| 분야           | 사용 기술 |
|----------------|-----------|
| 데이터 처리    | pandas, numpy, scikit-learn |
| 모델 실험 관리 | MLflow, MinIO (S3 호환) |
| 자동화         | Prefect 2.x (Docker Agent)|
| 서빙/API       | FastAPI, Streamlit |
| 모니터링       | Prometheus, Grafana |
| 배포           | Docker, GitHub Actions (확장 예정) |

---

## MLflow 실험 및 모델 관리

- `mlflow.sklearn.log_model()`을 통해 모델 자동 로깅 및 Registry 등록
- 실험 이름, 파라미터, 메트릭, 아티팩트를 MLflow UI에서 체계적으로 관리
- 예측 결과 및 평가 리포트를 `.csv`, `.png`로 저장하여 Artifact로 등록

---

## Prefect 기반 파이프라인 자동화

- `@flow`, `@task` 기반으로 모델 학습 → 예측 → 평가 Task 구성
- Docker 기반 Prefect Agent에서 스케줄링 및 자동 실행
- `.env` 환경변수로 민감 정보 및 경로 설정 통일

---

## FastAPI 서빙 및 Prometheus 모니터링

- `/predict` 엔드포인트를 통해 실시간 예측 가능
- `/metrics` 엔드포인트에서 Prometheus가 지표 수집
- Grafana에서 API 요청 수, 지연 시간, 오류율 실시간 시각화 가능

---

## Streamlit 대시보드

- 실시간 예측 결과 또는 데이터 트렌드를 사용자 친화적으로 시각화
- 학습된 모델의 성능 지표와 평가 결과를 손쉽게 확인
- 운영 중인 예측 결과를 시각적으로 검토할 수 있는 인터페이스 제공

---

## 환경 변수 설정 예시

```env
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ARTIFACT_ROOT=s3://mlflow-artifacts
```

---

## 디렉토리 구조 예시

```
data_pipeline/
├── ml_pipeline/
│   ├── flows/                    # Prefect Flow 정의
│   ├── tasks/                    # Task별 Python 스크립트
│   └── artifacts/                # 평가 결과 및 plot 저장
├── model_serving/                # FastAPI 서빙 코드, Streamlit 대시보드 앱
├── docker/                       # Dockerfile, docker-compose 설정
├── monitoring/                   # prometheus.yml, grafana 설정
├── data/                         # 데이터 저장 디렉토리
├── logs/                         # Prefect, FastAPI 로그
└── data_pipeling_guide.png       # 폴더 가이드 라인
```

---

## 향후 확장 계획

- Redis 기반 Feature Store 구축 및 버전 관리 자동화
- ML Drift / Concept Drift 탐지 기능 연동
- GitHub Actions 기반 FastAPI, Prefect, Streamlit 자동 배포
- 예측 성능 변화 대시보드 고도화 및 운영 편의성 강화

---

## 마무리

이 프로젝트는 MLflow, Prefect, FastAPI, Streamlit, Prometheus 등 **실무 핵심 MLOps 기술**을 통합한 예제로,  
모델 개발부터 자동화, 서빙, 모니터링까지 모든 단계를 경험할 수 있도록 구성되었습니다.

데이터 엔지니어링 + 머신러닝 자동화 + 운영 모니터링을 모두 아우르는 구조로,  
포트폴리오 및 실무 적용에 적합한 확장성과 완성도를 갖추고 있습니다.
