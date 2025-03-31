# SportsDrink MLOps Pipeline: MLflow + Prefect 기반 모델 자동화 및 모니터링 시스템

이 프로젝트는 스포츠음료 검색 데이터를 기반으로 한 머신러닝 모델의 학습, 예측, 평가를 자동화하고,  
FastAPI 및 Prometheus-Grafana를 통해 실시간 모니터링 가능한 엔드-투-엔드 MLOps 파이프라인입니다.

___

## 프로젝트 개요

- Prefect를 활용해 모델 학습, 예측, 평가 단계를 하나의 Flow로 자동화
- MLflow로 실험 및 모델 버전을 체계적으로 관리
- FastAPI로 모델 서빙 API 제공, Prometheus + Grafana로 실시간 상태 모니터링
- MinIO(S3 호환) 기반의 모델 Artifact 저장소 구성

___

## 전체 파이프라인 흐름

```
Feature 데이터 생성 
→ 모델 학습 (MLflow 로깅) 
→ 예측 
→ 평가 (시각화 포함) 
→ 모델 Registry 등록 
→ FastAPI 모델 서빙 (predict)
→ Prometheus 수집 
→ Grafana 시각화 + Slack 알림 연동
```

___

## 사용 기술 스택

| 분야           | 사용 기술 |
|----------------|-----------|
| 데이터 처리    | pandas, numpy, scikit-learn |
| 모델 실험 관리 | MLflow, MinIO (S3 호환) |
| 자동화         | Prefect 2.x (Docker Agent)|
| 서빙/API       | FastAPI, Streamlit |
| 모니터링       | Prometheus, Grafana |
| 배포           | Docker |

___

## MLflow 실험 및 모델 관리

- `mlflow.sklearn.log_model()`을 통해 모델 자동 로깅 및 Registry 등록
- 실험 이름, 파라미터, 메트릭, 아티팩트를 MLflow UI에서 관리
- 예측 결과 및 평가 리포트를 `.csv` 및 `.png`로 저장 후 Artifact로 등록

___

## Prefect 기반 파이프라인 자동화

- `@flow`, `@task`로 구성된 Prefect Flow 정의
- 모델 학습 → 예측 → 평가 순서로 Task 실행
- Docker Agent 기반의 Prefect Work Pool에서 Flow 자동 실행
- `.env` 환경변수를 통해 실행 환경 관리

___

## FastAPI 서빙 및 모니터링

- `/predict` 엔드포인트로 RESTful API 요청 가능
- `/metrics` 엔드포인트를 통해 Prometheus 지표 수집
- Grafana에서 실시간 요청 수, 지연 시간, 에러율 등 모니터링 가능

___

## 환경 변수 설정 예시

```env
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ARTIFACT_ROOT=s3://mlflow-artifacts
```

- Prefect에서는 `prefect.yaml` 혹은 Docker Job Variables로 설정
- FastAPI는 `.env` 또는 Docker 환경변수로 주입 가능

___

## 디렉토리 구조 예시

```
data_pipeline/
├── ml_pipeline/
│   ├── flows/                    # Prefect Flow 정의
│   ├── tasks/                    # Task별 Python 스크립트
│   └── artifacts/                # 평가 결과 및 plot 저장
├── fastapi_server/               # FastAPI 서빙 코드
├── docker/                       # Dockerfile, docker-compose 설정
├── monitoring/                   # prometheus.yml, grafana 설정
├── data/                         # 데이터 저장 디렉토리
├── logs/                         # Prefect, FastAPI 로그
└── requirements.txt              # 의존 패키지 명시
```

___

## 향후 확장 계획

- Redis 기반 Feature Store 구축 및 버전 관리 자동화
- ML Drift / Concept Drift 탐지 및 Slack 알림 연동
- GitHub Actions 기반 FastAPI, Prefect 배포 자동화
- 실시간 예측 성능 변화 추적 및 운영 대시보드 고도화

___

## 마무리

이 프로젝트는 MLflow, Prefect, FastAPI, Prometheus 등 실무에서 사용되는 핵심 MLOps 기술을 통합한 예제입니다.  
단순한 모델 학습을 넘어, **모델 자동화 → 배포 → 모니터링까지 이어지는 전체 MLOps 파이프라인**을 경험할 수 있도록 구성되었습니다.

실무 적용 및 포트폴리오 용도로 모두 활용 가능하며, 확장성 있는 구조로 설계되었습니다.
