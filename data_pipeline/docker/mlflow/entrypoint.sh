#!/bin/bash

# 1️⃣ PostgreSQL DB 스키마 업그레이드
echo "👉 MLflow DB 스키마 업그레이드 진행 중..."
mlflow db upgrade ${BACKEND_STORE_URI}

# 2️⃣ MLflow 서버 실행
echo "🚀 MLflow 서버 시작!"
mlflow server \
  --backend-store-uri ${BACKEND_STORE_URI} \
  --default-artifact-root ${ARTIFACT_ROOT} \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts