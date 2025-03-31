# Python 기반 서빙/실행 환경
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# 기본 실행 명령 (컨테이너 유지용)
CMD ["tail", "-f", "/dev/null"]
