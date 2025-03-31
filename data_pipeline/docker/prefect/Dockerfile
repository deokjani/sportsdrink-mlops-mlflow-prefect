# ✅ Python 3.9 기반 Prefect 이미지 사용
FROM prefecthq/prefect:2-python3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# Prefect 서비스 실행
CMD ["prefect", "server", "start"]
