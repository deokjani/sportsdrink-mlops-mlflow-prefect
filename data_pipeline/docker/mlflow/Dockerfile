FROM ghcr.io/mlflow/mlflow:v2.9.2

# 필수 패키지 설치
RUN pip install --no-cache-dir psycopg2-binary boto3

# entrypoint 복사
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]