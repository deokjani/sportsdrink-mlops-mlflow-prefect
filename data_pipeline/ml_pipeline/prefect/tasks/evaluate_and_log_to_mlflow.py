import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import boto3
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------
# [1] 설정값 정의
# ------------------------------------------------------
serving_version = 1
bucket = "mlflow-artifacts"
prediction_prefix = f"predictions/v{serving_version}/"
evaluation_prefix = f"evaluation/v{serving_version}/"
experiment_name = "sportsdrink_searchshare_evaluation"

# ------------------------------------------------------
# [2] MinIO에서 최신 예측 결과 다운로드
# ------------------------------------------------------
minio_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000"
)

response = minio_client.list_objects_v2(Bucket=bucket, Prefix=prediction_prefix)
csv_files = [f["Key"] for f in response.get("Contents", []) if f["Key"].endswith(".csv")]
latest_file_key = sorted(csv_files)[-1]

local_prediction_path = f"/tmp/{latest_file_key.split('/')[-1]}"
minio_client.download_file(bucket, latest_file_key, local_prediction_path)
print(f"✅ 최신 예측 결과 다운로드 완료: {latest_file_key}")

# ------------------------------------------------------
# [3] 예측 결과 로드 및 평가 지표 계산
# ------------------------------------------------------
df = pd.read_csv(local_prediction_path)
y_true = df["ratio"]
y_pred = df["predicted_ratio"]

def mse_safe(y_true, y_pred, squared=False):
    try:
        return mean_squared_error(y_true, y_pred, squared=squared)
    except TypeError:
        mse = mean_squared_error(y_true, y_pred)
        return mse if squared else mse ** 0.5

rmse = mse_safe(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = (abs((y_true - y_pred) / y_true)).mean() * 100

print("\n📊 [모델 평가 결과]")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# ------------------------------------------------------
# [4] 평가 결과 저장 (CSV + 시각화)
# ------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"/tmp/evaluation_plot_v{serving_version}_{timestamp}.png"
report_path = f"/tmp/evaluation_report_v{serving_version}_{timestamp}.csv"

plt.figure(figsize=(14, 6))
plt.plot(df["period"], y_true, marker="o", label="Actual (ratio)")
plt.plot(df["period"], y_pred, marker="x", label="Predicted (predicted_ratio)")
plt.xticks(rotation=45)
plt.title(f"Sports Drink Market Share Evaluation - v{serving_version}")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

report_df = pd.DataFrame({
    "metric": ["RMSE", "MAE", "R2", "MAPE(%)"],
    "value": [rmse, mae, r2, mape]
})
report_df.to_csv(report_path, index=False)

print("✅ 평가 시각화 및 리포트 저장 완료")

# ------------------------------------------------------
# [5] MinIO 업로드
# ------------------------------------------------------
minio_client.upload_file(plot_path, bucket, f"{evaluation_prefix}evaluation_plot.png")
minio_client.upload_file(report_path, bucket, f"{evaluation_prefix}evaluation_report.csv")
print(f"✅ MinIO 업로드 완료: s3://{bucket}/{evaluation_prefix}")

# ------------------------------------------------------
# [6] MLflow 기록
# ------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"evaluation_run_v{serving_version}") as run:
    mlflow.log_artifact(plot_path, artifact_path=evaluation_prefix)
    mlflow.log_artifact(report_path, artifact_path=evaluation_prefix)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)

    print(f"✅ MLflow 저장 완료! Run ID: {run.info.run_id}")
