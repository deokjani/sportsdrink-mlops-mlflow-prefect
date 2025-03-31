import uuid
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import boto3
import os

# ------------------------------------------------------
# [1] 설정값 정의
# ------------------------------------------------------
model_name = "sportsdrink_search_ratio_rf"
model_version = 1
serving_version = 1
experiment_name = "sportsdrink_searchshare_prediction"

# ------------------------------------------------------
# [2] MLflow 모델 로드
# ------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(experiment_name)

model_uri = f"models:/{model_name}/{model_version}"
print(f"👉 모델 로딩 중... {model_uri}")
loaded_model = mlflow.pyfunc.load_model(model_uri)
print("✅ 모델 로딩 완료!")

# ------------------------------------------------------
# [3] 데이터 로드 및 전처리
# ------------------------------------------------------
naver_path = "/app/data_pipeline/data/features/sportsdrink_youtube_search_daily/SportsDrink_Naver_Search_Daily_20250325.csv"
youtube_path = "/app/data_pipeline/data/features/sportsdrink_youtube_search_daily/standard_feature_mid_large.csv"

naver_data = pd.read_csv(naver_path)
naver_data["period"] = pd.to_datetime(naver_data["period"])
naver_data["year"] = naver_data["period"].dt.year
naver_data["month"] = naver_data["period"].dt.month
naver_data["day"] = naver_data["period"].dt.day

categorical_cols = ["age_group", "gender", "brand"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(naver_data[categorical_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=naver_data.index
)

youtube_df = pd.read_csv(youtube_path)
tier_map = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3, "diamond": 4}
tier_numeric = youtube_df["channel_tier"].map(tier_map).iloc[0]
tier_feature = pd.DataFrame([tier_numeric] * len(naver_data), columns=["channel_tier_numeric"])

X_infer = pd.concat([
    tier_feature.reset_index(drop=True),
    naver_data[["year", "month", "day"]].reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)

# ------------------------------------------------------
# [4] 예측 수행
# ------------------------------------------------------
predictions = loaded_model.predict(X_infer)

# ------------------------------------------------------
# [5] 결과 저장
# ------------------------------------------------------
result_df = naver_data.copy()
result_df["predicted_ratio"] = predictions
result_df["inference_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
result_df["request_id"] = [str(uuid.uuid4()) for _ in range(len(result_df))]

result_df["ratio"] = result_df["ratio"].round(2)
result_df["predicted_ratio"] = result_df["predicted_ratio"].round(2)

cols_to_save = [
    "period", "age_group", "gender", "brand",
    "ratio", "predicted_ratio",
    "inference_time", "request_id"
]
result_df = result_df[cols_to_save]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"sportsdrink_prediction_v{serving_version}_{timestamp}.csv"
local_path = f"/app/data_pipeline/data/predictions/{file_name}"
os.makedirs(os.path.dirname(local_path), exist_ok=True)
result_df.to_csv(local_path, index=False)

print(f"✅ 예측 결과 저장 완료: {local_path}")

# ------------------------------------------------------
# [6] MinIO 업로드
# ------------------------------------------------------
minio_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000"
)
s3_key = f"predictions/v{serving_version}/{file_name}"
minio_client.upload_file(local_path, "mlflow-artifacts", s3_key)

print(f"✅ MinIO 업로드 완료: s3://mlflow-artifacts/{s3_key}")

# ------------------------------------------------------
# [7] MLflow Artifact 등록
# ------------------------------------------------------
with mlflow.start_run(run_name=f"sportsdrink_prediction_run_v{serving_version}") as run:
    mlflow.log_artifact(local_path, artifact_path=f"predictions/v{serving_version}")
    print(f"✅ MLflow artifact 등록 완료! Run ID: {run.info.run_id}")
