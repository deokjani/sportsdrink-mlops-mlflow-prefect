import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import boto3

# ------------------------------------------------------
# [1] 설정값
# ------------------------------------------------------
model_name = "sportsdrink_search_ratio_rf"
serving_version = 1
experiment_name = "sportsdrink_searchshare_train"

# ------------------------------------------------------
# [2] 데이터 로드 및 전처리
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

X = pd.concat([
    tier_feature.reset_index(drop=True),
    naver_data[["year", "month", "day"]].reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)
y = naver_data["ratio"].values

# ------------------------------------------------------
# [3] 학습/검증 데이터 분할
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# [4] MLflow 설정
# ------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(experiment_name)

# ------------------------------------------------------
# [5] 모델 학습 및 결과 저장
# ------------------------------------------------------
with mlflow.start_run(run_name=f"{model_name}_train_v{serving_version}") as run:
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_filename = f"train_data_{timestamp}.csv"
    train_path = f"/tmp/{train_filename}"

    train_df = X.copy()
    train_df["ratio"] = y
    train_df.to_csv(train_path, index=False)

    # S3 업로드 (MinIO 사용 가정)
    s3_key = f"train/v{serving_version}/latest_train_data.csv"
    minio_client = boto3.client("s3", endpoint_url="http://minio:9000")
    minio_client.upload_file(train_path, "mlflow-artifacts", s3_key)

    mlflow.log_artifact(train_path, artifact_path=f"train/v{serving_version}")
    print(f"✅ 학습 데이터 저장 완료: s3://mlflow-artifacts/{s3_key}")

    # 모델 등록
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )

    print(f"✅ MLflow 모델 등록 완료! Run ID: {run.info.run_id}")
