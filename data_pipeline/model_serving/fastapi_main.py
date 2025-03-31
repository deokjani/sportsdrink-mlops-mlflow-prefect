from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator  # ✅ Prometheus 연동

# ✅ MLflow 설정
mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_NAME = "sportsdrink_search_ratio_rf"
MODEL_VERSION = 1
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ 모델 로드 완료: {model_uri}")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    model = None

# ✅ 학습 시 사용한 피처 목록
FEATURE_COLUMNS = [
    "channel_tier_numeric", "year", "month", "day",
    "age_group_10대", "age_group_20대", "age_group_30대",
    "age_group_40대", "age_group_50대", "age_group_60대 이상",
    "gender_female", "gender_male",
    "brand_게토레이", "brand_링티", "brand_토레타",
    "brand_파워에이드", "brand_포카리스웨트"
]

# ✅ FastAPI 앱 생성
app = FastAPI(title="SportsDrink 소비 비율 예측 API", version="1.0")

# ✅ Prometheus 지표 노출
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# ✅ 입력 스키마
class PredictionInput(BaseModel):
    year: int
    month: int
    day: int
    age_group: str
    gender: str
    brand: str
    channel_tier_numeric: int

# ✅ 입력 변환 함수
def build_input_df(data: PredictionInput):
    gender_map = {"여성": "female", "남성": "male", "female": "female", "male": "male"}
    brand_map = {
        "게토레이": "게토레이", "링티": "링티", "토레타": "토레타", "파워에이드": "파워에이드",
        "포카리": "포카리스웨트", "포카리스웨트": "포카리스웨트"
    }

    input_dict = {col: 0 for col in FEATURE_COLUMNS}
    input_dict["year"] = data.year
    input_dict["month"] = data.month
    input_dict["day"] = data.day
    input_dict["channel_tier_numeric"] = data.channel_tier_numeric
    input_dict[f"age_group_{data.age_group}"] = 1

    gender_val = gender_map.get(data.gender, data.gender)
    brand_val = brand_map.get(data.brand, data.brand)

    input_dict[f"gender_{gender_val}"] = 1
    input_dict[f"brand_{brand_val}"] = 1

    return pd.DataFrame([input_dict])

# ✅ POST 요청
@app.post("/predict")
def predict_post(input_data: PredictionInput):
    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    input_df = build_input_df(input_data)
    try:
        prediction = model.predict(input_df)
        return {
            "method": "POST",
            "input": input_data.dict(),
            "predicted_ratio": round(float(prediction[0]), 4)
        }
    except Exception as e:
        return {"error": f"예측 실패: {e}"}

# ✅ GET 요청
@app.get("/predict")
def predict_get(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    age_group: str = Query(...),
    gender: str = Query(...),
    brand: str = Query(...),
    channel_tier_numeric: int = Query(...)
):
    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    input_data = PredictionInput(
        year=year,
        month=month,
        day=day,
        age_group=age_group,
        gender=gender,
        brand=brand,
        channel_tier_numeric=channel_tier_numeric
    )

    input_df = build_input_df(input_data)

    try:
        prediction = model.predict(input_df)
        return {
            "method": "GET",
            "input": input_data.dict(),
            "predicted_ratio": round(float(prediction[0]), 4)
        }
    except Exception as e:
        return {"error": f"예측 실패: {e}"}

# ✅ 로컬 실행 시
if __name__ == "__main__":
    print("🚀 FastAPI 서버 실행 중 (http://0.0.0.0:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)