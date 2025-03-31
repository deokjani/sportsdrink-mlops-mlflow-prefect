import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc

# ✅ MLflow URI 설정
mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_NAME = "sportsdrink_search_ratio_rf"
MODEL_VERSION = 1
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# ✅ 모델 로드
try:
    model = mlflow.pyfunc.load_model(model_uri)
    st.success(f"✅ 모델 로딩 완료: {MODEL_NAME} (v{MODEL_VERSION})")
except Exception as e:
    st.error(f"❌ 모델 로딩 실패: {e}")
    st.stop()

# ✅ 학습에 사용된 피처 컬럼 (고정 순서)
feature_columns = [
    "channel_tier_numeric", "year", "month", "day",
    "age_group_10대", "age_group_20대", "age_group_30대",
    "age_group_40대", "age_group_50대", "age_group_60대 이상",
    "gender_female", "gender_male",
    "brand_게토레이", "brand_링티", "brand_토레타",
    "brand_파워에이드", "brand_포카리스웨트"
]

# ✅ 매핑 딕셔너리
gender_map = {"여성": "female", "남성": "male"}
brand_map = {
    "게토레이": "게토레이",
    "링티": "링티",
    "토레타": "토레타",
    "파워에이드": "파워에이드",
    "포카리": "포카리스웨트"
}

# ✅ Streamlit UI
st.title("🚀 SportsDrink 소비 비율 예측 대시보드")
st.caption("📊 브랜드, 연령, 성별, 채널 티어 기반 예측 모델 (v8 기준)")

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("연도", [2024, 2025])
with col2:
    month = st.selectbox("월", list(range(1, 13)))
with col3:
    day = st.selectbox("일", list(range(1, 32)))

age_group = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대", "60대 이상"])
gender = st.radio("성별", ["남성", "여성"])
brand = st.selectbox("브랜드", ["게토레이", "링티", "토레타", "파워에이드", "포카리"])
tier = st.selectbox("채널 등급 (0=bronze ~ 4=diamond)", [0, 1, 2, 3, 4])

# ✅ 입력값을 feature_columns 에 맞춰 One-Hot Encoding
input_dict = {col: 0 for col in feature_columns}

input_dict.update({
    "channel_tier_numeric": tier,
    "year": year,
    "month": month,
    "day": day,
    f"age_group_{age_group}": 1,
    f"gender_{gender_map[gender]}": 1,
    f"brand_{brand_map[brand]}": 1
})

input_df = pd.DataFrame([input_dict])

# ✅ 예측 실행
if st.button("📈 예측 실행"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ 예측된 소비 비율: **{prediction[0]:.2f}**")
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")
