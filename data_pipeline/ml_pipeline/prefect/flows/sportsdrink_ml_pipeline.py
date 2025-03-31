import os
import sys
import subprocess
from datetime import datetime
from prefect import flow, task

# ✅ 공통 실행 함수 (환경변수 전달 포함)
def run_script(script_path: str, task_name: str):
    print(f"🚀 실행 중: {task_name}")
    print(f"📂 스크립트 경로: {script_path}")

    # ✅ 환경변수 전달 추가!
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        env=os.environ
    )

    # ✅ 로그 저장 디렉토리 설정
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "/app/data_pipeline/logs"
    os.makedirs(log_dir, exist_ok=True)

    # ✅ 로그 파일 경로
    stdout_path = os.path.join(log_dir, f"{task_name}_{log_time}_stdout.log")
    stderr_path = os.path.join(log_dir, f"{task_name}_{log_time}_stderr.log")

    # ✅ 로그 파일 저장
    with open(stdout_path, "w") as f:
        f.write(result.stdout)
    with open(stderr_path, "w") as f:
        f.write(result.stderr)

    # ✅ 콘솔 출력
    print("==== STDOUT ====")
    print(result.stdout.strip() or "No STDOUT")
    print("==== STDERR ====")
    print(result.stderr.strip() or "No STDERR")

    # ✅ 실패 처리
    if result.returncode != 0:
        print(f"❌ 스크립트 실패 로그 위치: {stderr_path}")
        raise RuntimeError(f"❌ {task_name} 스크립트 실행 실패!")

    print(f"✅ {task_name} 완료!\n")

# ✅ 각각의 스크립트를 Prefect Task로 래핑
@task(name="Train Model")
def train_model():
    script_path = "/app/data_pipeline/ml_pipeline/prefect/tasks/train_rf_model_with_mlflow.py"
    run_script(script_path, "모델 학습")

@task(name="Predict and Log")
def predict_and_log():
    script_path = "/app/data_pipeline/ml_pipeline/prefect/tasks/predict_and_log_to_mlflow.py"
    run_script(script_path, "예측 및 MLflow 로깅")

@task(name="Evaluate and Log")
def evaluate_and_log():
    script_path = "/app/data_pipeline/ml_pipeline/prefect/tasks/evaluate_and_log_to_mlflow.py"
    run_script(script_path, "모델 평가 및 MLflow 로깅")

# ✅ 메인 Flow 정의
@flow(name="SportsDrink ML Pipeline")
def sportsdrink_flow():
    train_model()
    predict_and_log()
    evaluate_and_log()

# ✅ CLI 실행 시 바로 실행
if __name__ == "__main__":
    sportsdrink_flow()
